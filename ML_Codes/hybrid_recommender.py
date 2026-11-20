import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # VS Code에서 창을 띄우지 않아도 PNG 저장되도록
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans, SVDpp


RATING_MIN = 1
RATING_MAX = 5
RANDOM_STATE = 42
RELEVANT_THRESHOLD = 4  # 평점 4~5점이면 사용자가 좋아한 장소로 봄
EVAL_K = 10               # 순위 평가용 상위 추천 개수 (명령행 --top-k와 별개)


def find_data_dir(script_dir, given_dir=None):
    required = [
        "ratings_train.csv",
        "ratings_validation.csv",
        "ratings_test.csv",
        "places_for_recsys.csv",
    ]

    if given_dir:
        folder = Path(given_dir).expanduser().resolve()
        missing = [name for name in required if not (folder / name).exists()]
        if missing:
            raise FileNotFoundError(f"필요한 파일이 없습니다: {missing}")
        return folder

    candidates = [script_dir] + [
        p.parent for name in required for p in script_dir.rglob(name)
    ]
    checked = set()
    for folder in candidates:
        folder = folder.resolve()
        if folder in checked:
            continue
        checked.add(folder)
        if all((folder / name).exists() for name in required):
            return folder

    raise FileNotFoundError("데이터 폴더를 찾지 못했습니다. --data-dir로 지정해주세요.")


def load_data(data_dir):
    train = pd.read_csv(data_dir / "ratings_train.csv")
    validation = pd.read_csv(data_dir / "ratings_validation.csv")
    test = pd.read_csv(data_dir / "ratings_test.csv")
    places = pd.read_csv(data_dir / "places_for_recsys.csv")

    required_cols = {"user_id", "place_id", "rating"}
    for name, df in [("train", train), ("validation", validation), ("test", test)]:
        if not required_cols.issubset(df.columns):
            raise ValueError(f"{name}: 필수 컬럼이 없습니다. {required_cols}")
        if df[["user_id", "place_id", "rating"]].isnull().any().any():
            raise ValueError(f"{name}: 결측치가 있습니다.")
        if df.duplicated(["user_id", "place_id"]).any():
            raise ValueError(f"{name}: 동일 사용자-장소 중복 평가가 있습니다.")
    if "place_id" not in places.columns:
        raise ValueError("places_for_recsys.csv에 place_id 컬럼이 필요합니다.")
    return train, validation, test, places


def make_trainset(train_df):
    reader = Reader(rating_scale=(RATING_MIN, RATING_MAX))
    data = Dataset.load_from_df(train_df[["user_id", "place_id", "rating"]], reader)
    return data.build_full_trainset()


def build_models(trainset):
    item_knn = KNNWithMeans(
        k=40,
        min_k=1,
        sim_options={"name": "cosine", "user_based": False, "min_support": 3},
        verbose=True,
    )
    svdpp = SVDpp(
        n_factors=40, n_epochs=20, lr_all=0.007, reg_all=0.02,
        random_state=RANDOM_STATE, verbose=True,
    )

    print("\n[1/2] Item-based KNN 학습")
    item_knn.fit(trainset)
    print("\n[2/2] SVD++ 학습")
    svdpp.fit(trainset)
    return item_knn, svdpp


def get_predictions(model, df):
    # Surprise의 predict()는 사용자와 장소 ID를 문자열로 넘기면 안전함.
    estimates = []
    for row in df.itertuples(index=False):
        pred = model.predict(str(row.user_id), str(row.place_id))
        estimates.append(float(pred.est))
    return np.asarray(estimates, dtype=float)


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted
    return {
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAE": float(np.mean(np.abs(errors))),
    }


def tune_alpha(validation, knn_model, svdpp_model):
    actual = validation["rating"].to_numpy(dtype=float)
    knn_pred = get_predictions(knn_model, validation)
    svdpp_pred = get_predictions(svdpp_model, validation)

    results = []
    best_alpha = None
    best_rmse = float("inf")
    for alpha in np.arange(0.0, 1.0001, 0.05):
        hybrid_pred = alpha * knn_pred + (1 - alpha) * svdpp_pred
        metrics = calculate_metrics(actual, hybrid_pred)
        results.append({
            "alpha_KNN": round(float(alpha), 2),
            "alpha_SVDpp": round(float(1 - alpha), 2),
            **metrics,
        })
        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_alpha = float(alpha)
    return best_alpha, pd.DataFrame(results)


def evaluate_models(test, knn_model, svdpp_model, alpha):
    actual = test["rating"].to_numpy(dtype=float)
    knn_pred = get_predictions(knn_model, test)
    svdpp_pred = get_predictions(svdpp_model, test)
    hybrid_pred = alpha * knn_pred + (1 - alpha) * svdpp_pred
    return pd.DataFrame({
        "Item-based KNN": calculate_metrics(actual, knn_pred),
        "SVD++": calculate_metrics(actual, svdpp_pred),
        "Hybrid": calculate_metrics(actual, hybrid_pred),
    }).T


def evaluate_topk(test, train, validation, places, knn_model, svdpp_model, alpha, k=EVAL_K):
    """테스트 데이터의 선호 장소를 상위 K개 중 얼마나 잘 찾아냈는지 확인.

    중요한 점: 평점이 없는 장소를 0점으로 취급하지 않음.
    테스트 데이터에서 실제로 4점 이상으로 관측된 장소만 '정답'으로 잡음.
    테스트 정답이 하나라도 있는 사용자만 평균에 포함함.
    앞서 평가/학습에 사용한 train, validation 장소는 추천 후보에서 제외함.

    전체 1,047개 후보 중 정답이 사용자당 몇 개 안 되는 구조라서
    Recall/NDCG가 낮을 수 있음. RMSE와는 평가 대상 자체가 다름.
    """
    # ID는 전부 문자열로 통일. 파일마다 타입이 달라도 비교할 수 있도록 함.
    test = test.copy()
    train = train.copy()
    validation = validation.copy()
    for df in (test, train, validation):
        df["user_id"] = df["user_id"].astype(str)
        df["place_id"] = df["place_id"].astype(str)

    place_ids = places["place_id"].astype(str).drop_duplicates().tolist()
    already_seen = (
        pd.concat([train[["user_id", "place_id"]], validation[["user_id", "place_id"]]])
        .groupby("user_id")["place_id"]
        .agg(set)
        .to_dict()
    )
    positives = (
        test.loc[test["rating"] >= RELEVANT_THRESHOLD]
        .groupby("user_id")["place_id"]
        .agg(set)
        .to_dict()
    )
    if not positives:
        raise ValueError("테스트 데이터에 4점 이상 평점을 남긴 사용자가 없습니다.")

    per_user = {m: {"Recall": [], "NDCG": []} for m in
                ["Item-based KNN", "SVD++", "Hybrid"]}
    excluded_count = 0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    print(f"\n===== Test Top-{k} 순위 평가 =====")
    print(f"정답: 테스트 평점 {RELEVANT_THRESHOLD}점 이상 / 후보: 평가하지 않은 전체 장소")
    print(f"평가 사용자 수: {len(positives):,}명 (테스트 정답이 1개 이상 있는 사용자)")

    # 테스트 정답이 있는 사용자만 계산하므로 전체 2,000명을 무조건 돌리지는 않음.
    for i, (user_id, relevant) in enumerate(positives.items(), start=1):
        seen = already_seen.get(user_id, set())
        candidate_ids = [pid for pid in place_ids if pid not in seen]
        if not candidate_ids:
            excluded_count += 1
            continue

        # Surprise의 모델 두 개로 모든 후보 장소를 점수화.
        knn_scores = np.array([knn_model.predict(user_id, pid).est for pid in candidate_ids])
        svd_scores = np.array([svdpp_model.predict(user_id, pid).est for pid in candidate_ids])
        score_map = {
            "Item-based KNN": knn_scores,
            "SVD++": svd_scores,
            "Hybrid": alpha * knn_scores + (1 - alpha) * svd_scores,
        }

        # 정답 장소 수는 테스트셋의 선호 장소 수를 그대로 사용.
        # 정답이 train/validation에 겹쳐 있으면 평가 전에 데이터 중복을 확인하는 게 좋음.
        for model_name, scores in score_map.items():
            # mergesort는 동점일 때 원래 장소 순서를 유지하는 안정 정렬.
            top_indices = np.argsort(-scores, kind="mergesort")[:k]
            top_ids = [candidate_ids[j] for j in top_indices]
            flags = np.array([pid in relevant for pid in top_ids], dtype=float)
            recall = float(flags.sum() / len(relevant))
            dcg = float(np.sum(flags * discounts[:len(flags)]))
            ideal = float(np.sum(discounts[:min(len(relevant), len(top_ids))]))
            ndcg = dcg / ideal if ideal else 0.0
            per_user[model_name]["Recall"].append(recall)
            per_user[model_name]["NDCG"].append(ndcg)

        if i % 200 == 0 or i == len(positives):
            print(f"  순위 평가 진행: {i:,}/{len(positives):,}명")

    rows = {}
    for name, values in per_user.items():
        rows[name] = {
            f"Recall@{k}": float(np.mean(values["Recall"])) if values["Recall"] else np.nan,
            f"NDCG@{k}": float(np.mean(values["NDCG"])) if values["NDCG"] else np.nan,
        }
    if excluded_count:
        print(f"후보 장소가 없어 제외한 사용자: {excluded_count:,}명")
    return pd.DataFrame.from_dict(rows, orient="index")


def save_metric_graphs(result_df, output_dir, k=EVAL_K):
    """네 지표를 한 장에 섞지 않고 각각 PNG로 저장. 비교하기 쉽게 값도 표시."""
    specs = [
        ("MAE", "mae_comparison.png", "Mean Absolute Error (lower is better)"),
        ("RMSE", "rmse_comparison.png", "Root Mean Squared Error (lower is better)"),
        (f"Recall@{k}", f"recall_at_{k}.png", f"Recall@{k} (higher is better)"),
        (f"NDCG@{k}", f"ndcg_at_{k}.png", f"NDCG@{k} (higher is better)"),
    ]
    for col, filename, title in specs:
        values = result_df[col].astype(float)
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        bars = ax.bar(values.index, values.to_numpy())
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(col)
        ax.set_ylim(0, max(values.max() * 1.22, 0.05))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)
        print(f"그래프 저장: {output_dir / filename}")


def recommend_places(user_id, top_k, train, places, knn_model, svdpp_model, alpha):
    user_id = str(user_id)
    known_users = set(train["user_id"].astype(str))
    if user_id not in known_users:
        raise ValueError(f"{user_id}는 학습에 없는 사용자입니다. 이 단독 모델은 신규 사용자를 지원하지 않습니다.")

    rated_places = set(train.loc[
        train["user_id"].astype(str) == user_id, "place_id"
    ].astype(str))
    candidates = places[
        ~places["place_id"].astype(str).isin(rated_places)
    ].copy()

    predictions = []
    for place_id in candidates["place_id"].astype(str):
        p_knn = knn_model.predict(user_id, place_id).est
        p_svdpp = svdpp_model.predict(user_id, place_id).est
        score = alpha * p_knn + (1 - alpha) * p_svdpp
        predictions.append(float(np.clip(score, RATING_MIN, RATING_MAX)))

    candidates["predicted_rating"] = predictions
    candidates = candidates.sort_values("predicted_rating", ascending=False).head(top_k)
    display_cols = [col for col in [
        "place_id", "name", "category", "district", "dong",
        "nearby_station", "predicted_rating"
    ] if col in candidates.columns]
    return candidates[display_cols].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None, help="데이터 폴더")
    parser.add_argument("--user-id", type=str, default="U0001", help="기존 사용자 ID")
    parser.add_argument("--top-k", type=int, default=10, help="추천 목록 출력 개수")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k는 1 이상이어야 합니다.")

    data_dir = find_data_dir(Path(__file__).resolve().parent, args.data_dir)
    print(f"Data Folder: {data_dir}")
    train, validation, test, places = load_data(data_dir)
    print("\n===== 데이터 확인 =====")
    print(f"Train: {len(train):,} / Validation: {len(validation):,} / Test: {len(test):,}")
    print(f"장소: {places['place_id'].nunique():,} / 학습 사용자: {train['user_id'].nunique():,}")

    train_users = set(train["user_id"].astype(str))
    train_items = set(train["place_id"].astype(str))
    for label, df in [("Validation", validation), ("Test", test)]:
        print(f"{label} 미학습 사용자: {len(set(df['user_id'].astype(str)) - train_users):,}")
        print(f"{label} 미학습 장소: {len(set(df['place_id'].astype(str)) - train_items):,}")

    trainset = make_trainset(train)
    knn_model, svdpp_model = build_models(trainset)

    print("\n===== Validation: 결합 가중치 탐색 =====")
    best_alpha, alpha_results = tune_alpha(validation, knn_model, svdpp_model)
    alpha_results.to_csv(data_dir / "alpha_validation_results.csv", index=False, encoding="utf-8-sig")
    print(alpha_results.to_string(index=False))
    print(f"선택한 가중치: KNN {best_alpha:.2f}, SVD++ {1-best_alpha:.2f}")

    print("\n===== Test: 평점 예측 오차 =====")
    final_results = evaluate_models(test, knn_model, svdpp_model, best_alpha)
    print(final_results.to_string(float_format=lambda x: f"{x:.4f}"))

    # RMSE/MAE는 관측된 평점에 대한 회귀 오차.
    # Recall/NDCG는 '추천 순위' 지표라서 별도로 전체 후보를 정렬해야 함.
    rank_results = evaluate_topk(
        test, train, validation, places, knn_model, svdpp_model, best_alpha, EVAL_K
    )
    print("\n===== Test: 추천 순위 평가 =====")
    print(rank_results.to_string(float_format=lambda x: f"{x:.4f}"))

    # 네 지표를 한 파일로 묶으면 모델별로 한눈에 비교하기 편함.
    all_results = final_results.join(rank_results)
    all_results.to_csv(data_dir / "cf_all_test_metrics.csv", encoding="utf-8-sig")
    final_results.to_csv(data_dir / "hybrid_test_results.csv", encoding="utf-8-sig")
    rank_results.to_csv(data_dir / "cf_topk_test_results.csv", encoding="utf-8-sig")
    save_metric_graphs(all_results, data_dir, EVAL_K)

    print("\n===== 기존 사용자 장소 추천 =====")
    recommendations = recommend_places(
        args.user_id, args.top_k, train, places, knn_model, svdpp_model, best_alpha
    )
    print(f"사용자 {args.user_id} 추천 결과")
    print(recommendations.to_string(index=False))
    recommendations.to_csv(
        data_dir / f"recommendations_{args.user_id}.csv",
        index=False, encoding="utf-8-sig"
    )
    print("\n완료: 평가 CSV 및 그래프 PNG를 데이터 폴더에 저장했습니다.")


if __name__ == "__main__":
    main()
