# -*- coding: utf-8 -*-
"""
서울 데이트 장소 추천 - Item-based KNN + SVD++ 하이브리드
필요 패키지:
    pip install scikit-surprise pandas numpy

실행:
    python hybrid_recommender.py
    python hybrid_recommender.py --data-dir "데이터셋의 03_협업필터링_하이브리드 폴더"
    python hybrid_recommender.py --user-id U0001 --top-k 10
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans, SVDpp


RATING_MIN = 1
RATING_MAX = 5
RANDOM_STATE = 42


def find_data_dir(script_dir, given_dir=None):
    """CSV 파일이 있는 데이터 폴더를 찾는다."""
    required = [
        "ratings_train.csv",
        "ratings_validation.csv",
        "ratings_test.csv",
        "places_for_recsys.csv",
    ]

    if given_dir:
        folder = Path(given_dir)
        missing = [name for name in required if not (folder / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"지정한 폴더에서 파일을 찾지 못했습니다: {missing}"
            )
        return folder

    # 스크립트가 데이터 폴더 안에 있거나, 압축 해제한 상위 폴더에 있는 경우
    candidates = [script_dir] + [
        p.parent
        for name in required
        for p in script_dir.rglob(name)
    ]

    checked = set()
    for folder in candidates:
        folder = folder.resolve()
        if folder in checked:
            continue
        checked.add(folder)
        if all((folder / name).exists() for name in required):
            return folder

    raise FileNotFoundError(
        "데이터 폴더를 자동으로 찾지 못했습니다.\n"
        "--data-dir 옵션으로 CSV 4개가 들어 있는 폴더를 지정하세요."
    )


def load_data(data_dir):
    train = pd.read_csv(data_dir / "ratings_train.csv")
    validation = pd.read_csv(data_dir / "ratings_validation.csv")
    test = pd.read_csv(data_dir / "ratings_test.csv")
    places = pd.read_csv(data_dir / "places_for_recsys.csv")

    required_cols = {"user_id", "place_id", "rating"}
    for name, df in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if not required_cols.issubset(df.columns):
            raise ValueError(f"{name} 데이터에 필요한 컬럼이 없습니다: {required_cols}")
        if df[["user_id", "place_id", "rating"]].isnull().any().any():
            raise ValueError(f"{name} 데이터에 결측치가 있습니다.")
        if df.duplicated(["user_id", "place_id"]).any():
            raise ValueError(f"{name} 데이터에 동일 사용자-장소 중복 평점이 있습니다.")

    return train, validation, test, places


def make_trainset(train_df):
    """Surprise 학습 데이터셋 생성."""
    reader = Reader(rating_scale=(RATING_MIN, RATING_MAX))
    data = Dataset.load_from_df(
        train_df[["user_id", "place_id", "rating"]],
        reader
    )
    return data.build_full_trainset()


def build_models(trainset):
    """Item-based KNN과 SVD++ 모델을 생성하고 학습한다."""

    # user_based=False가 Item-based 방식의 핵심 설정이다.
    item_knn = KNNWithMeans(
        k=40,
        min_k=1,
        sim_options={
            "name": "cosine",
            "user_based": False,
            "min_support": 3,
        },
        verbose=True,
    )

    svdpp = SVDpp(
        n_factors=40,
        n_epochs=20,
        lr_all=0.007,
        reg_all=0.02,
        random_state=RANDOM_STATE,
        verbose=True,
    )

    print("\n[1/2] Item-based KNN 학습")
    item_knn.fit(trainset)

    print("\n[2/2] SVD++ 학습")
    svdpp.fit(trainset)

    return item_knn, svdpp


def get_predictions(model, df):
    """데이터프레임의 각 사용자-장소 조합에 대한 예측값을 반환한다."""
    estimates = []
    for row in df.itertuples(index=False):
        pred = model.predict(str(row.user_id), str(row.place_id))
        estimates.append(float(pred.est))
    return np.asarray(estimates, dtype=float)


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    return {"RMSE": rmse, "MAE": mae}


def tune_alpha(validation, knn_model, svdpp_model):
    """
    Validation RMSE가 가장 낮은 KNN 가중치 alpha를 선택한다.
    Hybrid = alpha * Item-KNN + (1-alpha) * SVD++
    """
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

    result_df = pd.DataFrame(results)
    return best_alpha, result_df


def evaluate_models(test, knn_model, svdpp_model, alpha):
    """Test 데이터로 단일 모델과 하이브리드 모델의 최종 성능을 비교한다."""
    actual = test["rating"].to_numpy(dtype=float)

    knn_pred = get_predictions(knn_model, test)
    svdpp_pred = get_predictions(svdpp_model, test)
    hybrid_pred = alpha * knn_pred + (1 - alpha) * svdpp_pred

    results = {
        "Item-based KNN": calculate_metrics(actual, knn_pred),
        "SVD++": calculate_metrics(actual, svdpp_pred),
        "Hybrid": calculate_metrics(actual, hybrid_pred),
    }

    return pd.DataFrame(results).T


def recommend_places(
    user_id,
    top_k,
    train,
    places,
    knn_model,
    svdpp_model,
    alpha,
):
    """학습 데이터에서 이미 평가한 장소를 제외하고 전체 장소 중 Top-K 추천."""
    user_id = str(user_id)

    known_users = set(train["user_id"].astype(str))
    if user_id not in known_users:
        raise ValueError(
            f"{user_id}는 학습 데이터에 없는 사용자입니다. "
            "현재 모델은 신규 사용자 추천을 지원하지 않습니다."
        )

    rated_places = set(
        train.loc[train["user_id"].astype(str) == user_id, "place_id"].astype(str)
    )

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
    candidates = candidates.sort_values(
        "predicted_rating", ascending=False
    ).head(top_k)

    display_cols = [
        col for col in [
            "place_id", "name", "category", "district", "dong",
            "nearby_station", "predicted_rating"
        ]
        if col in candidates.columns
    ]

    return candidates[display_cols].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="ratings_train.csv 등이 들어 있는 데이터 폴더 경로",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="U0001",
        help="추천을 확인할 사용자 ID (기본값: U0001)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="추천할 장소 수 (기본값: 10)",
    )
    args = parser.parse_args()

    data_dir = find_data_dir(Path(__file__).resolve().parent, args.data_dir)
    print(f"데이터 폴더: {data_dir}")

    train, validation, test, places = load_data(data_dir)

    print("\n===== 데이터 확인 =====")
    print(f"Train      : {len(train):,}건")
    print(f"Validation : {len(validation):,}건")
    print(f"Test       : {len(test):,}건")
    print(f"장소 수     : {places['place_id'].nunique():,}개")
    print(f"사용자 수   : {train['user_id'].nunique():,}명")

    # 검증/테스트의 사용자와 장소가 학습 데이터에 존재하는지 확인
    train_users = set(train["user_id"].astype(str))
    train_items = set(train["place_id"].astype(str))

    print(
        "Validation 미학습 사용자:",
        len(set(validation["user_id"].astype(str)) - train_users),
    )
    print(
        "Validation 미학습 장소:",
        len(set(validation["place_id"].astype(str)) - train_items),
    )
    print(
        "Test 미학습 사용자:",
        len(set(test["user_id"].astype(str)) - train_users),
    )
    print(
        "Test 미학습 장소:",
        len(set(test["place_id"].astype(str)) - train_items),
    )

    trainset = make_trainset(train)
    knn_model, svdpp_model = build_models(trainset)

    print("\n===== Validation: 최적 가중치 탐색 =====")
    best_alpha, alpha_results = tune_alpha(
        validation, knn_model, svdpp_model
    )
    alpha_results.to_csv(
        data_dir / "alpha_validation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(alpha_results.to_string(index=False))
    print(f"\n선택된 alpha (Item-based KNN 가중치): {best_alpha:.2f}")
    print(f"SVD++ 가중치: {1 - best_alpha:.2f}")

    print("\n===== Test: 최종 성능 평가 =====")
    final_results = evaluate_models(
        test, knn_model, svdpp_model, best_alpha
    )
    print(final_results.to_string(float_format=lambda x: f"{x:.4f}"))

    final_results.to_csv(
        data_dir / "hybrid_test_results.csv",
        encoding="utf-8-sig",
    )

    print("\n===== 사용자 장소 추천 =====")
    recommendations = recommend_places(
        args.user_id,
        args.top_k,
        train,
        places,
        knn_model,
        svdpp_model,
        best_alpha,
    )
    print(f"\n사용자 {args.user_id} 추천 결과")
    print(recommendations.to_string(index=False))

    recommendations.to_csv(
        data_dir / f"recommendations_{args.user_id}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n결과 파일 저장 완료:")
    print("- alpha_validation_results.csv")
    print("- hybrid_test_results.csv")
    print(f"- recommendations_{args.user_id}.csv")


if __name__ == "__main__":
    main()
