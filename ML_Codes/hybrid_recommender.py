import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans, SVDpp


RATING_MIN = 1
RATING_MAX = 5
RANDOM_STATE = 42


def find_data_dir(script_dir, given_dir=None):
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
                f"File not Found: {missing}"
            )
        return folder


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
        "Data Folder not Found\n"
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
            raise ValueError(f"{name} Required column do not exist: {required_cols}")
        if df[["user_id", "place_id", "rating"]].isnull().any().any():
            raise ValueError(f"{name} There is an outlier in data")
        if df.duplicated(["user_id", "place_id"]).any():
            raise ValueError(f"{name} User-place Duplication in data")

    return train, validation, test, places


def make_trainset(train_df):
    reader = Reader(rating_scale=(RATING_MIN, RATING_MAX))
    data = Dataset.load_from_df(
        train_df[["user_id", "place_id", "rating"]],
        reader
    )
    return data.build_full_trainset()


def build_models(trainset):

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

    print("\n[1/2] Item-based KNN Train")
    item_knn.fit(trainset)

    print("\n[2/2] SVD++ Train")
    svdpp.fit(trainset)

    return item_knn, svdpp


def get_predictions(model, df):
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

    user_id = str(user_id)

    known_users = set(train["user_id"].astype(str))
    if user_id not in known_users:
        raise ValueError(
            f"{user_id} is not Trained user"
            "Current model don't service to Not Trained user"
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
        help="Data directory",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="U0001",
        help="user ID (Default: U0001)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="number of place for recommend (Default: 10)",
    )
    args = parser.parse_args()

    data_dir = find_data_dir(Path(__file__).resolve().parent, args.data_dir)
    print(f"Data Folder: {data_dir}")

    train, validation, test, places = load_data(data_dir)

    print("\n===== Data check =====")
    print(f"Train      : {len(train):,}건")
    print(f"Validation : {len(validation):,}건")
    print(f"Test       : {len(test):,}건")
    print(f"Count places     : {places['place_id'].nunique():,}개")
    print(f"Count users   : {train['user_id'].nunique():,}명")

    train_users = set(train["user_id"].astype(str))
    train_items = set(train["place_id"].astype(str))

    print(
        "Validation trained user:",
        len(set(validation["user_id"].astype(str)) - train_users),
    )
    print(
        "Validation Test Not trained place:",
        len(set(validation["place_id"].astype(str)) - train_items),
    )
    print(
        "Test Not trained user:",
        len(set(test["user_id"].astype(str)) - train_users),
    )
    print(
        "Test Not trained place:",
        len(set(test["place_id"].astype(str)) - train_items),
    )

    trainset = make_trainset(train)
    knn_model, svdpp_model = build_models(trainset)

    print("\n===== Validation: Weight Search =====")
    best_alpha, alpha_results = tune_alpha(
        validation, knn_model, svdpp_model
    )
    alpha_results.to_csv(
        data_dir / "alpha_validation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(alpha_results.to_string(index=False))
    print(f"\nSelected alpha (Item-based KNN Weight): {best_alpha:.2f}")
    print(f"SVD++ Weight: {1 - best_alpha:.2f}")

    print("\n===== Test =====")
    final_results = evaluate_models(
        test, knn_model, svdpp_model, best_alpha
    )
    print(final_results.to_string(float_format=lambda x: f"{x:.4f}"))

    final_results.to_csv(
        data_dir / "hybrid_test_results.csv",
        encoding="utf-8-sig",
    )

    print("\n===== User place recommend =====")
    recommendations = recommend_places(
        args.user_id,
        args.top_k,
        train,
        places,
        knn_model,
        svdpp_model,
        best_alpha,
    )
    print(f"\nUser {args.user_id} Recommendation Result")
    print(recommendations.to_string(index=False))

    recommendations.to_csv(
        data_dir / f"recommendations_{args.user_id}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nResult files saved.:")
    print("- alpha_validation_results.csv")
    print("- hybrid_test_results.csv")
    print(f"- recommendations_{args.user_id}.csv")


if __name__ == "__main__":
    main()
