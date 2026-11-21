# -*- coding: utf-8 -*-
"""Val weight search + 4-stop walking course selection.
Run: python final_course_pipeline.py --data-root /path/to/project --user-id U0001
Deps: pandas numpy scikit-learn scikit-surprise, contents_rec.py (same dir)
CSVs: places_for_recsys.csv, users_synthetic.csv, ratings_train.csv,
ratings_validation.csv, ratings_test.csv, four_stop_courses_estimated.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from surprise import Dataset, KNNWithMeans, Reader, SVDpp

import contents_rec as cb


def locate(root, filename):
    hits = list(root.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"{filename} not found. Check --data-root.")
    return hits[0]


def cf_fit(df):
    ds = Dataset.load_from_df(
        df[["user_id", "place_id", "rating"]],
        Reader(rating_scale=(1, 5)),
    )
    ts = ds.build_full_trainset()
    knn = KNNWithMeans(
        k=40,
        min_k=1,
        sim_options={"name": "cosine", "user_based": False, "min_support": 3},
        verbose=False,
    )
    svd = SVDpp(
        n_factors=40,
        n_epochs=20,
        lr_all=0.007,
        reg_all=0.02,
        random_state=42,
        verbose=False,
    )
    knn.fit(ts)
    svd.fit(ts)
    return knn, svd


def cf_predict(models, users, places, alpha):
    knn, svd = models
    return np.array([
        alpha * knn.predict(str(u), str(p)).est + (1 - alpha) * svd.predict(str(u), str(p)).est
        for u, p in zip(users, places)
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--user-id", default="U0001")
    ap.add_argument("--output-dir", default="final_course_output")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    req_files = [
        "places_for_recsys.csv",
        "users_synthetic.csv",
        "ratings_train.csv",
        "ratings_validation.csv",
        "ratings_test.csv",
    ]
    files = {x: locate(root, x) for x in req_files}

    # Match suffixed filenames (e.g., (2))
    courses_hits = list(root.rglob("four_stop_courses_estimated*.csv"))
    if not courses_hits:
        raise FileNotFoundError("four_stop_courses_estimated*.csv required.")

    places = pd.read_csv(files["places_for_recsys.csv"])
    users = pd.read_csv(files["users_synthetic.csv"])
    train = pd.read_csv(files["ratings_train.csv"])
    valid = pd.read_csv(files["ratings_validation.csv"])
    test = pd.read_csv(files["ratings_test.csv"])
    courses = pd.read_csv(courses_hits[0])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for df in (train, valid, test):
        df["user_id"] = df["user_id"].astype(str)
        df["place_id"] = df["place_id"].astype(str)
    users["user_id"] = users["user_id"].astype(str)
    places["place_id"] = places["place_id"].astype(str)

    places["district"] = places["district"].fillna(cb.UNKNOWN)
    users["preferred_district"] = users["preferred_district"].fillna(cb.UNKNOWN)

    up = {v: i for i, v in enumerate(users.user_id)}
    pp = {v: i for i, v in enumerate(places.place_id)}
    tr = cb.to_index(train, up, pp)
    va = cb.to_index(valid, up, pp)
    te = cb.to_index(test, up, pp)

    pcat, pdist, districts = cb.build_place_features(places)
    ucat, udist = cb.build_explicit_user_profiles(users, districts)
    centers = cb.district_centers(places)
    lat = users.preferred_district.map(centers.latitude).to_numpy(dtype=float)
    lon = users.preferred_district.map(centers.longitude).to_numpy(dtype=float)

    # CB model F config: exp_cat, exp_near, generosity; scale 3km
    near = cb.proximity_matrix(lat, lon, places, 3)
    gen = users.rating_generosity.to_numpy(dtype=float)

    def cb_features(u, p):
        return np.column_stack([
            cb.cosine_rows(ucat[u], pcat[p]),
            near[u, p],
            gen[u],
        ])

    xtr = cb_features(tr[0], tr[1])
    xva = cb_features(va[0], va[1])

    cb_rows = []
    for ridge_alpha in cb.RIDGE_ALPHAS:
        model = Ridge(alpha=ridge_alpha).fit(xtr, tr[2])
        pred = np.clip(model.predict(xva), 1, 5)
        cb_rows.append((cb.rmse(va[2], pred), ridge_alpha, model, pred))
    cb_rmse, ridge_alpha, cb_model, cb_val = min(cb_rows, key=lambda r: r[0])

    cf_models = cf_fit(train)
    knn_val = cf_predict(cf_models, valid.user_id, valid.place_id, 1.0)
    svd_val = cf_predict(cf_models, valid.user_id, valid.place_id, 0.0)

    cf_alpha_rows = []
    for a in np.round(np.arange(0, 1.001, 0.05), 2):
        pred = a * knn_val + (1 - a) * svd_val
        cf_alpha_rows.append({
            "cf_knn_weight": a,
            "cf_svdpp_weight": round(1 - a, 2),
            "validation_rmse": cb.rmse(va[2], pred),
        })
    cf_alpha_df = pd.DataFrame(cf_alpha_rows)
    cf_alpha = float(cf_alpha_df.loc[cf_alpha_df.validation_rmse.idxmin(), "cf_knn_weight"])
    cf_val = cf_alpha * knn_val + (1 - cf_alpha) * svd_val

    rows = []
    for beta in np.round(np.arange(0, 1.001, 0.05), 2):
        pred = beta * cb_val + (1 - beta) * cf_val
        rows.append({
            "content_weight": beta,
            "cf_weight": round(1 - beta, 2),
            "validation_rmse": cb.rmse(va[2], pred),
            "validation_mae": cb.mae(va[2], pred),
        })
    weights = pd.DataFrame(rows)
    best = float(weights.loc[weights.validation_rmse.idxmin(), "content_weight"])

    weights.to_csv(out / "validation_weight_search.csv", index=False, encoding="utf-8-sig")
    cf_alpha_df.to_csv(out / "cf_internal_weight_search.csv", index=False, encoding="utf-8-sig")

    # Refit on train + val; eval on test
    tv = pd.concat([train, valid], ignore_index=True)
    tvi = cb.to_index(tv, up, pp)
    cb_final = Ridge(alpha=ridge_alpha).fit(cb_features(tvi[0], tvi[1]), tvi[2])
    cf_final = cf_fit(tv)

    cb_test = np.clip(cb_final.predict(cb_features(te[0], te[1])), 1, 5)
    cf_test = cf_predict(cf_final, test.user_id, test.place_id, cf_alpha)
    final_test = best * cb_test + (1 - best) * cf_test

    metrics = pd.DataFrame([
        {"model": "Content", "test_rmse": cb.rmse(te[2], cb_test), "test_mae": cb.mae(te[2], cb_test)},
        {"model": "CF Hybrid", "test_rmse": cb.rmse(te[2], cf_test), "test_mae": cb.mae(te[2], cf_test)},
        {"model": "Combined", "test_rmse": cb.rmse(te[2], final_test), "test_mae": cb.mae(te[2], final_test)},
    ])
    metrics.to_csv(out / "test_metrics.csv", index=False, encoding="utf-8-sig")

    if args.user_id not in up:
        raise ValueError(f"User {args.user_id} not in users_synthetic.csv.")

    u = up[args.user_id]
    n = len(places)
    p = np.arange(n, dtype=int)
    uu = np.full(n, u, dtype=int)

    cb_scores = np.clip(cb_final.predict(cb_features(uu, p)), 1, 5)
    cf_scores = cf_predict(cf_final, [args.user_id] * n, places.place_id, cf_alpha)

    place_scores = places[["place_id", "name", "category", "district"]].copy()
    place_scores["content_rating"] = cb_scores
    place_scores["cf_rating"] = cf_scores
    place_scores["combined_rating"] = best * cb_scores + (1 - best) * cf_scores

    # Exclude already-rated places
    rated = set(tv.loc[tv.user_id == args.user_id, "place_id"])
    place_scores["already_rated"] = place_scores.place_id.isin(rated)
    place_scores.to_csv(out / f"place_scores_{args.user_id}.csv", index=False, encoding="utf-8-sig")

    cols = ["place_1_id", "place_2_id", "place_3_id", "place_4_id"]
    for col in cols:
        courses[col] = courses[col].astype(str)

    eligible = courses.loc[
        (courses[["leg1_min", "leg2_min", "leg3_min"]] <= 40).all(axis=1)
        & (courses.total_estimated_minutes <= 75)
    ].copy()
    eligible = eligible.loc[eligible[cols].nunique(axis=1) == 4].copy()

    score_map = place_scores.set_index("place_id").combined_rating
    for i, col in enumerate(cols, 1):
        eligible[f"rating_{i}"] = eligible[col].map(score_map)

    eligible = eligible.dropna(subset=[f"rating_{i}" for i in range(1, 5)])
    eligible = eligible.loc[~eligible[cols].isin(rated).any(axis=1)].copy()
    eligible["rating_sum"] = eligible[[f"rating_{i}" for i in range(1, 5)]].sum(axis=1)
    eligible["rating_mean"] = eligible.rating_sum / 4

    # Objective: max rating_sum under travel constraints; tie-break by shorter duration
    eligible = eligible.sort_values(
        ["rating_sum", "total_estimated_minutes"],
        ascending=[False, True],
    )
    eligible.to_csv(out / f"ranked_courses_{args.user_id}.csv", index=False, encoding="utf-8-sig")

    print("\n=== Validation Weights ===")
    print(f"Content {best:.2f}, CF {1 - best:.2f} (CF KNN {cf_alpha:.2f}, SVD++ {1 - cf_alpha:.2f})")
    print("\n=== Test Metrics ===\n", metrics.to_string(index=False))
    print(f"\nEligible unseen courses: {len(eligible)}")

    if len(eligible):
        c = eligible.iloc[0]
        print("\n=== Top 4 Stops ===")
        for i, col in enumerate(cols, 1):
            print(f"{i}. {c[f'place_{i}_name']} ({c[col]}), pred rating {c[f'rating_{i}']:.4f}")
        print(f"Walk time (min): {c.leg1_min}, {c.leg2_min}, {c.leg3_min} / Total {c.total_estimated_minutes}")
        print("Note: Walking times are estimates, not verified routing.")
    else:
        print("No valid courses found. Expand candidate pool.")

    print("Saved to:", out.resolve())


if __name__ == "__main__":
    main()