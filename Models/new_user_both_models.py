# -*- coding: utf-8 -*-
"""New user prefs -> CB Ridge + CF (KNNWithMeans/SVD++) hybrid -> 4-stop rec.
Run: python new_user_both_models.py --data-root .
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import contents_rec as cb
from final_course_pipeline import cf_fit, cf_predict

CATS = ("명소", "문화", "쇼핑")


def ask_preferences():
    print("\n=== New User Seoul Date Course Rec (CB + CF) ===")
    weights = []
    for cat in CATS:
        while True:
            try:
                val = float(input(f"{cat} preference (0-10): ").strip())
                if not np.isfinite(val) or not 0 <= val <= 10:
                    raise ValueError
                weights.append(val)
                break
            except ValueError:
                print("Enter a number between 0 and 10.")
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0:
        weights[:] = 1
        print("All zero; using equal weights.")
    return weights / weights.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--output-dir", default="new_user_both_output")
    ap.add_argument("--neighbors", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.data_root)

    def load(name):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")
        return pd.read_csv(path)

    places = load("places_for_recsys.csv")
    users = load("users_synthetic.csv")
    train = load("ratings_train.csv")
    valid = load("ratings_validation.csv")
    test = load("ratings_test.csv")
    courses = load("four_stop_courses_estimated.csv")

    places["place_id"] = places.place_id.astype(str)
    places["district"] = places.district.fillna(cb.UNKNOWN)
    users["user_id"] = users.user_id.astype(str)
    users["preferred_district"] = users.preferred_district.fillna(cb.UNKNOWN)
    for d in (train, valid, test):
        d["user_id"] = d.user_id.astype(str)
        d["place_id"] = d.place_id.astype(str)

    pref = ask_preferences()
    districts = sorted(d for d in places.district.unique() if d != cb.UNKNOWN)
    print("Available districts:", ", ".join(districts))
    while True:
        district = input("Preferred district: ").strip()
        if district in districts:
            break
        print("Enter a valid district from the list above.")

    print("\nTraining models & tuning validation weights...")
    up = {uid: i for i, uid in enumerate(users.user_id)}
    pp = {pid: i for i, pid in enumerate(places.place_id)}
    tr, va, te = [cb.to_index(df, up, pp) for df in (train, valid, test)]

    pcat, _, dnames = cb.build_place_features(places)
    ucat, _ = cb.build_explicit_user_profiles(users, dnames)
    centers = cb.district_centers(places)
    near = cb.proximity_matrix(
        users.preferred_district.map(centers.latitude).to_numpy(dtype=float),
        users.preferred_district.map(centers.longitude).to_numpy(dtype=float),
        places,
        3,
    )
    generosity = users.rating_generosity.to_numpy(dtype=float)

    def features(idx):
        u, p, _ = idx
        return np.column_stack([cb.cosine_rows(ucat[u], pcat[p]), near[u, p], generosity[u]])

    xtr, xva = features(tr), features(va)
    ridge_trials = []
    for a in cb.RIDGE_ALPHAS:
        model = Ridge(alpha=a).fit(xtr, tr[2])
        pred = np.clip(model.predict(xva), 1, 5)
        ridge_trials.append((cb.rmse(va[2], pred), a, pred))
    _, ridge_alpha, cb_val = min(ridge_trials, key=lambda x: x[0])

    cf_models = cf_fit(train)
    knn_val = cf_predict(cf_models, valid.user_id, valid.place_id, 1.0)
    svd_val = cf_predict(cf_models, valid.user_id, valid.place_id, 0.0)
    cf_trials = [
        (cb.rmse(va[2], a * knn_val + (1 - a) * svd_val), float(a))
        for a in np.round(np.arange(0, 1.001, 0.05), 2)
    ]
    _, cf_alpha = min(cf_trials)
    cf_val = cf_alpha * knn_val + (1 - cf_alpha) * svd_val

    trials = []
    for beta in np.round(np.arange(0, 1.001, 0.05), 2):
        pred = beta * cb_val + (1 - beta) * cf_val
        trials.append({
            "content_weight": float(beta),
            "cf_weight": float(1 - beta),
            "validation_rmse": cb.rmse(va[2], pred),
            "validation_mae": cb.mae(va[2], pred),
        })
    weights_df = pd.DataFrame(trials)
    beta = float(weights_df.loc[weights_df.validation_rmse.idxmin(), "content_weight"])

    tv = pd.concat([train, valid], ignore_index=True)
    tvi = cb.to_index(tv, up, pp)
    cb_model = Ridge(alpha=ridge_alpha).fit(features(tvi), tvi[2])
    cf_models = cf_fit(tv)

    cb_test = np.clip(cb_model.predict(features(te)), 1, 5)
    cf_test = cf_predict(cf_models, test.user_id, test.place_id, cf_alpha)
    blended_test = beta * cb_test + (1 - beta) * cf_test
    metrics = pd.DataFrame([
        {"model": "Content Ridge", "rmse": cb.rmse(te[2], cb_test), "mae": cb.mae(te[2], cb_test)},
        {"model": "CF KNN + SVD++", "rmse": cb.rmse(te[2], cf_test), "mae": cb.mae(te[2], cf_test)},
        {"model": "Combined", "rmse": cb.rmse(te[2], blended_test), "mae": cb.mae(te[2], blended_test)},
    ])

    # CB: build profile from user input (same 3 features)
    n = len(places)
    center = centers.loc[district]
    near_new = cb.proximity_matrix(
        np.array([center.latitude]),
        np.array([center.longitude]),
        places,
        3,
    )[0]
    new_features = np.column_stack([
        cb.cosine_rows(np.repeat(pref[None, :], n, axis=0), pcat),
        near_new,
        np.full(n, float(np.nanmean(generosity))),
    ])
    content_scores = np.clip(cb_model.predict(new_features), 1, 5)

    # CF proxy: weighted avg of similar users' KNN+SVD++ predictions
    norms = np.linalg.norm(ucat, axis=1) * np.linalg.norm(pref)
