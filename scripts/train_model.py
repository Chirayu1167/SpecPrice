import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(r"C:\Users\Mahaj\Downloads\archive\smartprix_smartphones_april_2026.csv")
OUT_DIR = BASE_DIR / "models"

RANDOM_STATE = 42

CHARGING_TYPE_MAP = {"Slow Charging": "Standard"}


def load_and_clean():
    df = pd.read_csv(DATA_PATH)

    df["charging_speed_type"] = df["charging_speed_type"].map(CHARGING_TYPE_MAP).fillna(df["charging_speed_type"])
    df["brand_name"] = df["brand_name"].replace("samaung", "samsung")
    df["processor_brand"] = df["processor_brand"].fillna("unknown").replace("apple", "unknown")

    df = df.dropna(subset=["price"])
    df = df[(df["price"] >= 5000) & (df["price"] <= 300000)]

    df = df[df["fast_charging(W)"].fillna(45) <= 300]
    df = df[df["battery_capacity(mAh)"] <= 8000]
    df = df[df["screen_size"].between(4.5, 8.0)]
    df = df[df["refresh_rate"].fillna(120).between(60, 200)]

    df["processor_speed"] = df["processor_speed"].fillna(df["processor_speed"].median())
    df["fast_charging(W)"] = df["fast_charging(W)"].fillna(45)
    df["refresh_rate"] = df["refresh_rate"].fillna(120)
    df["rear_camera"] = df["rear_camera"].fillna(df["rear_camera"].median())
    df["front_camera"] = df["front_camera"].fillna(df["front_camera"].median())
    df["num_core"] = df["num_core"].fillna(8).astype(int)
    df["os"] = df["os"].fillna("Android v14")

    df["charging_ratio"] = df["battery_capacity(mAh)"] / df["fast_charging(W)"]

    df = df.reset_index(drop=True)
    print(f"cleaned: {len(df)} rows (of 997)")
    print(f"price range: {df['price'].min():,.0f} - {df['price'].max():,.0f}")
    return df


NUMERIC = [
    "has_5G", "has_NFC", "has_IR", "num_core", "processor_speed", "ram", "memory",
    "battery_capacity(mAh)", "fast_charging(W)", "charging_ratio", "screen_size",
    "refresh_rate", "rear_camera", "front_camera", "rear_camera_count",
]
CATEGORICAL = ["brand_name", "processor_brand", "os", "charging_speed_type"]
HAS_FLAG = {"True": 1.0, "False": 0.0}


def encode(df, cats=None):
    X, y = df[NUMERIC].copy(), np.log(df["price"]).to_numpy()
    for c in ("has_5G", "has_NFC", "has_IR"):
        X[c] = X[c].map(HAS_FLAG).astype(float)
    order = cats
    if order is None:
        order = {c: [v for v in df[c].dropna().unique()] for c in CATEGORICAL}
    for c in CATEGORICAL:
        for val in order[c]:
            X[f"{c}_{val}"] = (df[c] == val).astype(float)
    return X, y, order


MODEL_HYPERPARAMS = {
    "n_estimators": 500, "max_depth": 7, "learning_rate": 0.03,
    "subsample": 0.7, "colsample_bytree": 0.4, "min_child_weight": 3,
    "reg_lambda": 0.1, "reg_alpha": 0.0, "max_bin": 128,
}


def main():
    df = load_and_clean()
    y_all = np.log(df["price"].to_numpy())

    order = {c: [v for v in df[c].dropna().unique()] for c in CATEGORICAL}

    gkf = GroupKFold(n_splits=5)
    oof = np.empty(len(df))
    for tr_idx, va_idx in gkf.split(df, groups=df["brand_name"]):
        Xtr, ytr, _ = encode(df.iloc[tr_idx], order)
        Xva, _, _ = encode(df.iloc[va_idx], order)
        m = xgb.XGBRegressor(**MODEL_HYPERPARAMS, random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, y_all[va_idx])], verbose=False)
        oof[va_idx] = m.predict(Xva)

    oof_price = np.exp(oof)
    print(f"\n5-fold CV (grouped by brand) MAPE: {mean_absolute_percentage_error(df['price'], oof_price) * 100:.1f}%")

    final_cols, y_final, order = encode(df)
    final_cols = final_cols.astype(float)
    print(f"features: {final_cols.shape[1]}")

    final = xgb.XGBRegressor(**MODEL_HYPERPARAMS, random_state=RANDOM_STATE, n_jobs=-1)
    final.fit(final_cols, y_final)

    OUT_DIR.mkdir(exist_ok=True)
    model_path = OUT_DIR / "phone_price_model_v3.pkl"
    cols_path = OUT_DIR / "model_columns_v3.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final, f)
    with open(cols_path, "wb") as f:
        pickle.dump(list(final_cols.columns), f)

    metrics = {
        "rows": len(df),
        "grouped_cv_mape_pct": round(mean_absolute_percentage_error(df["price"], oof_price) * 100, 1),
        "features": final_cols.shape[1],
        "price_min": float(df["price"].min()),
        "price_max": float(df["price"].max()),
        "hyperparameters": MODEL_HYPERPARAMS,
    }
    with open(OUT_DIR / "metrics_v3.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("saved:", model_path, cols_path, "| metrics:", metrics)
    print("columns:", list(final_cols.columns))


if __name__ == "__main__":
    main()