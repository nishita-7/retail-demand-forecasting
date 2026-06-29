"""Shared prediction path: raw rows -> clean -> features -> euro predictions.
Used by BOTH a quick test-set run and the FastAPI app, so the serving logic is identical to training.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

# When run as `python src/predict.py`, src/ is on sys.path, so these siblings import cleanly.
from data_prep import clean
from featurize import build_features

PARAMS = yaml.safe_load(open("params.yaml"))
TARGET = PARAMS["features"]["target"]
LOG_TGT = PARAMS["features"]["log_target"]
CYCLICAL = PARAMS["features"].get("cyclical", [])

_model = None
_feature_cols = None


def _load():
    # Load model + feature order once and cache (so the API doesn't reload per request)
    global _model, _feature_cols
    if _model is None:
        _model = joblib.load("models/model.joblib")
        _feature_cols = list(json.load(open("models/feature_columns.json")).values())
    return _model, _feature_cols


def predict_raw(raw_df: pd.DataFrame, store_df: pd.DataFrame) -> np.ndarray:
    """Take raw rows (train/test schema) + store metadata, return euro sales predictions."""
    model, feature_cols = _load()
    df = raw_df.copy()

    # test.csv has a few missing Open values -> Kaggle convention: assume the store is open.
    if "Open" in df.columns:
        df["Open"] = df["Open"].fillna(1).astype(int)

    # Same merge + cleaning as training, but DON'T drop closed days — we still emit a prediction row for every input row (closed ones get forced to 0 below).
    cleaned = clean(df, store_df, drop_closed=False)
    open_mask = cleaned["Open"].eq(1).to_numpy() if "Open" in cleaned.columns else None

    feats = build_features(cleaned, cyclical_cols=CYCLICAL)

    # Align to the EXACT training feature set: missing one-hot columns -> 0, extras dropped, order matched. This is what makes serving robust to categories absent from the input.
    X = feats.reindex(columns=feature_cols, fill_value=0).astype(float)

    pred = model.predict(X)
    pred = np.expm1(pred) if LOG_TGT else np.asarray(pred)
    pred = np.clip(pred, 0, None)            # sales can't be negative
    if open_mask is not None:
        pred[~open_mask] = 0                  # a closed store sells nothing
    return pred


def main() -> None:
    raw = Path(PARAMS["data"]["raw_dir"])
    test = pd.read_csv(raw / "test.csv", parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(raw / "store.csv")

    preds = predict_raw(test, store)

    out = pd.DataFrame({"Id": test["Id"], "Sales": np.round(preds, 2)})
    out.to_csv("predictions.csv", index=False)
    print(f"wrote predictions.csv with {len(out):,} rows")
    print(out.head())
    print(f"\nsanity: mean predicted sales = EUR {preds[preds > 0].mean():,.0f} "
          f"| zeros (closed) = {(preds == 0).sum():,}")


if __name__ == "__main__":
    main()