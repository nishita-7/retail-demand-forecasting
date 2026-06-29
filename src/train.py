# Train candidate models, log each as an MLflow run, register the best one.

import json
from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv() 

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

REGISTERED_NAME = "rossmann-demand-forecaster"


def metrics_euros(y_true, y_pred, log_tgt):
    # All metrics in euros: invert the log BEFORE scoring, else they're log units.
    yt = np.expm1(y_true) if log_tgt else np.asarray(y_true)
    yp = np.expm1(y_pred) if log_tgt else np.asarray(y_pred)
    mask = yt > 0
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mape": float((np.abs((yt[mask] - yp[mask]) / yt[mask])).mean() * 100),
    }


def main(params_path: str = "params.yaml") -> None:
    params = yaml.safe_load(open(params_path))
    target = params["features"]["target"]
    log_tgt = params["features"]["log_target"]
    test_weeks = params["data"]["test_weeks"]
    m = params["model"]

    feat_path = Path(params["data"]["processed_path"]).parent / "features.parquet"
    df = pd.read_parquet(feat_path)
    feature_cols = list(json.load(open("models/feature_columns.json")).values())

    # Time-based split: last `test_weeks` weeks are validation (used to pick the winner)
    df = df.sort_values("Date")
    cutoff = df["Date"].max() - pd.Timedelta(weeks=test_weeks)
    tr, va = df[df["Date"] <= cutoff], df[df["Date"] > cutoff]
    X_tr, X_va = tr[feature_cols].astype(float), va[feature_cols].astype(float)
    y_tr = np.log1p(tr[target]) if log_tgt else tr[target]
    y_va = np.log1p(va[target]) if log_tgt else va[target]

    # The 3 candidates
    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=m["random_state"]
        ),
        "xgboost": xgb.XGBRegressor(
            n_estimators=m["n_estimators"], max_depth=m["max_depth"],
            learning_rate=m["learning_rate"], subsample=m["subsample"],
            colsample_bytree=m["colsample_bytree"], random_state=m["random_state"],
            tree_method="hist", n_jobs=-1,
        ),
    }
    
    # Use DagsHub if its tracking URI is set (via env vars), else fall back to local sqlite.
    if os.environ.get("MLFLOW_TRACKING_URI"):
        pass 
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("rossmann-demand-forecasting")

    runs = []
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_tr, y_tr)
            mets = metrics_euros(y_va, model.predict(X_va), log_tgt)

            mlflow.log_param("model_type", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(mets)

            log_model_fn = mlflow.xgboost.log_model if name == "xgboost" else mlflow.sklearn.log_model
            info = log_model_fn(model, artifact_path="model")  

            runs.append({"name": name, "run_id": run.info.run_id,
                         "model_uri": info.model_uri, **mets})

            runs.append({"name": name, "run_id": run.info.run_id, **mets})
            print(f"{name:>18}:  MAE EUR {mets['mae']:,.0f} | RMSE EUR {mets['rmse']:,.0f} | MAPE {mets['mape']:.1f}%")

    # winner = lowest validation MAPE
    best = min(runs, key=lambda r: r["mape"])
    print(f"\nbest model: {best['name']} (MAPE {best['mape']:.1f}%)")

    # Save the serving artifact FIRST — this is what the API loads, so it must always land.
    final = candidates[best["name"]]
    X_all = df[feature_cols].astype(float)
    y_all = np.log1p(df[target]) if log_tgt else df[target]
    final.fit(X_all, y_all)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(final, "models/model.joblib")
    print("saved models/model.joblib")

    # Register the winner using the URI MLflow returned (version-proof).
    # Wrapped so a registry hiccup never blocks the save above.
    try:
        mv = mlflow.register_model(best["model_uri"], REGISTERED_NAME)
        print(f"registered '{REGISTERED_NAME}' version {mv.version}")
    except Exception as e:
        print(f"registry step skipped ({e})")

if __name__ == "__main__":
    main()