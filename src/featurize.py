# Turn the cleaned data into model-ready features.

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _cyclical(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """Encode a cyclical integer column as sin/cos.

    Month 12 (Dec) and month 1 (Jan) are adjacent in reality but look far apart
    as plain numbers. sin/cos on a circle restores that adjacency for the model.
    """
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df


def build_features(df: pd.DataFrame, cyclical_cols=None) -> pd.DataFrame:
    """Engineer features. Pure function -> reusable in training AND the API."""
    df = df.copy()

    # calendar parts from the Date (the model can't read a raw timestamp)
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["Date"].dt.dayofweek + 1  # 1=Mon..7=Sun

    # cyclical encoding for the periodic columns
    periods = {"month": 12, "day_of_week": 7, "week_of_year": 52}
    for col in (cyclical_cols or []):
        if col in periods:
            df = _cyclical(df, col, periods[col])

    # months since the nearest competitor opened (0-filled rows = unknown -> 0)
    has_comp = df["CompetitionOpenSinceYear"] > 0
    df["months_since_competition"] = np.where(
        has_comp,
        12 * (df["year"] - df["CompetitionOpenSinceYear"])
        + (df["month"] - df["CompetitionOpenSinceMonth"]),
        0,
    ).clip(min=0)

    # log-compress the long, weak competition-distance tail (confirmed weak in EDA)
    df["competition_distance_log"] = np.log1p(df["CompetitionDistance"])

    # is this month an active Promo2 month for the store?
    month_abbr = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                  7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    df["month_abbr"] = df["month"].map(month_abbr)
    df["is_promo2_month"] = df.apply(
        lambda r: int(str(r["month_abbr"]) in str(r["PromoInterval"]).split(",")),
        axis=1,
    )

    # one-hot encode the unordered categoricals
    df = pd.get_dummies(df, columns=["StoreType", "Assortment", "StateHoliday"],
                        prefix=["stype", "assort", "stateholiday"])

    # get_dummies makes bool columns; cast to int so every model + parquet is happy
    bool_cols = df.select_dtypes("bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # drop leakage + raw columns we've replaced. Note: Date is KEPT (needed for the
    # time-based train/val split) but excluded from the model features in main().
    # Customers is dropped: unknown at predict time + near-proxy for Sales = leakage.
    drop_cols = ["Customers", "Open", "month_abbr",
                 "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                 "PromoInterval", "Promo2SinceWeek", "Promo2SinceYear"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def main(params_path: str = "params.yaml") -> None:
    params = yaml.safe_load(open(params_path))
    in_path = Path(params["data"]["processed_path"])
    out_path = in_path.parent / "features.parquet"
    target = params["features"]["target"]
    cyclical = params["features"].get("cyclical", [])

    df = pd.read_parquet(in_path)
    feats = build_features(df, cyclical_cols=cyclical)

    # model features = everything except the target AND Date (Date is for splitting only)
    feature_cols = [c for c in feats.columns if c not in (target, "Date")]
    Path("models").mkdir(exist_ok=True)
    pd.Series(feature_cols).to_json("models/feature_columns.json")

    feats.to_parquet(out_path, index=False)
    print(f"saved {len(feats):,} rows x {feats.shape[1]} cols -> {out_path}")
    print(f"feature count (excl. target & Date): {len(feature_cols)}")


if __name__ == "__main__":
    main()