"""Load, merge, and clean the Rossmann data.

This is the production version of the cleaning you explore in 01_eda.ipynb.
It lives in a module so that:
  1. The DVC pipeline calls it as the `prepare` stage.
  2. The FastAPI app imports the same `clean()` logic at predict time,
     so the exact same transforms run at train AND serve time (no skew).

Run standalone:  python src/data_prep.py
Run via DVC:     dvc repro prepare
"""

from pathlib import Path

import pandas as pd
import yaml

def load_raw(raw_dir: Path):
    """Read the two source files."""
    # parse_dates: we derive month / weekday / seasonality features later, so Date must be a real datetime from the start (avoids re-parsing).
    train = pd.read_csv(raw_dir / "train.csv", parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(raw_dir / "store.csv")
    return train, store


def clean(train: pd.DataFrame, store: pd.DataFrame, drop_closed: bool = True) -> pd.DataFrame:
    """Merge store metadata onto sales rows and handle the dataset's quirks.

    Kept as a standalone function so the API can call clean() on a single incoming row with the identical logic used in training.
    """
    train = train.copy()
    train["StateHoliday"] = train["StateHoliday"].astype(str)

    # Left join: train is the source of truth (one row per store-day) -> guarantees we never drop a sales record even if metadata were missing.
    df = train.merge(store, on="Store", how="left")

    if drop_closed:
        # Closed days record Sales=0 -> that's "no opportunity", not "no demand".
        # Training on them teaches the store calendar, not buying behaviour.
        df = df[df["Open"] == 1].copy()

    # Missingness here is structural, not random -> fill by meaning, not a blanket median.
    # Promo2* fields are NaN only when the store isn't in Promo2 -> the truthful fill is 0 / "none".
    for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
        df[col] = df[col].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("none")

    # CompetitionDistance: a few genuine unknowns -> median is a safe neutral fill.
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())

    # Unknown competitor open date -> 0 (the model can still lean on CompetitionDistance).
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
        df[col] = df[col].fillna(0)

    return df


def main(params_path: str = "params.yaml") -> None:
    params = yaml.safe_load(open(params_path))
    raw_dir = Path(params["data"]["raw_dir"])
    out_path = Path(params["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train, store = load_raw(raw_dir)
    df = clean(train, store, drop_closed=params["data"]["drop_closed"])

    # parquet keeps dtypes (dates stay dates) and is far smaller/faster than csv.
    df.to_parquet(out_path, index=False)
    print(f"saved {len(df):,} rows x {df.shape[1]} cols -> {out_path}")


if __name__ == "__main__":
    main()
