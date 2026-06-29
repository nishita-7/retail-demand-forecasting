"""FastAPI serving layer.

A thin wrapper over predict_raw() — the API does NOT re-implement any prediction
logic, it builds a one-row request and hands it to the same function training/
batch use. Caller sends the essentials (store, date, promo); the store's
attributes are merged in automatically by clean() inside predict_raw().

"""

import sys
from datetime import date as date_type
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Make src/ importable regardless of where uvicorn is launched from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from predict import predict_raw  # noqa: E402

app = FastAPI(
    title="Retail Demand Forecasting API",
    description="Forecast daily store sales. Send a store, date, and whether a promo runs.",
    version="1.0",
)

# Store metadata is small (~1,115 rows) — load once at startup. predict_raw()'s clean() merges these onto each incoming request, so the caller never sends them.
STORE_DF = pd.read_csv(ROOT / "data" / "raw" / "store.csv")
VALID_STORES = set(STORE_DF["Store"].unique())


class ForecastRequest(BaseModel):
    store_id: int = Field(..., examples=[259], description="Store ID")
    date: date_type = Field(..., examples=["2015-08-01"], description="Date to forecast (YYYY-MM-DD)")
    promo: int = Field(0, ge=0, le=1, description="1 if a promo runs that day, else 0")
    school_holiday: int = Field(0, ge=0, le=1, description="1 if a school holiday, else 0")
    state_holiday: str = Field("0", description="'0'=none, 'a'=public, 'b'=Easter, 'c'=Christmas")


class ForecastResponse(BaseModel):
    store_id: int
    date: date_type
    predicted_sales: float


@app.get("/")
def health():
    return {"status": "ok", "stores_loaded": len(VALID_STORES)}


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    if req.store_id not in VALID_STORES:
        raise HTTPException(status_code=404, detail=f"Unknown store_id {req.store_id}")

    ts = pd.Timestamp(req.date)
    # Build a single raw row in the dataset's schema; predict_raw + clean do the rest.
    raw = pd.DataFrame([{
        "Store": req.store_id,
        "Date": ts,
        "DayOfWeek": ts.dayofweek + 1,   # 1=Mon..7=Sun
        "Open": 1,                        # forecasting a trading day
        "Promo": req.promo,
        "StateHoliday": str(req.state_holiday),
        "SchoolHoliday": req.school_holiday,
    }])

    pred = float(predict_raw(raw, STORE_DF)[0])
    return ForecastResponse(store_id=req.store_id, date=req.date, predicted_sales=round(pred, 2))