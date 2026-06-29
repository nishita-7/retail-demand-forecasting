"""Streamlit dashboard — the clickable demo."""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from predict import predict_raw  # noqa: E402

st.set_page_config(page_title="Retail Demand Forecast", page_icon="📈", layout="centered")


@st.cache_data
def load_stores():
    return pd.read_csv(ROOT / "data" / "raw" / "store.csv")


STORE_DF = load_stores()
STORE_IDS = sorted(int(s) for s in STORE_DF["Store"].unique())


def make_row(store_id, d, promo, school=0, state="0"):
    ts = pd.Timestamp(d)
    return {"Store": store_id, "Date": ts, "DayOfWeek": ts.dayofweek + 1,
            "Open": 1, "Promo": promo, "StateHoliday": str(state), "SchoolHoliday": school}


def forecast(rows):
    return predict_raw(pd.DataFrame(rows), STORE_DF)


st.title("📈 Retail Demand Forecasting")
st.caption("Daily store-sales forecasts from an XGBoost model (Rossmann dataset).")

# Inputs 
c1, c2 = st.columns(2)
with c1:
    default_ix = STORE_IDS.index(262) if 262 in STORE_IDS else 0
    store_id = st.selectbox("Store", STORE_IDS, index=default_ix)
    fdate = st.date_input("Date", value=date(2015, 9, 17))
with c2:
    promo = st.toggle("Promotion running", value=False)
    school = st.toggle("School holiday", value=False)

promo_i, school_i = int(promo), int(school)

# Single-day prediction + live promo lift 
pred_sel     = forecast([make_row(store_id, fdate, promo_i, school_i)])[0]
pred_promo   = forecast([make_row(store_id, fdate, 1, school_i)])[0]
pred_nopromo = forecast([make_row(store_id, fdate, 0, school_i)])[0]
lift = pred_promo - pred_nopromo
lift_pct = (lift / pred_nopromo * 100) if pred_nopromo > 0 else 0

m1, m2 = st.columns(2)
m1.metric("Predicted sales", f"€{pred_sel:,.0f}")
m2.metric("Promo lift (this day)", f"€{lift:,.0f}", f"{lift_pct:.0f}%")

# 14-day forecast chart
st.subheader("14-day forecast")
days = [fdate + timedelta(days=i) for i in range(14)]
base_rows  = [make_row(store_id, d, 0, school_i) for d in days]
promo_rows = [make_row(store_id, d, 1, school_i) for d in days]
chart_df = pd.DataFrame({
    "date": days,
    "no promo": forecast(base_rows),
    "with promo": forecast(promo_rows),
}).set_index("date")
st.line_chart(chart_df)

total_base = chart_df["no promo"].sum()
st.caption(f"14-day total (no promo): €{total_base:,.0f}  ·  "
           f"with promo every day: €{chart_df['with promo'].sum():,.0f}")