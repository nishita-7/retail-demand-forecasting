# 🛒 Retail Demand Forecasting

End-to-end machine learning project that forecasts daily store sales — from raw data and statistical analysis through a deployed, interactive forecasting app. Built on the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset (1,115 stores, ~844k trading days).

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://retail-demand-forecasting-psdjqymkbqjstvepgnpccu.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5252?logo=xgboost&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-13ADC7?logo=dvc&logoColor=white)
---

## The business problem

Retailers need accurate short-term sales forecasts to plan inventory, staffing and promotions. This project predicts each store's daily sales up to six weeks ahead, surfaces the drivers behind sales and serves those forecasts through both an interactive dashboard and a REST API.

## Headline results

| Metric | Value |
|---|---|
| Best model | **XGBoost** |
| Forecast accuracy (MAPE) | **~11%** — forecasts land within ~11% of actual sales |
| Models compared | Linear Regression, Random Forest, XGBoost |
| Validation method | Time-based holdout (last 6 weeks) — no future leakage |

## Key business insights

- **Promotions are the biggest controllable lever.** A two-sample t-test confirmed promotions raise average daily sales by **€2,299 per store (+38.8%)** — statistically significant (p < 0.001) and practically large (Cohen's d = 0.80).
- **December is the annual demand peak** — inventory should ramp ~6 weeks ahead.
- **Store type B dominates** (~€10,300/day vs ~€6,800 for other types) despite being a small segment.
- **Competition distance has a weak effect** — closer competitors don't meaningfully reduce sales (they cluster in high-footfall areas).
- **`Customers` was deliberately excluded** as a feature: it's unknown at prediction time and is a near-proxy for sales (data leakage).

## Architecture

```
Raw data → Clean (DVC) → Feature engineering (DVC) → Train + compare models (MLflow)
                                                              │
                                              best model → MLflow Registry (DagsHub)
                                                              │
                                            model.joblib → shared predict_raw()
                                                          ┌──────┴──────┐
                                              FastAPI API (Docker)   Streamlit dashboard
                                                                     (Streamlit Cloud)
```

A single `predict_raw()` function powers the batch script, the API, and the dashboard — so the serving logic can never drift from training.

## Tech stack

- **ML / stats:** Python, pandas, NumPy, scikit-learn, XGBoost, statsmodels, SciPy
- **MLOps:** DVC (pipeline + data versioning), MLflow (tracking + model registry), DagsHub (remote)
- **Serving:** FastAPI + Pydantic, Streamlit, Docker
- **Deployment:** Streamlit Community Cloud (dashboard); Docker-containerized API

## Project structure

```
retail-demand-forecasting/
├── data/raw/                  # Kaggle CSVs (download separately)
├── notebooks/
│   ├── 01_eda.ipynb           # exploration + business-framed EDA
│   ├── 02_hypothesis_tests.ipynb   # promo t-test (effect size + CI) + ANOVA
│   └── 03_modeling.ipynb      # model comparison + feature importance
├── src/
│   ├── data_prep.py           # load, merge, clean
│   ├── featurize.py           # feature engineering
│   ├── train.py               # train 3 models, log to MLflow, register best
│   └── predict.py             # shared prediction path (API + batch reuse it)
├── app/
│   ├── api.py                 # FastAPI service
│   └── dashboard.py           # Streamlit dashboard
├── params.yaml                # all configuration
├── dvc.yaml                   # reproducible pipeline (prepare → featurize → train)
├── Dockerfile                 # containerized API
└── requirements.txt
```

## Run it locally

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Get the data
# download train.csv + store.csv from the Kaggle Rossmann competition into data/raw/

# 3. Run the pipeline (clean → featurize → train + log to MLflow)
dvc repro

# 4. Launch the API
uvicorn app.api:app --reload      # http://127.0.0.1:8000/docs

# 5. Launch the dashboard
streamlit run app/dashboard.py    # http://localhost:8501
```

### Run the API in Docker

```bash
docker build -t retail-demand-api .
docker run -p 8000:8000 retail-demand-api
# open http://localhost:8000/docs
```

## Modeling notes

- **Target transform:** sales are right-skewed (skew 1.59), so the model trains on `log1p(Sales)` and inverts with `expm1` — errors are proportional, not dominated by a few large days.
- **Features:** calendar parts with cyclical (sin/cos) encoding, months-since-competition, log competition distance, an active-Promo2-month flag, and one-hot store attributes.
- **Validation:** a strict time-based split (train on the past, validate on the final 6 weeks) — the only correct approach for forecasting.

## Future work

- Hyperparameter tuning (Optuna) and per-store models for the highest-volume stores
- Reorder-point / safety-stock recommendations from the forecast
- Scheduled retraining as new sales data arrives
- Deploy the containerized API to a public host (e.g. AWS Cloud, Hugging Face Spaces)