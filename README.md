# Retail Demand Forecasting — Walmart M5 (PySpark)

An end-to-end demand forecasting pipeline on the **Walmart M5** dataset, built with **PySpark** on **Databricks**. The goal is to predict daily unit sales for 30,490 item–store series and quantify how much a Gradient Boosted Trees model with engineered time-series features improves over a naive baseline — on data where **68% of item-days have zero sales**.

---

## Results

| Model | Test RMSE | Test MAE |
|-------|:---------:|:--------:|
| Naive baseline — repeat last week | 2.68 | 1.24 |
| **Gradient Boosted Trees** | **2.14** | **1.03** |

The GBT model improves over the naive baseline by **20.1% (RMSE)** and **17.5% (MAE)**. At the aggregate level, daily total demand is predicted within **4.1% mean absolute error**.

Evaluated on a held-out final 28 days (Apr 25 – May 22, 2016) — a strict chronological split with no future leakage.

---

## Dataset

- **M5 Forecasting** — 5+ years of daily Walmart sales (2011-01-29 → 2016-05-22, 1,941 days)
- **30,490 series**: 3,049 products × 10 stores across CA, TX, WI
- **3 categories** (FOODS, HOUSEHOLD, HOBBIES), 7 departments
- Auxiliary tables: daily calendar (holidays, sporting events, SNAP benefit days) and weekly item prices
- Highly intermittent demand: median 0, mean 1.13, max 763 — **68% zeros**

After unpivoting from wide format (1,941 day-columns) to long format: **~59M rows**.

---

## Pipeline

**`m5_01_data_loading`** — data engineering

- Read 3 CSVs with **explicit schemas** (no double-scan from `inferSchema` on a 1,947-column file)
- **Unpivot** wide sales table → 59M-row long format
- **Broadcast join** with calendar (tiny table); **sort-merge join** with 6.8M-row price table
- Feature engineering via **window functions**: sales lags (7, 28), rolling mean/std (7, 28), price lag & change ratio, SNAP and event flags
- Rolling frames **exclude the current row** — no target leakage
- Persist to a **Delta table** for fast downstream reads

**`m5_02_modeling`** — machine learning

- **Chronological split**: last 28 days held out (the actual forecast horizon)
- MLlib Pipeline: `StringIndexer → OneHotEncoder → VectorAssembler → GBTRegressor`
- Trained on the most recent 2 years (~21.7M rows) to address upward demand drift
- Evaluated against a naive lag-7 baseline, plus feature importance and error analysis

---

## Feature Importance

<img width="1187" height="709" alt="fig1_feature_importance" src="https://github.com/user-attachments/assets/f82ed147-a5e5-46ab-9576-e12aa173b33e" />

The top four features — all derived from sales history — account for **79%** of total importance. The 7-day rolling mean alone contributes **40%**: in demand forecasting, the recent past is the best predictor of the near future. Price, store identity and SNAP days act as refinements, not drivers.

---

## Forecast vs Actual

<img width="1429" height="1137" alt="fig2_forecast_vs_actual" src="https://github.com/user-attachments/assets/01c9321a-1a25-41b6-b9e2-b55a5e3fc2b8" />

The model tracks steady mid-volume products closely (middle panel) but **systematically under-predicts demand spikes** — the expected cost of minimizing squared error on zero-inflated data, where the safest guess is always a low one.

---

## Aggregate Accuracy

<img width="1549" height="589" alt="fig3_aggregate" src="https://github.com/user-attachments/assets/fb181dbd-51a5-4c53-bf6e-61f7a1b24025" />

Summed across all 30,490 series, individual errors cancel: daily total demand lands within **4.1%** on average with only **−1.2%** bias. Aggregate forecasts are far more reliable than item-level ones — exactly the level at which warehouse inventory planning operates.

---

## Residual Analysis

<img width="1309" height="589" alt="fig4_residuals" src="https://github.com/user-attachments/assets/268402e9-f2a3-419f-9276-f3868c450db6" />

Residuals on non-zero-sales days center at **−0.79 units** rather than zero — the under-prediction is systematic, not noise, and points directly at the loss function as the next lever.

---

## Summary

**What the model gets right:**

- **+20% over naive baseline** — engineered lag/rolling features add real signal
- **Strong aggregate accuracy (4.1%)** — reliable at the inventory-planning level
- **Leakage-free by construction** — chronological split, current-row-excluded rolling windows

**Where it falls short, and why:**

- **Peak shaving** — squared-error loss + 68% zeros pushes predictions low
- **Fix**: a Tweedie or quantile objective would trade some over-forecasting for catching peaks — usually the right trade-off, since a missed sale costs more than a spare unit on the shelf
- **Two-stage design** (sell / don't sell, then quantity) would address zero-inflation head-on

---

## Stack

Python · PySpark · Spark MLlib · Delta Lake · Databricks · Matplotlib

---

## Usage

1. Download the [M5 dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and upload `calendar.csv`, `sales_train_evaluation.csv`, `sell_prices.csv` to a Databricks volume
2. Run `m5_01_data_loading` to build the feature table
3. Run `m5_02_modeling` to train and evaluate

Runs end-to-end on **Databricks Free Edition** (serverless compute).
