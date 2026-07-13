# Retail Demand Forecasting on Walmart M5 (PySpark + Databricks)

End-to-end demand forecasting pipeline built on the **Walmart M5** dataset using **PySpark** on **Databricks**. Starting from 59M raw sales records, the project engineers time-series features and trains a Gradient Boosted Trees model that beats a naive baseline by **20% on RMSE**.

**TL;DR:** Predicted daily unit sales for 30,490 item–store series across 3 US states. GBT model reached **Test RMSE 2.14 / MAE 1.03**, outperforming a naive lag-7 baseline (RMSE 2.68) by **20%**. At the aggregate level, daily total demand is predicted within **4.1% mean absolute error**.

---

## Results

| Model | Test RMSE | Test MAE |
|---|---|---|
| Naive baseline (last week's sales) | 2.68 | 1.24 |
| **Gradient Boosted Trees** | **2.14** | **1.03** |
| Improvement | **−20.1%** | **−17.5%** |

At the **aggregate** (all-products daily total) level, the model is far more accurate than on individual series — mean absolute error of **4.1%** and a small **−1.2%** bias.

---

## Dataset

The [M5 Forecasting dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy) contains 5+ years of daily Walmart sales:

- **30,490** item–store combinations (3,049 products × 10 stores)
- **1,941 days** of history (2011-01-29 → 2016-05-22)
- **3 categories** (FOODS, HOUSEHOLD, HOBBIES), **7 departments**, **3 states** (CA, TX, WI)
- Auxiliary tables: daily calendar (events, SNAP benefit days) and weekly item prices

After reshaping from wide to long format, the working dataset is **~59M rows**.

---

## Pipeline

The project runs entirely on Spark, in two notebooks:

**`m5_01_data_loading`** — data engineering
- Read 3 CSVs with **explicit schemas** (avoids costly `inferSchema` on 1,947-column sales file)
- **Unpivot** the wide sales table (1,941 day-columns → long format, 59M rows)
- **Broadcast join** with the tiny calendar table; **sort-merge join** with the 6.8M-row price table
- Feature engineering via **window functions**: lag (7, 28), rolling mean/std (7, 28), price lags & change ratio, SNAP and event flags
- Persist to a **Delta table** for fast downstream reads

**`m5_02_modeling`** — machine learning
- **Chronological** train/test split (last 28 days held out — no leakage)
- MLlib `Pipeline`: `StringIndexer` → `OneHotEncoder` → `VectorAssembler` → `GBTRegressor`
- Trained on a recent 2-year window (addresses distribution shift; ~21.7M rows)
- Evaluation against a naive baseline, feature importance, and error analysis

---

## Key Findings

### 1. Past sales dominate the signal
The top four features — all derived from sales history — account for **79%** of total feature importance. The 7-day rolling mean alone contributes 40%.

![Feature importance](images/fig1_feature_importance.png)

### 2. The model tracks trends but shaves peaks
On individual products, the model follows the overall shape of demand but **systematically under-predicts spikes** — a known consequence of minimizing squared error on zero-inflated data (68% of rows have zero sales).

![Forecast vs actual](images/fig2_forecast_vs_actual.png)

### 3. Errors cancel at the aggregate level
Individual under-predictions offset each other when summed. Daily **total** demand is predicted within **4.1%** mean absolute error and only **−1.2%** bias — mirroring a core principle of hierarchical forecasting: aggregate forecasts are far more reliable than item-level ones, which is exactly what warehouse-level inventory planning needs.

![Aggregate forecast](images/fig3_aggregate.png)

### 4. Under-prediction is systematic, not random
Residuals (prediction − actual) on non-zero sales are centered at **−0.79**, confirming the conservative bias observed above.

![Residual distribution](images/fig4_residuals.png)

---

## Techniques Demonstrated

- **Distributed data processing** on 59M rows with PySpark
- **Wide-to-long reshaping** with `unpivot`
- **Join strategy**: broadcast vs sort-merge, chosen by table size
- **Window functions** for time-series lag and rolling features (with leakage-safe framing that excludes the current row)
- **Delta Lake** for materialized, fast-read intermediate storage
- **MLlib Pipelines** with proper categorical encoding
- **Leakage-free chronological splitting** for time-series validation
- **Baseline-relative evaluation** and multi-level error analysis

---

## Stack

Python · PySpark · Spark MLlib · Delta Lake · Databricks · Matplotlib

## Next Steps

- **Tweedie / quantile loss** to correct the under-prediction bias on demand peaks
- **Two-stage model** (zero vs non-zero, then count) for the 68% intermittent-demand rows
- **Hyperparameter tuning** with time-series cross-validation
- **Item-level intro-date handling** to distinguish "not yet on shelf" from genuine zeros

## Reproducing

1. Download the [M5 dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and upload `calendar.csv`, `sales_train_evaluation.csv`, `sell_prices.csv` to a Databricks volume.
2. Run `m5_01_data_loading` to build the feature table.
3. Run `m5_02_modeling` to train and evaluate.

Built and run on **Databricks Free Edition** (serverless compute).