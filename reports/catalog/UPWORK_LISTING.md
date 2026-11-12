# Upwork catalog listing

Every number below is shown in the app and comes from the rolling backtest in this repository
(final test: 25 Apr 2011 to 28 Nov 2011, 200 products, 6,400 out-of-sample product-weeks).

## Title

**Demand Forecasting & Inventory Planning Dashboard — Weekly SKU Forecasts, Rolling Backtest, and What-If Sales Scenarios**

Short version, if the field has a character limit (65 characters):

**Demand Forecasting & Inventory Planning Dashboard (Python)**

## Description

A demand forecasting and inventory planning dashboard built in Python on two years of real transactions from a UK
online gift retailer (public data, about 1 million rows). It forecasts weekly units for 200 products, 1 to 4 weeks
ahead, and tests every forecast on weeks the model never saw.

**What it does**

- Weekly SKU forecasts with an 80% prediction range, compared against simple methods a planner would use anyway
  (13-week average, same week last year).
- Rolling backtest with 13 forecast dates. Model choices were made on the first 5; the last 8 are the final test that
  every screen reports.
- Model lab: LightGBM, XGBoost, exponential smoothing (Holt), TSB / Croston for intermittent demand, SARIMAX and an
  ensemble, all on the same test, with feature importance (SHAP).
- Inventory simulation: the same reorder policy run with the forecast and with the benchmark, with editable lead
  time, service target, unit cost and holding rate.
- 26-week sales projection with what-if economic scenarios (base, recession, boom) and a price-change slider.
- Live replay of recorded sales week by week, with stock-risk alerts. It is a replay of history, labelled as such.
- Desktop app (PyQt6, dark mode) and a lighter web version (Streamlit).

**Results on the final test (out of sample)**

- Weekly totals: 81.4% accurate, with the forecast level within 2.1% of the units actually sold. The 13-week average
  benchmark is 80.7% accurate and 8.8% low. On accuracy the two are level (the 95% range of the difference includes
  zero); the gain is in the level.
- Per product: a second setting tuned for product-level accuracy reaches 46.7%, which is 5.1 points above the
  benchmark (95% range +3.7 to +6.8). It runs 23.2% low on totals because it aims at the typical week, so the
  dashboard shows both settings side by side and says which to use for which decision.
- Inventory simulation (1-week lead time, 95% service target): 91.5% of demand filled against 90.2% with the
  benchmark, and about £59,000 a year of lost sales avoided. The product-level setting needs 11.3% less stock for the
  same fill rate. These are simulation results with stated assumptions, not measured savings.

**How the numbers are kept honest**

- Products were selected from the first year only, and automated tests change the future and confirm that no model
  input moves.
- Gains are reported with 95% ranges, and the dashboard says "level with the benchmark" when a range includes zero.
- Weak spots are on screen: single gift products are hard to forecast week by week for every method.
- Scenarios are what-if assumptions and are labelled that way.

**What I can build for you**

The same pipeline on your sales data: data cleaning, a backtest against your current method, forecasts by SKU, store
or category, safety stock and reorder points, and a dashboard your planners can use.

**Tools**

Python, pandas, LightGBM, XGBoost, statsmodels, scikit-learn, SciPy, PyQt6, pyqtgraph, Streamlit, Altair, pytest.

## Skills / tags

Demand Forecasting · Time Series Analysis · Inventory Management · Python · Machine Learning · Data Visualization ·
Dashboard · LightGBM · Supply Chain Analytics · Streamlit

## Images and video

- Cover: `reports/catalog/cover_1000x750.png`
- Executive summary: `reports/screens/1_executive_summary.png`
- The second setting: `reports/screens/1b_summary_per_product_accuracy.png`
- Live replay: `reports/screens/2_live_replay.png`
- Model lab: `reports/screens/7_model_lab.png`
- Sales projection: `reports/screens/8_sales_projection.png`
- Video: `reports/demo/live_replay_demo.mp4`
