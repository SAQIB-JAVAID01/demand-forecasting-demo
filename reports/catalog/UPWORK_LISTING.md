# Upwork portfolio listing

Every number below is computed by the code in this repository and shown in the app (final test: 25 Apr 2011 to
28 Nov 2011, 200 products, 6,400 out-of-sample product-weeks, 13-week average as the benchmark). Upwork limits for a
portfolio project: title 70 characters, role 100, description 600, five skill tags, images 1000 x 750.

## Title (59 characters)

Demand Forecasting & Inventory Planning Dashboard in Python

## Role (97 characters)

Data scientist and Python developer: data cleaning, forecasting, backtesting, desktop and web app

## Description (493 characters)

Weekly demand forecasts for 200 products of a UK online retailer (public data, about 1M sales rows), 1–4 weeks ahead, scored on 32 weeks the models never saw. Weekly totals: 81.4% accurate, bias −2.1% (13-week average: 80.7%, −8.8%), so level on accuracy but closer on level. A second setting adds +5.1 pts per product. Also: model lab, simulated inventory plan (91.5% fill rate vs 90.2%), 26-week what-if scenarios and a replay of recorded sales. PyQt6 desktop app plus Streamlit web version.

## Skills (5)

Demand Forecasting · Time Series Analysis · Inventory Management · Python · Dashboard

## Images and video (upload in this order)

1. Cover: `reports/catalog/cover_1000x750.png`
2. Executive summary (opens on "Unbiased totals"): `reports/screens/1_executive_summary.png`
3. The other setting, side by side: `reports/screens/1b_summary_per_product_accuracy.png`
4. Live replay of recorded sales: `reports/screens/2_live_replay.png`
5. Model lab: `reports/screens/7_model_lab.png`
6. Sales projection, what-if scenarios: `reports/screens/8_sales_projection.png`
7. Video, 22 seconds: `reports/demo/live_replay_demo.mp4` (replay of recorded sales, not live data)

## Longer version (for a linked page or a Project Catalog listing)

A demand forecasting and inventory planning dashboard built in Python on two years of real transactions from a UK
online gift retailer (public data, about 1 million rows). It forecasts weekly units for 200 products, 1 to 4 weeks
ahead, and scores every forecast on weeks the models never saw.

**What it does**

- Weekly product forecasts with an 80% prediction range, compared with the simple method a planner would use anyway
  (13-week average).
- Rolling backtest with 13 forecast dates: choices were made on the first 5, the last 8 are the final test every
  screen reports.
- Model lab: LightGBM, XGBoost, Holt exponential smoothing, TSB / Croston for intermittent demand, SARIMAX and an
  ensemble on the same test, with SHAP feature importance.
- Inventory simulation: the same reorder policy run with the forecast and with the benchmark, with editable lead
  time, service target, unit cost and holding rate.
- 26-week sales projection with prediction intervals and what-if economic scenarios (recession, base, boom).
- Replay of recorded sales week by week with stock-risk alerts, labelled as a replay.
- PyQt6 desktop app (dark mode) and a Streamlit web version that uses the same KPI code.

**Results on the final test (out of sample)**

- "Unbiased totals" setting (default): weekly totals 81.4% accurate, bias −2.1%. The benchmark is 80.7% accurate
  and 8.8% low. On accuracy the two are level with each other (95% range of the difference −4.0 to +5.1 pts); the
  improvement is in the level.
- "Best per-product accuracy" setting: 46.7% per product-week, +5.1 pts over the benchmark (95% range +3.7 to
  +6.8). It runs 23.2% low on totals because it aims at the typical week, so the app shows both settings side by
  side and says which to use for which decision.
- Inventory simulation (1-week lead time, 95% service target): "Unbiased totals" fills 91.5% of demand against
  90.2% for the benchmark, about £59,000 a year of lost sales avoided; "Best per-product accuracy" needs 11.3% less
  stock for the benchmark's fill rate. Simulation on recorded history with stated assumptions, not measured savings.

**How the numbers are kept honest**

- Products were selected from the first year only; automated tests change the future and confirm that no model
  input moves.
- Every gain comes with a 95% range, and the app says "level with the benchmark" when the range includes zero.
- Weak spots stay on screen: single gift products are hard to forecast week by week for every method.
- Scenarios are what-if assumptions and are labelled that way.

**What I can build for you**

The same pipeline on your sales data: cleaning, a backtest against your current method, forecasts by product, store
or category, safety stock and reorder points, and a dashboard your planners can use.

**Tools:** Python, pandas, LightGBM, XGBoost, statsmodels, scikit-learn, SciPy, PyQt6, pyqtgraph, Streamlit, Altair,
pytest.
