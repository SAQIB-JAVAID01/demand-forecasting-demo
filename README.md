# Demand forecasting for a UK online retailer

Weekly unit forecasts, 1 to 4 weeks ahead, for the 200 highest-revenue products of a UK online gift retailer,
with an honest backtest against simple methods.

**Full write-up with all numbers and charts: [reports/REPORT.md](reports/REPORT.md)**

![Cover](reports/figures/00_cover.png)

## What is in here

| Path | What it does |
|---|---|
| `src/prepare.py` | Cleans 1.07M raw transactions (cancellations, non-product codes, sheet overlap) into a weekly product panel |
| `src/features.py` | Builds model inputs that only use data available at forecast time |
| `src/backtest.py` | Rolling-origin backtest: 5 simple methods and LightGBM, 13 forecast dates |
| `src/evaluate.py` | WAPE, bias, bootstrap confidence intervals, segment breakdowns, interval calibration |
| `src/report.py` | Charts and `reports/REPORT.md`, generated from the results |
| `app/dashboard.py` | Streamlit dashboard for planners |
| `tests/` | Tests that change the future and check the model inputs do not move |

## How the test is kept honest

- Products are selected from the first year only.
- The model setting and the reference baseline are chosen on the first 5 forecast dates. The last 8 are the final
  test and are not used for any choice.
- The gain over the baseline is reported with a 95% confidence interval.
- Weak spots are reported, including a forecast bias and the segments where the gain is small.

## Run it

```
pip install -r requirements.txt
python run_all.py            # download, clean, backtest, score, report (about 10 minutes)
python -m pytest tests -q
streamlit run app/dashboard.py
```

## Data

UCI Machine Learning Repository, "Online Retail II" (Chen, 2019), licence CC BY 4.0.
https://archive.ics.uci.edu/dataset/502/online+retail+ii
The raw file is downloaded by `run_all.py` and is not stored in this repository.
