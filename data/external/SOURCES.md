# External data

`uk_macro_monthly.csv` is downloaded by `python -m src.macro` from FRED (Federal Reserve Bank of St. Louis),
which republishes OECD Main Economic Indicators for the United Kingdom:

| Column | FRED id | Description |
|---|---|---|
| cpi | GBRCPIALLMINMEI | Consumer price index, all items, UK, 2015 = 100 (OECD) |
| unemployment | LRHUTTTTGBM156S | Harmonised unemployment rate, UK, % of labour force (OECD) |
| confidence | CSCICP03GBM665S | Consumer confidence indicator, UK, amplitude adjusted, long-term average = 100 (OECD) |
| retail_volume | GBRSLRTTO02IXOBSAM | Retail trade volume index, UK, seasonally adjusted, 2015 = 100 (OECD) |

Links: https://fred.stlouisfed.org/series/GBRCPIALLMINMEI, https://fred.stlouisfed.org/series/LRHUTTTTGBM156S,
https://fred.stlouisfed.org/series/CSCICP03GBM665S, https://fred.stlouisfed.org/series/GBRSLRTTO02IXOBSAM

UK bank holidays for 2009 to 2012 (England and Wales) are listed in `src/features_ext.py`, from the UK
government's published bank holiday dates (gov.uk/bank-holidays).
