"""UK economic indicators used by the model lab and the sales-projection page.

Run:  python -m src.macro
Downloads four monthly series from FRED (Federal Reserve Bank of St. Louis), which republishes OECD and ONS
data, and caches them in data/external/uk_macro_monthly.csv. Later runs use the cache, so no network is needed.

Series
------
cpi            GBRCPIALLMINMEI     Consumer price index, all items, UK (OECD, 2015 = 100)
unemployment   LRHUTTTTGBM156S     Harmonised unemployment rate, UK, % (OECD)
confidence     CSCICP03GBM665S     Consumer confidence indicator, UK, amplitude adjusted, long-term average = 100 (OECD)
retail_volume  GBRSLRTTO02IXOBSAM  Retail trade volume index, UK, seasonally adjusted (OECD, 2015 = 100)

Leakage rule: a monthly figure is published several weeks after the month ends, so a week may only use the
figure for a month that ended at least PUBLICATION_LAG_DAYS before it (see `weekly`).
"""
from __future__ import annotations

import io
import urllib.request

import pandas as pd

from . import config as C

EXTERNAL_DIR = C.ROOT / "data" / "external"
CACHE = EXTERNAL_DIR / "uk_macro_monthly.csv"
SERIES = {
    "cpi": "GBRCPIALLMINMEI",
    "unemployment": "LRHUTTTTGBM156S",
    "confidence": "CSCICP03GBM665S",
    "retail_volume": "GBRSLRTTO02IXOBSAM",
}
SOURCE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd=2008-01-01&coed=2012-12-31"
PUBLICATION_LAG_DAYS = 45


def download() -> pd.DataFrame:
    frames = []
    for name, sid in SERIES.items():
        with urllib.request.urlopen(SOURCE_URL.format(id=sid), timeout=60) as r:
            df = pd.read_csv(io.BytesIO(r.read()))
        df.columns = ["month", name]
        df["month"] = pd.to_datetime(df["month"])
        frames.append(df.set_index("month"))
    out = pd.concat(frames, axis=1).sort_index()
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE)
    return out


def monthly() -> pd.DataFrame:
    """Monthly indicators plus inflation (CPI change over 12 months, %)."""
    df = pd.read_csv(CACHE, parse_dates=["month"]).set_index("month") if CACHE.exists() else download()
    df["inflation"] = 100 * (df["cpi"] / df["cpi"].shift(12) - 1)
    return df


def weekly(weeks: pd.Series | pd.DatetimeIndex, lag_days: int = PUBLICATION_LAG_DAYS) -> pd.DataFrame:
    """The newest monthly figure that was public in each week (month end + lag <= week start)."""
    m = monthly()
    avail = m.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag_days)
    m = m.set_axis(avail)
    w = pd.DatetimeIndex(pd.Series(weeks).drop_duplicates().sort_values())
    out = m.reindex(m.index.union(w)).ffill().reindex(w)
    out.index.name = "week"
    return out


if __name__ == "__main__":
    print(download().tail())
