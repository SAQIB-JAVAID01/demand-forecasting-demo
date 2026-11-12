"""Step 2: leakage-safe features.

For a forecast made h weeks ahead, the newest sales figure we are allowed to
use for target week t is week t-h. Every history feature below is therefore
built from `shift(h)`: nothing after week t-h can enter a row for week t.
`tests/test_no_leakage.py` checks this by changing the future and confirming
the features do not move.

Calendar features (week of year, trading days, weeks to Christmas) describe the
target week itself. They come from the business calendar, which a planner knows
ahead of time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_FEATURES = [
    "last", "lag_1", "lag_2", "lag_3", "mean_4", "mean_8", "mean_13", "mean_26",
    "std_8", "max_13", "nonzero_8", "weeks_since_sale", "same_week_last_year",
    "trend_4_13", "price_last", "price_ratio", "orders_mean_4",
    "total_last", "total_mean_4", "total_yoy", "share_13", "trading_days_last",
    "age_weeks",
]
CALENDAR_FEATURES = ["weekofyear", "month", "weeks_to_xmas", "trading_days"]
FEATURES = HISTORY_FEATURES + CALENDAR_FEATURES


def _roll(s: pd.Series, by: pd.Series, window: int, fn: str, min_periods: int = 1) -> pd.Series:
    r = s.groupby(by, sort=False).rolling(window, min_periods=min_periods)
    return getattr(r, fn)().reset_index(level=0, drop=True)


def build_features(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """Return one row per (sku, week) with features for an h-week-ahead forecast.

    `panel` must be sorted by sku, week and have a complete weekly grid per sku.
    """
    p = panel[["sku", "week", "t", "units", "orders", "price", "trading_days", "closed"]].copy()
    by = p["sku"]
    g = p.groupby("sku", sort=False)

    known = g["units"].shift(h)                      # newest usable sales figure
    f = pd.DataFrame({"sku": p["sku"], "week": p["week"], "t": p["t"], "h": h})
    f["last"] = known
    for k in (1, 2, 3):
        f[f"lag_{k}"] = g["units"].shift(h + k)
    for w in (4, 8, 13, 26):
        f[f"mean_{w}"] = _roll(known, by, w, "mean")
    f["std_8"] = _roll(known, by, 8, "std", min_periods=3)
    f["max_13"] = _roll(known, by, 13, "max")
    f["nonzero_8"] = _roll((known > 0).astype(float).where(known.notna()), by, 8, "mean")
    f["trend_4_13"] = f["mean_4"] / (f["mean_13"] + 1.0)
    f["same_week_last_year"] = g["units"].shift(52) if h <= 52 else np.nan

    # weeks since the last week with a sale (as of t-h)
    idx = p["t"].where(p["units"] > 0)
    last_sale_t = idx.groupby(by, sort=False).ffill()
    f["weeks_since_sale"] = (p["t"] - last_sale_t).groupby(by, sort=False).shift(h)

    price_known = g["price"].shift(h)
    f["price_last"] = price_known
    f["price_ratio"] = price_known / _roll(price_known, by, 13, "mean")
    f["orders_mean_4"] = _roll(g["orders"].shift(h), by, 4, "mean")

    # demand across all selected products (same h-week delay)
    total = p.groupby("t")["units"].sum()
    total_known = total.shift(h)
    f["total_last"] = p["t"].map(total_known)
    f["total_mean_4"] = p["t"].map(total_known.rolling(4, min_periods=1).mean())
    f["total_yoy"] = p["t"].map(total_known / total.shift(h + 52))
    f["share_13"] = f["mean_13"] / (p["t"].map(total_known.rolling(13, min_periods=1).mean()) + 1.0)
    f["trading_days_last"] = g["trading_days"].shift(h)
    f["age_weeks"] = g.cumcount() - h

    # calendar of the target week
    iso = p["week"].dt.isocalendar()
    f["weekofyear"] = iso["week"].astype(int).to_numpy()
    f["month"] = p["week"].dt.month.to_numpy()
    xmas = pd.to_datetime(p["week"].dt.year.astype(str) + "-12-25")
    f["weeks_to_xmas"] = ((xmas - p["week"]).dt.days // 7).clip(-2, 30).to_numpy()
    f["trading_days"] = p["trading_days"].to_numpy()

    f["closed"] = p["closed"].to_numpy()
    f["y"] = p["units"].to_numpy()
    return f
