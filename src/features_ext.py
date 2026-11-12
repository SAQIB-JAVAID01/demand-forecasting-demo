"""Extended, leakage-safe features for the model lab.

Builds on `features.build_features` (lags, rolling means, trend, price, totals, calendar) and adds:

- Smoothing and volatility: exponentially weighted mean, 13-week coefficient of variation, recent slope.
- Intermittency: average demand interval (ADI) and squared CV of the non-zero sales (CV2) over 26 weeks, and the
  Syntetos-Boylan demand class (smooth, erratic, intermittent, lumpy) they imply.
- Seasonality: Fourier terms of the week of the year (3 sine / cosine pairs) and a Christmas-peak flag.
- Holidays: number of UK (England and Wales) bank holidays in the target week, and in the week after it.
- Price and promotion proxies: price against its 52-week average and discount depth from the 13-week high.
- Product attributes fixed in the first year only: revenue class (A / B / C) and regular vs intermittent.
- Economy: UK consumer confidence, unemployment and inflation, only as published by the forecast date.

The same rule holds as in `features.py`: for a forecast h weeks ahead, history enters only through `shift(h)`, and
calendar items (week of year, bank holidays) describe the target week, which a planner knows in advance.
`tests/test_no_leakage.py` checks these features too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES, build_features

UK_BANK_HOLIDAYS = pd.to_datetime([
    "2009-01-01", "2009-04-10", "2009-04-13", "2009-05-04", "2009-05-25", "2009-08-31", "2009-12-25", "2009-12-28",
    "2010-01-01", "2010-04-02", "2010-04-05", "2010-05-03", "2010-05-31", "2010-08-30", "2010-12-27", "2010-12-28",
    "2011-01-03", "2011-04-22", "2011-04-25", "2011-04-29", "2011-05-02", "2011-05-30", "2011-08-29", "2011-12-26",
    "2011-12-27", "2012-01-02", "2012-04-06", "2012-04-09", "2012-05-07", "2012-06-04", "2012-06-05", "2012-08-27",
    "2012-12-25", "2012-12-26",
])
FOURIER_K = 3
MACRO = ["confidence", "unemployment", "inflation"]

EXTRA_HISTORY = ["ewm_8", "cv_13", "slope_13", "adi_26", "cv2_26", "price_vs_52", "discount_13"]
EXTRA_STATIC = ["abc_code", "intermittent", "sb_class"]
EXTRA_CALENDAR = [f"{f}_{k}" for k in range(1, FOURIER_K + 1) for f in ("sin", "cos")] + \
                 ["bank_holidays", "bank_holidays_next", "xmas_peak"]
EXTRA_MACRO = [f"macro_{m}" for m in MACRO]
EXT_FEATURES = FEATURES + EXTRA_HISTORY + EXTRA_STATIC + EXTRA_CALENDAR + EXTRA_MACRO
EXT_HISTORY = EXTRA_HISTORY + EXTRA_MACRO          # must not move when the future changes

FEATURE_GROUPS = {
    "Lags and recent sales": ["last", "lag_1", "lag_2", "lag_3", "same_week_last_year", "weeks_since_sale"],
    "Rolling statistics": ["mean_4", "mean_8", "mean_13", "mean_26", "std_8", "max_13", "nonzero_8", "ewm_8",
                           "cv_13"],
    "Trend": ["trend_4_13", "slope_13", "age_weeks", "total_yoy"],
    "Intermittency": ["adi_26", "cv2_26", "sb_class", "intermittent"],
    "Seasonality and holidays": ["weekofyear", "month", "weeks_to_xmas", "xmas_peak", "trading_days",
                                 "trading_days_last", "bank_holidays", "bank_holidays_next"]
                                + [f"{f}_{k}" for k in range(1, FOURIER_K + 1) for f in ("sin", "cos")],
    "Price and promotion": ["price_last", "price_ratio", "price_vs_52", "discount_13", "orders_mean_4"],
    "Store-wide demand": ["total_last", "total_mean_4", "share_13"],
    "Product class": ["abc_code"],
    "Economy": EXTRA_MACRO,
}


def _roll(s: pd.Series, by: pd.Series, window: int, fn: str, min_periods: int = 1) -> pd.Series:
    r = s.groupby(by, sort=False).rolling(window, min_periods=min_periods)
    return getattr(r, fn)().reset_index(level=0, drop=True)


def bank_holidays_in_week(weeks: pd.Series, offset_weeks: int = 0) -> np.ndarray:
    start = pd.DatetimeIndex(weeks) + pd.Timedelta(weeks=offset_weeks)
    hol = UK_BANK_HOLIDAYS.to_numpy()
    s = start.to_numpy()
    return ((hol[None, :] >= s[:, None]) & (hol[None, :] < (s + np.timedelta64(7, "D"))[:, None])).sum(1)


def build_features_ext(panel: pd.DataFrame, h: int, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    f = build_features(panel, h)
    p = panel
    by = p["sku"]
    g = p.groupby("sku", sort=False)
    known = g["units"].shift(h)

    f["ewm_8"] = known.groupby(by, sort=False).transform(lambda s: s.ewm(halflife=4, min_periods=1).mean())
    std13 = _roll(known, by, 13, "std", min_periods=3)
    f["cv_13"] = std13 / (f["mean_13"] + 1.0)
    # slope of the last 13 known weeks, relative to their mean
    tt = p["t"].astype(float)
    k_t = known * tt
    n13 = _roll(known.notna().astype(float).where(known.notna()), by, 13, "sum")
    st, sy = _roll(tt.where(known.notna()), by, 13, "sum"), _roll(known, by, 13, "sum")
    stt, sty = _roll((tt ** 2).where(known.notna()), by, 13, "sum"), _roll(k_t, by, 13, "sum")
    den = n13 * stt - st ** 2
    f["slope_13"] = ((n13 * sty - st * sy) / den.where(den > 0)) / (f["mean_13"] + 1.0)

    nz = (known > 0).astype(float).where(known.notna())
    n_nz = _roll(nz, by, 26, "sum")
    f["adi_26"] = 26.0 / (n_nz + 1.0)
    pos = known.where(known > 0)
    m_pos, s_pos = _roll(pos, by, 26, "mean"), _roll(pos, by, 26, "std", min_periods=2)
    f["cv2_26"] = (s_pos / m_pos) ** 2
    lumpy = f["cv2_26"].fillna(0) >= 0.49
    inter = f["adi_26"] >= 1.32
    f["sb_class"] = np.select([~inter & ~lumpy, ~inter & lumpy, inter & ~lumpy], [0, 1, 2], 3)  # smooth .. lumpy

    price_known = g["price"].shift(h)
    f["price_vs_52"] = price_known / _roll(price_known, by, 52, "mean")
    f["discount_13"] = 1 - price_known / _roll(price_known, by, 13, "max")

    if "abc" in p:
        f["abc_code"] = p["abc"].map({"A": 0, "B": 1, "C": 2}).to_numpy()
    else:
        f["abc_code"] = np.nan
    f["intermittent"] = (p["demand_type"] == "intermittent").astype(float).to_numpy() if "demand_type" in p else np.nan

    woy = f["weekofyear"].to_numpy(float)
    for k in range(1, FOURIER_K + 1):
        f[f"sin_{k}"] = np.sin(2 * np.pi * k * woy / 52.18)
        f[f"cos_{k}"] = np.cos(2 * np.pi * k * woy / 52.18)
    f["bank_holidays"] = bank_holidays_in_week(p["week"])
    f["bank_holidays_next"] = bank_holidays_in_week(p["week"], 1)
    f["xmas_peak"] = f["weeks_to_xmas"].between(0, 6).astype(float)

    if macro is not None:
        # value public at the forecast date (week t-h), not at the target week
        origin = p["week"] - pd.to_timedelta(7 * h, unit="D")
        for m in MACRO:
            ser = macro[m]
            f[f"macro_{m}"] = ser.reindex(ser.index.union(pd.DatetimeIndex(origin.unique()))).ffill() \
                                 .reindex(origin).to_numpy()
    else:
        for m in MACRO:
            f[f"macro_{m}"] = np.nan
    return f.replace([np.inf, -np.inf], np.nan)
