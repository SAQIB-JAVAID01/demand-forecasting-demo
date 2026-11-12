"""Headline numbers shared by the desktop app and the web app.

Everything is computed from the backtest predictions (one row per product, week and horizon).
Nothing here is a constant result: change the predictions and every number moves with them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

BASE = "ma13"                       # strongest simple method on the final test (the tougher comparison)
BASE_LABEL = "13-week average"
SEASONAL = "snaive"                 # same week last year
BIAS_TARGET = 0.05                  # |bias| within 5% counts as unbiased
DEFAULT_SETTING = "gbm_tweedie"     # opens on the setting whose totals are unbiased
SETTINGS = {"gbm_tweedie": "Unbiased totals", "ensemble": "Best per-product accuracy"}
SETTING_USE = {"gbm_tweedie": "Total stock, purchasing and capacity",
               "ensemble": "Ranking products and typical-week planning"}
SETTING_WHY = {
    "gbm_tweedie": "Aims at the average week, so the forecasts add up to the units actually sold.",
    "ensemble": "Aims at the typical (median) week. Gift sales have occasional bulk orders, so the typical week "
                "is below the average: closer per product, but totals come out low by design.",
}


def settings_for(preds: pd.DataFrame) -> dict[str, str]:
    """The forecast settings available in this backtest output, default first."""
    out = {k: v for k, v in SETTINGS.items() if k in preds.columns}
    if "ensemble" not in out and "gbm_log" in preds.columns:       # before the model lab has been run
        out["gbm_log"] = SETTINGS["ensemble"]
    return out


def final_rows(preds: pd.DataFrame, abc=None, demand=None, horizons=None) -> pd.DataFrame:
    """Out-of-sample rows of the final test, closed weeks removed, with the optional filters."""
    m = (preds["fold_type"] == "final") & ~preds["closed"]
    if abc is not None:
        m &= preds["abc"].isin(list(abc))
    if demand is not None:
        m &= preds["demand_type"].isin(list(demand))
    if horizons is not None:
        m &= preds["h"].isin(list(horizons))
    return preds[m]


def wape(df: pd.DataFrame, col: str) -> float:
    return float((df[col] - df["y"]).abs().sum() / df["y"].sum())


def bias(df: pd.DataFrame, col: str) -> float:
    return float((df[col] - df["y"]).sum() / df["y"].sum())


def weekly(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Units per week, summed over the products in the frame."""
    return df.groupby("week")[["y"] + cols].sum()


def total_wape(df: pd.DataFrame, col: str) -> float:
    """Error of the weekly total: products are added up first, then compared week by week."""
    g = weekly(df, [col])
    return float((g[col] - g["y"]).abs().sum() / g["y"].sum())


def product_ci(df: pd.DataFrame, col: str, base: str = BASE, n: int = 1000) -> tuple[float, float]:
    """95% range of (benchmark WAPE - forecast WAPE) per product-week, resampling whole products."""
    g = df.assign(em=(df[col] - df["y"]).abs(), eb=(df[base] - df["y"]).abs()) \
          .groupby("sku")[["em", "eb", "y"]].sum().to_numpy()
    rng = np.random.default_rng(C.SEED)
    s = g[rng.integers(0, len(g), size=(n, len(g)))].sum(axis=1)
    d = (s[:, 1] - s[:, 0]) / s[:, 2]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def total_ci(df: pd.DataFrame, col: str, base: str = BASE, n: int = 1000) -> tuple[float, float]:
    """95% range of (benchmark - forecast) weekly-total WAPE, resampling whole weeks."""
    g = weekly(df, [col, base])
    a = np.column_stack([(g[col] - g["y"]).abs(), (g[base] - g["y"]).abs(), g["y"]])
    rng = np.random.default_rng(C.SEED)
    s = a[rng.integers(0, len(a), size=(n, len(a)))].sum(axis=1)
    d = (s[:, 1] - s[:, 0]) / s[:, 2]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def verdict(lo: float, hi: float) -> str:
    """'better', 'worse' or 'level': level means the 95% range includes zero."""
    return "better" if lo > 0 else ("worse" if hi < 0 else "level")


def headline(df: pd.DataFrame, col: str, base: str = BASE) -> dict:
    """The numbers the first screen leads with, for one forecast setting on the filtered final test."""
    tw, tb = total_wape(df, col), total_wape(df, base)
    pw, pb = wape(df, col), wape(df, base)
    t_lo, t_hi = total_ci(df, col, base)
    p_lo, p_hi = product_ci(df, col, base)
    return dict(
        total_acc=1 - tw, total_acc_base=1 - tb, total_fva=tb - tw, total_ci=(t_lo, t_hi),
        total_verdict=verdict(t_lo, t_hi),
        product_acc=1 - pw, product_acc_base=1 - pb, product_fva=pb - pw, product_ci=(p_lo, p_hi),
        product_verdict=verdict(p_lo, p_hi),
        bias=bias(df, col), bias_base=bias(df, base),
        weeks=int(df["week"].nunique()), products=int(df["sku"].nunique()), rows=int(len(df)),
        units=float(df["y"].sum()),
    )


def bias_tag(b: float) -> str:
    return "Unbiased (within ±5%)" if abs(b) <= BIAS_TARGET else ("Under-forecast" if b < 0 else "Over-forecast")


def inventory_view(run: dict, cost: float, hold: float) -> dict:
    """Money and stock figures for one precomputed simulation run, at a unit cost (share of price) and holding rate."""
    m, b, e = run["model_run"], run["base_run"], run["model_same_service"]
    years = m["weeks"] / 52.0
    dv = b["avg_inventory_value"] - e["avg_inventory_value"]           # at unit cost = price
    return dict(run=run, lead=run["lead"], service=run["service"], cost=cost, hold=hold, m=m, b=b, e=e,
                inv_change=e["avg_inventory_value"] / b["avg_inventory_value"] - 1,
                saving=dv * cost * hold, saving_lo=dv * cost * 0.15, saving_hi=dv * cost * 0.35,
                lost_rev=(b["lost_revenue"] - m["lost_revenue"]) / years,
                lost_margin=(b["lost_revenue"] - m["lost_revenue"]) * (1 - cost) / years)


def find_run(inv: dict | None, model: str, lead: int, service: float) -> dict | None:
    if inv is None:
        return None
    return next((r for r in inv["runs"] if r["model"] == model and r["lead"] == lead
                 and abs(r["service"] - service) < 1e-9), None)
