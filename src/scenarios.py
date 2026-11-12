"""Sales projection under economic what-if scenarios.

Run:  python -m src.scenarios        (writes data/processed/projection.parquet and reports/scenarios.json)

1. Baseline: a SARIMAX model of the weekly store total (log units) with Fourier seasonality, UK bank holidays and
   trading days, fitted on all 104 weeks and projected 26 weeks past the end of the data, with 80% and 95%
   prediction intervals. Trading days for future weeks repeat the same week of the previous year.
2. Economic layer: the baseline is scaled by a multiplier built from changes in UK consumer confidence,
   unemployment and inflation (phased in over 13 weeks) and an optional price change.
   - The price effect uses an own-price elasticity estimated from this data (log units on log price with product
     and week effects, bootstrap over products for the 95% range).
   - The macro sensitivities are editable assumptions. The same model fitted with the real UK series
     (data/external) gives estimates that are far too wide to rely on with two years of data, and they are
     shown next to the assumptions so the gap is visible.
   Scenarios are what-if assumptions, not predictions of the real economy.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from . import config as C
from . import macro

warnings.filterwarnings("ignore")

HORIZON_WEEKS = 26
PHASE_IN_WEEKS = 13
FOURIER_K = 3
MACRO = ["confidence", "unemployment", "inflation"]

# Sales change for a one-unit change in each indicator (editable in the app). Conservative assumptions.
ASSUMED_SENSITIVITY = {"confidence": 0.015, "unemployment": -0.04, "inflation": -0.01}
SCENARIOS = {
    "Recession": {"confidence": -3.0, "unemployment": 1.5, "inflation": 1.0},
    "Base": {"confidence": 0.0, "unemployment": 0.0, "inflation": 0.0},
    "Boom": {"confidence": 2.0, "unemployment": -1.0, "inflation": -1.0},
}


def _exog(weeks: pd.DatetimeIndex, trading_days: np.ndarray) -> np.ndarray:
    """Fourier terms of the week of the year and the number of trading days lost (out of 6)."""
    woy = weeks.isocalendar().week.to_numpy(float)
    cols = []
    for k in range(1, FOURIER_K + 1):
        cols += [np.sin(2 * np.pi * k * woy / 52.18), np.cos(2 * np.pi * k * woy / 52.18)]
    cols.append(6.0 - np.asarray(trading_days, float))
    return np.column_stack(cols)


def future_calendar(cal: pd.DataFrame, n: int = HORIZON_WEEKS) -> pd.DataFrame:
    """Future weeks: Christmas to New Year repeats last year's shutdown; other weeks lose one trading day per
    UK bank holiday falling Monday to Saturday."""
    from .features_ext import UK_BANK_HOLIDAYS
    last = cal["week"].max()
    weeks = pd.date_range(last + pd.Timedelta(weeks=1), periods=n, freq="W-MON")
    prev = cal.set_index("week")
    td, closed = [], []
    for w in weeks:
        md = (w.month, w.day)
        if md >= (12, 15) or md <= (1, 7):
            ref = prev.index[np.argmin(np.abs((prev.index - (w - pd.Timedelta(weeks=52))).days))]
            td.append(int(prev.loc[ref, "trading_days"]))
            closed.append(bool(prev.loc[ref, "closed"]))
        else:
            hol = ((UK_BANK_HOLIDAYS >= w) & (UK_BANK_HOLIDAYS < w + pd.Timedelta(days=6))).sum()
            td.append(int(6 - hol))
            closed.append(False)
    return pd.DataFrame({"week": weeks, "trading_days": td, "closed": closed})


def segment_shares(panel: pd.DataFrame, weeks: int = 26) -> dict:
    recent = panel[panel["week"] > panel["week"].max() - pd.Timedelta(weeks=weeks)]
    units = recent.groupby("abc")["units"].sum()
    rev = recent.groupby("abc")["revenue"].sum()
    return {"units": (units / units.sum()).to_dict(), "price": float(rev.sum() / units.sum())}


def baseline(panel: pd.DataFrame, cal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = cal.sort_values("t").reset_index(drop=True)
    tot = panel.groupby("week")["units"].sum().reindex(cal["week"]).fillna(0).to_numpy(float)
    op = ~cal["closed"].to_numpy()
    X = _exog(pd.DatetimeIndex(cal["week"]), cal["trading_days"].to_numpy())
    from statsmodels.tsa.statespace.sarimax import SARIMAX   # only needed to fit, not to apply scenarios
    res = SARIMAX(np.log1p(tot[op]), exog=X[op], order=(1, 0, 1), trend="ct").fit(disp=False)
    fut = future_calendar(cal)
    Xf = _exog(pd.DatetimeIndex(fut["week"]), fut["trading_days"].to_numpy())
    fc = res.get_forecast(len(fut), exog=Xf)
    mean = fc.predicted_mean
    se = np.sqrt(fc.var_pred_mean)
    proj = pd.DataFrame({"week": fut["week"], "closed": fut["closed"], "mean": np.expm1(mean)})
    for lvl, zq in (("80", 1.2816), ("95", 1.96)):
        proj[f"lo{lvl}"] = np.expm1(mean - zq * se)
        proj[f"hi{lvl}"] = np.expm1(mean + zq * se)
    proj.loc[proj["closed"], ["mean", "lo80", "hi80", "lo95", "hi95"]] = 0.0
    proj[["mean", "lo80", "hi80", "lo95", "hi95"]] = proj[["mean", "lo80", "hi80", "lo95", "hi95"]].clip(lower=0)
    hist = pd.DataFrame({"week": cal["week"], "units": tot, "closed": cal["closed"]})
    return hist, proj


def macro_estimates(panel: pd.DataFrame, cal: pd.DataFrame) -> dict:
    """The same baseline model with the real UK indicators added: coefficient and 95% range per indicator."""
    cal = cal.sort_values("t").reset_index(drop=True)
    tot = panel.groupby("week")["units"].sum().reindex(cal["week"]).fillna(0).to_numpy(float)
    op = ~cal["closed"].to_numpy()
    X = _exog(pd.DatetimeIndex(cal["week"]), cal["trading_days"].to_numpy())
    M = macro.weekly(cal["week"])[MACRO].to_numpy(float)
    from statsmodels.tsa.statespace.sarimax import SARIMAX   # only needed to fit, not to apply scenarios
    res = SARIMAX(np.log1p(tot[op]), exog=np.column_stack([X, M])[op], order=(1, 0, 1), trend="ct").fit(disp=False)
    k = X.shape[1] + 2                                   # after intercept and drift
    params, ci = np.asarray(res.params), np.asarray(res.conf_int(0.05))
    out = {}
    for i, m in enumerate(MACRO):
        out[m] = {"effect": float(np.expm1(params[k + i])), "lo95": float(np.expm1(ci[k + i, 0])),
                  "hi95": float(np.expm1(ci[k + i, 1])), "p_value": float(np.asarray(res.pvalues)[k + i])}
    last = macro.monthly()[MACRO].dropna()
    last = last[last.index <= cal["week"].max()].iloc[-1]
    out["latest"] = {m: float(last[m]) for m in MACRO}
    out["latest_month"] = str(macro.monthly()[MACRO].dropna().loc[:cal["week"].max()].index[-1].date())
    return out


def price_elasticity(panel: pd.DataFrame, n_boot: int = 300) -> dict:
    p = panel[(panel["units"] > 0) & (panel["price"] > 0) & ~panel["closed"]].copy()
    p["lu"], p["lp"] = np.log(p["units"]), np.log(p["price"])
    for c in ("lu", "lp"):
        p[c + "d"] = p[c] - p.groupby("sku")[c].transform("mean") - (p.groupby("week")[c].transform("mean") - p[c].mean())
    g = p.groupby("sku")[["lud", "lpd"]].apply(lambda d: pd.Series({"xy": (d["lud"] * d["lpd"]).sum(),
                                                                   "xx": (d["lpd"] ** 2).sum()}))
    est = float(g["xy"].sum() / g["xx"].sum())
    rng = np.random.default_rng(C.SEED)
    a = g.to_numpy()
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    b = a[idx].sum(1)
    boots = b[:, 0] / b[:, 1]
    return {"elasticity": est, "lo95": float(np.percentile(boots, 2.5)), "hi95": float(np.percentile(boots, 97.5))}


def multiplier(n: int, deltas: dict, sens: dict, price_change: float = 0.0, elasticity: float = 0.0) -> np.ndarray:
    """Weekly sales multiplier: indicator changes phased in linearly over PHASE_IN_WEEKS, then held."""
    ramp = np.minimum(np.arange(1, n + 1) / PHASE_IN_WEEKS, 1.0)
    logm = sum(np.log1p(sens[m]) * deltas.get(m, 0.0) for m in MACRO) * ramp
    return np.exp(logm) * (1 + price_change) ** elasticity


def apply(proj: pd.DataFrame, deltas: dict, sens: dict, price_change: float = 0.0, elasticity: float = 0.0) -> pd.DataFrame:
    m = multiplier(len(proj), deltas, sens, price_change, elasticity)
    out = proj.copy()
    for c in ("mean", "lo80", "hi80", "lo95", "hi95"):
        out[c] = proj[c].to_numpy() * m
    return out


def run() -> dict:
    panel = pd.read_parquet(C.PROCESSED_DIR / "panel.parquet")
    cal = pd.read_parquet(C.PROCESSED_DIR / "calendar.parquet")
    hist, proj = baseline(panel, cal)
    proj.to_parquet(C.PROCESSED_DIR / "projection.parquet", index=False)
    info = {"macro_estimates": macro_estimates(panel, cal), "price": price_elasticity(panel),
            "assumed_sensitivity": ASSUMED_SENSITIVITY, "scenarios": SCENARIOS, "segments": segment_shares(panel),
            "horizon_weeks": HORIZON_WEEKS, "phase_in_weeks": PHASE_IN_WEEKS}
    base_total = float(proj["mean"].sum())
    for name, d in SCENARIOS.items():
        s = apply(proj, d, ASSUMED_SENSITIVITY)
        info.setdefault("totals", {})[name] = float(s["mean"].sum())
    info["base_total"] = base_total
    (C.REPORTS_DIR / "scenarios.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: info[k] for k in ("macro_estimates", "price", "totals")}, indent=1))
    return info


if __name__ == "__main__":
    run()
