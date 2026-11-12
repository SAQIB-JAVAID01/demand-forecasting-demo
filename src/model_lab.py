"""Step 5: model lab. More forecasting models on the same rolling backtest.

Run:  python -m src.model_lab          (after src.backtest and src.evaluate)
Adds one column per model to data/processed/predictions.parquet and writes reports/model_lab.json.

Every model below forecasts from the same forecast dates as the original backtest and sees only data up to that
date. The ensemble's members are picked on the selection folds only; the leaderboard reports the final folds,
which no choice ever looked at.

Models
------
ets          Damped-trend exponential smoothing (Holt), per product (statsmodels).
tsb          Teunter-Syntetos-Babai method for intermittent demand (a Croston variant), per product.
sarimax_td   Top-down: a SARIMAX model of the weekly store total with Fourier seasonality, UK bank holidays and
             trading days (statsmodels), split across products by each product's recent share. Fitted on the
             total because single products have too little history for a seasonal model.
gbm_fe       LightGBM, log target, on the extended feature set (src/features_ext.py).
xgb          XGBoost, Tweedie objective, extended features.
hgb          scikit-learn HistGradientBoosting, Poisson loss, extended features.
ensemble     Equal-weight average of the three models with the lowest selection-fold error.
"""
from __future__ import annotations

import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import config as C
from . import macro
from .backtest import LGB_BASE, make_folds
from .features_ext import EXT_FEATURES, FEATURE_GROUPS, FOURIER_K, bank_holidays_in_week, build_features_ext

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

NEW_MODELS = ["ets", "tsb", "sarimax_td", "gbm_fe", "xgb", "hgb"]
LABELS = {
    "naive": "Last week's sales", "ma4": "4-week average", "ma13": "13-week average", "snaive": "Same week last year",
    "ses": "Exponential smoothing", "gbm_tweedie": "LightGBM (Tweedie)", "gbm_l1": "LightGBM (L1)",
    "gbm_log": "LightGBM (log target)", "ets": "Holt damped trend (ETS)", "tsb": "TSB / Croston (intermittent)",
    "sarimax_td": "SARIMAX top-down", "gbm_fe": "LightGBM + engineered features", "xgb": "XGBoost (Tweedie)",
    "hgb": "HistGradientBoosting (Poisson)", "ensemble": "Ensemble (top 3 on selection folds)",
}
FAMILY = {"naive": "Baseline", "ma4": "Baseline", "ma13": "Baseline", "snaive": "Baseline", "ses": "Baseline",
          "ets": "Statistical", "tsb": "Statistical", "sarimax_td": "Statistical", "gbm_tweedie": "Machine learning",
          "gbm_l1": "Machine learning", "gbm_log": "Machine learning", "gbm_fe": "Machine learning",
          "xgb": "Machine learning", "hgb": "Machine learning", "ensemble": "Ensemble"}
CANDIDATES = ["naive", "ma4", "ma13", "snaive", "ses", "gbm_tweedie", "gbm_l1", "gbm_log"] + NEW_MODELS


def ets_forecast(y: np.ndarray, h: int) -> np.ndarray:
    if len(y) < 10 or y.sum() == 0:
        return np.full(h, float(np.mean(y[-13:])) if len(y) else 0.0)
    try:
        m = ExponentialSmoothing(y, trend="add", damped_trend=True, initialization_method="estimated").fit(optimized=True)
        f = m.forecast(h)
        if not np.all(np.isfinite(f)):
            raise ValueError
        return np.clip(f, 0, None)
    except Exception:
        return np.full(h, float(np.mean(y[-13:])))


def tsb_forecast(y: np.ndarray, alpha: float = 0.1, beta: float = 0.1) -> float:
    """TSB: smooths the demand size and, separately, the probability that a week has any demand."""
    if len(y) == 0:
        return 0.0
    nz = y[y > 0]
    z = float(nz[:4].mean()) if len(nz) else 0.0
    p = float((y[:8] > 0).mean())
    for v in y:
        if v > 0:
            z += alpha * (v - z)
            p += beta * (1 - p)
        else:
            p += beta * (0 - p)
    return p * z


def per_product(panel: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    rows = []
    open_p = panel[~panel["closed"]]
    for sku, g in open_p.groupby("sku", sort=False):
        t, y = g["t"].to_numpy(), g["units"].to_numpy(float)
        for f in folds:
            c = f["cutoff_t"]
            hist = y[t <= c]
            e = ets_forecast(hist, C.HORIZON)
            ts = tsb_forecast(hist)
            for h in range(1, C.HORIZON + 1):
                rows.append((sku, c, h, float(e[h - 1]), ts))
    return pd.DataFrame(rows, columns=["sku", "cutoff_t", "h", "ets", "tsb"])


def calendar_exog(weeks, trading_days: np.ndarray) -> np.ndarray:
    woy = pd.DatetimeIndex(weeks).isocalendar().week.to_numpy(float)
    cols = []
    for k in range(1, FOURIER_K + 1):
        cols += [np.sin(2 * np.pi * k * woy / 52.18), np.cos(2 * np.pi * k * woy / 52.18)]
    cols += [bank_holidays_in_week(pd.Series(pd.DatetimeIndex(weeks))), np.asarray(trading_days, float) / 6.0]
    return np.column_stack(cols)


def sarimax_total(panel: pd.DataFrame, cal: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    tot = panel.groupby("t")["units"].sum().reindex(cal["t"]).fillna(0).to_numpy(float)
    X = calendar_exog(cal["week"], cal["trading_days"].to_numpy(float))
    open_ = ~cal["closed"].to_numpy()
    rows = []
    for f in folds:
        c = f["cutoff_t"]
        idx = np.arange(c + 1)
        idx = idx[open_[idx]]
        tgt = np.arange(c + 1, c + C.HORIZON + 1)
        try:
            res = SARIMAX(np.log1p(tot[idx]), exog=X[idx], order=(1, 0, 1), trend="ct").fit(disp=False)
            fc = np.expm1(res.forecast(len(tgt), exog=X[tgt]))
        except Exception:
            fc = np.repeat(tot[idx][-13:].mean(), C.HORIZON)
        for h in range(1, C.HORIZON + 1):
            rows.append((c, h, max(float(fc[h - 1]), 0.0)))
    return pd.DataFrame(rows, columns=["cutoff_t", "h", "total_fc"])


def fit_ml(name: str, train: pd.DataFrame, test: pd.DataFrame):
    X, Xt = train[EXT_FEATURES], test[EXT_FEATURES]
    if name == "gbm_fe":
        m = lgb.LGBMRegressor(**LGB_BASE, objective="regression")
        m.fit(X, np.log1p(train["y"]))
        return np.clip(np.expm1(m.predict(Xt)), 0, None), m
    if name == "xgb":
        m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, min_child_weight=10, subsample=0.8,
                             colsample_bytree=0.8, objective="reg:tweedie", tweedie_variance_power=1.3,
                             random_state=C.SEED, n_jobs=2, tree_method="hist")
        m.fit(X, train["y"])
        return np.clip(m.predict(Xt), 0, None), m
    keep = [c for c in EXT_FEATURES if X[c].nunique() > 1]          # constant columns carry nothing
    X, Xt = X[keep], Xt[keep]
    m = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.06, max_iter=300, max_leaf_nodes=15,
                                      min_samples_leaf=40, l2_regularization=1.0, random_state=C.SEED)
    m.fit(X, train["y"])
    return np.clip(m.predict(Xt), 0, None), m


def wape(y, p) -> float:
    return float(np.abs(np.asarray(p) - np.asarray(y)).sum() / np.asarray(y).sum())


def bias(y, p) -> float:
    return float((np.asarray(p) - np.asarray(y)).sum() / np.asarray(y).sum())


def total_wape(df: pd.DataFrame, col: str) -> float:
    g = df.groupby("week")[["y", col]].sum()
    return wape(g["y"], g[col])


def boot_ci(df: pd.DataFrame, model: str, base: str, rng, n: int = 1000) -> list[float]:
    """95% range of the relative error change against `base`, resampling whole products."""
    a = df.assign(em=(df[model] - df["y"]).abs(), eb=(df[base] - df["y"]).abs()).groupby("sku")[["em", "eb"]].sum().to_numpy()
    s = a[rng.integers(0, len(a), size=(n, len(a)))].sum(axis=1)
    rel = s[:, 0] / s[:, 1] - 1
    return [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))]


def run() -> dict:
    t0 = time.time()
    panel = pd.read_parquet(C.PROCESSED_DIR / "panel.parquet").sort_values(["sku", "week"]).reset_index(drop=True)
    cal = pd.read_parquet(C.PROCESSED_DIR / "calendar.parquet").sort_values("t").reset_index(drop=True)
    preds = pd.read_parquet(C.PROCESSED_DIR / "predictions.parquet")
    preds = preds.drop(columns=[c for c in NEW_MODELS + ["ensemble"] if c in preds])
    folds = make_folds(int(panel["t"].max()) + 1)
    mw = macro.weekly(pd.date_range(panel["week"].min() - pd.Timedelta(weeks=60), panel["week"].max(), freq="W-MON"))

    print("statistical models per product ...", flush=True)
    pp = per_product(panel, folds)
    st = sarimax_total(panel, cal, folds)
    print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    print("machine learning models ...", flush=True)
    ml_rows, imp_gain, shap_abs, n_imp = [], np.zeros(len(EXT_FEATURES)), np.zeros(len(EXT_FEATURES)), 0
    for h in range(1, C.HORIZON + 1):
        F = build_features_ext(panel, h, mw)
        usable = F["last"].notna() & ~F["closed"]
        for f in folds:
            c = f["cutoff_t"]
            train = F[usable & (F["t"] <= c)]
            test = F[(F["t"] == c + h) & F["last"].notna()]
            if test.empty:
                continue
            res = test[["sku", "t", "h"]].copy()
            res["share"] = (test["mean_13"] / test.groupby("t")["mean_13"].transform("sum")).to_numpy()
            for name in ("gbm_fe", "xgb", "hgb"):
                p_, m_ = fit_ml(name, train, test)
                res[name] = p_
                if name == "gbm_fe" and f["fold_type"] == "final":
                    g_ = m_.booster_.feature_importance(importance_type="gain")
                    imp_gain += g_ / g_.sum()
                    contrib = m_.booster_.predict(test[EXT_FEATURES], pred_contrib=True)[:, :-1]   # TreeSHAP
                    shap_abs += np.abs(contrib).mean(0)
                    n_imp += 1
            ml_rows.append(res)
        print(f"  horizon {h} done ({time.time() - t0:.0f}s)", flush=True)
    ml = pd.concat(ml_rows, ignore_index=True)

    out = preds.merge(ml, on=["sku", "t", "h"], how="left").merge(pp, on=["sku", "cutoff_t", "h"], how="left") \
               .merge(st, on=["cutoff_t", "h"], how="left")
    out["sarimax_td"] = out["total_fc"] * out["share"]
    out = out.drop(columns=["total_fc", "share"])
    for m in NEW_MODELS:
        out[m] = out[m].fillna(out["ma13"]).clip(lower=0)
        out.loc[out["closed"], m] = 0.0

    sel = out[~out["closed"] & (out["fold_type"] == "selection")]
    sel_w = {m: wape(sel["y"], sel[m]) for m in CANDIDATES}
    members = sorted(CANDIDATES, key=sel_w.get)[:3]
    out["ensemble"] = out[members].mean(axis=1)
    scored = out[~out["closed"]]
    sel, fin = scored[scored["fold_type"] == "selection"], scored[scored["fold_type"] == "final"]
    sel_w["ensemble"] = wape(sel["y"], sel["ensemble"])
    models = CANDIDATES + ["ensemble"]

    rng = np.random.default_rng(C.SEED)
    fa = fin[fin["abc"] == "A"]
    board = []
    for m in models:
        board.append({"model": m, "label": LABELS[m], "family": FAMILY[m], "selection_wape": sel_w[m],
                      "wape": wape(fin["y"], fin[m]), "bias": bias(fin["y"], fin[m]), "total_wape": total_wape(fin, m),
                      "class_a_wape": wape(fa["y"], fa[m]),
                      "vs_ma13": wape(fin["y"], fin[m]) / wape(fin["y"], fin["ma13"]) - 1,
                      "vs_ma13_ci95": boot_ci(fin, m, "ma13", rng)})
    board = sorted(board, key=lambda r: r["wape"])
    best_sel = min(models, key=sel_w.get)

    imp = pd.DataFrame({"feature": EXT_FEATURES, "gain": imp_gain / max(n_imp, 1), "shap": shap_abs / max(n_imp, 1)})
    imp["shap"] = imp["shap"] / imp["shap"].sum()
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    imp["group"] = imp["feature"].map(group_of).fillna("Other")

    lab = {"leaderboard": board, "ensemble_members": members, "best_on_selection": best_sel,
           "feature_importance": imp.sort_values("shap", ascending=False).to_dict("records"),
           "feature_groups": FEATURE_GROUPS, "n_features_base": 27, "n_features_ext": len(EXT_FEATURES),
           "final_rows": int(len(fin)), "runtime_s": round(time.time() - t0, 1)}
    out.to_parquet(C.PROCESSED_DIR / "predictions.parquet", index=False)
    (C.REPORTS_DIR / "model_lab.json").write_text(json.dumps(lab, indent=2))
    show = pd.DataFrame(board)[["label", "selection_wape", "wape", "total_wape", "bias", "vs_ma13"]]
    print(show.round(3).to_string(index=False))
    print("ensemble members:", members, "| best on selection folds:", best_sel, f"| {time.time() - t0:.0f}s")
    return lab


if __name__ == "__main__":
    run()
