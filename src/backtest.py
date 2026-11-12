"""Step 3: rolling-origin backtest of baselines and gradient boosting.

Run:  python -m src.backtest
Writes data/processed/predictions.parquet and feature_importance.json.

How the test works
------------------
The forecast date moves forward 4 weeks at a time. At each date the models see
only history up to that date and forecast the next 4 weeks. The first
N_SELECT_FOLDS dates are "selection folds" (used to choose the model setting);
the later dates are the "final test" and are never used for any choice.
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURES, build_features

LGB_BASE = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=C.SEED,
                n_jobs=2, verbose=-1)

# Candidate settings for the point forecast. The winner is picked on the
# selection folds only.
GBM_CANDIDATES = {
    "gbm_tweedie": dict(objective="tweedie", tweedie_variance_power=1.3),
    "gbm_l1": dict(objective="regression_l1"),
    "gbm_log": dict(objective="regression"),   # trained on log1p(units)
}
BASELINES = ["naive", "ma4", "ma13", "snaive", "ses"]


def make_folds(n_weeks: int) -> list[dict]:
    folds, c, i = [], C.INITIAL_TRAIN_WEEKS - 1, 0
    while c + C.HORIZON <= n_weeks - 1:
        folds.append({"fold": i, "cutoff_t": c,
                      "fold_type": "selection" if i < C.N_SELECT_FOLDS else "final"})
        c += C.STEP
        i += 1
    return folds


def ses_level(y: np.ndarray) -> float:
    """Simple exponential smoothing; alpha chosen per series by one-step error."""
    if len(y) == 0:
        return np.nan
    if len(y) < 4:
        return float(np.mean(y))
    best_level, best_sse = float(y[-1]), np.inf
    for alpha in np.arange(0.05, 1.0, 0.05):
        level, sse = y[0], 0.0
        for v in y[1:]:
            sse += (v - level) ** 2
            level = alpha * v + (1 - alpha) * level
        if sse < best_sse:
            best_sse, best_level = sse, float(level)
    return best_level


def ses_forecasts(panel: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    """SES level per (sku, cutoff), fitted on open weeks up to the cutoff."""
    rows = []
    open_panel = panel[~panel["closed"]]
    for sku, g in open_panel.groupby("sku", sort=False):
        t, y = g["t"].to_numpy(), g["units"].to_numpy(float)
        for f in folds:
            rows.append((sku, f["cutoff_t"], ses_level(y[t <= f["cutoff_t"]])))
    return pd.DataFrame(rows, columns=["sku", "cutoff_t", "ses"])


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, params: dict,
                log_target: bool = False) -> tuple[np.ndarray, lgb.LGBMRegressor]:
    y = np.log1p(train["y"]) if log_target else train["y"]
    model = lgb.LGBMRegressor(**LGB_BASE, **params)
    model.fit(train[FEATURES], y)
    pred = model.predict(test[FEATURES])
    if log_target:
        pred = np.expm1(pred)
    return np.clip(pred, 0, None), model


def run() -> pd.DataFrame:
    panel = pd.read_parquet(C.PROCESSED_DIR / "panel.parquet")
    n_weeks = int(panel["t"].max()) + 1
    folds = make_folds(n_weeks)
    print(f"{n_weeks} weeks, {panel['sku'].nunique()} products, {len(folds)} folds "
          f"({C.N_SELECT_FOLDS} selection + {len(folds) - C.N_SELECT_FOLDS} final)")

    ses = ses_forecasts(panel, folds)
    meta = panel[["sku", "abc", "demand_type", "description"]].drop_duplicates("sku")
    importance = {name: 0.0 for name in FEATURES}
    n_imp = 0
    out = []
    t0 = time.time()

    for h in range(1, C.HORIZON + 1):
        F = build_features(panel, h).replace([np.inf, -np.inf], np.nan)
        usable = F["last"].notna() & ~F["closed"]
        for f in folds:
            c = f["cutoff_t"]
            train = F[usable & (F["t"] <= c)]
            test = F[(F["t"] == c + h) & F["last"].notna()].copy()
            if test.empty:
                continue

            res = test[["sku", "week", "t", "h", "y", "closed"]].copy()
            res["fold"], res["fold_type"], res["cutoff_t"] = f["fold"], f["fold_type"], c
            res["scale"] = test["mean_13"].to_numpy() + 1.0

            # baselines: all use only data up to the cutoff
            res["naive"] = test["last"].to_numpy()
            res["ma4"] = test["mean_4"].to_numpy()
            res["ma13"] = test["mean_13"].to_numpy()
            res["snaive"] = test["same_week_last_year"].fillna(test["mean_4"]).to_numpy()
            res = res.merge(ses, on=["sku", "cutoff_t"], how="left")

            for name, params in GBM_CANDIDATES.items():
                pred, model = fit_predict(train, test, params, log_target=(name == "gbm_log"))
                res[name] = pred
                if name == "gbm_tweedie" and f["fold_type"] == "final":
                    gain = model.booster_.feature_importance(importance_type="gain")
                    gain = gain / gain.sum()
                    for k, v in zip(FEATURES, gain):
                        importance[k] += float(v)
                    n_imp += 1

            for col, q in (("q_lo", C.Q_LO), ("q_hi", C.Q_HI)):
                res[col], _ = fit_predict(train, test, dict(objective="quantile", alpha=q))
            res["q_lo"], res["q_hi"] = (np.minimum(res["q_lo"], res["q_hi"]),
                                        np.maximum(res["q_lo"], res["q_hi"]))

            # the business is shut in closed weeks: every method forecasts zero
            pred_cols = BASELINES + list(GBM_CANDIDATES) + ["q_lo", "q_hi"]
            res.loc[res["closed"], pred_cols] = 0.0
            out.append(res)
        print(f"  horizon {h} done ({time.time() - t0:.0f}s)")

    preds = pd.concat(out, ignore_index=True).merge(meta, on="sku")
    preds.to_parquet(C.PROCESSED_DIR / "predictions.parquet", index=False)
    imp = {k: v / max(n_imp, 1) for k, v in importance.items()}
    (C.PROCESSED_DIR / "feature_importance.json").write_text(json.dumps(imp, indent=2))
    print(f"saved {len(preds):,} forecast rows")
    return preds


if __name__ == "__main__":
    run()
