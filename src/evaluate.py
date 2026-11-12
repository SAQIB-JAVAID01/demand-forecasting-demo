"""Step 4: score the backtest.

Run:  python -m src.evaluate
Writes reports/metrics.json and reports/tables/*.csv.

Rules followed here:
- The model setting and the "best baseline" are both chosen on the selection
  folds. Headline numbers come from the final folds only.
- The improvement over the baseline is reported with a 95% confidence interval
  (bootstrap over products), not just a point value.
- Closed weeks (business shut) are excluded from scoring.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .backtest import BASELINES, GBM_CANDIDATES

LABELS = {
    "naive": "Last week's sales",
    "ma4": "4-week average",
    "ma13": "13-week average",
    "snaive": "Same week last year",
    "ses": "Exponential smoothing",
    "gbm_tweedie": "Gradient boosting (Tweedie)",
    "gbm_l1": "Gradient boosting (L1)",
    "gbm_log": "Gradient boosting (log target)",
}


def wape(y, p) -> float:
    return float(np.abs(p - y).sum() / y.sum())


def bias(y, p) -> float:
    return float((p - y).sum() / y.sum())


def score(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = df["y"].to_numpy()
    rows = []
    for c in cols:
        p = df[c].to_numpy()
        rows.append({"model": c, "label": LABELS[c], "wape": wape(y, p), "bias": bias(y, p),
                     "mae": float(np.abs(p - y).mean())})
    return pd.DataFrame(rows)


def weekly_total_wape(df: pd.DataFrame, col: str) -> float:
    g = df.groupby("week")[["y", col]].sum()
    return wape(g["y"].to_numpy(), g[col].to_numpy())


def bootstrap_diff(df: pd.DataFrame, model: str, base: str, rng) -> dict:
    """95% CI for WAPE(model) - WAPE(base), resampling whole products."""
    g = df.assign(em=(df[model] - df["y"]).abs(), eb=(df[base] - df["y"]).abs()) \
          .groupby("sku")[["em", "eb", "y"]].sum().to_numpy()
    n = len(g)
    idx = rng.integers(0, n, size=(C.N_BOOTSTRAP, n))
    s = g[idx].sum(axis=1)                       # (B, 3)
    d = s[:, 0] / s[:, 2] - s[:, 1] / s[:, 2]
    rel = s[:, 0] / s[:, 1] - 1
    return {"diff": float(g[:, 0].sum() / g[:, 2].sum() - g[:, 1].sum() / g[:, 2].sum()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "relative_change": float(g[:, 0].sum() / g[:, 1].sum() - 1),
            "relative_ci95": [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))],
            "share_of_resamples_model_better": float((d < 0).mean())}


def conformalize(df: pd.DataFrame) -> pd.DataFrame:
    """Widen/narrow the raw 80% interval using errors from EARLIER folds only."""
    df = df.copy()
    df["lo"], df["hi"] = df["q_lo"], df["q_hi"]
    level = C.Q_HI - C.Q_LO
    for f in sorted(df["fold"].unique()):
        past = df[(df["fold"] < f) & ~df["closed"]]
        if len(past) < 200:
            continue
        s = np.maximum(past["q_lo"] - past["y"], past["y"] - past["q_hi"]) / past["scale"]
        q = float(np.quantile(s, min(1.0, level * (1 + 1 / len(s)))))
        m = df["fold"] == f
        df.loc[m, "lo"] = np.clip(df.loc[m, "q_lo"] - q * df.loc[m, "scale"], 0, None)
        df.loc[m, "hi"] = df.loc[m, "q_hi"] + q * df.loc[m, "scale"]
    df.loc[df["closed"], ["lo", "hi"]] = 0.0
    df["hi"] = np.maximum(df["hi"], df["lo"])
    return df


def main() -> dict:
    preds = pd.read_parquet(C.PROCESSED_DIR / "predictions.parquet")
    preds = conformalize(preds)
    preds.to_parquet(C.PROCESSED_DIR / "predictions.parquet", index=False)
    scored = preds[~preds["closed"]]
    sel, fin = scored[scored["fold_type"] == "selection"], scored[scored["fold_type"] == "final"]
    all_models = BASELINES + list(GBM_CANDIDATES)
    rng = np.random.default_rng(C.SEED)
    tables = C.REPORTS_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    # choices made on selection folds only
    sel_scores = score(sel, all_models)
    best_gbm = sel_scores[sel_scores["model"].isin(GBM_CANDIDATES)].sort_values("wape").iloc[0]["model"]
    best_base = sel_scores[sel_scores["model"].isin(BASELINES)].sort_values("wape").iloc[0]["model"]

    fin_scores = score(fin, all_models)
    fin_scores["weekly_total_wape"] = [weekly_total_wape(fin, m) for m in fin_scores["model"]]
    fin_scores["chosen"] = fin_scores["model"].isin([best_gbm])
    fin_scores["best_baseline_on_selection"] = fin_scores["model"] == best_base
    sel_scores.to_csv(tables / "scores_selection_folds.csv", index=False)
    fin_scores.to_csv(tables / "scores_final_folds.csv", index=False)

    # the baseline that turned out best on the final folds (reported for honesty)
    base_fin = fin_scores[fin_scores["model"].isin(BASELINES)].sort_values("wape").iloc[0]["model"]

    def seg(by: str, frame: pd.DataFrame = fin) -> pd.DataFrame:
        rows = []
        for key, g in frame.groupby(by):
            if g["y"].sum() <= 0:
                continue
            rows.append({by: key, "rows": len(g), "units": float(g["y"].sum()),
                         "wape_model": wape(g["y"], g[best_gbm]),
                         "wape_baseline": wape(g["y"], g[best_base]),
                         "bias_model": bias(g["y"], g[best_gbm])})
        d = pd.DataFrame(rows)
        d["relative_change"] = d["wape_model"] / d["wape_baseline"] - 1
        return d

    fin = fin.assign(season=np.where(fin["week"].dt.month >= 9, "Sep-Nov peak", "Dec-Aug"))
    segments = {"horizon": seg("h", fin), "abc": seg("abc", fin),
                "demand_type": seg("demand_type", fin), "season": seg("season", fin),
                "fold": seg("fold", fin)}
    for k, d in segments.items():
        d.to_csv(tables / f"segment_{k}.csv", index=False)

    def coverage(frame, lo, hi):
        return float(((frame["y"] >= frame[lo]) & (frame["y"] <= frame[hi])).mean())

    interval = {
        "target": C.Q_HI - C.Q_LO,
        "raw_coverage_final": coverage(fin, "q_lo", "q_hi"),
        "calibrated_coverage_final": coverage(fin, "lo", "hi"),
        "calibrated_below_upper_final": float((fin["y"] <= fin["hi"]).mean()),
        "raw_mean_width_over_scale": float(((fin["q_hi"] - fin["q_lo"]) / fin["scale"]).mean()),
        "calibrated_mean_width_over_scale": float(((fin["hi"] - fin["lo"]) / fin["scale"]).mean()),
        "calibrated_coverage_by_horizon": {int(h): coverage(g, "lo", "hi") for h, g in fin.groupby("h")},
    }

    per_sku = fin.groupby("sku").apply(
        lambda g: pd.Series({"wape_model": wape(g["y"], g[best_gbm]),
                             "wape_baseline": wape(g["y"], g[best_base]),
                             "units": g["y"].sum()}) if g["y"].sum() > 0 else None,
        include_groups=False).dropna()
    per_sku.to_csv(tables / "per_product.csv")

    metrics = {
        "data": json.loads((C.PROCESSED_DIR / "cleaning_log.json").read_text()),
        "design": {
            "folds_total": int(preds["fold"].nunique()),
            "folds_selection": int(sel["fold"].nunique()),
            "folds_final": int(fin["fold"].nunique()),
            "horizon_weeks": C.HORIZON,
            "final_first_week": str(fin["week"].min().date()),
            "final_last_week": str(fin["week"].max().date()),
            "final_rows_scored": int(len(fin)),
            "final_units": float(fin["y"].sum()),
            "closed_rows_excluded": int(preds["closed"].sum()),
        },
        "chosen_model": best_gbm, "chosen_model_label": LABELS[best_gbm],
        "best_baseline": best_base, "best_baseline_label": LABELS[best_base],
        "best_baseline_on_final_folds": base_fin,
        "best_baseline_on_final_folds_label": LABELS[base_fin],
        "selection_scores": sel_scores.to_dict("records"),
        "final_scores": fin_scores.to_dict("records"),
        "vs_best_baseline": bootstrap_diff(fin, best_gbm, best_base, rng),
        "vs_best_baseline_on_final_folds": bootstrap_diff(fin, best_gbm, base_fin, rng),
        "vs_naive": bootstrap_diff(fin, best_gbm, "naive", rng),
        "products_where_model_beats_baseline": float((per_sku["wape_model"] < per_sku["wape_baseline"]).mean()),
        "segments": {k: d.to_dict("records") for k, d in segments.items()},
        "interval": interval,
        "feature_importance": json.loads((C.PROCESSED_DIR / "feature_importance.json").read_text()),
    }
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print(fin_scores[["label", "wape", "bias", "weekly_total_wape"]].round(4).to_string(index=False))
    print("chosen:", best_gbm, "| best baseline (selection folds):", best_base,
          "| best baseline (final folds):", base_fin)
    print("vs best baseline:", json.dumps(metrics["vs_best_baseline"], indent=1))
    print("interval:", json.dumps(interval, indent=1))
    return metrics


if __name__ == "__main__":
    main()
