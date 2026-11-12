"""Step 6: build a single-file interactive dashboard (no server, no CDN).

Run:  python -m src.build_dashboard
Writes app/dashboard.html. Open it by double-clicking; it also works on
GitHub Pages. All data is embedded in the file.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C

TEMPLATE = C.ROOT / "app" / "dashboard_template.html"
OUT = C.ROOT / "app" / "dashboard.html"


def r1(x) -> float | None:
    return None if pd.isna(x) else round(float(x), 1)


def build_data() -> dict:
    preds = pd.read_parquet(C.PROCESSED_DIR / "predictions.parquet")
    panel = pd.read_parquet(C.PROCESSED_DIR / "panel.parquet")
    cal = pd.read_parquet(C.PROCESSED_DIR / "calendar.parquet")
    m = json.loads((C.REPORTS_DIR / "metrics.json").read_text())
    model, base = m["chosen_model"], m["best_baseline"]
    n_weeks = len(cal)

    products = []
    for sku, g in panel.groupby("sku", sort=True):
        units = [None] * n_weeks
        for t, u in zip(g["t"], g["units"]):
            units[int(t)] = r1(u)
        p = preds[preds["sku"] == sku].sort_values("t")
        rows = [[int(t), int(h), 1 if ft == "final" else 0, 1 if cl else 0, r1(y), r1(a), r1(b), r1(tw), r1(lo), r1(hi)]
                for t, h, ft, cl, y, a, b, tw, lo, hi in zip(
                    p["t"], p["h"], p["fold_type"], p["closed"], p["y"], p[model], p[base],
                    p["gbm_tweedie"], p["lo"], p["hi"])]
        products.append({"s": sku, "d": str(g["description"].iloc[0]).title(), "a": g["abc"].iloc[0],
                         "t": g["demand_type"].iloc[0], "u": units, "f": rows})

    return {
        "weeks": [w.strftime("%Y-%m-%d") for w in cal["week"]],
        "closed": [int(c) for c in cal["closed"]],
        "products": products,
        "modelLabel": m["chosen_model_label"],
        "baseLabel": m["best_baseline_label"],
        "finalScores": sorted(
            [{"label": r["label"], "wape": r["wape"], "bias": r["bias"], "total": r["weekly_total_wape"],
              "chosen": bool(r["chosen"])} for r in m["final_scores"]], key=lambda r: r["wape"]),
        "vsStrongest": m["vs_best_baseline_on_final_folds"],
        "strongestLabel": m["best_baseline_on_final_folds_label"],
        "design": m["design"],
        "data": {k: m["data"][k] for k in ("raw_rows", "clean_rows", "skus_selected", "weeks",
                                           "cancellations_matched_to_a_sale", "date_min", "date_max")},
        "interval": m["interval"],
    }


def main() -> None:
    data = build_data()
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/", json.dumps(data, separators=(",", ":"), allow_nan=False))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
