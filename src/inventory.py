"""Replenishment simulation: what the forecast is worth in stock.

The same stock policy is run twice over the backtest weeks, once fed by the forecast and once by the benchmark
(13-week average), product by product:

- Periodic review every week, order-up-to level, lost sales (unmet demand is gone), orders arrive after the lead
  time.
- Order-up-to level = (lead time + 1 week) x (forecast for the week + recent average miss) + safety stock.
- Safety stock = z x (spread of the recent misses) x sqrt(lead time + 1 week), with z from the service target.
  "Recent" means the 13 latest weeks whose sales were already known; nothing later is used.

Only the final-test weeks are scored (the earlier weeks warm the simulation up). Money figures use a unit cost of
`cost_share` x the product's average selling price and a yearly holding rate; both are assumptions, shown in the
app and editable there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

WINDOW = 13


@dataclass
class Grid:
    skus: np.ndarray
    weeks: pd.DatetimeIndex
    y: np.ndarray            # (products, weeks) actual units
    final: np.ndarray        # (weeks,) True for final-test weeks
    price: np.ndarray        # (products,) average selling price
    fc: dict                 # method -> (products, weeks) forecast for the week, made before it


def make_grid(preds: pd.DataFrame, panel: pd.DataFrame, methods: list[str]) -> Grid:
    p = preds.sort_values(["sku", "week"])
    skus = np.array(sorted(p["sku"].unique()))
    weeks = pd.DatetimeIndex(sorted(p["week"].unique()))
    piv = lambda c: p.pivot_table(index="sku", columns="week", values=c, aggfunc="first").reindex(index=skus, columns=weeks)
    y = piv("y").fillna(0).to_numpy(float)
    first = weeks[0]
    hist = panel[panel["week"] < first]
    pr = (hist.groupby("sku")["revenue"].sum() / hist.groupby("sku")["units"].sum().replace(0, np.nan)).reindex(skus)
    pr = pr.fillna(panel.groupby("sku")["price"].median().reindex(skus)).fillna(1.0).to_numpy(float)
    ft = p.drop_duplicates("week").set_index("week")["fold_type"].reindex(weeks)
    return Grid(skus, weeks, y, (ft == "final").to_numpy(), pr, {m: piv(m).fillna(0).to_numpy(float) for m in methods})


def simulate(g: Grid, method: str, lead: int = 1, z: float = 1.645, cost_share: float = 0.45) -> dict:
    """Run the policy for every product at once; returns totals over the final-test weeks."""
    y, f = g.y, g.fc[method]
    n, T = y.shape
    cover = lead + 1
    err = np.full((n, T), np.nan)
    on_hand = np.zeros(n)
    pipe = np.zeros((n, lead + 1))                     # pipe[:, k] arrives in k weeks
    served = short = inv_sum = ss_sum = demand = 0.0
    nf = 0
    for t in range(T):
        lo = max(0, t - WINDOW)
        e = err[:, lo:t]
        cnt = np.isfinite(e).sum(1)
        mean_e = np.where(cnt > 0, np.nanmean(np.where(cnt[:, None] > 0, e, 0.0), 1), 0.0) if t > 0 else np.zeros(n)
        if t > 0:
            sd = np.nanstd(e, 1)
        else:
            sd = np.zeros(n)
        sd = np.where(cnt >= 4, sd, 0.5 * f[:, t] + 1.0)
        rate = np.maximum(f[:, t] + mean_e, 0.0)
        ss = max(z, -5) * sd * np.sqrt(cover)
        S = np.maximum(cover * rate + ss, 0.0)
        if t == 0:
            on_hand = S.copy()
        on_hand += pipe[:, 0]
        pipe = np.roll(pipe, -1, axis=1)
        pipe[:, -1] = 0.0
        position = on_hand + pipe.sum(1)
        order = np.maximum(S - position, 0.0)
        pipe[:, lead - 1 if lead >= 1 else 0] += order
        d = y[:, t]
        sv = np.minimum(on_hand, d)
        on_hand -= sv
        err[:, t] = d - f[:, t]
        if g.final[t]:
            served += sv.sum()
            short += (d - sv).sum()
            demand += d.sum()
            inv_sum += (on_hand * g.price).sum()
            ss_sum += (np.maximum(ss, 0) * g.price).sum()
            nf += 1
            if nf == 1:
                short_by = np.zeros(n)
                inv_units = np.zeros(n)
            short_by += d - sv
            inv_units += on_hand
    weekly_demand = demand / max(nf, 1)
    inv_units_total = inv_units.sum() / max(nf, 1)
    return {
        "fill_rate": served / demand if demand else np.nan,
        "units_short": short, "share_short": short / demand if demand else np.nan,
        "lost_revenue": float((short_by * g.price).sum()),
        "avg_inventory_units": inv_units_total,
        "avg_inventory_value": cost_share * inv_sum / max(nf, 1),
        "avg_safety_stock_value": cost_share * ss_sum / max(nf, 1),
        "weeks_of_cover": inv_units_total / weekly_demand if weekly_demand else np.nan,
        "weeks": nf,
    }


def z_for_fill(g: Grid, method: str, target_fill: float, lead: int, cost_share: float) -> float:
    lo, hi = -2.0, 5.0
    for _ in range(16):
        mid = (lo + hi) / 2
        if simulate(g, method, lead, mid, cost_share)["fill_rate"] < target_fill:
            lo = mid
        else:
            hi = mid
    return hi


def compare(g: Grid, model: str, base: str = "ma13", lead: int = 1, service: float = 0.95,
            cost_share: float = 0.45, holding: float = 0.25) -> dict:
    """Same policy for both methods, then the model's stock when it is tuned to the benchmark's fill rate."""
    z = float(norm.ppf(service))
    m, b = simulate(g, model, lead, z, cost_share), simulate(g, base, lead, z, cost_share)
    z_eq = z_for_fill(g, model, b["fill_rate"], lead, cost_share)
    m_eq = simulate(g, model, lead, z_eq, cost_share)
    years = m["weeks"] / 52.0
    margin = 1 - cost_share
    return {
        "inputs": {"lead_weeks": lead, "review_weeks": 1, "service_target": service, "z": z, "cost_share": cost_share,
                   "holding_rate": holding, "weeks_scored": m["weeks"]},
        "model": m, "base": b, "model_same_service": m_eq, "z_same_service": z_eq,
        "inventory_change_same_service": m_eq["avg_inventory_value"] / b["avg_inventory_value"] - 1,
        "carrying_saving_per_year": (b["avg_inventory_value"] - m_eq["avg_inventory_value"]) * holding,
        "lost_margin_avoided_per_year": (b["lost_revenue"] - m["lost_revenue"]) * margin / years,
        "lost_revenue_avoided_per_year": (b["lost_revenue"] - m["lost_revenue"]) / years,
    }


LEADS = (1, 2, 3, 4)
SERVICES = (0.90, 0.95, 0.98)


def run(methods: tuple[str, ...] = ("ensemble", "gbm_tweedie", "gbm_log"), base: str = "ma13") -> dict:
    """Every lead time x service target for each forecast setting, money at unit cost = price (scaled in the app).

    Writes reports/inventory.json. Money values scale linearly with the unit-cost share and the holding rate, so the
    app applies those two editable assumptions without rerunning the simulation.
    """
    import json
    import time

    from src import config as C
    t0 = time.time()
    preds = pd.read_parquet(C.PROCESSED_DIR / "predictions.parquet")
    panel = pd.read_parquet(C.PROCESSED_DIR / "panel.parquet")
    methods = tuple(m for m in methods if m in preds.columns)
    g = make_grid(preds, panel, list(methods) + [base])
    out = {"base": base, "products": int(len(g.skus)), "weeks_scored": int(g.final.sum()),
           "final_start": str(g.weeks[g.final][0].date()), "final_end": str(g.weeks[g.final][-1].date()),
           "weekly_demand": float(g.y[:, g.final].sum() / g.final.sum()), "runs": []}
    for lead in LEADS:
        for s in SERVICES:
            z = float(norm.ppf(s))
            b = simulate(g, base, lead, z, 1.0)
            for m in methods:
                r = simulate(g, m, lead, z, 1.0)
                z_eq = z_for_fill(g, m, b["fill_rate"], lead, 1.0)
                r_eq = simulate(g, m, lead, z_eq, 1.0)
                out["runs"].append({"model": m, "lead": lead, "service": s, "z": z, "z_same_service": z_eq,
                                    "model_run": r, "base_run": b, "model_same_service": r_eq})
    out["runtime_s"] = round(time.time() - t0, 1)
    (C.REPORTS_DIR / "inventory.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"inventory.json: {len(out['runs'])} runs in {out['runtime_s']} s")
    return out


if __name__ == "__main__":
    run()
