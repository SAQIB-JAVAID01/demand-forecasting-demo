"""Step 5: figures and the written report, generated from metrics.json.

Run:  python -m src.report
Every number in reports/REPORT.md is filled in from the backtest output, so the
text cannot drift from the results.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .evaluate import LABELS

# palette (validated with the dataviz validator: blue + orange, light surface)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, NEUTRAL, BAND = "#2a78d6", "#eb6834", "#c3c2b7", "#cde2fb"

FEATURE_NAMES = {
    "last": "Most recent known week", "lag_1": "Week before that", "lag_2": "2 weeks before",
    "lag_3": "3 weeks before", "mean_4": "4-week average", "mean_8": "8-week average",
    "mean_13": "13-week average", "mean_26": "26-week average", "std_8": "8-week variability",
    "max_13": "13-week maximum", "nonzero_8": "Share of weeks with sales (8w)",
    "weeks_since_sale": "Weeks since last sale", "same_week_last_year": "Same week last year",
    "trend_4_13": "Short vs long average (trend)", "price_last": "Latest price",
    "price_ratio": "Price vs 13-week average", "orders_mean_4": "Orders per week (4w)",
    "total_last": "All-product demand, latest", "total_mean_4": "All-product demand (4w)",
    "total_yoy": "All-product demand vs last year", "share_13": "Product share of demand",
    "trading_days_last": "Trading days, latest week", "age_weeks": "Weeks since launch",
    "weekofyear": "Week of year", "month": "Month", "weeks_to_xmas": "Weeks to Christmas",
    "trading_days": "Trading days in target week",
}


def style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 2, "legend.frameon": False,
    })


def title(ax, head: str, sub: str = ""):
    ax.set_title(head, pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, color=INK2, fontsize=10, va="bottom")


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def fig_accuracy(m: dict):
    df = pd.DataFrame(m["final_scores"]).sort_values("wape", ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = [BLUE if c else NEUTRAL for c in df["chosen"]]
    ax.barh(df["label"], df["wape"], color=colors, height=0.62)
    for y, (v, ch) in enumerate(zip(df["wape"], df["chosen"])):
        ax.text(v + 0.006, y, pct(v), va="center", color=INK, fontweight="bold" if ch else "normal")
    ax.set_xlim(0, df["wape"].max() * 1.15)
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Forecast error (WAPE), lower is better")
    title(ax, "Forecast error by method",
          f"Final test folds, {m['design']['final_first_week']} to {m['design']['final_last_week']}; "
          "blue = setting chosen on earlier folds")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "01_error_by_method.png", dpi=170)
    plt.close(fig)


def fig_weekly(preds: pd.DataFrame, m: dict):
    fin = preds[preds["fold_type"] == "final"]
    g = fin.groupby("week")[["y", m["chosen_model"], m["best_baseline"]]].sum()
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.plot(g.index, g["y"], color=INK2, label="Actual units sold")
    ax.plot(g.index, g[m["chosen_model"]], color=BLUE, label="Model forecast (1 to 4 weeks ahead)")
    ax.plot(g.index, g[m["best_baseline"]], color=ORANGE, linewidth=1.6,
            label=f"Baseline: {m['best_baseline_label'].lower()}")
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=1)
    ax.set_ylabel("Units per week, all forecast products")
    title(ax, "Total weekly demand: actual vs forecast",
          "Each week's forecast was made before that week, using only earlier data")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "02_weekly_total.png", dpi=170)
    plt.close(fig)


def fig_horizon(m: dict):
    d = pd.DataFrame(m["segments"]["horizon"])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(d["h"], d["wape_baseline"], color=ORANGE, marker="o", markersize=7,
            label=f"Baseline: {m['best_baseline_label'].lower()}")
    ax.plot(d["h"], d["wape_model"], color=BLUE, marker="o", markersize=7, label="Model")
    for x, a, b in zip(d["h"], d["wape_model"], d["wape_baseline"]):
        ax.text(x, a - 0.02, pct(a), ha="center", va="top", color=INK)
        ax.text(x, b + 0.02, pct(b), ha="center", va="bottom", color=INK)
    ax.set_xticks(d["h"])
    lo = min(d["wape_model"].min(), d["wape_baseline"].min())
    hi = max(d["wape_model"].max(), d["wape_baseline"].max())
    ax.set_ylim(max(0, lo - 0.12), hi + 0.12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_xlabel("Weeks ahead")
    ax.set_ylabel("Forecast error (WAPE)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    title(ax, "Error by how far ahead the forecast is", "Final test folds")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "03_error_by_horizon.png", dpi=170)
    plt.close(fig)


def fig_segments(m: dict):
    parts = [("abc", "abc", {"A": "A: top 20% by revenue", "B": "B: next 30%", "C": "C: remaining 50%"}),
             ("demand_type", "demand_type", {"regular": "Regular sellers", "intermittent": "Intermittent sellers"}),
             ("season", "season", {"Sep-Nov peak": "Sep to Nov (pre-Christmas peak)", "Dec-Aug": "Dec to Aug"})]
    rows = []
    for key, col, names in parts:
        for r in m["segments"][key]:
            rows.append((names.get(r[col], str(r[col])), r["wape_model"], r["wape_baseline"]))
    d = pd.DataFrame(rows, columns=["segment", "model", "baseline"])[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    y = np.arange(len(d))
    ax.barh(y + 0.19, d["baseline"], height=0.34, color=ORANGE,
            label=f"Baseline: {m['best_baseline_label'].lower()}")
    ax.barh(y - 0.19, d["model"], height=0.34, color=BLUE, label="Model")
    for i, (a, b) in enumerate(zip(d["model"], d["baseline"])):
        ax.text(b + 0.006, i + 0.19, pct(b), va="center", color=INK, fontsize=9.5)
        ax.text(a + 0.006, i - 0.19, pct(a), va="center", color=INK, fontsize=9.5)
    ax.set_yticks(y, d["segment"])
    ax.set_xlim(0, max(d["model"].max(), d["baseline"].max()) * 1.55)
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Forecast error (WAPE), lower is better")
    ax.legend(loc="center right")
    title(ax, "Where the model helps and where it does not", "Final test folds, by product group and season")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "04_error_by_segment.png", dpi=170)
    plt.close(fig)


def pick_examples(preds: pd.DataFrame, m: dict) -> list[str]:
    """Not hand-picked: the products at the 25th, 50th and 75th percentile of
    model error among the A and B classes."""
    fin = preds[(preds["fold_type"] == "final") & ~preds["closed"] & preds["abc"].isin(["A", "B"])]
    e = fin.assign(ae=(fin[m["chosen_model"]] - fin["y"]).abs()).groupby("sku")[["ae", "y"]].sum()
    e = (e["ae"] / e["y"]).sort_values()
    return [e.index[int(q * (len(e) - 1))] for q in (0.25, 0.5, 0.75)]


def fig_examples(preds: pd.DataFrame, panel: pd.DataFrame, m: dict):
    skus = pick_examples(preds, m)
    fig, axes = plt.subplots(len(skus), 1, figsize=(9, 8.2), sharex=True)
    start = preds["week"].min() - pd.Timedelta(weeks=12)
    for ax, sku, tag in zip(axes, skus, ("better than most (25th percentile of error)",
                                          "typical (median error)",
                                          "worse than most (75th percentile of error)")):
        hist = panel[(panel["sku"] == sku) & (panel["week"] >= start)]
        p = preds[preds["sku"] == sku].sort_values("week")
        ax.fill_between(p["week"], p["lo"], p["hi"], color=BAND, linewidth=0, label="80% interval")
        ax.plot(hist["week"], hist["units"], color=INK2, linewidth=1.6, label="Actual")
        ax.plot(p["week"], p[m["chosen_model"]], color=BLUE, label="Forecast")
        ax.axvline(preds.loc[preds["fold_type"] == "final", "week"].min(), color=AXIS, linewidth=1, linestyle="--")
        ax.set_ylim(0, None)
        ax.grid(axis="x", visible=False)
        desc = str(p["description"].iloc[0]).title()
        ax.set_title(f"{desc} ({sku}): {tag}", fontsize=10.5, pad=6)
    axes[0].legend(loc="upper right", ncols=3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle("Three products chosen by rule, not by eye", x=0.06, y=0.985, ha="left", fontweight="bold", fontsize=13)
    fig.text(0.06, 0.935, "Units per week; dashed line = start of the final test folds", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(C.FIGURES_DIR / "05_example_products.png", dpi=170)
    plt.close(fig)
    return skus


def fig_importance(m: dict):
    imp = pd.Series(m["feature_importance"]).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.barh([FEATURE_NAMES.get(k, k) for k in imp.index], imp.values, color=BLUE, height=0.62)
    for y, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.012, y, pct(v, 0), va="center", color=INK)
    ax.set_xlim(0, imp.max() * 1.14)
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of the model's total gain")
    title(ax, "What the model relies on", "Top 12 inputs; Tweedie-setting models on the final folds")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "06_feature_importance.png", dpi=170)
    plt.close(fig)


def fig_cover(preds: pd.DataFrame, m: dict):
    """One image for a portfolio thumbnail: three headline figures and the weekly chart."""
    fin = preds[preds["fold_type"] == "final"]
    g = fin.groupby("week")[["y", m["chosen_model"]]].sum()
    row = {r["model"]: r for r in m["final_scores"]}
    vs = m["vs_best_baseline_on_final_folds"]
    fig = plt.figure(figsize=(10, 5.6))
    fig.text(0.05, 0.92, "Demand forecasting for a UK online retailer", fontsize=19, fontweight="bold")
    fig.text(0.05, 0.865, f"{m['data']['skus_selected']} products, weekly forecasts 1 to 4 weeks ahead, "
             f"tested on {m['design']['folds_final']} later periods the model never saw", color=INK2, fontsize=11)
    tiles = [(pct(row[m["chosen_model"]]["wape"]), "forecast error (WAPE), product-week level"),
             (f"{100 * vs['relative_change']:+.0f}%", f"error vs the strongest simple method\n(95% CI {100 * vs['relative_ci95'][0]:+.0f}% to {100 * vs['relative_ci95'][1]:+.0f}%)"),
             (f"{100 * row[m['chosen_model']]['bias']:+.0f}%", "bias: forecasts run low on average\n(trade-off explained in the report)")]
    for i, (big, small) in enumerate(tiles):
        x = 0.05 + i * 0.31
        fig.text(x, 0.74, big, fontsize=26, fontweight="bold", color=INK)
        fig.text(x, 0.725, small, fontsize=10, color=INK2, va="top")
    ax = fig.add_axes((0.07, 0.1, 0.88, 0.5))
    ax.plot(g.index, g["y"], color=INK2, label="Actual units sold")
    ax.plot(g.index, g[m["chosen_model"]], color=BLUE, label="Forecast")
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2)
    fig.text(0.05, 0.02, "Public data: UCI Online Retail II (CC BY 4.0). Python, pandas, LightGBM.",
             color=MUTED, fontsize=9)
    fig.savefig(C.FIGURES_DIR / "00_cover.png", dpi=170)
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def write_report(m: dict, example_skus: list[str]):
    d, des, vs, iv = m["data"], m["design"], m["vs_best_baseline"], m["interval"]
    fs = pd.DataFrame(m["final_scores"]).sort_values("wape")
    row = {r["model"]: r for r in m["final_scores"]}
    model, base = row[m["chosen_model"]], row[m["best_baseline"]]
    lo, hi = vs["relative_ci95"]
    clear = hi < 0
    if clear:
        verdict = (f"The model's error is {abs(100 * vs['relative_change']):.0f}% lower than the reference simple method "
                   f"(95% confidence interval: {abs(100 * hi):.0f}% to {abs(100 * lo):.0f}% lower).")
    elif lo > 0:
        verdict = (f"The model did WORSE than the reference simple method: its error is {100 * vs['relative_change']:.0f}% higher "
                   f"(95% confidence interval: {100 * lo:.0f}% to {100 * hi:.0f}% higher).")
    else:
        verdict = (f"The model's error is {100 * vs['relative_change']:+.0f}% against the reference simple method, but the 95% "
                   f"confidence interval ({100 * lo:+.0f}% to {100 * hi:+.0f}%) includes zero, so this test does not show a clear gain.")

    t_scores = pd.DataFrame({
        "Method": fs["label"] + np.where(fs["chosen"], " (chosen)", ""),
        "WAPE": fs["wape"].map(pct), "Bias": fs["bias"].map(lambda v: f"{100 * v:+.1f}%"),
        "WAPE on weekly total": fs["weekly_total_wape"].map(pct)})

    def seg_table(key, col, name):
        s = pd.DataFrame(m["segments"][key])
        return md_table(pd.DataFrame({
            name: s[col], "Model WAPE": s["wape_model"].map(pct), "Baseline WAPE": s["wape_baseline"].map(pct),
            "Change": s["relative_change"].map(lambda v: f"{100 * v:+.0f}%"),
            "Model bias": s["bias_model"].map(lambda v: f"{100 * v:+.1f}%")}))

    other_base = ""
    if m["best_baseline_on_final_folds"] != m["best_baseline"]:
        v2 = m["vs_best_baseline_on_final_folds"]
        other_base = (f"\n\nOn the final folds a different simple method, {m['best_baseline_on_final_folds_label'].lower()}, "
                      f"turned out to be the strongest baseline. Against that one the model's error is "
                      f"{100 * v2['relative_change']:+.0f}% (95% CI {100 * v2['relative_ci95'][0]:+.0f}% to "
                      f"{100 * v2['relative_ci95'][1]:+.0f}%).")

    v2 = m["vs_best_baseline_on_final_folds"]
    strongest = (f"Against the strongest simple method on the final folds, {m['best_baseline_on_final_folds_label'].lower()} "
                 f"({pct(row[m['best_baseline_on_final_folds']]['wape'])}), the model's error is "
                 f"{100 * v2['relative_change']:+.0f}% (95% CI {100 * v2['relative_ci95'][0]:+.0f}% to "
                 f"{100 * v2['relative_ci95'][1]:+.0f}%). This is the tougher comparison.")
    tw = row["gbm_tweedie"]
    best_simple_total = min(row[b]["weekly_total_wape"] for b in ("naive", "ma4", "ma13", "snaive", "ses"))
    if abs(model["bias"]) > 0.10:
        tradeoff = (
            f"The chosen setting was picked for the lowest WAPE on the selection folds, and that rule has a cost. "
            f"Minimising absolute error pulls a forecast toward the median, and for skewed demand the median is below "
            f"the average. The chosen model therefore forecasts {abs(100 * model['bias']):.0f}% fewer units than were "
            f"actually sold, and its error on total weekly demand ({pct(model['weekly_total_wape'])}) is worse than the "
            f"best simple method's ({pct(best_simple_total)}).\n\n"
            f"The Tweedie setting aims at the average instead. Its bias is {100 * tw['bias']:+.1f}% and its error on "
            f"total weekly demand is {pct(tw['weekly_total_wape'])}, but its product-level WAPE is {pct(tw['wape'])}, "
            f"only slightly better than the simple averages.\n\n"
            f"What this means in practice: use the chosen setting when the question is 'what will a typical week look "
            f"like for this product', and the Tweedie setting (or the upper interval bound) when the question is 'how "
            f"much stock do we need in total'. Neither setting is good at both on this data. No setting was changed "
            f"after the final folds were scored; the three loss settings above are the only ones that were run.")
    else:
        tradeoff = (f"The chosen model's bias is {100 * model['bias']:+.1f}%, so its totals are close to actual sales.")
    cov_h = ", ".join(f"{h} wk: {pct(v, 0)}" for h, v in iv["calibrated_coverage_by_horizon"].items())
    text = f"""# Demand forecasting for a UK online retailer

Weekly unit forecasts, 1 to 4 weeks ahead, for the {d['skus_selected']} highest-revenue products of a UK online
gift retailer. Public data, fully reproducible. Every number below is generated by the code in this repository.

![Cover](figures/00_cover.png)

## Result in short

- On the final test folds ({des['final_first_week']} to {des['final_last_week']}, {des['final_rows_scored']:,} product-week
  forecasts) the chosen model has a forecast error (WAPE) of **{pct(model['wape'])}**. The reference simple method,
  {m['best_baseline_label'].lower()} (the best one on the selection folds), has {pct(base['wape'])}.
- {verdict}
- {strongest}
- On total weekly demand across all products the model's error is {pct(model['weekly_total_wape'])}.
- The model beats the baseline on {pct(m['products_where_model_beats_baseline'], 0)} of products.
- The 80% prediction interval contained the actual value {pct(iv['calibrated_coverage_final'], 0)} of the time.
- Model bias is {100 * model['bias']:+.1f}% of actual units (negative means it forecasts too low on average). See
  "The trade-off" below; this matters for stock planning.

WAPE is total absolute error divided by total actual units. A WAPE of 40% means the forecasts were off by 40 units
for every 100 sold. Product-week demand for gift items is noisy, so errors at this level are large for every method;
the comparison with simple methods is what matters.

## Data

Source: UCI Machine Learning Repository, "Online Retail II" (licence CC BY 4.0). All transactions of a UK online
retailer between {d['date_min']} and {d['date_max']}.

| Cleaning step | Rows |
|---|---|
| Raw transaction rows | {d['raw_rows']:,} |
| Removed: duplicate period shared by the two source sheets | {d['rows_dropped_sheet_overlap']:,} |
| Cancellation rows | {d['cancellation_rows']:,} |
| Cancellations matched to their original sale (sale removed too) | {d['cancellations_matched_to_a_sale']:,} |
| Removed: zero or negative quantity or price | {d['rows_dropped_nonpositive_qty_or_price']:,} |
| Removed: non-product codes (postage, fees, adjustments) | {d['rows_dropped_non_product_codes']:,} |
| Clean rows kept | {d['clean_rows']:,} |

Cancelled sales matter: {d['units_removed_as_cancelled_sales']:,} units were sold and then cancelled. Leaving them in
would count orders that never shipped as demand.

The data was grouped into {d['weeks']} full weeks ({d['week_first']} to {d['week_last']}). The business was shut in
{len(d['closed_weeks'])} of them (Christmas and New Year); those weeks are forecast as zero by every method and are
not scored.

Products: {d['skus_selected']} were selected out of {d['skus_total']:,}, using the first {C.INITIAL_TRAIN_WEEKS} weeks only
(sold in at least {C.MIN_ACTIVE_WEEKS} of those weeks, then ranked by revenue). They made up
{pct(d['selected_share_of_first_year_revenue'], 0)} of first-year revenue. Choosing them from the first year alone
means no information from the test period decided which products are in the study. {pct(d['panel_zero_share'], 0)} of
product-weeks have zero sales.

## Method

**Test design.** A rolling-origin backtest: the forecast date moves forward {C.STEP} weeks at a time, and at each date
the methods see only earlier data and forecast the next {C.HORIZON} weeks. There are {des['folds_total']} forecast dates.
The first {des['folds_selection']} were used to choose the model setting and the reference baseline. The last
{des['folds_final']} are the final test and were not used for any choice.

**Simple methods.** Last week's sales; 4-week average; 13-week average; same week last year; exponential smoothing
with the smoothing factor tuned per product.

**Model.** One gradient-boosting model (LightGBM) trained across all products, with a separate model for each of the
4 horizons. Inputs: recent sales and averages, variability, weeks since the last sale, same week last year, price and
price change, demand across all products, product age, and the calendar of the target week (week of year, weeks to
Christmas, trading days). For a forecast h weeks ahead, the newest sales figure used is h weeks old; automated tests
in `tests/` change the future and confirm the inputs do not move. Three loss settings were tried; the one with the
lowest error on the selection folds was kept ({m['chosen_model_label']}).

**Prediction intervals.** Two quantile models give a 10% and 90% forecast. The interval is then adjusted using the
errors from earlier folds only (conformal calibration), so the adjustment never sees the fold it is applied to.

**Trading days.** The number of trading days in the target week is used as an input. It is taken from the data, on
the assumption that a planner knows the shutdown calendar in advance.

## Results on the final test folds

{md_table(t_scores)}

![Error by method](figures/01_error_by_method.png)

Difference between the model and {m['best_baseline_label'].lower()} (the best simple method on the selection folds):
{100 * vs['diff']:+.1f} WAPE points, 95% CI {100 * vs['ci95'][0]:+.1f} to {100 * vs['ci95'][1]:+.1f}. The interval comes
from {C.N_BOOTSTRAP:,} bootstrap resamples of whole products. The model was better in
{pct(vs['share_of_resamples_model_better'], 0)} of resamples.{other_base}

The other two loss settings are listed in the table for transparency; they were not chosen and their final-fold
scores played no part in the choice.

### The trade-off: lower error per product, but forecasts that run low

{tradeoff}

![Weekly total](figures/02_weekly_total.png)

### By horizon

{seg_table('horizon', 'h', 'Weeks ahead')}

![Error by horizon](figures/03_error_by_horizon.png)

### By product group and season

{seg_table('abc', 'abc', 'Revenue class')}

{seg_table('demand_type', 'demand_type', 'Demand pattern')}

{seg_table('season', 'season', 'Season')}

![Error by segment](figures/04_error_by_segment.png)

Revenue class and demand pattern were assigned from the first year only. "Intermittent" means the product had no
sales in more than 20% of first-year weeks.

### Prediction intervals

| | Coverage of the 80% interval | Average width (relative to recent demand) |
|---|---|---|
| Raw quantile models | {pct(iv['raw_coverage_final'])} | {iv['raw_mean_width_over_scale']:.2f} |
| After calibration on earlier folds | {pct(iv['calibrated_coverage_final'])} | {iv['calibrated_mean_width_over_scale']:.2f} |

Calibrated coverage by horizon: {cov_h}. Actual demand was at or below the upper bound in
{pct(iv['calibrated_below_upper_final'])} of product-weeks; that upper bound is the natural starting point for a
stock level.

### Example products

![Examples](figures/05_example_products.png)

The three products are the ones at the 25th, 50th and 75th percentile of model error among class A and B products
({', '.join(example_skus)}), so the chart shows a good, a typical and a poor case instead of hand-picked ones.

### What the model relies on

![Feature importance](figures/06_feature_importance.png)

Importance was recorded for the Tweedie-setting models. Recent averages carry most of the signal, which is why the
simple averages are hard to beat on this data.

## Limits

- Two years of data. Only one earlier Christmas season is available for training, so the pre-Christmas peak is the
  hardest period and "same week last year" has a single year behind it.
- Sales are not the same as demand: weeks with no stock look like weeks with no demand. The data has no stock levels.
- No promotion calendar is available. Price changes are visible only after they happen.
- The {d['skus_selected']} products are the established top sellers. New and slow products are harder and are not covered.
- One retailer, one period. The size of the gain will differ on other data.
- The model was run once with fixed settings. Likely improvements that were NOT tried: scaling each product's sales
  by its own level before training, a promotion or price-plan input, and hierarchical reconciliation so product
  forecasts add up to an unbiased total.

## Reproduce

```
pip install -r requirements.txt
python run_all.py            # download, clean, backtest, score, report (about 10 minutes)
python -m pytest tests -q    # leakage tests
streamlit run app/dashboard.py
```
"""
    (C.REPORTS_DIR / "REPORT.md").write_text(text, encoding="utf-8")


def main():
    style()
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    m = json.loads((C.REPORTS_DIR / "metrics.json").read_text())
    preds = pd.read_parquet(C.PROCESSED_DIR / "predictions.parquet")
    panel = pd.read_parquet(C.PROCESSED_DIR / "panel.parquet")
    fig_accuracy(m)
    fig_weekly(preds, m)
    fig_horizon(m)
    fig_segments(m)
    skus = fig_examples(preds, panel, m)
    fig_importance(m)
    fig_cover(preds, m)
    write_report(m, skus)
    print("figures and REPORT.md written to", C.REPORTS_DIR)


if __name__ == "__main__":
    main()
