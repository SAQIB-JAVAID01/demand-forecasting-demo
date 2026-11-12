"""Planner dashboard for the backtest results.

Run:  streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.report import BAND, BLUE, INK2, ORANGE, pct, style  # noqa: E402

st.set_page_config(page_title="Demand forecast", layout="wide")
style()


@st.cache_data
def load():
    preds = pd.read_parquet(C.PROCESSED_DIR / "predictions.parquet")
    panel = pd.read_parquet(C.PROCESSED_DIR / "panel.parquet")
    metrics = json.loads((C.REPORTS_DIR / "metrics.json").read_text())
    return preds, panel, metrics


preds, panel, m = load()
model, base = m["chosen_model"], m["best_baseline"]
row = {r["model"]: r for r in m["final_scores"]}

st.title("Demand forecast: UK online retailer")
st.caption(f"{m['data']['skus_selected']} products, weekly forecasts 1 to 4 weeks ahead. "
           f"Final test period {m['design']['final_first_week']} to {m['design']['final_last_week']}. "
           "Every forecast shown was made before the week it covers.")

a, b, c, d = st.columns(4)
a.metric("Model error (WAPE)", pct(row[model]["wape"]))
b.metric(f"Baseline: {m['best_baseline_label'].lower()}", pct(row[base]["wape"]))
vs = m["vs_best_baseline"]
c.metric("Error vs baseline", f"{100 * vs['relative_change']:+.0f}%",
         help=f"95% CI {100 * vs['relative_ci95'][0]:+.0f}% to {100 * vs['relative_ci95'][1]:+.0f}%")
d.metric("80% interval coverage", pct(m["interval"]["calibrated_coverage_final"], 0))

with st.sidebar:
    st.header("Filters")
    abc = st.multiselect("Revenue class", ["A", "B", "C"], default=["A", "B", "C"])
    kinds = st.multiselect("Demand pattern", ["regular", "intermittent"], default=["regular", "intermittent"])
    horizons = st.multiselect("Weeks ahead", [1, 2, 3, 4], default=[1, 2, 3, 4])

sel = preds[preds["abc"].isin(abc) & preds["demand_type"].isin(kinds) & preds["h"].isin(horizons)
            & ~preds["closed"]]
fin = sel[sel["fold_type"] == "final"]
if fin.empty:
    st.warning("No forecasts match these filters.")
    st.stop()

left, right = st.columns([3, 2])
with left:
    st.subheader("Total weekly demand")
    g = fin.groupby("week")[["y", model, base]].sum()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(g.index, g["y"], color=INK2, label="Actual")
    ax.plot(g.index, g[model], color=BLUE, label="Model")
    ax.plot(g.index, g[base], color=ORANGE, linewidth=1.6, label="Baseline")
    ax.set_ylim(0, None)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=3)
    st.pyplot(fig)
with right:
    st.subheader("Error for this selection")
    y = fin["y"].sum()
    st.dataframe(pd.DataFrame({
        "Method": ["Model", "Baseline"],
        "WAPE": [pct((fin[model] - fin["y"]).abs().sum() / y), pct((fin[base] - fin["y"]).abs().sum() / y)],
        "Bias": [f"{100 * (fin[model] - fin['y']).sum() / y:+.1f}%", f"{100 * (fin[base] - fin['y']).sum() / y:+.1f}%"],
    }), hide_index=True, width="stretch")
    st.caption(f"{len(fin):,} product-week forecasts, {y:,.0f} units.")

st.subheader("One product")
names = (sel.drop_duplicates("sku").assign(name=lambda d: d["sku"] + "  " + d["description"].str.title())
            .sort_values("name"))
choice = st.selectbox("Product", names["name"])
sku = names.loc[names["name"] == choice, "sku"].iloc[0]
p = preds[preds["sku"] == sku].sort_values("week")
hist = panel[panel["sku"] == sku]
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.fill_between(p["week"], p["lo"], p["hi"], color=BAND, linewidth=0, label="80% interval")
ax.plot(hist["week"], hist["units"], color=INK2, linewidth=1.6, label="Actual")
ax.plot(p["week"], p[model], color=BLUE, label="Forecast")
ax.set_ylim(0, None)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", ncols=3)
st.pyplot(fig)

table = p[["week", "h", "fold_type", "y", model, "lo", "hi", base]].rename(columns={
    "week": "Week", "h": "Weeks ahead", "fold_type": "Fold type", "y": "Actual", model: "Forecast",
    "lo": "Low (10%)", "hi": "High (90%)", base: "Baseline"})
st.dataframe(table.round({c: 1 for c in ["Actual", "Forecast", "Low (10%)", "High (90%)", "Baseline"]}), hide_index=True, width="stretch")
st.download_button("Download this product's forecasts (CSV)", table.to_csv(index=False), f"{sku}_forecast.csv")
