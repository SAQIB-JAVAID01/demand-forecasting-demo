"""Demand Forecasting Studio, web version (Streamlit).

Run:  streamlit run web/streamlit_app.py
A lighter view of the desktop app's key pages. It reads the same backtest output
(data/processed, reports/*.json) and uses the same KPI code (src/kpis.py), so the
numbers match the desktop app. Nothing is trained here and no number is typed in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src import kpis as K  # noqa: E402
from src import scenarios as SC  # noqa: E402

BLUE, ORANGE, GREY, GREEN, AMBER, GRID = "#5b8def", "#f0915a", "#c3cad4", "#3fcf8e", "#f5a524", "#1b212a"
INDICATORS = [("confidence", "Consumer confidence (index pts)", -6.0, 6.0, 0.5),
              ("unemployment", "Unemployment rate (pts)", -2.0, 3.0, 0.1),
              ("inflation", "Inflation, CPI yearly (pts)", -2.0, 4.0, 0.1)]

st.set_page_config(page_title="Demand Forecasting Studio", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data
def load() -> dict:
    def js(name):
        p = C.REPORTS_DIR / name
        return json.loads(p.read_text()) if p.exists() else None
    pp = C.PROCESSED_DIR / "projection.parquet"
    return dict(preds=pd.read_parquet(C.PROCESSED_DIR / "predictions.parquet"),
                panel=pd.read_parquet(C.PROCESSED_DIR / "panel.parquet"),
                metrics=js("metrics.json"), inv=js("inventory.json"), lab=js("model_lab.json"),
                scen=js("scenarios.json"), proj=pd.read_parquet(pp) if pp.exists() else None)


@st.cache_data
def headlines(abc: tuple, demand: tuple, horizons: tuple) -> dict:
    fin = K.final_rows(load()["preds"], abc, demand, horizons)
    return {k: K.headline(fin, k) for k in K.settings_for(load()["preds"])}


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def spct(x: float, d: int = 1) -> str:
    return f"{100 * x:+.{d}f}%"


def pts(x: float) -> str:
    return f"{100 * x:+.1f} pts"


VERDICT = {"better": "better than the benchmark", "worse": "below the benchmark", "level": "level with the benchmark"}


def style(chart: alt.Chart, height: int = 320) -> alt.Chart:
    return (chart.properties(height=height)
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID, labelColor="#a3abb7", titleColor="#a3abb7",
                            labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(labelColor="#c3cad4", titleColor="#a3abb7", orient="top", labelFontSize=12,
                              symbolStrokeWidth=3))


def table(df: pd.DataFrame, **cfg):
    """A read-only table sized to show every row (no inner scroll bar)."""
    st.dataframe(df, hide_index=True, use_container_width=True, height=36 * (len(df) + 1) + 3,
                 column_config={k: st.column_config.TextColumn(k, width=w) for k, w in cfg.items()} or None)


def metric(col, label: str, value: str, note: str, tone: str = "neutral", help: str | None = None):
    """A headline tile. tone: good (green), bad (red) or neutral (grey). No arrow: the note is a label, not a change."""
    neg = note.lstrip()[:1] in ("-", "−")    # Streamlit colours a note by its sign: "normal" = positive green
    color = "off" if tone == "neutral" else ("inverse" if (tone == "good") == neg else "normal")
    col.metric(label, value, note, delta_color=color, delta_arrow="off", help=help)


def line_chart(df: pd.DataFrame, colors: dict, y_title: str = "Units", height: int = 320) -> alt.Chart:
    long = df.reset_index().melt(df.index.name or "index", var_name="Series", value_name="value")
    x = df.index.name or "index"
    ch = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X(f"{x}:T", title=None), y=alt.Y("value:Q", title=y_title),
        color=alt.Color("Series:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), title=None),
        tooltip=[alt.Tooltip(f"{x}:T", title="Week"), "Series:N", alt.Tooltip("value:Q", format=",.0f")])
    return style(ch, height)


# ---------------------------------------------------------------- sidebar
D = load()
preds, panel = D["preds"], D["panel"]
SETTINGS = K.settings_for(preds)

st.sidebar.title("Demand Forecasting Studio")
st.sidebar.caption("UK online gift retailer · 200 products · public data (UCI Online Retail II)")
page = st.sidebar.radio("Page", ["Executive summary", "Forecast accuracy", "Inventory simulation", "Sales projection",
                                 "Data and method"])
setting = st.sidebar.radio("Forecast setting", list(SETTINGS), format_func=SETTINGS.get,
                           help="Unbiased totals aims at the average week. Best per-product accuracy aims at the "
                                "typical week: closer per product, low on totals by design.")
with st.sidebar.expander("Filters"):
    abc = tuple(st.multiselect("Revenue class", ["A", "B", "C"], ["A", "B", "C"]) or ["A", "B", "C"])
    demand = tuple(st.multiselect("Demand type", ["regular", "intermittent"], ["regular", "intermittent"])
                   or ["regular", "intermittent"])
    horizons = tuple(st.multiselect("Weeks ahead", [1, 2, 3, 4], [1, 2, 3, 4]) or [1, 2, 3, 4])
st.sidebar.caption("Results are a rolling backtest on recorded history: forecasts made 1 to 4 weeks ahead, scored on "
                   "weeks never used to build or tune the model. Simulations are labelled as simulations.")

fin = K.final_rows(preds, abc, demand, horizons)
if fin.empty or fin["y"].sum() <= 0:
    st.warning("No product-weeks match these filters.")
    st.stop()
HALL = headlines(abc, demand, horizons)
H = HALL[setting]
other = next(k for k in SETTINGS if k != setting)
span = f"{fin['week'].min():%d %b %Y} to {fin['week'].max():%d %b %Y}"

# ---------------------------------------------------------------- pages
if page == "Executive summary":
    st.title("Executive summary")
    b_ok = abs(H["bias"]) <= K.BIAS_TARGET
    if b_ok:
        st.subheader(f"Out of sample, weekly totals are {pct(H['total_acc'])} accurate and the forecast level is "
                     f"within {abs(100 * H['bias']):.1f}% of the units sold.")
    else:
        st.subheader(f"Out of sample, weekly totals are {pct(H['total_acc'])} accurate; this setting runs "
                     f"{abs(100 * H['bias']):.0f}% {'low' if H['bias'] < 0 else 'high'} on totals because it aims at "
                     f"the typical week, and adds {pts(H['product_fva'])} per product.")
    st.caption(f"Benchmark ({K.BASE_LABEL}): {pct(H['total_acc_base'])} weekly-total accuracy, {spct(H['bias_base'])} bias. "
               f"Units, weekly, 1–4 weeks ahead, {H['products']} products, {H['rows']:,} out-of-sample forecasts, {span}.")

    c = st.columns(4)
    tone = {"better": "good", "worse": "bad", "level": "neutral"}
    metric(c[0], "Weekly-total accuracy", pct(H["total_acc"]), f"benchmark {pct(H['total_acc_base'])}",
           help="1 − Σ|forecast total − actual total| / Σ actual total, week by week. The order-quantity view.")
    metric(c[1], "Value added vs benchmark", pts(H["total_fva"]),
           VERDICT[H["total_verdict"]],
           tone[H["total_verdict"]],
           help=f"Weekly-total accuracy points over the {K.BASE_LABEL}. 95% range {pts(H['total_ci'][0])} to "
                f"{pts(H['total_ci'][1])} (resampling weeks). Level = the range includes zero.")
    metric(c[2], "Forecast bias", spct(H["bias"]), K.bias_tag(H["bias"]),
           "good" if b_ok else "neutral",
           help=f"Σ(forecast − actual) / Σ actual, target ±5%. Benchmark {spct(H['bias_base'])}.")
    metric(c[3], "Product-level accuracy", pct(H["product_acc"]),
           f"{pts(H['product_fva'])}, level" if H["product_verdict"] == "level" else f"{pts(H['product_fva'])} vs benchmark",
           tone[H["product_verdict"]],
           help=f"1 − WAPE per product-week, volume-weighted. 95% range of the gain {pts(H['product_ci'][0])} to "
                     f"{pts(H['product_ci'][1])} (resampling products). Single gift products sell in small, irregular "
                     f"amounts, so product-week error is high for every method.")

    st.caption(f"95% ranges of the value added, {K.BASE_LABEL} as benchmark: weekly total {pts(H['total_ci'][0])} to "
               f"{pts(H['total_ci'][1])} ({VERDICT[H['total_verdict']]}); per product {pts(H['product_ci'][0])} to "
               f"{pts(H['product_ci'][1])} ({VERDICT[H['product_verdict']]}). Benchmark bias {spct(H['bias_base'])}. "
               "Level means the range includes zero.")

    st.markdown("#### Two forecast settings, one trade-off")
    rows = [{"Setting": SETTINGS[k] + (" (shown now)" if k == setting else ""),
             "Weekly-total accuracy": pct(h["total_acc"]), "Forecast bias": spct(h["bias"]),
             "Product-level accuracy": pct(h["product_acc"]),
             "Per product vs benchmark": f"{pts(h['product_fva'])} ({h['product_verdict']})",
             "Use for": K.SETTING_USE.get(k, "")} for k, h in HALL.items()]
    rows.append({"Setting": f"{K.BASE_LABEL} (benchmark)", "Weekly-total accuracy": pct(H["total_acc_base"]),
                 "Forecast bias": spct(H["bias_base"]), "Product-level accuracy": pct(H["product_acc_base"]),
                 "Per product vs benchmark": "–", "Use for": "Comparison"})
    st.table(pd.DataFrame(rows).set_index("Setting"))
    st.caption("No setting wins both. " + " ".join(f"**{SETTINGS[k]}.** {K.SETTING_WHY[k]}" for k in SETTINGS
                                                   if k in K.SETTING_WHY))

    st.markdown("#### Forecast vs actual, weekly total")
    g = K.weekly(fin, [setting, K.BASE]).rename(columns={"y": "Actual units", setting: "Forecast", K.BASE: K.BASE_LABEL})
    st.altair_chart(line_chart(g, {"Actual units": GREY, "Forecast": BLUE, K.BASE_LABEL: ORANGE}), use_container_width=True)
    st.caption("Each forecast was made 1 to 4 weeks before the week it is compared with.")

elif page == "Forecast accuracy":
    st.title("Forecast accuracy")
    st.caption(f"Final test, {span}. Accuracy = 1 − WAPE per product-week unless stated. Setting: {SETTINGS[setting]}.")
    seg = []
    f2 = fin.assign(season=np.where(fin["week"].dt.month >= 9, "Sep to Nov (peak)", "Apr to Aug"))
    for col, names in (("abc", {"A": "Class A", "B": "Class B", "C": "Class C"}),
                       ("demand_type", {"regular": "Regular sellers", "intermittent": "Intermittent sellers"}),
                       ("season", {"Apr to Aug": "Apr to Aug", "Sep to Nov (peak)": "Sep to Nov (peak)"}),
                       ("h", {h: f"{h} wk ahead" for h in (1, 2, 3, 4)})):
        for key, name in names.items():
            d = f2[f2[col] == key]
            if len(d) and d["y"].sum() > 0:
                seg.append({"Segment": name, "Forecast": 1 - K.wape(d, setting), K.BASE_LABEL: 1 - K.wape(d, K.BASE),
                            "Value added (pts)": 100 * (K.wape(d, K.BASE) - K.wape(d, setting)),
                            "Bias": K.bias(d, setting)})
    sg = pd.DataFrame(seg)
    best, worst = sg.loc[sg["Value added (pts)"].idxmax()], sg.loc[sg["Value added (pts)"].idxmin()]
    st.subheader(f"Largest value added: {best['Segment']} ({best['Value added (pts)']:+.1f} pts). "
                 f"Smallest: {worst['Segment']} ({worst['Value added (pts)']:+.1f} pts).")
    bars = alt.Chart(sg).mark_bar().encode(
        y=alt.Y("Segment:N", sort=None, title=None),
        x=alt.X("Value added (pts):Q", title=f"Accuracy points vs {K.BASE_LABEL} (product-week)"),
        color=alt.condition(alt.datum["Value added (pts)"] >= 0, alt.value(GREEN), alt.value("#f2555a")),
        tooltip=["Segment", alt.Tooltip("Value added (pts):Q", format="+.1f")])
    st.altair_chart(style(bars, 34 * len(sg) + 40), use_container_width=True)
    show = sg.assign(**{"Forecast": sg["Forecast"].map(pct), K.BASE_LABEL: sg[K.BASE_LABEL].map(pct),
                        "Value added (pts)": sg["Value added (pts)"].map("{:+.1f}".format), "Bias": sg["Bias"].map(spct)})
    table(show)

    lab = D["lab"]
    if lab:
        st.markdown("#### Model lab: every model on the same rolling backtest")
        lb = pd.DataFrame(lab["leaderboard"]).sort_values("wape")
        table(pd.DataFrame({
            "Model": lb["label"], "Family": lb["family"],
            "Weekly-total accuracy": (1 - lb["total_wape"]).map(pct), "Bias": lb["bias"].map(spct),
            "Product-level accuracy": (1 - lb["wape"]).map(pct),
            "Error vs 13-wk avg (95% range)": [
                f"{100 * v:+.1f}% ({100 * ci[0]:+.0f} to {100 * ci[1]:+.0f})" for v, ci in zip(lb["vs_ma13"], lb["vs_ma13_ci95"])]}))
        names = {r["model"]: r["label"] for r in lab["leaderboard"]}
        st.caption(f"All product-weeks of the final test ({lab['final_rows']:,}), no filters. The first 5 forecast dates "
                   f"chose the ensemble members ({', '.join(names.get(m, m) for m in lab['ensemble_members'])}); "
                   f"the last 8 are the test shown here. Negative error change = less error than the benchmark.")

elif page == "Inventory simulation":
    st.title("Inventory simulation")
    st.caption("Simulation on recorded history, not measured savings. The same weekly order-up-to policy with lost "
               f"sales runs once with the forecast and once with the {K.BASE_LABEL}.")
    inv = D["inv"]
    if not inv:
        st.info("Run  python -m src.inventory  to fill this page.")
        st.stop()
    leads = sorted({r["lead"] for r in inv["runs"]})
    servs = sorted({r["service"] for r in inv["runs"]})
    c = st.columns(4)
    lead = c[0].radio("Lead time (weeks)", leads, horizontal=True)
    serv = c[1].radio("Service target", servs, index=servs.index(0.95) if 0.95 in servs else 0, horizontal=True,
                      format_func=lambda v: pct(v, 0))
    cost = c[2].slider("Unit cost, % of price", 30, 60, 45) / 100
    hold = c[3].slider("Holding rate per year, %", 15, 35, 25) / 100
    run = K.find_run(inv, setting, lead, serv)
    if run is None:
        st.info("This combination was not simulated for the selected setting.")
        st.stop()
    r = K.inventory_view(run, cost, hold)
    m, b, e = r["m"], r["b"], r["e"]
    ic = r["inv_change"]
    st.subheader(f"With the same stock policy, the forecast fills {pct(m['fill_rate'])} of demand vs {pct(b['fill_rate'])} "
                 f"with the {K.BASE_LABEL}, and needs {abs(100 * ic):.0f}% {'less' if ic < 0 else 'more'} stock for the "
                 f"same fill rate.")
    k = st.columns(5)
    dfill, dshort = m["fill_rate"] - b["fill_rate"], m["units_short"] - b["units_short"]
    metric(k[0], "Simulated fill rate", pct(m["fill_rate"]), f"{100 * dfill:+.1f} pts vs benchmark",
           "good" if dfill > 0 else ("bad" if dfill < 0 else "neutral"))
    metric(k[1], "Units short", f"{m['units_short']:,.0f}", f"{dshort:+,.0f} vs benchmark",
           "good" if dshort < 0 else ("bad" if dshort > 0 else "neutral"))
    metric(k[2], "Stock for the same fill rate", f"{100 * ic:+.1f}%", f"vs benchmark at {pct(b['fill_rate'])} fill",
           "good" if ic < 0 else "neutral")
    metric(k[3], "Carrying-cost saving", f"£{r['saving']:,.0f}/yr", f"£{r['saving_lo']:,.0f} to £{r['saving_hi']:,.0f} at 15–35%")
    metric(k[4], "Lost sales avoided", f"£{r['lost_rev']:,.0f}/yr", f"£{r['lost_margin']:,.0f} margin")
    st.caption(f"{inv['products']} products over the {inv['weeks_scored']} final-test weeks ({inv['final_start']} to "
               f"{inv['final_end']}). Safety stock = z × σ × √(L + R): z = {run['z']:.2f} for {pct(serv, 0)}, "
               f"L = {lead} wk lead time, R = 1 wk review, σ = spread of the last 13 weekly misses. Weeks of cover "
               f"{m['weeks_of_cover']:.1f} (benchmark {b['weeks_of_cover']:.1f}). Unit cost and holding rate are "
               f"assumptions you can change above.")
    comp = pd.DataFrame([{"Setting": SETTINGS[k_], **{
        "Fill rate": pct(v["m"]["fill_rate"]), "Benchmark fill rate": pct(v["b"]["fill_rate"]),
        "Stock for the same fill rate": f"{100 * v['inv_change']:+.1f}%",
        "Carrying-cost saving / yr": f"£{v['saving']:,.0f}", "Lost sales avoided / yr": f"£{v['lost_rev']:,.0f}"}}
        for k_ in SETTINGS if (rr := K.find_run(inv, k_, lead, serv)) for v in [K.inventory_view(rr, cost, hold)]])
    st.markdown("#### Both settings under these inputs")
    table(comp)

elif page == "Sales projection":
    st.title("Sales projection under economic scenarios")
    st.caption("What-if assumptions, not predictions of the economy. A SARIMAX model of weekly demand projects 26 weeks "
               "with prediction intervals; scenarios scale it by assumed sensitivities, phased in over 13 weeks.")
    proj, info = D["proj"], D["scen"]
    if proj is None or not info:
        st.info("Run  python -m src.scenarios  to fill this page.")
        st.stop()
    name = st.radio("Scenario", ["Recession", "Base", "Boom", "Custom"], index=1, horizontal=True)
    deltas, sens = dict(SC.SCENARIOS.get(name, SC.SCENARIOS["Base"])), dict(SC.ASSUMED_SENSITIVITY)
    price = 0.0
    if name == "Custom":
        cc = st.columns(4)
        for i, (key, label_, lo, hi, step) in enumerate(INDICATORS):
            deltas[key] = cc[i].slider(label_, lo, hi, 0.0, step)
        price = cc[3].slider("Own price change, %", -15, 15, 0) / 100
        with st.expander("Sensitivities (% of sales per unit change)"):
            sc_ = st.columns(3)
            for i, (key, label_, *_ ) in enumerate(INDICATORS):
                sens[key] = sc_[i].number_input(label_, -20.0, 20.0, 100 * SC.ASSUMED_SENSITIVITY[key], 0.5) / 100
    el = info["price"]["elasticity"]
    sel = SC.apply(proj, deltas, sens, price, el)
    named = {n: SC.apply(proj, SC.SCENARIOS[n], sens) for n in ("Recession", "Base", "Boom")}
    tot = {n: float(v["mean"].sum()) for n, v in named.items()}
    s_tot = float(sel["mean"].sum())
    k = st.columns(3)
    chg = s_tot / tot["Base"] - 1
    metric(k[0], "Base projection, next 26 weeks", f"{tot['Base']:,.0f} units",
           f"80% interval {proj['lo80'].sum():,.0f} to {proj['hi80'].sum():,.0f}")
    metric(k[1], f"Selected scenario: {name}", f"{s_tot:,.0f} units",
           "same as base" if abs(chg) < 5e-4 else f"{100 * chg:+.1f}% vs base",
           "neutral" if abs(chg) < 5e-4 else ("good" if chg > 0 else "bad"))
    metric(k[2], "Recession to boom", f"{100 * (tot['Recession'] / tot['Base'] - 1):+.0f}% to "
                                      f"{100 * (tot['Boom'] / tot['Base'] - 1):+.0f}%", "vs base, 26-week total")
    hist = panel.groupby("week")["units"].sum().sort_index().iloc[-52:].rename("value").reset_index()
    base_x = alt.X("week:T", title=None)
    band95 = alt.Chart(sel).mark_area(opacity=0.18, color=BLUE).encode(x=base_x, y=alt.Y("lo95:Q", title="Units per week"), y2="hi95:Q")
    band80 = alt.Chart(sel).mark_area(opacity=0.30, color=BLUE).encode(x=base_x, y="lo80:Q", y2="hi80:Q")
    mean = alt.Chart(sel).mark_line(color=BLUE, strokeWidth=2.5).encode(
        x=base_x, y="mean:Q", tooltip=[alt.Tooltip("week:T", title="Week"), alt.Tooltip("mean:Q", format=",.0f", title=name)])
    past = alt.Chart(hist).mark_line(color=GREY, strokeWidth=2).encode(
        x=base_x, y="value:Q", tooltip=[alt.Tooltip("week:T", title="Week"), alt.Tooltip("value:Q", format=",.0f", title="Actual")])
    layers = [band95, band80, past, mean]
    for n, col in (("Recession", ORANGE), ("Base", "#6e7887"), ("Boom", GREEN)):
        if n != name:
            layers.append(alt.Chart(named[n]).mark_line(color=col, strokeDash=[5, 4], strokeWidth=1.5).encode(
                x=base_x, y="mean:Q", tooltip=[alt.Tooltip("week:T", title="Week"), alt.Tooltip("mean:Q", format=",.0f", title=n)]))
    st.altair_chart(style(alt.layer(*layers), 380), use_container_width=True)
    st.caption("Grey: the last 52 weeks of actual sales. Blue: the selected scenario with 80% and 95% prediction intervals. "
               "Dashed: recession (orange), base (grey), boom (green).")
    me = info["macro_estimates"]
    st.markdown("#### Assumptions, visible")
    table(pd.DataFrame([{
        "Indicator": label_, "Change in this scenario": f"{deltas[key]:+.1f}",
        "Assumed effect per unit": f"{100 * sens[key]:+.1f}%",
        "Estimated from 2009–2011 (95% range)": f"{100 * me[key]['effect']:+.0f}% ({100 * me[key]['lo95']:+.0f} to {100 * me[key]['hi95']:+.0f})"}
        for key, label_, *_ in INDICATORS]))
    st.caption(f"With two years of data the estimated effects are wide and mixed up with the trend, so the projection uses "
               f"the stated assumptions. Price elasticity {el:.2f} (95% {info['price']['lo95']:.2f} to "
               f"{info['price']['hi95']:.2f}), estimated from the retailer's own price changes.")

else:
    st.title("Data and method")
    d, des = D["metrics"]["data"], D["metrics"]["design"]
    st.markdown(f"""
**Data.** Every transaction of a UK online gift retailer from {d['date_min']} to {d['date_max']} (public data, UCI
Online Retail II, CC BY 4.0). {d['clean_rows']:,} of {d['raw_rows']:,} rows were kept after removing cancellations,
duplicates and non-product lines. {d['skus_selected']} products were selected from the first year only; they made up
{pct(d['selected_share_of_first_year_revenue'], 0)} of first-year revenue.

**Test.** A rolling backtest with {des['folds_total']} forecast dates, 4 weeks apart. At each date the models see only
the history up to that date and forecast 1 to {des['horizon_weeks']} weeks ahead. The first {des['folds_selection']}
dates were used for choices; the last {des['folds_final']} ({des['final_rows_scored']:,} product-weeks,
{des['final_first_week']} to {des['final_last_week']}) are the final test every page reports. Automated tests change
the future and confirm that no model input moves.

**Two settings.** "Unbiased totals" (gradient boosting with a Tweedie loss) aims at the average week, so forecasts add
up to the units sold. "Best per-product accuracy" (an ensemble of the three models with the lowest error on the
selection dates) aims at the typical week: closer per product, low on totals by design. Both were fixed before the
final test was scored.

**Limits.** Two years of data, one earlier Christmas, no stock levels and no promotion calendar. The inventory page is
a simulation and the scenarios are what-if assumptions. Results will differ on other data.

**Desktop app.** The full PyQt6 app adds a live replay of recorded sales, a product explorer and risk views.
""")
