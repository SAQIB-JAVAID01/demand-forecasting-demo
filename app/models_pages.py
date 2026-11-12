"""Model lab and Sales projection pages for Demand Forecasting Studio.

Model lab: every forecasting model on the same rolling backtest (reports/model_lab.json, made by
`python -m src.model_lab`), the engineered features and what drives the forecast (TreeSHAP from LightGBM).

Sales projection: the next 26 weeks of total demand with 80% / 95% prediction intervals (data/processed/
projection.parquet, made by `python -m src.scenarios`), scaled by what-if economic scenarios. Sensitivities are
editable assumptions; the estimates from the real UK indicators are shown beside them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QButtonGroup, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                             QPushButton, QSlider, QTableWidgetItem, QVBoxLayout, QWidget)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src import scenarios as SC  # noqa: E402
from src.report import FEATURE_NAMES  # noqa: E402

EXTRA_NAMES = {
    "ewm_8": "Smoothed recent sales (EWM)", "cv_13": "13-week variability (CV)", "slope_13": "13-week slope",
    "adi_26": "Average gap between sales (ADI)", "cv2_26": "Size variability of sales (CV²)",
    "price_vs_52": "Price vs 52-week average", "discount_13": "Discount from 13-week high",
    "abc_code": "Revenue class (A/B/C)", "intermittent": "Intermittent seller", "sb_class": "Demand pattern class",
    "sin_1": "Season wave 1 (sin)", "cos_1": "Season wave 1 (cos)", "sin_2": "Season wave 2 (sin)",
    "cos_2": "Season wave 2 (cos)", "sin_3": "Season wave 3 (sin)", "cos_3": "Season wave 3 (cos)",
    "bank_holidays": "UK bank holidays in week", "bank_holidays_next": "Bank holidays next week",
    "xmas_peak": "Christmas peak (6 weeks)", "macro_confidence": "UK consumer confidence",
    "macro_unemployment": "UK unemployment rate", "macro_inflation": "UK inflation (CPI)",
}
NAMES = {**FEATURE_NAMES, **EXTRA_NAMES}
FAMILY_COLOR = {"Baseline": "neutral", "Statistical": "muted", "Machine learning": "soft", "Ensemble": "s1"}
INDICATORS = [("confidence", "Consumer confidence", "index pts", -6.0, 6.0, 0.5),
              ("unemployment", "Unemployment rate", "pts", -2.0, 3.0, 0.1),
              ("inflation", "Inflation (CPI, yearly)", "pts", -2.0, 4.0, 0.1)]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def tile(lay_row, title: str, label_fn):
    f = QFrame()
    f.setObjectName("card")
    l = QVBoxLayout(f)
    l.setContentsMargins(20, 16, 20, 16)
    l.setSpacing(4)
    val, chip, note = label_fn("", "kv"), label_fn(""), label_fn("", "kn", True)
    l.addWidget(label_fn(title, "kl"))
    l.addWidget(val)
    l.addWidget(chip, 0, Qt.AlignmentFlag.AlignLeft)
    l.addWidget(note)
    l.addStretch(1)
    lay_row.addWidget(f)
    return val, chip, note


# ===================================================================== model lab
class ModelLab:
    def __init__(self, win, lay, Card, label, chip_css, make_table):
        self.win, self.label, self.chip_css = win, label, chip_css
        path = C.REPORTS_DIR / "model_lab.json"
        self.lab = json.loads(path.read_text()) if path.exists() else None
        row = QHBoxLayout()
        row.setSpacing(16)
        self.k = [tile(row, t_, label) for t_ in ("Models compared", "Chosen on selection folds",
                                                  "Out-of-sample accuracy", "Engineered features")]
        lay.addLayout(row)
        self.c_board, self.c_groups = Card("Leaderboard · final test", 400), Card("Feature groups", 400)
        win._row(lay, self.c_board, self.c_groups, stretch=(3, 2), weight=1)
        self.c_tab = Card("All models")
        self.table = make_table(500)
        self.c_tab.lay.insertWidget(2, self.table, 1)
        lay.addWidget(self.c_tab, 1)
        self.c_shap, self.c_feat = Card("What drives the forecast · TreeSHAP", 420), Card("Feature engineering", text=True)
        win._row(lay, self.c_shap, self.c_feat, stretch=(3, 2), weight=1)
        self.c_how = Card("How the models were tested", text=True)
        lay.addWidget(self.c_how)

    def refresh(self, t: dict):
        lab = self.lab
        if lab is None:
            self.c_board.say("Run  python -m src.model_lab  to fill this page.")
            return
        b = pd.DataFrame(lab["leaderboard"])
        best = lab["best_on_selection"]
        r = b.set_index("model").loc[best]
        base = b.set_index("model").loc["ma13"]
        new = sum(1 for f in b["family"] if f in ("Statistical", "Ensemble")) + 3
        vals = [
            (str(len(b)), f"{len(b) - 8} added in the model lab", t["s1"],
             "Baselines, statistical models, gradient boosting and an ensemble, on the same 13 forecast dates."),
            (r["label"].split(" (")[0], "Lowest error on the selection folds", t["good"],
             f"Members: {', '.join(self._lbl(m) for m in lab['ensemble_members'])}." if best == "ensemble" else ""),
            (pct(1 - r["wape"]), f"+{100 * (base['wape'] - r['wape']):.1f} pts vs 13-week average", t["good"],
             f"1 − WAPE, product-week, final test. 95% range of the error change vs the 13-week average: "
             f"{100 * r['vs_ma13_ci95'][0]:+.0f}% to {100 * r['vs_ma13_ci95'][1]:+.0f}%."),
            (str(lab["n_features_ext"]), f"from {lab['n_features_base']} in the first model", t["s1"],
             "Lags, rolling statistics, intermittency, Fourier seasonality, UK bank holidays, price and promotion "
             "proxies, product class and UK economic indicators."),
        ]
        for (val, chip, note), (v, c, col, n) in zip(self.k, vals):
            val.setText(v)
            chip.setText(c)
            chip.setStyleSheet(self.chip_css(col))
            note.setText(n)
        val0 = self.k[1][0]
        val0.setStyleSheet("font-size: 19pt; font-weight: 600;")

        bs = b.sort_values("wape")
        self.c_board.say(f"{r['label']} has the highest out-of-sample accuracy: {pct(1 - r['wape'])} per product-week.",
                         "Accuracy = 1 − WAPE over every product-week of the final test, which no choice looked at. "
                         "Blue = chosen on the selection folds; dark blue = machine learning; grey = statistical "
                         "models; dark grey = simple baselines.")
        self.c_board.chart.hbars(t, [x for x in bs["label"]], list(1 - bs["wape"]),
                                 ["s1" if m == best else FAMILY_COLOR[f] for m, f in zip(bs["model"], bs["family"])])
        self.c_tab.say("Error, bias and weekly-total accuracy for every model",
                       "Selection error chose the ensemble members; every other column is the final test. "
                       "Change vs 13-week average: negative = less error, with a 95% bootstrap range over products.")
        cols = ["Model", "Family", "Selection WAPE", "Accuracy", "Weekly total acc.", "Bias", "A-item acc.", "vs 13-wk avg"]
        tb = self.table
        tb.setSortingEnabled(False)
        tb.setRowCount(len(bs))
        tb.setColumnCount(len(cols))
        tb.setHorizontalHeaderLabels(cols)
        for i, row in enumerate(bs.itertuples()):
            cells = [row.label, row.family, pct(row.selection_wape), pct(1 - row.wape), pct(1 - row.total_wape),
                     f"{100 * row.bias:+.1f}%", pct(1 - row.class_a_wape),
                     f"{100 * row.vs_ma13:+.1f}% ({100 * row.vs_ma13_ci95[0]:+.0f} to {100 * row.vs_ma13_ci95[1]:+.0f})"]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                it.setTextAlignment((Qt.AlignmentFlag.AlignLeft if c < 2 else Qt.AlignmentFlag.AlignRight)
                                    | Qt.AlignmentFlag.AlignVCenter)
                if c == 7:
                    from PyQt6.QtGui import QColor
                    it.setForeground(QColor(t["good"] if row.vs_ma13 < 0 else t["muted"] if row.vs_ma13 == 0 else t["bad"]))
                tb.setItem(i, c, it)
        tb.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        tb.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

        imp = pd.DataFrame(lab["feature_importance"]).sort_values("shap", ascending=False)
        top = imp.head(14)
        self.c_shap.say(f"{NAMES.get(top.iloc[0]['feature'], top.iloc[0]['feature'])} and "
                        f"{NAMES.get(top.iloc[1]['feature'], top.iloc[1]['feature'])} carry the most signal.",
                        "Mean absolute SHAP value per feature (TreeSHAP, computed by LightGBM) on the final-test "
                        "forecasts of the engineered-feature model, as a share of the total.")
        self.c_shap.chart.hbars(t, [NAMES.get(f, f) for f in top["feature"]], list(top["shap"]), ["s1"] * len(top),
                                fmt=lambda v: pct(v, 1))
        gs = imp.groupby("group")["shap"].sum().sort_values(ascending=False)
        self.c_groups.say(f"{gs.index[0]} explains {pct(gs.iloc[0], 0)} of the forecast.",
                          "Share of the SHAP total by feature group. Economy features are lagged to the date each "
                          "figure was published.")
        self.c_groups.chart.hbars(t, list(gs.index), list(gs.values), ["s1"] + ["soft"] * (len(gs) - 1),
                                  fmt=lambda v: pct(v, 0))
        grp = lab["feature_groups"]
        self.c_feat.say(f"{lab['n_features_ext']} features, all known at the forecast date")
        self.c_feat.body.setText("".join(
            f"<b>{g}.</b> {', '.join(NAMES.get(f, f) for f in fs)}.<br><br>" for g, fs in grp.items()))
        self.c_how.say("Same dates, same rules, no peeking")
        self.c_how.body.setText(
            "<b>Rolling backtest.</b> 13 forecast dates, 4 weeks apart; at each date every model sees only the "
            "history up to that date and forecasts 1 to 4 weeks ahead.<br><br>"
            "<b>Selection vs final test.</b> The first 5 dates chose the ensemble members (the 3 models with the "
            f"lowest error there); the last 8 dates ({lab['final_rows']:,} product-weeks) are the out-of-sample "
            "test reported here.<br><br>"
            "<b>Statistical models.</b> Holt damped trend and TSB (a Croston method for intermittent demand) per "
            "product; SARIMAX on the weekly store total, split by each product's recent share, because single "
            "products have too little history for a seasonal model.<br><br>"
            "<b>Machine learning.</b> LightGBM, XGBoost and scikit-learn HistGradientBoosting on the engineered "
            "features. Automated tests change the future and check that no feature moves.")

    def _lbl(self, m):
        b = {r["model"]: r["label"] for r in self.lab["leaderboard"]}
        return b.get(m, m)


# ===================================================================== projection
class Projection:
    def __init__(self, win, lay, Card, label, chip_css):
        self.win, self.label, self.chip_css = win, label, chip_css
        ppath, jpath = C.PROCESSED_DIR / "projection.parquet", C.REPORTS_DIR / "scenarios.json"
        self.ok = ppath.exists() and jpath.exists()
        if self.ok:
            self.proj = pd.read_parquet(ppath)
            self.info = json.loads(jpath.read_text())
        self.scenario = "Base"
        self._busy = False

        ctl = QFrame()
        ctl.setObjectName("hero")
        cl = QVBoxLayout(ctl)
        cl.setContentsMargins(24, 18, 24, 18)
        cl.setSpacing(12)
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(label("SCENARIO", "flabel"))
        self.b_sc, grp = {}, QButtonGroup(win)
        for name in ("Recession", "Base", "Boom", "Custom"):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setChecked(name == "Base")
            b.clicked.connect(lambda _=False, n=name: self.pick(n))
            grp.addButton(b)
            top.addWidget(b)
            self.b_sc[name] = b
        top.addStretch(1)
        self.l_note = label("", "hero2")
        top.addWidget(self.l_note)
        cl.addLayout(top)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        self.sl, self.sl_val, self.sens = {}, {}, {}
        for r, (key, name, unit, lo, hi, step) in enumerate(INDICATORS):
            grid.addWidget(label(name, "kl"), r, 0)
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(int(round(lo / step)), int(round(hi / step)))
            s.setProperty("step", step)
            s.valueChanged.connect(self._moved)
            grid.addWidget(s, r, 1)
            v = label("", "kn")
            v.setMinimumWidth(130)
            grid.addWidget(v, r, 2)
            grid.addWidget(label("Sensitivity, % sales per unit", "kn"), r, 3)
            sp = QDoubleSpinBox()
            sp.setRange(-20, 20)
            sp.setSingleStep(0.5)
            sp.setDecimals(1)
            sp.setSuffix(" %")
            sp.setValue(100 * SC.ASSUMED_SENSITIVITY[key])
            sp.valueChanged.connect(self._schedule)
            grid.addWidget(sp, r, 4)
            self.sl[key], self.sl_val[key], self.sens[key] = s, v, sp
        r = len(INDICATORS)
        grid.addWidget(label("Own price change", "kl"), r, 0)
        self.sl_price = QSlider(Qt.Orientation.Horizontal)
        self.sl_price.setRange(-15, 15)
        self.sl_price.valueChanged.connect(self._moved)
        grid.addWidget(self.sl_price, r, 1)
        self.l_price = label("", "kn")
        grid.addWidget(self.l_price, r, 2)
        self.l_elast = label("", "kn")
        grid.addWidget(self.l_elast, r, 3, 1, 2)
        grid.setColumnStretch(1, 1)
        cl.addLayout(grid)
        lay.addWidget(ctl)

        row = QHBoxLayout()
        row.setSpacing(16)
        self.k = [tile(row, t_, label) for t_ in ("Projected units, next 26 weeks", "Selected scenario vs base",
                                                  "Range across scenarios", "Resilience")]
        lay.addLayout(row)
        self.c_fan = Card("Sales projection · 80% and 95% prediction intervals", 380)
        lay.addWidget(self.c_fan, 1)
        self.c_seg, self.c_ass = Card("By revenue class (ABC segmentation)", 260), Card("Assumptions, visible", text=True)
        win._row(lay, self.c_seg, self.c_ass, stretch=(1, 1), weight=1)
        self.timer = QTimer(win)
        self.timer.setSingleShot(True)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self.draw)
        self.pick("Base")

    def deltas(self) -> dict:
        return {k: s.value() * s.property("step") for k, s in self.sl.items()}

    def sens_now(self) -> dict:
        return {k: sp.value() / 100 for k, sp in self.sens.items()}

    def pick(self, name: str):
        self.scenario = name
        self.b_sc[name].setChecked(True)
        if name != "Custom":
            self._busy = True
            for k, s in self.sl.items():
                s.setValue(int(round(SC.SCENARIOS[name][k] / s.property("step"))))
            self.sl_price.setValue(0)
            self._busy = False
        self._schedule()

    def _moved(self):
        if not self._busy and self.scenario != "Custom":
            self.scenario = "Custom"
            self.b_sc["Custom"].setChecked(True)
        self._schedule()

    def _schedule(self):
        self.timer.start()

    def draw(self):
        t = self.win.themes[self.win.theme]
        if not self.ok:
            self.c_fan.say("Run  python -m src.scenarios  to fill this page.")
            return
        info, proj = self.info, self.proj
        sens, d = self.sens_now(), self.deltas()
        el = info["price"]["elasticity"]
        pc = self.sl_price.value() / 100
        latest = info["macro_estimates"]["latest"]
        for k, s in self.sl.items():
            unit = next(u for k_, _, u, *_ in INDICATORS if k_ == k)
            self.sl_val[k].setText(f"{d[k]:+.1f} {unit}  (now {latest[k]:.1f})")
        self.l_price.setText(f"{100 * pc:+.0f}%")
        self.l_elast.setText(f"Price elasticity {el:.2f} (estimated, 95% {info['price']['lo95']:.2f} to "
                             f"{info['price']['hi95']:.2f})")
        sel = SC.apply(proj, d, sens, pc, el)
        named = {n: SC.apply(proj, SC.SCENARIOS[n], sens) for n in ("Recession", "Base", "Boom")}
        base = named["Base"]
        tot = {n: float(v["mean"].sum()) for n, v in named.items()}
        s_tot = float(sel["mean"].sum())
        price = info["segments"]["price"]
        inside = int(((sel["mean"] >= base["lo80"]) & (sel["mean"] <= base["hi80"]) | proj["closed"]).sum())
        rec_inside = int(((named["Recession"]["mean"] >= base["lo80"]) | proj["closed"]).sum())
        n = len(proj)
        chg = s_tot / tot["Base"] - 1
        vals = [
            (f"{tot['Base']:,.0f}", f"≈ £{tot['Base'] * price / 1e6:.2f}M at recent prices", t["s1"],
             f"Base scenario, {proj['week'].iloc[0]:%d %b %Y} to {proj['week'].iloc[-1]:%d %b %Y}. "
             f"80% interval {proj['lo80'].sum():,.0f} to {proj['hi80'].sum():,.0f} (weekly bounds summed)."),
            (f"{100 * chg:+.1f}%", f"{self.scenario}: {s_tot:,.0f} units",
             t["good"] if chg >= 0 else (t["warn"] if chg > -0.10 else t["bad"]), "Change against the base projection."),
            (f"{100 * (tot['Recession'] / tot['Base'] - 1):+.0f}% to {100 * (tot['Boom'] / tot['Base'] - 1):+.0f}%",
             "Recession to boom", t["s1"], "Total over 26 weeks, against the base projection."),
            (f"{inside} of {n}", "weeks inside the base 80% interval", t["good"] if inside >= 0.8 * n else t["warn"],
             f"Weeks where the selected scenario stays inside the base projection's 80% interval. In the "
             f"recession scenario, {rec_inside} of {n} weeks stay above its lower bound."),
        ]
        for (val, chip, note), (v, c, col, nt) in zip(self.k, vals):
            val.setText(v)
            chip.setText(c)
            chip.setStyleSheet(self.chip_css(col))
            note.setText(nt)
        self.l_note.setText("What-if assumptions, not predictions of the economy")

        hist = self.win.panel.groupby("week")["units"].sum().sort_index().iloc[-52:]
        lines = [(n_, named[n_]["mean"].to_numpy(), ck, True)
                 for n_, ck in (("Recession", "s2"), ("Base", "muted"), ("Boom", "good")) if n_ != self.scenario]
        self.c_fan.say(f"{self.scenario}: {s_tot:,.0f} units over the next 26 weeks ({100 * chg:+.1f}% vs base).",
                       "Blue line and bands: the selected scenario with 80% and 95% prediction intervals from a "
                       "SARIMAX model of weekly demand (Fourier seasonality, trading days). Dashed: the other scenarios. "
                       "Changes in the indicators phase in over 13 weeks.")
        self.c_fan.chart.fan(t, hist.index, hist.to_numpy(float), sel["week"], sel["mean"].to_numpy(),
                             sel["lo80"].to_numpy(), sel["hi80"].to_numpy(), sel["lo95"].to_numpy(), sel["hi95"].to_numpy(),
                             lines=lines, name=f"{self.scenario} projection")
        sh = info["segments"]["units"]
        keys = sorted(sh)
        self.c_seg.say(f"Class A carries {pct(sh.get('A', 0), 0)} of projected units.",
                       "Recent 26-week share of units per revenue class applied to the projection. Blue = selected "
                       "scenario, orange = base.")
        self.c_seg.chart.grouped(t, [f"Class {k}" for k in keys], [s_tot * sh[k] for k in keys],
                                 [tot["Base"] * sh[k] for k in keys], f"Selected: {self.scenario}", "Base",
                                 fmt=lambda v: f"{v / 1000:,.0f}k")
        me = info["macro_estimates"]
        rows = "".join(
            f"<tr><td>{name}</td><td align='right'>{100 * sens[k]:+.1f}%</td>"
            f"<td align='right'>{100 * me[k]['effect']:+.0f}% ({100 * me[k]['lo95']:+.0f} to {100 * me[k]['hi95']:+.0f})</td></tr>"
            for k, name, *_ in INDICATORS)
        self.c_ass.say("Sensitivities are assumptions; the data alone cannot pin them down")
        self.c_ass.body.setText(
            f"<table cellspacing='0' cellpadding='4' width='100%'><tr><td><b>Indicator</b></td>"
            f"<td align='right'><b>Assumed, per unit</b></td><td align='right'><b>Estimated from 2009–2011 (95%)</b></td>"
            f"</tr>{rows}</table><br>"
            f"Estimates come from the same weekly model with the real UK series (OECD via FRED, published values "
            f"only, latest {me['latest_month'][:7]}). With two years of data the ranges are wide and mixed up with the 2010–11 trend, so the "
            f"projection uses the editable assumptions. The price effect uses the elasticity estimated from this "
            f"retailer's own price changes ({el:.2f}).")
