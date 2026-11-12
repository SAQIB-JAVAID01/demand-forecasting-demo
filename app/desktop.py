"""Demand forecasting desktop app (PyQt6).

Run:  python app/desktop.py
Reads the backtest output in data/processed and reports/metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                             QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

THEMES = {
    "light": dict(page="#f4f4f1", surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", border="#e3e2dc", s1="#2a78d6", s2="#eb6834",
                  actual="#52514e", band="#cde2fb", neutral="#c3c2b7", chip="#ecebe6", side="#ffffff",
                  good="#006300", bad="#b3261e", sel="#e4eefb"),
    "dark": dict(page="#0d0d0d", surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", border="#2c2c2a", s1="#3987e5", s2="#d95926",
                 actual="#c3c2b7", band="#104281", neutral="#52514e", chip="#262624", side="#141413",
                 good="#0ca30c", bad="#e66767", sel="#17304f"),
}


def qss(t: dict) -> str:
    return f"""
    QWidget {{ background: {t['page']}; color: {t['ink']}; font-family: 'Segoe UI'; font-size: 10.5pt; }}
    #side {{ background: {t['side']}; border-right: 1px solid {t['border']}; }}
    #side QWidget {{ background: {t['side']}; }}
    QFrame#card {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 12px; }}
    QFrame#card QWidget {{ background: {t['surface']}; }}
    QLabel#h1 {{ font-size: 17pt; font-weight: 700; }}
    QLabel#brand {{ font-size: 13pt; font-weight: 700; }}
    QLabel#sub, QLabel#cap {{ color: {t['ink2']}; }}
    QLabel#section {{ color: {t['muted']}; font-size: 8.5pt; font-weight: 600; letter-spacing: 1px; }}
    QLabel#kv {{ font-size: 22pt; font-weight: 700; }}
    QLabel#kl {{ color: {t['ink2']}; }}
    QLabel#kn {{ color: {t['muted']}; font-size: 9pt; }}
    QLabel#h2 {{ font-size: 12pt; font-weight: 700; }}
    QPushButton {{ background: {t['chip']}; border: none; border-radius: 8px; padding: 6px 14px; }}
    QPushButton:hover {{ background: {t['grid']}; }}
    QPushButton:checked {{ background: {t['ink']}; color: {t['surface']}; }}
    #side QPushButton, QFrame#card QPushButton {{ background: {t['chip']}; color: {t['ink']}; }}
    #side QPushButton:hover, QFrame#card QPushButton:hover {{ background: {t['grid']}; }}
    #side QPushButton:checked {{ background: {t['ink']}; color: {t['side']}; }}
    QFrame#card QPushButton#primary {{ background: {t['s1']}; color: #ffffff; font-weight: 600; padding: 7px 16px; }}
    QLineEdit {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 8px; padding: 6px 9px; }}
    QListWidget {{ border: 1px solid {t['border']}; border-radius: 8px; outline: none; }}
    QListWidget::item {{ padding: 6px 8px; border-radius: 6px; }}
    QListWidget::item:selected {{ background: {t['sel']}; color: {t['ink']}; }}
    QTabWidget::pane {{ border: none; }}
    QTabBar::tab {{ background: transparent; color: {t['ink2']}; padding: 8px 16px; margin-right: 4px;
                    border-bottom: 2px solid transparent; font-weight: 600; }}
    QTabBar::tab:selected {{ color: {t['ink']}; border-bottom: 2px solid {t['s1']}; }}
    QTableWidget {{ background: {t['surface']}; border: none; gridline-color: {t['grid']};
                    alternate-background-color: {t['page']}; }}
    QHeaderView::section {{ background: {t['surface']}; color: {t['ink2']}; border: none;
                            border-bottom: 1px solid {t['grid']}; padding: 6px; font-weight: 600; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; }}
    QScrollBar::handle:vertical {{ background: {t['axis']}; border-radius: 5px; min-height: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    """


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def spct(x: float, d: int = 0) -> str:
    return f"{100 * x:+.{d}f}%"


def card(layout_cls=QVBoxLayout) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("card")
    lay = layout_cls(f)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(4)
    return f, lay


def label(text: str = "", name: str = "", wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


class Chart(FigureCanvasQTAgg):
    """Matplotlib canvas with a hover crosshair and tooltip."""

    def __init__(self, height: int = 300):
        self.fig = Figure(figsize=(6, 3), dpi=100)
        super().__init__(self.fig)
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.t = THEMES["light"]
        self._hover = None
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("axes_leave_event", self._on_leave)
        self.mpl_connect("resize_event", self._on_resize)

    def _on_resize(self, _):
        try:
            self.fig.tight_layout(pad=0.8)
        except Exception:
            pass

    def _base(self):
        t, ax = self.t, self.ax
        ax.clear()
        self.fig.set_facecolor(t["surface"])
        ax.set_facecolor(t["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["axis"])
        ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=9, length=3)
        ax.grid(axis="y", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)

    def lines(self, t: dict, x, series: list[dict], band=None, mark=None, extra=None):
        """series: [{name, color, values, width}]; band: (lo, hi); mark: x position of a dashed line."""
        self.t = t
        self._base()
        ax = self.ax
        x = pd.DatetimeIndex(x)
        if band is not None:
            ax.fill_between(x, band[0], band[1], color=t["band"], linewidth=0, label="80% interval")
        for s in series:
            ax.plot(x, s["values"], color=t[s["color"]], linewidth=s.get("width", 2), label=s["name"],
                    solid_capstyle="round")
        if mark is not None:
            ax.axvline(mark, color=t["axis"], linewidth=1, linestyle="--")
        ax.set_ylim(0, None)
        ax.margins(x=0.01)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if ax.get_ylim()[1] >= 10000 else f"{v:,.0f}")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=4, frameon=False, fontsize=9,
                        handlelength=1.6, borderaxespad=0.2)
        for txt in leg.get_texts():
            txt.set_color(t["ink2"])
        self.fig.tight_layout(pad=0.8)
        self._hover = dict(x=x, xnum=mdates.date2num(x.to_pydatetime()), series=series, band=band, extra=extra,
                           line=ax.axvline(x[0], color=t["muted"], linewidth=1, visible=False),
                           note=ax.annotate("", xy=(0, 0), xytext=(12, -12), textcoords="offset points",
                                            va="top", fontsize=9, color=t["ink"], visible=False,
                                            bbox=dict(boxstyle="round,pad=0.5", fc=t["surface"], ec=t["axis"], lw=1)))
        self.draw_idle()

    def bars(self, t: dict, labels: list[str], values: list[float], highlight: list[bool]):
        self.t = t
        self._hover = None
        self._base()
        ax = self.ax
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=t["grid"], linewidth=0.8)
        y = np.arange(len(labels))[::-1]
        ax.barh(y, values, height=0.6, color=[t["s1"] if h else t["neutral"] for h in highlight])
        for yi, v, h in zip(y, values, highlight):
            ax.text(v + max(values) * 0.012, yi, pct(v), va="center", fontsize=9, color=t["ink"],
                    fontweight="bold" if h else "normal")
        ax.set_yticks(y, labels)
        ax.set_xlim(0, max(values) * 1.16)
        ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        self.fig.tight_layout(pad=0.8)
        self.draw_idle()

    def _on_move(self, ev):
        h = self._hover
        if not h or ev.inaxes is not self.ax or ev.xdata is None:
            return
        i = int(np.abs(h["xnum"] - ev.xdata).argmin())
        rows = [f"Week of {h['x'][i]:%d %b %Y}"]
        for s in h["series"]:
            v = s["values"][i]
            if v is not None and not np.isnan(v):
                rows.append(f"{s['name']}: {v:,.0f}")
        if h["band"] is not None and not np.isnan(h["band"][1][i]):
            rows.append(f"80% interval: {h['band'][0][i]:,.0f} to {h['band'][1][i]:,.0f}")
        if h["extra"]:
            e = h["extra"](i)
            if e:
                rows.append(e)
        h["line"].set_xdata([h["x"][i], h["x"][i]])
        h["line"].set_visible(True)
        right = i > len(h["x"]) * 0.62
        h["note"].set_text("\n".join(rows))
        h["note"].xy = (h["xnum"][i], self.ax.get_ylim()[1])
        h["note"].set_ha("right" if right else "left")
        h["note"].xyann = (-12 if right else 12, -12)
        h["note"].set_visible(True)
        self.draw_idle()

    def _on_leave(self, _):
        if self._hover:
            self._hover["line"].set_visible(False)
            self._hover["note"].set_visible(False)
            self.draw_idle()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demand Forecasting: UK online retailer")
        self.resize(1360, 860)
        self.theme = "light"
        self.preds = pd.read_parquet(C.PROCESSED_DIR / "predictions.parquet")
        self.panel = pd.read_parquet(C.PROCESSED_DIR / "panel.parquet")
        self.m = json.loads((C.REPORTS_DIR / "metrics.json").read_text())
        self.model, self.base = self.m["chosen_model"], self.m["best_baseline"]
        self.final_start = self.preds.loc[self.preds["fold_type"] == "final", "week"].min()
        self.sku = None
        self._build()
        self.apply_theme()

    # ---------- layout ----------
    def _chips(self, lay, title: str, options: list[tuple], exclusive=False) -> list[QPushButton]:
        lay.addWidget(label(title, "section"))
        row = QHBoxLayout()
        row.setSpacing(6)
        buttons = []
        for value, text in options:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setChecked(True)
            b.setProperty("value", value)
            b.clicked.connect(self.refresh)
            row.addWidget(b)
            buttons.append(b)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addSpacing(8)
        return buttons

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # sidebar
        side = QFrame()
        side.setObjectName("side")
        side.setFixedWidth(310)
        s = QVBoxLayout(side)
        s.setContentsMargins(18, 18, 18, 18)
        s.setSpacing(6)
        s.addWidget(label("Demand Forecasting", "brand"))
        s.addWidget(label("UK online retailer · 200 products", "cap"))
        s.addSpacing(14)
        self.b_abc = self._chips(s, "REVENUE CLASS", [("A", "A"), ("B", "B"), ("C", "C")])
        self.b_type = self._chips(s, "DEMAND PATTERN", [("regular", "Regular"), ("intermittent", "Intermittent")])
        self.b_h = self._chips(s, "WEEKS AHEAD", [(1, "1"), (2, "2"), (3, "3"), (4, "4")])
        s.addWidget(label("PRODUCT", "section"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search product or code")
        self.search.textChanged.connect(self.fill_products)
        s.addWidget(self.search)
        self.plist = QListWidget()
        self.plist.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.plist.setWordWrap(True)
        self.plist.currentItemChanged.connect(self.on_product)
        s.addWidget(self.plist, 1)
        self.b_theme = QPushButton("Dark mode")
        self.b_theme.clicked.connect(self.toggle_theme)
        s.addWidget(self.b_theme)
        outer.addWidget(side)

        # main
        main = QWidget()
        v = QVBoxLayout(main)
        v.setContentsMargins(22, 18, 22, 18)
        v.setSpacing(12)
        d = self.m["design"]
        v.addWidget(label("Weekly demand forecast, 1 to 4 weeks ahead", "h1"))
        v.addWidget(label(f"Tested on {d['folds_final']} forecast dates ({d['final_first_week']} to {d['final_last_week']}) "
                          "that were never used to choose the model. Every forecast was made before the week it covers.", "sub", True))

        kp = QHBoxLayout()
        kp.setSpacing(12)
        self.k = []
        for title in ("Model forecast error (WAPE)", "Error vs baseline", "Bias (forecast vs actual units)",
                      "Actuals inside the 80% interval"):
            f, lay = card()
            val, lab, note = label("", "kv"), label(title, "kl"), label("", "kn")
            for w in (val, lab, note):
                lay.addWidget(w)
            kp.addWidget(f)
            self.k.append((val, lab, note))
        v.addLayout(kp)

        self.tabs = QTabWidget()
        v.addWidget(self.tabs, 1)

        # overview tab
        ov = QWidget()
        g = QGridLayout(ov)
        g.setContentsMargins(0, 10, 0, 0)
        g.setSpacing(12)
        f1, l1 = card()
        l1.addWidget(label("Total weekly demand", "h2"))
        l1.addWidget(label("Final test period, selected products. Hover for values.", "cap"))
        self.c_total = Chart(230)
        l1.addWidget(self.c_total, 1)
        f2, l2 = card()
        l2.addWidget(label("Forecast error by method", "h2"))
        l2.addWidget(label("All 200 products, final test. Lower is better. Blue = chosen on earlier folds.", "cap", True))
        self.c_methods = Chart(230)
        l2.addWidget(self.c_methods, 1)
        g.addWidget(f1, 0, 0)
        g.addWidget(f2, 0, 1)
        g.setColumnStretch(0, 3)
        g.setColumnStretch(1, 2)
        f3, l3 = card()
        l3.addWidget(label("How to read this", "h2"))
        self.notes = label("", "cap", True)
        self.notes.setTextFormat(Qt.TextFormat.RichText)
        l3.addWidget(self.notes)
        g.addWidget(f3, 1, 0, 1, 2)
        g.setRowStretch(0, 1)
        self.tabs.addTab(ov, "Overview")

        # product tab
        pr = QWidget()
        pv = QVBoxLayout(pr)
        pv.setContentsMargins(0, 10, 0, 0)
        pv.setSpacing(12)
        f4, l4 = card()
        head = QHBoxLayout()
        self.p_title = label("", "h2")
        head.addWidget(self.p_title, 1)
        self.b_csv = QPushButton("Export CSV")
        self.b_csv.setObjectName("primary")
        self.b_csv.clicked.connect(self.export_csv)
        head.addWidget(self.b_csv)
        l4.addLayout(head)
        self.p_stats = label("", "cap", True)
        l4.addWidget(self.p_stats)
        self.c_product = Chart(330)
        l4.addWidget(self.c_product, 1)
        pv.addWidget(f4, 3)
        f5, l5 = card()
        l5.addWidget(label("Forecast table", "h2"))
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        l5.addWidget(self.table)
        pv.addWidget(f5, 2)
        self.tabs.addTab(pr, "Product")
        outer.addWidget(main, 1)

    # ---------- state ----------
    @staticmethod
    def _on(buttons) -> list:
        vals = [b.property("value") for b in buttons if b.isChecked()]
        if not vals:                       # never allow an empty filter
            for b in buttons:
                b.setChecked(True)
            vals = [b.property("value") for b in buttons]
        return vals

    def filtered(self) -> pd.DataFrame:
        p = self.preds
        return p[p["abc"].isin(self._on(self.b_abc)) & p["demand_type"].isin(self._on(self.b_type))
                 & p["h"].isin(self._on(self.b_h)) & ~p["closed"] & (p["fold_type"] == "final")]

    def refresh(self):
        t = THEMES[self.theme]
        fin = self.filtered()
        y = fin["y"].sum()
        wm = (fin[self.model] - fin["y"]).abs().sum() / y
        wb = (fin[self.base] - fin["y"]).abs().sum() / y
        bias = (fin[self.model] - fin["y"]).sum() / y
        cov = ((fin["y"] >= fin["lo"]) & (fin["y"] <= fin["hi"])).mean()
        ch = wm / wb - 1
        self.k[0][0].setText(pct(wm))
        self.k[0][2].setText(f"{len(fin):,} forecasts, {fin['sku'].nunique()} products")
        self.k[1][0].setText(spct(ch))
        self.k[1][0].setStyleSheet(f"color: {t['good'] if ch < 0 else t['bad']};")
        self.k[1][1].setText(f"Error vs {self.m['best_baseline_label'].lower()} ({pct(wb)})")
        self.k[1][2].setText("model error is lower" if ch < 0 else "model error is higher")
        self.k[2][0].setText(spct(bias))
        self.k[2][2].setText("forecasts run low" if bias < 0 else "forecasts run high")
        self.k[3][0].setText(pct(cov, 0))
        self.k[3][2].setText("target 80%")

        g = fin.groupby("week")[["y", self.model, self.base]].sum()
        self.c_total.lines(t, g.index, [
            dict(name="Actual", color="actual", values=g["y"].to_numpy()),
            dict(name="Model", color="s1", values=g[self.model].to_numpy()),
            dict(name=f"Baseline ({self.m['best_baseline_label'].lower()})", color="s2", width=1.6,
                 values=g[self.base].to_numpy())])
        fs = sorted(self.m["final_scores"], key=lambda r: r["wape"])
        self.c_methods.bars(t, [r["label"].replace("Gradient boosting", "LightGBM") for r in fs], [r["wape"] for r in fs], [r["chosen"] for r in fs])
        self.fill_products()

    def fill_products(self):
        q = self.search.text().strip().lower()
        meta = self.preds.drop_duplicates("sku")
        meta = meta[meta["abc"].isin(self._on(self.b_abc)) & meta["demand_type"].isin(self._on(self.b_type))]
        meta = meta.assign(name=meta["description"].str.title()).sort_values("name")
        if q:
            meta = meta[meta["name"].str.lower().str.contains(q, regex=False) | meta["sku"].str.lower().str.contains(q, regex=False)]
        self.plist.blockSignals(True)
        self.plist.clear()
        for sku, name, abc in zip(meta["sku"], meta["name"], meta["abc"]):
            it = QListWidgetItem(f"{name}\n{sku} · class {abc}")
            it.setData(Qt.ItemDataRole.UserRole, sku)
            self.plist.addItem(it)
        skus = list(meta["sku"])
        if self.sku not in skus:
            self.sku = skus[0] if skus else None
        if self.sku:
            self.plist.setCurrentRow(skus.index(self.sku))
        self.plist.blockSignals(False)
        self.show_product()

    def on_product(self, item, _prev=None):
        if item:
            self.sku = item.data(Qt.ItemDataRole.UserRole)
            self.show_product()
            self.tabs.setCurrentIndex(1)

    def product_table(self) -> pd.DataFrame:
        p = self.preds[self.preds["sku"] == self.sku].sort_values("week")
        return pd.DataFrame({
            "Week": p["week"].dt.strftime("%Y-%m-%d"), "Weeks ahead": p["h"], "Actual": p["y"].round(0),
            "Forecast": p[self.model].round(1), "Low (10%)": p["lo"].round(1), "High (90%)": p["hi"].round(1),
            "Baseline": p[self.base].round(1),
            "Period": np.where(p["closed"], "closed", np.where(p["fold_type"] == "final", "final test", "selection"))})

    def show_product(self):
        t = THEMES[self.theme]
        if not self.sku:
            self.p_title.setText("No product matches")
            return
        hist = self.panel[self.panel["sku"] == self.sku].sort_values("week")
        p = self.preds[self.preds["sku"] == self.sku].set_index("week").reindex(hist["week"])
        row = self.preds[self.preds["sku"] == self.sku].iloc[0]
        self.p_title.setText(f"{str(row['description']).title()} ({self.sku})")
        fin = self.filtered()
        fin = fin[fin["sku"] == self.sku]
        y = fin["y"].sum()
        if y > 0:
            self.p_stats.setText(
                f"Model WAPE {pct((fin[self.model] - fin['y']).abs().sum() / y)}   ·   "
                f"Baseline WAPE {pct((fin[self.base] - fin['y']).abs().sum() / y)}   ·   "
                f"Bias {spct((fin[self.model] - fin['y']).sum() / y)}   ·   "
                f"{y:,.0f} units in the test period   ·   class {row['abc']}, {row['demand_type']}")
        else:
            self.p_stats.setText("")
        hs = p["h"].to_numpy()
        self.c_product.lines(
            t, hist["week"],
            [dict(name="Actual", color="actual", width=1.6, values=hist["units"].to_numpy(float)),
             dict(name="Forecast", color="s1", values=p[self.model].to_numpy(float))],
            band=(p["lo"].to_numpy(float), p["hi"].to_numpy(float)), mark=self.final_start,
            extra=lambda i: f"Made {int(hs[i])} wk ahead" if not np.isnan(hs[i]) else "")

        tb = self.product_table()
        self.table.setRowCount(len(tb))
        self.table.setColumnCount(len(tb.columns))
        self.table.setHorizontalHeaderLabels(list(tb.columns))
        for r, rec in enumerate(tb.itertuples(index=False)):
            for c, v in enumerate(rec):
                text = f"{v:,.0f}" if isinstance(v, float) else str(v)
                it = QTableWidgetItem(text)
                it.setTextAlignment((Qt.AlignmentFlag.AlignLeft if c in (0, 7) else Qt.AlignmentFlag.AlignRight)
                                    | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, it)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def export_csv(self):
        if not self.sku:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export forecast", f"{self.sku}_forecast.csv", "CSV (*.csv)")
        if path:
            self.product_table().to_csv(path, index=False)

    def set_notes(self):
        m = self.m
        row = {r["model"]: r for r in m["final_scores"]}
        ch, tw, v, d = row[self.model], row["gbm_tweedie"], m["vs_best_baseline_on_final_folds"], m["design"]
        self.notes.setText(
            f"<b>WAPE</b> is total absolute error divided by total units sold. Weekly demand for single gift products is "
            f"noisy, so every method has a large error; the comparison between methods is what matters.<br><br>"
            f"<b>Against the strongest simple method</b> ({m['best_baseline_on_final_folds_label'].lower()}), the model's "
            f"error is {spct(v['relative_change'])} (95% confidence interval {spct(v['relative_ci95'][0])} to "
            f"{spct(v['relative_ci95'][1])}).<br><br>"
            f"<b>Trade-off.</b> The chosen setting forecasts {abs(100 * ch['bias']):.0f}% fewer units than were sold, so its "
            f"error on total weekly demand ({pct(ch['weekly_total_wape'])}) is worse than the simple methods'. The Tweedie "
            f"setting has a bias of {spct(tw['bias'], 1)} and a weekly-total error of {pct(tw['weekly_total_wape'])}, but a "
            f"product-level error of {pct(tw['wape'])}.<br><br>"
            f"<b>Test design.</b> {d['folds_total']} forecast dates, 4 weeks apart. The first {d['folds_selection']} chose "
            f"the model setting; the last {d['folds_final']} are the final test shown here.")

    # ---------- theme ----------
    def apply_theme(self):
        self.setStyleSheet(qss(THEMES[self.theme]))
        self.b_theme.setText("Light mode" if self.theme == "dark" else "Dark mode")
        self.set_notes()
        self.refresh()

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self.apply_theme()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    w = Window()
    if "--screenshot" in sys.argv:           # used to produce portfolio images without a display
        out = Path(sys.argv[sys.argv.index("--screenshot") + 1])
        w.show()
        app.processEvents()
        for name, tab, theme in (("overview", 0, "light"), ("product", 1, "light"), ("overview_dark", 0, "dark")):
            if w.theme != theme:
                w.toggle_theme()
            w.tabs.setCurrentIndex(tab)
            app.processEvents()
            w.grab().save(str(out / f"desktop_{name}.png"))
        return
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
