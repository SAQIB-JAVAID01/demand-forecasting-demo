"""Demand Forecasting Studio (PyQt6 desktop app).

Run:  python app/desktop.py
Reads the backtest output in data/processed and reports/metrics.json.
Every sentence and number on screen is computed from that output.
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
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QPushButton, QScrollArea,
                             QSizePolicy, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src.report import FEATURE_NAMES  # noqa: E402

BASE = "ma13"                       # strongest simple method on the final test (the tougher comparison)
BASE_LABEL = "13-week average"
MODELS = {"gbm_log": "Best per-product accuracy", "gbm_tweedie": "Unbiased totals"}

THEMES = {
    "light": dict(page="#f3f4f6", surface="#ffffff", ink="#0f172a", ink2="#475569", muted="#94a3b8",
                  grid="#e5e7eb", axis="#cbd5e1", border="#e5e7eb", s1="#2a78d6", s2="#eb6834",
                  actual="#475569", band="#cde2fb", neutral="#cbd5e1", chip="#eef0f3", nav="#0f1b2d",
                  navink="#cbd5e1", navsel="#1d3557", good="#0a7d32", bad="#c0392b", warn="#b45309",
                  sel="#e4eefb", tint="#f1f6fd", soft="#9ec5f4"),
    "dark": dict(page="#0b0f17", surface="#141a24", ink="#f1f5f9", ink2="#cbd5e1", muted="#7c8799",
                 grid="#232b38", axis="#334155", border="#232b38", s1="#3987e5", s2="#d95926",
                 actual="#cbd5e1", band="#104281", neutral="#3b4556", chip="#1e2633", nav="#070a10",
                 navink="#94a3b8", navsel="#16233a", good="#34c36b", bad="#f07171", warn="#f5a524",
                 sel="#17304f", tint="#111c2e", soft="#1c5cab"),
}


def qss(t: dict) -> str:
    return f"""
    QWidget {{ background: {t['page']}; color: {t['ink']}; font-family: 'Segoe UI'; font-size: 10.5pt; }}
    QScrollArea {{ border: none; }}
    #nav {{ background: {t['nav']}; }}
    #nav QLabel {{ background: {t['nav']}; color: #ffffff; }}
    #nav QLabel#navsub {{ color: {t['navink']}; font-size: 9pt; }}
    #nav QLabel#navfoot {{ color: {t['navink']}; font-size: 8.5pt; }}
    #nav QPushButton {{ background: transparent; color: {t['navink']}; text-align: left; padding: 11px 14px;
                        border-radius: 8px; font-size: 10.5pt; }}
    #nav QPushButton:hover {{ background: {t['navsel']}; color: #ffffff; }}
    #nav QPushButton:checked {{ background: {t['navsel']}; color: #ffffff; font-weight: 600;
                                border-left: 3px solid {t['s1']}; }}
    #topbar, #topbar QWidget {{ background: {t['surface']}; }}
    #topbar {{ border-bottom: 1px solid {t['border']}; }}
    QFrame#card {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 12px; }}
    QFrame#card QWidget {{ background: {t['surface']}; }}
    QFrame#hero {{ background: {t['tint']}; border: 1px solid {t['border']}; border-left: 5px solid {t['s1']};
                   border-radius: 12px; }}
    QFrame#hero QWidget {{ background: {t['tint']}; }}
    QLabel#title {{ font-size: 16pt; font-weight: 700; }}
    QLabel#sub {{ color: {t['ink2']}; }}
    QLabel#flabel {{ color: {t['muted']}; font-size: 8.5pt; font-weight: 600; }}
    QLabel#hero1 {{ font-size: 14pt; font-weight: 600; }}
    QLabel#hero2 {{ color: {t['ink2']}; font-size: 10.5pt; }}
    QLabel#kv {{ font-size: 23pt; font-weight: 700; }}
    QLabel#kl {{ color: {t['ink']}; font-weight: 600; }}
    QLabel#kn {{ color: {t['ink2']}; font-size: 9.5pt; }}
    QLabel#eyebrow {{ color: {t['s1']}; font-size: 8.5pt; font-weight: 700; }}
    QLabel#h2 {{ font-size: 12pt; font-weight: 700; }}
    QLabel#cap {{ color: {t['ink2']}; font-size: 9.5pt; }}
    QLabel#body {{ color: {t['ink2']}; font-size: 10.5pt; }}
    QPushButton {{ background: {t['chip']}; color: {t['ink']}; border: none; border-radius: 8px; padding: 6px 13px; }}
    QPushButton:hover {{ background: {t['grid']}; }}
    #topbar QPushButton, QFrame#card QPushButton {{ background: {t['chip']}; color: {t['ink']}; }}
    #topbar QPushButton:checked {{ background: {t['ink']}; color: {t['surface']}; }}
    #topbar QPushButton#seg:checked {{ background: {t['s1']}; color: #ffffff; }}
    QFrame#card QPushButton#primary {{ background: {t['s1']}; color: #ffffff; font-weight: 600; padding: 7px 16px; }}
    QLineEdit {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 8px; padding: 7px 10px; }}
    QListWidget {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 8px; outline: none; }}
    QListWidget::item {{ padding: 6px 8px; border-radius: 6px; }}
    QListWidget::item:selected {{ background: {t['sel']}; color: {t['ink']}; }}
    QTableWidget {{ background: {t['surface']}; border: none; alternate-background-color: {t['page']}; }}
    QTableWidget::item:selected {{ background: {t['sel']}; color: {t['ink']}; }}
    QHeaderView::section {{ background: {t['surface']}; color: {t['ink2']}; border: none;
                            border-bottom: 1px solid {t['grid']}; padding: 7px; font-weight: 600; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; }}
    QScrollBar::handle:vertical {{ background: {t['axis']}; border-radius: 5px; min-height: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    """


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def spct(x: float, d: int = 0) -> str:
    return f"{100 * x:+.{d}f}%"


def label(text: str = "", name: str = "", wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


class Chart(FigureCanvasQTAgg):
    """Matplotlib canvas in the app's style, with a hover read-out on line charts."""

    def __init__(self, height: int = 280):
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
            self.fig.tight_layout(pad=0.9)
        except Exception:
            pass

    def _base(self, t: dict, grid: str = "y"):
        self.t = t
        self._hover = None
        ax = self.ax
        ax.clear()
        self.fig.set_facecolor(t["surface"])
        ax.set_facecolor(t["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(t["axis"])
        ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=9, length=3)
        ax.grid(False)
        ax.grid(axis=grid, color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        return ax

    def _legend(self, ncols: int = 4):
        leg = self.ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=ncols, frameon=False, fontsize=9,
                             handlelength=1.5, borderaxespad=0.2, columnspacing=1.4)
        for txt in leg.get_texts():
            txt.set_color(self.t["ink2"])

    def _done(self):
        try:
            self.fig.tight_layout(pad=0.9)
        except Exception:
            pass
        self.draw_idle()

    # ----- chart types -----
    def lines(self, t, x, series, band=None, mark=None, shade=None, extra=None):
        ax = self._base(t)
        x = pd.DatetimeIndex(x)
        if shade is not None:
            ax.axvspan(shade[0], shade[1], color=t["band"], alpha=0.45, linewidth=0, label=shade[2])
        if band is not None:
            ax.fill_between(x, band[0], band[1], color=t["band"], linewidth=0, label="80% range")
        for s in series:
            ax.plot(x, s["values"], color=t[s["color"]], linewidth=s.get("width", 2.2), label=s["name"],
                    solid_capstyle="round")
        if mark is not None:
            ax.axvline(mark, color=t["axis"], linewidth=1, linestyle="--")
        ax.set_ylim(0, None)
        ax.margins(x=0.01)
        top = ax.get_ylim()[1]
        ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if top >= 10000 else f"{v:,.0f}")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        self._legend()
        self._hover = dict(x=x, xnum=mdates.date2num(x.to_pydatetime()), series=series, band=band, extra=extra,
                           line=ax.axvline(x[0], color=t["muted"], linewidth=1, visible=False),
                           note=ax.annotate("", xy=(0, 0), xytext=(12, -12), textcoords="offset points",
                                            va="top", fontsize=9, color=t["ink"], visible=False, zorder=10,
                                            bbox=dict(boxstyle="round,pad=0.5", fc=t["surface"], ec=t["axis"], lw=1)))
        self._done()

    def hbars(self, t, labels, values, colors, fmt=pct, signed=False):
        ax = self._base(t, grid="x")
        y = np.arange(len(labels))[::-1]
        ax.barh(y, values, height=0.62, color=[t[c] for c in colors])
        span = max(abs(min(values + [0])), abs(max(values + [0]))) or 1
        for yi, v, c in zip(y, values, colors):
            ax.text(v + span * 0.015 if v >= 0 else span * 0.015, yi, fmt(v), va="center",
                    ha="left", fontsize=9, color=t["ink"],
                    fontweight="bold" if c == "s1" else "normal")
        ax.set_yticks(y, labels)
        ax.set_xlim(min(0, min(values)) * 1.04, max(values + [0]) + span * 0.18)
        ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%" if fmt is pct else fmt(v))
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if signed:
            ax.axvline(0, color=t["axis"], linewidth=1)
        self._done()

    def grouped(self, t, labels, a, b, name_a, name_b, horizontal=True):
        """Two bars per category: a = model (blue), b = simple method (orange)."""
        ax = self._base(t, grid="x" if horizontal else "y")
        n = np.arange(len(labels))
        top = max(a + b) * 1.17
        if horizontal:
            y = n[::-1]
            ax.barh(y + 0.19, a, height=0.34, color=t["s1"], label=name_a)
            ax.barh(y - 0.19, b, height=0.34, color=t["s2"], label=name_b)
            for yi, va, vb in zip(y, a, b):
                ax.text(va + top * 0.008, yi + 0.19, pct(va), va="center", fontsize=8.5, color=t["ink"], fontweight="bold")
                ax.text(vb + top * 0.008, yi - 0.19, pct(vb), va="center", fontsize=8.5, color=t["ink"])
            ax.set_yticks(y, labels)
            ax.set_xlim(0, top)
            ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        else:
            ax.bar(n - 0.19, a, width=0.34, color=t["s1"], label=name_a)
            ax.bar(n + 0.19, b, width=0.34, color=t["s2"], label=name_b)
            for xi, va, vb in zip(n, a, b):
                ax.text(xi - 0.19, va + top * 0.012, pct(va), ha="center", fontsize=8.5, color=t["ink"], fontweight="bold")
                ax.text(xi + 0.19, vb + top * 0.012, pct(vb), ha="center", fontsize=8.5, color=t["ink"])
            ax.set_xticks(n, labels)
            ax.set_ylim(0, top)
            ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
        self._legend()
        self._done()

    def vbars(self, t, labels, values, colors, target=None, target_label="", fmt=pct, ymax=None, signed=False,
              inside=False):
        ax = self._base(t)
        n = np.arange(len(labels))
        ax.bar(n, values, width=0.6, color=[t[c] for c in colors])
        span = max(abs(min(values + [0])), abs(max(values + [0]))) or 1
        for xi, v in zip(n, values):
            if inside:
                ax.text(xi, v - span * 0.04, fmt(v), ha="center", va="top", fontsize=10, color="#ffffff", fontweight="bold")
            else:
                ax.text(xi, v + (span * 0.025 if v >= 0 else -span * 0.025), fmt(v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=9, color=t["ink"])
        if target is not None:
            ax.axhline(target, color=t["ink2"], linewidth=1.2, linestyle="--", label=target_label)
            self._legend()
        if signed:
            ax.axhline(0, color=t["axis"], linewidth=1)
        ax.set_xticks(n, labels)
        lo = min(0, min(values)) - (span * 0.2 if min(values) < 0 else 0)
        ax.set_ylim(lo, ymax if ymax else max(values + [0]) + span * 0.2)
        ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%" if fmt is pct else fmt(v))
        self._done()

    def scatter(self, t, x, y, colors, xlabel, ylabel):
        ax = self._base(t, grid="both")
        lo, hi = max(1, min(x.min(), y.min()) * 0.8), max(x.max(), y.max()) * 1.25
        ax.plot([lo, hi], [lo, hi], color=t["ink2"], linewidth=1.2, linestyle="--", label="Perfect forecast")
        for c, name in (("s1", "Model beats the simple method"), ("s2", "Simple method is better")):
            k = [i for i, v in enumerate(colors) if v == c]
            ax.scatter(x[k], y[k], s=36, color=t[c], edgecolors=t["surface"], linewidths=0.8, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:,.0f}")
            axis.set_minor_formatter(lambda v, _: "")
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9)
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9)
        self._legend(ncols=1)
        self._done()

    def hist(self, t, values, bins, xlabel):
        ax = self._base(t)
        counts, edges = np.histogram(values, bins=bins)
        mids = (edges[:-1] + edges[1:]) / 2
        ax.bar(mids, counts, width=(edges[1] - edges[0]) * 0.88, color=[t["s1"] if m < 0 else t["s2"] for m in mids])
        ax.axvline(0, color=t["ink2"], linewidth=1.2, linestyle="--")
        ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:+.0f}%")
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9)
        ax.set_ylabel("Number of products", color=t["ink2"], fontsize=9)
        self._done()

    def timeline(self, t, folds, weeks):
        ax = self._base(t, grid="x")
        x0 = mdates.date2num(weeks[0])
        seen = set()
        for i, (cut, kind) in enumerate(folds):
            yy = len(folds) - 1 - i
            c = mdates.date2num(weeks[cut])
            ax.barh(yy, c + 7 - x0, left=x0, height=0.55, color=t["neutral"],
                    label=None if "h" in seen else "History the model can see")
            seen.add("h")
            ax.barh(yy, 28, left=c + 7, height=0.55, color=t["soft"] if kind == "selection" else t["s1"],
                    label=None if kind in seen else {"selection": "4 weeks forecast: used to choose settings",
                                                     "final": "4 weeks forecast: final test"}[kind])
            seen.add(kind)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_yticks([len(folds) - 1, 0], ["First\nforecast date", "Last\nforecast date"])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        self._legend(ncols=1)
        self._done()

    # ----- hover -----
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
            rows.append(f"80% range: {h['band'][0][i]:,.0f} to {h['band'][1][i]:,.0f}")
        if h["extra"]:
            e = h["extra"](i)
            if e:
                rows.append(e)
        h["line"].set_xdata([h["x"][i], h["x"][i]])
        h["line"].set_visible(True)
        right = i > len(h["x"]) * 0.6
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


class Card(QFrame):
    """A card: category line, a plain-language finding, optional chart or text, and how to read it."""

    def __init__(self, eyebrow: str, chart_height: int = 0, text: bool = False):
        super().__init__()
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(18, 15, 18, 15)
        self.lay.setSpacing(5)
        self.finding = label("", "h2", True)
        self.lay.addWidget(label(eyebrow.upper(), "eyebrow"))
        self.lay.addWidget(self.finding)
        self.chart = self.body = None
        if chart_height:
            self.chart = Chart(chart_height)
            self.lay.addWidget(self.chart, 1)
        if text:
            self.body = label("", "body", True)
            self.body.setTextFormat(Qt.TextFormat.RichText)
            self.body.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.lay.addWidget(self.body, 1)
        self.how = label("", "cap", True)
        self.lay.addWidget(self.how)

    def say(self, finding: str, how: str = ""):
        self.finding.setText(finding)
        self.how.setText(how)
        self.how.setVisible(bool(how))


class NumItem(QTableWidgetItem):
    """Table cell that shows formatted text but sorts by its number."""

    def __init__(self, text: str, value: float):
        super().__init__(text)
        self.value = value

    def __lt__(self, other):
        return self.value < getattr(other, "value", 0.0)


PAGES = [("Executive summary", "The answer in one screen"),
         ("Forecast accuracy", "Where the forecast is strong and where it is weak"),
         ("Products", "How every product performed in the final test"),
         ("Product explorer", "History, forecast and range for one product"),
         ("Risk and stock", "How far actual demand can move from the forecast"),
         ("Data and method", "What went in and how it was tested")]


def make_table(min_height: int) -> QTableWidget:
    tb = QTableWidget()
    tb.setAlternatingRowColors(True)
    tb.verticalHeader().setVisible(False)
    tb.setShowGrid(False)
    tb.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    tb.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    tb.setMinimumHeight(min_height)
    return tb


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demand Forecasting Studio")
        self.resize(1480, 940)
        self.theme = "light"
        self.preds = pd.read_parquet(C.PROCESSED_DIR / "predictions.parquet")
        self.panel = pd.read_parquet(C.PROCESSED_DIR / "panel.parquet")
        self.cal = pd.read_parquet(C.PROCESSED_DIR / "calendar.parquet")
        self.m = json.loads((C.REPORTS_DIR / "metrics.json").read_text())
        self.model = self.m["chosen_model"]
        self.final_start = self.preds.loc[self.preds["fold_type"] == "final", "week"].min()
        self.final_end = self.preds["week"].max()
        self.sku = None
        self._ci_cache = {}
        self._build()
        self.apply_theme()

    # ================= layout =================
    def _page(self) -> QVBoxLayout:
        area = QScrollArea()
        area.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(24, 18, 24, 24)
        lay.setSpacing(14)
        area.setWidget(inner)
        self.stack.addWidget(area)
        return lay

    @staticmethod
    def _row(lay, *widgets, stretch=None, weight: int = 0):
        row = QHBoxLayout()
        row.setSpacing(14)
        for i, w in enumerate(widgets):
            row.addWidget(w, stretch[i] if stretch else 1)
        lay.addLayout(row, weight)

    def _chips(self, lay, title, options) -> list[QPushButton]:
        lay.addWidget(label(title, "flabel"))
        out = []
        for value, text in options:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setChecked(True)
            b.setProperty("value", value)
            b.clicked.connect(self.refresh)
            lay.addWidget(b)
            out.append(b)
        lay.addSpacing(14)
        return out

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        nav = QFrame()
        nav.setObjectName("nav")
        nav.setFixedWidth(232)
        n = QVBoxLayout(nav)
        n.setContentsMargins(14, 20, 14, 16)
        n.setSpacing(4)
        brand = label("Demand Forecasting\nStudio")
        brand.setStyleSheet("font-size: 13.5pt; font-weight: 700;")
        n.addWidget(brand)
        n.addWidget(label("UK online retailer · 200 products", "navsub"))
        n.addSpacing(18)
        self.nav_group = QButtonGroup(self)
        for i, (name, _) in enumerate(PAGES):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setChecked(i == 0)
            b.clicked.connect(lambda _=False, k=i: self.go(k))
            self.nav_group.addButton(b, i)
            n.addWidget(b)
        n.addStretch(1)
        self.b_theme = QPushButton("Dark mode")
        self.b_theme.clicked.connect(self.toggle_theme)
        n.addWidget(self.b_theme)
        n.addSpacing(6)
        n.addWidget(label("Public data: UCI Online Retail II\nPython · pandas · LightGBM · PyQt6", "navfoot"))
        outer.addWidget(nav)

        right = QWidget()
        r = QVBoxLayout(right)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(0)
        top = QFrame()
        top.setObjectName("topbar")
        tl = QVBoxLayout(top)
        tl.setContentsMargins(24, 14, 24, 12)
        tl.setSpacing(6)
        self.title, self.subtitle = label("", "title"), label("", "sub")
        tl.addWidget(self.title)
        tl.addWidget(self.subtitle)
        tl.addSpacing(4)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(label("FORECAST SETTING", "flabel"))
        self.b_model = []
        grp = QButtonGroup(self)
        for key, text in MODELS.items():
            b = QPushButton(text)
            b.setObjectName("seg")
            b.setCheckable(True)
            b.setChecked(key == self.model)
            b.setProperty("value", key)
            b.clicked.connect(self.on_model)
            grp.addButton(b)
            bar.addWidget(b)
            self.b_model.append(b)
        bar.addSpacing(14)
        self.b_abc = self._chips(bar, "REVENUE CLASS", [("A", "A"), ("B", "B"), ("C", "C")])
        self.b_type = self._chips(bar, "DEMAND", [("regular", "Regular"), ("intermittent", "Intermittent")])
        self.b_h = self._chips(bar, "WEEKS AHEAD", [(1, "1"), (2, "2"), (3, "3"), (4, "4")])
        bar.addStretch(1)
        tl.addLayout(bar)
        r.addWidget(top)
        self.stack = QStackedWidget()
        r.addWidget(self.stack, 1)
        outer.addWidget(right, 1)

        # ----- 1 executive summary -----
        lay = self._page()
        hero = QFrame()
        hero.setObjectName("hero")
        hl = QVBoxLayout(hero)
        hl.setContentsMargins(20, 16, 20, 16)
        hl.setSpacing(6)
        self.hero1, self.hero2 = label("", "hero1", True), label("", "hero2", True)
        hl.addWidget(self.hero1)
        hl.addWidget(self.hero2)
        lay.addWidget(hero)
        self.k = []
        krow = QHBoxLayout()
        krow.setSpacing(14)
        for title in ("Forecast error", "Against the best simple method", "Forecast level", "Range reliability"):
            f = QFrame()
            f.setObjectName("card")
            l = QVBoxLayout(f)
            l.setContentsMargins(18, 14, 18, 14)
            l.setSpacing(2)
            val, note = label("", "kv"), label("", "kn", True)
            l.addWidget(label(title, "kl"))
            l.addWidget(val)
            l.addWidget(note)
            krow.addWidget(f)
            self.k.append((val, note))
        lay.addLayout(krow)
        self.s_total, self.s_methods = Card("Forecast vs actual", 290), Card("Method comparison", 290)
        self._row(lay, self.s_total, self.s_methods, stretch=(3, 2), weight=1)
        self.s_take, self.s_hist = Card("What this means for the business", text=True), Card("Two years of demand", 240)
        self._row(lay, self.s_take, self.s_hist, stretch=(2, 3), weight=1)

        # ----- 2 accuracy -----
        lay = self._page()
        self.a_h, self.a_seg = Card("By forecast distance", 300), Card("By product group and season", 300)
        self._row(lay, self.a_h, self.a_seg, stretch=(2, 3), weight=1)
        self.a_bias, self.a_dist = Card("Forecast level by month", 300), Card("Product by product", 300)
        self._row(lay, self.a_bias, self.a_dist, weight=1)

        # ----- 3 products -----
        lay = self._page()
        self.p_scatter, self.p_tab = Card("Forecast vs actual, per product", 420), Card("All products")
        self.ptable = make_table(420)
        self.ptable.cellDoubleClicked.connect(self.open_from_table)
        self.p_tab.lay.insertWidget(2, self.ptable, 1)
        self._row(lay, self.p_scatter, self.p_tab, stretch=(4, 7), weight=1)

        # ----- 4 explorer -----
        lay = self._page()
        left = QFrame()
        left.setObjectName("card")
        left.setFixedWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 14, 14, 14)
        ll.addWidget(label("PRODUCT", "eyebrow"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search product or code")
        self.search.textChanged.connect(self.fill_products)
        ll.addWidget(self.search)
        self.plist = QListWidget()
        self.plist.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.plist.setWordWrap(True)
        self.plist.currentItemChanged.connect(self.on_product)
        ll.addWidget(self.plist, 1)
        col = QVBoxLayout()
        col.setSpacing(14)
        self.e_card = Card("One product", 330)
        self.b_csv = QPushButton("Export CSV")
        self.b_csv.setObjectName("primary")
        self.b_csv.clicked.connect(self.export_csv)
        head = QHBoxLayout()
        head.addStretch(1)
        head.addWidget(self.b_csv)
        self.e_card.lay.addLayout(head)
        col.addWidget(self.e_card, 3)
        tcard = Card("Week by week")
        self.table = make_table(200)
        tcard.lay.insertWidget(2, self.table, 1)
        tcard.say("Every forecast for this product",
                  "Low and High are the 80% range. Period shows whether the week was used to choose settings (selection) "
                  "or belongs to the final test.")
        col.addWidget(tcard, 2)
        row = QHBoxLayout()
        row.setSpacing(14)
        row.addWidget(left)
        row.addLayout(col, 1)
        lay.addLayout(row, 1)

        # ----- 5 risk -----
        lay = self._page()
        self.r_cov, self.r_serv = Card("Is the range trustworthy?", 300), Card("Upper bound as a stock level", 300)
        self._row(lay, self.r_cov, self.r_serv, weight=1)
        self.r_note, self.r_bias = Card("How to use the range", text=True), Card("Forecast level by method", 240)
        self._row(lay, self.r_note, self.r_bias, weight=1)

        # ----- 6 method -----
        lay = self._page()
        self.d_clean, self.d_time = Card("Data cleaning", 290), Card("Test design", 290)
        self._row(lay, self.d_clean, self.d_time, weight=1)
        self.d_imp, self.d_txt = Card("What drives the forecast", 300), Card("Method in plain words", text=True)
        self._row(lay, self.d_imp, self.d_txt, weight=1)

    # ================= state =================
    def go(self, i: int):
        self.stack.setCurrentIndex(i)
        self.nav_group.button(i).setChecked(True)
        self.title.setText(PAGES[i][0])
        self.subtitle.setText(PAGES[i][1])

    def on_model(self):
        self.model = next(b.property("value") for b in self.b_model if b.isChecked())
        self.refresh()

    @staticmethod
    def _on(buttons) -> list:
        vals = [b.property("value") for b in buttons if b.isChecked()]
        if not vals:                       # an empty filter is never allowed
            for b in buttons:
                b.setChecked(True)
            vals = [b.property("value") for b in buttons]
        return vals

    def filtered(self) -> pd.DataFrame:
        p = self.preds
        return p[p["abc"].isin(self._on(self.b_abc)) & p["demand_type"].isin(self._on(self.b_type))
                 & p["h"].isin(self._on(self.b_h)) & ~p["closed"] & (p["fold_type"] == "final")]

    @staticmethod
    def wape(df, col) -> float:
        return float((df[col] - df["y"]).abs().sum() / df["y"].sum())

    @staticmethod
    def bias(df, col) -> float:
        return float((df[col] - df["y"]).sum() / df["y"].sum())

    def ci(self, fin: pd.DataFrame) -> tuple[float, float]:
        """95% range for the relative error change vs the simple method, resampling whole products."""
        key = (self.model, tuple(self._on(self.b_abc)), tuple(self._on(self.b_type)), tuple(self._on(self.b_h)))
        if key not in self._ci_cache:
            g = fin.assign(em=(fin[self.model] - fin["y"]).abs(), eb=(fin[BASE] - fin["y"]).abs()) \
                   .groupby("sku")[["em", "eb"]].sum().to_numpy()
            rng = np.random.default_rng(C.SEED)
            s = g[rng.integers(0, len(g), size=(1000, len(g)))].sum(axis=1)
            rel = s[:, 0] / s[:, 1] - 1
            self._ci_cache[key] = (float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5)))
        return self._ci_cache[key]

    # ================= rendering =================
    def refresh(self):
        t = THEMES[self.theme]
        fin = self.filtered()
        mdl = self.model
        wm, wb, bs = self.wape(fin, mdl), self.wape(fin, BASE), self.bias(fin, mdl)
        ch = wm / wb - 1
        lo, hi = self.ci(fin)
        cov = float(((fin["y"] >= fin["lo"]) & (fin["y"] <= fin["hi"])).mean())
        gain, loss = hi < 0, lo > 0

        # ----- headline -----
        if gain:
            h1 = f"The forecast is {abs(100 * ch):.0f}% more accurate per product than the best simple method."
        elif loss:
            h1 = f"The forecast is {abs(100 * ch):.0f}% less accurate per product than the best simple method."
        else:
            h1 = "Per product, the forecast is about as accurate as the best simple method. The difference is not clear."
        if bs < -0.10:
            h2 = (f"But it forecasts {abs(100 * bs):.0f}% fewer units than were actually sold. For total stock, switch to "
                  f"\"Unbiased totals\" above or plan from the upper range.")
        elif bs > 0.10:
            h2 = f"But it forecasts {abs(100 * bs):.0f}% more units than were actually sold, so totals run high."
        else:
            h2 = f"Its totals are close to actual sales ({spct(bs, 1)} of units), so it is safe to use for total stock planning."
        self.hero1.setText(h1)
        self.hero2.setText(f"{h2}  Based on {len(fin):,} weekly forecasts for {fin['sku'].nunique()} products, "
                           f"{self.final_start:%d %b %Y} to {self.final_end:%d %b %Y}: a period never used to build or tune the model.")

        self.k[0][0].setText(pct(wm))
        self.k[0][1].setText(f"Off by about {100 * wm:.0f} units for every 100 sold, per product and week.")
        self.k[1][0].setText(spct(ch))
        self.k[1][0].setStyleSheet(f"color: {t['good'] if gain else t['bad'] if loss else t['warn']};")
        self.k[1][1].setText(f"The {BASE_LABEL} has {pct(wb)} error. 95% confidence range: {spct(lo)} to {spct(hi)}"
                             f"{'' if (gain or loss) else ' (includes zero)'}.")
        self.k[2][0].setText(spct(bs))
        self.k[2][0].setStyleSheet(f"color: {t['bad'] if abs(bs) > 0.10 else t['good']};")
        self.k[2][1].setText("Forecast units against actual units. "
                             + ("Runs low." if bs < -0.03 else "Runs high." if bs > 0.03 else "On target."))
        self.k[3][0].setText(pct(cov, 0))
        self.k[3][0].setStyleSheet(f"color: {t['good'] if abs(cov - 0.8) <= 0.05 else t['warn']};")
        self.k[3][1].setText("of actual weeks fell inside the 80% range. Target: 80%.")

        # ----- summary charts -----
        g = fin.groupby("week")[["y", mdl, BASE]].sum()
        te = float((g[mdl] - g["y"]).abs().sum() / g["y"].sum())
        tb_ = float((g[BASE] - g["y"]).abs().sum() / g["y"].sum())
        self.s_total.say(f"On total weekly demand the forecast is off by {pct(te, 0)}; the simple method by {pct(tb_, 0)}.",
                         "Grey = units actually sold. Blue = forecast made 1 to 4 weeks earlier. Orange = simple method. "
                         "Move the mouse over the chart for values.")
        self.s_total.chart.lines(t, g.index, [
            dict(name="Actual units", color="actual", values=g["y"].to_numpy()),
            dict(name="Forecast", color="s1", values=g[mdl].to_numpy()),
            dict(name=BASE_LABEL, color="s2", width=1.6, values=g[BASE].to_numpy())])

        names = {"gbm_log": "Model: best per-product", "gbm_tweedie": "Model: unbiased totals", "ma13": "13-week average",
                 "ses": "Exponential smoothing", "ma4": "4-week average", "naive": "Last week's sales",
                 "snaive": "Same week last year"}
        sc = sorted((self.wape(fin, c), c) for c in names)
        rank = [c for _, c in sc].index(mdl) + 1
        self.s_methods.say(f"The selected setting ranks {rank} of {len(sc)} methods on per-product error.",
                           "Shorter bar = smaller error. Blue = the setting selected at the top.")
        self.s_methods.chart.hbars(t, [names[c] for _, c in sc], [v for v, _ in sc],
                                   ["s1" if c == mdl else "neutral" for _, c in sc])

        ps = fin.groupby("sku").apply(lambda d: pd.Series({
            "units": d["y"].sum(), "fc": d[mdl].sum(),
            "wm": self.wape(d, mdl) if d["y"].sum() > 0 else np.nan,
            "wb": self.wape(d, BASE) if d["y"].sum() > 0 else np.nan}), include_groups=False).dropna()
        share = float((ps["wm"] < ps["wb"]).mean())
        other = "gbm_tweedie" if mdl == "gbm_log" else "gbm_log"
        self.s_take.say("Three things to take away")
        self.s_take.body.setText(
            f"<b>1. Accuracy.</b> The forecast beats the {BASE_LABEL} on {pct(share, 0)} of products. Gift demand is erratic "
            f"week to week, so every method has a large error. The gain over simple methods is the value added.<br><br>"
            f"<b>2. Level.</b> This setting's total is {spct(bs)} against actual sales. The other setting, "
            f"\"{MODELS[other]}\", is {spct(self.bias(fin, other))} with a per-product error of {pct(self.wape(fin, other))}. "
            f"Use the switch at the top to compare.<br><br>"
            f"<b>3. Risk.</b> The 80% range held {pct(cov, 0)} of the time, so its upper bound is a usable starting point "
            f"for safety stock.")

        tot = self.panel.groupby("week")["units"].sum()
        self.s_hist.say(f"Demand peaks before Christmas. The busiest week was {tot.idxmax():%d %b %Y} with {tot.max():,.0f} units.",
                        "All 200 products. The shaded area is the final test period that every other chart reports on.")
        self.s_hist.chart.lines(t, tot.index, [dict(name="Units sold per week", color="actual", values=tot.to_numpy())],
                                shade=(self.final_start, self.final_end, "Final test period"))

        # ----- accuracy -----
        hh = fin.groupby("h").apply(lambda d: pd.Series({"m": self.wape(d, mdl), "b": self.wape(d, BASE)}), include_groups=False)
        if len(hh) > 1:
            self.a_h.say(f"Error is lowest {hh['m'].idxmin()} wk ahead ({pct(hh['m'].min())}) and highest "
                         f"{hh['m'].idxmax()} wk ahead ({pct(hh['m'].max())}).", "Blue = model, orange = simple method. Lower is better.")
        else:
            self.a_h.say(f"Error {hh.index[0]} wk ahead is {pct(hh['m'].iloc[0])}.", "Blue = model, orange = simple method.")
        self.a_h.chart.grouped(t, [f"{h} wk ahead" for h in hh.index], list(hh["m"]), list(hh["b"]), "Model", BASE_LABEL)

        f2 = fin.assign(season=np.where(fin["week"].dt.month >= 9, "peak", "rest"))
        segs = []
        for col_, mapping in (("abc", {"A": "Class A: top sellers", "B": "Class B", "C": "Class C"}),
                              ("demand_type", {"regular": "Regular sellers", "intermittent": "Intermittent sellers"}),
                              ("season", {"rest": "Apr to Aug", "peak": "Sep to Nov (peak)"})):
            for key in mapping:
                d = f2[f2[col_] == key]
                if len(d) and d["y"].sum() > 0:
                    segs.append((mapping[key], self.wape(d, mdl), self.wape(d, BASE)))
        gains = [(n_, a / b - 1) for n_, a, b in segs]
        bg, wg = min(gains, key=lambda v: v[1]), max(gains, key=lambda v: v[1])
        self.a_seg.say(f"Biggest gain: {bg[0]} ({spct(bg[1])} error). Smallest: {wg[0]} ({spct(wg[1])}).",
                       "Each pair compares the model (blue) with the simple method (orange) for one group of products or weeks.")
        self.a_seg.chart.grouped(t, [s[0] for s in segs], [s[1] for s in segs], [s[2] for s in segs], "Model", BASE_LABEL)

        mb = fin.groupby(fin["week"].dt.to_period("M")).apply(lambda d: self.bias(d, mdl), include_groups=False)
        self.a_bias.say(f"The forecast ran lowest in {mb.idxmin().strftime('%B %Y')} ({spct(mb.min())} against actual units).",
                        "Below zero (orange) = forecast under actual sales that month. Above zero (blue) = forecast over.")
        self.a_bias.chart.vbars(t, [p.strftime("%b") for p in mb.index], list(mb.values),
                                ["s2" if v < 0 else "s1" for v in mb.values], fmt=spct, signed=True)

        chg = (ps["wm"] / ps["wb"] - 1).clip(-0.6, 0.6)
        self.a_dist.say(f"{pct(share, 0)} of products are forecast better than with the simple method.",
                        "Each bar counts products. Blue, left of the dashed line = the model has the lower error for that product.")
        self.a_dist.chart.hist(t, chg.to_numpy(), np.linspace(-0.6, 0.6, 25), "Model error against the simple method, per product")

        # ----- products -----
        meta = self.preds.drop_duplicates("sku").set_index("sku")
        ps = ps.join(meta[["description", "abc"]])
        within = float((ps["fc"] / ps["units"]).between(0.8, 1.2).mean())
        self.p_scatter.say(f"{pct(within, 0)} of products have a forecast total within 20% of actual sales.",
                           "Each dot is one product over the whole test period. On the dashed line = perfect. "
                           "Below the line = forecast too low.")
        self.p_scatter.chart.scatter(t, ps["units"].to_numpy(), ps["fc"].clip(lower=1).to_numpy(),
                                     ["s1" if a < b else "s2" for a, b in zip(ps["wm"], ps["wb"])],
                                     "Actual units in the test period", "Forecast units in the test period")
        self.p_tab.say("Click a column to sort. Double-click a row to open the product.",
                       "Level = forecast units against actual units. Change = model error against the simple method (negative is better).")
        self._fill_ptable(ps, t)

        # ----- risk -----
        cv = fin.groupby("h").apply(lambda d: float(((d["y"] >= d["lo"]) & (d["y"] <= d["hi"])).mean()), include_groups=False)
        self.r_cov.say(f"The 80% range contained actual demand {pct(cov, 0)} of the time.",
                       "One bar per forecast distance. The closer to the dashed 80% line, the more honest the range.")
        self.r_cov.chart.vbars(t, [f"{h} wk ahead" for h in cv.index], list(cv.values), ["s1"] * len(cv),
                               target=0.8, target_label="Target: 80%", ymax=1.0, inside=True)
        sv = fin.groupby("abc").apply(lambda d: float((d["y"] <= d["hi"]).mean()), include_groups=False)
        serv = float((fin["y"] <= fin["hi"]).mean())
        self.r_serv.say(f"Stocking to the upper bound would have covered demand in {pct(serv, 0)} of product-weeks.",
                        "Share of weeks where actual demand was at or below the upper bound, by revenue class.")
        self.r_serv.chart.vbars(t, [f"Class {k}" for k in sv.index], list(sv.values), ["s1"] * len(sv),
                                target=0.9, target_label="Design level: 90%", ymax=1.0, inside=True)
        width = float(((fin["hi"] - fin["lo"]) / fin["scale"]).mean())
        self.r_note.say("Read the range as a planning tool")
        self.r_note.body.setText(
            f"<b>What it is.</b> For every product and week the model gives a low and a high value. Actual demand should "
            f"fall between them about 8 times out of 10.<br><br>"
            f"<b>How wide it is.</b> On average the range is {width:.1f} times the product's recent weekly demand. That is "
            f"wide, because single gift products sell erratically.<br><br>"
            f"<b>How it was checked.</b> The range was adjusted using mistakes from earlier forecast dates only, then "
            f"tested on later dates. It is the same for both forecast settings.")
        keys = ["gbm_log", "gbm_tweedie", BASE]
        self.r_bias.say("For total stock, pick the method whose level is closest to zero.",
                        "Forecast units against actual units for the same weeks. Left of zero = too low.")
        self.r_bias.chart.hbars(t, ["Model: best per-product", "Model: unbiased totals", BASE_LABEL],
                                [self.bias(fin, k_) for k_ in keys], ["s1" if k_ == mdl else "neutral" for k_ in keys],
                                fmt=lambda v: spct(v, 1), signed=True)
        self.fill_products()

    def _fill_ptable(self, ps: pd.DataFrame, t: dict):
        cols = ["Product", "Class", "Sold", "Forecast", "Level", "Model err.", "Simple err.", "Change"]
        tb = self.ptable
        tb.setSortingEnabled(False)
        tb.setRowCount(len(ps))
        tb.setColumnCount(len(cols))
        tb.setHorizontalHeaderLabels(cols)
        for r, (sku, row) in enumerate(ps.sort_values("units", ascending=False).iterrows()):
            lvl, chg = row["fc"] / row["units"] - 1, row["wm"] / row["wb"] - 1
            cells = [(str(row["description"]).title(), None), (row["abc"], None), (f"{row['units']:,.0f}", row["units"]),
                     (f"{row['fc']:,.0f}", row["fc"]), (spct(lvl), lvl), (pct(row["wm"], 0), row["wm"]),
                     (pct(row["wb"], 0), row["wb"]), (spct(chg), chg)]
            for c, (text, val) in enumerate(cells):
                it = QTableWidgetItem(text) if val is None else NumItem(text, float(val))
                it.setData(Qt.ItemDataRole.UserRole, sku)
                it.setToolTip(f"{cells[0][0]} ({sku})")
                it.setTextAlignment((Qt.AlignmentFlag.AlignLeft if c == 0 else Qt.AlignmentFlag.AlignRight)
                                    | Qt.AlignmentFlag.AlignVCenter)
                if c == 7:
                    it.setForeground(QColor(t["good"] if chg < 0 else t["bad"]))
                tb.setItem(r, c, it)
        tb.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        tb.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        tb.setSortingEnabled(True)

    def open_from_table(self, row: int, _col: int):
        self.sku = self.ptable.item(row, 0).data(Qt.ItemDataRole.UserRole)
        self.search.clear()
        self.fill_products()
        self.go(3)

    def fill_products(self):
        q = self.search.text().strip().lower()
        meta = self.preds.drop_duplicates("sku")
        meta = meta[meta["abc"].isin(self._on(self.b_abc)) & meta["demand_type"].isin(self._on(self.b_type))]
        meta = meta.assign(name=meta["description"].str.title()).sort_values("name")
        if q:
            meta = meta[meta["name"].str.lower().str.contains(q, regex=False)
                        | meta["sku"].str.lower().str.contains(q, regex=False)]
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

    def product_table(self) -> pd.DataFrame:
        p = self.preds[self.preds["sku"] == self.sku].sort_values("week")
        return pd.DataFrame({
            "Week": p["week"].dt.strftime("%Y-%m-%d"), "Weeks ahead": p["h"], "Actual": p["y"].round(0),
            "Forecast": p[self.model].round(1), "Low": p["lo"].round(1), "High": p["hi"].round(1),
            "Simple method": p[BASE].round(1),
            "Period": np.where(p["closed"], "closed", np.where(p["fold_type"] == "final", "final test", "selection"))})

    def show_product(self):
        t = THEMES[self.theme]
        if not self.sku:
            self.e_card.say("No product matches the search")
            return
        hist = self.panel[self.panel["sku"] == self.sku].sort_values("week")
        p = self.preds[self.preds["sku"] == self.sku].set_index("week").reindex(hist["week"])
        row = self.preds[self.preds["sku"] == self.sku].iloc[0]
        fin = self.filtered()
        fin = fin[fin["sku"] == self.sku]
        name = f"{str(row['description']).title()} ({self.sku})"
        if len(fin) and fin["y"].sum() > 0:
            wm, wb, bs = self.wape(fin, self.model), self.wape(fin, BASE), self.bias(fin, self.model)
            self.e_card.say(f"{name}: forecast error {pct(wm, 0)}, {'better' if wm < wb else 'worse'} than the simple method ({pct(wb, 0)}).",
                            f"Class {row['abc']}, {row['demand_type']} seller. {fin['y'].sum():,.0f} units sold in the test period, "
                            f"forecast level {spct(bs)}. Grey = actual, blue = forecast, light blue = 80% range, "
                            f"dashed line = start of the final test.")
        else:
            self.e_card.say(name, "No sales in the selected test weeks.")
        hs = p["h"].to_numpy()
        self.e_card.chart.lines(
            t, hist["week"],
            [dict(name="Actual units", color="actual", width=1.6, values=hist["units"].to_numpy(float)),
             dict(name="Forecast", color="s1", values=p[self.model].to_numpy(float))],
            band=(p["lo"].to_numpy(float), p["hi"].to_numpy(float)), mark=self.final_start,
            extra=lambda i: f"Made {int(hs[i])} wk ahead" if not np.isnan(hs[i]) else "")
        tb = self.product_table()
        self.table.setRowCount(len(tb))
        self.table.setColumnCount(len(tb.columns))
        self.table.setHorizontalHeaderLabels(list(tb.columns))
        for r, rec in enumerate(tb.itertuples(index=False)):
            for c, v in enumerate(rec):
                it = QTableWidgetItem(f"{v:,.0f}" if isinstance(v, float) else str(v))
                it.setTextAlignment((Qt.AlignmentFlag.AlignLeft if c == 0 else Qt.AlignmentFlag.AlignRight)
                                    | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, it)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def export_csv(self):
        if not self.sku:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export forecast", f"{self.sku}_forecast.csv", "CSV (*.csv)")
        if path:
            self.product_table().to_csv(path, index=False)

    def render_static(self):
        """Data and method page: does not depend on the filters."""
        t, d, des = THEMES[self.theme], self.m["data"], self.m["design"]
        v1 = d["raw_rows"] - d["rows_dropped_sheet_overlap"]
        v2 = v1 - d["cancellation_rows"] - d["cancellations_matched_to_a_sale"]
        v3 = v2 - d["rows_dropped_nonpositive_qty_or_price"]
        steps = [("Raw transaction rows", d["raw_rows"]), ("Duplicated period removed", v1),
                 ("Cancellations removed", v2), ("Zero or negative lines removed", v3),
                 ("Clean rows used", d["clean_rows"])]
        self.d_clean.say(f"{d['clean_rows']:,} of {d['raw_rows']:,} transaction rows were kept ({pct(d['clean_rows'] / d['raw_rows'], 0)}).",
                         f"{d['cancellations_matched_to_a_sale']:,} cancelled orders were matched to their original sale and both "
                         f"removed: {d['units_removed_as_cancelled_sales']:,} units that never shipped. Postage and fee lines were dropped last.")
        self.d_clean.chart.hbars(t, [s[0] for s in steps], [float(s[1]) for s in steps],
                                 ["neutral"] * (len(steps) - 1) + ["s1"], fmt=lambda x: f"{x / 1e6:.3f}M")
        folds = self.preds.drop_duplicates("fold").sort_values("fold")
        self.d_time.say(f"{des['folds_total']} forecast dates: {des['folds_selection']} to choose settings, {des['folds_final']} as the final test.",
                        "Each row is one forecast date. Grey = history the model could see. Blue = the 4 weeks it had to forecast. "
                        "Only the dark blue rows are reported in this app.")
        self.d_time.chart.timeline(t, [(int(c), k) for c, k in zip(folds["cutoff_t"], folds["fold_type"])], list(self.cal["week"]))
        imp = pd.Series(self.m["feature_importance"]).sort_values(ascending=False).head(10)
        self.d_imp.say(f"Recent sales averages carry {pct(float(imp.iloc[:2].sum()), 0)} of the model's signal.",
                       "Share of the model's total gain per input (unbiased-totals setting, final test). "
                       "This is why simple averages are hard to beat on this data.")
        self.d_imp.chart.hbars(t, [FEATURE_NAMES.get(k, k) for k in imp.index], list(imp.values), ["s1"] * len(imp),
                               fmt=lambda x: pct(x, 0))
        self.d_txt.say("How the forecast is made and tested")
        self.d_txt.body.setText(
            f"<b>Data.</b> Every transaction of a UK online gift retailer from {d['date_min']} to {d['date_max']} "
            f"(public data, UCI Online Retail II). {d['skus_selected']} products were selected from the first year only; "
            f"they made up {pct(d['selected_share_of_first_year_revenue'], 0)} of first-year revenue.<br><br>"
            f"<b>Model.</b> One gradient-boosting model (LightGBM) learns from all products at once. For a forecast made "
            f"h weeks ahead it only uses sales that were known h weeks earlier. Automated tests change the future and "
            f"confirm the inputs do not move.<br><br>"
            f"<b>Two settings.</b> \"Best per-product accuracy\" had the lowest error on the selection dates and is the "
            f"pre-chosen setting. \"Unbiased totals\" aims at the average and is shown so the trade-off is visible.<br><br>"
            f"<b>Limits.</b> Two years of data, one earlier Christmas, no stock levels and no promotion calendar. "
            f"Results will differ on other data.")

    # ================= theme =================
    def apply_theme(self):
        self.setStyleSheet(qss(THEMES[self.theme]))
        self.b_theme.setText("Light mode" if self.theme == "dark" else "Dark mode")
        self.render_static()
        self.refresh()
        self.go(self.stack.currentIndex())

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self.apply_theme()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    w = Window()
    if "--screenshot" in sys.argv:           # produce portfolio images without clicking through the app
        out = Path(sys.argv[sys.argv.index("--screenshot") + 1])
        w.show()
        app.processEvents()
        for i, (name, _) in enumerate(PAGES):
            w.go(i)
            app.processEvents()
            w.grab().save(str(out / f"{i + 1}_{name.lower().replace(' ', '_')}.png"))
        w.b_model[1].click()
        w.go(0)
        app.processEvents()
        w.grab().save(str(out / "7_summary_unbiased_totals.png"))
        w.b_model[0].click()
        w.toggle_theme()
        w.go(0)
        app.processEvents()
        w.grab().save(str(out / "8_executive_summary_dark.png"))
        return
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
