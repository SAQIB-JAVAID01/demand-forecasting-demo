"""Demand Forecasting Studio (PyQt6 desktop app).

Run:  python app/desktop.py
Reads the backtest output in data/processed and reports/metrics.json.
Every sentence and number on screen is computed from that output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QSettings, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QFileDialog, QFrame, QGraphicsOpacityEffect, QGridLayout,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                             QPushButton, QScrollArea, QSizePolicy, QSlider, QSplashScreen, QStackedWidget, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src.report import FEATURE_NAMES  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from brand import logo_icon, logo_pixmap, save_assets, splash_pixmap  # noqa: E402
from charts import Chart  # noqa: E402
from icons import icon  # noqa: E402
from live import LivePage  # noqa: E402
from models_pages import ModelLab, Projection  # noqa: E402
from src import inventory as INV  # noqa: E402
from src import kpis as K  # noqa: E402
from src.kpis import BASE, BASE_LABEL, BIAS_TARGET, SEASONAL  # noqa: E402

MODELS: dict[str, str] = {}         # forecast settings found in the backtest output, default first (filled on start)

# Design tokens. Same hues in both modes; dark mode lifts lightness and lowers saturation so nothing glows.
# Neutral surfaces carry 80% of the screen, one accent (blue) marks the model and the selected state,
# orange is always the simple method, green / red / amber are reserved for good / bad / warning.
THEMES = {
    "light": dict(page="#f5f6f8", surface="#ffffff", raised="#fafbfc", ink="#0f172a", ink2="#475569", muted="#8a94a6",
                  grid="#eef0f4", axis="#d6dbe3", border="#e4e7ec", s1="#2f6fed", s2="#f0883e", actual="#64748b",
                  band="#d9e6fd", neutral="#d3d9e2", chip="#f0f2f5", nav="#0b1220", navink="#94a0b4",
                  navsel="#18233a", navline="#1b2537", good="#139a52", bad="#e0464b", warn="#c97a06", sel="#e9f0fe",
                  tint="#f3f7ff", soft="#a8c3fa", track="#f0f2f5"),
    "dark": dict(page="#0b0e13", surface="#11151c", raised="#151a22", ink="#e6e8eb", ink2="#a3abb7", muted="#6e7887",
                 grid="#1b212a", axis="#2a313c", border="#1f2630", s1="#5b8def", s2="#f0915a", actual="#c3cad4",
                 band="#1a3560", neutral="#2b3441", chip="#1a2029", nav="#080a0f", navink="#8a94a3",
                 navsel="#151c28", navline="#151a22", good="#3fcf8e", bad="#f2555a", warn="#f5a524", sel="#18243a",
                 tint="#111a29", soft="#2e5bb8", track="#1a2029"),
}


def qss(t: dict) -> str:
    return f"""
    QWidget {{ background: {t['page']}; color: {t['ink']}; font-family: 'Inter', 'Segoe UI'; font-size: 10pt; }}
    QToolTip {{ background: {t['raised']}; color: {t['ink']}; border: 1px solid {t['border']}; padding: 6px 8px;
                border-radius: 6px; }}
    QScrollArea {{ border: none; }}
    #nav {{ background: {t['nav']}; border-right: 1px solid {t['navline']}; }}
    #nav QWidget {{ background: {t['nav']}; }}
    #nav QLabel {{ color: #ffffff; }}
    #nav QLabel#navsub {{ color: {t['navink']}; font-size: 8.5pt; }}
    #nav QLabel#navsec {{ color: #5d6878; font-size: 7.5pt; font-weight: 600; letter-spacing: 1px; }}
    #nav QLabel#navfoot {{ color: #5d6878; font-size: 7.5pt; }}
    #nav QPushButton {{ background: transparent; color: {t['navink']}; text-align: left; padding: 8px 12px;
                        border-radius: 8px; font-size: 9.8pt; font-weight: 500; }}
    #nav QPushButton:hover {{ background: {t['navsel']}; color: #ffffff; }}
    #nav QPushButton:checked {{ background: {t['navsel']}; color: #ffffff; font-weight: 600; }}
    QFrame#navcard, QFrame#navcard QWidget {{ background: #111a2b; }}
    QFrame#navcard {{ border: 1px solid #1d2840; border-radius: 12px; }}
    QFrame#navcard QLabel {{ color: #c8d1df; font-size: 8.5pt; }}
    QFrame#navcard QLabel#navval {{ color: #ffffff; font-weight: 600; }}
    QFrame#navcard QLabel#navsec {{ color: #6f7c90; }}
    #topbar, #topbar QWidget {{ background: {t['page']}; }}
    #topbar {{ border-bottom: 1px solid {t['border']}; }}
    QFrame#card {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 14px; }}
    QFrame#card QWidget {{ background: {t['surface']}; }}
    QFrame#hero {{ border: 1px solid {t['border']}; border-radius: 14px;
                   background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t['tint']}, stop:1 {t['surface']}); }}
    QFrame#hero QWidget {{ background: transparent; }}
    QFrame#card QPushButton:checked {{ background: {t['s1']}; color: #ffffff; font-weight: 600; }}
    QFrame#inner {{ background: {t['raised']}; border: 1px solid {t['border']}; border-radius: 12px; }}
    QFrame#inner QWidget {{ background: {t['raised']}; }}
    QLabel#kvs {{ font-size: 21pt; font-weight: 600; }}
    QDoubleSpinBox {{ background: {t['surface']}; color: {t['ink']}; border: 1px solid {t['border']}; border-radius: 6px;
                      padding: 3px 6px; }}
    QFrame#hero QComboBox, QFrame#hero QLineEdit {{ background: {t['surface']}; }}
    QFrame#hero QAbstractItemView, QFrame#hero QListView {{ background: {t['raised']}; color: {t['ink']}; }}
    QFrame#hero QPushButton {{ background: {t['surface']}; color: {t['ink']}; border: 1px solid {t['border']}; }}
    QFrame#hero QPushButton:hover {{ background: {t['chip']}; }}
    QFrame#hero QPushButton:checked {{ background: {t['ink']}; color: {t['surface']}; border-color: {t['ink']}; }}
    QFrame#hero QPushButton#primary {{ background: {t['s1']}; color: #ffffff; font-weight: 600; border: none; }}
    QSlider {{ background: transparent; }}
    QSlider::groove:horizontal {{ height: 4px; background: {t['grid']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {t['s1']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: #ffffff; border: 2px solid {t['s1']}; width: 12px; margin: -6px 0;
                                  border-radius: 8px; }}
    QLabel#title {{ font-size: 17pt; font-weight: 700; }}
    QLabel#sub {{ color: {t['ink2']}; font-size: 9.5pt; }}
    QLabel#flabel {{ color: {t['muted']}; font-size: 7.5pt; font-weight: 600; letter-spacing: 1px; }}
    QLabel#hero1 {{ font-size: 13.5pt; font-weight: 600; background: transparent; }}
    QLabel#hero2 {{ color: {t['ink2']}; font-size: 9.8pt; background: transparent; }}
    QLabel#kv, QWidget#kv {{ font-size: 27pt; font-weight: 600; }}
    QWidget#chip {{ font-size: 8.4pt; font-weight: 600; }}
    QLabel#heroBig {{ font-size: 16pt; font-weight: 700; background: transparent; }}
    QWidget#badge {{ font-size: 8pt; font-weight: 700; }}
    QLabel#kl, QWidget#kl {{ color: {t['ink2']}; font-size: 9pt; font-weight: 500; }}
    QLabel#kn, QWidget#kn {{ color: {t['muted']}; font-size: 8.8pt; }}
    QLabel#eyebrow {{ color: {t['muted']}; font-size: 7.5pt; font-weight: 600; letter-spacing: 1px; }}
    QLabel#info {{ color: {t['muted']}; font-size: 10pt; }}
    QLabel#h2 {{ font-size: 11.5pt; font-weight: 600; }}
    QLabel#cap {{ color: {t['muted']}; font-size: 8.8pt; }}
    QLabel#body {{ color: {t['ink2']}; font-size: 10pt; }}
    QPushButton {{ background: {t['chip']}; color: {t['ink']}; border: 1px solid transparent; border-radius: 8px;
                   padding: 5px 12px; font-weight: 500; }}
    QPushButton:hover {{ background: {t['grid']}; }}
    #topbar QPushButton {{ background: {t['surface']}; color: {t['ink2']}; border: 1px solid {t['border']};
                           border-radius: 14px; padding: 4px 12px; font-size: 9pt; }}
    #topbar QPushButton:hover {{ color: {t['ink']}; }}
    #topbar QPushButton:checked {{ background: {t['sel']}; color: {t['s1']}; border-color: {t['soft']}; font-weight: 600; }}
    #topbar QPushButton#seg:checked {{ background: {t['s1']}; color: #ffffff; border-color: {t['s1']}; }}
    QFrame#card QPushButton {{ background: {t['chip']}; color: {t['ink']}; }}
    QFrame#card QPushButton#primary {{ background: {t['s1']}; color: #ffffff; font-weight: 600; padding: 6px 16px; }}
    QLineEdit {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 8px; padding: 6px 10px; }}
    QLineEdit:focus {{ border-color: {t['s1']}; }}
    QComboBox {{ background: {t['surface']}; color: {t['ink']}; border: 1px solid {t['border']}; border-radius: 8px;
                 padding: 4px 10px; }}
    QComboBox QAbstractItemView {{ background: {t['raised']}; color: {t['ink']}; selection-background-color: {t['sel']};
                                   selection-color: {t['ink']}; border: 1px solid {t['border']}; }}
    QListWidget {{ background: {t['surface']}; border: none; outline: none; }}
    QListWidget::item {{ padding: 6px 8px; border-radius: 8px; }}
    QListWidget::item:hover {{ background: {t['raised']}; }}
    QListWidget::item:selected {{ background: {t['sel']}; color: {t['ink']}; }}
    QTableWidget {{ background: {t['surface']}; border: none; alternate-background-color: {t['raised']};
                    gridline-color: {t['grid']}; }}
    QTableWidget::item {{ padding: 4px; }}
    QTableWidget::item:selected {{ background: {t['sel']}; color: {t['ink']}; }}
    QHeaderView::section {{ background: {t['surface']}; color: {t['muted']}; border: none;
                            border-bottom: 1px solid {t['border']}; padding: 8px 6px; font-size: 8.5pt; font-weight: 600; }}
    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {t['axis']}; border-radius: 4px; min-height: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {t['axis']}; border-radius: 4px; min-width: 30px; }}
    """


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def spct(x: float, d: int = 0) -> str:
    return f"{100 * x:+.{d}f}%"


def chip_css(color: str) -> str:
    """Rounded status pill: the status colour on a 14% tint of itself."""
    c = QColor(color)
    return (f"background: rgba({c.red()},{c.green()},{c.blue()},0.14); color: {color}; border-radius: 10px; "
            f"padding: 3px 10px; font-size: 8.4pt; font-weight: 600;")


def label(text: str = "", name: str = "", wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


class Card(QFrame):
    """A card: category line, a short plain-language finding, an optional chart or text.

    The "how to read it" note lives in the (i) tooltip next to the category, which keeps the screen calm.
    """

    def __init__(self, eyebrow: str, chart_height: int = 0, text: bool = False):
        super().__init__()
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(20, 16, 20, 16)
        self.lay.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(label(eyebrow.upper(), "eyebrow"))
        self.info = label("ⓘ", "info")
        self.info.setCursor(Qt.CursorShape.WhatsThisCursor)
        head.addWidget(self.info)
        head.addStretch(1)
        self.head = head
        self.lay.addLayout(head)
        self.finding = label("", "h2", True)
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
        self.how = label("", "cap", True)        # kept for callers; the text is shown as a tooltip
        self.how.hide()

    def say(self, finding: str, how: str = ""):
        self.finding.setText(finding)
        self.how.setText(how)
        self.info.setToolTip(f"<div style='max-width:340px'>{how}</div>" if how else "")
        self.info.setVisible(bool(how))


class NumItem(QTableWidgetItem):
    """Table cell that shows formatted text but sorts by its number."""

    def __init__(self, text: str, value: float):
        super().__init__(text)
        self.value = value

    def __lt__(self, other):
        return self.value < getattr(other, "value", 0.0)


PAGES = [("Executive summary", "Forecast accuracy, value added and what it means for inventory"),
         ("Live replay", "Real-time replay of a year of real sales, with each forecast drawn before its week"),
         ("Forecast accuracy", "Where the forecast is strong and where it is weak"),
         ("Products", "How every product performed in the final test"),
         ("Product explorer", "History, forecast and range for one product"),
         ("Risk and stock", "How far actual demand can move from the forecast"),
         ("Model lab", "Statistical models, machine learning and an ensemble on the same rolling backtest"),
         ("Sales projection", "The next 26 weeks under what-if economic scenarios"),
         ("Data and method", "What went in and how it was tested")]
ICONS = ("summary", "live", "accuracy", "products", "explorer", "risk", "lab", "projection", "method")


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
        self.setWindowIcon(logo_icon())
        self.resize(1480, 940)
        self.settings = QSettings("Saqib Jawed", "Demand Forecasting Studio")
        self.theme = str(self.settings.value("theme", "dark"))     # dark on first launch, then the last choice
        if self.theme not in THEMES:
            self.theme = "dark"
        self.persist = self.animate = True
        self._fade = None
        self.preds = pd.read_parquet(C.PROCESSED_DIR / "predictions.parquet")
        self.panel = pd.read_parquet(C.PROCESSED_DIR / "panel.parquet")
        self.cal = pd.read_parquet(C.PROCESSED_DIR / "calendar.parquet")
        self.m = json.loads((C.REPORTS_DIR / "metrics.json").read_text())
        MODELS.clear()
        MODELS.update(K.settings_for(self.preds))
        self.model = next(iter(MODELS))                 # opens on "Unbiased totals"; the other setting is one click away
        ipath = C.REPORTS_DIR / "inventory.json"
        self.inv = json.loads(ipath.read_text()) if ipath.exists() else None
        self.themes = THEMES
        self.final_start = self.preds.loc[self.preds["fold_type"] == "final", "week"].min()
        self.final_end = self.preds["week"].max()
        self.sku = None
        self._ci_cache = {}
        self._head_cache = {}
        self._build()
        self.apply_theme()

    # ================= layout =================
    def _page(self) -> QVBoxLayout:
        area = QScrollArea()
        area.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(28, 22, 28, 28)
        lay.setSpacing(16)
        area.setWidget(inner)
        self.stack.addWidget(area)
        return lay

    @staticmethod
    def _row(lay, *widgets, stretch=None, weight: int = 0):
        row = QHBoxLayout()
        row.setSpacing(16)
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
        nav.setFixedWidth(244)
        n = QVBoxLayout(nav)
        n.setContentsMargins(16, 22, 16, 18)
        n.setSpacing(3)
        head = QHBoxLayout()
        head.setSpacing(11)
        mark = QLabel()
        mark.setPixmap(logo_pixmap(120))
        mark.setScaledContents(True)
        mark.setFixedSize(38, 38)
        head.addWidget(mark)
        brand = label("Demand Forecasting\nStudio")
        brand.setStyleSheet("font-size: 11pt; font-weight: 700;")
        head.addWidget(brand, 1)
        n.addLayout(head)
        n.addSpacing(4)
        n.addWidget(label("UK online retailer · 200 products", "navsub"))
        n.addSpacing(12)
        self.nav_group = QButtonGroup(self)
        self.nav_buttons = []
        sections = {0: "OVERVIEW", 2: "ANALYSIS", 6: "MODELS", 8: "METHOD"}
        for i, (name, _) in enumerate(PAGES):
            if i in sections:
                if i:
                    n.addSpacing(10)
                n.addWidget(label(sections[i], "navsec"))
                n.addSpacing(2)
            b = QPushButton(name)
            b.setCheckable(True)
            b.setChecked(i == 0)
            b.setIconSize(QSize(17, 17))
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=i: self.go(k))
            self.nav_group.addButton(b, i)
            self.nav_buttons.append(b)
            n.addWidget(b)
        n.addStretch(1)
        snap = QFrame()
        snap.setObjectName("navcard")
        sl = QGridLayout(snap)
        sl.setContentsMargins(14, 12, 14, 12)
        sl.setHorizontalSpacing(8)
        sl.setVerticalSpacing(7)
        sl.addWidget(label("FINAL TEST SNAPSHOT", "navsec"), 0, 0, 1, 2)
        self.snap = []
        for r_, text in enumerate(("Weekly-total accuracy", "vs benchmark", "Forecast bias", "Sim. fill rate"), 1):
            v = label("", "navval")
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            sl.addWidget(label(text), r_, 0)
            sl.addWidget(v, r_, 1)
            self.snap.append(v)
        n.addWidget(snap)
        n.addSpacing(10)
        self.b_theme = QPushButton("Dark mode")
        self.b_theme.setIconSize(QSize(17, 17))
        self.b_theme.setCursor(Qt.CursorShape.PointingHandCursor)
        self.b_theme.clicked.connect(self.toggle_theme)
        n.addWidget(self.b_theme)
        n.addSpacing(8)
        n.addWidget(label("Public data · UCI Online Retail II", "navfoot"))
        outer.addWidget(nav)

        right = QWidget()
        r = QVBoxLayout(right)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(0)
        top = QFrame()
        top.setObjectName("topbar")
        tl = QVBoxLayout(top)
        tl.setContentsMargins(28, 18, 28, 14)
        tl.setSpacing(4)
        self.title, self.subtitle = label("", "title"), label("", "sub")
        tl.addWidget(self.title)
        tl.addWidget(self.subtitle)
        tl.addSpacing(8)
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
        hl.setContentsMargins(24, 20, 24, 20)
        hl.setSpacing(6)
        self.hero1, self.hero2 = label("", "heroBig", True), label("", "hero2", True)
        self.hero1.setTextFormat(Qt.TextFormat.RichText)
        hl.addWidget(self.hero1)
        hl.addWidget(self.hero2)
        lay.addWidget(hero)
        self.k = self._kpi_row(lay, ("Weekly-total accuracy (1–4 wk)", "Value added vs benchmark", "Forecast bias",
                                     "Product-level accuracy"))
        self.s_trade = Card("Two forecast settings, one trade-off", text=True)
        lay.addWidget(self.s_trade)
        self._build_inventory(lay)
        self.k2 = self._kpi_row(lay, ("Validated on", "Peak season (Nov 2011)", "A-item accuracy",
                                      "Inside the 80% interval"), small=True)
        self.s_total, self.s_methods = Card("Forecast vs actual", 290), Card("Forecast Value Added by segment", 290)
        self._row(lay, self.s_total, self.s_methods, stretch=(3, 2), weight=1)
        self.s_take, self.s_hist = Card("What this means for the business", text=True), Card("Two years of demand", 240)
        self._row(lay, self.s_take, self.s_hist, stretch=(2, 3), weight=1)

        # ----- live replay (real-time charts) -----
        lay = self._page()
        self.live = LivePage(self, C.PROCESSED_DIR / "transactions.parquet", BASE, BASE_LABEL, THEMES)
        lay.addWidget(self.live, 1)

        # ----- accuracy -----
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
        self._row(lay, self.p_scatter, self.p_tab, stretch=(5, 7), weight=1)

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

        # ----- model lab and sales projection -----
        lay = self._page()
        self.lab = ModelLab(self, lay, Card, label, chip_css, make_table)
        lay = self._page()
        self.proj = Projection(self, lay, Card, label, chip_css)

        # ----- 6 method -----
        lay = self._page()
        self.d_clean, self.d_time = Card("Data cleaning", 290), Card("Test design", 290)
        self._row(lay, self.d_clean, self.d_time, weight=1)
        self.d_imp, self.d_txt = Card("What drives the forecast", 300), Card("Method in plain words", text=True)
        self._row(lay, self.d_imp, self.d_txt, weight=1)

    def _kpi_row(self, lay, titles, small=False) -> list:
        out = []
        krow = QHBoxLayout()
        krow.setSpacing(16)
        for title in titles:
            f = QFrame()
            f.setObjectName("card")
            l = QVBoxLayout(f)
            l.setContentsMargins(20, 14 if small else 16, 20, 14 if small else 16)
            l.setSpacing(4)
            val, note, chip = label("", "kvs" if small else "kv"), label("", "kn", True), label("")
            l.addWidget(label(title, "kl"))
            l.addWidget(val)
            l.addWidget(chip, 0, Qt.AlignmentFlag.AlignLeft)
            l.addSpacing(2)
            l.addWidget(note)
            l.addStretch(1)
            krow.addWidget(f)
            out.append((val, note, chip))
        lay.addLayout(krow)
        return out

    def _build_inventory(self, lay):
        """Inventory planning card: one replenishment policy run with the forecast and with the benchmark."""
        f = QFrame()
        f.setObjectName("card")
        v = QVBoxLayout(f)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)
        head = QHBoxLayout()
        head.addWidget(label("INVENTORY PLANNING · SIMULATED, SAME POLICY MODEL VS BASELINE", "eyebrow"))
        self.inv_info = label("ⓘ", "info")
        head.addWidget(self.inv_info)
        head.addStretch(1)
        v.addLayout(head)
        self.inv_find = label("", "h2", True)
        v.addWidget(self.inv_find)
        ctl = QHBoxLayout()
        ctl.setSpacing(6)
        self.inv_lead, self.inv_serv = [], []
        for title, opts, store, default in (("LEAD TIME", [(1, "1 wk"), (2, "2 wk"), (3, "3 wk"), (4, "4 wk")], self.inv_lead, 1),
                                            ("SERVICE TARGET", [(0.90, "90%"), (0.95, "95%"), (0.98, "98%")], self.inv_serv, 0.95)):
            ctl.addWidget(label(title, "flabel"))
            grp = QButtonGroup(self)
            for val, text in opts:
                b = QPushButton(text)
                b.setCheckable(True)
                b.setChecked(val == default)
                b.setProperty("value", val)
                b.clicked.connect(self.refresh_inventory)
                grp.addButton(b)
                ctl.addWidget(b)
                store.append(b)
            ctl.addSpacing(14)
        self.inv_cost, self.inv_hold = QSlider(Qt.Orientation.Horizontal), QSlider(Qt.Orientation.Horizontal)
        for title, sl, lo, hi, val in (("UNIT COST, % OF PRICE", self.inv_cost, 30, 60, 45),
                                       ("HOLDING RATE / YR", self.inv_hold, 15, 35, 25)):
            ctl.addWidget(label(title, "flabel"))
            sl.setRange(lo, hi)
            sl.setValue(val)
            sl.setFixedWidth(120)
            sl.valueChanged.connect(self.refresh_inventory)
            ctl.addWidget(sl)
            vl = label("", "kn")
            vl.setMinimumWidth(34)
            sl.setProperty("lbl", vl)
            ctl.addWidget(vl)
            ctl.addSpacing(10)
        ctl.addStretch(1)
        v.addLayout(ctl)
        row = QHBoxLayout()
        row.setSpacing(12)
        self.inv_k = []
        for title in ("Simulated fill rate", "Stock-out risk", "Inventory for the same service",
                      "Carrying-cost saving", "Lost sales avoided"):
            fr = QFrame()
            fr.setObjectName("inner")
            l = QVBoxLayout(fr)
            l.setContentsMargins(16, 12, 16, 12)
            l.setSpacing(3)
            val, chip, note = label("", "kvs"), label(""), label("", "kn", True)
            l.addWidget(label(title, "kl"))
            l.addWidget(val)
            l.addWidget(chip, 0, Qt.AlignmentFlag.AlignLeft)
            l.addWidget(note)
            l.addStretch(1)
            row.addWidget(fr)
            self.inv_k.append((val, chip, note))
        v.addLayout(row)
        self.inv_foot = label("", "kn", True)
        self.inv_foot.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self.inv_foot)
        lay.addWidget(f)

    def inventory_result(self) -> dict | None:
        """The precomputed simulation for the selected setting, lead time and service target, priced with the inputs."""
        if self.inv is None:
            return None
        lead = next(b.property("value") for b in self.inv_lead if b.isChecked())
        serv = next(b.property("value") for b in self.inv_serv if b.isChecked())
        run = K.find_run(self.inv, self.model, lead, serv)
        if run is None:
            return None
        return K.inventory_view(run, self.inv_cost.value() / 100, self.inv_hold.value() / 100)

    def refresh_inventory(self):
        t = THEMES[self.theme]
        for sl in (self.inv_cost, self.inv_hold):
            sl.property("lbl").setText(f"{sl.value()}%")
        r = self.inventory_result()
        if r is None:
            self.inv_find.setText("Run  python -m src.inventory  to fill this card.")
            return
        m, b, e = r["m"], r["b"], r["e"]
        df = m["fill_rate"] - b["fill_rate"]
        ds = m["units_short"] - b["units_short"]
        ic = r["inv_change"]
        self.inv_find.setText(
            f"With the same stock policy, the forecast fills {pct(m['fill_rate'])} of demand vs {pct(b['fill_rate'])} "
            f"with the {BASE_LABEL}, and needs {abs(100 * ic):.0f}% {'less' if ic < 0 else 'more'} stock for the same fill rate.")
        self.inv_info.setToolTip(
            "<div style='max-width:360px'>Weekly review, order-up-to policy, lost sales, orders arrive after the lead time. "
            f"The same policy runs once with the forecast and once with the {BASE_LABEL}, for all "
            f"{self.inv['products']} products over the {self.inv['weeks_scored']} final-test weeks "
            f"({self.inv['final_start']} to {self.inv['final_end']}). Money uses the unit cost and holding rate set here; "
            "both are assumptions. Recorded history, not a live store.</div>")
        c = r["cost"]
        vals = [
            (pct(m["fill_rate"]), f"{'+' if df >= 0 else '−'}{abs(100 * df):.1f} pts vs baseline", t["good"] if df >= 0 else t["bad"],
             f"Units served from stock. Baseline {pct(b['fill_rate'])}. Target {pct(r['service'], 0)} cycle service."),
            (f"{m['units_short']:,.0f}", f"{pct(m['share_short'])} of demand short", t["good"] if ds <= 0 else t["warn"],
             f"Units short over {m['weeks']} weeks; baseline {b['units_short']:,.0f} ({pct(b['share_short'])})."),
            (f"{100 * ic:+.1f}%", "Same fill rate as the baseline", t["good"] if ic < 0 else t["warn"],
             f"Average stock £{e['avg_inventory_value'] * c:,.0f} vs £{b['avg_inventory_value'] * c:,.0f} at cost."),
            (f"£{r['saving']:,.0f}/yr", f"{pct(r['hold'], 0)} holding rate", t["good"] if r["saving"] >= 0 else t["warn"],
             f"£{r['saving_lo']:,.0f} to £{r['saving_hi']:,.0f} at 15–35% holding; unit cost {pct(c, 0)} of price."),
            (f"£{r['lost_rev']:,.0f}/yr", f"£{r['lost_margin']:,.0f} margin", t["good"] if r["lost_rev"] >= 0 else t["warn"],
             "Revenue the baseline missed and the forecast served, same policy and service target."),
        ]
        for (val, chip, note), (v_, ct, col, n_) in zip(self.inv_k, vals):
            val.setText(v_)
            chip.setText(ct)
            chip.setStyleSheet(chip_css(col))
            note.setText(n_)
        run = r["run"]
        self.inv_foot.setText(
            f"<b>Safety stock</b> = z × σ × √(L + R): z = {run['z']:.2f} for {pct(r['service'], 0)}, L = {r['lead']} wk lead "
            f"time, R = 1 wk review, σ = spread of the last 13 weekly misses. Average safety stock "
            f"£{m['avg_safety_stock_value'] * c:,.0f} (baseline £{b['avg_safety_stock_value'] * c:,.0f}). "
            f"<b>Recommended cover</b> {m['weeks_of_cover']:.1f} weeks ({7 * m['weeks_of_cover']:.0f} days) of demand on hand "
            f"(baseline {b['weeks_of_cover']:.1f}). <b>Reorder point</b> = (L + R) × forecast + safety stock, checked weekly. "
            f"Same service needs z = {run['z_same_service']:.2f} with the forecast.")
        self.snap[3].setText(pct(m["fill_rate"]))

    # ================= state =================
    def go(self, i: int):
        changed = i != self.stack.currentIndex()
        self.stack.setCurrentIndex(i)
        if changed and self.animate and self.isVisible():
            page = self.stack.currentWidget()
            eff = QGraphicsOpacityEffect(page)
            page.setGraphicsEffect(eff)
            anim = QPropertyAnimation(eff, b"opacity", self)
            anim.setDuration(240)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.finished.connect(lambda pg_=page: pg_.setGraphicsEffect(None))
            anim.start()
            self._fade = anim
        self.live.activate(PAGES[i][0] == "Live replay")
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

        # ----- headline (no model names on this page) -----
        acc, acc_b = 1 - wm, 1 - wb
        fva = wb - wm                                       # product-level value added, accuracy points
        fva_s = self.wape(fin, SEASONAL) - wm
        hk = (tuple(self._on(self.b_abc)), tuple(self._on(self.b_type)), tuple(self._on(self.b_h)))
        if hk not in self._head_cache:                      # both settings, for the tiles and the trade-off card
            self._head_cache[hk] = {k_: K.headline(fin, k_) for k_ in MODELS}
        HALL = self._head_cache[hk]
        H = HALL[mdl]
        fva_lo, fva_hi = H["product_ci"]
        weeks_hist = self.cal["week"].nunique()
        n_p = fin["sku"].nunique()
        good = t["good"]
        pts = lambda x: f"{'+' if x >= 0 else '−'}{abs(100 * x):.1f} pts"
        other = next(k for k in MODELS if k != mdl)
        bo = self.bias(fin, other)
        vcol = {"better": good, "worse": t["bad"], "level": t["s1"]}
        vtxt = {"better": "Better than the benchmark", "worse": "Below the benchmark", "level": "Level with the benchmark"}
        b_ok = abs(bs) <= BIAS_TARGET
        if b_ok:
            h1 = (f"Out of sample, weekly totals are <span style='color:{good}'>{pct(H['total_acc'])} accurate</span> and "
                  f"the forecast level is <span style='color:{good}'>within {abs(100 * bs):.1f}%</span> of the units sold.")
        else:
            h1 = (f"Out of sample, weekly totals are {pct(H['total_acc'])} accurate; this setting runs "
                  f"<span style='color:{t['warn']}'>{abs(100 * bs):.0f}% {'low' if bs < 0 else 'high'}</span> on totals "
                  f"because it aims at the typical week, and adds <span style='color:{good}'>{pts(fva)}</span> per product.")
        h2 = (f"The {BASE_LABEL} benchmark: {pct(H['total_acc_base'])} weekly-total accuracy, {spct(H['bias_base'], 1)} bias. "
              f"\"{MODELS[other]}\" is one click away at the top. ")
        self.hero1.setText(h1)
        self.hero2.setText(f"{h2}Units, weekly, 1–4 weeks ahead, {n_p} products, rolling backtest "
                           f"{self.final_start:%d %b %Y} to {self.final_end:%d %b %Y}: weeks never used to build or tune the model.")

        g = fin.groupby("week")[["y", mdl, BASE]].sum()
        te = float((g[mdl] - g["y"]).abs().sum() / g["y"].sum())
        tb_ = float((g[BASE] - g["y"]).abs().sum() / g["y"].sum())
        ps = fin.groupby("sku").apply(lambda d: pd.Series({
            "units": d["y"].sum(), "fc": d[mdl].sum(),
            "wm": self.wape(d, mdl) if d["y"].sum() > 0 else np.nan,
            "wb": self.wape(d, BASE) if d["y"].sum() > 0 else np.nan}), include_groups=False).dropna()
        share = float((ps["wm"] < ps["wb"]).mean())
        cov_ok = abs(cov - 0.8) <= 0.05
        dt = tb_ - te
        fa = fin[fin["abc"] == "A"]
        rev = self.panel.groupby("sku")["revenue"].sum()
        a_skus = self.preds.loc[self.preds["abc"] == "A", "sku"].unique()
        a_share = float(rev.reindex(a_skus).sum() / rev.sum())
        tag = K.bias_tag(bs)
        inter = ("Accuracy = 1 − WAPE: total absolute error divided by total units, so big sellers count more. "
                 "Single gift products sell in small, irregular amounts with many zero weeks, so product-week error "
                 "is high for every method; the points added over the benchmark are what the forecast contributes.")
        t_lo, t_hi = H["total_ci"]
        db = abs(H["bias_base"]) - abs(bs)                  # how much closer to zero than the benchmark's level
        kpis = [
            (pct(H["total_acc"]), "Out-of-sample · rolling backtest", t["s1"],
             f"All products in the filter added up, week by week: the order-quantity view. "
             f"Benchmark ({BASE_LABEL}) {pct(H['total_acc_base'])}.",
             "Weekly-total accuracy = 1 − (Σ |forecast total − actual total| / Σ actual total) over the final-test "
             "weeks. Product errors that cancel out inside a week do not count, which is what matters for "
             "purchasing and capacity."),
            (pts(H["total_fva"]), vtxt[H["total_verdict"]], vcol[H["total_verdict"]],
             f"Weekly-total accuracy vs {BASE_LABEL}; 95% range {pts(t_lo)} to {pts(t_hi)}. "
             f"Per product: {pts(fva)} ({vtxt[H['product_verdict']].lower()}).",
             "Forecast Value Added = benchmark error − forecast error, in accuracy points, on the same weeks. "
             "\"Level\" means the 95% range includes zero, so the difference could be chance. The total range "
             "resamples whole weeks, the per-product range resamples whole products, 1,000 times each."),
            (spct(bs, 1), tag, good if b_ok else t["warn"],
             f"Forecast vs actual units, target ±5%. Benchmark {spct(H['bias_base'], 1)}"
             + (f": {pts(db)[:-4]} points closer to zero." if db > 0 else "."),
             "Forecast bias = Σ(forecast − actual) / Σ actual. Below zero = under-forecast. A setting that aims at "
             "the typical (median) week is low on totals by design, because occasional bulk orders lift the average "
             "above the typical week."),
            (pct(acc), f"{pts(fva)} vs {BASE_LABEL}", vcol[H["product_verdict"]],
             f"Each product, each week. 95% range {pts(fva_lo)} to {pts(fva_hi)}; {pts(fva_s)} vs same week last year.",
             inter),
        ]
        if len(fa) and fa["y"].sum() > 0:
            fa_m, fa_b = self.wape(fa, mdl), self.wape(fa, BASE)
            a_item = (pct(1 - fa_m), f"{pts(fa_b - fa_m)} vs {BASE_LABEL}", good if fa_b >= fa_m else t["bad"],
                      f"{fa['sku'].nunique()} class-A products, {pct(a_share, 0)} of revenue, per product-week.")
        else:
            a_item = ("–", "Class A not in the filter", t["muted"], "Select class A at the top.")
        for (val, note, chip), (v_, c_text, c_col, n_, tip) in zip(self.k, kpis):
            val.setText(v_)
            chip.setText(c_text)
            chip.setStyleSheet(chip_css(c_col))
            note.setToolTip(f"<div style='max-width:320px'>{tip}</div>" if tip else "")
            note.setText(n_)

        fn = fin[fin["week"].dt.month == 11]
        if len(fn) and fn["y"].sum() > 0:
            pk_m, pk_b = self.wape(fn, mdl), self.wape(fn, BASE)
            peak = (pct(1 - pk_m), f"{pts(pk_b - pk_m)} vs {BASE_LABEL}", good if pk_b >= pk_m else t["bad"],
                    f"{fn['week'].nunique()} weeks of the Christmas run-up. December is after the final test.")
        else:
            peak = ("–", "Not in the filter", t["muted"], "")
        k2 = [
            (f"{n_p} products", "Rolling origin · 13 forecast dates", t["s1"],
             f"{weeks_hist} weeks of history · {len(fin):,} out-of-sample forecasts, 1–4 weeks ahead."),
            peak,
            a_item,
            (pct(cov, 0), "● On the 80% target" if cov_ok else ("↑ Above the 80% target" if cov > 0.8 else "↓ Below the 80% target"),
             good if cov_ok else t["warn"], "Share of actual weeks inside the 80% prediction interval."),
        ]
        for (val, note, chip), (v_, c_text, c_col, n_) in zip(self.k2, k2):
            val.setText(v_)
            chip.setText(c_text)
            chip.setStyleSheet(chip_css(c_col))
            note.setText(n_)
        self.snap[0].setText(pct(H["total_acc"]))
        self.snap[1].setText(pts(H["total_fva"]))
        self.snap[2].setText(spct(bs, 1))
        self.refresh_inventory()

        # ----- the two settings side by side -----
        def cell(text, bold=False, right=True, color=None):
            style = f"color:{color};" if color else ""
            return (f"<td align='{'right' if right else 'left'}' style='padding:5px 14px 5px 0;{style}'>"
                    f"{'<b>' if bold else ''}{text}{'</b>' if bold else ''}</td>")
        head = "".join(cell(x, True, i > 0) for i, x in enumerate(
            ("Setting", "Weekly-total accuracy", "Forecast bias", "Product-level accuracy", "Per product vs benchmark")))
        body = ""
        for k_ in list(MODELS) + [BASE]:
            if k_ == BASE:
                h_ = HALL[mdl]
                vals_ = (h_["total_acc_base"], h_["bias_base"], h_["product_acc_base"], None)
                name = f"{BASE_LABEL} (benchmark)"
            else:
                h_ = HALL[k_]
                vals_ = (h_["total_acc"], h_["bias"], h_["product_acc"], h_["product_fva"])
                name = MODELS[k_] + ("  ·  shown now" if k_ == mdl else "")
            body += "<tr>" + cell(name, k_ == mdl, False, t["s1"] if k_ == mdl else None) + cell(pct(vals_[0])) + \
                cell(spct(vals_[1], 1), color=good if abs(vals_[1]) <= BIAS_TARGET else t["warn"]) + cell(pct(vals_[2])) + \
                cell("–" if vals_[3] is None else f"{pts(vals_[3])} ({vtxt[h_['product_verdict']].split()[0].lower()})",
                     color=None if vals_[3] is None else vcol[h_["product_verdict"]]) + "</tr>"
        self.s_trade.say("No setting wins both: one gets the totals right, the other is closer product by product.",
                         "Both settings come from the same rolling backtest and the same final-test weeks. The benchmark "
                         "row is the 13-week average. \"Level\" means the 95% range of the difference includes zero.")
        self.s_trade.body.setText(
            f"<table cellspacing='0' cellpadding='0'><tr>{head}</tr>{body}</table><br>"
            + "<br>".join(f"<b>{MODELS[k_]}.</b> {K.SETTING_WHY.get(k_, '')} <i>Use for: "
                          f"{K.SETTING_USE.get(k_, '').lower()}.</i>" for k_ in MODELS if k_ in K.SETTING_WHY))

        # ----- summary charts -----
        self.s_total.say(f"Weekly total: {pct(1 - te, 0)} accurate; the {BASE_LABEL} {pct(1 - tb_, 0)}.",
                         "Grey = units actually sold. Blue = forecast made 1 to 4 weeks earlier. Orange = simple method. "
                         "Move the mouse over the chart for values.")
        self.s_total.chart.lines(t, g.index, [
            dict(name="Actual units", color="actual", values=g["y"].to_numpy()),
            dict(name="Forecast", color="s1", values=g[mdl].to_numpy()),
            dict(name=BASE_LABEL, color="s2", width=1.8, values=g[BASE].to_numpy())], area="Actual units")

        f2 = fin.assign(season=np.where(fin["week"].dt.month >= 9, "peak", "rest"))
        segs = []
        for col_, mapping in (("abc", {"A": "Class A", "B": "Class B", "C": "Class C"}),
                              ("demand_type", {"regular": "Regular sellers", "intermittent": "Intermittent (Croston-type)"}),
                              ("season", {"peak": "Sep to Nov (peak)", "rest": "Apr to Aug"}),
                              ("h", {1: "1 wk ahead", 4: "4 wk ahead"})):
            for key in mapping:
                d = f2[f2[col_] == key]
                if len(d) and d["y"].sum() > 0:
                    segs.append((mapping[key], self.wape(d, BASE) - self.wape(d, mdl)))
        best = max(segs, key=lambda v: v[1])
        neg = [n_ for n_, v in segs if v < 0]
        self.s_methods.say(f"Largest value added: {best[0]} ({pts(best[1])})." +
                           (f" Below the benchmark: {', '.join(neg)}." if neg else " Positive in every segment."),
                           f"Accuracy points added over the {BASE_LABEL} in each segment: benchmark WAPE − forecast WAPE. "
                           "Green = the forecast adds accuracy, red = it loses accuracy there.")
        self.s_methods.chart.hbars(t, [n_ for n_, _ in segs], [v for _, v in segs],
                                   ["good" if v >= 0 else "bad" for _, v in segs],
                                   fmt=lambda v: f"{'+' if v >= 0 else '−'}{abs(100 * v):.1f}", signed=True)

        r_inv = self.inventory_result()
        self.s_take.say("Three things to take away")
        inv_txt = (f"With the same policy at a {pct(r_inv['service'], 0)} target and {r_inv['lead']}-week lead time, the "
                   f"forecast needs {abs(100 * r_inv['inv_change']):.0f}% {'less' if r_inv['inv_change'] < 0 else 'more'} stock "
                   f"for the same fill rate and avoids about £{r_inv['lost_rev']:,.0f} a year of lost sales "
                   f"(simulation, assumptions in the card above)." if r_inv else "Run the inventory simulation to see the effect.")
        if H["total_verdict"] == "level":
            t1 = (f"Weekly totals are {pct(H['total_acc'])} accurate, level with the {BASE_LABEL} ({pct(H['total_acc_base'])}; "
                  f"the 95% range of the difference includes zero).")
        else:
            t1 = (f"Weekly totals are {pct(H['total_acc'])} accurate, {pts(H['total_fva'])} "
                  f"{'over' if H['total_fva'] >= 0 else 'against'} the {BASE_LABEL} ({pct(H['total_acc_base'])}).")
        t2 = (f"The forecast level is {spct(bs, 1)} against units sold; the benchmark is {spct(H['bias_base'], 1)}."
              if b_ok else
              f"This setting is {spct(bs, 1)} on totals because it aims at the typical week, not the average. "
              f"For total stock use \"{MODELS[other]}\" ({spct(bo, 1)}).")
        self.s_take.body.setText(
            f"<b>1. Totals.</b> {t1} {t2}<br><br>"
            f"<b>2. Products.</b> Per product-week this setting is {pct(acc)} accurate ({pts(fva)} vs the benchmark, "
            f"better on {pct(share, 0)} of products); \"{MODELS[other]}\" is {pct(HALL[other]['product_acc'])} "
            f"({pts(HALL[other]['product_fva'])}). Single gift products sell in small, irregular amounts, so "
            f"product-week error is high for every method.<br><br>"
            f"<b>3. Inventory.</b> {inv_txt} The 80% interval held {pct(cov, 0)} of the time.")

        tot = self.panel.groupby("week")["units"].sum()
        self.s_hist.say(f"Demand peaks before Christmas. The busiest week was {tot.idxmax():%d %b %Y} with {tot.max():,.0f} units.",
                        "All 200 products. The shaded area is the final test period that every other chart reports on.")
        self.s_hist.chart.lines(t, tot.index, [dict(name="Units sold per week", color="actual", values=tot.to_numpy())],
                                shade=(self.final_start, self.final_end, "Final test period"), area="Units sold per week")

        # ----- accuracy -----
        hh = fin.groupby("h").apply(lambda d: pd.Series({"m": self.wape(d, mdl), "b": self.wape(d, BASE)}), include_groups=False)
        if len(hh) > 1:
            self.a_h.say(f"Error is lowest {hh['m'].idxmin()} wk ahead ({pct(hh['m'].min())}) and highest "
                         f"{hh['m'].idxmax()} wk ahead ({pct(hh['m'].max())}).", "Blue = model, orange = simple method. Lower is better.")
        else:
            self.a_h.say(f"Error {hh.index[0]} wk ahead is {pct(hh['m'].iloc[0])}.", "Blue = model, orange = simple method.")
        mae = lambda c: float((fin[c] - fin["y"]).abs().mean())
        rmse = lambda c: float(np.sqrt(((fin[c] - fin["y"]) ** 2).mean()))
        self.a_h.how.setText(self.a_h.how.text() + (
            f" All product-weeks in the filter: WAPE {pct(wm)} vs {pct(wb)}, MAE {mae(mdl):.1f} vs {mae(BASE):.1f} units, "
            f"RMSE {rmse(mdl):.1f} vs {rmse(BASE):.1f} units (forecast vs {BASE_LABEL}). Product-week error is noisy for "
            "single gift products with many zero weeks; MAPE is not shown because it is undefined in weeks with no sales."))
        self.a_h.info.setToolTip(f"<div style='max-width:340px'>{self.a_h.how.text()}</div>")
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
        keys = list(MODELS) + [BASE]
        self.r_bias.say("For total stock, pick the method whose level is closest to zero.",
                        "Forecast units against actual units for the same weeks. Left of zero = too low.")
        self.r_bias.chart.hbars(t, [f"Setting: {MODELS[k_].lower()}" for k_ in MODELS] + [BASE_LABEL],
                                [self.bias(fin, k_) for k_ in keys], ["s1" if k_ == mdl else "neutral" for k_ in keys],
                                fmt=lambda v: spct(v, 1), signed=True)
        self.fill_products()
        self.lab.refresh(t)
        self.proj.draw()
        self.live.prepare()

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
        self.go(4)

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
            self.live.follow(self.sku)

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
            self.e_card.say(f"{name}: {pct(1 - wm, 0)} accurate, {'ahead of' if wm < wb else 'behind'} the 13-week average ({pct(1 - wb, 0)}).",
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
                                 ["neutral"] * (len(steps) - 1) + ["s1"], fmt=lambda x: f"{x:,.0f}")
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
            f"<b>Model.</b> Gradient-boosting models (LightGBM) learn from all products at once; the Model lab page "
            f"compares them with statistical models and an ensemble. For a forecast made "
            f"h weeks ahead it only uses sales that were known h weeks earlier. Automated tests change the future and "
            f"confirm the inputs do not move.<br><br>"
            f"<b>Two settings.</b> \"Unbiased totals\" aims at the average week, so forecasts add up to the units "
            f"sold; the app opens on it because totals drive purchasing. \"Best per-product accuracy\" had the lowest "
            f"product-week error on the selection dates"
            f"{' (an ensemble of the three best models there)' if 'ensemble' in MODELS else ''}; it aims at the "
            f"typical week and is low on totals by design. Both were fixed before the final test was scored.<br><br>"
            f"<b>Limits.</b> Two years of data, one earlier Christmas, no stock levels and no promotion calendar. "
            f"Results will differ on other data.")

    # ================= theme =================
    def apply_theme(self):
        self.setStyleSheet(qss(THEMES[self.theme]))
        self.live.apply_theme(THEMES[self.theme])
        t = THEMES[self.theme]
        self.b_theme.setText("Light mode" if self.theme == "dark" else "Dark mode")
        self.b_theme.setIcon(icon("sun" if self.theme == "dark" else "moon", t["navink"], "#ffffff"))
        for b, kind in zip(self.nav_buttons, ICONS):
            b.setIcon(icon(kind, t["navink"], "#ffffff"))
        self.render_static()
        self.refresh()
        self.go(self.stack.currentIndex())

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        if self.persist:
            self.settings.setValue("theme", self.theme)
        self.apply_theme()


LIVE = 1                                     # index of the live replay page


def live_shots(w, app, out: Path):
    """Frames of the real-time page: all products, one product and a class (dark), and all products in light."""
    from catalog import full_page
    out.mkdir(parents=True, exist_ok=True)
    w.show()
    live = LIVE
    if w.theme != "dark":
        w.toggle_theme()
    w.go(live)
    top = w.preds.groupby("sku")["y"].sum().idxmax()          # best seller, for the one-product frame
    w.live.cb_prod.setCurrentIndex(max(w.live.cb_prod.findData(top), 0))
    for pos, scope, theme, name in ((30.45, 0, "dark", "2_live_replay.png"),
                                    (47.6, 6, "dark", "2b_live_replay_product.png"),
                                    (50.62, 1, "dark", "2c_live_replay_class_a.png"),
                                    (30.45, 0, "light", "2_live_replay_light.png")):
        if w.theme != theme:
            w.toggle_theme()
            w.go(live)
        w.live.cb_scope.setCurrentIndex(scope)
        w.live.snapshot(pos)
        for _ in range(3):
            app.processEvents()
        full_page(w).save(str(out / name))
    w.live.cb_scope.setCurrentIndex(0)
    if w.theme != "dark":
        w.toggle_theme()


def demo_frames(w, app, out: Path, fps: int = 30) -> int:
    """Frames for a short screen recording of the live page (dark): 8 s at 1x, then 14 s at 2x.

    The replay is stepped one animation frame per video frame, so the clip plays at the app's real speed.
    Near the end the hover read-out is shown on the main chart, as when the mouse rests on it.
    """
    from catalog import full_page
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    w.show()
    if w.theme != "dark":
        w.toggle_theme()
    w.go(LIVE)
    lv = w.live
    lv.cb_scope.setCurrentIndex(0)
    lv.timer.stop()
    lv.snapshot(29.62)
    lv.play(True)
    n, plan = 0, [(1, 8), (2, 14)]
    total = sum(sec for _, sec in plan) * fps
    for speed, sec in plan:
        for b in lv.b_speed:
            b.setChecked(b.property("value") == speed)
        lv._on_speed()
        for _ in range(sec * fps):
            lv._frame()
            if n > total - 5 * fps and lv._cur is not None:          # hover read-out, three weeks back
                vb = lv.p_stream.getPlotItem().vb
                k = max(lv._cur - 3, 0)
                lv.h_stream.move(vb.mapViewToScene(QPointF(lv.x[k], lv.S.y[k])))
            app.processEvents()
            pm = full_page(w)
            h = int(pm.width() * 0.75) // 2 * 2
            pm.copy(0, 0, pm.width() // 2 * 2, min(h, pm.height() // 2 * 2)).save(str(out / f"{n:05d}.jpg"), "JPG", 90)
            n += 1
    lv.h_stream.hide()
    return n


def load_fonts() -> str:
    """Register the bundled Inter font (SIL Open Font License, see app/fonts/OFL.txt)."""
    fams = set()
    for f in sorted((Path(__file__).resolve().parent / "fonts").glob("*.ttf")):
        fid = QFontDatabase.addApplicationFont(str(f))
        fams.update(QFontDatabase.applicationFontFamilies(fid))
    return "Inter" if "Inter" in fams else "Segoe UI"


def main():
    app = QApplication(sys.argv)
    font = QFont(load_fonts(), 10)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)
    app.setWindowIcon(logo_icon())
    batch = any(a in sys.argv for a in ("--screenshot", "--live-shots", "--catalog", "--demo", "--cover"))
    splash = None
    if not batch:
        splash = QSplashScreen(splash_pixmap())
        splash.show()
        app.processEvents()
    w = Window()
    if batch:                                # image modes never change the saved theme and skip the fades
        w.persist = w.animate = False
        if w.theme != "dark":
            w.toggle_theme()
    if "--screenshot" in sys.argv:           # produce portfolio images without clicking through the app
        out = Path(sys.argv[sys.argv.index("--screenshot") + 1])
        from catalog import full_page
        w.show()
        app.processEvents()
        for theme in ("dark", "light"):
            if w.theme != theme:
                w.toggle_theme()
            for i, (name, _) in enumerate(PAGES):
                w.go(i)
                if name == "Live replay":
                    w.live.snapshot(40.55)
                app.processEvents()
                app.processEvents()
                suffix = "" if theme == "dark" else "_light"
                full_page(w).save(str(out / f"{i + 1}_{name.lower().replace(' ', '_')}{suffix}.png"))
        w.toggle_theme()
        w.b_model[1].click()                 # the second setting, for the trade-off screenshot
        w.go(0)
        app.processEvents()
        full_page(w).save(str(out / "1b_summary_per_product_accuracy.png"))
        w.b_model[0].click()
        w.go(next(i for i, (n_, _) in enumerate(PAGES) if n_ == "Sales projection"))
        w.proj.pick("Recession")
        w.proj.draw()
        app.processEvents()
        full_page(w).save(str(out / "8b_sales_projection_recession.png"))
        w.proj.pick("Base")
        w.proj.draw()
        live_shots(w, app, out)
        return
    if "--demo" in sys.argv:                 # frames for a short screen recording of the live page
        n = demo_frames(w, app, Path(sys.argv[sys.argv.index("--demo") + 1]))
        print(f"{n} frames")
        return
    if "--cover" in sys.argv:                # 4:3 cover image for a portfolio or project catalog
        from catalog import cover
        cover(w, app, Path(sys.argv[sys.argv.index("--cover") + 1]))
        return
    if "--live-shots" in sys.argv:           # only the real-time page, light and dark
        live_shots(w, app, Path(sys.argv[sys.argv.index("--live-shots") + 1]))
        return
    if "--catalog" in sys.argv:              # 4:3 images for a portfolio or project catalog
        from catalog import export
        out = Path(sys.argv[sys.argv.index("--catalog") + 1])
        w.show()
        app.processEvents()
        export(w, app, out)
        save_assets(ROOT / "assets")
        return
    w.show()
    if splash is not None:
        splash.finish(w)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
