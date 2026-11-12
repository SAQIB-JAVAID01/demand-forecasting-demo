"""Live replay page for Demand Forecasting Studio.

A simulated live feed built from recorded history. The retailer's real sales are played back day by day
(from the cleaned transactions), and each forecast appears only once the date it was made has passed, so it
is always drawn ahead of "today". Nothing is fitted after the fact and every number comes from the backtest
output in data/processed.

Libraries: pyqtgraph draws the real-time charts (30 frames a second without blocking the window),
statsmodels gives the LOWESS trend line, scipy gives the robust anomaly scores and scikit-learn the
running error metrics.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QRectF, QSize, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QGradient, QIcon, QLinearGradient, QPainter, QPalette, QPixmap
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QCompleter, QFrame, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QPushButton, QSizePolicy, QSlider, QStyle, QStyledItemDelegate, QVBoxLayout,
                             QWidget)
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from statsmodels.nonparametric.smoothers_lowess import lowess

pg.setConfigOptions(antialias=True)

INTER = ("Accuracy = 1 - WAPE (weighted absolute percentage error). Single gift products sell in small, irregular amounts week to week, so per-product accuracy is low for every method; the gain over the benchmark is what the model adds.")
FRAME_MS = 33                 # about 30 frames a second
FRAMES_PER_WEEK = 360         # at 1x one replayed week takes about 12 s (one day every 1.7 s)
WARMUP_WEEKS = 6              # running scores are drawn once this many weeks have been checked
WEEK_S = 7 * 86400.0
TREND = "#8b5cf6"
SCOPES = [("all", "All products"), ("A", "Class A: top sellers"), ("B", "Class B"), ("C", "Class C"),
          ("regular", "Regular sellers"), ("intermittent", "Intermittent sellers"), ("one", "One product")]


def epoch(index) -> np.ndarray:
    return ((pd.DatetimeIndex(index) - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy(float)


def qcolor(hex_: str, alpha: int = 255) -> QColor:
    c = QColor(hex_)
    c.setAlpha(alpha)
    return c


def fade(hex_: str, alpha: int) -> QBrush:
    """Vertical gradient for areas under lines: colour at the top, transparent at the baseline."""
    g = QLinearGradient(0, 0, 0, 1)
    g.setCoordinateMode(QGradient.CoordinateMode.ObjectBoundingMode)
    g.setColorAt(0, qcolor(hex_, alpha))
    g.setColorAt(1, qcolor(hex_, 0))
    return QBrush(g)


def dot_icon(hex_: str) -> QIcon:
    pm = QPixmap(14, 14)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(hex_))
    p.drawEllipse(2, 2, 10, 10)
    p.end()
    return QIcon(pm)


def lab(text: str = "", name: str = "", wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


def fmt_units(v: float) -> str:
    if abs(v) >= 10_000:
        return f"{v / 1000:.0f}k"
    if abs(v) >= 1000:
        return f"{v / 1000:.1f}k"
    return f"{v:,.0f}"


class UnitsAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [fmt_units(v) for v in values]


class PercentAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [f"{100 * v:.0f}%" for v in values]


class FastText(QWidget):
    """One line of text painted directly (no QLabel).

    Changing the text only repaints this widget and never triggers a relayout of the page, which keeps the
    animation at full frame rate. Font comes from the style sheet (object name); colours are set in code.
    """

    def __init__(self, name: str, sample: str, opaque: bool = True):
        super().__init__()
        self.setObjectName(name)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.opaque = opaque
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, opaque)
        self.sample, self._text = sample, ""
        self.fg, self.bg = QColor("#0f172a"), QColor("#ffffff")

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        return QSize(fm.horizontalAdvance(self.sample), fm.height() + 6)

    def minimumSizeHint(self) -> QSize:
        return QSize(40, self.sizeHint().height())

    def changeEvent(self, e):
        if e.type() == QEvent.Type.FontChange:
            self.updateGeometry()
        super().changeEvent(e)

    def text(self) -> str:
        return self._text

    def setText(self, text: str):
        if text != self._text:
            self._text = text
            self.update()

    def set_colors(self, fg=None, bg=None):
        if fg is not None:
            self.fg = QColor(fg)
        if bg is not None:
            self.bg = QColor(bg)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        if self.opaque:
            p.fillRect(self.rect(), self.bg)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.setFont(self.font())
        p.setPen(self.fg)
        text = self.fontMetrics().elidedText(self._text, Qt.TextElideMode.ElideRight, self.width())
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
        p.end()


class Ticker(FastText):
    """A number that counts towards its new value instead of jumping."""

    def __init__(self, fmt):
        super().__init__("kv", "000,000.0%")
        self.fmt, self.cur, self.target, self._text = fmt, None, None, "–"

    def set(self, v: float, instant: bool = False):
        self.target = v
        if instant or self.cur is None or not np.isfinite(v) or not np.isfinite(self.cur):
            self.cur = v
            self._show()

    def step(self):
        if self.target is None or self.cur == self.target or not np.isfinite(self.target):
            return
        d = self.target - self.cur
        self.cur = self.target if abs(d) <= 1e-4 * max(1.0, abs(self.target)) else self.cur + 0.09 * d
        self._show()

    def _show(self):
        self.setText(self.fmt(self.cur) if self.cur is not None and np.isfinite(self.cur) else "–")


class Chip(FastText):
    """A small rounded status pill ("↓ 17% vs 13-week average"), painted directly like FastText."""

    def __init__(self, sample: str = "↓ 100% vs 13-week average  "):
        super().__init__("chip", sample)
        self.col = QColor("#8a94a6")

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        return QSize(fm.horizontalAdvance(self.sample) + 22, fm.height() + 10)

    def set(self, text: str, color: str):
        if text != self._text or QColor(color) != self.col:
            self._text, self.col = text, QColor(color)
            self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), self.bg)
        if not self._text:
            p.end()
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(self.font())
        fm = self.fontMetrics()
        w = min(fm.horizontalAdvance(self._text) + 18, self.width() - 1)
        r = QRectF(0.5, 2.5, w, self.height() - 5)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(self.col.name(), 34))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.setPen(self.col)
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, fm.elidedText(self._text, Qt.TextElideMode.ElideRight, int(w - 12)))
        p.end()


def set_style(w: QWidget, css: str):
    if w.styleSheet() != css:
        w.setStyleSheet(css)


def set_text(label: QLabel, text: str):
    """Change a label only when its text really changes (avoids needless relayouts)."""
    if label.text() != text:
        label.setText(text)


def make_plot(left=None, date_axis: bool = True) -> pg.PlotWidget:
    axes = {}
    if date_axis:
        axes["bottom"] = pg.DateAxisItem(orientation="bottom")
    if left is not None:
        axes["left"] = left
    w = pg.PlotWidget(axisItems=axes)
    w.setMouseEnabled(x=False, y=False)
    w.setMenuEnabled(False)
    w.hideButtons()
    opaque(w)
    return w


def opaque(w: pg.PlotWidget):
    """Charts paint their own background, so Qt need not repaint the cards behind them every frame."""
    w.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
    w.viewport().setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
    w.viewport().setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)


class Hover:
    """Crosshair and a small read-out box that follows the mouse over a live chart."""

    def __init__(self, plot: pg.PlotWidget, text_fn):
        self.plot, self.text_fn = plot, text_fn
        pi = plot.getPlotItem()
        self.line = pg.InfiniteLine(angle=90, movable=False)
        self.dot = pg.ScatterPlotItem(size=9)
        self.tip = pg.TextItem(html="", anchor=(0, 0))
        for it in (self.line, self.dot, self.tip):
            it.setZValue(200)
            it.setVisible(False)
            pi.addItem(it, ignoreBounds=True)
        plot.scene().sigMouseMoved.connect(self.move)
        plot.leaveEvent = lambda _e: self.hide()

    def style(self, t: dict):
        self.line.setPen(pg.mkPen(qcolor(t["ink2"], 150), width=1, style=Qt.PenStyle.DashLine))
        self.dot.setBrush(pg.mkBrush(t["ink"]))
        self.dot.setPen(pg.mkPen(t["surface"], width=2))
        self.tip.fill = pg.mkBrush(qcolor(t["raised"], 245))
        self.tip.border = pg.mkPen(t["axis"])
        self.t = t

    def hide(self):
        for it in (self.line, self.dot, self.tip):
            it.setVisible(False)

    def move(self, pos):
        vb = self.plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(pos):
            self.hide()
            return
        mp = vb.mapSceneToView(pos)
        got = self.text_fn(mp.x())
        if not got:
            self.hide()
            return
        x, y, html = got
        (x0, x1), (y0, y1) = vb.viewRange()
        self.line.setValue(x)
        if y is not None and np.isfinite(y):
            self.dot.setData([x], [y])
        else:
            self.dot.setData([], [])
        self.tip.setHtml(f"<div style='color:{self.t['ink']}; font-size:8.5pt; padding:2px 4px'>{html}</div>")
        right = (x - x0) > 0.6 * (x1 - x0)
        self.tip.setAnchor((1.04, 0) if right else (-0.04, 0))
        self.tip.setPos(x, y1)
        for it in (self.line, self.dot, self.tip):
            it.setVisible(True)


class LiveCard(QFrame):
    def __init__(self, eyebrow: str, widget: QWidget | None = None, min_h: int = 0, body: str = ""):
        super().__init__()
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(20, 16, 20, 16)
        self.lay.setSpacing(6)
        self.head = QHBoxLayout()
        self.head.setSpacing(6)
        self.head.addWidget(lab(eyebrow.upper(), "eyebrow"))
        self.info = lab("ⓘ", "info")
        self.info.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.head.addWidget(self.info)
        self.head.addStretch(1)
        self.lay.addLayout(self.head)
        self.finding = lab("", "h2", True)
        self.lay.addWidget(self.finding)
        if widget is not None:
            if min_h:
                widget.setMinimumHeight(min_h)
            self.lay.addWidget(widget, 1)
        self.how = lab(body, "body" if body else "cap", True)
        self.how.setVisible(bool(body))
        self.lay.addWidget(self.how)
        self.info.setVisible(False)

    def say(self, finding: str, how: str | None = None):
        set_text(self.finding, finding)
        if how is not None and self.how.isHidden():
            self.info.setToolTip(f"<div style='max-width:340px'>{how}</div>" if how else "")
            self.info.setVisible(bool(how))

class FeedDelegate(QStyledItemDelegate):
    """One feed row: coloured risk marker, product, what happened, and the size of the miss."""

    ROW = 60

    def __init__(self, page):
        super().__init__(page)
        self.page = page

    def sizeHint(self, option, index):
        return QSize(option.rect.width(), self.ROW)

    def paint(self, p: QPainter, option, index):
        d = index.data(Qt.ItemDataRole.UserRole)
        if not d:
            return
        t = self.page.themes[self.page.win.theme]
        r = QRectF(option.rect).adjusted(0, 3, -6, -3)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        p.setBrush(QColor(t["chip"] if hover else t["raised"]))
        p.drawRoundedRect(r, 10, 10)
        flash = index.data(Qt.ItemDataRole.UserRole + 1) or 0.0
        if flash > 0:
            p.setBrush(qcolor(t["s1"], int(80 * flash)))
            p.drawRoundedRect(r, 10, 10)
        col = QColor(t["bad"] if d["up"] else t["warn"])
        p.setBrush(col)
        p.drawRoundedRect(QRectF(r.left() + 10, r.top() + 12, 3, r.height() - 24), 1.5, 1.5)
        base = QFont(option.font)
        right_w = 96
        x0, w = r.left() + 24, r.width() - 24 - right_w - 8
        f1 = QFont(base)
        f1.setPointSizeF(9.6)
        f1.setWeight(QFont.Weight.DemiBold)
        p.setFont(f1)
        p.setPen(QColor(t["ink"]))
        fm = QFontMetrics(f1)
        p.drawText(QRectF(x0, r.top() + 9, w, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   fm.elidedText(d["title"], Qt.TextElideMode.ElideRight, int(w)))
        f2 = QFont(base)
        f2.setPointSizeF(8.4)
        p.setFont(f2)
        p.setPen(QColor(t["muted"]))
        fm2 = QFontMetrics(f2)
        p.drawText(QRectF(x0, r.top() + 30, w, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   fm2.elidedText(d["sub"], Qt.TextElideMode.ElideRight, int(w)))
        f3 = QFont(base)
        f3.setPointSizeF(11.5)
        f3.setWeight(QFont.Weight.DemiBold)
        p.setFont(f3)
        p.setPen(col)
        rx = r.right() - right_w - 10
        p.drawText(QRectF(rx, r.top() + 7, right_w, 24), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, d["big"])
        p.setFont(f2)
        p.setPen(QColor(t["muted"]))
        p.drawText(QRectF(rx, r.top() + 30, right_w, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, d["small"])
        p.restore()


# ===================================================================== data
class ReplayData:
    """Everything the page needs, prepared once; per scope only fast group-bys run."""

    def __init__(self, preds: pd.DataFrame, panel: pd.DataFrame, cal: pd.DataFrame, tx_path: Path):
        self.preds, self.panel = preds, panel
        self.weeks = pd.DatetimeIndex(sorted(preds["week"].unique()))
        self.x = epoch(self.weeks)
        wt = dict(zip(pd.DatetimeIndex(cal["week"]), cal["t"]))
        self.week_t = np.array([wt[w] for w in self.weeks], float)
        self.names = preds.drop_duplicates("sku").set_index("sku")["description"].astype(str).str.title()
        self.daily = self._daily(Path(tx_path))
        self._events: dict[str, pd.DataFrame] = {}

    def _daily(self, path: Path):
        """Units per product, week and weekday from the cleaned transactions (used to fill each week day by day)."""
        if not path.exists():
            return None
        tx = pd.read_parquet(path, columns=["sku", "date", "quantity"])
        tx = tx[tx["sku"].isin(set(self.names.index))]
        day = tx["date"].dt.normalize()
        tx = tx.assign(week=day - pd.to_timedelta(day.dt.dayofweek, unit="D"), dow=day.dt.dayofweek)
        tx = tx[tx["week"].isin(self.weeks)]
        return (tx.groupby(["sku", "week", "dow"])["quantity"].sum().unstack("dow", fill_value=0)
                .reindex(columns=range(7), fill_value=0))

    def events(self, model: str) -> pd.DataFrame:
        """Product-weeks where sales fell outside the 80% range by at least 2 robust standard deviations.

        The scale is the median absolute deviation (scipy) of that product's earlier misses only, so a flag
        never uses information from later weeks.
        """
        if model in self._events:
            return self._events[model]
        p = self.preds[~self.preds["closed"]].sort_values(["sku", "week"])
        out = []
        for sku, d in p.groupby("sku", sort=False):
            y, m, lo, hi = (d[c].to_numpy(float) for c in ("y", model, "lo", "hi"))
            r = y - m
            wk, hs = d["week"].to_numpy(), d["h"].to_numpy()
            for j in range(len(d)):
                if lo[j] <= y[j] <= hi[j]:
                    continue
                s = float(stats.median_abs_deviation(r[:j], scale="normal")) if j >= 6 else 0.0
                if not s > 0:
                    s = max((hi[j] - lo[j]) / 2.563, 1.0)      # width of a normal 80% range
                z = r[j] / s
                if abs(z) >= 2:
                    out.append((wk[j], sku, hs[j], y[j], m[j], z))
        ev = pd.DataFrame(out, columns=["week", "sku", "h", "y", "m", "z"])
        ev["p"] = 2 * stats.norm.sf(ev["z"].abs())
        self._events[model] = ev
        return ev

    def build(self, skus, race_skus, rows: pd.DataFrame, model: str, base: str, one: bool, hs) -> SimpleNamespace:
        S = SimpleNamespace(one=one)
        n = len(self.weeks)
        ph = self.panel[self.panel["sku"].isin(skus)]
        tot = ph.groupby("week")["units"].sum()
        ctx = tot[tot.index < self.weeks[0]].tail(26)
        S.ctx_x, S.ctx_y = epoch(ctx.index), ctx.to_numpy(float)
        S.y = tot.reindex(self.weeks).fillna(0).to_numpy(float)
        S.closed = ph.groupby("week")["closed"].all().reindex(self.weeks).fillna(False).to_numpy(bool)

        g = rows.groupby("week").agg(m=(model, "sum"), b=(base, "sum"), lo=("lo", "sum"), hi=("hi", "sum"),
                                     cut=("cutoff_t", "max")).reindex(self.weeks)
        S.m, S.b, S.lo, S.hi = (g[c].to_numpy(float) for c in ("m", "b", "lo", "hi"))
        S.cut = g["cut"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            S.ratio = np.where((S.m > 0) & ~S.closed & np.isfinite(S.m), S.y / S.m, np.nan)

        ok = rows[~rows["closed"]]
        S.rows = pd.DataFrame({"widx": self.weeks.get_indexer(ok["week"]), "sku": ok["sku"].to_numpy(),
                               "y": ok["y"].to_numpy(float), "m": ok[model].to_numpy(float),
                               "b": ok[base].to_numpy(float), "lo": ok["lo"].to_numpy(float),
                               "hi": ok["hi"].to_numpy(float)})
        r = S.rows
        c = (r.assign(em=(r["m"] - r["y"]).abs(), eb=(r["b"] - r["y"]).abs(),
                      inside=((r["y"] >= r["lo"]) & (r["y"] <= r["hi"])).astype(float), cnt=1.0)
             .groupby("widx")[["y", "m", "em", "eb", "inside", "cnt"]].sum().reindex(range(n)).fillna(0).cumsum())
        with np.errstate(divide="ignore", invalid="ignore"):
            S.wape_m = np.where(c["y"] > 0, c["em"] / c["y"], np.nan)
            S.wape_b = np.where(c["y"] > 0, c["eb"] / c["y"], np.nan)
            S.bias = np.where(c["y"] > 0, c["m"] / c["y"] - 1, np.nan)
            S.cov = np.where(c["cnt"] > 0, c["inside"] / c["cnt"], np.nan)
        S.cum_y, S.cum_m, S.cum_n = c["y"].to_numpy(), c["m"].to_numpy(), c["cnt"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            S.gain = np.where(S.wape_b > 0, 1 - S.wape_m / S.wape_b, np.nan)     # share of error removed
        # accuracy on the weekly total of the scope (what a planner orders against)
        ok_w = np.isfinite(S.m) & np.isfinite(S.b) & ~S.closed
        cy = np.cumsum(np.where(ok_w, S.y, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            S.tacc_m = np.where(cy > 0, 1 - np.cumsum(np.where(ok_w, np.abs(S.m - S.y), 0.0)) / cy, np.nan)
            S.tacc_b = np.where(cy > 0, 1 - np.cumsum(np.where(ok_w, np.abs(S.b - S.y), 0.0)) / cy, np.nan)
        checked = np.cumsum(np.diff(np.concatenate([[0.0], S.cum_n])) > 0)
        need = WARMUP_WEEKS * (2 if one else 1)          # one product is noisier, so it warms up longer
        S.warm = int(np.searchsorted(checked, need)) if checked[-1] >= need else n - 1

        # day-by-day fill of each week, from the real transactions of these products
        share = np.tile(np.arange(1, 8) / 7.0, (n, 1))
        if self.daily is not None:
            dd = self.daily[self.daily.index.get_level_values("sku").isin(skus)]
            dd = dd.groupby(level="week").sum().reindex(self.weeks).fillna(0).to_numpy(float)
            t7 = dd.sum(1, keepdims=True)
            share = np.where(t7 > 0, np.cumsum(dd, 1) / np.where(t7 > 0, t7, 1), share)
        S.cum = np.hstack([np.zeros((n, 1)), share])

        pw = (self.panel[self.panel["sku"].isin(race_skus) & self.panel["week"].isin(self.weeks)]
              .pivot_table(index="sku", columns="week", values="units", aggfunc="sum")
              .reindex(columns=self.weeks).fillna(0))
        S.race_skus, S.race_cum = pw.index.to_numpy(), pw.to_numpy(float).cumsum(1)

        ev = self.events(model)
        ev = ev[ev["sku"].isin(skus) & ev["h"].isin(hs)]
        S.events, S.ev_widx = ev, self.weeks.get_indexer(ev["week"])

        hi_top = np.nanmax(S.hi) if one and np.isfinite(S.hi).any() else 0
        m_top = np.nanmax(S.m) if np.isfinite(S.m).any() else 0
        S.ytop = max(float(S.ctx_y.max()) if len(S.ctx_y) else 0, float(S.y.max()), m_top, hi_top, 1.0) * 1.12
        wm_, wb_ = S.wape_m[S.warm:], S.wape_b[S.warm:]
        fin = np.concatenate([wm_[np.isfinite(wm_)], wb_[np.isfinite(wb_)], [0.3]])
        S.err_top = min(float(np.quantile(fin, 0.95)) * 1.2, 2.0)     # an early swing may run off the top
        S.err_bot = max(0.0, np.floor(float(np.min(fin[:-1] if len(fin) > 1 else fin)) * 0.8 * 10) / 10)
        return S


# ===================================================================== page
class LivePage(QWidget):
    def __init__(self, win, tx_path: Path, base: str, base_label: str, themes: dict):
        super().__init__()
        self.win, self.base, self.base_label, self.themes = win, base, base_label, themes
        self.data = ReplayData(win.preds, win.panel, win.cal, tx_path)
        self.n = len(self.data.weeks)
        self.x = self.data.x
        self.pos, self.speed, self.playing, self.frame = 0.0, 1, False, 0
        self._cur = self._done = None
        self._drawn = None
        self.race_state: dict[str, list] = {}
        self.race_text: dict[str, pg.TextItem] = {}
        self.race_val: dict[str, pg.TextItem] = {}
        self.race_target: dict[str, tuple] = {}
        self.race_xmax = 1.0
        self.s_xmax = self.s_goal = self.a_xmax = self.a_goal = None
        self.flash: dict[int, list] = {}
        self.S = None
        self.timer = QTimer(self)
        self.timer.setInterval(FRAME_MS)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)        # steady frame pacing on Windows
        self.timer.timeout.connect(self._frame)
        self._build()

    # ---------------------------------------------------------------- layout
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        ctl = QFrame()
        ctl.setObjectName("hero")
        cl = QVBoxLayout(ctl)
        cl.setContentsMargins(24, 18, 24, 14)
        cl.setSpacing(10)
        top = QHBoxLayout()
        top.setSpacing(8)
        txt = QVBoxLayout()
        txt.setSpacing(3)
        r1 = QHBoxLayout()
        r1.setSpacing(12)
        self.badge = FastText("badge", "●  LIVE REPLAY  ", opaque=False)
        self.l_week = lab("", "hero1")
        r1.addWidget(self.badge)
        r1.addWidget(self.l_week)
        r1.addStretch(1)
        txt.addLayout(r1)
        self.l_status = lab("", "hero2", True)
        txt.addWidget(self.l_status)
        top.addLayout(txt, 1)
        self.b_play = QPushButton("Play")
        self.b_play.setObjectName("primary")
        self.b_play.setMinimumWidth(92)
        self.b_play.clicked.connect(self.toggle_play)
        b_restart = QPushButton("Restart")
        b_restart.clicked.connect(self.restart)
        top.addWidget(self.b_play)
        top.addWidget(b_restart)
        cl.addLayout(top)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(lab("SPEED", "flabel"))
        self.b_speed, grp = [], QButtonGroup(self)
        for v in (0.5, 1, 2, 4):
            b = QPushButton(f"{v:g}x")
            b.setCheckable(True)
            b.setChecked(v == 1)
            b.setProperty("value", v)
            b.clicked.connect(self._on_speed)
            grp.addButton(b)
            bar.addWidget(b)
            self.b_speed.append(b)
        bar.addSpacing(14)
        bar.addWidget(lab("SCOPE", "flabel"))
        self.cb_scope = QComboBox()
        for key, text in SCOPES:
            self.cb_scope.addItem(text, key)
        self.cb_scope.currentIndexChanged.connect(self._on_scope)
        bar.addWidget(self.cb_scope)
        bar.addSpacing(8)
        bar.addWidget(lab("PRODUCT", "flabel"))
        self.cb_prod = QComboBox()
        self.cb_prod.setEditable(True)
        self.cb_prod.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.cb_prod.setMinimumWidth(300)
        comp = self.cb_prod.completer()
        comp.setFilterMode(Qt.MatchFlag.MatchContains)
        comp.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.cb_prod.activated.connect(self._on_product)
        bar.addWidget(self.cb_prod)
        bar.addStretch(1)
        self.l_day = FastText("kl", "Wednesday 28 Sep 2011   ", opaque=False)
        bar.addWidget(self.l_day)
        cl.addLayout(bar)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.n - 1)
        self.slider.valueChanged.connect(self.seek)
        cl.addWidget(self.slider)
        root.addWidget(ctl)

        # KPI tiles with sparklines
        krow = QHBoxLayout()
        krow.setSpacing(16)
        self.tiles = []
        fmts = [("Units this week", lambda v: f"{v:,.0f}"), ("Accuracy gain vs benchmark", lambda v: f"{100 * v:+.1f}%"),
                ("Total-demand accuracy", lambda v: f"{100 * v:.1f}%"), ("Inside the 80% range", lambda v: f"{100 * v:.0f}%")]
        for title, fmt in fmts:
            f = QFrame()
            f.setObjectName("card")
            l = QVBoxLayout(f)
            l.setContentsMargins(20, 16, 20, 12)
            l.setSpacing(3)
            tk = Ticker(fmt)
            chip = Chip()
            note = FastText("kn", "Forecast 00,000 · day 7 of 7") if not self.tiles else lab("", "kn", True)
            l.addWidget(lab(title, "kl"))
            l.addWidget(tk)
            l.addWidget(chip)
            l.addWidget(note)
            sp = pg.PlotWidget()
            sp.setFixedHeight(36)
            sp.setMouseEnabled(x=False, y=False)
            sp.setMenuEnabled(False)
            sp.hideButtons()
            opaque(sp)
            sp.getPlotItem().hideAxis("left")
            sp.getPlotItem().hideAxis("bottom")
            sp.getPlotItem().setContentsMargins(0, 0, 0, 0)
            curve = sp.plot()
            ref = pg.InfiniteLine(angle=0, movable=False)
            ref.setVisible(False)
            sp.addItem(ref)
            l.addWidget(sp)
            krow.addWidget(f)
            self.tiles.append(SimpleNamespace(tk=tk, chip=chip, note=note, sp=sp, curve=curve, ref=ref))
        root.addLayout(krow)

        # main stream + anomaly feed
        self.p_stream = make_plot(UnitsAxis(orientation="left"))
        pi = self.p_stream.getPlotItem()
        self.legend = pi.addLegend(offset=(12, 6), brush=None, pen=None, colCount=4)
        self.region = pg.LinearRegionItem(values=(0, 1), movable=False)
        for ln in self.region.lines:
            ln.setPen(pg.mkPen(None))
        pi.addItem(self.region)
        self.band_lo, self.band_hi = pg.PlotCurveItem(), pg.PlotCurveItem()
        self.band = pg.FillBetweenItem(self.band_lo, self.band_hi)
        pi.addItem(self.band)
        pi.addItem(self.band_lo)
        pi.addItem(self.band_hi)
        self.g_fc = pg.PlotCurveItem()                 # soft glow under the forecast line
        pi.addItem(self.g_fc)
        self.c_actual = pi.plot(name="Actual units (replayed)", fillLevel=0)
        self.c_trend = pi.plot(name="Trend (LOWESS)")
        self.c_fc_past = pi.plot(name="Forecast")
        self.c_fc_ahead = pi.plot(name="Forecast ahead of today", symbol="o", symbolSize=6)
        self.band_legend = pg.PlotDataItem()
        self.legend.addItem(self.band_legend, "80% range")
        self.today = pg.InfiniteLine(angle=90, movable=False)
        pi.addItem(self.today)
        self.halo = pg.ScatterPlotItem(pen=None)
        self.head = pg.ScatterPlotItem(size=9)
        pi.addItem(self.halo)
        pi.addItem(self.head)
        self.k_stream = LiveCard("Demand as it arrives", self.p_stream, 360)
        self.h_stream = Hover(self.p_stream, self._read_stream)

        self.feed = QListWidget()
        self.feed.setItemDelegate(FeedDelegate(self))
        self.feed.setUniformItemSizes(True)
        self.feed.setMouseTracking(True)
        self.feed.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.feed.setMinimumHeight(360)
        self.feed_items, self.feed_counts = {}, {}
        self.k_feed = LiveCard("Live anomaly and stock-risk feed", self.feed)
        self.feed_count = lab("", "kn")
        self.k_feed.head.addWidget(self.feed_count)
        row = QHBoxLayout()
        row.setSpacing(16)
        row.addWidget(self.k_stream, 3)
        row.addWidget(self.k_feed, 2)
        root.addLayout(row, 1)

        # running accuracy, sales race, forecast vs actual scatter
        self.p_acc = make_plot(PercentAxis(orientation="left"))
        api = self.p_acc.getPlotItem()
        self.acc_legend = api.addLegend(offset=(10, 4), brush=None, pen=None, colCount=2)
        self.c_wm = api.plot(name="Model", fillLevel=0)
        self.c_wb = api.plot(name=self.base_label)
        self.t_wm, self.t_wb = pg.TextItem(anchor=(0, 0.5)), pg.TextItem(anchor=(0, 0.5))
        api.addItem(self.t_wm)
        api.addItem(self.t_wb)
        self.k_acc = LiveCard("Running accuracy", self.p_acc, 250)
        self.h_acc = Hover(self.p_acc, self._read_acc)

        self.p_race = make_plot(date_axis=False)
        rpi = self.p_race.getPlotItem()
        rpi.hideAxis("left")
        rpi.getAxis("bottom").setTickPen(pg.mkPen(None))
        self.bars = pg.BarGraphItem(x0=[], y=[], width=[], height=0.72)
        rpi.addItem(self.bars)
        rpi.setYRange(-0.6, 7.6, padding=0)
        self.k_race = LiveCard("Top sellers so far", self.p_race, 250)

        self.p_sc = make_plot(date_axis=False)
        spi = self.p_sc.getPlotItem()
        spi.setLogMode(x=True, y=True)
        spi.setLabel("bottom", "Actual units")
        spi.setLabel("left", "Forecast units")
        self.c_diag = spi.plot()
        self.c_g_hi, self.c_g_lo = spi.plot(), spi.plot()            # forecast within 50% of actual
        self.c_old = spi.plot(pen=None, symbol="o", symbolSize=6)
        self.c_recent = spi.plot(pen=None, symbol="o", symbolSize=6)
        self.c_new = spi.plot(pen=None, symbol="o", symbolSize=12)
        self.k_sc = LiveCard("Forecast vs actual, product-weeks", self.p_sc, 250)
        for k_ in (self.k_acc, self.k_race, self.k_sc):
            k_.finding.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        row2 = QHBoxLayout()
        row2.setSpacing(16)
        row2.addWidget(self.k_acc, 1)
        row2.addWidget(self.k_race, 1)
        row2.addWidget(self.k_sc, 1)
        root.addLayout(row2, 1)

        self.k_txt = LiveCard("About this replay", body=(
            "Real sales from a public data set, replayed day by day as a simulated live feed, not a live connection. "
            "Forecasts appear only after the date they were made and use earlier data only. Built with pyqtgraph, "
            "statsmodels (LOWESS trend), SciPy (robust anomaly scores) and scikit-learn (error and quantile loss)."))
        self.k_txt.say("Recorded history, replayed honestly")
        root.addWidget(self.k_txt)

    # ---------------------------------------------------------------- data / scope
    def prepare(self):
        """Rebuild for the current filters, forecast setting and scope; keeps the replay position."""
        w = self.win
        p = w.preds
        meta = p.drop_duplicates("sku")
        meta = meta[meta["abc"].isin(w._on(w.b_abc)) & meta["demand_type"].isin(w._on(w.b_type))]
        if meta.empty:
            meta = p.drop_duplicates("sku")
        self._fill_products(meta)
        key = self.cb_scope.currentData()
        if key in ("A", "B", "C"):
            sel = meta[meta["abc"] == key]
        elif key in ("regular", "intermittent"):
            sel = meta[meta["demand_type"] == key]
        elif key == "one":
            sel = meta[meta["sku"] == self.cb_prod.currentData()]
        else:
            sel = meta
        self.scope_note = ""
        if sel.empty:
            sel, self.scope_note = meta, " (no product matches this scope with the current filters, showing all)"
        one = key == "one" and len(sel) == 1
        skus = list(sel["sku"])
        race = skus if not one else list(meta["sku"])
        hs = w._on(w.b_h)
        rows = p[p["sku"].isin(skus) & p["h"].isin(hs)]
        self.S = self.data.build(skus, race, rows, w.model, self.base, one, hs)
        self.scope_name = self.data.names[skus[0]] if one else f"{len(skus)} products"
        self.sel_sku = skus[0] if one else None
        S = self.S
        self.s_x0 = S.ctx_x[0] if len(S.ctx_x) else self.x[0]
        self.s_xmax = self.a_xmax = None             # snap the windows to the new scope
        self.p_stream.setYRange(0, S.ytop, padding=0)
        self.p_acc.setYRange(S.err_bot, S.err_top, padding=0)
        self.c_wm.setFillLevel(S.err_bot)
        top = np.log10(max(S.rows["y"].max(), S.rows["m"].max(), 1) + 1) * 1.04 if len(S.rows) else 1
        self.p_sc.setXRange(0, top, padding=0)
        self.p_sc.setYRange(0, top, padding=0)
        self.c_diag.setData([1, 10 ** top], [1, 10 ** top])
        self.c_g_hi.setData([1, 10 ** top], [1.5, 1.5 * 10 ** top])
        self.c_g_lo.setData([1, 10 ** top], [0.5, 0.5 * 10 ** top])
        for t_ in list(self.race_text.values()) + list(self.race_val.values()):
            self.p_race.removeItem(t_)
        self.race_text.clear()
        self.race_val.clear()
        self.race_state.clear()
        self._cur = self._done = None
        self._draw_pos(force=True)
        self.settle()

    def _fill_products(self, meta: pd.DataFrame):
        cur = self.cb_prod.currentData() or self.win.sku
        meta = meta.assign(name=meta["description"].astype(str).str.title()).sort_values("name")
        self.cb_prod.blockSignals(True)
        self.cb_prod.clear()
        for sku, name in zip(meta["sku"], meta["name"]):
            self.cb_prod.addItem(f"{name} ({sku})", sku)
        i = self.cb_prod.findData(cur)
        self.cb_prod.setCurrentIndex(i if i >= 0 else 0)
        self.cb_prod.blockSignals(False)

    def _on_scope(self):
        self.prepare()

    def _on_product(self):
        if self.cb_scope.currentData() != "one":
            self.cb_scope.blockSignals(True)
            self.cb_scope.setCurrentIndex(len(SCOPES) - 1)
            self.cb_scope.blockSignals(False)
        self.prepare()

    def follow(self, sku: str):
        """Keep the product picker in step with the product explorer."""
        i = self.cb_prod.findData(sku)
        if i >= 0 and i != self.cb_prod.currentIndex():
            self.cb_prod.setCurrentIndex(i)
            if self.cb_scope.currentData() == "one":
                self.prepare()

    # ---------------------------------------------------------------- controls
    def activate(self, on: bool):
        if on:
            self.timer.start()
            if self.pos < self.n - 1:
                self.play(True)
        else:
            self.play(False)
            self.timer.stop()

    def play(self, on: bool):
        self.playing = on
        self.b_play.setText("Pause" if on else ("Replay" if self.pos >= self.n - 1e-3 else "Play"))
        self._badge(True)

    def toggle_play(self):
        if self.playing:
            self.play(False)
        else:
            if self.pos >= self.n - 1e-3:
                self.pos = 0.0
            self.play(True)
            if not self.timer.isActive():
                self.timer.start()

    def restart(self):
        self.pos = 0.0
        self._draw_pos()
        self.play(True)
        if not self.timer.isActive():
            self.timer.start()

    def seek(self, k: int):
        self.pos = min(float(k) + 0.9999, self.n - 1e-6)
        self._draw_pos()

    def snapshot(self, pos: float):
        """Jump to a position and let every animation settle (used for screenshots)."""
        self.play(False)
        self.pos = min(pos, self.n - 1e-6)
        self._draw_pos()
        self.settle()

    def settle(self):
        for _ in range(90):
            self._animate()
        for f in list(self.flash.values()):
            f[1] = 0.5
            f[0].setData(Qt.ItemDataRole.UserRole + 1, 0.5)

    def _on_speed(self):
        self.speed = next(b.property("value") for b in self.b_speed if b.isChecked())

    # ---------------------------------------------------------------- frame loop
    def _frame(self):
        self.frame += 1
        if self.playing:
            self.pos = min(self.pos + self.speed / FRAMES_PER_WEEK, self.n - 1e-6)
            if self.pos >= self.n - 1e-6:
                self.play(False)
        if self.pos != self._drawn:
            self._draw_pos()
        self._animate()
        if self.frame % 30 == 0:
            self._badge(self.frame % 60 == 0)

    def _draw_pos(self, force: bool = False):
        S = self.S
        if S is None:
            return
        cur = int(self.pos)
        frac = self.pos - cur
        done = cur if frac >= 0.999 else cur - 1
        if force or cur != self._cur:
            self._cur = cur
            self._week_changed()
        if force or done != self._done:
            prev = self._done
            self._done = done
            self._complete_changed(prev)
        dpos = min(frac, 1.0) * 7
        live = S.y[cur] * float(np.interp(dpos, np.arange(8), S.cum[cur]))
        xs = np.concatenate([S.ctx_x, self.x[:cur + 1]])
        ys = np.concatenate([S.ctx_y, S.y[:cur], [live]])
        self.c_actual.setData(xs, ys)
        hx = self.x[cur]
        self.head.setData([hx], [live])
        self.halo.setData([hx], [live])
        self.region.setRegion((hx, hx + max(frac, 0.003) * WEEK_S))
        self.today.setValue(hx)
        self.slider.blockSignals(True)
        self.slider.setValue(cur)
        self.slider.blockSignals(False)
        day = self.data.weeks[cur] + pd.Timedelta(days=min(int(dpos), 6))
        set_text(self.l_day, f"{day:%A %d %b %Y}")
        tile = self.tiles[0]
        tile.tk.set(live, instant=True)
        m_now = S.m[cur]
        t = self.themes[self.win.theme]
        day_n = min(int(dpos) + 1, 7)
        if S.closed[cur]:
            note, chip = "Business shut this week (holiday)", ("", t["muted"])
        elif np.isfinite(m_now) and m_now > 0:
            reached = 10 * int(10 * live / m_now)      # in steps of 10%, so it changes about once a day
            note = f"Forecast {m_now:,.0f} · day {day_n} of 7"
            chip = (f"{reached}% of this week's forecast", t["s1"])
        else:
            note, chip = "No forecast for this week under the filters", ("", t["muted"])
        set_text(tile.note, note)
        tile.chip.set(*chip)
        self._drawn = self.pos

    def _week_changed(self):
        S, cur = self.S, self._cur
        wk = self.data.weeks[cur]
        self.l_week.setText(f"Week of {wk:%d %B %Y}")
        self.l_status.setText(
            f"{self.scope_name}{self.scope_note}  ·  week {cur + 1} of {self.n}  ·  recorded sales replayed as a "
            f"simulated live feed; every forecast was made before its week")
        self._draw_forecast()

    def _draw_forecast(self):
        S, cur, done = self.S, self._cur, self._done if self._done is not None else -1
        idx = np.arange(self.n)
        issued = np.isfinite(S.m) & (S.cut < self.data.week_t[cur])
        past = np.where(issued & (idx <= cur), S.m, np.nan)
        self.c_fc_past.setData(self.x, past, connect="finite")
        self.g_fc.setData(self.x, np.where(issued, S.m, np.nan), connect="finite")
        self.c_fc_ahead.setData(self.x, np.where(issued & (idx >= cur), S.m, np.nan), connect="finite")
        run = []
        j = cur
        while j < self.n and issued[j]:
            run.append(j)
            j += 1
        lo = hi = None
        if run and S.one:                           # the model's own range, for every week already issued
            run = [j for j in range(run[-1] + 1) if issued[j] and np.isfinite(S.lo[j]) and np.isfinite(S.hi[j])]
        if run:
            ri = np.array(run)
            if S.one:
                lo, hi = S.lo[ri], S.hi[ri]
            else:                                   # range for the total from the misses seen so far
                r = S.ratio[:done + 1]
                r = r[np.isfinite(r)]
                if len(r) >= 6:
                    q1, q9 = np.quantile(r, [0.1, 0.9])
                    lo, hi = S.m[ri] * q1, S.m[ri] * q9
        if lo is not None:
            xs = self.x[np.array(run)]
            if len(xs) == 1:
                xs, lo, hi = np.array([xs[0] - WEEK_S * 0.3, xs[0] + WEEK_S * 0.3]), np.repeat(lo, 2), np.repeat(hi, 2)
            self.band_lo.setData(xs, lo)
            self.band_hi.setData(xs, hi)
            self.band.setVisible(True)
        else:
            self.band.setVisible(False)

    def _complete_changed(self, prev):
        S, done = self.S, self._done
        self._draw_forecast()
        # trend line on everything seen so far
        xs = np.concatenate([S.ctx_x, self.x[:done + 1]])
        ys = np.concatenate([S.ctx_y, S.y[:done + 1]])
        if len(xs) >= 8:
            self.c_trend.setData(xs, lowess(ys, xs, frac=0.3, it=1, return_sorted=False))
        else:
            self.c_trend.setData([], [])

        t = self.themes[self.win.theme]
        sel = S.rows[S.rows["widx"] <= done]
        if len(sel) and sel["y"].sum() > 0:
            y, ybar = sel["y"].to_numpy(), sel["y"].mean()
            wm = mean_absolute_error(y, sel["m"]) / ybar
            wb = mean_absolute_error(y, sel["b"]) / ybar
            bs = sel["m"].sum() / sel["y"].sum() - 1
            cov = float(((sel["y"] >= sel["lo"]) & (sel["y"] <= sel["hi"])).mean())
            ql = (mean_pinball_loss(y, sel["lo"], alpha=0.1) + mean_pinball_loss(y, sel["hi"], alpha=0.9)) / 2
            gain = 1 - wm / wb if wb > 0 else 0.0
            self.tiles[1].tk.set(gain)
            self.tiles[1].tk.set_colors(fg=t["ink"])
            self.tiles[1].chip.set("✓ Ahead of the benchmark" if gain > 0 else
                                   "✕ Behind the benchmark", t["good"] if gain > 0 else t["bad"])
            self.tiles[1].note.setText(f"Per-product accuracy {100 * (1 - wm):.0f}% vs {100 * (1 - wb):.0f}%")
            self.tiles[1].note.setToolTip(f"<div style='max-width:320px'>{INTER}</div>")
            ta, tb_ = S.tacc_m[done], S.tacc_b[done]
            self.tiles[2].tk.set(ta)
            self.tiles[2].tk.set_colors(fg=t["ink"])
            d = ta - tb_ if np.isfinite(ta) and np.isfinite(tb_) else 0.0
            self.tiles[2].chip.set(f"{'↑' if d >= 0 else '↓'} {abs(100 * d):.1f} pts vs benchmark",
                                   t["good"] if d >= 0 else (t["warn"] if d > -0.05 else t["bad"]))
            self.tiles[2].note.setText(f"Forecast level {100 * bs:+.0f}% of units sold")
            self.tiles[2].note.setToolTip("<div style='max-width:320px'>1 - WAPE of the weekly total for the products "
                                          "in scope: how close the total order quantity would have been.</div>")
            self.tiles[3].tk.set(cov)
            self.tiles[3].tk.set_colors(fg=t["ink"])
            ok = abs(cov - 0.8) <= 0.05
            self.tiles[3].chip.set("● On the 80% target" if ok else ("↑ Above the 80% target" if cov > 0.8
                                                                     else "↓ Below the 80% target"),
                                   t["good"] if ok else t["warn"])
            self.tiles[3].note.setText(f"{len(sel):,} forecasts checked · quantile loss {ql:,.1f}")
        else:
            for tl in self.tiles[1:]:
                tl.tk.set(np.nan)
                tl.tk.set_colors(fg=t["ink"])
                tl.chip.set("", t["muted"])
                tl.note.setText("Waiting for the first full week")
        for tl in self.tiles[1:]:
            tl.curve.setPen(pg.mkPen(tl.chip.col if tl.chip.text() else QColor(t["s1"]), width=1.8))
        lo_ = max(0, done - 25)
        for tl, arr, ref in zip(self.tiles, (S.y, S.gain, S.tacc_m, S.cov), (None, 0.0, None, 0.8)):
            a = arr[lo_:done + 1] if done >= 0 else np.array([])
            tl.curve.setData(np.arange(len(a)), a, connect="finite")
            tl.ref.setVisible(ref is not None and len(a) > 0)
            if ref is not None:
                tl.ref.setValue(ref)

        # running accuracy
        k, w0 = done + 1, S.warm
        end = self.x[max(done, w0)]
        span = end - self.x[w0]
        self.a_goal = (self.x[w0] - WEEK_S * 0.5, end + max(3 * WEEK_S, 0.2 * span, 8 * WEEK_S - span))
        self.c_wm.setData(self.x[w0:k], S.wape_m[w0:k], connect="finite")
        self.c_wb.setData(self.x[w0:k], S.wape_b[w0:k], connect="finite")
        if done < w0:
            self.t_wm.setText("")
            self.t_wb.setText("")
            self.k_acc.say(f"Warming up: scores start after {WARMUP_WEEKS * (2 if S.one else 1)} checked weeks", "")
        elif k and np.isfinite(S.wape_m[done]) and np.isfinite(S.wape_b[done]):
            self.t_wm.setText(f"{100 * S.wape_m[done]:.1f}%")
            self.t_wm.setPos(self.x[done] + WEEK_S * 0.6, S.wape_m[done])
            self.t_wb.setText(f"{100 * S.wape_b[done]:.1f}%")
            self.t_wb.setPos(self.x[done] + WEEK_S * 0.6, S.wape_b[done])
            gap = S.wape_m[done] / S.wape_b[done] - 1
            self.k_acc.say(f"Model error {abs(100 * gap):.0f}% {'lower' if gap < 0 else 'higher'} than the {self.base_label}",
                           f"Error (WAPE) over every product-week checked so far, lower is better: model "
                           f"{100 * S.wape_m[done]:.1f}%, {self.base_label} {100 * S.wape_b[done]:.1f}%. Drawn after a "
                           f"warm-up, because the first few weeks swing wildly.")
        else:
            self.t_wm.setText("")
            self.t_wb.setText("")
            self.k_acc.say("Waiting for the first full week.", "")

        # main card text
        if done >= 0:
            wk = self.data.weeks[done]
            m_ = S.m[done]
            fc = f" · forecast {m_:,.0f}" if np.isfinite(m_) else ""
            self.k_stream.say(f"{S.y[done]:,.0f} units sold in the week of {wk:%d %b}{fc}")
        else:
            self.k_stream.say("The first replay week is filling in")
        band_txt = "the model's 80% range" if S.one else "an 80% range for the total, from the misses seen so far"
        self.k_stream.say(self.k_stream.finding.text(),
                          f"Light line: sales arriving day by day (the shaded strip is today's week). Blue: forecast; "
                          f"dots are forecasts already made for weeks still ahead. Shaded blue: {band_txt}. "
                          f"Purple: LOWESS trend (statsmodels) of the sales seen so far. Move the mouse over the chart for values.")

        # race
        if done >= 0 and len(S.race_skus):
            vals = S.race_cum[:, done]
            order = np.argsort(-vals)[:8]
            order = [i for i in order if vals[i] > 0]
            self.race_target = {S.race_skus[i]: (float(vals[i]), r) for r, i in enumerate(order)}
            if order:
                lead = S.race_skus[order[0]]
                nm = self.data.names[lead]
                nm = nm if len(nm) <= 20 else nm[:19].rstrip() + "…"
                self.k_race.say(f"{nm} leads · {vals[order[0]]:,.0f} units",
                                f"{self.data.names[lead]}. Units sold since the replay started; bars glide as the order changes."
                                + (" Blue = the selected product." if S.one else " Blue = the leader."))
        else:
            self.race_target = {}
            self.k_race.say("The race starts with the first full week", "")

        # scatter
        old, new = sel[sel["widx"] < done - 3], sel[sel["widx"] == done]
        rec = sel[(sel["widx"] >= done - 3) & (sel["widx"] < done)]
        self.c_old.setSymbolSize(10 if S.one else 4)
        self.c_recent.setSymbolSize(11 if S.one else 5)
        self.c_new.setSymbolSize(15 if S.one else 6.5)
        self.c_old.setData(old["y"].to_numpy() + 1, old["m"].to_numpy() + 1)
        self.c_recent.setData(rec["y"].to_numpy() + 1, rec["m"].to_numpy() + 1)
        self.c_new.setData(new["y"].to_numpy() + 1, new["m"].to_numpy() + 1)
        pos = new[new["y"] > 0]
        if len(pos) == 1:
            r_ = pos.iloc[0]
            self.k_sc.say(f"Latest week: {r_['y']:,.0f} sold, {r_['m']:,.0f} forecast ({r_['m'] / r_['y'] - 1:+.0%})",
                          "Each dot is one week of this product (log scale). On the diagonal = exact; dotted lines = within 50%. Orange = the latest week.")
        elif len(pos):
            within = float(((pos["m"] - pos["y"]).abs() <= 0.5 * pos["y"]).mean())
            self.k_sc.say(f"{within:.0%} of {len(pos)} products within 50% this week",
                          "Each dot is one product-week (log scale). On the diagonal = exact; dotted lines = within 50%. Brighter = the last 4 weeks, orange = the latest week.")
        else:
            self.k_sc.say("Dots appear as weeks are checked", "")

        # feed
        incremental = prev is not None and done == prev + 1
        self._feed(done, incremental)

    # ---------------------------------------------------------------- feed
    def _feed(self, done: int, incremental: bool):
        """Newest first. One row per product (one per week for a single product), with a running count."""
        S = self.S
        ev = S.events.assign(widx=S.ev_widx, az=S.events["z"].abs())
        ev = ev[ev["widx"] <= done]
        top = ev.sort_values(["widx", "az"]).groupby("widx").tail(4)
        if incremental:
            add = top[top["widx"] == done]
        else:
            self.feed.clear()
            self.flash.clear()
            self.feed_items, self.feed_counts = {}, {}
            for k_ in (top["week"] if S.one else top["sku"]):
                self.feed_counts[k_] = self.feed_counts.get(k_, 0) + 1
            add = top.drop_duplicates("week" if S.one else "sku", keep="last").tail(40)
        for r in add.itertuples():
            self._feed_item(r, incremental)
        while self.feed.count() > 40:
            it = self.feed.takeItem(self.feed.count() - 1)
            self.flash.pop(id(it), None)
            self.feed_items.pop(it.data(Qt.ItemDataRole.UserRole + 2), None)
        nflag = len(top)
        self.feed_count.setText(f"{nflag:,} flagged")
        self.k_feed.say("Sales that broke out of the forecast range" if nflag else "No unusual weeks yet",
                        "A product-week is flagged when sales fall outside its 80% range by at least 2 robust "
                        "standard deviations of its earlier misses (median absolute deviation, SciPy). Above the "
                        "range = stock-out risk, below = overstock risk. Up to 4 per week.")

    def _feed_item(self, r, flash: bool):
        S = self.S
        key = r.week if S.one else r.sku
        if flash:
            self.feed_counts[key] = self.feed_counts.get(key, 0) + 1
        old = self.feed_items.pop(key, None)
        if old is not None:
            self.flash.pop(id(old), None)
            self.feed.takeItem(self.feed.row(old))
        up = r.y > r.m
        kind = "Stock-out risk" if up else "Overstock risk"
        name = self.data.names.get(r.sku, r.sku)
        big = f"{r.y / r.m - 1:+.0%}" if r.m > 0 else "new"
        n = self.feed_counts.get(key, 1)
        d = dict(up=up, big=big,
                 title=f"Week of {r.week:%d %b %Y}" if S.one else name,
                 sub=f"{kind}  ·  {r.y:,.0f} sold vs {r.m:,.0f} forecast" if S.one else
                     f"{kind}  ·  {r.week:%d %b}  ·  {r.y:,.0f} sold vs {r.m:,.0f}",
                 small=f"{n} flags" if n > 1 and not S.one else f"score {abs(r.z):.1f}")
        it = QListWidgetItem()
        it.setData(Qt.ItemDataRole.UserRole, d)
        it.setData(Qt.ItemDataRole.UserRole + 1, 1.0 if flash else 0.0)
        it.setData(Qt.ItemDataRole.UserRole + 2, key)
        it.setToolTip(f"{name} ({r.sku}). Chance of a miss at least this large if the forecast were right: "
                      f"{r.p:.2%} (normal approximation).")
        self.feed.insertItem(0, it)
        self.feed_items[key] = it
        if flash:
            self.flash[id(it)] = [it, 1.0]

    # ---------------------------------------------------------------- animation
    def _animate(self):
        for tl in self.tiles[1:]:
            tl.tk.step()
        self._windows()
        # pulsing head on the live sales line
        s = 16 + (9 * abs(np.sin(self.frame / 16.0)) if self.playing else 4)
        self.halo.setSize(s)
        # flashing new feed rows
        if self.flash:
            t = self.themes[self.win.theme]
            for key, f in list(self.flash.items()):
                f[1] *= 0.975
                if f[1] < 0.04:
                    f[0].setData(Qt.ItemDataRole.UserRole + 1, 0.0)
                    del self.flash[key]
                elif self.frame % 3 == 0:
                    f[0].setData(Qt.ItemDataRole.UserRole + 1, f[1])
        self._race_step()

    def _windows(self):
        """The demand and accuracy charts end a few weeks after "today" and glide when a week completes."""
        S = self.S
        if S is None or self._cur is None:
            return
        goal = min(self.x[self._cur] + 4.6 * WEEK_S, self.x[-1] + 1.2 * WEEK_S)
        if self.s_xmax is None or abs(goal - self.s_xmax) > 0.5 * WEEK_S * 30:
            self.s_xmax = goal
            self.p_stream.setXRange(self.s_x0, goal, padding=0)
        elif abs(goal - self.s_xmax) > 600:
            self.s_xmax += 0.06 * (goal - self.s_xmax)
            self.p_stream.setXRange(self.s_x0, self.s_xmax, padding=0)
        if self.a_goal is not None:
            a0, a1 = self.a_goal
            if self.a_xmax is None or abs(a1 - self.a_xmax) > WEEK_S * 15:
                self.a_xmax = a1
                self.p_acc.setXRange(a0, a1, padding=0)
            elif abs(a1 - self.a_xmax) > 600:
                self.a_xmax += 0.06 * (a1 - self.a_xmax)
                self.p_acc.setXRange(a0, self.a_xmax, padding=0)

    # ---------------------------------------------------------------- hover read-outs
    def _week_at(self, xv: float, last: int) -> int | None:
        if last < 0:
            return None
        i = int(round((xv - self.x[0]) / WEEK_S))
        return min(max(i, 0), last)

    def _read_stream(self, xv: float):
        S = self.S
        if S is None or self._cur is None:
            return None
        cur = self._cur
        last = cur
        idx = np.arange(self.n)
        issued = np.isfinite(S.m) & (S.cut < self.data.week_t[cur])
        ahead = np.flatnonzero(issued & (idx > cur))
        if len(ahead):
            last = int(ahead[-1])
        i = self._week_at(xv, last)
        if i is None:
            return None
        wk = self.data.weeks[i]
        rows = [f"<b>Week of {wk:%d %b %Y}</b>"]
        y = None
        if i < cur:
            y = S.y[i]
            rows.append(f"Sold <b>{y:,.0f}</b>")
        elif i == cur:
            y = float(self.c_actual.yData[-1]) if self.c_actual.yData is not None and len(self.c_actual.yData) else None
            if y is not None:
                rows.append(f"Sold so far <b>{y:,.0f}</b>")
        else:
            rows.append("Still ahead")
        if issued[i]:
            rows.append(f"<span style='color:{self.t_['s1']}'>Forecast <b>{S.m[i]:,.0f}</b></span>")
            if y is None:
                y = S.m[i]
            if S.one and np.isfinite(S.lo[i]):
                rows.append(f"80% range {S.lo[i]:,.0f} to {S.hi[i]:,.0f}")
            if i < cur and S.m[i] > 0 and S.y[i] > 0:
                rows.append(f"Miss {S.m[i] / S.y[i] - 1:+.0%}")
        elif S.closed[i]:
            rows.append("Holiday week")
        return self.x[i], y, "<br>".join(rows)

    def _read_acc(self, xv: float):
        S = self.S
        if S is None or self._done is None or self._done < S.warm:
            return None
        i = self._week_at(xv, self._done)
        if i is None or i < S.warm or not np.isfinite(S.wape_m[i]):
            return None
        wk = self.data.weeks[i]
        return self.x[i], S.wape_m[i], (
            f"<b>Up to {wk:%d %b %Y}</b><br>"
            f"<span style='color:{self.t_['s1']}'>Model <b>{100 * S.wape_m[i]:.1f}%</b></span><br>"
            f"<span style='color:{self.t_['s2']}'>{self.base_label} <b>{100 * S.wape_b[i]:.1f}%</b></span>")

    def _race_step(self):
        t = self.themes[self.win.theme]
        tgt = self.race_target
        moving = False
        for sku, (val, rank) in tgt.items():
            st = self.race_state.setdefault(sku, [0.0, -1.5])
            ty = 7 - rank
            dv, dy = val - st[0], ty - st[1]
            if abs(dv) > 0.5 or abs(dy) > 0.01:
                st[0] += 0.07 * dv
                st[1] += 0.07 * dy
                moving = True
            else:
                st[0], st[1] = val, ty
        for sku in [k for k in self.race_state if k not in tgt]:
            st = self.race_state[sku]
            st[0] *= 0.9
            st[1] += 0.07 * (-1.6 - st[1])
            moving = True
            if st[0] < 1:
                del self.race_state[sku]
                for store in (self.race_text, self.race_val):
                    txt = store.pop(sku, None)
                    if txt is not None:
                        self.p_race.removeItem(txt)
        vmax = max([v for v, _ in tgt.values()] or [1.0])
        goal = vmax * 1.22
        if abs(goal - self.race_xmax) > 1e-3 * goal:
            self.race_xmax += 0.05 * (goal - self.race_xmax)
            moving = True
        if not moving and getattr(self, "_race_still", False):
            return
        self._race_still = not moving
        skus = list(self.race_state)
        lead = max(tgt, key=lambda k: tgt[k][0]) if tgt else None
        hi = self.sel_sku if self.S is not None and self.S.one and self.sel_sku in tgt else lead
        self.bars.setOpts(x0=np.zeros(len(skus)), y=[self.race_state[k][1] for k in skus],
                          width=[self.race_state[k][0] for k in skus], height=0.7,
                          brushes=[pg.mkBrush(t["s1"] if k == hi else t["neutral"]) for k in skus],
                          pen=pg.mkPen(None))
        self.p_race.setXRange(0, self.race_xmax, padding=0)
        self.p_race.setYRange(-0.6, 7.6, padding=0)
        vw = self.p_race.getPlotItem().vb.width() or 300
        per_px = self.race_xmax / vw                       # data units per pixel
        fam = self.font().family()
        f_name, f_val = QFont(fam, 8), QFont(fam, 8)
        f_name.setWeight(QFont.Weight.Medium)
        f_val.setWeight(QFont.Weight.DemiBold)
        fm = QFontMetrics(f_name)
        for k in skus:
            v, y = self.race_state[k]
            txt, val = self.race_text.get(k), self.race_val.get(k)
            if txt is None:
                txt, val = pg.TextItem(anchor=(0, 0.5)), pg.TextItem(anchor=(0, 0.5))
                txt.setFont(f_name)
                val.setFont(f_val)
                txt.setZValue(10)
                val.setZValue(10)
                self.p_race.addItem(txt)
                self.p_race.addItem(val)
                self.race_text[k], self.race_val[k] = txt, val
            name = self.data.names.get(k, k)
            name = name if len(name) <= 24 else name[:23].rstrip() + "…"
            if txt.toPlainText() != name:
                txt.setText(name)
            name_w = (fm.horizontalAdvance(name) + 14) * per_px
            inside = name_w + 4 * per_px < v
            txt.setColor("#ffffff" if (k == hi and inside) else t["ink"])
            txt.setPos(4 * per_px, y)
            vt = f"{v:,.0f}"
            if val.toPlainText() != vt:
                val.setText(vt)
            val.setColor(t["s1"] if k == hi else t["ink2"])
            val.setPos(max(v, name_w) + 4 * per_px, y)

    def _badge(self, bright: bool):
        t = self.themes[self.win.theme]
        if self.playing:
            col = t["bad"] if bright else qcolor(t["bad"], 120)
            text = "●  LIVE REPLAY"
        else:
            col, text = t["muted"], "❚❚  PAUSED"
        self.badge.setText(text)
        self.badge.set_colors(fg=col)

    # ---------------------------------------------------------------- theme
    def apply_theme(self, t: dict):
        font = QFont(self.font().family(), 8)
        for w in (self.p_stream, self.p_acc, self.p_race, self.p_sc):
            w.setBackground(t["surface"])
            pi = w.getPlotItem()
            for name in ("left", "bottom"):
                ax = pi.getAxis(name)
                ax.setPen(pg.mkPen(t["axis"]))
                ax.setTextPen(pg.mkPen(t["ink2"]))
                ax.setStyle(tickFont=font)
            pi.showGrid(x=False, y=True, alpha=0.18)
            for name in ("left", "bottom"):
                pi.getAxis(name).setPen(pg.mkPen(t["grid"]))
                pi.getAxis(name).setTextPen(pg.mkPen(t["ink2"]))
                pi.getAxis(name).label.setDefaultTextColor(QColor(t["muted"]))
        self.badge.set_colors(bg=t["tint"])
        self.l_day.set_colors(fg=t["ink"], bg=t["tint"])
        self.t_ = t
        self.tiles[0].note.set_colors(fg=t["ink2"], bg=t["surface"])
        for tl in self.tiles:
            tl.chip.set_colors(bg=t["surface"])
        for h in (self.h_stream, self.h_acc):
            h.style(t)
        for tl in self.tiles:
            tl.tk.set_colors(fg=t["ink"], bg=t["surface"])
            tl.sp.setBackground(t["surface"])
            tl.curve.setPen(pg.mkPen(t["s1"], width=1.8))
            tl.curve.setFillLevel(None)
            tl.ref.setPen(pg.mkPen(t["muted"], width=1, style=Qt.PenStyle.DotLine))
        self.tiles[0].curve.setFillLevel(0)
        self.tiles[0].curve.setBrush(fade(t["s1"], 70))
        self.c_actual.setPen(pg.mkPen(t["actual"], width=2.2))
        self.c_actual.setBrush(fade(t["actual"], 38))
        self.g_fc.setPen(pg.mkPen(qcolor(t["s1"], 46), width=8))
        self.c_trend.setPen(pg.mkPen(TREND, width=2, style=Qt.PenStyle.DashDotLine))
        self.c_fc_past.setPen(pg.mkPen(t["s1"], width=2.2))
        self.c_fc_ahead.setPen(pg.mkPen(t["s1"], width=2.4, style=Qt.PenStyle.DashLine))
        self.c_fc_ahead.setSymbolBrush(t["surface"])
        self.c_fc_ahead.setSymbolPen(pg.mkPen(t["s1"], width=2))
        self.band.setBrush(qcolor(t["s1"], 64))
        self.band_lo.setPen(pg.mkPen(qcolor(t["s1"], 150), width=1))
        self.band_hi.setPen(pg.mkPen(qcolor(t["s1"], 150), width=1))
        self.band_legend.setPen(pg.mkPen(qcolor(t["s1"], 110), width=8))
        self.region.setBrush(qcolor(t["s1"], 30))
        self.today.setPen(pg.mkPen(t["muted"], width=1.2, style=Qt.PenStyle.DashLine))
        self.head.setBrush(pg.mkBrush(t["ink"]))
        self.head.setPen(pg.mkPen(t["surface"], width=2))
        self.halo.setBrush(pg.mkBrush(qcolor(t["s1"], 80)))
        self.c_wm.setPen(pg.mkPen(t["s1"], width=2.4))
        self.c_wm.setBrush(fade(t["s1"], 26))
        self.c_wb.setPen(pg.mkPen(t["s2"], width=2, style=Qt.PenStyle.DashLine))
        self.t_wm.setColor(t["s1"])
        self.t_wb.setColor(t["s2"])
        self.c_diag.setPen(pg.mkPen(t["muted"], width=1.2, style=Qt.PenStyle.DashLine))
        self.c_old.setSymbolBrush(qcolor(t["s1"], 48))
        self.c_old.setSymbolPen(pg.mkPen(None))
        self.c_recent.setSymbolBrush(qcolor(t["s1"], 175))
        self.c_recent.setSymbolPen(pg.mkPen(None))
        for g_ in (self.c_g_hi, self.c_g_lo):
            g_.setPen(pg.mkPen(qcolor(t["muted"], 150), width=1, style=Qt.PenStyle.DotLine))
        self.c_new.setSymbolBrush(qcolor(t["s2"], 230))
        self.c_new.setSymbolPen(pg.mkPen(t["surface"], width=1))
        for lg in (self.legend, self.acc_legend):
            lg.setLabelTextColor(t["ink2"])
            lg.setLabelTextSize("8pt")
        for txt in list(self.race_text.values()) + list(self.race_val.values()):
            self.p_race.removeItem(txt)
        self.race_text.clear()
        self.race_val.clear()
        self._race_still = False
        for k_ in (self.k_acc, self.k_race, self.k_sc):     # two-line headers, so the three charts align
            k_.finding.ensurePolished()
            k_.finding.setFixedHeight(2 * k_.finding.fontMetrics().lineSpacing() + 4)
        self._badge(True)
