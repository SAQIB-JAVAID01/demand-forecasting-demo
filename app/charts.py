"""Chart canvas for the desktop app: one visual language for every chart.

Blue = model, orange = simple method, slate = actual. Bars sit on a light
track, numbers are printed next to the marks, and line charts have a hover
read-out.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams
from matplotlib.patches import Polygon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QSizePolicy

for _f in sorted((Path(__file__).resolve().parent / "fonts").glob("*.ttf")):     # bundled Inter
    font_manager.fontManager.addfont(str(_f))
rcParams["font.family"] = ["Inter", "Segoe UI", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def gradient_fill(ax, x, y, color: str, alpha: float = 0.26):
    """Area under a line that fades to transparent at the baseline."""
    x = np.asarray(x, float)
    y = np.nan_to_num(np.asarray(y, float))
    if len(x) < 2 or y.max() <= 0:
        return
    z = np.empty((128, 1, 4))
    z[:, :, :3] = mcolors.to_rgb(color)
    z[:, :, 3] = np.linspace(0, alpha, 128)[:, None] ** 1.3 / alpha ** 0.3
    im = ax.imshow(z, aspect="auto", extent=[x.min(), x.max(), 0, y.max()], origin="lower", zorder=1,
                   interpolation="bilinear")
    clip = Polygon(np.vstack([[x[0], 0], np.column_stack([x, y]), [x[-1], 0]]), facecolor="none",
                   edgecolor="none", closed=True)
    ax.add_patch(clip)
    im.set_clip_path(clip)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def units(v: float) -> str:
    if abs(v) >= 1_000_000:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 10_000:
        return f"{v / 1000:.0f}k"
    if abs(v) >= 1000:
        return f"{v / 1000:.1f}k"
    return f"{v:,.0f}"


class Chart(FigureCanvasQTAgg):
    def __init__(self, height: int = 280):
        self.fig = Figure(figsize=(6, 3), dpi=100)
        super().__init__(self.fig)
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.t: dict = {}
        self._hover = None
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("axes_leave_event", self._on_leave)
        self.mpl_connect("figure_leave_event", self._on_leave)
        self.mpl_connect("resize_event", self._on_resize)

    # ---------- shared styling ----------
    def _on_resize(self, _):
        self._layout()

    def _layout(self):
        try:
            self.fig.tight_layout(pad=1.0)
        except Exception:
            pass

    def _base(self, t: dict, grid: str = "y", frame: str = "bottom"):
        self.t = t
        self._hover = None
        ax = self.ax
        ax.clear()
        self.fig.set_facecolor(t["surface"])
        ax.set_facecolor(t["surface"])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(side in frame.split("+"))
            ax.spines[side].set_color(t["axis"])
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=t["axis"], labelcolor=t["ink2"], labelsize=8.8, length=0, pad=7)
        ax.grid(False)
        if grid:
            ax.grid(axis=grid, color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        return ax

    def _legend(self, ncols: int = 4):
        leg = self.ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncols=ncols, frameon=False, fontsize=8.6,
                             handlelength=1.1, handleheight=0.7, borderaxespad=0.1, columnspacing=1.8, handletextpad=0.5)
        for txt in leg.get_texts():
            txt.set_color(self.t["ink2"])

    def _done(self):
        self._layout()
        self.draw_idle()

    # ---------- line chart ----------
    def lines(self, t, x, series, band=None, mark=None, shade=None, extra=None, area: str | None = None,
              ymax: float | None = None, yfmt=None):
        """series: [{name, color, values, width}]. area = name of the series to fill under."""
        ax = self._base(t)
        x = pd.DatetimeIndex(x)
        if shade is not None:
            ax.axvspan(shade[0], shade[1], color=t["band"], alpha=0.5, linewidth=0, label=shade[2])
        if band is not None:
            ax.fill_between(x, band[0], band[1], color=t["band"], linewidth=0, label="80% range")
        for s in series:
            v = np.asarray(s["values"], float)
            if area == s["name"]:
                gradient_fill(ax, mdates.date2num(x.to_pydatetime()), v, t[s["color"]])
            ax.plot(x, v, color=t[s["color"]], linewidth=s.get("width", 2.1), label=s["name"], zorder=3,
                    solid_capstyle="round", solid_joinstyle="round")
            ok = np.flatnonzero(~np.isnan(v))
            if len(ok):                                   # dot on the latest point
                ax.plot([x[ok[-1]]], [v[ok[-1]]], marker="o", markersize=6, color=t[s["color"]],
                        markeredgecolor=t["surface"], markeredgewidth=1.8, zorder=5)
        if mark is not None:
            ax.axvline(mark, color=t["muted"], linewidth=1, linestyle=(0, (4, 4)))
        ax.set_ylim(0, ymax)
        ax.set_xlim(x[0], x[-1])
        ax.margins(x=0.015)
        ax.yaxis.set_major_formatter(lambda v, _: (yfmt or units)(v))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, min_n_ticks=3))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        self._legend()
        self._hover = dict(x=x, xnum=mdates.date2num(x.to_pydatetime()), series=series, band=band, extra=extra, yfmt=yfmt,
                           line=ax.axvline(x[0], color=t["muted"], linewidth=1, visible=False),
                           note=ax.annotate("", xy=(0, 0), xytext=(12, -12), textcoords="offset points",
                                            va="top", fontsize=9, color=t["ink"], visible=False, zorder=10,
                                            linespacing=1.6,
                                            bbox=dict(boxstyle="round,pad=0.7,rounding_size=0.5", fc=t.get("raised", t["surface"]),
                                                      ec=t["border"], lw=1)))
        self._done()

    # ---------- horizontal bars on a track ----------
    def hbars(self, t, labels, values, colors, fmt=pct, signed=False):
        ax = self._base(t, grid="", frame="")
        y = np.arange(len(labels))[::-1]
        lo, hi = min(0.0, min(values)), max(0.0, max(values))
        span = (hi - lo) or 1.0
        if not signed:
            ax.barh(y, [hi] * len(y), height=0.52, color=t["track"], linewidth=0)
        ax.barh(y, values, height=0.52, color=[t[c] for c in colors], linewidth=0)
        for yi, v, c in zip(y, values, colors):
            ax.text((v if v >= 0 else 0) + span * 0.02, yi, fmt(v), va="center", ha="left", fontsize=9.2,
                    color=t["ink"], fontweight="semibold" if c == "s1" else "normal")
        ax.set_yticks(y, labels)
        ax.tick_params(axis="y", labelsize=9.2)
        ax.set_xticks([])
        ax.set_xlim(lo - span * 0.02, hi + span * 0.2)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        if signed:
            ax.axvline(0, color=t["axis"], linewidth=1.2)
        self._done()

    # ---------- two bars per category ----------
    def grouped(self, t, labels, a, b, name_a, name_b):
        """a = model (blue), b = simple method (orange)."""
        ax = self._base(t, grid="", frame="")
        y = np.arange(len(labels))[::-1]
        top = max(a + b)
        ax.barh(y + 0.19, a, height=0.32, color=t["s1"], label=name_a, linewidth=0)
        ax.barh(y - 0.19, b, height=0.32, color=t["s2"], label=name_b, linewidth=0)
        for yi, va, vb in zip(y, a, b):
            ax.text(va + top * 0.015, yi + 0.19, pct(va), va="center", fontsize=8.8, color=t["ink"], fontweight="semibold")
            ax.text(vb + top * 0.015, yi - 0.19, pct(vb), va="center", fontsize=8.8, color=t["ink2"])
        ax.set_yticks(y, labels)
        ax.tick_params(axis="y", labelsize=9.2)
        ax.set_xticks([])
        ax.set_xlim(0, top * 1.16)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        self._legend()
        self._done()

    # ---------- vertical bars ----------
    def vbars(self, t, labels, values, colors, target=None, target_label="", fmt=pct, ymax=None, signed=False,
              inside=False):
        ax = self._base(t, frame="" if signed else "bottom")
        n = np.arange(len(labels))
        ax.bar(n, values, width=0.5, color=[t[c] for c in colors], linewidth=0)
        lo, hi = min(0.0, min(values)), max(0.0, max(values))
        span = (hi - lo) or 1.0
        for xi, v in zip(n, values):
            if inside:
                ax.text(xi, v - span * 0.04, fmt(v), ha="center", va="top", fontsize=10, color="#ffffff", fontweight="semibold")
            else:
                ax.text(xi, v + (span * 0.03 if v >= 0 else -span * 0.03), fmt(v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=9.2, color=t["ink"], fontweight="semibold")
        if target is not None:
            ax.axhline(target, color=t["ink"], linewidth=1.1, linestyle=(0, (4, 4)), label=target_label)
            self._legend()
        if signed:
            ax.axhline(0, color=t["axis"], linewidth=1.2)
        ax.set_xticks(n, labels)
        ax.tick_params(axis="x", labelsize=9.2)
        ax.set_ylim(lo - (span * 0.16 if lo < 0 else 0), ymax if ymax else hi + span * 0.16)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%" if fmt is pct else fmt(v))
        self._done()

    # ---------- scatter ----------
    def scatter(self, t, x, y, colors, xlabel, ylabel):
        ax = self._base(t, grid="both", frame="left+bottom")
        lo, hi = max(1, min(x.min(), y.min()) * 0.7), max(x.max(), y.max()) * 1.4
        ax.fill_between([lo, hi], [lo * 0.8, hi * 0.8], [lo * 1.2, hi * 1.2], color=t["band"], alpha=0.55, linewidth=0,
                        label="Within 20% of actual")
        ax.plot([lo, hi], [lo, hi], color=t["ink2"], linewidth=1.2, linestyle=(0, (5, 4)), label="Perfect forecast")
        colors = np.asarray(colors)
        for c, name in (("s1", "Model beats the simple method"), ("s2", "Simple method is better")):
            k = colors == c
            ax.scatter(x[k], y[k], s=40, color=t[c], edgecolors=t["surface"], linewidths=0.8, alpha=0.85, label=name, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(lambda v, _: units(v))
            axis.set_minor_formatter(lambda v, _: "")
        ax.tick_params(which="minor", length=0)
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=8.8, labelpad=6)
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=8.8, labelpad=6)
        self._legend(ncols=2)
        self._done()

    # ---------- histogram ----------
    def hist(self, t, values, bins, xlabel):
        ax = self._base(t)
        counts, edges = np.histogram(values, bins=bins)
        mids = (edges[:-1] + edges[1:]) / 2
        w = (edges[1] - edges[0]) * 0.86
        ax.bar(mids[mids < 0], counts[mids < 0], width=w, color=t["s1"], linewidth=0, label="Model is better")
        ax.bar(mids[mids >= 0], counts[mids >= 0], width=w, color=t["s2"], linewidth=0, label="Simple method is better")
        ax.axvline(0, color=t["ink"], linewidth=1.1, linestyle=(0, (4, 4)))
        ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:+.0f}%")
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, integer=True))
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=8.8, labelpad=6)
        ax.set_ylabel("Products", color=t["ink2"], fontsize=8.8, labelpad=6)
        self._legend()
        self._done()

    # ---------- backtest timeline ----------
    def timeline(self, t, folds, weeks):
        ax = self._base(t, grid="x")
        x0 = mdates.date2num(weeks[0])
        seen = set()
        for i, (cut, kind) in enumerate(folds):
            yy = len(folds) - 1 - i
            c = mdates.date2num(weeks[cut])
            ax.barh(yy, c + 7 - x0, left=x0, height=0.52, color=t["track"], linewidth=0,
                    label=None if "h" in seen else "History the model can see")
            seen.add("h")
            ax.barh(yy, 28, left=c + 7, height=0.52, color=t["soft"] if kind == "selection" else t["s1"], linewidth=0,
                    label=None if kind in seen else {"selection": "Forecast used to choose settings",
                                                     "final": "Final test forecast"}[kind])
            seen.add(kind)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_yticks([len(folds) - 1, 0], ["First\nforecast", "Last\nforecast"])
        ax.set_ylim(-0.7, len(folds) - 0.3)
        self._legend(ncols=3)
        self._done()

    # ---------- hover ----------
    def _on_move(self, ev):
        h = self._hover
        if not h or ev.inaxes is not self.ax or ev.xdata is None:
            return
        i = int(np.abs(h["xnum"] - ev.xdata).argmin())
        rows = [f"Week of {h['x'][i]:%d %b %Y}"]
        for s in h["series"]:
            v = s["values"][i]
            if v is not None and not np.isnan(v):
                rows.append(f"{s['name']}: {h['yfmt'](v) if h['yfmt'] else format(v, ',.0f')}")
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
