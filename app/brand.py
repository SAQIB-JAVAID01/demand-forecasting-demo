"""Logo for Demand Forecasting Studio, drawn in code (no image files needed).

The mark: three rising bars for sales history, then a dashed step and a dot
for the forecast, on a rounded blue tile. Original artwork for this project.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap

BLUE_TOP, BLUE_BOTTOM, ORANGE = "#3b8bea", "#1b57a8", "#f2783f"


def logo_pixmap(size: int = 256) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 100.0

    tile = QPainterPath()
    tile.addRoundedRect(QRectF(0, 0, size, size), 24 * s, 24 * s)
    grad = QLinearGradient(0, 0, size, size)
    grad.setColorAt(0, QColor(BLUE_TOP))
    grad.setColorAt(1, QColor(BLUE_BOTTOM))
    p.fillPath(tile, QBrush(grad))

    # history: three rising bars
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#ffffff"))
    for x, h in ((20, 22), (36, 34), (52, 46)):
        p.drawRoundedRect(QRectF(x * s, (78 - h) * s, 11 * s, h * s), 3.2 * s, 3.2 * s)

    # forecast: dashed outline of the next bar, and its expected level
    pen = QPen(QColor(255, 255, 255, 215), 2.6 * s, Qt.PenStyle.CustomDashLine, Qt.PenCapStyle.RoundCap)
    pen.setDashPattern([1.6, 2.2])
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(68 * s, 24 * s, 11 * s, 54 * s), 3.2 * s, 3.2 * s)
    p.setPen(QPen(QColor("#ffffff"), 2.4 * s))
    p.setBrush(QColor(ORANGE))
    p.drawEllipse(QPointF(73.5 * s, 24 * s), 7.2 * s, 7.2 * s)
    p.end()
    return pm


def logo_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(logo_pixmap(size))
    return icon


SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="512" height="512">
  <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{BLUE_TOP}"/><stop offset="1" stop-color="{BLUE_BOTTOM}"/></linearGradient></defs>
  <rect width="100" height="100" rx="24" fill="url(#g)"/>
  <rect x="20" y="56" width="11" height="22" rx="3.2" fill="#fff"/>
  <rect x="36" y="44" width="11" height="34" rx="3.2" fill="#fff"/>
  <rect x="52" y="32" width="11" height="46" rx="3.2" fill="#fff"/>
  <rect x="68" y="24" width="11" height="54" rx="3.2" fill="none" stroke="#fff" stroke-opacity="0.85"
        stroke-width="2.6" stroke-dasharray="4 6" stroke-linecap="round"/>
  <circle cx="73.5" cy="24" r="7.2" fill="{ORANGE}" stroke="#fff" stroke-width="2.4"/>
</svg>
"""


def save_assets(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    logo_pixmap(1024).save(str(folder / "logo.png"))
    logo_pixmap(256).save(str(folder / "logo_256.png"))
    (folder / "logo.svg").write_text(SVG, encoding="utf-8")
