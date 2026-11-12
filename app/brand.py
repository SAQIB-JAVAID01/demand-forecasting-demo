"""Logo for Demand Forecasting Studio, drawn in code (no image files needed).

The mark: three rising bars for sales history, then a dashed step and a dot
for the forecast, on a rounded blue tile. Original artwork for this project.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (QBrush, QColor, QFont, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                         QRadialGradient)

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


def splash_pixmap(w: int = 640, h: int = 360) -> QPixmap:
    """Start-up card shown while the data loads: dark surface, soft blue glow, logo and name."""
    pm = QPixmap(w, h)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    card = QPainterPath()
    card.addRoundedRect(QRectF(0, 0, w, h), 18, 18)
    p.fillPath(card, QColor("#0b0e13"))
    glow = QRadialGradient(QPointF(w * 0.82, h * 0.05), w * 0.75)
    glow.setColorAt(0, QColor(59, 139, 234, 95))
    glow.setColorAt(1, QColor(59, 139, 234, 0))
    p.fillPath(card, QBrush(glow))
    p.setPen(QPen(QColor(255, 255, 255, 22), 1))
    p.drawPath(card)
    p.drawPixmap(44, 52, logo_pixmap(76))
    p.setPen(QColor("#ffffff"))
    f = QFont("Inter", 22)
    f.setWeight(QFont.Weight.Bold)
    p.setFont(f)
    p.drawText(QRectF(44, 150, w - 88, 44), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               "Demand Forecasting Studio")
    p.setPen(QColor("#a3abb7"))
    p.setFont(QFont("Inter", 11))
    p.drawText(QRectF(44, 196, w - 88, 28), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               "Weekly forecasts for 200 products, tested on a year never used to build them")
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#1a2029"))
    p.drawRoundedRect(QRectF(44, h - 70, w - 88, 4), 2, 2)
    bar = QLinearGradient(44, 0, w * 0.6, 0)
    bar.setColorAt(0, QColor("#5b8def"))
    bar.setColorAt(1, QColor("#8b5cf6"))
    p.setBrush(QBrush(bar))
    p.drawRoundedRect(QRectF(44, h - 70, (w - 88) * 0.62, 4), 2, 2)
    p.setPen(QColor("#6e7887"))
    p.setFont(QFont("Inter", 9))
    p.drawText(QRectF(44, h - 56, w - 88, 24), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               "Loading the sales history and forecasts…")
    p.end()
    return pm


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
