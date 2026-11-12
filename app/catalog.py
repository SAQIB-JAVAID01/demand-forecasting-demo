"""Portfolio / project-catalog images (4:3, 1600 x 1200) made from the running app.

Run:  python app/desktop.py --catalog reports/catalog
Each image is a real screenshot of the app on a branded frame, with a caption.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPixmap

from brand import logo_pixmap

W, H = 1600, 1200
CAPTIONS = [
    ("Executive summary", "The answer in one screen: accuracy, level and risk, in plain words"),
    ("Live replay", "Real-time replay of real sales: live forecast, KPIs, anomaly feed and sales race"),
    ("Forecast accuracy", "Where the forecast is strong and where it is weak"),
    ("Products", "Every product: forecast against actual sales"),
    ("Product explorer", "History, forecast and 80% range for any product, with CSV export"),
    ("Risk and stock", "How reliable the range is, and how to use it for stock"),
    ("Data and method", "Cleaning steps, test design and what drives the forecast"),
]


def frame(shot: QPixmap, title: str, caption: str) -> QPixmap:
    pm = QPixmap(W, H)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    g = QLinearGradient(0, 0, W, H)
    g.setColorAt(0, QColor("#0f1b2d"))
    g.setColorAt(1, QColor("#1b3a66"))
    p.fillRect(0, 0, W, H, g)

    p.drawPixmap(60, 44, logo_pixmap(84))
    p.setPen(QColor("#ffffff"))
    f = QFont("Segoe UI", 27)
    f.setBold(True)
    p.setFont(f)
    p.drawText(QRectF(166, 40, W - 220, 50), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               f"Demand Forecasting Studio  ·  {title}")
    p.setPen(QColor("#c7d6ea"))
    p.setFont(QFont("Segoe UI", 15))
    p.drawText(QRectF(166, 92, W - 220, 36), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, caption)

    top, margin = 168, 60
    box_w, box_h = W - 2 * margin, H - top - 86
    scaled = shot.scaled(box_w, box_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    x, y = (W - scaled.width()) // 2, top
    clip = QPainterPath()
    clip.addRoundedRect(QRectF(x, y, scaled.width(), scaled.height()), 16, 16)
    p.fillPath(clip.translated(0, 8), QColor(0, 0, 0, 70))      # soft shadow
    p.setClipPath(clip)
    p.drawPixmap(x, y, scaled)
    p.setClipping(False)

    p.setPen(QColor("#9fb4d0"))
    p.setFont(QFont("Segoe UI", 12))
    p.drawText(QRectF(margin, H - 66, W - 2 * margin, 40), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               "Real screenshot. Public data (UCI Online Retail II). Python · pandas · LightGBM · PyQt6 · pyqtgraph. "
               "Results from a final test period never used to tune the model.")
    p.end()
    return pm


def full_page(window) -> QPixmap:
    """The current page at its full height (no scrolling), with the nav rail and top bar."""
    from PyQt6.QtWidgets import QFrame
    nav = window.findChild(QFrame, "nav").grab()
    top = window.findChild(QFrame, "topbar").grab()
    body = window.stack.currentWidget().widget().grab()
    for pm in (nav, top, body):
        pm.setDevicePixelRatio(1)
    w, h = nav.width() + max(top.width(), body.width()), max(nav.height(), top.height() + body.height())
    out = QPixmap(w, h)
    out.fill(nav.toImage().pixelColor(4, nav.height() - 4))
    p = QPainter(out)
    p.drawPixmap(0, 0, nav)
    p.fillRect(nav.width(), 0, w - nav.width(), h, body.toImage().pixelColor(3, 3))
    p.drawPixmap(nav.width(), 0, top)
    p.drawPixmap(nav.width(), top.height(), body)
    p.end()
    return out


def export(window, app, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    app.processEvents()
    for i, (title, caption) in enumerate(CAPTIONS):
        window.go(i)
        if title == "Live replay":
            window.live.snapshot(40.55)
        app.processEvents()
        app.processEvents()
        frame(full_page(window), title, caption).save(str(out / f"{i + 1}_{title.lower().replace(' ', '_')}.png"))
    window.toggle_theme()
    window.go(0)
    app.processEvents()
    app.processEvents()
    frame(full_page(window), "Dark mode", "The same app in dark mode").save(str(out / "7_dark_mode.png"))
    window.toggle_theme()
