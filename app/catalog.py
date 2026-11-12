"""Portfolio / project-catalog images (4:3, 1600 x 1200) made from the running app.

Run:  python app/desktop.py --catalog reports/catalog
Each image is a real screenshot of the app on a branded frame, with a caption.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient

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
    f = QFont("Inter", 27)
    f.setBold(True)
    p.setFont(f)
    p.drawText(QRectF(166, 40, W - 220, 50), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               f"Demand Forecasting Studio  ·  {title}")
    p.setPen(QColor("#c7d6ea"))
    p.setFont(QFont("Inter", 15))
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
    p.setFont(QFont("Inter", 12))
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
    frame(full_page(window), "Light mode", "The same app in light mode").save(str(out / "8_light_mode.png"))
    window.toggle_theme()


def cover(window, app, path: Path) -> None:
    """Catalog cover (4:3, 1600 x 1200): the live page in dark mode on a dark stage, with the name and three tags.

    Upwork shows catalog images at 1000 x 750 and asks for little text, so the screenshot carries the image.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    window.go(1)
    window.live.cb_scope.setCurrentIndex(0)
    window.live.snapshot(40.55)
    app.processEvents()
    app.processEvents()
    shot = full_page(window)
    shot = shot.copy(0, 0, shot.width(), int(shot.width() * 0.62))   # nav, hero, live tiles and the main chart

    pm = QPixmap(W, H)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    bg = QLinearGradient(0, 0, 0, H)
    bg.setColorAt(0, QColor("#05070b"))
    bg.setColorAt(1, QColor("#0b111c"))
    p.fillRect(0, 0, W, H, bg)
    for cx, cy, r, col in ((W * 0.86, -40, 900, QColor(59, 130, 246, 80)), (W * 0.05, H * 1.05, 800, QColor(139, 92, 246, 46))):
        g = QRadialGradient(QPointF(cx, cy), r)
        g.setColorAt(0, col)
        g.setColorAt(1, QColor(col.red(), col.green(), col.blue(), 0))
        p.fillRect(0, 0, W, H, QBrush(g))

    p.drawPixmap(80, 74, logo_pixmap(72))
    p.setPen(QColor("#ffffff"))
    f = QFont("Inter", 34)
    f.setWeight(QFont.Weight.Bold)
    p.setFont(f)
    p.drawText(QRectF(176, 66, W - 240, 56), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               "Demand Forecasting Studio")
    p.setPen(QColor("#a3abb7"))
    p.setFont(QFont("Inter", 16))
    p.drawText(QRectF(178, 120, W - 240, 36), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               "Real-time sales forecasting dashboard in Python")

    x = 80
    fp = QFont("Inter", 12)
    fp.setWeight(QFont.Weight.DemiBold)
    p.setFont(fp)
    for text, col in (("Live replay", "#5b8def"), ("Forecast vs actual", "#3fcf8e"), ("Stock-risk alerts", "#f0915a")):
        tw = p.fontMetrics().horizontalAdvance(text) + 44
        r = QRectF(x, 190, tw, 38)
        c = QColor(col)
        p.setPen(QPen(QColor(c.red(), c.green(), c.blue(), 110), 1))
        p.setBrush(QColor(c.red(), c.green(), c.blue(), 30))
        p.drawRoundedRect(r, 19, 19)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawEllipse(QPointF(x + 18, 209), 4.5, 4.5)
        p.setPen(QColor("#e6e8eb"))
        p.drawText(r.adjusted(30, 0, -12, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
        x += tw + 12

    top, side = 276, 80
    sw = W - 2 * side
    scaled = shot.scaledToWidth(sw, Qt.TransformationMode.SmoothTransformation)
    win = QRectF(side, top, sw, H - top + 40)                        # runs off the bottom edge
    glow = QRadialGradient(QPointF(W / 2, top + 40), W * 0.6)
    glow.setColorAt(0, QColor(91, 141, 239, 70))
    glow.setColorAt(1, QColor(91, 141, 239, 0))
    p.fillRect(QRectF(0, top - 120, W, 500), QBrush(glow))
    clip = QPainterPath()
    clip.addRoundedRect(win, 18, 18)
    p.fillPath(clip.translated(0, 10), QColor(0, 0, 0, 120))
    p.setClipPath(clip)
    p.fillRect(win, QColor("#0b0e13"))
    p.drawPixmap(int(side), int(top), scaled)
    p.setClipping(False)
    p.setPen(QPen(QColor(255, 255, 255, 34), 1.2))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(clip)
    p.end()
    pm.save(str(path))
    pm.scaled(1000, 750, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation).save(
        str(path.with_name(path.stem + "_1000x750" + path.suffix)))
