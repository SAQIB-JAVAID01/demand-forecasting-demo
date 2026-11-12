"""Line icons for the sidebar, drawn in code on a 24 x 24 grid (no image files needed)."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF


def _pts(*xy):
    return QPolygonF([QPointF(x, y) for x, y in xy])


def _draw(kind: str, color: str, px: int) -> QPixmap:
    pm = QPixmap(px, px)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(px / 24.0, px / 24.0)
    pen = QPen(QColor(color), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    if kind == "summary":
        for x, y in ((3.5, 3.5), (13.5, 3.5), (3.5, 13.5), (13.5, 13.5)):
            p.drawRoundedRect(QRectF(x, y, 7, 7), 2, 2)
    elif kind == "live":
        p.drawPolyline(_pts((2, 12.5), (6.5, 12.5), (9.5, 5), (14, 19.5), (17, 12.5), (22, 12.5)))
    elif kind == "accuracy":
        p.drawEllipse(QPointF(12, 12), 9, 9)
        p.drawEllipse(QPointF(12, 12), 5, 5)
        p.setBrush(QColor(color))
        p.drawEllipse(QPointF(12, 12), 1.4, 1.4)
    elif kind == "products":
        p.drawPolygon(_pts((12, 3), (20.5, 7.5), (12, 12), (3.5, 7.5)))
        p.drawPolyline(_pts((3.5, 7.5), (3.5, 16.5), (12, 21), (20.5, 16.5), (20.5, 7.5)))
        p.drawLine(QPointF(12, 12), QPointF(12, 21))
    elif kind == "explorer":
        p.drawEllipse(QPointF(10.5, 10.5), 6.5, 6.5)
        p.drawLine(QPointF(15.5, 15.5), QPointF(20.5, 20.5))
    elif kind == "risk":
        path = QPainterPath(QPointF(12, 3))
        path.lineTo(19.5, 6)
        path.lineTo(19.5, 11)
        path.cubicTo(19.5, 16, 16, 19.5, 12, 21)
        path.cubicTo(8, 19.5, 4.5, 16, 4.5, 11)
        path.lineTo(4.5, 6)
        path.closeSubpath()
        p.drawPath(path)
        p.drawPolyline(_pts((8.8, 12), (11.2, 14.4), (15.5, 9.8)))
    elif kind == "method":
        p.drawPolygon(_pts((12, 3), (21, 7.5), (12, 12), (3, 7.5)))
        p.drawPolyline(_pts((3, 12), (12, 16.5), (21, 12)))
        p.drawPolyline(_pts((3, 16.5), (12, 21), (21, 16.5)))
    elif kind == "lab":
        p.drawPolyline(_pts((9, 3), (9, 9.5), (4, 19), (5.2, 21), (18.8, 21), (20, 19), (15, 9.5), (15, 3)))
        p.drawLine(QPointF(7.5, 3), QPointF(16.5, 3))
        p.drawLine(QPointF(6.4, 15), QPointF(17.6, 15))
    elif kind == "projection":
        p.drawPolyline(_pts((3, 18), (8, 13), (11.5, 15.5), (15, 10)))
        pen2 = QPen(QColor(color), 1.6, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen2)
        p.drawLine(QPointF(15, 10), QPointF(21, 4.5))
        p.drawLine(QPointF(15, 10), QPointF(21, 9.5))
        p.drawLine(QPointF(15, 10), QPointF(21, 14.5))
    elif kind == "moon":
        path = QPainterPath()
        path.addEllipse(QPointF(12, 12), 8.5, 8.5)
        cut = QPainterPath()
        cut.addEllipse(QPointF(16.5, 8), 7, 7)
        p.drawPath(path.subtracted(cut))
    elif kind == "sun":
        p.drawEllipse(QPointF(12, 12), 4, 4)
        for a in range(8):
            import math
            r = math.radians(a * 45)
            p.drawLine(QPointF(12 + 7 * math.cos(r), 12 + 7 * math.sin(r)),
                       QPointF(12 + 9.5 * math.cos(r), 12 + 9.5 * math.sin(r)))
    p.end()
    return pm


def icon(kind: str, normal: str, active: str | None = None, size: int = 18) -> QIcon:
    """An icon with one colour for the normal state and another for checked/hover."""
    ic = QIcon()
    for color, state in ((normal, QIcon.State.Off), (active or normal, QIcon.State.On)):
        for scale in (1, 2):
            pm = _draw(kind, color, size * scale)
            pm.setDevicePixelRatio(scale)
            ic.addPixmap(pm, QIcon.Mode.Normal, state)
    if active:
        for scale in (1, 2):
            pm = _draw(kind, active, size * scale)
            pm.setDevicePixelRatio(scale)
            ic.addPixmap(pm, QIcon.Mode.Active, QIcon.State.Off)
    return ic
