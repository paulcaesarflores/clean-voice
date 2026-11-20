"""The 10-band EQ's panel in Fine-tune's properties area (B012): a dark card with a frequency-response graph on top
(a dot per band to drag, the wheel over a dot changes its Q) and a row per band below it (on, number, gain, frequency,
Q, type), then the output trim. It is Clean Voice's own look in the dark palette's colours, with no branding from any
other EQ. The maths is ``parametric_eq``; the panel only edits the filter's settings and reports every change as
``changed(key, value)``, which the filter window publishes like any other property edit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPainterPath, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractButton,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from clean_voice import parametric_eq as pe
from clean_voice.theme import DARK, rgba

PAL = DARK  # the card is dark in both themes, like a piece of studio kit
# One colour per band, picked to stay apart on the dark card.
BAND_COLORS = (
    "#F06A63",
    "#F0903C",
    "#F0C43C",
    "#B6D84A",
    "#3DCB8D",
    "#3CC8D8",
    "#6FA0FF",
    "#9A86FF",
    "#D27CF0",
    "#F07CB4",
)
HIT = 14.0  # how close (px) a click must be to a dot to take it
DOT = 9.0  # a dot's radius
DB_STEP = 6.0  # a horizontal gridline every 6 dB
WHEEL_Q = 1.12  # one wheel notch multiplies Q by this
GRID_LABELS = ("31", "62", "125", "250", "500", "1k", "2k", "4k", "8k", "16k")
LOG_SPAN = math.log(pe.FREQ_RANGE[1] / pe.FREQ_RANGE[0])
CURVE_POINTS = 240


def _alpha(color: str, a: float) -> QColor:
    c = QColor(color)
    c.setAlphaF(a)
    return c


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, v))


class EqGraph(QWidget):
    """The graph: log frequency 20 Hz-20 kHz across, +-18 dB up, the summed curve filled, a numbered dot per band at
    (frequency, gain). ``moved(band, freq, gain)`` while a dot is dragged, ``q_turned(band, q)`` on the wheel."""

    moved = Signal(int, float, float)
    q_turned = Signal(int, float)

    LEFT, RIGHT, TOP, BOTTOM = 34.0, 12.0, 12.0, 22.0

    def __init__(self, values: dict[str, Any]) -> None:
        super().__init__()
        self.values = values  # the panel's settings, read live at every paint
        self.dragging: int | None = None
        self.hover: int | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(210)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("EQ curve")

    def sizeHint(self) -> QSize:
        return QSize(640, 280)

    # -------------------------------------------------------------- the two scales

    def plot(self) -> QRectF:
        return QRectF(
            self.LEFT, self.TOP, self.width() - self.LEFT - self.RIGHT, self.height() - self.TOP - self.BOTTOM
        )

    def x_of(self, freq: float) -> float:
        r = self.plot()
        return r.left() + r.width() * math.log(freq / pe.FREQ_RANGE[0]) / LOG_SPAN

    def freq_of(self, x: float) -> float:
        r = self.plot()
        return pe.FREQ_RANGE[0] * math.exp(LOG_SPAN * (x - r.left()) / r.width())

    def y_of(self, db: float) -> float:
        r = self.plot()
        return r.top() + r.height() * (pe.GAIN_RANGE[1] - db) / (pe.GAIN_RANGE[1] - pe.GAIN_RANGE[0])

    def db_of(self, y: float) -> float:
        r = self.plot()
        return pe.GAIN_RANGE[1] - (y - r.top()) / r.height() * (pe.GAIN_RANGE[1] - pe.GAIN_RANGE[0])

    # -------------------------------------------------------------- the dots

    def band(self, i: int) -> pe.Band:
        v = self.values
        return pe.Band(
            v[pe.key(i, "on")] is True,
            str(v[pe.key(i, "type")]),
            float(v[pe.key(i, "freq")]),  # type: ignore[arg-type]
            float(v[pe.key(i, "gain")]),  # type: ignore[arg-type]
            float(v[pe.key(i, "q")]),  # type: ignore[arg-type]
        )

    def dot_pos(self, i: int) -> QPointF:
        """Band ``i``'s dot: at (frequency, gain); a cut or notch has no gain, so its dot sits on the 0 dB line."""
        b = self.band(i)
        return QPointF(self.x_of(b.freq), self.y_of(b.gain if b.type in pe.GAINED else 0.0))

    def curve_db(self, freq: float) -> float:
        return float(pe.response_db(self.values, np.array([freq]))[0])

    def dot_at(self, pos: QPointF) -> int | None:
        """The nearest dot within reach (the higher number wins a tie: it is drawn on top)."""
        best, best_d = None, HIT
        for i in range(1, pe.BANDS + 1):
            p = self.dot_pos(i)
            d = math.hypot(p.x() - pos.x(), p.y() - pos.y())
            if d <= best_d:
                best, best_d = i, d
        return best

    # -------------------------------------------------------------- the mouse

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = self.dot_at(event.position())
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        i = self.dragging
        if i is None:
            over = self.dot_at(event.position())
            if over != self.hover:
                self.hover = over
                self.setCursor(Qt.CursorShape.OpenHandCursor if over else Qt.CursorShape.ArrowCursor)
                self.update()
            return
        pos = event.position()
        freq = round(_clamp(self.freq_of(pos.x()), *pe.FREQ_RANGE))
        gain = self.band(i).gain
        if self.band(i).type in pe.GAINED:
            gain = round(_clamp(self.db_of(pos.y()), *pe.GAIN_RANGE), 1)
        self.moved.emit(i, float(freq), gain)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.dragging = None
        self.hover = self.dot_at(event.position())
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        i = self.dot_at(event.position())
        if i is None:
            event.ignore()  # not over a dot: the wheel scrolls the pane
            return
        q = self.band(i).q * WHEEL_Q ** (event.angleDelta().y() / 120.0)
        self.q_turned.emit(i, round(_clamp(q, *pe.Q_RANGE), 2))
        event.accept()

    def leaveEvent(self, event: object) -> None:
        self.hover = None
        self.update()

    # -------------------------------------------------------------- painting

    def paintEvent(self, event: object) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.plot()
        p.fillRect(self.rect(), QColor(PAL.band))
        p.fillRect(r, QColor(PAL.surface))
        small = QFont(self.font())
        small.setPixelSize(11)
        p.setFont(small)
        grid = QColor(PAL.line)
        # dB gridlines, 0 dB stronger
        db = pe.GAIN_RANGE[0]
        while db <= pe.GAIN_RANGE[1] + 1e-9:
            y = self.y_of(db)
            p.setPen(QPen(QColor(PAL.muted) if db == 0 else grid, 1.2 if db == 0 else 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(QColor(PAL.muted))
            p.drawText(
                QRectF(0, y - 8, self.LEFT - 5, 16),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                f"{db:+.0f}" if db else "0",
            )
            db += DB_STEP
        # frequency gridlines
        for f, label in zip(pe.GRID_FREQS, GRID_LABELS, strict=True):
            x = self.x_of(f)
            p.setPen(QPen(grid, 1))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(QColor(PAL.muted))
            p.drawText(QRectF(x - 20, r.bottom() + 3, 40, 14), int(Qt.AlignmentFlag.AlignCenter), label)
        # the summed curve, filled down or up to the 0 dB line
        freqs = pe.FREQ_RANGE[0] * np.exp(np.linspace(0.0, LOG_SPAN, CURVE_POINTS))
        resp = np.clip(pe.response_db(self.values, freqs), *pe.GAIN_RANGE)
        line = QPainterPath(QPointF(self.x_of(float(freqs[0])), self.y_of(float(resp[0]))))
        for f, d in zip(freqs[1:], resp[1:], strict=True):
            line.lineTo(self.x_of(float(f)), self.y_of(float(d)))
        fill = QPainterPath(line)
        fill.lineTo(self.x_of(float(freqs[-1])), self.y_of(0.0))
        fill.lineTo(self.x_of(float(freqs[0])), self.y_of(0.0))
        fill.closeSubpath()
        p.save()
        p.setClipRect(r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_alpha(PAL.desk, 0.28))
        p.drawPath(fill)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(PAL.desk), 2.0))
        p.drawPath(line)
        p.restore()
        # the dots
        bold = QFont(small)
        bold.setPixelSize(11)
        bold.setWeight(QFont.Weight.Bold)
        p.setFont(bold)
        for i in range(1, pe.BANDS + 1):
            on = self.band(i).on
            c = QPointF(self.dot_pos(i))
            c.setX(_clamp(c.x(), r.left(), r.right()))
            c.setY(_clamp(c.y(), r.top(), r.bottom()))
            lit = i in (self.dragging, self.hover)
            if lit:
                halo = QColor(BAND_COLORS[i - 1] if on else PAL.muted)
                halo.setAlphaF(0.3)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(halo)
                p.drawEllipse(c, DOT + 5, DOT + 5)
            p.setPen(QPen(QColor(PAL.band), 1.5))
            p.setBrush(QColor(BAND_COLORS[i - 1] if on else "#56636F"))
            p.drawEllipse(c, DOT, DOT)
            p.setPen(QColor(PAL.on_desk if on else PAL.ink))
            p.drawText(QRectF(c.x() - DOT, c.y() - DOT, 2 * DOT, 2 * DOT), int(Qt.AlignmentFlag.AlignCenter), str(i))
        # what the band under the pointer says
        i = self.dragging or self.hover
        if i:
            b = self.band(i)
            text = f"Band {i}   {b.freq:.0f} Hz   Q {b.q:.2f}" + (f"   {b.gain:+.1f} dB" if b.type in pe.GAINED else "")
            p.setFont(small)
            p.setPen(QColor(PAL.ink))
            p.drawText(QRectF(r.left() + 8, r.top() + 4, r.width() - 16, 16), int(Qt.AlignmentFlag.AlignLeft), text)


# ------------------------------------------------------------------ a row's small controls


class _Spin(QDoubleSpinBox):
    """A typeable number box that leaves the wheel to the pane unless it has the focus (as the window's boxes do)."""

    def __init__(self, lo: float, hi: float, decimals: int, step: float, suffix: str, name: str) -> None:
        super().__init__()
        self.setRange(lo, hi)
        self.setDecimals(decimals)
        self.setSingleStep(step)
        self.setSuffix(suffix)
        # No arrows (the style sheet gives the buttons no width): Qt's NoButtons with a style sheet collapses the text.
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setKeyboardTracking(False)  # the value is taken at Enter or when the box loses focus: no half-typed jumps
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(name)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class _OnButton(QAbstractButton):
    """A band's on/off: a ring in the band's colour that fills when the band is on."""

    def __init__(self, color: str, name: str) -> None:
        super().__init__()
        self.color = color
        self.setCheckable(True)
        self.setFixedSize(26, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName(name)
        self.setToolTip(name)

    def paintEvent(self, event: object) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        on = self.isChecked()
        p.setPen(QPen(QColor(self.color if on else PAL.muted), 2))
        p.setBrush(QColor(self.color) if on else Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, 7.5, 7.5)
        if self.hasFocus():
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(PAL.desk), 2))
            p.drawEllipse(c, 11.5, 11.5)

    def sizeHint(self) -> QSize:
        return QSize(26, 26)


# Each type's icon, as a polyline on a 0..1 square (x along, y up).
SHAPES: dict[str, list[tuple[float, float]]] = {
    "bell": [(0.0, 0.3), (0.3, 0.3), (0.42, 0.55), (0.5, 0.8), (0.58, 0.55), (0.7, 0.3), (1.0, 0.3)],
    "low_shelf": [(0.0, 0.75), (0.3, 0.75), (0.7, 0.3), (1.0, 0.3)],
    "high_shelf": [(0.0, 0.3), (0.3, 0.3), (0.7, 0.75), (1.0, 0.75)],
    "low_cut": [(0.0, 0.05), (0.35, 0.3), (0.55, 0.6), (0.75, 0.75), (1.0, 0.75)],
    "high_cut": [(0.0, 0.75), (0.25, 0.75), (0.45, 0.6), (0.65, 0.3), (1.0, 0.05)],
    "notch": [(0.0, 0.7), (0.38, 0.7), (0.5, 0.05), (0.62, 0.7), (1.0, 0.7)],
}


class _TypeButton(QAbstractButton):
    """One type's button: its response shape as a small line drawing; the chosen one is filled."""

    def __init__(self, kind: str, label: str) -> None:
        super().__init__()
        self.kind = kind
        self.setCheckable(True)
        self.setFixedSize(28, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(label)
        self.setAccessibleName(label)

    def paintEvent(self, event: object) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        on = self.isChecked()
        p.setPen(QPen(QColor(PAL.desk if on else PAL.line), 1))
        p.setBrush(
            _alpha(PAL.desk, 0.22) if on else (QColor(PAL.hover) if self.underMouse() else Qt.BrushStyle.NoBrush)
        )
        p.drawRoundedRect(box, 4, 4)
        inner = box.adjusted(6, 6, -6, -6)
        path = QPainterPath()
        for n, (x, y) in enumerate(SHAPES[self.kind]):
            pt = QPointF(inner.left() + x * inner.width(), inner.bottom() - y * inner.height() / 0.8)
            path.moveTo(pt) if n == 0 else path.lineTo(pt)
        p.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(QColor(PAL.desk if on else PAL.muted), 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPath(path)
        if self.hasFocus():
            p.setPen(QPen(QColor(PAL.desk), 2))
            p.drawRoundedRect(box, 4, 4)


class BandRow:
    """One band's controls; ``on``, ``gain``, ``freq``, ``q`` and ``types`` (kind -> button) are for the tests."""

    def __init__(self, number: int) -> None:
        color = BAND_COLORS[number - 1]
        self.number = number
        self.on = _OnButton(color, f"Band {number} on")
        self.label = QLabel(str(number))
        self.label.setObjectName("eqNumber")
        self.label.setStyleSheet(f"color: {color}; font-weight: 700; font-size: 14px;")
        self.label.setFixedWidth(18)
        self.gain = _Spin(*pe.GAIN_RANGE, 1, 0.5, " dB", f"Band {number} gain")
        self.freq = _Spin(*pe.FREQ_RANGE, 0, 1.0, " Hz", f"Band {number} frequency")
        self.q = _Spin(*pe.Q_RANGE, 2, 0.1, "", f"Band {number} Q")
        self.types = {kind: _TypeButton(kind, label) for kind, label in pe.TYPES}
        for w in (self.gain, self.freq, self.q):
            w.setFixedWidth(86 if w is self.freq else 78)

    def show_type(self, kind: str) -> None:
        for k, button in self.types.items():
            button.setChecked(k == kind)
        self.gain.setEnabled(kind in pe.GAINED)  # a cut or notch has no gain


def panel_style() -> str:
    return f"""
QWidget#eqPanel {{ background: {PAL.band}; border: 1px solid {PAL.line}; border-radius: 8px; }}
QLabel {{ color: {PAL.ink}; font-size: 12px; background: transparent; }}
QLabel[role="eqHead"] {{ color: {PAL.muted}; font-size: 11px; }}
QDoubleSpinBox {{ background: {PAL.surface}; color: {PAL.ink}; border: 1px solid {PAL.line}; border-radius: 4px;
    padding: 3px 6px; font-size: 12px; selection-background-color: {rgba(PAL.desk, 0.3)}; }}
QDoubleSpinBox:focus {{ border: 1px solid {PAL.desk}; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}
QAbstractSpinBox QLineEdit {{ background: transparent; color: {PAL.ink}; border: none; padding: 0; }}
QDoubleSpinBox:disabled {{ color: {PAL.muted}; background: {PAL.band}; }}
"""


class EqPanel(QWidget):
    """The panel. ``values``: the filter's full settings, kept current. ``changed(key, value)`` for every edit."""

    changed = Signal(str, object)

    def __init__(self, settings: Mapping[str, object]) -> None:
        super().__init__()
        self.setObjectName("eqPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet(panel_style())
        self.values: dict[str, Any] = dict(settings)
        column = QVBoxLayout(self)
        column.setContentsMargins(12, 12, 12, 12)
        column.setSpacing(10)
        self.graph = EqGraph(self.values)
        self.graph.moved.connect(self._dragged)
        self.graph.q_turned.connect(lambda i, q: self._put(pe.key(i, "q"), q))
        column.addWidget(self.graph, 1)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(5)
        for col, text in ((3, "Gain"), (4, "Frequency"), (5, "Q"), (6, "Type")):
            head = QLabel(text)
            head.setProperty("role", "eqHead")
            grid.addWidget(head, 0, col)
        self.rows: list[BandRow] = []
        for n in range(1, pe.BANDS + 1):
            row = BandRow(n)
            self.rows.append(row)
            grid.addWidget(row.on, n, 0)
            grid.addWidget(row.label, n, 1)
            grid.addWidget(row.gain, n, 3)
            grid.addWidget(row.freq, n, 4)
            grid.addWidget(row.q, n, 5)
            kinds = QHBoxLayout()
            kinds.setSpacing(4)
            for button in row.types.values():
                kinds.addWidget(button)
            grid.addLayout(kinds, n, 6)
            row.on.toggled.connect(lambda on, n=n: self._put(pe.key(n, "on"), on))
            row.gain.valueChanged.connect(lambda v, n=n: self._put(pe.key(n, "gain"), round(v, 1)))
            row.freq.valueChanged.connect(lambda v, n=n: self._put(pe.key(n, "freq"), v))
            row.q.valueChanged.connect(lambda v, n=n: self._put(pe.key(n, "q"), v))
            for kind, button in row.types.items():
                button.clicked.connect(lambda _=False, n=n, kind=kind: self._put(pe.key(n, "type"), kind))
        grid.setColumnStretch(7, 1)
        column.addLayout(grid)
        trim_row = QHBoxLayout()
        label = QLabel("Output")
        label.setProperty("role", "eqHead")
        self.trim = _Spin(*pe.TRIM_RANGE, 1, 0.5, " dB", "Output trim")
        self.trim.setFixedWidth(86)
        self.trim.valueChanged.connect(lambda v: self._put("trim", round(v, 1)))
        trim_row.addWidget(label)
        trim_row.addWidget(self.trim)
        trim_row.addStretch(1)
        column.addLayout(trim_row)
        self._load()

    def _load(self) -> None:
        """Show ``values`` in every box and button (no change is emitted)."""
        for row in self.rows:
            n = row.number
            for widget, k in ((row.gain, "gain"), (row.freq, "freq"), (row.q, "q")):
                widget.blockSignals(True)
                widget.setValue(float(self.values[pe.key(n, k)]))  # type: ignore[arg-type]
                widget.blockSignals(False)
            row.on.blockSignals(True)
            row.on.setChecked(self.values[pe.key(n, "on")] is True)
            row.on.blockSignals(False)
            row.show_type(str(self.values[pe.key(n, "type")]))
        self.trim.blockSignals(True)
        self.trim.setValue(float(self.values["trim"]))  # type: ignore[arg-type]
        self.trim.blockSignals(False)

    def _put(self, key: str, value: object) -> None:
        """One setting edited, from a box, a button, the wheel or a drag."""
        if self.values.get(key) == value:
            return
        self.values[key] = value
        if key.endswith("_type"):
            self.rows[int(key[1 : key.index("_")]) - 1].show_type(str(value))
        elif key.endswith(("_freq", "_gain", "_q")):  # a drag or the wheel: the box shows it too
            n = int(key[1 : key.index("_")])
            box = {"freq": self.rows[n - 1].freq, "gain": self.rows[n - 1].gain, "q": self.rows[n - 1].q}[
                key.rsplit("_", 1)[1]
            ]
            if box.value() != value:
                box.blockSignals(True)
                box.setValue(float(value))  # type: ignore[arg-type]
                box.blockSignals(False)
        self.graph.update()
        self.changed.emit(key, value)

    def _dragged(self, band: int, freq: float, gain: float) -> None:
        if self.values[pe.key(band, "on")] is not True:  # taking a grey dot switches its band on
            row = self.rows[band - 1].on
            row.blockSignals(True)
            row.setChecked(True)
            row.blockSignals(False)
            self._put(pe.key(band, "on"), True)
        self._put(pe.key(band, "freq"), freq)
        self._put(pe.key(band, "gain"), gain)
