"""Clean Voice's own controls for the one window (docs/plans/J10-one-window.md §7).

- ``PowerButton``: the header's power button; its ring says what the call hears (``signal``, ``line`` or ``hot``).
- ``StepToggle``: Step 2's Denoise, Voice style and Fine-tune switches: an outline and an LED when on.
- ``BarSlider``: a slim slider with its value on the right; it keeps the original's handling (main.js:62-122).
- ``LevelMeter``: OBS's audio mixer meter, with its dB scale, zones, peak hold and decay (VolumeMeter.cpp).
- ``ToggleSwitch``, ``IconButton``, ``TestButton`` (Test your mic, which becomes its own 5-second progress bar).
- ``fader_db``/``fader_deflection``/``db_text``: OBS's fader mapping and its dB readout, for the volume sliders.

Every widget paints only on a change (the status tick must leave the GIL to the audio threads) and reads its colours
from ``theme.current()`` when it paints, so a light/dark switch is one repaint.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

from PySide6.QtCore import QByteArray, QEvent, QPointF, QRectF, QSize, Qt, QVariantAnimation, Signal
from PySide6.QtGui import (
    QColor,
    QFocusEvent,
    QFont,
    QFontMetricsF,
    QHoverEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QResizeEvent,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QAbstractButton, QLineEdit, QSizePolicy, QWidget

from clean_voice import theme

# Clean Voice's own line icons, on a 24 x 24 grid, drawn at 1.5 px.
ICONS = {
    "headphones": '<path d="M4 15v-3a8 8 0 0 1 16 0v3" /><rect x="3" y="14" width="4.5" height="6.5" rx="1.5" />'
    '<rect x="16.5" y="14" width="4.5" height="6.5" rx="1.5" />',
    "chevron": '<path d="M7.5 10l4.5 4.5 4.5-4.5" />',
    "tick": '<path d="M4.5 12.5l4.5 4.5 10.5-10.5" />',
    # The filter window's (J10 plan §7): the eye, the eye crossed out, add, remove, up and down.
    "eye": '<path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6z" /><circle cx="12" cy="12" r="2.75" />',
    "eye-off": '<path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6z" /><path d="M4.5 4.5l15 15" />',
    "plus": '<path d="M12 5v14M5 12h14" />',
    "bin": '<path d="M4.5 7h15M9.5 7V4.5h5V7M6.5 7l1 12.5h9l1-12.5M10 10.5v5.5M14 10.5v5.5" />',
    "chevron-up": '<path d="M7.5 14l4.5-4.5 4.5 4.5" />',
}
STROKE = 1.5


@lru_cache(maxsize=128)
def _icon(name: str, color: str, stroke: float) -> QSvgRenderer:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        f'<g fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        f"{ICONS[name]}</g></svg>"
    )
    return QSvgRenderer(QByteArray(svg.encode()))


def paint_icon(p: QPainter, name: str, rect: QRectF, color: str, stroke: float = STROKE) -> None:
    _icon(name, color, stroke).render(p, rect)


def js_round(x: float) -> float:
    """Math.round: halves go up (Python's round goes to even)."""
    return math.floor(x + 0.5)


def ui_font(widget: QWidget, pixels: float, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """The widget's own font (Barlow with its fallbacks) at a size and weight."""
    f = QFont(widget.font())
    f.setPixelSize(round(pixels))
    f.setWeight(weight)
    return f


def tabular(font: QFont) -> QFont:
    """The font with tabular figures, so timers and dB values do not jitter (§7 Type)."""
    f = QFont(font)
    f.setFeature(QFont.Tag("tnum"), 1)
    return f


def keyboard_focus(widget: QWidget) -> bool:
    """Focus reached with the keyboard: only then is a focus ring drawn (a click or the start-up focus draws none)."""
    return widget.hasFocus() and widget.window().testAttribute(Qt.WidgetAttribute.WA_KeyboardFocusChange)


def _focus_ring(p: QPainter, rect: QRectF, radius: float, color: str | None = None) -> None:
    """The 2 px keyboard-focus ring, ``desk`` unless the control is itself filled ``desk``: then ``ink``, as the
    primary style-sheet buttons do, so the ring never disappears into the fill (round 15)."""
    p.setPen(QPen(QColor(color or theme.current().desk), 2))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(rect, radius, radius)


def _mix(a: str, b: str, t: float) -> QColor:
    """``a`` moved ``t`` (0..1) of the way to ``b``."""
    ca, cb = QColor(a), QColor(b)
    return QColor(
        round(ca.red() + (cb.red() - ca.red()) * t),
        round(ca.green() + (cb.green() - ca.green()) * t),
        round(ca.blue() + (cb.blue() - ca.blue()) * t),
    )


def _enter_clicks(button: QAbstractButton, event: QKeyEvent) -> bool:
    """Enter and Return work a button too (Space already does, as in every QAbstractButton)."""
    if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and button.isEnabled():
        button.click()
        return True
    return False


# ------------------------------------------------------------------ OBS's fader and its dB readout

# OBS's mixer fader is OBS_FADER_LOG (frontend/components/VolumeControl.cpp:49), whose deflection 0..1 maps to dB as
# libobs/obs-audio-controls.c log_def_to_db and log_db_to_def do (:159-185): 0 dB at the top, -96 dB and below at the
# bottom, with a 6 dB offset that spreads the loud end over most of the travel. Our own formulas from those constants.
FADER_OFFSET_DB = 6.0  # LOG_OFFSET_DB
FADER_RANGE_DB = 96.0  # LOG_RANGE_DB
INF_BELOW_DB = -96.0  # VolumeControl::updateText :854: below -96 dB the label reads "-inf dB"


def fader_db(deflection: float) -> float:
    """The fader's dB at ``deflection`` (0..1); -inf at the bottom."""
    if deflection >= 1.0:
        return 0.0
    if deflection <= 0.0:
        return -math.inf
    span = FADER_RANGE_DB + FADER_OFFSET_DB
    return -span * (span / FADER_OFFSET_DB) ** (-deflection) + FADER_OFFSET_DB


def fader_deflection(db: float) -> float:
    """The inverse: where ``db`` sits on the fader (0..1)."""
    if db >= 0.0:
        return 1.0
    if db <= -FADER_RANGE_DB:
        return 0.0
    top = -math.log10(FADER_OFFSET_DB)
    bottom = -math.log10(FADER_RANGE_DB + FADER_OFFSET_DB)
    return (-math.log10(-db + FADER_OFFSET_DB) - bottom) / (top - bottom)


def db_text(db: float) -> str:
    """OBS's volume label (VolumeControl::updateText :849-860): one decimal and " dB", or "-inf dB" below -96."""
    if not db >= INF_BELOW_DB:  # also NaN and -inf
        return "-inf dB"
    return f"{0.0 if abs(db) < 0.05 else db:.1f} dB"


# ------------------------------------------------------------------ BarSlider


class _ValueEdit(QLineEdit):
    """The field a BarSlider's value turns into: Enter or leaving applies, Esc cancels."""

    finished = Signal(bool)  # True: apply what was typed

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setFrame(False)
        self.setFont(tabular(ui_font(self, 13, QFont.Weight.Medium)))
        pal = theme.current()
        self.setStyleSheet(
            f"QLineEdit {{ background: {pal.surface}; color: {pal.ink}; border: 1px solid {pal.desk};"
            " border-radius: 4px; padding: 0 4px; }"
        )
        self._done = False
        self.returnPressed.connect(lambda: self._finish(True))

    def _finish(self, apply: bool) -> None:
        if not self._done:
            self._done = True
            self.finished.emit(apply)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._finish(False)
            event.accept()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            super().keyPressEvent(event)  # emits returnPressed, which ends the edit
            event.accept()  # the slider below must not see this Enter and open the field again
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self._finish(True)


class BarSlider(QWidget):
    """A 0..100 slider: the label above, a 3 px track filled in ``desk`` with a 14 px handle, the value on the right
    of the track. It keeps the original's handling (main.js:64-122): click or drag anywhere on the track (left of it
    gives 0, right of it 100), arrows +-1 (Shift +-10), Home/End, double-click resets (unless ``reset`` is None).
    ``valueChanged`` fires only when the guest changes the value; ``set_value`` is silent. Disabled, it is drawn grey;
    ``dimmed`` draws it grey too but keeps it working (Step 2's rows while their switch is off)."""

    valueChanged = Signal(int)
    LABEL_H = 20
    TRACK_H = 24
    VALUE_W = 64
    HANDLE_R = 7.0

    def __init__(
        self,
        label: str,
        value: float = 50,
        reset: float | None = 50,
        parent: QWidget | None = None,
        formatter: Callable[[int], str] | None = None,
        compact: bool = False,
        value_width: int | None = None,
        parse: Callable[[str], float | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.parse = parse  # typed text -> slider position (None: not a number); default: the number is the position
        self.parse_disabled = False  # the caller cannot map typed text to a position right now
        self._editor: _ValueEdit | None = None
        self.LABEL_H = 0 if compact else type(self).LABEL_H  # compact: one row, the label is the tooltip
        self.VALUE_W = value_width if value_width is not None else (46 if compact else type(self).VALUE_W)
        if compact:
            self.setToolTip(label)
        self.label = label
        self.reset = None if reset is None else int(js_round(reset))
        self.formatter = formatter or str
        self.dimmed = False
        self._value = 0
        self._dragging = False
        self._hover = False  # the pointer is over the rows a press acts on (below the label row)
        self.setFixedHeight(self.LABEL_H + self.TRACK_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self._sync_cursor()
        self.setAccessibleName(label)
        self.set_value(value)

    def sizeHint(self) -> QSize:
        return QSize(300, self.height())

    def minimumSizeHint(self) -> QSize:
        return QSize(self.VALUE_W + 80, self.height())

    def value(self) -> int:
        return self._value

    def value_text(self) -> str:
        return self.formatter(self._value)

    def set_value(self, value: float) -> None:
        v = int(js_round(max(0.0, min(100.0, float(value)))))
        if v != self._value:
            self._value = v
            self.setAccessibleDescription(self.value_text())
            self.update()

    def set_dimmed(self, dimmed: bool) -> None:
        if dimmed != self.dimmed:
            self.dimmed = dimmed
            self.update()

    def refresh_text(self) -> None:
        """The formatter's words changed for the same value (the mic's dB came in): repaint and tell screen readers."""
        self.setAccessibleDescription(self.value_text())
        self.update()

    def _change(self, value: float) -> None:
        before = self._value
        self.set_value(value)
        if self._value != before:
            self.valueChanged.emit(self._value)

    def track_rect(self) -> QRectF:
        """Where a click maps to a value: the track's row, inset by the handle, left of the value. The label row above
        takes no clicks."""
        left = self.HANDLE_R
        right = self.width() - self.VALUE_W - self.HANDLE_R
        return QRectF(left, float(self.LABEL_H), max(1.0, right - left), float(self.height() - self.LABEL_H))

    def _on_label(self, event: QMouseEvent) -> bool:
        return event.position().y() < self.LABEL_H

    def cut_off(self) -> bool:
        """Whether the label or the value does not fit its room."""
        if self.LABEL_H and QFontMetricsF(ui_font(self, 13, QFont.Weight.Medium)).horizontalAdvance(self.label) > (
            self.width()
        ):
            return True
        metrics = QFontMetricsF(tabular(ui_font(self, 13, QFont.Weight.Medium)))
        return metrics.horizontalAdvance(self.value_text()) > self.VALUE_W

    def _track_y(self) -> float:
        return self.LABEL_H + self.TRACK_H / 2.0

    def _from_x(self, x: float) -> float:
        track = self.track_rect()
        return (x - track.left()) / track.width() * 100.0

    # --- typing a value: a click on the value text (or Enter/F2) swaps it for a small field in the same spot.
    # Clean Voice's own addition (the user asked, 2026-11-20); OBS's volume control has a spin box beside its fader.

    def _on_value(self, event: QMouseEvent) -> bool:
        pos = event.position()
        return pos.x() >= self.width() - self.VALUE_W and pos.y() >= self.LABEL_H

    def editing(self) -> bool:
        return self._editor is not None

    def _typed_position(self, text: str) -> float | None:
        """The slider position for what the guest typed, or None when it is not a number."""
        if self.parse is not None:
            return self.parse(text)
        try:
            number = float(text.replace(",", ".").strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None

    def begin_edit(self) -> None:
        if self._editor is not None or not self.isEnabled() or self.parse_disabled:
            return
        editor = _ValueEdit(self)
        shown = self.value_text()
        for unit in (" dB", " %"):
            shown = shown.removesuffix(unit)
        editor.setText(shown)
        editor.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        editor.setAccessibleName(f"{self.label} value")
        editor.setGeometry(self.width() - self.VALUE_W, self.LABEL_H, self.VALUE_W, self.TRACK_H)
        editor.finished.connect(self._end_edit)
        self._editor = editor
        editor.show()
        editor.setFocus()
        editor.selectAll()

    def _end_edit(self, apply: bool) -> None:
        editor, self._editor = self._editor, None
        if editor is None:
            return
        text = editor.text()
        editor.hide()
        editor.setParent(None)  # no longer one of the slider's children; freed later
        editor.deleteLater()
        self.setFocus()
        if apply:
            position = self._typed_position(text)
            if position is not None:
                self._change(position)  # clamps to 0..100 and rounds to the closest position
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        if self._editor is not None:
            self._editor.setGeometry(self.width() - self.VALUE_W, self.LABEL_H, self.VALUE_W, self.TRACK_H)
        super().resizeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        if self._on_label(event):  # a click on the name sets nothing (home-gain-labels L1)
            return
        if self._on_value(event):  # the value text is for typing, not for dragging
            self.begin_edit()
            return
        self._dragging = True
        self._sync_cursor()
        self._change(self._from_x(event.position().x()))
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._change(self._from_x(event.position().x()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._dragging = False
            pos = event.position()
            self._set_hover(QRectF(self.rect()).contains(pos) and pos.y() >= self.LABEL_H)
            self._sync_cursor()
            self.update()

    def event(self, event: QEvent) -> bool:
        kind = event.type()
        if kind in (QEvent.Type.HoverEnter, QEvent.Type.HoverMove) and isinstance(event, QHoverEvent):
            self._set_hover(event.position().y() >= self.LABEL_H)
        elif kind == QEvent.Type.HoverLeave:
            self._set_hover(False)
        return super().event(event)

    def _set_hover(self, on: bool) -> None:
        if on != self._hover:  # repaint and re-cursor only when the state changes
            self._hover = on
            self._sync_cursor()
            self.update()

    def _sync_cursor(self) -> None:
        # The hand only where a press acts (home-gain-labels m1); compact or disabled sliders keep it everywhere.
        if not self.LABEL_H or not self.isEnabled() or self._dragging or self._hover:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

    @property
    def handle_grown(self) -> bool:
        """The handle is drawn larger: grabbed, or the pointer over the track."""
        return self.isEnabled() and (self._dragging or self._hover)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self.reset is not None and not self._on_label(event) and not self._on_value(event):
            self._change(self.reset)

    @property
    def dragging(self) -> bool:
        return self._dragging

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.EnabledChange:
            if not self.isEnabled():
                self._dragging = False  # a disabled widget never gets the release
            self._sync_cursor()
            self.setAccessibleDescription(self.value_text())  # the formatter may show the state ("–" when greyed)
        super().changeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
        key = int(event.key())
        moves = {Qt.Key.Key_Right: step, Qt.Key.Key_Up: step, Qt.Key.Key_Left: -step, Qt.Key.Key_Down: -step}
        move = next((d for k, d in moves.items() if int(k) == key), None)
        if move is not None:
            self._change(self._value + move)
        elif key == int(Qt.Key.Key_Home):
            self._change(0)
        elif key == int(Qt.Key.Key_End):
            self._change(100)
        elif key in (int(Qt.Key.Key_Return), int(Qt.Key.Key_Enter), int(Qt.Key.Key_F2)):
            self.begin_edit()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.LABEL_H:
            p.setPen(QColor(pal.ink))
            p.setFont(ui_font(self, 13, QFont.Weight.Medium))
            p.drawText(QRectF(0, 0, self.width(), self.LABEL_H), Qt.AlignmentFlag.AlignVCenter, self.label)
        track = self.track_rect()
        y = self._track_y()
        hx = track.left() + track.width() * self._value / 100.0
        lit = self.isEnabled() and not self.dimmed
        fill = pal.desk if lit else pal.muted
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(pal.line))
        p.drawRoundedRect(QRectF(track.left(), y - 1.5, track.width(), 3.0), 1.5, 1.5)
        p.setBrush(QColor(fill))
        p.drawRoundedRect(QRectF(track.left(), y - 1.5, hx - track.left(), 3.0), 1.5, 1.5)
        # The handle: a disc in a 2 px surface ring, a larger one when grabbed, a focus ring for the keyboard.
        r = self.HANDLE_R + (1.0 if self.handle_grown else 0.0)
        p.setPen(QPen(QColor(pal.surface), 2.0))
        p.setBrush(QColor(fill))
        p.drawEllipse(QPointF(hx, y), r, r)
        if keyboard_focus(self):
            p.setPen(QPen(QColor(pal.desk), 2.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(hx, y), r + 4.0, r + 4.0)
        p.setPen(QColor(pal.ink if self.isEnabled() else pal.muted))
        p.setFont(tabular(ui_font(self, 13, QFont.Weight.Medium)))
        box = QRectF(self.width() - self.VALUE_W, self.LABEL_H, self.VALUE_W, self.TRACK_H)
        p.drawText(box, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, self.value_text())


# ------------------------------------------------------------------ the header's power button


class PowerButton(QAbstractButton):
    """The power button on the header band: a 56 px ring around the power glyph. Checked is on (the voice is
    cleaned); off is a bypass. ``tone`` is the ring's colour: "signal" while the call hears the clean voice, "line"
    when the power is off or the engine is starting, "hot" when there is a problem. The ring changes over 150 ms, in
    answer to the change (§7 Motion)."""

    RING = 56
    FADE_MS = 150

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tone = "line"
        self._from_tone = "line"
        self._amount = 1.0  # how far the ring has gone from ``_from_tone`` to ``tone``
        self._fade = QVariantAnimation(self)
        self._fade.setDuration(self.FADE_MS)
        self._fade.valueChanged.connect(self._step)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedSize(self.RING + 8, self.RING + 8)  # room for the focus ring

    def sizeHint(self) -> QSize:
        return QSize(self.RING + 8, self.RING + 8)

    def set_tone(self, tone: str) -> None:
        if tone == self.tone:
            return
        self._from_tone, self.tone = self.tone, tone
        if self.isVisible():
            self._fade.stop()
            self._fade.setStartValue(0.0)
            self._fade.setEndValue(1.0)
            self._fade.start()
        else:
            self._amount = 1.0
            self.update()

    def _step(self, value: object) -> None:
        self._amount = float(value)  # type: ignore[arg-type]
        self.update()

    def ring_color(self) -> QColor:
        pal = theme.current()
        return _mix(getattr(pal, self._from_tone), getattr(pal, self.tone), self._amount)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not _enter_clicks(self, event):
            super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(self.width() / 2.0, self.height() / 2.0)
        r = self.RING / 2.0 - 2.0
        ring = self.ring_color()
        if self.underMouse() or self.isDown():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(pal.hover))
            p.drawEllipse(c, r, r)
        p.setPen(QPen(ring, 3.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, r, r)
        # The power glyph: an open circle with a bar through its gap, drawn for Clean Voice.
        glyph = QPen(QColor(pal.ink), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(glyph)
        g = 10.0
        # Qt's angles run counter-clockwise from 3 o'clock: 290 degrees from 125 leave the 70-degree gap at the top.
        p.drawArc(QRectF(c.x() - g, c.y() - g + 1.5, 2 * g, 2 * g), 125 * 16, 290 * 16)
        p.drawLine(QPointF(c.x(), c.y() - g - 1.5), QPointF(c.x(), c.y() + 1.5))
        if keyboard_focus(self):
            p.setPen(QPen(QColor(pal.desk), 2.0))
            p.drawEllipse(c, r + 4.0, r + 4.0)


# ------------------------------------------------------------------ Step 2's switches


class StepToggle(QAbstractButton):
    """A Step 2 switch (Denoise, Voice style, Fine-tune (advanced)): a 36 px squared button (radius 6) with a small
    LED square on its left. On: a 2 px ``desk`` outline, ``ink`` words, the LED lit ``signal``. Off: a 1 px ``line``
    outline, ``muted`` words, the LED unlit. The outline and the LED both carry the state, never colour alone."""

    H = 36
    W = 184
    PAD = 2  # room around the button for the focus ring

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setText(text)
        self.setAccessibleName(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setFixedSize(self.W + 2 * self.PAD, self.H + 2 * self.PAD)

    def sizeHint(self) -> QSize:
        return QSize(self.W + 2 * self.PAD, self.H + 2 * self.PAD)

    def cut_off(self) -> bool:
        return QFontMetricsF(ui_font(self, 13, QFont.Weight.Medium)).horizontalAdvance(self.text()) > self.W - 40

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not _enter_clicks(self, event):
            super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        on = self.isChecked()
        box = QRectF(self.PAD, self.PAD, self.W, self.H)
        width = 2.0 if on else 1.0
        p.setPen(QPen(QColor(pal.desk if on else pal.line), width))
        p.setBrush(QColor(pal.hover if self.underMouse() else pal.surface))
        p.drawRoundedRect(box.adjusted(width / 2, width / 2, -width / 2, -width / 2), 6, 6)
        led = QRectF(box.left() + 14, box.center().y() - 4, 8, 8)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(pal.signal if on else pal.line))
        p.drawRoundedRect(led, 1.5, 1.5)
        p.setPen(QColor(pal.ink if on else pal.muted))
        p.setFont(ui_font(self, 13, QFont.Weight.Medium))
        p.drawText(box.adjusted(32, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())
        if keyboard_focus(self):
            _focus_ring(p, QRectF(self.rect()).adjusted(1, 1, -1, -1), 7)


# ------------------------------------------------------------------ buttons


class ToggleSwitch(QAbstractButton):
    """A squared pill switch (radius 6), 40 x 22, filled in ``desk`` when on."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedSize(self.sizeHint())

    def sizeHint(self) -> QSize:
        return QSize(46, 28)  # the 40 x 22 pill plus room for the focus ring

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not _enter_clicks(self, event):
            super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QRectF(3, 3, 40, 22)
        on = self.isChecked()
        if not self.isEnabled():
            p.setOpacity(0.45)
        p.setPen(Qt.PenStyle.NoPen if on else QPen(QColor(pal.muted), 1.2))
        p.setBrush(QColor(pal.desk if on else pal.surface))
        p.drawRoundedRect(track.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(pal.on_desk if on else pal.muted))
        knob = 16.0
        x = track.right() - 3 - knob if on else track.left() + 3
        p.drawRoundedRect(QRectF(x, track.top() + 3, knob, knob), 4, 4)
        if keyboard_focus(self):
            _focus_ring(p, track.adjusted(-2, -2, 2, 2), 8)


class IconButton(QAbstractButton):
    """A squared button with a line icon and its words ("Hear myself"); checkable ones fill in ``desk`` when on."""

    def __init__(
        self, icon: str, tooltip: str, checkable: bool = False, text: str = "", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.icon_name = icon
        self.setText(text)
        self.setCheckable(checkable)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setFixedHeight(36)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        if not self.text():
            return QSize(36, 36)
        width = QFontMetricsF(ui_font(self, 13, QFont.Weight.Medium)).horizontalAdvance(self.text())
        return QSize(math.ceil(width) + 50, 36)

    def cut_off(self) -> bool:
        width = QFontMetricsF(ui_font(self, 13, QFont.Weight.Medium)).horizontalAdvance(self.text())
        return bool(self.text()) and width > self.width() - 40

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not _enter_clicks(self, event):
            super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        on = self.isCheckable() and self.isChecked()
        bg = pal.desk if on else (pal.hover if self.underMouse() else pal.surface)
        p.setPen(Qt.PenStyle.NoPen if on else QPen(QColor(pal.line), 1))
        p.setBrush(QColor(bg))
        p.drawRoundedRect(r, 6, 6)
        ink = pal.on_desk if on else pal.ink
        left = r.left() + 10 if self.text() else r.center().x() - 8  # an icon alone sits in the middle
        paint_icon(p, self.icon_name, QRectF(left, r.center().y() - 8, 16, 16), ink)
        if self.text():
            p.setPen(QColor(ink))
            p.setFont(ui_font(self, 13, QFont.Weight.Medium))
            p.drawText(r.adjusted(34, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter, self.text())
        if keyboard_focus(self):
            _focus_ring(p, QRectF(self.rect()).adjusted(1, 1, -1, -1), 7, pal.ink if on else None)


class TestButton(QAbstractButton):
    """Test your mic: the filled desk primary button. While a test runs it becomes its own 5-second progress bar
    ("Speak now", then "Playing back"), and a click cancels the test."""

    def __init__(self, parent: QWidget | None = None, height: int = 36) -> None:
        super().__init__(parent)
        self.progress = 0.0  # 0..1 of the current phase; only drawn while ``running``
        self.running = False
        self.setText("Test your mic")
        self.setToolTip("Record 5 seconds, then hear them played back")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(140, self.height())

    def set_phase(self, text: str, running: bool, progress: float) -> None:
        progress = max(0.0, min(1.0, progress))
        if (text, running, progress) != (self.text(), self.running, self.progress):
            self.setText(text)
            self.running, self.progress = running, progress
            self.setToolTip("Click to stop the test" if running else "Record 5 seconds, then hear them played back")
            self.update()

    def _paint_font(self) -> QFont:
        return tabular(ui_font(self, 13, QFont.Weight.Medium if self.running else QFont.Weight.DemiBold))

    def cut_off(self) -> bool:
        """Measured with the font it paints in."""
        return QFontMetricsF(self._paint_font()).horizontalAdvance(self.text()) > self.width() - 12

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not _enter_clicks(self, event):
            super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        if not self.isEnabled():
            p.setOpacity(0.5)
        filled = not self.running  # idle: the primary button; running: the progress bar on the surface
        p.setPen(QPen(QColor(pal.desk), 1))
        p.setBrush(QColor(pal.desk if filled else pal.surface))
        p.drawRoundedRect(r, 6, 6)
        if self.running and self.progress > 0:
            fill = QColor(pal.desk)
            fill.setAlphaF(0.25)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(fill)
            p.drawRoundedRect(QRectF(r.left(), r.top(), max(12.0, r.width() * self.progress), r.height()), 6, 6)
        p.setPen(QColor(pal.on_desk if filled else pal.ink))
        p.setFont(self._paint_font())
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.text())
        if keyboard_focus(self):
            _focus_ring(p, QRectF(self.rect()).adjusted(1, 1, -1, -1), 7, pal.ink if filled else None)


# ------------------------------------------------------------------ the level meter (OBS's VolumeMeter)


class LevelMeter(QWidget):
    """OBS's audio mixer meter, horizontal, from reference/obs-studio/frontend/components/VolumeMeter.cpp (its
    approach, written anew here):

    - **Scale:** linear in dB from -60 dB (minimumLevel, :314) to 0 dB; a position is
      ``length - int(dB * length / -60)`` (updateBackgroundCache :707-710, paintEvent :784). Under the bar, a 2 px
      tick and its number every 6 dB from 0 down to -60 (paintHTicks :604-632, TICK_DB_INTERVAL :18, TICK_SIZE :17),
      each number centred on its tick, 0's ending at its tick.
    - **Zones:** nominal below -20 dB, warning to -9 dB, error above (warningLevel/errorLevel :315-316, the sample-peak
      meter's levels :269-277), each drawn dim behind the bar and bright in it (:697-727, :803-821).
    - **Peak:** rises at once, falls 11.76 dB/s (peakDecayRate :319, calculateBallisticsForChannel :519-529). A peak at
      0 dB fills the bar in the error colour for 1 s (CLIP_FLASH_DURATION_MS :16, :804-810).
    - **Peak hold:** a 3 px mark in its zone's colour, held 20 s (peakHoldDuration :321, :531-544, :823-839).
    - **Magnitude:** a 3 px mark (:841-850) that integrates to 99 % in 300 ms (magnitudeIntegrationTime :320,
      :561-573), drawn only when a magnitude is given.
    - **Input indicator:** a 3 px square left of the bar, its colour the peak held 1 s (inputPeakHoldDuration :322,
      :546-559, getPeakColor :585-602, minimumInputLevel -50 dB :318).
    - **Idle:** no audio resets everything (detectIdle/resetLevels :397-412, :506-515).

    Clean Voice differs where it must: one mono channel, so one bar 6 px thick (doLayout's
    ``clamp(floor(22 / channels), 3, 6)``, :496-497); the engine reports one peak per meter, so the input indicator
    reads that peak too (OBS's is the source's input before its fader), and the magnitude mark waits for a magnitude
    (``set_level``'s ``magnitude``). Its colours are the theme's: ``signal``/``hot``/``clip`` for the zones, ``ink`` for
    the magnitude and the clip square, ``muted`` for the scale; ``muted`` greys replace OBS's disabled colours when the
    call can't hear this. When the meter is too narrow for every number, every other number is left out (the ticks
    stay); OBS sizes its meter to fit them instead (sizeHint :891-907)."""

    MIN_DB = -60.0  # minimumLevel
    WARNING_DB = -20.0  # warningLevel
    ERROR_DB = -9.0  # errorLevel
    CLIP_DB = 0.0  # clipLevel
    MIN_INPUT_DB = -50.0  # minimumInputLevel
    PEAK_DECAY_DB_S = 11.76  # peakDecayRate: 20 dB in 1.7 s
    MAGNITUDE_S = 0.3  # magnitudeIntegrationTime
    PEAK_HOLD_S = 20.0  # peakHoldDuration
    INPUT_HOLD_S = 1.0  # inputPeakHoldDuration
    CLIP_FLASH_S = 1.0  # CLIP_FLASH_DURATION_MS
    TICK_DB = 6  # TICK_DB_INTERVAL
    TICK_SIZE = 2
    INDICATOR = 3  # INDICATOR_THICKNESS
    THICKNESS = 6  # one channel
    MARK = 3  # the peak-hold and magnitude marks
    FONT_PX = 12  # the scale's numbers: the hint size, the smallest the window uses

    def __init__(self, clean: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.clean = clean  # the call's meter (Step 3) rather than the mic's (Step 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self.THICKNESS + self.TICK_SIZE + 2 + 15)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.active = True
        self.muted = False  # the level is real but does not reach the call: OBS's disabled (grey) colours
        self._now = 0.0
        self.reset_levels()

    def sizeHint(self) -> QSize:
        return QSize(280, self.height())

    def minimumSizeHint(self) -> QSize:
        return QSize(120, self.height())

    # -- the numbers

    def reset_levels(self) -> None:
        self.peak = -math.inf  # the display peak, dB
        self.peak_hold = -math.inf
        self.input_hold = -math.inf
        self.magnitude = -math.inf
        self._hold_at = self._input_at = 0.0
        self._clip_until = -1.0

    def set_muted(self, muted: bool) -> None:
        if muted != self.muted:
            self.muted = muted
            self.update()

    @staticmethod
    def to_db(level: float) -> float:
        return 20.0 * math.log10(level) if level > 0.0 else -math.inf

    def set_level(self, peak: float, dt: float, magnitude: float | None = None) -> None:
        """One status tick: ``peak`` (0..1, linear) since the last tick, ``dt`` seconds later, and optionally the
        magnitude (RMS, 0..1). Repaints only when a drawn position changed."""
        before = self._drawn()
        self._now += max(0.0, dt)
        if not self.active:
            self.reset_levels()
        else:
            self._ballistics(self.to_db(peak), None if magnitude is None else self.to_db(magnitude), dt)
        if self._drawn() != before:
            self.update()

    def _ballistics(self, current: float, magnitude: float | None, dt: float) -> None:
        if current >= self.peak or math.isnan(self.peak):
            self.peak = current  # the attack is immediate
        else:
            self.peak = min(max(self.peak - self.PEAK_DECAY_DB_S * dt, min(current, 0.0)), 0.0)
        # The hold's attack is immediate; after its hold time it falls back to the peak.
        held = self._now - self._hold_at > self.PEAK_HOLD_S
        if current >= self.peak_hold or not math.isfinite(self.peak_hold) or held:
            self.peak_hold, self._hold_at = current, self._now
        held = self._now - self._input_at > self.INPUT_HOLD_S
        if current >= self.input_hold or not math.isfinite(self.input_hold) or held:
            self.input_hold, self._input_at = current, self._now
        if magnitude is not None:
            if not math.isfinite(self.magnitude):
                self.magnitude = magnitude
            else:
                step = (magnitude - self.magnitude) * (dt / self.MAGNITUDE_S) * 0.99
                self.magnitude = min(max(self.magnitude + step, self.MIN_DB), 0.0)
        length = self.meter_length()
        if self.position(self.peak, length) >= self.position(self.CLIP_DB, length) and self._clip_until < self._now:
            self._clip_until = self._now + self.CLIP_FLASH_S

    @property
    def clipping(self) -> bool:
        return self._now < self._clip_until

    @property
    def fraction(self) -> float:
        """The display peak's place on the scale, 0..1."""
        length = 1000
        return self.position(self.peak, length) / length

    # -- the geometry

    def meter_length(self) -> int:
        return max(1, self.width() - (self.INDICATOR + 2))

    @classmethod
    def position(cls, db: float, length: int) -> int:
        """Where ``db`` falls on a bar ``length`` px long, from its start (VolumeMeter.cpp :707-710, :784)."""
        if math.isnan(db) or db == -math.inf:
            return 0
        return max(0, min(length, length - int(db * (length / cls.MIN_DB))))

    def zone_color(self, db: float, bright: bool = True) -> str:
        pal = theme.current()
        if self.muted:
            return pal.muted if bright else pal.line
        if db >= self.ERROR_DB:
            return pal.clip
        if db >= self.WARNING_DB:
            return pal.hot
        return pal.signal

    def input_color(self) -> QColor:
        """getPeakColor (:585-602): below -50 dB the dim nominal colour, then each zone's, then the clip colour."""
        pal = theme.current()
        db = self.input_hold
        if not db >= self.MIN_INPUT_DB:
            return self._dim(self.zone_color(self.MIN_DB))
        if db >= self.CLIP_DB:
            return QColor(pal.ink)
        return QColor(self.zone_color(db))

    def _dim(self, color: str) -> QColor:
        pal = theme.current()
        if self.muted:
            return QColor(pal.line)
        return _mix(pal.surface, color, 0.28)

    def _drawn(self) -> tuple[int, int, int, int, bool, str]:
        length = self.meter_length()
        return (
            self.position(self.peak, length),
            self.position(self.peak_hold, length),
            self.position(self.magnitude, length),
            self.position(self.input_hold, length),
            self.clipping,
            self.input_color().name(),
        )

    def tick_labels(self) -> list[tuple[int, str]]:
        """The scale's ticks (their x) and the numbers drawn under them; "" where a number is left out."""
        x0 = self.INDICATOR + 3
        width = self.width() - x0
        scale = width / self.MIN_DB
        metrics = QFontMetricsF(tabular(ui_font(self, self.FONT_PX)))
        widest = metrics.horizontalAdvance("-60")
        steps = int(-self.MIN_DB // self.TICK_DB)
        room = width / steps
        every = 1 if room >= widest + 8 else 2  # a clear gap between two numbers
        out = []
        for k in range(steps + 1):
            db = -k * self.TICK_DB
            x = int(x0 + width - (db * scale) - 1)
            out.append((x, str(db) if k % every == 0 else ""))
        return out

    def paintEvent(self, event: QPaintEvent) -> None:
        pal = theme.current()
        p = QPainter(self)
        length = self.meter_length()
        start = self.INDICATOR + 2
        warning = self.position(self.WARNING_DB, length)
        error = self.position(self.ERROR_DB, length)
        t = self.THICKNESS
        # The dim zones behind the bar.
        p.fillRect(start, 0, length, t, self._dim(self.zone_color(self.CLIP_DB)))
        p.fillRect(start, 0, error, t, self._dim(self.zone_color(self.ERROR_DB - 1)))
        p.fillRect(start, 0, warning, t, self._dim(self.zone_color(self.MIN_DB)))
        # The peak bar, bright in each zone it reaches.
        peak = self.position(self.peak, length)
        if self.active:
            if self.clipping:
                p.fillRect(start, 0, length, t, QColor(self.zone_color(self.CLIP_DB)))
            else:
                if peak > error:
                    p.fillRect(start, 0, min(peak, length), t, QColor(self.zone_color(self.CLIP_DB)))
                if peak > warning:
                    p.fillRect(start, 0, min(peak, error), t, QColor(self.zone_color(self.ERROR_DB - 1)))
                if peak > 0:
                    p.fillRect(start, 0, min(peak, warning), t, QColor(self.zone_color(self.MIN_DB)))
            hold = self.position(self.peak_hold, length)
            if hold - self.MARK > 0:
                p.fillRect(start + hold - self.MARK, 0, self.MARK, t, QColor(self.zone_color(self.peak_hold)))
            magnitude = self.position(self.magnitude, length)
            if math.isfinite(self.magnitude) and magnitude - self.MARK >= 0:
                p.fillRect(start + magnitude - self.MARK, 0, self.MARK, t, QColor(pal.ink))
            p.fillRect(0, 0, self.INDICATOR, t, self.input_color())
        # The scale under the bar.
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = tabular(ui_font(self, self.FONT_PX))
        p.setFont(font)
        metrics = QFontMetricsF(font)
        p.setPen(QColor(pal.muted))
        y = t + 1
        for x, label in self.tick_labels():
            p.fillRect(x, y, 1, self.TICK_SIZE, QColor(pal.muted))
            if not label:
                continue
            w = metrics.horizontalAdvance(label)
            left = x - w if label == "0" else max(0.0, x - w / 2.0)
            p.drawText(QPointF(left, y + 3 + metrics.capHeight()), label)
