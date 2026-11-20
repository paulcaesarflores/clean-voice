"""The Clean Voice application and its one window (docs/plans/J10-one-window.md §7, DESIGN "One-window redesign").

On top, the header band: the headline says what the call hears right now, with the power button on its right (the
one bold element). Below, five numbered steps a beginner reads top to bottom: 1 Choose your microphone (with Test your
mic and Hear myself), 2 Clean your voice (Denoise, Voice style, Fine-tune), 3 Adjust volume, 4 Send clean voice to,
5 Record backup (optional). The footer holds Start with Windows (or at login), the Theme choice, the version and Check
for updates.
Every behaviour of the original app is kept (simple-clean-mic ui/main.js, src-tauri/src/main.rs); the look and the
words are Clean Voice's own, light or dark as the guest chooses (Automatic: with the system)."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import itertools
import logging
import math
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from PySide6.QtCore import QEvent, QMargins, QObject, QPointF, QRect, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QPen,
    QTextLayout,
    QTextOption,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProxyStyle,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleFactory,
    QStyleHintReturn,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

from clean_voice import (
    __version__,
    autostart,
    filter_window,
    fonts,
    mark,
    mic_level,
    nvafx,
    prefs,
    theme,
    updates,
    vst,
)
from clean_voice.devices import CABLE_DOWNLOAD_URL, is_cable_output
from clean_voice.dsp import loudness_db
from clean_voice.engine import ECHO_MAC, EchoState, MicTestState, State, Status
from clean_voice.settings import PRESETS, Settings, replace, style_bands
from clean_voice.theme import Palette
from clean_voice.widgets import (
    BarSlider,
    IconButton,
    LevelMeter,
    PowerButton,
    StepToggle,
    TestButton,
    ToggleSwitch,
    db_text,
    fader_db,
    fader_deflection,
    keyboard_focus,
    paint_icon,
    tabular,
    ui_font,
)

if TYPE_CHECKING:
    from clean_voice.tray import Tray

APP_NAME = "Clean Voice"
SHOWN_MESSAGE = "Clean Voice window shown"
NOT_SHOWN_MESSAGE = "Clean Voice window NOT shown"
# Exit code of a --quit-after-ms run whose window never became visible.
NOT_SHOWN_EXIT = 2
# Exit code of a --quit-after-ms run whose bundled RNNoise library would not load.
NO_RNNOISE_EXIT = 3
# Exit code of a --quit-after-ms run whose build is missing one of the audio libraries.
NO_AUDIO_EXIT = 4
AUDIO_MODULES = ("numpy", "scipy.signal", "samplerate", "sounddevice")
# Echo removal (J06): AEC3 and WASAPI loopback, Windows only (macOS has no loopback capture in v1).
ECHO_MODULES = ("pywebrtc_audio", "pyaudiowpatch")


def audio_modules(platform: str = sys.platform) -> tuple[str, ...]:
    """The audio libraries a build for ``platform`` must hold; the launch check imports each one."""
    return AUDIO_MODULES + ECHO_MODULES if platform == "win32" else AUDIO_MODULES


TICK_MS = 33
HIDDEN_TICK_MS = 500  # while the window is hidden in the tray, to leave the GIL to the audio (main.rs:216-223)
SAVE_DELAY_MS = 500  # the saved choices are written this long after the last change, and on quit
# Barlow first (bundled), then the system fonts, with Japanese glyphs (Qt falls back per glyph).
UI_FONTS = fonts.body()
# J10 plan, open question 9: the opening size fits a 1366 x 768 laptop with its taskbar (768 - 48 px taskbar leaves
# 720 for the window with its title bar). All five steps show without scrolling at this size (round 13; 640 cut off
# Steps 4-5). Below it the steps scroll under the header, which never does.
WINDOW_SIZE = (760, 684)
MIN_SIZE = (680, 520)


def opening_geometry(available: QRect, frame: QMargins) -> QRect:
    """The window's first geometry (its inside, without the ``frame``): WINDOW_SIZE when it fits the screen's
    ``available`` area (QScreen.availableGeometry: the taskbar or Dock left out) with its frame, else the largest size
    that fits, never under MIN_SIZE (the steps scroll, the header and footer stay). Centred in ``available``; a frame
    taller than the screen keeps its title bar on it."""
    room_w = available.width() - frame.left() - frame.right()
    room_h = available.height() - frame.top() - frame.bottom()
    w = max(MIN_SIZE[0], min(WINDOW_SIZE[0], room_w))
    h = max(MIN_SIZE[1], min(WINDOW_SIZE[1], room_h))
    x = available.x() + max(0, (room_w - w) // 2) + frame.left()
    y = available.y() + max(0, (room_h - h) // 2) + frame.top()
    return QRect(x, y, w, h)


MARGIN = 24  # §7 Spacing: window margins
NUMBER_W = 32  # the step numbers' column; every control starts at its right edge
SECOND_W = StepToggle.W + 2 * StepToggle.PAD  # Step 2's switches, and Steps 3-5's inline titles: one edge after them
GUTTER = 12  # §7: the control gutter, between the left column and the controls right of it
METER_GAP = 16  # Steps 1 and 3: between a meter and its slider
VALUE_W = 70  # the dB readouts of Mic level, Hear myself and Voice volume
STYLE_W = 116  # the Voice style dropdown; the Denoise row keeps the same room, so both Strength sliders line up
STEP_GAP = 14  # §7 says 18; 14 so all five steps fit (round 13)
# Above Step 1, and at least below Step 5. The steps end in a stretch, so the opening size has room to spare under Step
# 5: on real Windows text 12 px (§7's figure) at 100 % and 7 px at 125 % and above (ci-fixes r1, probe_layout_slack),
# as Windows and macOS draw each line up to 1 px taller than offscreen does (Barlow's fractional descent; CI run
# 43368578097). With a fixed 12 px here and OUTPUT_H unset, the steps scrolled 4 px at 100 % and 9 px at 125 %+.
BODY_TOP, BODY_BOTTOM = 14, 4
OUTPUT_H = 34  # Step 4's dropdown: fixed, as the theme dropdown is; Windows' style made it 38 px at 125 %
SNAP_ROOM = 6  # a scrolled view starts this far above a step or row (round 16)
HEADER_TOP, HEADER_BOTTOM = 12, 12  # §7 says 20 and 16; tightened with STEP_GAP
ROW_GAP = 8  # §7 says 10; 8 so all five steps fit (round 13)
STEPS = ("Choose your microphone", "Clean your voice", "Adjust volume", "Send clean voice to", "Record backup")
ECHO_TEXT = "For speakers instead of headphones: stops the call hearing its own sound."
MIC_TEST_TEXT = "Test your mic"
NO_MIC_TEXT = "No microphone found"
NO_MIC_TITLE = "Plug in a microphone"  # the notice says what to do; the dropdown above already says none was found
NO_OUTPUT_TEXT = "No output found"
CALL_APPS = "Zoom, Meet, Discord or Teams"
SHOW_PLUGIN = "Show plugin"
# The headline, from the guest's side: what the call hears right now.
HEAD_CLEAN = "Your call hears your clean voice"
HEAD_RAW = "Your call hears your raw mic"
HEAD_NO_CALL = "Your call can't hear you yet"
HEAD_NO_MIC = "Clean Voice can't use your microphone"
HEAD_NO_DEVICE = "Clean Voice can't open your sound devices"  # the engine does not say which device failed
HEAD_STARTING = "Starting…"
NO_CLEANUP = "No clean-up: your voice goes out as your mic hears it"
POWER_OFF_SUB = "Clean-up is off. Press the power button to turn it back on."
# Step 1's and Step 3's sliders (DESIGN "Home gain sliders", "OBS-style meters and faders").
MIC_LEVEL_TIP = "Your mic's own level, as in your computer's sound settings. Lower it if this meter reaches the red."
MIC_LEVEL_FIXED = "This mic's level can't be changed from here."
CALL_VOLUME_TIP = "How loud your voice is in the call"
MONITOR_VOLUME_TIP = "How loud you hear yourself. The call doesn't change."
# The footer's theme choice (DESIGN "Theme choice", 2026-11-13), in theme.MODES' order.
THEME_TEXTS = ("Automatic", "Light", "Dark")
THEME_H = 28  # the footer's switch row: the dropdown is no taller, so the footer keeps its height


class EngineLike(Protocol):
    recordings_dir: Path
    vst: vst.VstRegistry  # Fine-tune's loaded VST plug-ins (J10 §5)

    def mics(self) -> list[str]: ...
    def outputs(self) -> list[str]: ...
    def default_mic(self) -> str | None: ...
    def start(self, mic: str | None) -> None: ...
    def stop(self) -> None: ...
    def select_mic(self, mic: str) -> None: ...
    def select_output(self, output: str | None) -> None: ...
    def refresh(self) -> None: ...
    def set_recording(self, on: bool) -> None: ...
    def set_settings(self, settings: Settings) -> None: ...
    def settings(self) -> Settings: ...
    def set_monitor(self, on: bool) -> None: ...
    def set_monitor_volume(self, volume: float) -> None: ...
    def set_echo(self, on: bool) -> None: ...
    def start_mic_test(self) -> None: ...
    def cancel_mic_test(self) -> None: ...
    def status(self) -> Status: ...
    def close(self) -> None: ...


# ------------------------------------------------------------------ what each state says


def cable_mic_name(platform: str = sys.platform) -> str:
    """What the guest picks as the microphone in the call app: the cable's recording side (devices.py:30)."""
    return "CABLE Output" if platform == "win32" else "VB-Cable"


def use_hint(platform: str = sys.platform) -> str:
    """Step 4's one instruction."""
    return (
        f"Then pick <b>{cable_mic_name(platform)}</b> as the microphone in {CALL_APPS}, and turn off its noise "
        "cancellation."
    )


_GENERIC_MIC = {"microphone", "microphone array", "mic", "headset", "headset microphone", "input", "line in", "マイク"}


def short_mic_name(name: str) -> str:
    """A device name short enough for a small label: "Video Call L + R (Focusrite USB Audio)" gives
    "Video Call L + R"; a generic name such as "Microphone (A4tech FHD Camera)" gives what is in the brackets."""
    match = re.match(r"^(.*?)\s*\((.*)\)\s*$", name)
    if match is None:
        return name.strip()
    head, inner = match.group(1).strip(), match.group(2).strip()
    return inner if not head or head.lower() in _GENERIC_MIC else head


@dataclass(frozen=True)
class Notice:
    """What happened and what to do, shown in the step it concerns."""

    kind: str  # "cable" or "output" (Step 4), or "mic" (Step 1)
    title: str
    body: str
    action: str | None = None  # button text: the VB-CABLE download page
    credit: str | None = None
    details: str | None = None  # the sound system's own words, kept for the tooltip

    @property
    def step(self) -> int:
        return 1 if self.kind == "mic" else 4


def cable_notice(platform: str = sys.platform) -> Notice:
    return Notice(
        "cable",
        "Install VB-CABLE",
        "1.  Download VB-CABLE and install it.\n"
        "2.  Restart your computer.\n"
        f"3.  Open Clean Voice again, then pick {cable_mic_name(platform)} as the microphone in your call app.",
        "Download VB-CABLE",
        "VB-CABLE is free software by VB-Audio (vb-cable.com); they welcome donations.",
    )


@dataclass(frozen=True)
class View:
    headline: str
    tone: str  # the headline's colour: "signal", "ink" or "hot" (drawn in hot_text)
    power: str  # the power ring's colour: "signal" (lit), "line" or "hot"
    notice: Notice | None


def describe(status: Status, cleaning: bool = True, platform: str = sys.platform) -> View:
    """The headline, the power ring and the notice for one engine status (§7 States)."""
    state = status.state
    if state is State.OFF:
        return View(HEAD_STARTING, "ink", "line", None)
    if state is State.NO_MIC:
        notice = Notice("mic", NO_MIC_TITLE, "Clean Voice notices it within a few seconds.")
        return View(HEAD_NO_MIC, "hot", "hot", notice)
    if state is State.MIC_DENIED:
        notice = Notice(
            "mic",
            "Clean Voice isn't allowed to use your microphone",
            "Open System Settings, Privacy & Security, Microphone, and switch Clean Voice on. "
            "Clean Voice notices within a few seconds.",
        )
        return View(HEAD_NO_MIC, "hot", "hot", notice)
    if state is State.MIC_LOST:
        notice = Notice(
            "mic",
            "Your microphone was unplugged",
            "Plug it back in and Clean Voice carries on, with the backup in the same file. Or pick another one above.",
        )
        return View(HEAD_NO_MIC, "hot", "hot", notice)
    if state is State.ERROR:
        # The engine does not say whether the mic or the cable failed (J07 r1 m2): say neither, and keep the sound
        # system's raw words out of the text (they stay in the tooltip).
        notice = Notice(
            "mic",
            "A sound device won't open",
            "Another app may be using your microphone or VB-CABLE. Close it, or pick another microphone above. "
            "Clean Voice tries again every few seconds.",
            details=status.error,
        )
        return View(HEAD_NO_DEVICE, "hot", "hot", notice)
    if state is State.CABLE_LOST:
        notice = Notice(
            "cable",
            "VB-CABLE stopped responding",
            "Clean Voice tries again every few seconds. If it goes on, restart your computer.",
        )
        return View(HEAD_NO_CALL, "hot", "hot", notice)
    if state is State.NO_CABLE or status.cable is None:
        return View(HEAD_NO_CALL, "hot", "hot", cable_notice(platform))
    output = status.output or status.cable
    if output is not None and not is_cable_output(output):  # the guest sends the voice to speakers, not the call
        notice = Notice(
            "output",
            f"Your voice goes to {output}",
            f"Your call can't hear it there. Choose {status.cable} above.",
        )
        return View(HEAD_NO_CALL, "hot", "hot", notice)
    if cleaning:
        return View(HEAD_CLEAN, "signal", "signal", None)
    return View(HEAD_RAW, "ink", "line", None)


def describe_chain(settings: Settings, preset: str) -> str:
    """The subline: only what is on ("Denoise 70, Warm style, 2 Fine-tune filters"), §7."""
    parts = []
    if settings.denoise:
        parts.append(f"Denoise {round(settings.strength)}")
    if settings.style:
        parts.append(f"{PRESETS[preset].name} style")
    if settings.fine_tune:
        n = len(settings.filters)
        parts.append(f"{n} Fine-tune filter{'' if n == 1 else 's'}" if n else "Fine-tune with no filters yet")
    return ", ".join(parts) if parts else NO_CLEANUP


def format_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ------------------------------------------------------------------ the volume sliders' dB (DESIGN "OBS-style meters")


def voice_volume_text(value: int) -> str:
    """Voice volume (0..100, 50 = 0 dB, -12..+12 dB in the chain: dsp.loudness_db) in OBS's readout."""
    return db_text(loudness_db(float(value)))


def monitor_db(position: int) -> float:
    """Hear myself's slider is an OBS fader: its position (0..100) is the fader's deflection (widgets.fader_db)."""
    return fader_db(position / 100.0)


def monitor_percent(position: int) -> float:
    """The engine's Hear myself volume (0..100 %, gain = (percent / 100)^2, Engine.set_monitor_volume) for a fader
    position: the same gain as the fader's dB."""
    db = monitor_db(position)
    return 0.0 if db == -math.inf else 100.0 * 10.0 ** (db / 40.0)


def monitor_position(percent: float) -> float:
    """Where a saved Hear myself volume sits on the fader (saved files keep the engine's percent)."""
    if percent <= 0.0:
        return 0.0
    return 100.0 * fader_deflection(40.0 * math.log10(min(percent, 100.0) / 100.0))


def monitor_text(position: int) -> str:
    return db_text(monitor_db(position))


# ------------------------------------------------------------------ small window-only widgets


class Logo(QWidget):
    def __init__(self, size: int = 16, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        mark.paint_mark(p, self.width())


class RecordingDot(QWidget):
    """The backup's dot: steady red while recording, grey otherwise (only the meters move, §7 Motion)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self.recording = False

    def set_recording(self, recording: bool) -> None:
        if recording != self.recording:
            self.recording = recording
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(pal.clip if self.recording else pal.muted))
        p.drawEllipse(QPointF(6, 6), 4.5, 4.5)


class HeaderBand(QFrame):
    """The header band: tinted ``band``, with a 2 px ``signal`` bottom edge while the call hears the clean voice and a
    1 px ``line`` hairline otherwise."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("header")
        self.lit = False

    def set_lit(self, lit: bool) -> None:
        if lit != self.lit:
            self.lit = lit
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(pal.band))
        edge = 2 if self.lit else 1
        p.fillRect(0, self.height() - edge, self.width(), edge, QColor(pal.signal if self.lit else pal.line))


class DevicePicker(QAbstractButton):
    """Step 1's dropdown: the mic's full name, on two lines if needed, and a chevron. A click opens the microphone
    menu under it (the original's dock menu, main.js:549-593). ``expanded`` shows the menu is open."""

    device_name = True  # a very long device name may be cut, with the full name in the tooltip

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.expanded = False
        self.setToolTip("Choose microphone")
        self.setAccessibleName("Choose microphone")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setFixedHeight(46)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(300, 46)

    def minimumSizeHint(self) -> QSize:
        return QSize(260, 46)

    def set_expanded(self, expanded: bool) -> None:
        if expanded != self.expanded:
            self.expanded = expanded
            self.update()

    def _text_rect(self) -> QRectF:
        return QRectF(self.rect()).adjusted(10, 3, -26, -3)

    def lines(self) -> tuple[str, str, bool]:
        """The name on at most two lines, broken at a space; True when the second line had to be cut with "…"."""
        metrics = QFontMetricsF(ui_font(self, 13))
        width = self._text_rect().width()
        text = self.text()
        if metrics.horizontalAdvance(text) <= width:
            return text, "", False
        # "Video Call L + R (Focusrite USB Audio)": the name, then its interface, when each fits a line (round 15).
        head, bracket, tail = text.partition(" (")
        if bracket and head and max(metrics.horizontalAdvance(head), metrics.horizontalAdvance("(" + tail)) <= width:
            return head, "(" + tail, False
        words, first = text.split(" "), ""
        while words and metrics.horizontalAdvance((first + " " + words[0]).strip()) <= width:
            first = (first + " " + words.pop(0)).strip()
        if not first:  # one long word: break it where it reaches the edge
            first = metrics.elidedText(text, Qt.TextElideMode.ElideNone, width)
            rest = text[len(first) :]
        else:
            rest = " ".join(words)
        second = metrics.elidedText(rest, Qt.TextElideMode.ElideRight, width)
        return first, second, second != rest

    def cut_off(self) -> bool:
        """Whether the name needs more than the two lines it has."""
        return self.lines()[2]

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Down):
            self.click()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        focused = keyboard_focus(self)
        lit = self.expanded or focused
        p.setPen(QPen(QColor(pal.desk if lit else (pal.muted if self.underMouse() else pal.line)), 2 if focused else 1))
        p.setBrush(QColor(pal.surface))
        p.drawRoundedRect(r, 4, 4)
        p.setFont(ui_font(self, 13))
        p.setPen(QColor(pal.ink if self.isEnabled() else pal.muted))
        first, second, _ = self.lines()
        words = f"{first}\n{second}" if second else first
        p.drawText(self._text_rect(), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), words)
        paint_icon(p, "chevron", QRectF(r.right() - 24, r.center().y() - 8, 16, 16), pal.muted, 1.8)


class MenuItem(QAbstractButton):
    """One microphone in the menu: its full name, cut with "…" only if it must be, and a tick on the current one."""

    device_name = True

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setText(text)
        self.setToolTip(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setFixedHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(300, 34)

    def cut_off(self) -> bool:
        return QFontMetricsF(ui_font(self, 13, QFont.Weight.Medium)).horizontalAdvance(self.text()) > self.width() - 44

    def paintEvent(self, event: QPaintEvent) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        if self.isEnabled() and (self.underMouse() or self.hasFocus()):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(pal.hover))
            p.drawRoundedRect(r, 4, 4)
        on = self.isChecked()
        font = ui_font(self, 13, QFont.Weight.Medium if on else QFont.Weight.Normal)
        p.setFont(font)
        p.setPen(QColor(pal.desk if on else (pal.ink if self.isEnabled() else pal.muted)))
        text_rect = r.adjusted(10, 0, -34, 0)
        shown = QFontMetricsF(font).elidedText(self.text(), Qt.TextElideMode.ElideRight, text_rect.width())
        p.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, shown)
        if on:
            paint_icon(p, "tick", QRectF(r.right() - 26, r.center().y() - 8, 16, 16), pal.desk, 2.0)


class MicMenu(QFrame):
    """The microphone menu that opens under Step 1's dropdown."""

    chosen = Signal(str)
    dismissed = Signal()  # Escape: the window closes the menu, the picker and its event filter with it

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("micMenu")
        self.items: list[MenuItem] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)
        head = QLabel("Microphone")
        head.setProperty("role", "menuHead")
        layout.addWidget(head)
        self._head = head
        self._scroll = QScrollArea()
        self._scroll.setObjectName("menuScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list = QWidget()
        self._list.setObjectName("menuList")
        self._list_layout = QVBoxLayout(self._list)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(0)
        self._scroll.setWidget(self._list)
        layout.addWidget(self._scroll)
        self.hide()

    def fill(self, names: list[str], current: str | None) -> None:
        for item in self.items:
            item.deleteLater()
        self.items = []
        for name in names or [NO_MIC_TEXT]:
            item = MenuItem(name, self._list)
            item.setEnabled(bool(names))
            item.setChecked(name == current)
            item.clicked.connect(lambda _=False, name=name: self.chosen.emit(name))
            self._list_layout.addWidget(item)
            self.items.append(item)

    def names(self) -> list[str]:
        return [i.text() for i in self.items if i.isEnabled()]

    def place(self, under: QRect, area: QRect) -> None:
        """Just under ``under`` (the picker, in the parent's coordinates), at least 320 px wide, inside ``area``."""
        width = min(max(under.width(), 320), area.width() - 16)
        content = 12 + 2 + self._head.sizeHint().height() + 34 * len(self.items)
        height = min(content, area.bottom() - under.bottom() - 12)
        x = min(under.x(), area.right() - width - 8)
        self.setGeometry(x, under.bottom() + 4, width, max(height, 80))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.dismissed.emit()
            return
        if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            items = [i for i in self.items if i.isEnabled()]
            if items:
                step = 1 if event.key() == Qt.Key.Key_Down else -1
                focused = next((k for k, i in enumerate(items) if i.hasFocus()), -1)
                items[(focused + step) % len(items)].setFocus()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            focused_item = next((i for i in self.items if i.hasFocus() and i.isEnabled()), None)
            if focused_item is not None:
                focused_item.click()
                return
        super().keyPressEvent(event)


class Dropdown(QComboBox):
    """A dropdown with Clean Voice's chevron. Its focus border shows for the keyboard only (TabFocus: a click opens it
    without taking the focus)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_icon(p, "chevron", QRectF(self.width() - 26.0, self.height() / 2.0 - 8.0, 16, 16), pal.muted, 1.8)


class DeviceComboBox(Dropdown):
    """Step 4's device dropdown; opening it lets the window refresh the list first (only while no stream runs)."""

    opening = Signal()

    def showPopup(self) -> None:
        self.opening.emit()
        super().showPopup()


class CloseNote(QLabel):
    """The footer's close note, word-wrapped, its lines at most ``THEME_H / 2`` apart: two lines fill the switch row
    and no more, at every display scale. Windows' 12 px text has 15 px lines at 125 % and 150 % (14 at 100 %), and
    two of those made the footer 2 px taller at the minimum width (theme-choice r1 F1). The last line keeps its whole
    height below its step, so its descenders are never cut (theme-choice r2 F2). Its last two words are joined by a
    no-break space, so a wrapped note never ends with one word alone ("bar." / "tray.", ci-fixes r1)."""

    def __init__(self, text: str) -> None:
        super().__init__("\N{NO-BREAK SPACE}".join(text.rsplit(" ", 1)))
        self.setWordWrap(True)

    def _line_box(self) -> float:
        """One line's ink height (ascent + descent), rounded up to whole device pixels plus one for the antialiased
        edge, in logical pixels."""
        metrics = QFontMetricsF(self.font())
        dpr = self.devicePixelRatioF()
        return (math.ceil((metrics.ascent() + metrics.descent()) * dpr) + 1) / dpr

    def _layout(self, width: int) -> tuple[QTextLayout, float]:
        """The text broken at ``width``, each line placed one step below the last; and that step."""
        step = min(QFontMetricsF(self.font()).lineSpacing(), THEME_H - self._line_box())
        layout = QTextLayout(self.text(), self.font())
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WordWrap)
        layout.setTextOption(option)
        layout.beginLayout()
        n = 0
        while (line := layout.createLine()).isValid():
            line.setLineWidth(width)
            line.setPosition(QPointF(0, n * step))
            n += 1
        layout.endLayout()
        return layout, step

    def _height(self, width: int) -> int:
        layout, step = self._layout(width)
        if layout.lineCount() <= 1:
            return math.ceil(QFontMetricsF(self.font()).height())
        return math.ceil((layout.lineCount() - 1) * step + self._line_box())

    def heightForWidth(self, width: int) -> int:
        margins = self.contentsMargins()
        return self._height(width - margins.left() - margins.right()) + margins.top() + margins.bottom()

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(hint.width(), self.heightForWidth(hint.width()))

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(hint.width(), self.heightForWidth(hint.width()))

    def paintEvent(self, event: QPaintEvent) -> None:
        rect = self.contentsRect()
        layout, _ = self._layout(rect.width())
        p = QPainter(self)
        p.setPen(self.palette().color(self.foregroundRole()))
        # Centred in the row, as the QLabel it replaces.
        layout.draw(p, QPointF(rect.left(), rect.top() + (rect.height() - self._height(rect.width())) / 2))


# ------------------------------------------------------------------ the style sheet, per theme


def style_sheet(pal: Palette) -> str:
    return f"""
QWidget#root, QWidget#body, QScrollArea#bodyScroll, QWidget#footer {{ background: {pal.surface}; }}
QLabel {{ color: {pal.ink}; font-size: 13px; }}
QLabel[role="headline"] {{ font-size: 26px; font-weight: 600; }}
QLabel[role="headline"][tone="signal"] {{ color: {pal.signal}; }}
QLabel[role="headline"][tone="hot"] {{ color: {pal.hot_text}; }}
QLabel[role="subline"] {{ color: {pal.muted}; font-size: 13px; }}
QLabel[role="stepNumber"] {{ color: {pal.muted}; font-size: 16px; font-weight: 600; }}
QLabel[role="stepTitle"] {{ font-size: 16px; font-weight: 600; }}
QLabel[role="hint"], QLabel[role="version"] {{ color: {pal.muted}; font-size: 12px; }}
QLabel[role="hint"][warn="true"] {{ color: {pal.hot_text}; }}
QLabel[role="noteTitle"] {{ color: {pal.hot_text}; font-size: 13px; font-weight: 600; }}
QLabel[role="note"] {{ color: {pal.ink}; font-size: 13px; }}
QLabel[role="option"] {{ font-size: 13px; font-weight: 500; }}
QLabel[role="option"][dim="true"] {{ color: {pal.muted}; }}
QLabel[role="menuHead"] {{ color: {pal.muted}; font-size: 12px; padding: 4px 8px 6px 8px; }}
QLabel[role="time"] {{ color: {pal.ink}; font-size: 13px; }}
QLabel[role="instruction"] {{ color: {pal.ink}; font-size: 13px; }}
QFrame[role="rule"] {{ background: {pal.line}; border: none; max-height: 1px; min-height: 1px; }}
QFrame[role="noteBar"] {{ background: {pal.hot}; border: none; }}
QFrame#micMenu {{ background: {pal.surface}; border: 1px solid {pal.line}; border-radius: 4px; }}
QWidget#menuList, QScrollArea#menuScroll {{ background: transparent; }}
QComboBox {{ background: {pal.surface}; color: {pal.ink}; border: 1px solid {pal.line}; border-radius: 4px;
    padding: 7px 30px 7px 10px; font-size: 13px; }}
QComboBox[dim="true"] {{ color: {pal.muted}; }}
QComboBox:hover {{ border-color: {pal.muted}; }}
QComboBox:focus {{ border: 2px solid {pal.desk}; padding: 6px 29px 6px 9px; }}
QComboBox:disabled {{ color: {pal.muted}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{ image: none; width: 0; height: 0; border: none; }}
QComboBox QAbstractItemView {{ background: {pal.surface}; color: {pal.ink}; border: 1px solid {pal.line};
    selection-background-color: {pal.hover}; selection-color: {pal.desk}; padding: 4px; outline: none; }}
QComboBox#themeCombo {{ font-size: 12px; padding: 3px 0 3px 8px; combobox-popup: 0; }}
QComboBox#themeCombo:focus {{ padding: 2px 0 2px 7px; }}
QComboBox#themeCombo QAbstractItemView::item {{ min-height: 24px; padding: 0 4px; }}
QPushButton[role="btn"] {{ background: {pal.surface}; color: {pal.ink}; border: 1px solid {pal.line};
    border-radius: 6px; padding: 7px 14px; font-size: 13px; font-weight: 500; }}
QPushButton[role="btn"]:hover {{ border-color: {pal.muted}; background: {pal.hover}; }}
QPushButton[role="btn"]:focus {{ border: 2px solid {pal.desk}; padding: 6px 13px; }}
QPushButton[role="btn"]:disabled {{ color: {pal.muted}; }}
QPushButton[role="btn"][primary="true"] {{ background: {pal.desk}; border-color: {pal.desk}; color: {pal.on_desk}; }}
QPushButton[role="btn"][primary="true"]:focus {{ border: 2px solid {pal.ink}; }}
QPushButton[role="link"] {{ background: transparent; color: {pal.desk}; border: none; padding: 2px 0;
    font-size: 12px; text-align: left; }}
QPushButton[role="link"]:hover {{ text-decoration: underline; }}
QPushButton[role="link"]:focus {{ text-decoration: underline; border-bottom: 2px solid {pal.desk}; }}
QPushButton[role="link"]:disabled {{ color: {pal.muted}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px 4px 0; }}
QScrollBar::handle:vertical {{ background: {pal.line}; border-radius: 3px; min-height: 40px; }}
QScrollBar::handle:vertical:hover {{ background: {pal.muted}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QToolTip {{ background: {pal.surface}; color: {pal.ink}; border: 1px solid {pal.line}; }}
"""


def _label(text: str = "", role: str | None = None, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    if role:
        label.setProperty("role", role)
    label.setWordWrap(wrap)
    return label


def _button(text: str, primary: bool = False, role: str = "btn") -> QPushButton:
    button = QPushButton(text)
    button.setProperty("role", role)
    if primary:
        button.setProperty("primary", True)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setFocusPolicy(Qt.FocusPolicy.TabFocus)  # the focus border only for the keyboard (§7 Focus)
    return button


def _row_group(spacing: int) -> QHBoxLayout:
    """Footer widgets that belong together, ``spacing`` apart, in the footer's row."""
    group = QHBoxLayout()
    group.setContentsMargins(0, 0, 0, 0)
    group.setSpacing(spacing)
    return group


def _rule() -> QFrame:
    line = QFrame()
    line.setProperty("role", "rule")
    line.setFixedHeight(1)
    return line


def _headline_font(pixels: int) -> QFont:
    font = QFont()
    font.setFamilies(fonts.headline())
    font.setPixelSize(pixels)
    font.setWeight(QFont.Weight.DemiBold)
    return font


def _set_text(label: QLabel, text: str) -> None:
    if label.text() != text:  # the tick runs 30 times a second: touch a label only when its text changes
        label.setText(text)


def _set_prop(widget: QWidget, name: str, value: object) -> None:
    if widget.property(name) != value:
        widget.setProperty(name, value)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


def _theme_mode(name: str) -> str:
    """A ``theme_name`` argument as a choice: "auto" and "dark" as they are, anything else light (as ``for_name``)."""
    return name if name in theme.MODES else "light"


def _select_data(combo: QComboBox, data: str) -> None:
    index = combo.findData(data)
    if index >= 0 and combo.currentIndex() != index:
        combo.setCurrentIndex(index)


def _set_visible(widget: QWidget, visible: bool) -> None:
    if widget.isHidden() == visible:
        widget.setVisible(visible)


class Step(QWidget):
    """One numbered step: the number in a 32 px column, then the title over its controls (Steps 1 and 2), or the
    title inline on the left of them (Steps 3-5, so the window stays at 640 px)."""

    def __init__(self, number: int, title: str, inline: bool = False, note: str | None = None) -> None:
        super().__init__()
        self.number = number
        self.title = _label(title, "stepTitle")
        self.title.setFont(_headline_font(16))
        number_label = _label(str(number), "stepNumber")
        number_label.setFont(_headline_font(16))
        number_label.setFixedWidth(NUMBER_W)
        number_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setAccessibleName(f"Step {number}: {title}")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(number_label, 0, Qt.AlignmentFlag.AlignTop)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(ROW_GAP)
        if inline:
            head = QVBoxLayout()
            head.setSpacing(0)
            head.setContentsMargins(0, 0, 0, 0)
            head.addWidget(self.title)
            if note:
                head.addWidget(_label(note, "hint"))
            head.addStretch(1)
            holder = QWidget()
            holder.setLayout(head)
            holder.setFixedWidth(SECOND_W + 12)  # as wide as Step 2's switches and their gap: one edge
            row.addWidget(holder)
            row.addLayout(self.body, 1)
        else:
            column = QVBoxLayout()
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(8)  # §7: step title to its controls
            column.addWidget(self.title)
            column.addLayout(self.body)
            row.addLayout(column, 1)


class _ActivationWatcher(QObject):
    """Shows the hidden window when the app is activated (macOS: Finder or the Dock opened it again)."""

    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self.window = window

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ApplicationActivate and self.window.isHidden() and not self.window._closed:
            self.window.show_from_tray()
        return False


log = logging.getLogger(__name__)

# ------------------------------------------------------------------ Check for updates (DESIGN "Updates")

UPDATE_DELAY_MS = 10_000  # the quiet check, 10 s after launch
CHECKING = "Checking for updates…"


class UpdateController(QObject):
    """Check for updates and Install update. The network and the download run on a worker thread (``run``); their
    results come back to the UI thread through queued signals. Nothing installs without the guest's click on Install
    update."""

    changed = Signal()
    _checked = Signal(object)  # CheckResult
    _progress = Signal(float)
    _downloaded = Signal(object)  # Path, or UpdateError

    def __init__(
        self,
        current: str,
        repo: str | None,
        platform: str,
        quit_app: Callable[[], None],
        opener: Any = None,
        run: Callable[[Callable[[], None]], None] | None = None,
        launch: Callable[[list[str]], object] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.current, self.repo, self.platform = current, repo, platform
        self.state = "idle"  # idle, checking, latest, newer, failed, confirm, downloading, install-failed
        self.message = ""
        self.fraction = 0.0
        self.result: updates.CheckResult | None = None
        self._quiet = False
        self._quit = quit_app
        self._opener = opener
        self._run = run or (lambda job: threading.Thread(target=job, name="clean-voice-update", daemon=True).start())
        self._launch = launch or (lambda command: subprocess.Popen(command))  # an argument list, no shell
        self._checked.connect(self._on_checked)
        self._progress.connect(self._on_progress)
        self._downloaded.connect(self._on_downloaded)

    def _set(self, state: str, message: str) -> None:
        self.state, self.message = state, message
        self.changed.emit()

    def check(self, quiet: bool = False) -> None:
        """Ask for the latest release. ``quiet`` (the launch check) shows only an update, never "latest" or a
        failure."""
        if self.state in ("checking", "downloading"):
            return
        self._quiet = quiet
        if not quiet:
            self._set("checking", CHECKING)
        opener = self._opener

        def job() -> None:
            kwargs = {} if opener is None else {"opener": opener}
            try:
                result = updates.check(self.current, self.repo, **kwargs)
            except Exception:  # check() never raises; and if it did, the view must not stick on "Checking…"
                log.exception("update check failed")
                result = updates.CheckResult("failed", updates.FAILED)
            self._checked.emit(result)

        self._run(job)

    def _on_checked(self, result: updates.CheckResult) -> None:
        self.result = result
        if self._quiet and not result.installable:
            if self.state == "checking":
                self._set("idle", "")
            return
        self._set(result.state, result.message)

    def confirm(self) -> None:
        """Install update's first click: say that Clean Voice will close, with Install and Cancel."""
        if self.result is not None and self.result.installable and self.state in ("newer", "install-failed"):
            self._set("confirm", updates.CONFIRM)

    def cancel(self) -> None:
        if self.state == "confirm" and self.result is not None:
            self._set("newer", self.result.message)

    def primary(self) -> None:
        """The install button: Install update asks first; Install (after the question) downloads and installs."""
        if self.state == "confirm":
            self.install()
        else:
            self.confirm()

    def install(self) -> None:
        result = self.result
        if result is None or not result.installable or self.state != "confirm":
            return  # never without the guest's Install after the question
        self.fraction = 0.0
        self._set("downloading", result.message)
        folder = Path(tempfile.mkdtemp(prefix="clean-voice-update-"))
        opener = self._opener

        def job() -> None:
            kwargs = {} if opener is None else {"opener": opener}
            try:
                path: object = updates.download_verified(result, self.platform, folder, self._progress.emit, **kwargs)
            except updates.UpdateError as exc:
                path = exc
            except Exception:  # never a stuck "Downloading…"
                log.exception("update download failed")
                path = updates.UpdateError(updates.DOWNLOAD_FAILED)
            self._downloaded.emit(path)

        self._run(job)

    def _on_progress(self, fraction: float) -> None:
        self.fraction = fraction
        self.changed.emit()

    def _on_downloaded(self, outcome: object) -> None:
        if isinstance(outcome, updates.UpdateError):
            self._set("install-failed", str(outcome))
            return
        assert isinstance(outcome, Path)
        try:
            self._launch(updates.install_command(outcome, self.platform))
        except (OSError, updates.UpdateError):
            log.exception("could not start the installer")
            self._set("install-failed", "Couldn't start the installer. Try again later.")
            return
        self._quit()  # the installer replaces the app; Clean Voice must not be running


class UpdateView(QWidget):
    """The footer's updates: ``check_button`` (Check for updates, placed beside the version by the window) and, on a
    line of their own when there is something to say, what the check found, with Install update and Cancel."""

    def __init__(self, controller: UpdateController) -> None:
        super().__init__()
        self.controller = controller
        self.check_button = _button("Check for updates", role="link")
        self.check_button.clicked.connect(lambda: controller.check())
        self.status = _label("", "hint", wrap=True)
        self.install_button = _button("Install update", primary=True)
        self.install_button.clicked.connect(controller.primary)
        self.cancel_button = _button("Cancel")
        self.cancel_button.clicked.connect(controller.cancel)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        row.addStretch(1)
        row.addWidget(self.status, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.install_button, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.cancel_button, 0, Qt.AlignmentFlag.AlignVCenter)
        controller.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        c = self.controller
        busy = c.state in ("checking", "downloading")
        self.check_button.setEnabled(not busy and c.state != "confirm")
        _set_text(self.status, c.message)
        _set_visible(self.status, bool(c.message))
        installing = c.state in ("newer", "confirm", "downloading", "install-failed")
        _set_visible(self.install_button, installing)
        _set_visible(self.cancel_button, c.state == "confirm")
        self.install_button.setEnabled(c.state in ("newer", "confirm", "install-failed"))
        if c.state == "downloading":
            text = f"Downloading {round(c.fraction * 100)} %"
        else:
            text = "Install" if c.state == "confirm" else "Install update"
        self.install_button.setText(text)
        _set_visible(self, bool(c.message) or installing)


class MainWindow(QMainWindow):
    """The one window, wired to the engine and the saved choices. ``theme_name`` "light" or "dark" fixes the theme
    (the renders) without changing the saved choice; None takes the saved choice (Automatic follows the system live)."""

    def __init__(
        self,
        engine: EngineLike | None = None,
        prefs_path: Path | None = None,
        login_item: autostart.LoginItem | None = None,
        platform: str = sys.platform,
        theme_name: str | None = None,
        updater: UpdateController | None = None,
        mic_levels: mic_level.MicLevel | None = None,
        vst_loader: filter_window.Loader = vst.create,
        vst_scan: Callable[[], list[tuple[str, str]]] = vst.scan,
    ) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(mark.icon())
        self.platform = platform
        self.mic_levels = mic_levels if mic_levels is not None else mic_level.system()
        self.updates = updater or UpdateController(
            __version__, updates.repo_of(updates.homepage()), platform, self.quit_app, parent=self
        )
        self.engine: EngineLike = engine if engine is not None else _make_engine()
        self._owns_engine = engine is None
        self.prefs_path = prefs_path if prefs_path is not None else prefs.default_path()
        self.prefs = prefs.load(self.prefs_path)
        self.login_item = login_item if login_item is not None else autostart.login_item()
        self.tray: Tray | None = None
        # The footer's Light / Dark / Automatic choice (DESIGN "Theme choice"); ``theme_name`` (the renders) fixes it
        # for this window without changing the saved one.
        self.theme_mode = self.prefs.theme if theme_name is None else _theme_mode(theme_name)
        theme.request_scheme(self.theme_mode)
        self.palette_now = theme.palette_for(self.theme_mode)
        theme.use(self.palette_now)
        self._quitting = False
        self._closed = False
        self._dirty = False
        self._mics: list[str] | None = None  # the lists last shown, None until the first sync
        self._outputs: list[str] | None = None
        self._view: View | None = None
        self._status = Status()
        self._monitor_wanted = False
        self._monitor_pending_until = 0.0
        self._mic_gain_mic: str | None = None
        self._mic_db: float | None = None
        self._last_tick = time.monotonic()
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self.save_prefs)
        # Fine-tune's list and its VST plug-ins; the filter window opens from Show plugin.
        self.fine = filter_window.FilterModel(
            lambda: self.prefs.settings, self._send_filters, self.engine.vst, vst_loader, vst_scan
        )
        self.filter_window: filter_window.FilterWindow | None = None
        self._build()
        self.resize(*WINDOW_SIZE)
        self.setMinimumSize(*MIN_SIZE)
        self._placed = False  # fit_to_screen runs once, before the window first shows
        self._apply_prefs()
        if QGuiApplication.instance() is not None:
            QGuiApplication.styleHints().colorSchemeChanged.connect(self.follow_system_theme)
        # As in the original (main.js:687-704): send every saved choice, then start with the saved mic. The engine
        # runs until Quit; the power button is a bypass, never the engine. The saved VST plug-ins load first: they are
        # in the engine's registry before the settings naming them reach it.
        self.fine.load_saved()
        self.engine.set_settings(self.prefs.settings)
        self.engine.set_echo(self.prefs.echo)
        self.engine.set_monitor_volume(self.prefs.monitor_volume)
        self.engine.set_recording(self.prefs.recording)
        if self.prefs.output is not None:
            self.engine.select_output(self.prefs.output)
        self.engine.start(self.prefs.mic)
        self._read_autostart()
        self.tick()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.tick)
        # Slow until the window is really shown: a --minimized start is never shown, so it never gets a hideEvent
        # either (J07 r1 F1). showEvent and hideEvent switch it from there.
        self._timer.start(TICK_MS if self.isVisible() else HIDDEN_TICK_MS)
        # macOS: opening Clean Voice again from Finder or the Dock only activates the running app (no second
        # process), so the hidden window comes back on that activation (J07 r1 m7).
        app = QApplication.instance()
        if platform == "darwin" and app is not None:
            self._activation = _ActivationWatcher(self)
            app.installEventFilter(self._activation)

    # ------------------------------------------------------------------ theme

    def follow_system_theme(self, *_: object) -> None:
        """The system switched between light and dark: follow it on Automatic; Light and Dark stay as chosen."""
        if self.theme_mode == "auto":
            self.set_theme(theme.palette_for("auto"))

    def choose_theme(self, mode: str) -> None:
        """The guest's choice in the footer: applied everywhere at once, and saved."""
        if mode not in theme.MODES:
            return
        self.theme_mode = mode
        theme.request_scheme(mode)  # first: Automatic then reads the system's own setting again
        self.set_theme(theme.palette_for(mode))
        _select_data(self.theme_combo, mode)
        if mode != self.prefs.theme:
            self._change(theme=mode)

    def _theme_chosen(self, index: int) -> None:
        self.choose_theme(str(self.theme_combo.itemData(index)))

    def set_theme(self, pal: Palette) -> None:
        self.palette_now = pal
        theme.use(pal)
        self.root.setStyleSheet(style_sheet(pal))
        for widget in self.root.findChildren(QWidget):
            widget.update()
        if self.filter_window is not None:
            self.filter_window.set_theme(pal)

    def style_text(self) -> str:
        return self.root.styleSheet()

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        fonts.load()
        # Style-sheet fonts take their family from the application font, so the families go there too.
        app_font = QApplication.font()
        if app_font.families() != UI_FONTS:
            app_font.setFamilies(UI_FONTS)
            QApplication.setFont(app_font)
        font = QFont()
        font.setFamilies(UI_FONTS)
        font.setPixelSize(13)
        self.setFont(font)
        root = QWidget()
        root.setObjectName("root")
        root.setFont(font)
        root.setStyleSheet(style_sheet(self.palette_now))
        self.root = root
        column = QVBoxLayout(root)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self._build_header())
        body = QWidget()
        body.setObjectName("body")
        steps = QVBoxLayout(body)
        steps.setContentsMargins(MARGIN, BODY_TOP, MARGIN, BODY_BOTTOM)
        self._steps_layout = steps
        steps.setSpacing(STEP_GAP)
        self.steps = [
            self._build_step1(),
            self._build_step2(),
            self._build_step3(),
            self._build_step4(),
            self._build_step5(),
        ]
        for step in self.steps:
            steps.addWidget(step)
        steps.addStretch(1)
        self.body = QScrollArea()
        self.body.setObjectName("bodyScroll")
        self.body.setWidget(body)
        self.body.setWidgetResizable(True)
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body.verticalScrollBar().valueChanged.connect(lambda _: self.close_mic_menu())
        # The steps settle their heights after the first layout (wrapped notes): reveal the notice again each time.
        self._revealing = False
        self.body.verticalScrollBar().rangeChanged.connect(lambda *_: self._reveal_notice())
        column.addWidget(self.body, 1)
        column.addWidget(self._build_footer())
        self.mic_menu = MicMenu(root)
        self.mic_menu.chosen.connect(self._menu_chosen)
        self.mic_menu.dismissed.connect(self.close_mic_menu)
        self.setCentralWidget(root)
        self._set_tab_order()

    def _build_header(self) -> QWidget:
        """The one bold element: the headline (what the call hears) and the power button, on a tinted band."""
        header = HeaderBand()
        row = QHBoxLayout(header)
        row.setContentsMargins(MARGIN, HEADER_TOP, MARGIN - 4, HEADER_BOTTOM)
        row.setSpacing(16)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.headline = _label("", "headline", wrap=True)
        self.headline.setFont(_headline_font(26))
        self.subline = _label("", "subline", wrap=True)
        text.addStretch(1)
        text.addWidget(self.headline)
        text.addWidget(self.subline)
        text.addStretch(1)
        row.addLayout(text, 1)
        self.power_button = PowerButton()
        self.power_button.setAccessibleName("Clean voice on or off")
        self.power_button.clicked.connect(self.set_cleaning)
        row.addWidget(self.power_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.header = header
        return header

    @staticmethod
    def _lead(widget: QWidget) -> QHBoxLayout:
        """A control in the left column, as wide as Step 2's switch boxes and on their edges."""
        lead = QHBoxLayout()
        lead.setContentsMargins(StepToggle.PAD, 0, StepToggle.PAD, 0)
        lead.setSpacing(0)
        widget.setFixedWidth(StepToggle.W)
        lead.addWidget(widget, 0, Qt.AlignmentFlag.AlignTop)
        return lead

    @staticmethod
    def _meter_row(meter: LevelMeter, slider: BarSlider) -> QHBoxLayout:
        """Steps 1 and 3: a meter, and its slider beside it with the label above, in two equal halves of the right
        column, so the mic's and the call's meters share their width and scale (round 13)."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(METER_GAP)
        column = QVBoxLayout()
        column.setContentsMargins(0, BarSlider.LABEL_H + 1, 0, 0)  # the bar just under the slider's label row
        column.setSpacing(0)
        column.addWidget(meter)
        column.addStretch(1)
        row.addLayout(column, 1)
        row.addWidget(slider, 1, Qt.AlignmentFlag.AlignTop)
        return row

    def _build_step1(self) -> Step:
        step = Step(1, STEPS[0])
        row = QHBoxLayout()
        row.setSpacing(GUTTER)
        self.input_button = DevicePicker()
        self.input_button.clicked.connect(self.toggle_mic_menu)
        row.addLayout(self._lead(self.input_button))
        self.in_meter = LevelMeter(clean=False)
        self.in_meter.setToolTip("Your mic, before cleaning")
        # The mic's own OS level: no double-click reset, a jump to a fixed level could make it clip.
        self.mic_gain = BarSlider("Mic level", value=0, reset=None, formatter=self._mic_gain_text, value_width=VALUE_W)
        self.mic_gain.setToolTip(MIC_LEVEL_FIXED)
        self.mic_gain.setEnabled(False)  # until the first read says the mic has a level
        self.mic_gain.valueChanged.connect(self._mic_gain_changed)
        row.addLayout(self._meter_row(self.in_meter, self.mic_gain), 1)
        step.body.addLayout(row)
        self.mic_title = _label("", "noteTitle", wrap=True)
        self.mic_body = _label("", "note", wrap=True)
        step.body.addWidget(self.mic_title)
        step.body.addWidget(self.mic_body)
        # Test your mic and Hear myself, on one row (J10 plan, open question 1): both answer "how do I sound?".
        # Hear myself's slider sits under Mic level's, on the same edges.
        hear = QHBoxLayout()
        hear.setSpacing(GUTTER)
        self.test_button = TestButton()
        self.test_button.clicked.connect(self._test_clicked)
        hear.addLayout(self._lead(self.test_button))
        right = QHBoxLayout()
        right.setSpacing(METER_GAP)
        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        self.monitor_button = IconButton("headphones", "Hear myself", checkable=True, text="Hear myself")
        self.monitor_button.setFixedWidth(self.monitor_button.sizeHint().width())
        self.monitor_button.clicked.connect(self._monitor_clicked)
        buttons.addWidget(self.monitor_button)
        buttons.addStretch(1)
        self.monitor_volume_label = _label("Volume", "option")
        buttons.addWidget(self.monitor_volume_label)
        right.addLayout(buttons, 1)
        self.monitor_volume = BarSlider(
            "Hear myself volume", reset=100, compact=True, formatter=monitor_text, value_width=VALUE_W
        )
        self.monitor_volume.setToolTip(MONITOR_VOLUME_TIP)
        self.monitor_volume.valueChanged.connect(self._monitor_volume_changed)
        right.addWidget(self.monitor_volume, 1)
        hear.addLayout(right, 1)
        step.body.addLayout(hear)
        self.test_label = _label("", "hint", wrap=True)  # a mic test problem, when there is one
        _set_prop(self.test_label, "warn", True)
        self.test_label.hide()
        step.body.addWidget(self.test_label)
        self.dock_note = _label("", "hint", wrap=True)  # a Hear myself problem, when there is one
        _set_prop(self.dock_note, "warn", True)
        self.dock_note.hide()
        step.body.addWidget(self.dock_note)
        return step

    def _strength_row(self, toggle: StepToggle, slider: BarSlider, extra: QWidget | None = None) -> QHBoxLayout:
        """A Step 2 row: the switch, "Strength" and its slider, then the style dropdown or room for it, so both
        Strength sliders start and end on the same edges (round 13)."""
        row = QHBoxLayout()
        row.setSpacing(GUTTER)
        row.addWidget(toggle)
        label = _label("Strength", "option")
        row.addWidget(label)
        row.addWidget(slider, 1)
        if extra is not None:
            extra.setFixedWidth(STYLE_W)
            row.addWidget(extra)
        else:
            row.addSpacing(GUTTER + STYLE_W)  # a box layout puts no spacing next to a spacer
        self._dim_labels[slider] = label
        return row

    def _build_step2(self) -> Step:
        step = Step(2, STEPS[1])
        step.body.setSpacing(6)  # three rows and the echo row: a little tighter than the other steps
        self._dim_labels: dict[BarSlider, QLabel] = {}
        self.denoise_toggle = StepToggle("Denoise")
        self.denoise_toggle.clicked.connect(lambda on: self._switch(denoise=on))
        self.strength_bar = BarSlider("Denoise strength", reset=70, compact=True)
        self.strength_bar.setToolTip("Higher takes out more fan, keyboard and room sound in the pauses")
        self.strength_bar.valueChanged.connect(self._strength_changed)
        step.body.addLayout(self._strength_row(self.denoise_toggle, self.strength_bar))
        # Remove speaker echo sits under Denoise; in the engine it stays independent of the Denoise switch.
        echo = QHBoxLayout()
        echo.setSpacing(12)
        echo.setContentsMargins(SECOND_W + 12, 0, 0, 0)  # under the Strength sliders
        self.echo_switch = ToggleSwitch()
        self.echo_switch.setAccessibleName("Remove speaker echo")
        self.echo_switch.clicked.connect(self._echo_clicked)
        echo.addWidget(self.echo_switch, 0, Qt.AlignmentFlag.AlignTop)
        echo_text = QVBoxLayout()
        echo_text.setSpacing(0)
        echo_text.addWidget(_label("Remove speaker echo", "option"))
        self.echo_desc = _label(ECHO_TEXT, "hint", wrap=True)
        echo_text.addWidget(self.echo_desc)
        echo.addLayout(echo_text, 1)
        if self.platform != "win32":
            self.echo_switch.setEnabled(False)  # v1: Windows only (J06 plan, open question 2)
            self.echo_switch.setToolTip(ECHO_MAC)
        step.body.addLayout(echo)
        self.style_toggle = StepToggle("Voice style")
        self.style_toggle.clicked.connect(lambda on: self._switch(style=on))
        self.style_strength_bar = BarSlider(
            "Voice style strength", reset=50, compact=True, formatter=lambda v: f"{v / 100:.2f}"
        )
        self.style_strength_bar.setToolTip("How much of the style: 0 leaves your voice as it is, 1 doubles the style")
        self.style_strength_bar.valueChanged.connect(self._style_strength_changed)
        self.style_combo = Dropdown()
        self.style_combo.setAccessibleName("Voice style")
        self.style_combo.addItems([p.name for p in PRESETS.values()])
        self.style_combo.activated.connect(self._style_chosen)
        step.body.addLayout(self._strength_row(self.style_toggle, self.style_strength_bar, self.style_combo))
        fine = QHBoxLayout()
        fine.setSpacing(12)
        self.fine_tune_toggle = StepToggle("Fine-tune (advanced)")
        self.fine_tune_toggle.clicked.connect(lambda on: self._switch(fine_tune=on))
        fine.addWidget(self.fine_tune_toggle)
        self.show_plugin_button = _button(SHOW_PLUGIN)
        self.show_plugin_button.setToolTip("Opens the filter window, as in OBS Studio")
        self.show_plugin_button.clicked.connect(self.show_filters)
        fine.addWidget(self.show_plugin_button)
        self.filters_label = _label("", "hint")
        fine.addWidget(self.filters_label, 1)
        step.body.addLayout(fine)
        return step

    def _build_step3(self) -> Step:
        step = Step(3, STEPS[2], inline=True)
        self.out_meter = LevelMeter(clean=True)
        self.out_meter.setToolTip("What your call hears")
        self.call_volume = BarSlider("Voice volume", reset=50, formatter=voice_volume_text, value_width=VALUE_W)
        self.call_volume.setToolTip(CALL_VOLUME_TIP)
        self.call_volume.valueChanged.connect(self._loudness_changed)
        step.body.addLayout(self._meter_row(self.out_meter, self.call_volume))
        return step

    def _build_step4(self) -> Step:
        step = Step(4, STEPS[3], inline=True)
        self.output_combo = DeviceComboBox()
        self.output_combo.setAccessibleName("Send clean voice to")
        self.output_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.output_combo.setMinimumContentsLength(16)
        self.output_combo.setPlaceholderText("VB-CABLE (not installed)")
        self.output_combo.setFixedHeight(OUTPUT_H)
        self.output_combo.opening.connect(self._picker_opening)
        self.output_combo.activated.connect(self._output_combo_chosen)
        step.body.addWidget(self.output_combo)
        self.use_hint = _label(use_hint(self.platform), "instruction", wrap=True)
        self.use_hint.setTextFormat(Qt.TextFormat.RichText)
        step.body.addWidget(self.use_hint)
        self.cable_title = _label("", "noteTitle", wrap=True)
        self.cable_body = _label("", "note", wrap=True)
        self.download_button = _button("Download VB-CABLE", primary=True)
        self.download_button.clicked.connect(self.open_cable_download)
        self.cable_credit = _label("", "hint", wrap=True)
        for w in (self.cable_title, self.cable_body):
            step.body.addWidget(w)
        step.body.addWidget(self.download_button, 0, Qt.AlignmentFlag.AlignLeft)
        step.body.addWidget(self.cable_credit)
        return step

    def _build_step5(self) -> Step:
        step = Step(5, STEPS[4], inline=True, note="(optional)")
        rec_row = QHBoxLayout()
        rec_row.setSpacing(10)
        self.rec_dot = RecordingDot()
        rec_row.addWidget(self.rec_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        self.rec_label = _label("", "option")
        rec_row.addWidget(self.rec_label)
        self.rec_time = _label("", "time")
        self.rec_time.setFont(tabular(self.rec_time.font()))
        rec_row.addWidget(self.rec_time)
        rec_row.addStretch(1)
        self.open_folder_button = _button("Open folder")
        self.open_folder_button.clicked.connect(self.open_folder)
        rec_row.addWidget(self.open_folder_button)
        self.rec_switch = ToggleSwitch()
        self.rec_switch.setAccessibleName("Backup recording on or off")
        self.rec_switch.clicked.connect(self._recording_clicked)
        rec_row.addWidget(self.rec_switch)
        step.body.addLayout(rec_row)
        self.rec_detail = _label("", "hint", wrap=True)
        self.rec_detail.hide()
        step.body.addWidget(self.rec_detail)
        self.backup_bar = step
        return step

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        footer.setObjectName("footer")
        footer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        column = QVBoxLayout(footer)
        column.setContentsMargins(MARGIN, 0, MARGIN, 8)
        column.setSpacing(6)
        column.addWidget(_rule())
        row = QHBoxLayout()
        row.setSpacing(12)
        mac = self.platform == "darwin"
        self.close_note = CloseNote(
            "Closing keeps Clean Voice running in the menu bar."
            if mac
            else "Closing keeps Clean Voice open in the tray."  # "running" did not fit one line at 760 px
        )
        self.close_note.setProperty("role", "hint")
        self.close_note.setToolTip(
            "To quit, choose Quit from its menu-bar menu."
            if mac
            else "To quit, right-click the tray icon and choose Quit."
        )
        row.addWidget(self.close_note, 1)
        login = "Start at login" if mac else "Start with Windows"
        self.autostart_label = _label(login, "hint")
        self.autostart_label.setToolTip(
            "Clean Voice opens hidden in the menu bar each time you sign in."
            if mac
            else "Clean Voice opens hidden in the system tray each time you sign in."
        )
        # Each label sits closer to its own control than to the next group, and the mark to its version: the room
        # this saves keeps the close note on one line at 760 px at every display scale (theme-choice r1 F1).
        login_group = _row_group(8)
        login_group.addWidget(self.autostart_label)
        self.autostart_switch = ToggleSwitch()
        self.autostart_switch.setAccessibleName(login)
        self.autostart_switch.clicked.connect(self._autostart_clicked)
        login_group.addWidget(self.autostart_switch)
        row.addLayout(login_group)
        row.addLayout(self._build_theme_choice())
        version_group = _row_group(6)
        version_group.addWidget(Logo(16), 0, Qt.AlignmentFlag.AlignVCenter)
        self.version_label = _label(f"v{__version__}", "version")
        version_group.addWidget(self.version_label)
        row.addLayout(version_group)
        self.updates_view = UpdateView(self.updates)
        row.addWidget(self.updates_view.check_button)
        column.addLayout(row)
        self.autostart_note = _label("", "hint", wrap=True)
        _set_prop(self.autostart_note, "warn", True)
        self.autostart_note.hide()
        column.addWidget(self.autostart_note)
        column.addWidget(self.updates_view)
        self.footer = footer
        return footer

    def _build_theme_choice(self) -> QHBoxLayout:
        """The footer's "Theme" dropdown: Automatic, Light, Dark. As compact as the switch row it sits in, so the
        footer keeps its height and the five steps their room."""
        group = _row_group(8)  # the row's 12 px from the switch before it
        self.theme_label = _label("Theme", "hint")
        self.theme_label.setToolTip("Automatic follows your computer's light or dark setting")
        group.addWidget(self.theme_label)
        combo = Dropdown()
        combo.setObjectName("themeCombo")
        combo.setAccessibleName("Theme")
        combo.setToolTip(self.theme_label.toolTip())
        for mode, text in zip(theme.MODES, THEME_TEXTS, strict=True):
            combo.addItem(text, mode)
        _select_data(combo, self.theme_mode)
        combo.setFixedHeight(THEME_H)
        widest = max(QFontMetricsF(ui_font(combo, 12)).horizontalAdvance(t) for t in THEME_TEXTS)
        # The borders, the text's left padding, the text, then the style sheet's 28 px drop-down zone, which holds the
        # chevron (3 px clear of the text).
        combo.setFixedWidth(2 + 8 + math.ceil(widest) + 28)
        combo.activated.connect(self._theme_chosen)
        self.theme_label.setBuddy(combo)
        group.addWidget(combo, 0, Qt.AlignmentFlag.AlignVCenter)
        self.theme_combo = combo
        return group

    def tab_chain(self) -> list[QWidget]:
        """The keyboard's order (T30): the header, then Steps 1-5, then the footer."""
        return [
            self.power_button,
            self.input_button,
            self.mic_gain,
            self.test_button,
            self.monitor_button,
            self.monitor_volume,
            self.denoise_toggle,
            self.strength_bar,
            self.echo_switch,
            self.style_toggle,
            self.style_strength_bar,
            self.style_combo,
            self.fine_tune_toggle,
            self.show_plugin_button,
            self.call_volume,
            self.output_combo,
            self.download_button,
            self.open_folder_button,
            self.rec_switch,
            self.autostart_switch,
            self.theme_combo,
            self.updates_view.check_button,
            self.updates_view.install_button,
            self.updates_view.cancel_button,
        ]

    def _set_tab_order(self) -> None:
        chain = self.tab_chain()
        for first, second in itertools.pairwise(chain):
            QWidget.setTabOrder(first, second)

    # ------------------------------------------------------------------ saved choices

    def _apply_prefs(self) -> None:
        """Show the saved choices on the controls (main.js:687-689)."""
        s = self.prefs.settings
        self.call_volume.set_value(s.loudness)
        self.strength_bar.set_value(s.strength)
        self.echo_switch.setChecked(self.prefs.echo and self.platform == "win32")
        self.rec_switch.setChecked(self.prefs.recording)
        self.power_button.setChecked(s.cleaning)
        self.style_strength_bar.set_value(self.prefs.style_strength * 100)
        self.monitor_volume.set_value(monitor_position(self.prefs.monitor_volume))
        self._show_switches()

    def _change(self, **changes: object) -> None:
        self.prefs = dataclasses.replace(self.prefs, **changes)  # type: ignore[arg-type]
        self._dirty = True
        self._save_timer.start()

    def save_prefs(self) -> None:
        self._save_timer.stop()
        self._take_plugin_state()
        if not self._dirty:
            return
        try:
            prefs.save(self.prefs_path, self.prefs)
            self._dirty = False
        except OSError:
            pass  # a read-only config folder: the choices last for this session only

    def _take_plugin_state(self) -> None:
        """Every loaded VST plug-in's state into its filter, before a save (obs-vst.cpp vst_save :140-145): what the
        guest changed in a plug-in's own editor is saved without any change on this screen."""
        settings = self.prefs.settings
        taken = self.fine.with_state(settings)
        if taken is not settings:
            self.engine.set_settings(taken)  # the same list: the chain keeps every instance
            self.prefs = dataclasses.replace(self.prefs, settings=taken)
            self._dirty = True

    def _send(self, settings: Settings) -> None:
        self.engine.set_settings(settings)
        self._change(settings=settings)
        self._show_switches()

    # ------------------------------------------------------------------ Step 2: the three switches and their rows

    def _switch(self, **switch: bool) -> None:
        """Denoise, Voice style or Fine-tune: only what the guest switches changes the chain."""
        self._send(replace(self.prefs.settings, **switch))
        self.apply(self._status)

    def _show_switches(self) -> None:
        s = self.prefs.settings
        for toggle, on in (
            (self.denoise_toggle, s.denoise),
            (self.style_toggle, s.style),
            (self.fine_tune_toggle, s.fine_tune),
        ):
            if toggle.isChecked() != on:
                toggle.setChecked(on)
        # A row whose switch is off, or every row while the power is off, is greyed but stays usable, so a guest can
        # preset it.
        for slider, on in ((self.strength_bar, s.denoise), (self.style_strength_bar, s.style)):
            dim = not (on and s.cleaning)
            slider.set_dimmed(dim)
            _set_prop(self._dim_labels[slider], "dim", dim)
        _set_prop(self.style_combo, "dim", not (s.style and s.cleaning))
        index = list(PRESETS).index(self.prefs.preset)
        if self.style_combo.currentIndex() != index:
            self.style_combo.setCurrentIndex(index)
        n = len(s.filters)
        _set_text(self.filters_label, f"{n} filter{'' if n == 1 else 's'}" if n else "No filters yet")

    def _send_style(self, preset: str, strength: float) -> None:
        self._change(preset=preset, style_strength=strength)
        self._send(replace(self.prefs.settings, bands=style_bands(preset, strength)))

    def _style_chosen(self, index: int) -> None:
        keys = list(PRESETS)
        if 0 <= index < len(keys):
            self._send_style(keys[index], self.prefs.style_strength)

    def _style_strength_changed(self, value: int) -> None:
        self._send_style(self.prefs.preset, value / 100.0)

    def _strength_changed(self, value: int) -> None:
        self._send(replace(self.prefs.settings, strength=float(value)))

    def _send_filters(self, settings: Settings) -> None:
        """The filter window's edits: to the engine at once, saved after the usual delay, shown in the subline."""
        self._send(settings)
        self.apply(self._status)

    def show_filters(self) -> None:
        """Show plugin: opens the Fine-tune filter window, non-modal; a second click brings the open one to the front
        (OBS opens one Filters window per source)."""
        window = self.filter_window
        if window is None:
            window = filter_window.FilterWindow(self.fine, self.palette_now, self.platform)
            window.setWindowIcon(self.windowIcon())
            window.closed.connect(self.save_prefs)  # OBSBasicFilters::closeEvent saves the project (:602-612)
            self.filter_window = window
        window.show()
        window.setWindowState(window.windowState() & ~Qt.WindowState.WindowMinimized)
        window.raise_()
        window.activateWindow()

    # ------------------------------------------------------------------ Step 3: Voice volume

    def _loudness_changed(self, value: int) -> None:
        self._send(replace(self.prefs.settings, loudness=float(value)))

    # ------------------------------------------------------------------ Step 1: the mic's own level, Hear myself

    def _mic_gain_text(self, value: int) -> str:
        slider = getattr(self, "mic_gain", None)  # None while the slider itself is being built
        if slider is None or not slider.isEnabled():
            return "–"
        return db_text(self._mic_db) if self._mic_db is not None else f"{value} %"

    def _gain_mic(self) -> str | None:
        return self.current_mic() if self._mics else None

    def sync_mic_gain(self) -> None:
        """Show the selected mic's OS level and its dB: read on every tick, so a change in the system settings or
        another mic shows within a tick. Greyed out when the mic has no level that can be set from here."""
        mic = self._gain_mic()
        try:
            level = self.mic_levels.level(mic) if mic else None
            db = self.mic_levels.level_db(mic) if mic and level is not None else None
        except Exception:  # noqa: BLE001 - never crash the window over a level
            level, db = None, None
        slider = self.mic_gain
        ok = level is not None
        if slider.isEnabled() != ok:
            slider.setEnabled(ok)
            slider.setToolTip(MIC_LEVEL_TIP if ok else MIC_LEVEL_FIXED)
        if db != self._mic_db:
            self._mic_db = db
            slider.refresh_text()
        if level is None:
            slider.set_value(0)  # greyed: nothing of the previous mic's level stays on the handle
        elif not (slider.dragging and mic == self._mic_gain_mic):
            slider.set_value(level * 100.0)
        self._mic_gain_mic = mic

    def _mic_gain_changed(self, value: int) -> None:
        mic = self._gain_mic()
        if mic is None:
            return
        try:
            self.mic_levels.set_level(mic, value / 100.0)
        except Exception:  # the next read shows the level the OS really has
            log.debug("setting the mic level failed", exc_info=True)

    def _monitor_volume_changed(self, position: int) -> None:
        percent = monitor_percent(position)
        self._change(monitor_volume=percent)
        self.engine.set_monitor_volume(percent)

    # ------------------------------------------------------------------ the power button

    def set_cleaning(self, on: bool) -> None:
        """The power button and the tray's "Clean my voice": off is a bypass, others hear the raw mic."""
        self.power_button.setChecked(on)
        if on != self.prefs.settings.cleaning:
            self._send(replace(self.prefs.settings, cleaning=on))
        if self.tray is not None:
            self.tray.sync(on, self._status.recording and self._status.state is State.ON)
        self.apply(self._status)

    def cleaning(self) -> bool:
        return self.prefs.settings.cleaning

    # ------------------------------------------------------------------ devices

    def _picker_opening(self) -> None:
        # A fresh list only while no stream runs (report cause 5); the engine ignores it otherwise too.
        if self._status.state is not State.ON:
            self.engine.refresh()

    def current_mic(self) -> str | None:
        return self._status.mic or self.prefs.mic or self.engine.default_mic()

    def current_output(self) -> str | None:
        return self._status.output or self.prefs.output or self._status.cable

    def _sync_devices(self) -> None:
        mics, outputs = self.engine.mics(), self.engine.outputs()
        if mics != self._mics:
            self._mics = mics
            if self.mic_menu.isVisible():
                self.mic_menu.fill(mics, self.current_mic())
        if outputs != self._outputs and not self.output_combo.view().isVisible():
            self._outputs = outputs
            self._fill(self.output_combo, outputs, NO_OUTPUT_TEXT)
        self._select(self.output_combo, self._outputs or [], self.current_output())
        if self.current_output() is None and self.output_combo.currentIndex() != -1 and self._outputs:
            self.output_combo.blockSignals(True)
            self.output_combo.setCurrentIndex(-1)  # nothing goes anywhere yet (J07 r1 m8)
            self.output_combo.blockSignals(False)
        mic = self.current_mic() if self._mics else None
        text = mic or NO_MIC_TEXT
        if self.input_button.text() != text:
            self.input_button.setText(text)
            self.input_button.setToolTip(text)
            self.input_button.update()

    @staticmethod
    def _fill(combo: QComboBox, names: list[str], empty: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        if names:
            combo.addItems(names)
            combo.setEnabled(True)
        else:
            combo.addItem(empty)
            combo.setEnabled(False)
        combo.blockSignals(False)

    @staticmethod
    def _select(combo: QComboBox, names: list[str], current: str | None) -> None:
        if current in names and combo.currentText() != current and not combo.view().isVisible():
            combo.blockSignals(True)
            combo.setCurrentIndex(names.index(current))
            combo.blockSignals(False)

    def choose_mic(self, name: str) -> None:
        if name != self.current_mic():
            self.engine.select_mic(name)
        if name != self.prefs.mic:
            self._change(mic=name)

    def _output_combo_chosen(self, index: int) -> None:
        if self._outputs and 0 <= index < len(self._outputs):
            name = self._outputs[index]
            if name != self.current_output():
                self.engine.select_output(name)
            if name != self.prefs.output:
                self._change(output=name)

    def toggle_mic_menu(self) -> None:
        if self.mic_menu.isVisible():
            self.close_mic_menu()
        else:
            self.open_mic_menu()

    def _place_menu(self) -> None:
        picker = self.input_button
        under = QRect(picker.mapTo(self.root, picker.rect().topLeft()), picker.size())
        self.mic_menu.place(under, self.root.rect())

    def open_mic_menu(self) -> None:
        self.body.ensureWidgetVisible(self.input_button, 0, 0)
        self._picker_opening()
        self._sync_devices()
        self.mic_menu.fill(self._mics or [], self.current_mic())
        self._place_menu()
        self.mic_menu.show()
        self.mic_menu.raise_()
        self.input_button.set_expanded(True)
        current = next((i for i in self.mic_menu.items if i.isChecked()), None)
        (current or (self.mic_menu.items[0] if self.mic_menu.items else self.mic_menu)).setFocus()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def close_mic_menu(self) -> None:
        if not hasattr(self, "mic_menu") or self.mic_menu.isHidden():
            return
        self.mic_menu.hide()
        self.input_button.set_expanded(False)
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)

    def _menu_chosen(self, name: str) -> None:
        self.close_mic_menu()
        self.choose_mic(name)
        self.input_button.setFocus(Qt.FocusReason.OtherFocusReason)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # A click anywhere but the menu or its picker closes the menu (main.js:579-581).
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(event, QMouseEvent):
            where = event.globalPosition().toPoint()
            inside = self.mic_menu.geometry().contains(self.root.mapFromGlobal(where))
            on_button = self.input_button.rect().contains(self.input_button.mapFromGlobal(where))
            if not inside and not on_button:
                self.close_mic_menu()
        return False

    # ------------------------------------------------------------------ the other switches

    def _recording_clicked(self, on: bool) -> None:
        self.engine.set_recording(on)
        self._change(recording=on)

    def _echo_clicked(self, on: bool) -> None:
        self.engine.set_echo(on)
        self._change(echo=on)
        self.apply(self.engine.status())

    def _monitor_clicked(self, on: bool) -> None:
        self._monitor_wanted = on
        self._monitor_pending_until = time.monotonic() + 1.0  # until the engine has answered
        self.engine.set_monitor(on)
        self._show_monitor(on)

    def _show_monitor(self, on: bool) -> None:
        tip = "Stop hearing myself" if on else "Hear myself"  # main.js:522
        button = self.monitor_button
        if button.isChecked() != on:
            button.setChecked(on)
        if button.toolTip() != tip:
            button.setToolTip(tip)
            button.setAccessibleName(tip)

    def _test_clicked(self) -> None:
        if self._status.mic_test is MicTestState.IDLE:
            self.engine.start_mic_test()
        else:
            self.engine.cancel_mic_test()
        self.apply(self.engine.status())

    def _read_autostart(self) -> None:
        try:
            on = self.login_item.is_enabled()
        except Exception:  # noqa: BLE001 - an unreadable login item just shows off
            on = False
        self.autostart_switch.setChecked(on)

    def _autostart_clicked(self, on: bool) -> None:
        try:
            if on:
                self.login_item.enable(autostart.current_command())
            else:
                self.login_item.disable()
            failed = False
        except Exception:  # noqa: BLE001 - shown on screen, the switch reads the real state back
            failed = True
        self._read_autostart()  # main.js:638: the switch shows what really happened
        _set_text(self.autostart_note, "Clean Voice could not change this. Try again, or restart your computer.")
        _set_visible(self.autostart_note, failed)

    def open_folder(self) -> None:
        folder = self.engine.recordings_dir
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def open_cable_download(self) -> None:
        QDesktopServices.openUrl(QUrl(CABLE_DOWNLOAD_URL))

    # ------------------------------------------------------------------ the tick

    def tick(self) -> None:
        status = self.engine.status()
        self._status = status  # the device sync below reads the current mic and output from it
        self._sync_devices()
        self.sync_mic_gain()
        self.apply(status)
        self.fine.tick()  # unload the VST plug-ins the chain has let go of

    def apply(self, status: Status) -> None:
        now = time.monotonic()
        dt = min(0.5, max(0.0, now - self._last_tick)) or TICK_MS / 1000
        self._last_tick = now
        self._status = status
        on = status.state is State.ON
        cleaning = self.prefs.settings.cleaning
        view = describe(status, cleaning, self.platform)
        if view != self._view:
            self._view = view
            self._show_view(view)
        reaching = view.headline in (HEAD_CLEAN, HEAD_RAW)
        _set_text(self.subline, self._subline(view))
        self.header.set_lit(view.power == "signal")
        self.power_button.set_tone(view.power)
        if self.power_button.isChecked() != cleaning:
            self.power_button.setChecked(cleaning)
        self.out_meter.set_muted(not reaching)  # grey: the call can't hear this (J07 r2 n4)
        self.in_meter.set_muted(
            status.state in (State.NO_MIC, State.MIC_LOST, State.MIC_DENIED, State.ERROR)
        )  # and no mic to show
        self.in_meter.active = self.out_meter.active = on
        self.in_meter.set_level(status.input_level if on else 0.0, dt)
        self.out_meter.set_level(status.level if on else 0.0, dt)
        self._show_recording(status, on)
        self._show_mic_test(status, on)
        self._show_echo(status)
        # Hear myself: the button shows the guest's click until the engine has answered, then the engine's state.
        if now >= self._monitor_pending_until:
            self._monitor_wanted = status.monitor
        self._show_monitor(self._monitor_wanted)
        note = status.monitor_error or ""
        _set_text(self.dock_note, note)
        _set_visible(self.dock_note, bool(note))
        if self.tray is not None:
            self.tray.sync(cleaning, status.recording and on)

    def _subline(self, view: View) -> str:
        if view.notice is not None:
            return f"Step {view.notice.step} shows what to do."
        if view.headline == HEAD_RAW:
            return POWER_OFF_SUB
        return describe_chain(self.prefs.settings, self.prefs.preset)

    def _show_view(self, view: View) -> None:
        _set_text(self.headline, view.headline)
        _set_prop(self.headline, "tone", view.tone)
        notice = view.notice
        mic_notice = notice if notice is not None and notice.kind == "mic" else None
        call_notice = notice if notice is not None and notice.kind in ("cable", "output") else None
        _set_text(self.mic_title, mic_notice.title if mic_notice else "")
        _set_text(self.mic_body, mic_notice.body if mic_notice else "")
        _set_visible(self.mic_title, mic_notice is not None)
        _set_visible(self.mic_body, mic_notice is not None)
        self.mic_body.setToolTip(mic_notice.details or "" if mic_notice else "")
        _set_text(self.cable_title, call_notice.title if call_notice else "")
        _set_text(self.cable_body, call_notice.body if call_notice else "")
        _set_text(self.cable_credit, (call_notice.credit or "") if call_notice else "")
        _set_visible(self.cable_title, call_notice is not None)
        _set_visible(self.cable_body, call_notice is not None)
        _set_visible(self.cable_credit, call_notice is not None and call_notice.credit is not None)
        _set_visible(self.download_button, call_notice is not None and call_notice.action is not None)
        _set_visible(self.use_hint, call_notice is None)
        if call_notice is not None and call_notice.kind == "output":
            # No Settings page any more: the dropdown the guest needs is right here.
            self.output_combo.setFocus(Qt.FocusReason.OtherFocusReason)
        self._reveal_notice()

    def _reveal_notice(self) -> None:
        """A Step 4 problem: scroll Step 4 into view, as the subline sends the guest there (Step 1 is always in view).
        A backup error under Step 5 is scrolled into view too (round 15); a Step 4 problem comes first."""
        if self._revealing:  # its own scrolling and room change the range too
            return
        self._revealing = True
        try:
            self._reveal()
        finally:
            self._revealing = False

    def _reveal(self) -> None:
        self._steps_layout.activate()  # the notice that just appeared has its place
        view = self._view
        step4 = view is not None and view.notice is not None and view.notice.step == 4
        backup = self.rec_detail.isVisibleTo(self)
        if not (step4 or backup):
            self._set_room(0)
            return
        self.body.verticalScrollBar().setValue(0)  # from the top: scroll only as far as the notice needs
        if backup:
            self.body.ensureWidgetVisible(self.rec_detail, 0, 8)
        if step4:
            last = self.cable_credit if self.cable_credit.isVisibleTo(self) else self.cable_body
            self.body.ensureWidgetVisible(last, 0, 8)
            self.body.ensureWidgetVisible(self.output_combo, 0, 8)
        self._snap_to_step()

    def row_tops(self) -> list[int]:
        """Where each step and each row of controls starts, in the scrolled steps' coordinates."""
        tops = []
        for step in self.steps:
            tops.append(step.y())
            for i in range(step.body.count()):
                item = step.body.itemAt(i)
                if item is not None and not item.isEmpty():
                    tops.append(step.y() + item.geometry().top())
        return sorted(set(tops))

    def _room(self) -> int:
        return self._steps_layout.contentsMargins().bottom() - BODY_BOTTOM

    def _set_room(self, room: int) -> None:
        """Extra room under Step 5, so a snapped view can scroll that far; changed only when it must (each change
        changes the scroll range, which reveals again)."""
        if room != self._room():
            self._steps_layout.setContentsMargins(MARGIN, BODY_TOP, MARGIN, BODY_BOTTOM + room)
            self._steps_layout.activate()
            # The scroll area grows its widget on the layout's request: deliver it now, so the new end is scrollable.
            QApplication.sendPostedEvents(self.body.widget(), QEvent.Type.LayoutRequest)

    def _snap_to_step(self) -> None:
        """Move the scroll down to just above the next step or row, so the view never starts halfway through a control
        (round 16). Further down only shows more of what is below, so the notice stays in view; when the steps end
        too soon for that, a few pixels of room go under Step 5."""
        bar = self.body.verticalScrollBar()
        value = bar.value()
        snaps = [top - SNAP_ROOM for top in self.row_tops() if top - SNAP_ROOM >= value]
        if value <= 0 or not snaps:
            self._set_room(0)
            return
        self._set_room(max(0, snaps[0] - (bar.maximum() - self._room())))
        bar.setValue(snaps[0])

    def _show_recording(self, status: Status, on: bool) -> None:
        recording = status.recording and on
        paused = status.recording and not on
        self.rec_dot.set_recording(recording)
        _set_visible(self.rec_dot, status.recording_enabled)
        if self.rec_switch.isChecked() != status.recording_enabled:
            self.rec_switch.setChecked(status.recording_enabled)
        if status.recording_error:
            label, detail = "Backup not saved", status.recording_error  # the engine's words, naming the folder
        elif recording:
            label, detail = "Backup recording", "Your clean voice and your raw mic are saved on this computer."
        elif paused:
            label, detail = "Backup paused", "It carries on in the same file when your mic is back."
        elif status.recording_enabled:
            label, detail = "Backup on", "Recording starts by itself once your microphone and VB-CABLE are ready."
        else:
            label, detail = "Backup off", "Switch it on to save your clean voice and your raw mic on this computer."
        _set_text(self.rec_label, label)
        self.rec_label.setToolTip(detail)
        _set_text(self.rec_time, format_duration(status.recording_seconds) if status.recording else "")
        _set_visible(self.rec_time, status.recording)
        _set_text(self.rec_detail, status.recording_error or "")
        _set_prop(self.rec_detail, "warn", status.recording_error is not None)
        appears = status.recording_error is not None and self.rec_detail.isHidden()
        _set_visible(self.rec_detail, status.recording_error is not None)
        if appears and self.isVisible():
            QTimer.singleShot(0, self._reveal_notice)

    def _show_mic_test(self, status: Status, on: bool) -> None:
        test = status.mic_test
        if test is MicTestState.RECORDING:
            left = max(1, math.ceil(5.0 - status.mic_test_seconds))
            phase = (f"Speak now  {left} s", True, status.mic_test_seconds / 5.0)
        elif test is MicTestState.PLAYING:
            phase = ("Playing back", True, status.mic_test_seconds / 5.0)
        else:
            phase = (MIC_TEST_TEXT, False, 0.0)
        enabled = on or test is not MicTestState.IDLE
        self.test_button.set_phase(*phase)
        if self.test_button.isEnabled() != enabled:
            self.test_button.setEnabled(enabled)
        error = status.mic_test_error if test is MicTestState.IDLE else None
        _set_text(self.test_label, error or "")
        _set_visible(self.test_label, error is not None)

    def _show_echo(self, status: Status) -> None:
        if self.platform != "win32":
            text, warn = ECHO_MAC, False
        elif status.echo in (EchoState.PAUSED, EchoState.UNAVAILABLE) and status.echo_reason:
            text, warn = status.echo_reason, True  # main.js:611-617
        else:
            text, warn = ECHO_TEXT, False
        _set_text(self.echo_desc, text)
        _set_prop(self.echo_desc, "warn", warn)

    # ------------------------------------------------------------------ window life

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        if self.mic_menu.isVisible():
            self._place_menu()

    def showEvent(self, event: object) -> None:
        super().showEvent(event)  # type: ignore[arg-type]
        if hasattr(self, "_timer"):
            self._timer.setInterval(TICK_MS)
        QTimer.singleShot(0, self._reveal_notice)  # once the steps have their sizes

    def hideEvent(self, event: object) -> None:
        super().hideEvent(event)  # type: ignore[arg-type]
        if hasattr(self, "_timer"):
            self._timer.setInterval(HIDDEN_TICK_MS)

    def fit_to_screen(self) -> None:
        """Size and centre the window for its screen before it first shows (``opening_geometry``): a short screen, or
        one at 150 %, would otherwise put the footer under the taskbar."""
        if self._placed:
            return
        self._placed = True
        self.winId()  # makes the native window, so that the frame's size is known
        frame, inside = self.frameGeometry(), self.geometry()
        margins = QMargins(
            inside.left() - frame.left(),
            inside.top() - frame.top(),
            frame.right() - inside.right(),
            frame.bottom() - inside.bottom(),
        )
        self.setGeometry(opening_geometry(self.screen().availableGeometry(), margins))

    def show_from_tray(self) -> None:
        self.fit_to_screen()  # a --minimized launch shows for the first time here
        self.show()
        self.setWindowState((self.windowState() & ~Qt.WindowState.WindowMinimized) | Qt.WindowState.WindowActive)
        self.raise_()
        self.activateWindow()

    def quit_app(self) -> None:
        """Quit for real (the tray's Quit, or the end of a launch check): save, close the engine, leave."""
        self._quitting = True
        self.close()
        if not self._closed:
            self._shutdown(close_engine=True)
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _shutdown(self, close_engine: bool) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        app = QApplication.instance()
        watcher = getattr(self, "_activation", None)
        if app is not None and watcher is not None:
            app.removeEventFilter(watcher)
        self.close_mic_menu()
        if self.filter_window is not None:
            self.filter_window.close()  # saves, as OBS's closeEvent does
        self.save_prefs()
        self.mic_levels.close()
        if self.tray is not None:
            self.tray.hide()
        if close_engine:
            self.engine.close()

    def closeEvent(self, event: object) -> None:
        if self.tray is not None and not self._quitting:
            # Close to the tray: the voice keeps going (main.rs:230-236).
            event.ignore()  # type: ignore[attr-defined]
            self.save_prefs()
            self.hide()
            return
        self._shutdown(close_engine=self._owns_engine or self._quitting)
        super().closeEvent(event)  # type: ignore[arg-type]


def recordings_folder() -> Path:
    from PySide6.QtCore import QStandardPaths

    music = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.MusicLocation)
    return Path(music or Path.home()) / APP_NAME


def _make_engine() -> EngineLike:
    from clean_voice.engine import Engine

    return Engine(recordings_folder())


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(prog="clean_voice", add_help=False)
    # Hidden: lets a check with no one at the screen prove the window opened, then exit.
    parser.add_argument("--quit-after-ms", type=int, default=None, help=argparse.SUPPRESS)
    # Start hidden in the tray: what the login item runs (main.rs:206-209).
    parser.add_argument(autostart.MINIMIZED, action="store_true", help=argparse.SUPPRESS)
    return parser.parse_known_args(argv)


def follow_scheme_style(app: QApplication) -> None:
    """Run a Qt style whose palette follows the Theme choice, so Qt draws the title bars (every window's, the VST
    editor's frame included) and the tray menu light or dark with it. Windows 10's default, windowsvista, keeps a
    light palette in Dark, and Qt then keeps those light (theme-choice r1 m2); Fusion follows the scheme. Windows 11's
    windows11 and macOS's own style already follow, and stay."""
    if app.style().name().lower() == "windowsvista":
        app.setStyle(WindowsFusion())


class WindowsFusion(QProxyStyle):
    """Fusion with windowsvista's question boxes (theme-choice r2 m4, r3 m5): Yes and No underline their letters when
    windowsvista would (while Alt is held, or always with Windows' "Underline access keys" on), the "?" icon at the
    Windows styles' 32 px rather than Fusion's 48, and windowsvista's 9 px above the buttons."""

    def __init__(self) -> None:
        super().__init__("fusion")
        # Asked about the underlines; polished with the app so it watches for Alt, as it does when it is the style.
        self._vista = QStyleFactory.create("windowsvista")

    def polish(self, target: QWidget | QApplication | QPalette) -> None:  # type: ignore[override]
        super().polish(target)  # type: ignore[arg-type]  # one of its three overloads
        if isinstance(target, QApplication) and self._vista is not None:
            self._vista.polish(target)
        if isinstance(target, QMessageBox) and (buttons := target.findChild(QDialogButtonBox)) is not None:
            buttons.setContentsMargins(0, 9, 0, 0)

    def unpolish(self, target: QWidget | QApplication) -> None:  # type: ignore[override]
        super().unpolish(target)  # type: ignore[arg-type]  # one of its two overloads
        if isinstance(target, QApplication) and self._vista is not None:
            self._vista.unpolish(target)

    def styleHint(
        self,
        hint: QStyle.StyleHint,
        option: QStyleOption | None = None,
        widget: QWidget | None = None,
        returnData: QStyleHintReturn | None = None,
    ) -> int:
        if hint == QStyle.StyleHint.SH_UnderlineShortcut and self._vista is not None:
            return self._vista.styleHint(hint, option, widget, returnData)
        return super().styleHint(hint, option, widget, returnData)

    def pixelMetric(
        self, metric: QStyle.PixelMetric, option: QStyleOption | None = None, widget: QWidget | None = None
    ) -> int:
        if metric == QStyle.PixelMetric.PM_MessageBoxIconSize:
            return 32
        return super().pixelMetric(metric, option, widget)


def show_window(window: MainWindow, minimized: bool) -> None:
    """``--minimized`` starts hidden, but only when there is a tray to bring the window back from."""
    if minimized and window.tray is not None:
        return
    window.fit_to_screen()
    window.show()


def main(argv: list[str] | None = None, engine: EngineLike | None = None, prefs_path: Path | None = None) -> int:
    """Start the app and return its exit code."""
    from clean_voice import tray

    args, qt_args = _parse_args(sys.argv[1:] if argv is None else argv)

    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication([sys.argv[0], *qt_args])
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(mark.icon())  # the title bar, the taskbar and (in briefcase dev) the Dock
    follow_scheme_style(app)
    fonts.load()
    font = QFont()
    font.setFamilies(UI_FONTS)
    QApplication.setFont(font)

    server: tray.InstanceServer | None = None
    if args.quit_after_ms is None:
        # One copy only: a second launch brings the first to the front and leaves (main.rs:198).
        name = tray.server_name()
        if tray.notify_existing(name):
            return 0
        server = tray.InstanceServer(name)

    nvafx.probe_async()  # is NVIDIA's audio effects runtime here? OBS asks once at start-up, never in the audio path
    window = MainWindow(engine, prefs_path)
    if server is not None:
        server.show_requested.connect(window.show_from_tray)
    if args.quit_after_ms is None and tray.tray_available():
        window.tray = tray.Tray(window)
        app.setQuitOnLastWindowClosed(False)
    show_window(window, args.minimized)
    # The quiet check for updates, 10 s after launch, off the UI thread; it shows only an update that exists.
    update_timer = QTimer(window)  # a child of the window: it cannot outlive it
    update_timer.setSingleShot(True)
    update_timer.timeout.connect(lambda: window.updates.check(quiet=True))
    update_timer.start(UPDATE_DELAY_MS)

    if args.quit_after_ms is not None:
        # A child of the window, so it cannot outlive this run and quit a later event loop.
        quit_timer = QTimer(window)
        quit_timer.setSingleShot(True)
        quit_timer.timeout.connect(window.close)
        quit_timer.timeout.connect(app.quit)

        def report_shown() -> None:
            # The exit code is the proof on every OS: on macOS stdout goes to a redacted system log.
            if not window.isVisible():
                print(NOT_SHOWN_MESSAGE, file=sys.stderr, flush=True)
                quit_timer.stop()
                window.close()
                app.exit(NOT_SHOWN_EXIT)
                return
            print(SHOWN_MESSAGE, flush=True)
            # The launch check also proves the audio stack is bundled. sounddevice is otherwise imported only on the
            # engine's thread, where a failure shows on screen instead of failing the check (a build once held only
            # PySide6, J05 Step 5).
            for module in audio_modules():
                try:
                    importlib.import_module(module)
                except ImportError as exc:
                    print(f"Audio library missing: {module} ({exc})", file=sys.stderr, flush=True)
                    quit_timer.stop()
                    window.close()
                    app.exit(NO_AUDIO_EXIT)
                    return
            # And that the bundled RNNoise library loads in the built app.
            try:
                from clean_voice import rnnoise

                rnnoise.load()
            except OSError as exc:
                print(f"RNNoise failed to load: {exc}", file=sys.stderr, flush=True)
                quit_timer.stop()
                window.close()
                app.exit(NO_RNNOISE_EXIT)

        # Runs once the event loop has started, i.e. after the window has been shown.
        QTimer.singleShot(0, report_shown)
        quit_timer.start(max(args.quit_after_ms, 0))

    code = app.exec()
    if window.tray is not None:
        window.quit_app()
    else:
        window.close()
    if server is not None:
        server.close()
    # Closing the last window and app.quit() each post a Quit; exec() takes one, and the other would end the next event
    # loop in this process (the in-process launch tests are followed by tests that wait on signals).
    QApplication.removePostedEvents(app, QEvent.Type.Quit)
    return code
