"""A VST 2 plug-in's own editor, in its own window (J10 plan §5 "Editor window").

It follows obs-vst's EditorWidget (``reference/obs-studio/plugins/obs-vst/EditorWidget.cpp``, ``win/EditorWidget-win.cpp``,
``mac/EditorWidget-osx.mm``) and VSTPlugin's openEditor / closeEditor / onEditorClosed (``VSTPlugin.cpp`` :291-343):
a separate top-level window, opened only for a plug-in with an editor, which hands the plug-in a native parent, sizes
itself from the plug-in's rect, follows the plug-in's resize requests, and tells the plug-in when it closes. Like OBS,
it never sends effEditIdle.

*Difference:* OBS makes a raw Win32 window (``CreateWindowW``) or an ``NSView`` and wraps it with
``QWidget::createWindowContainer``. Here the parent is a native child ``QWidget`` (``WA_NativeWindow``), whose
``winId()`` is an ``HWND`` on Windows and an ``NSView*`` on macOS: no window class registration and no PyObjC.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QGridLayout, QWidget

from clean_voice import vst

SOURCE_NAME = "Clean Voice"  # OBS titles the window with its source's name (VSTPlugin.cpp:324-333)
DEFAULT_SIZE = 300  # win/EditorWidget-win.cpp:62 and mac/EditorWidget-osx.mm:26: a plug-in that gives no rect


class EditorWindow(QWidget):
    """EditorWidget: the plug-in's editor in a window of its own (``open_editor`` makes it)."""

    _resize_requested = Signal(int, int)  # a plug-in may ask from any thread: queued to the UI thread

    def __init__(self, plugin: vst.VstPlugin, filter_name: str = "", platform: str = sys.platform) -> None:
        super().__init__(None)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.MSWindowsFixedSizeDialogHint)  # EditorWidget.cpp:24
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)  # onEditorClosed's deleteLater (VSTPlugin.cpp:302)
        self.plugin = plugin
        self._platform = platform
        self._opened = False  # editorOpened: effEditOpen was sent, so closing sends effEditClose
        if filter_name:  # VSTPlugin.cpp:328-333: "<source>: <filter> - <effect>", or "<source> - <effect>"
            self.setWindowTitle(f"{SOURCE_NAME}: {filter_name} - {plugin.effect_name}")
        else:
            self.setWindowTitle(f"{SOURCE_NAME} - {plugin.effect_name}")
        self.view = QWidget(self)
        self.view.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view)
        self._resize_requested.connect(self._resize)
        self._build()

    def _build(self) -> None:
        """buildEffectContainer, as each OS's file orders it."""
        plugin, handle = self.plugin, int(self.view.winId())
        if self._platform == "darwin":
            rect = plugin.editor_rect()  # mac :37-47: the rect first, as a fixed size
            if rect is not None:
                self.setFixedSize(*rect)
            else:
                self.resize(DEFAULT_SIZE, DEFAULT_SIZE)  # the 300 x 300 view of :26-29
            self._open(handle)  # :49
            return
        self._open(handle)  # win :46
        rect = plugin.editor_rect()  # :48-63
        if rect is None:
            # :62 sizes only the container; in a layout the window sets the view's size, so the window gets the 300.
            self.resize(DEFAULT_SIZE, DEFAULT_SIZE)
            return
        width, height = self._scaled(rect)
        self.view.resize(width, height)
        self.resize(width, height)

    def _open(self, handle: int) -> None:
        self._opened = True
        self.plugin.dispatch(vst.EFF_EDIT_OPEN, ptr=handle)

    def _scaled(self, rect: tuple[int, int]) -> tuple[int, int]:
        """win :51-58: the plug-in's rect is in device pixels, so it is divided by the screen's scale."""
        scale = self.devicePixelRatioF()
        return int(rect[0] / scale), int(rect[1] / scale)

    def request_resize(self, width: int, height: int) -> None:
        """audioMasterSizeWindow (VSTPlugin.cpp:59-64), from whichever thread the plug-in calls on."""
        self._resize_requested.emit(width, height)

    def _resize(self, width: int, height: int) -> None:
        """handleResizeRequest. Windows (:66-89) asks the plug-in for its rect again, as some plug-ins can't resize
        their window themselves; macOS (:54-57) takes the size it was asked for."""
        if self._platform == "darwin":
            self.setFixedSize(width, height)
            return
        rect = self.plugin.editor_rect()
        if rect is not None:
            self.resize(*self._scaled(rect))

    def closeEvent(self, event: QCloseEvent) -> None:
        """EditorWidget::closeEvent -> onEditorClosed (VSTPlugin.cpp:296-309)."""
        if self.plugin.editor is self:
            self.plugin.editor = None
        if self._opened and self.plugin.loaded:
            self._opened = False
            self.plugin.dispatch(vst.EFF_EDIT_CLOSE)
        super().closeEvent(event)


def open_editor(plugin: vst.VstPlugin, filter_name: str = "") -> EditorWindow | None:
    """openEditor (VSTPlugin.cpp:311-336): only for a loaded plug-in that has an editor, and only one at a time (a
    second call shows the open one again). None when the plug-in has no editor."""
    if not plugin.has_editor:
        return None  # :315-318
    current = plugin.editor
    if isinstance(current, EditorWindow):
        current.raise_()
        current.activateWindow()
        return current
    window = EditorWindow(plugin, filter_name)
    plugin.editor = window
    window.show()
    return window


def close_editor(plugin: vst.VstPlugin) -> None:
    """closeEditor (VSTPlugin.cpp:338-343)."""
    if plugin.editor is not None:
        plugin.editor.close()


def interface_buttons(plugin: vst.VstPlugin | None) -> tuple[bool, bool]:
    """Whether "Open Plug-in Interface" and "Close Plug-in Interface" show, as vst_properties / vst_changed set them
    (obs-vst.cpp:270-294, :307-323): neither without a loaded plug-in, Close while its editor is open, else Open."""
    if plugin is None or not plugin.loaded:
        return False, False
    if plugin.editor is not None:
        return False, True
    return True, False


def open_if_wanted(plugin: vst.VstPlugin, filter_name: str = "") -> EditorWindow | None:
    """The "Open interface when active" checkbox: the editor opens as soon as the plug-in loads (VSTPlugin.cpp:211-213), at
    launch too."""
    return open_editor(plugin, filter_name) if plugin.open_when_active else None
