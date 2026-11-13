"""The tray icon and the single instance (simple-clean-mic src-tauri/src/main.rs:160-236 and :198).

- ``Tray``: "Open Clean Voice", a "Clean my voice" check item, a separator and "Quit" (main.rs:160-164); left-click
  shows the window. Closing the window hides it here, and the voice keeps going (main.rs:230-236).
- ``InstanceServer`` and ``notify_existing``: one copy per user. A second launch asks the first to show itself and
  leaves, so two copies never fight over the same mic and cable.
"""

from __future__ import annotations

import getpass
import re
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from clean_voice import mark
from clean_voice.app import APP_NAME

if TYPE_CHECKING:
    from clean_voice.app import MainWindow

SHOW = b"show\n"
SHOWN = b"ok\n"
CONNECT_MS = 500
REPLY_MS = 3000


def tray_available() -> bool:
    return QSystemTrayIcon.isSystemTrayAvailable()


class Tray(QObject):
    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.menu = QMenu()
        self.open_action = QAction("Open Clean Voice", self)
        self.open_action.triggered.connect(window.show_from_tray)
        self.clean_action = QAction("Clean my voice", self)
        self.clean_action.setCheckable(True)
        self.clean_action.setChecked(window.cleaning())
        self.clean_action.toggled.connect(window.set_cleaning)
        self.quit_action = QAction("Quit", self)
        self.quit_action.triggered.connect(window.quit_app)
        self.menu.addAction(self.open_action)
        self.menu.addAction(self.clean_action)
        self.menu.addSeparator()
        self.menu.addAction(self.quit_action)
        self.icon = QSystemTrayIcon(mark.icon(), self)  # the same QIcon as the window and the app
        self.icon.setContextMenu(self.menu)
        self.icon.activated.connect(self._activated)
        self._tip = ""
        self.sync(window.cleaning(), False)
        self.icon.show()

    def _activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:  # a left click (main.rs:174-186)
            self.window.show_from_tray()

    def actions(self) -> list[str]:
        return [a.text() for a in self.menu.actions()]

    def sync(self, cleaning: bool, recording: bool) -> None:
        """Mirror the power button, and say in the tooltip whether a backup is being recorded."""
        if self.clean_action.isChecked() != cleaning:
            self.clean_action.blockSignals(True)
            self.clean_action.setChecked(cleaning)
            self.clean_action.blockSignals(False)
        tip = f"{APP_NAME}: {'on' if cleaning else 'off'}" + (", recording a backup" if recording else "")
        if tip != self._tip:
            self._tip = tip
            self.icon.setToolTip(tip)

    def tooltip(self) -> str:
        return self._tip

    def hide(self) -> None:
        self.icon.hide()


def server_name() -> str:
    """One name per user, so two people signed in at once each get their own copy."""
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - no user name: one shared name is still better than none
        user = "user"
    return "clean-voice-" + re.sub(r"[^A-Za-z0-9_.-]", "_", user)


def notify_existing(name: str) -> bool:
    """Ask a copy that is already running to show its window. True: one answered, so this launch should leave."""
    socket = QLocalSocket()
    socket.connectToServer(name)
    if not socket.waitForConnected(CONNECT_MS):
        return False
    socket.write(SHOW)
    socket.flush()
    socket.waitForBytesWritten(CONNECT_MS)
    # Wait for the first copy's answer before leaving: on Windows, a pipe closed before the other side has read it
    # loses what was written.
    socket.waitForReadyRead(REPLY_MS)
    socket.disconnectFromServer()
    return True


class InstanceServer(QObject):
    """Listens for later launches; each one that says "show" emits ``show_requested``."""

    show_requested = Signal()

    def __init__(self, name: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.name = name
        self._server = QLocalServer(self)
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        if not self._server.listen(name):
            # A stale socket file left by a crash (Unix); on Windows the pipe name is free again by itself.
            QLocalServer.removeServer(name)
            self._server.listen(name)
        self._server.newConnection.connect(self._accept)
        self._sockets: list[QLocalSocket] = []

    def listening(self) -> bool:
        return self._server.isListening()

    def _accept(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            if socket is None:
                return
            self._sockets.append(socket)
            socket.readyRead.connect(self._ready)
            socket.disconnected.connect(self._gone)
            if socket.bytesAvailable():
                self._read(socket)

    def _ready(self) -> None:
        socket = self.sender()
        if isinstance(socket, QLocalSocket):
            self._read(socket)

    def _gone(self) -> None:
        socket = self.sender()
        if isinstance(socket, QLocalSocket) and socket in self._sockets:
            self._sockets.remove(socket)
            socket.deleteLater()

    def _read(self, socket: QLocalSocket) -> None:
        if SHOW.strip() in bytes(socket.readAll().data()):
            socket.write(SHOWN)
            socket.flush()
            self.show_requested.emit()

    def close(self) -> None:
        for socket in self._sockets:
            socket.abort()
        self._sockets.clear()
        self._server.close()
