"""Shared pytest setup for Clean Voice."""

import os
import sys
from pathlib import Path

import pytest

# Qt reads the platform when the QApplication is created; set it before any test creates one.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# The offscreen platform has no font list of its own: give it the system fonts, as tools/ui_snapshot.py does, so the
# tests see the glyphs the renders and the native app show (the Japanese device names, J07 U15).
for _fonts in (Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts", Path("/System/Library/Fonts")):
    if _fonts.is_dir():
        os.environ.setdefault("QT_QPA_FONTDIR", str(_fonts))
        break


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> object:
    """Delete each test's windows before the next test starts.

    pytest-qt closes a test's widgets and calls ``deleteLater`` on them, but only a running event loop carries that out
    (its ``processEvents`` does not). Few tests run one, so ~190 main windows piled up, and the next test to wait on a
    signal spent its first ~3 s (more on a busy machine) destroying them: long enough for test_tray's second launch to
    give up on an answer and its "show" to be lost. This wrapper runs after pytest-qt's teardown, which is ``trylast``.
    """
    result = yield
    if "PySide6.QtWidgets" in sys.modules:  # a test without Qt has nothing to delete
        from PySide6.QtCore import QCoreApplication, QEvent

        if QCoreApplication.instance() is not None:
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return result


@pytest.fixture
def make_engine(tmp_path: Path):
    """Engines on the fake sound system of test_engine.py, closed after the test. The backup is switched on unless
    ``record=False`` (the engine's default is off since 2026-11-11; most tests check the backup too)."""
    from test_engine import FakeBackend

    from clean_voice.devices import Device
    from clean_voice.engine import Engine

    engines: list[Engine] = []

    def make(present: list[Device], record: bool = True) -> tuple[Engine, FakeBackend]:
        backend = FakeBackend(present)
        engine = Engine(tmp_path / "Recordings", backend, retry_s=0.2, watchdog_s=0.3)
        if record:
            engine.set_recording(True)  # queued before any start, as the window sends the saved choice
        engines.append(engine)
        return engine, backend

    yield make
    for engine in engines:
        engine.close()


@pytest.fixture(autouse=True)
def no_real_mic_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """No window in a test reads or sets this computer's mic level (Home's mic slider): it gets none, unless the test
    passes its own fake. The real-device tests in tests/test_home_gain.py (Windows) and tests/test_core_audio.py
    (macOS) read the real one directly, never setting it."""
    from clean_voice import mic_level

    monkeypatch.setattr(mic_level, "system", mic_level.NoMicLevel)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches the internet (Check for updates fakes its own network)."""
    import urllib.request

    def refused(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(urllib.request, "urlopen", refused)
    # And below urllib, for openers built by hand (updates.open_url): no socket may connect anywhere.
    import socket

    monkeypatch.setattr(socket, "create_connection", refused)
