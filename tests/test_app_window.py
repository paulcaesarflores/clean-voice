"""The Clean Voice window opens, and the launch check catches a build that is missing a piece. The screen's states
are in test_main_screen.py."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from clean_voice.app import MainWindow, audio_modules, main
from clean_voice.engine import State, Status
from clean_voice.preview import PreviewEngine

# tests/conftest.py sets QT_QPA_PLATFORM=offscreen for the in-process qapp fixture.

STOPPED = Status(state=State.OFF)


def test_main_window_shows(qtbot: QtBot, tmp_path: Path) -> None:
    window = MainWindow(PreviewEngine(STOPPED), tmp_path / "settings.json")
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    assert window.windowTitle() == "Clean Voice"
    assert window.isVisible()
    assert (window.width(), window.height()) == (760, 684)  # the one window (J10 open question 9; round 13: 684)
    assert (window.minimumWidth(), window.minimumHeight()) == (680, 520)


def test_module_entry_point() -> None:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-m", "clean_voice", "--quit-after-ms", "300"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Clean Voice window shown" in result.stdout


def test_quit_after_ms_fails_when_window_not_visible(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # The exit code is the launch check's only signal on macOS, where the app's stdout goes to a redacted system log.
    monkeypatch.setattr(MainWindow, "isVisible", lambda self: False)
    assert main(["--quit-after-ms", "2000"], engine=PreviewEngine(STOPPED), prefs_path=tmp_path / "s.json") == 2
    out, err = capsys.readouterr()
    assert "Clean Voice window shown" not in out
    assert "Clean Voice window NOT shown" in err


def test_quit_after_ms_fails_when_rnnoise_does_not_load(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from clean_voice import rnnoise

    def broken() -> None:
        raise OSError("missing library")

    monkeypatch.setattr(rnnoise, "load", broken)
    assert main(["--quit-after-ms", "2000"], engine=PreviewEngine(STOPPED), prefs_path=tmp_path / "s.json") == 3
    assert "RNNoise failed to load" in capsys.readouterr().err


def test_the_windows_launch_check_covers_echo_removal() -> None:
    # L1 (J06): echo removal needs AEC3 (pywebrtc-audio) and WASAPI loopback (PyAudioWPatch), Windows only.
    from clean_voice.app import audio_modules

    assert {"pywebrtc_audio", "pyaudiowpatch"} <= set(audio_modules("win32"))
    assert not {"pywebrtc_audio", "pyaudiowpatch"} & set(audio_modules("darwin"))
    assert {"numpy", "scipy.signal", "samplerate", "sounddevice"} <= set(audio_modules("darwin"))


# The libraries this platform's build holds and checks (the echo ones on Windows only; the test above pins both lists).
@pytest.mark.parametrize("module", audio_modules())
def test_quit_after_ms_fails_when_an_audio_library_is_missing(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    module: str,
    tmp_path: Path,
) -> None:
    # A Briefcase build whose requirements were never refreshed held only PySide6 (J05 Step 5). sounddevice is only
    # imported on the engine's thread, where a failure becomes an on-screen error rather than an exit code.
    monkeypatch.setitem(sys.modules, module, None)  # makes `import module` raise ImportError
    assert main(["--quit-after-ms", "2000"], engine=PreviewEngine(STOPPED), prefs_path=tmp_path / "s.json") == 4
    assert f"Audio library missing: {module}" in capsys.readouterr().err


def test_both_engines_fit_the_window_protocol() -> None:
    # M4 (J05 review r1), E7 (J07): pyright checks these assignments, so a method missing from either engine (or from
    # EngineLike) fails the gate's type check.
    from clean_voice.app import EngineLike
    from clean_voice.engine import Engine

    def as_window_engine(engine: Engine) -> EngineLike:
        return engine

    def as_preview_engine(engine: PreviewEngine) -> EngineLike:
        return engine

    def choose(engine: EngineLike) -> None:  # what the screen calls, through the protocol
        engine.set_recording(False)
        engine.select_output(engine.outputs()[0])

    preview = PreviewEngine(STOPPED, outputs=["Out Y"])
    choose(as_preview_engine(preview))
    assert not preview.status().recording_enabled and preview.status().output == "Out Y"
    assert callable(as_window_engine)
