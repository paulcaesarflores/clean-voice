"""W1, W2: the backup recording's on/off switch, and readable backups after the engine process is killed."""

import struct
import subprocess
import sys
import textwrap
import time
import wave
from pathlib import Path

import pytest
from test_engine import CABLE, MIC, FakeBackend, wait_for

from clean_voice import engine as engine_module
from clean_voice import wav
from clean_voice.app import describe
from clean_voice.audio_path import AudioPath
from clean_voice.engine import Engine, State


def frames_in(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 48_000 and w.getsampwidth() == 3 and w.getnchannels() == 1
        return w.getnframes()


def test_recording_is_off_by_default(make_engine, tmp_path: Path) -> None:
    # The user's decision of 2026-11-11: the backup is off until the guest switches it on.
    engine, _ = make_engine([MIC, CABLE], record=False)
    assert not engine.status().recording_enabled
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    time.sleep(0.3)
    status = engine.status()
    assert not status.recording and not status.recording_enabled and status.recording_files == ()
    assert not (tmp_path / "Recordings").exists() or not any((tmp_path / "Recordings").iterdir())
    engine.set_recording(True)
    wait_for(lambda: engine.status().recording)
    status = engine.status()
    assert status.recording_enabled and len(status.recording_files) == 2


def test_switching_off_closes_the_pair_and_the_audio_keeps_going(make_engine) -> None:
    engine, backend = make_engine([MIC, CABLE])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    wait_for(lambda: engine.status().recording_seconds > 0.3)
    first = engine.status().recording_files
    engine.set_recording(False)
    wait_for(lambda: not engine.status().recording_enabled)
    status = engine.status()
    assert status.state is State.ON and not status.recording
    sizes = [frames_in(f) for f in first]  # final and readable while the app keeps running
    assert all(n > 0 for n in sizes)
    played = backend.opened[0].path.frames_played
    wait_for(lambda: backend.opened[0].path.frames_played > played + 4800)  # the voice still reaches the cable
    assert [frames_in(f) for f in first] == sizes  # nothing more is written to the closed pair
    assert len(backend.opened) == 1  # no reconnect

    engine.set_recording(True)
    wait_for(lambda: engine.status().recording and engine.status().recording_files != first)
    second = engine.status().recording_files
    assert all(f.exists() for f in first) and not set(first) & set(second)  # a new pair; the first is kept
    wait_for(lambda: engine.status().recording_seconds > 0.3)
    engine.stop()
    wait_for(lambda: engine.status().state is State.OFF)
    assert all(frames_in(f) > 0 for f in second)


def test_switched_off_before_starting_makes_no_files(make_engine) -> None:
    engine, _ = make_engine([MIC, CABLE])
    engine.set_recording(False)
    wait_for(lambda: not engine.status().recording_enabled)
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    time.sleep(0.3)
    assert not engine.status().recording
    assert not engine.recordings_dir.exists() or not any(engine.recordings_dir.iterdir())


# ------------------------------------------------------------------ the backup never stops the voice (J05 review r1)


def voice_flows(backend: FakeBackend, seconds: float = 0.5) -> bool:
    streams = backend.opened[-1]
    played = streams.path.frames_played
    time.sleep(seconds)
    return not streams.closed and streams.path.frames_played > played


def test_a_session_past_the_wav_size_limit_keeps_the_voice_and_the_backup(
    make_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F1: at 8 h 17 min the WAV passed the RIFF 4 GiB limit, the header write raised, and the worker thread died:
    # the voice went silent, the screen kept saying "on", and Stop did nothing. The limit is cut to 1.5 s here.
    monkeypatch.setattr(wav, "MAX_DATA_BYTES", 216_000)
    engine, backend = make_engine([MIC, CABLE])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    first = engine.status().recording_files
    wait_for(lambda: engine.status().recording_files != first, timeout=10)  # rolled over to a new pair
    status = engine.status()
    assert status.state is State.ON and status.recording and status.recording_error is None
    assert engine._thread.is_alive() and voice_flows(backend)
    assert len(backend.opened) == 1  # the streams stayed open throughout
    engine.stop()
    wait_for(lambda: engine.status().state is State.OFF)  # the worker still answers
    assert frames_in(first[0]) >= 48_000 and frames_in(status.recording_files[0]) > 0


def test_a_failing_backup_write_stops_only_the_backup(make_engine) -> None:
    # F1: any error in the backup (here the header write, as at the size limit) ends the recording, not the voice.
    engine, backend = make_engine([MIC, CABLE])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)

    def broken() -> None:
        raise struct.error("'I' format requires 0 <= number <= 4294967295")

    recorder = engine._recorder
    assert recorder is not None
    recorder.clean.flush = broken  # type: ignore[method-assign]
    wait_for(lambda: engine.status().recording_error is not None)
    status = engine.status()
    assert status.state is State.ON and not status.recording
    assert status.recording_error is not None and status.recording_error.startswith("Can't save backups to ")
    assert engine._thread.is_alive() and voice_flows(backend)
    engine.stop()
    wait_for(lambda: engine.status().state is State.OFF)


def test_a_failing_audio_step_never_ends_the_worker(make_engine, monkeypatch: pytest.MonkeyPatch) -> None:
    # F1, generally: an exception in the audio step, and in the clean-up after it (the last process() call when
    # closing), used to end the worker thread. It shows as an error, the audio is retried, and Stop still works.
    broken = [True]

    class FailingPath(AudioPath):
        def process(self) -> int:
            if broken[0]:
                raise RuntimeError("audio step failure")  # every call, the clean-up's last one included
            return super().process()

    monkeypatch.setattr(engine_module, "AudioPath", FailingPath)
    engine, backend = make_engine([MIC, CABLE])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ERROR)
    # A few retries, each failing again, and the screen says so throughout: each retry used to show "On" (error None)
    # from opening the streams until the failing step closed them again (CI run 43368578097 read it there).
    opened, seen = len(backend.opened), set()
    deadline = time.monotonic() + 10.0
    while len(backend.opened) < opened + 3 and time.monotonic() < deadline:
        status = engine.status()
        seen.add((status.state, status.error))
        time.sleep(0.001)
    assert len(backend.opened) >= opened + 3
    assert seen == {(State.ERROR, "audio step failure")}
    assert engine._thread.is_alive()
    broken[0] = False
    wait_for(lambda: engine.status().state is State.ON, timeout=15)  # retried, and running again
    assert engine._thread.is_alive() and voice_flows(backend)
    engine.stop()
    wait_for(lambda: engine.status().state is State.OFF)


def test_an_unwritable_backup_folder_does_not_block_the_voice(tmp_path: Path) -> None:
    # F2: the folder could not be created (here its parent is a file, as a protected or offline folder fails), so
    # opening the backup raised after the streams opened: the engine showed ERROR with the wrong advice ("close
    # other apps that use the microphone") and retried forever, with no voice at all.
    blocker = tmp_path / "Music"
    blocker.write_text("not a folder")
    folder = blocker / "Clean Voice"
    backend = FakeBackend([MIC, CABLE])
    engine = Engine(folder, backend, retry_s=0.2, watchdog_s=0.3)
    try:
        engine.set_recording(True)  # the backup is off by default (2026-11-11)
        engine.start(MIC.name)
        wait_for(lambda: engine.status().state is State.ON)
        time.sleep(0.5)
        status = engine.status()
        assert status.state is State.ON and len(backend.opened) == 1 and voice_flows(backend)
        assert not status.recording and status.recording_enabled
        assert status.recording_error == (
            f"Can't save backups to {folder}. Your voice still reaches the call. Check that the disk has space and "
            "that the folder can be written."
        )
        view = describe(status)
        assert view.notice is None and view.headline == "Your call hears your clean voice"  # no microphone advice
        engine.set_recording(False)  # switching it off clears the message; on again tries again, and says so
        wait_for(lambda: engine.status().recording_error is None)
        engine.set_recording(True)
        wait_for(lambda: engine.status().recording_error is not None)
        assert engine.status().state is State.ON and voice_flows(backend)
    finally:
        engine.close()


CHILD = textwrap.dedent(
    """
    import sys, threading, time
    from pathlib import Path
    import numpy as np
    from clean_voice.devices import Device
    from clean_voice.engine import Engine, State

    MIC = Device(1, "Microphone (USB Mic)", 1, 48_000.0)
    CABLE = Device(4, "CABLE Input (VB-Audio Virtual Cable)", 2, 48_000.0)

    class Streams:
        def __init__(self, path):
            self.path, self.closed = path, False
            threading.Thread(target=self.run, daemon=True).start()

        def run(self):
            n, out = 0, np.zeros((480, 2), np.float32)
            while not self.closed:
                tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(n, n + 480) / 48_000)
                self.path.input_callback(tone.astype(np.float32)[:, None], 480, None, None)
                self.path.output_callback(out, 480, None, None)
                n += 480
                time.sleep(0.01)

        @property
        def active(self):
            return not self.closed

        def close(self):
            self.closed = True

    class Backend:
        def rescan(self): pass
        def inputs(self): return [MIC]
        def outputs(self): return [CABLE]
        def default_input(self): return MIC
        def open(self, mic, cable, make_path):
            path = make_path(48_000, 1, 48_000, 2)
            return path, Streams(path)

    engine = Engine(Path(sys.argv[1]), Backend())
    engine.set_recording(True)  # the backup is off by default (2026-11-11)
    engine.start(None)
    while engine.status().recording_seconds < 3.0:
        time.sleep(0.05)
    print("recorded", flush=True)
    time.sleep(60)  # killed long before this
    """
)


def test_a_killed_engine_leaves_readable_backups(tmp_path: Path) -> None:
    # W2: the process dies mid-recording; both files play up to the last flush.
    folder = tmp_path / "Recordings"
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(folder)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "recorded"
    finally:
        child.kill()
        child.wait(timeout=10)
    files = sorted(folder.glob("*.wav"))
    assert [f.name.split(" ", 2)[2] for f in files] == ["clean voice.wav", "raw mic.wav"]
    for f in files:
        assert frames_in(f) >= 2 * 48_000
