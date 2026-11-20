"""Crash-safe 24-bit WAV backups."""

import datetime as dt
import struct
import subprocess
import sys
import textwrap
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from clean_voice import wav
from clean_voice.wav import MAX_DATA_BYTES, Recorder, WavWriter, encode_24bit


def read_wav(path: Path) -> tuple[wave._wave_params, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        params = w.getparams()
        raw = w.readframes(params.nframes)
    b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
    ints = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)) << 8 >> 8
    return params, ints.astype(np.float64) / 8_388_607


def test_24bit_encoding_round_trips(tmp_path: Path) -> None:
    x = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, 1e-6], np.float32)
    path = tmp_path / "a.wav"
    w = WavWriter(path)
    w.write(x)
    w.close()
    params, data = read_wav(path)
    assert (params.nchannels, params.sampwidth, params.framerate) == (1, 3, 48_000)
    assert np.allclose(data, np.clip(x, -1, 1), atol=1.5 / 8_388_607)
    assert len(encode_24bit(x)) == 3 * len(x)


def test_header_is_valid_after_every_flush(tmp_path: Path) -> None:
    path = tmp_path / "b.wav"
    w = WavWriter(path, flush_every_s=0.1)
    for i in range(5):
        w.write(np.full(4800, 0.25, np.float32))
        # Read the file while it is still open, as a crash would leave it.
        params, data = read_wav(path)
        assert params.nframes == 4800 * (i + 1)
        assert np.allclose(data, 0.25, atol=1e-6)
    w.close()


def test_odd_length_file_is_padded_and_readable(tmp_path: Path) -> None:
    path = tmp_path / "c.wav"
    w = WavWriter(path)
    w.write(np.zeros(3, np.float32))
    w.close()
    assert path.stat().st_size % 2 == 0
    params, _ = read_wav(path)
    assert params.nframes == 3


def test_recorder_writes_clean_and_raw(tmp_path: Path) -> None:
    rec = Recorder(tmp_path / "Recordings")
    rec.write(np.full(480, 0.1, np.float32), np.full(480, 0.2, np.float32))
    rec.close()
    _, clean = read_wav(rec.clean.path)
    _, raw = read_wav(rec.raw.path)
    assert rec.clean.path.name.endswith("clean voice.wav")
    assert rec.raw.path.name.endswith("raw mic.wav")
    assert np.allclose(clean, 0.1, atol=1e-6) and np.allclose(raw, 0.2, atol=1e-6)


def test_a_second_pair_in_the_same_second_keeps_the_first(tmp_path: Path) -> None:
    # The recording switch can close a pair and open the next within one second (J05 Step 3).
    now = dt.datetime(2026, 11, 10, 14, 3, 12)
    first = Recorder(tmp_path, now)
    first.write(np.full(480, 0.1, np.float32), np.full(480, 0.2, np.float32))
    first.close()
    second = Recorder(tmp_path, now)
    second.write(np.full(960, 0.3, np.float32), np.full(960, 0.4, np.float32))
    second.close()
    assert second.clean.path.name == "2026-11-10 14-03-12 (2) clean voice.wav"
    assert second.raw.path.name == "2026-11-10 14-03-12 (2) raw mic.wav"
    _, clean = read_wav(first.clean.path)
    assert len(clean) == 480 and np.allclose(clean, 0.1, atol=1e-6)  # the first pair is untouched


def test_the_size_limit_fits_a_riff_header() -> None:
    # F1 (J05 review r1): a RIFF header holds 32-bit sizes; at 144 000 B/s a file passes 4 GiB after 8 h 17 min, and
    # the header write then raised struct.error into the audio step.
    assert 36 + MAX_DATA_BYTES + 1 <= 2**32 - 1  # sizes plus the pad byte
    assert MAX_DATA_BYTES / 144_000 / 3600 > 8.2  # still over 8 hours per pair


def test_a_pair_rolls_over_before_the_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The limit cut to 1.5 s per file, so the test writes 3 s instead of 8 hours (a seek to 4 GiB fills a real 4 GB
    # file on Windows). Every pair is complete and readable; nothing is lost between them.
    monkeypatch.setattr(wav, "MAX_DATA_BYTES", 216_000)  # 1.5 s of 48 kHz 24-bit
    rec = Recorder(tmp_path)
    for _ in range(300):  # 3 s of 10 ms blocks
        rec.write(np.full(480, 0.25, np.float32), np.full(480, 0.5, np.float32))
    rec.close()  # the writer thread has written everything queued once close returns
    paths = list(zip(sorted(tmp_path.glob("* clean voice.wav")), sorted(tmp_path.glob("* raw mic.wav")), strict=True))
    assert rec.error is None
    assert rec.pairs == len(paths) == 2 and rec.frames_total == 300 * 480
    total = 0
    for clean_path, raw_path in paths:
        for path in (clean_path, raw_path):
            data = path.read_bytes()
            riff, size = struct.unpack("<I", data[4:8])[0], struct.unpack("<I", data[40:44])[0]
            assert size <= 216_000 and riff == 36 + size  # the header never passed the limit
        _, clean = read_wav(clean_path)
        assert np.allclose(clean, 0.25, atol=1e-6)
        total += len(clean)
    assert total == 300 * 480


def test_any_write_failure_ends_the_backup_and_never_raises(tmp_path: Path) -> None:
    # F1: Recorder.write caught OSError only; anything else left the audio step and took the worker down.
    rec = Recorder(tmp_path)

    def broken() -> None:
        raise struct.error("'I' format requires 0 <= number <= 4294967295")

    rec.clean.flush = broken  # type: ignore[method-assign]
    for _ in range(200):
        rec.write(np.zeros(480, np.float32), np.zeros(480, np.float32))  # must not raise
    rec.close()  # the writer thread has run once close returns
    assert not rec.active and rec.error and "format requires" in rec.error
    rec.close()  # closing a failed recorder is safe


def test_write_silence_adds_zeros_to_both_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # B002 item 7: a mic gap is written as silence (libobs/media-io/audio-io.c input_and_output clears the mix each
    # tick; obs-audio.c audio_callback mixes only sources that have audio). Across the roll-over limit too.
    monkeypatch.setattr(wav, "MAX_DATA_BYTES", 216_000)  # 1.5 s per file
    rec = Recorder(tmp_path)
    rec.write(np.full(480, 0.25, np.float32), np.full(480, 0.5, np.float32))
    rec.write_silence(48_000 * 2 + 5)
    rec.write(np.full(480, 0.25, np.float32), np.full(480, 0.5, np.float32))
    rec.close()
    total = 480 * 2 + 48_000 * 2 + 5
    assert rec.error is None and rec.pairs == 2 and rec.frames_total == total
    first_first = lambda p: "(" in p.name
    clean = np.concatenate([read_wav(p)[1] for p in sorted(tmp_path.glob("* clean voice.wav"), key=first_first)])
    raw = np.concatenate([read_wav(p)[1] for p in sorted(tmp_path.glob("* raw mic.wav"), key=first_first)])
    for data, level in ((clean, 0.25), (raw, 0.5)):
        assert len(data) == total
        assert np.allclose(data[:480], level, atol=1e-6) and np.allclose(data[-480:], level, atol=1e-6)
        assert not data[480:-480].any()


def test_a_slow_disk_write_does_not_block_the_caller(tmp_path: Path) -> None:
    # B002 item 12: the data is queued and a write thread writes it (plugins/obs-ffmpeg/obs-ffmpeg-output.c
    # write_thread, :892-919), so a slow disk never holds the caller.
    rec = Recorder(tmp_path)
    real = rec.clean.write

    disk_free = threading.Event()

    def slow(samples: np.ndarray) -> None:
        disk_free.wait(10)  # the disk stays stuck until the test lets go, however slow the runner is
        real(samples)

    rec.clean.write = slow  # type: ignore[method-assign]
    block = np.full(480, 0.25, np.float32)
    start = time.perf_counter()
    for _ in range(3):
        rec.write(block, block)
    returned_s = time.perf_counter() - start
    disk_free.set()
    assert returned_s < 5.0, returned_s  # a caller held by the stuck disk would have waited the full 10 s
    rec.close()
    assert rec.error is None
    assert len(read_wav(next(tmp_path.glob("* clean voice.wav")))[1]) == 3 * 480
    assert len(read_wav(next(tmp_path.glob("* raw mic.wav")))[1]) == 3 * 480


def test_a_write_failure_in_the_writer_thread_ends_the_backup_quietly(tmp_path: Path) -> None:
    rec = Recorder(tmp_path)

    def broken(samples: np.ndarray) -> None:
        raise OSError("disk full")

    rec.clean.write = broken  # type: ignore[method-assign]
    block = np.zeros(480, np.float32)
    for _ in range(5):
        rec.write(block, block)  # never raises
    rec.close()  # returns
    assert rec.error == "disk full" and not rec.active
    rec.write(block, block)  # after the error: dropped at once
    rec.write_silence(100)


CHILD = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    import numpy as np
    from clean_voice.wav import Recorder

    rec = Recorder(Path(sys.argv[1]))
    n = 0
    t = np.arange(480)
    printed = 0
    while True:
        tone = (0.5 * np.sin(2 * np.pi * 440 * (t + n) / 48000)).astype(np.float32)
        rec.write(tone, tone * 0.5)
        n += 480
        # The files are written on the recorder's own thread, so print only the seconds that reached the disk.
        # frames_flushed moves before the header is rewritten, so a flush is reported only once the next one began.
        flushed = min(rec.clean.frames_flushed, rec.raw.frames_flushed) // 48000 * 48000
        if flushed > printed:
            if printed:
                print(printed, flush=True)
            printed = flushed
        time.sleep(0.0005)
    """
)


def test_killed_process_leaves_readable_files(tmp_path: Path) -> None:
    folder = tmp_path / "Recordings"
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(folder)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout is not None
        seconds = 0
        deadline = time.monotonic() + 60
        while seconds < 3 and time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            seconds = int(line) // 48000
        assert seconds >= 3, proc.stderr.read() if proc.stderr else ""
    finally:
        proc.kill()  # mid-recording, no chance to close the files
        proc.wait(timeout=30)

    clean = next(folder.glob("* clean voice.wav"))
    raw = next(folder.glob("* raw mic.wav"))
    for path, amp in ((clean, 0.5), (raw, 0.25)):
        params, data = read_wav(path)
        assert params.nframes >= 3 * 48000
        assert params.nframes % 48000 == 0  # exactly the flushed seconds
        expected = amp * np.sin(2 * np.pi * 440 * np.arange(params.nframes) / 48000)
        assert np.allclose(data, expected, atol=2e-6)
