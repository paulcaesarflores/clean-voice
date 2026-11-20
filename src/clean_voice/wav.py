"""Crash-safe 24-bit WAV backup recording.

Each flush appends the audio, then rewrites the RIFF and data sizes in the header, so a killed process leaves a file
that every player reads up to the last flush.
"""

from __future__ import annotations

import datetime as dt
import logging
import queue
import struct
import threading
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)

SAMPLE_RATE = 48_000
BITS = 24
BYTES_PER_SAMPLE = BITS // 8
HEADER_SIZE = 44
_FULL_SCALE = 8_388_607  # 2**23 - 1
# The largest data chunk a file may hold: under the RIFF 32-bit limit (2**32 - 1 - 36 header bytes, and a pad
# byte), rounded down to whole flushes. About 8 h 16 min of 48 kHz 24-bit mono.
MAX_DATA_BYTES = (2**32 - 1 - 36 - 1) // 144_000 * 144_000


def _header(data_size: int, sample_rate: int, channels: int) -> bytes:
    block_align = channels * BYTES_PER_SAMPLE
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, sample_rate * block_align, block_align, BITS)
        + b"data"
        + struct.pack("<I", data_size)
    )


def encode_24bit(samples: npt.NDArray[np.float32]) -> bytes:
    """Little-endian 24-bit PCM bytes for float samples in -1..1 (clipped)."""
    ints = np.round(np.clip(samples, -1.0, 1.0) * _FULL_SCALE).astype("<i4")
    return ints.view(np.uint8).reshape(-1, 4)[:, :BYTES_PER_SAMPLE].tobytes()


class WavWriter:
    """A mono 48 kHz 24-bit WAV file that is readable after every flush."""

    def __init__(self, path: Path, sample_rate: int = SAMPLE_RATE, flush_every_s: float = 1.0) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self._flush_bytes = max(1, int(flush_every_s * sample_rate)) * BYTES_PER_SAMPLE
        self._file: BinaryIO | None = path.open("w+b")
        self._file.write(_header(0, sample_rate, 1))
        self._file.flush()
        self._pending = bytearray()
        self._data_size = 0
        self.frames_written = 0

    @property
    def frames_flushed(self) -> int:
        return self._data_size // BYTES_PER_SAMPLE

    def has_room(self, frames: int) -> bool:
        """Whether ``frames`` more still fit under the RIFF size limit."""
        return self._data_size + len(self._pending) + frames * BYTES_PER_SAMPLE <= MAX_DATA_BYTES

    def write(self, samples: npt.NDArray[np.float32]) -> None:
        """Queue samples; they reach the file at the next flush, which happens about once a second."""
        if self._file is None:
            raise ValueError("WAV file is closed")
        self._pending += encode_24bit(samples)
        self.frames_written += len(samples)
        if len(self._pending) >= self._flush_bytes:
            self.flush()

    def flush(self) -> None:
        f = self._file
        if f is None or not self._pending:
            return
        f.seek(HEADER_SIZE + self._data_size)
        f.write(self._pending)
        self._data_size += len(self._pending)
        self._pending.clear()
        # Audio first, sizes second: a crash in between leaves the previous, still valid, header.
        f.seek(4)
        f.write(struct.pack("<I", 36 + self._data_size))
        f.seek(40)
        f.write(struct.pack("<I", self._data_size))
        f.flush()

    def close(self) -> None:
        f = self._file
        if f is None:
            return
        self.flush()
        if self._data_size % 2:
            # RIFF chunks are padded to an even size; the pad byte is not part of the data.
            f.seek(HEADER_SIZE + self._data_size)
            f.write(b"\0")
            f.seek(4)
            f.write(struct.pack("<I", 36 + self._data_size + 1))
        f.close()
        self._file = None


class Recorder:
    """The backup pair: the clean voice and the raw mic, side by side in one folder.

    A RIFF header holds 32-bit sizes, so one file stops at 4 GiB, about 8 h 17 min of 48 kHz 24-bit mono. Before a
    pair gets there it is closed and the recording goes on in a new pair. The files are written on a thread of their
    own, so a slow disk never holds the caller (plugins/obs-ffmpeg/obs-ffmpeg-output.c: the producer queues and posts,
    ``write_thread`` drains and writes; the queue is unbounded, as there). A failure to write never leaves ``write``:
    it ends the recording (``error``) and the voice goes on.
    """

    def __init__(self, folder: Path, now: dt.datetime | None = None) -> None:
        self.folder = folder
        self.error: str | None = None
        self.pairs = 0  # pairs opened so far, the current one included
        self.frames_total = 0  # frames written to all pairs
        self.files: tuple[Path, Path]  # the current pair's paths, replaced whole on roll-over
        self.clean, self.raw = self._open_pair(now)
        self._queue: queue.SimpleQueue[tuple[str, object] | None] = queue.SimpleQueue()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="backup-writer", daemon=True)
        self._thread.start()

    def _open_pair(self, now: dt.datetime | None = None) -> tuple[WavWriter, WavWriter]:
        folder = self.folder
        folder.mkdir(parents=True, exist_ok=True)
        stamp = (now or dt.datetime.now()).strftime("%Y-%m-%d %H-%M-%S")
        # The recording switch (or a rollover) can start a new pair within the same second: never overwrite one.
        base, n = stamp, 2
        while (folder / f"{stamp} clean voice.wav").exists() or (folder / f"{stamp} raw mic.wav").exists():
            stamp, n = f"{base} ({n})", n + 1
        clean = WavWriter(folder / f"{stamp} clean voice.wav")
        try:
            raw = WavWriter(folder / f"{stamp} raw mic.wav")
        except OSError:
            clean.close()
            raise
        self.pairs += 1
        self.files = (clean.path, raw.path)
        return clean, raw

    @property
    def active(self) -> bool:
        return self.error is None

    def write(self, clean: npt.NDArray[np.float32], raw: npt.NDArray[np.float32]) -> None:
        """Queue a block for the writer thread (da_push_back + os_sem_post, obs-ffmpeg-output.c:697-701)."""
        if self.error is None and not self._closed:
            self._queue.put(("audio", (clean.copy(), raw.copy())))

    def write_silence(self, frames: int) -> None:
        """Queue ``frames`` of silence for both files: the time a lost mic was away (OBS's mix is cleared every tick,
        libobs/media-io/audio-io.c input_and_output, and only sources with audio are mixed, libobs/obs-audio.c)."""
        if self.error is None and not self._closed and frames > 0:
            self._queue.put(("silence", frames))

    def _run(self) -> None:
        # Drains the queue and writes (obs-ffmpeg-output.c write_thread :892-919); a write error ends this output only.
        while (item := self._queue.get()) is not None:
            if self.error is not None:
                continue
            kind, data = item
            try:
                if kind == "audio":
                    clean, raw = data  # type: ignore[misc]
                    self._write_block(clean, raw)
                else:
                    left = int(data)  # type: ignore[call-overload]
                    zeros = np.zeros(SAMPLE_RATE, np.float32)  # in 1 s pieces: no giant array for a long gap
                    while left > 0:
                        piece = zeros[: min(left, SAMPLE_RATE)]
                        self._write_block(piece, piece)
                        left -= len(piece)
            except Exception as exc:  # noqa: BLE001 - disk full, folder gone, anything: keep the voice, stop recording
                self.error = str(exc) or type(exc).__name__
                self._close_quietly()

    def _write_block(self, clean: npt.NDArray[np.float32], raw: npt.NDArray[np.float32]) -> None:
        if not self.clean.has_room(len(clean)) or not self.raw.has_room(len(raw)):
            self._roll_over()
        self.clean.write(clean)
        self.raw.write(raw)
        self.frames_total += len(clean)

    def _roll_over(self) -> None:
        for writer in (self.clean, self.raw):
            writer.close()
        self.clean, self.raw = self._open_pair()

    def _close_quietly(self) -> None:
        for writer in (self.clean, self.raw):
            try:
                writer.close()
            except Exception:
                log.debug("closing a failed backup file failed too", exc_info=True)

    def close(self) -> None:
        """Write what is queued, stop the thread and close both files (obs-ffmpeg-output.c:1122-1126: signal, join)."""
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join()
        for writer in (self.clean, self.raw):
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001 - report it; the other file is still closed
                self.error = self.error or str(exc)
