"""A preallocated single-producer, single-consumer ring of float32 audio frames.

The producer only advances ``_written`` and the consumer only advances ``_read``, each after its copy is done, so
the two sides never take a lock. Device callbacks use it to copy and nothing else.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class Ring:
    def __init__(self, capacity: int, channels: int = 1) -> None:
        if capacity <= 0 or channels <= 0:
            raise ValueError("capacity and channels must be positive")
        self.capacity = capacity
        self.channels = channels
        self._buf = np.zeros((capacity, channels), np.float32)
        self._written = 0  # frames ever written; only the producer changes it
        self._read = 0  # frames ever read; only the consumer changes it

    @property
    def total_written(self) -> int:
        return self._written

    def available(self) -> int:
        """Frames waiting to be read."""
        return self._written - self._read

    def space(self) -> int:
        """Frames that can be written without overwriting unread audio."""
        return self.capacity - (self._written - self._read)

    def write(self, data: npt.NDArray[np.float32]) -> int:
        """Copy as many frames of ``data`` (shape (n, channels), or (n,) for mono) as fit; return how many."""
        frames = data.reshape(-1, self.channels) if data.ndim == 1 else data
        n = min(len(frames), self.space())
        if n <= 0:
            return 0
        start = self._written % self.capacity
        first = min(n, self.capacity - start)
        self._buf[start : start + first] = frames[:first]
        if n > first:
            self._buf[: n - first] = frames[first:n]
        self._written += n  # publish only after the copy
        return n

    def read_into(self, out: npt.NDArray[np.float32]) -> int:
        """Copy up to ``len(out)`` frames into ``out`` (shape (n, channels), or (n,) for mono); return how many."""
        frames = out.reshape(-1, self.channels) if out.ndim == 1 else out
        n = min(len(frames), self.available())
        if n <= 0:
            return 0
        start = self._read % self.capacity
        first = min(n, self.capacity - start)
        frames[:first] = self._buf[start : start + first]
        if n > first:
            frames[first:n] = self._buf[: n - first]
        self._read += n  # free the space only after the copy
        return n
