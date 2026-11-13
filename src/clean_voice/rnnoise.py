"""ctypes binding to the bundled RNNoise library (BSD-3-Clause, Xiph.Org; see native/RNNOISE-COPYING.txt).

The libraries in ``native/`` are built by ``tools/rnnoise/build_rnnoise.py`` from the RNNoise 0.2 release.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

FRAME_SIZE = 480  # 10 ms at 48 kHz, RNNoise's fixed frame
NATIVE_DIR = Path(__file__).resolve().parent / "native"
# RNNoise works on 16-bit-scaled floats.
_SCALE = 32768.0


def library_path() -> Path:
    if sys.platform == "win32":
        return NATIVE_DIR / "rnnoise.dll"
    if sys.platform == "darwin":
        return NATIVE_DIR / "librnnoise.dylib"
    raise OSError(f"Clean Voice runs on Windows and macOS only, not {sys.platform}")


_lib: ctypes.CDLL | None = None


def load() -> ctypes.CDLL:
    """Load the library once and declare the functions Clean Voice uses."""
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(library_path()))
        lib.rnnoise_get_frame_size.restype = ctypes.c_int
        lib.rnnoise_get_frame_size.argtypes = []
        lib.rnnoise_create.restype = ctypes.c_void_p
        lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        lib.rnnoise_destroy.restype = None
        lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        lib.rnnoise_process_frame.restype = ctypes.c_float
        lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        if lib.rnnoise_get_frame_size() != FRAME_SIZE:
            raise OSError("unexpected RNNoise frame size")
        _lib = lib
    return _lib


class RNNoise:
    """One RNNoise denoiser state. ``process`` takes exactly one 480-sample frame in -1..1."""

    def __init__(self) -> None:
        self._lib = load()
        self._state = self._lib.rnnoise_create(None)
        if not self._state:
            raise MemoryError("rnnoise_create failed")
        self._in = np.zeros(FRAME_SIZE, np.float32)
        self._out = np.zeros(FRAME_SIZE, np.float32)
        self._in_ptr = self._in.ctypes.data
        self._out_ptr = self._out.ctypes.data

    def process(self, frame: npt.NDArray[np.float32], out: npt.NDArray[np.float32]) -> float:
        """Denoise ``frame`` into ``out`` (both 480 samples) and return the voice probability (0..1)."""
        np.multiply(frame, _SCALE, out=self._in)
        vad = float(self._lib.rnnoise_process_frame(self._state, self._out_ptr, self._in_ptr))
        np.multiply(self._out, 1.0 / _SCALE, out=out)
        return vad

    def close(self) -> None:
        if self._state:
            self._lib.rnnoise_destroy(self._state)
            self._state = None

    def __del__(self) -> None:
        self.close()
