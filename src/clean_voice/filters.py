"""The EQs' per-sample loop: scipy's compiled biquad cascade, called without ``scipy.signal.sosfilt``'s checks.

While an EQ glides it filters 1 ms sub-blocks, two cascades each, so the public wrapper (shape checks, axis moves and
copies, about 75 us a call) cost more than the filtering itself: the bench's worst case (both EQs retuned every block)
went over the 10 ms block. ``_sosfilt`` is the loop the wrapper runs, so the numbers are the same (test_filters checks
it bit for bit against ``scipy.signal.sosfilt``). It is private to scipy, which is pinned in pyproject.toml; that test
fails first if a scipy upgrade changes it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal._sosfilt import _sosfilt  # pyright: ignore[reportMissingModuleSource]


def sosfilt(sos: Any, x: npt.ArrayLike, zi: Any) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``scipy.signal.sosfilt(sos, x, zi=zi)`` for one 1-D signal: (output, final state), both float64."""
    y = np.array(x, dtype=np.float64, ndmin=2)  # a C-ordered copy, filtered in place
    z = np.array(zi, dtype=np.float64)[np.newaxis]
    _sosfilt(np.ascontiguousarray(sos, dtype=np.float64), y, z)
    return y[0], z[0]
