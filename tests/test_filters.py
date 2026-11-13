"""``filters.sosfilt``, the style EQ's per-sample loop: the same numbers as ``scipy.signal.sosfilt`` (moved here from
the deleted test_user_eq.py in J10 Step 9; it fails first if a scipy upgrade changes the private loop)."""

import numpy as np
from scipy import signal

from clean_voice.filters import sosfilt


def test_the_fast_filter_is_scipys_sosfilt() -> None:
    rng = np.random.default_rng(4)
    s = np.array([[0.2, 0.4, 0.2, 1.0, -0.5, 0.3], [1.0, -1.8, 0.9, 1.0, -1.7, 0.75]])  # two stable sections
    x = rng.standard_normal(333).astype(np.float32)
    zi = rng.standard_normal((len(s), 2))
    kept, x_kept = zi.copy(), x.copy()
    want, want_zi = signal.sosfilt(s, x, zi=zi)
    got, got_zi = sosfilt(s, x, zi)
    assert np.array_equal(got, want) and np.array_equal(got_zi, want_zi)
    assert np.array_equal(zi, kept) and np.array_equal(x, x_kept)  # the inputs are left as they were
