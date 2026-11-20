"""The 10-band parametric EQ's data and maths (B012). OBS has only a 3-band EQ (plugins/obs-filters/eq-filter.c), so
this filter is Clean Voice's own addition, approved by the user on 2026-11-20.

Each band is one biquad from the public RBJ "Audio EQ Cookbook" (bell, low and high shelf, low and high cut, notch),
written here from its formulas. ``fine_tune`` builds the filter's properties from the constants below, ``obs_filters``
runs the biquads, and the filter window's graph draws ``response_db``. Nothing here needs Qt.

A band's settings are flat keys ``b<N>_on``, ``b<N>_type``, ``b<N>_freq``, ``b<N>_gain``, ``b<N>_q`` (N = 1..10) and
the filter's ``trim``, in dB.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 48_000
BANDS = 10
FREQS = (32.0, 64.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
FREQ_RANGE = (20.0, 20000.0)
GAIN_RANGE = (-18.0, 18.0)
Q_RANGE = (0.1, 18.0)
TRIM_RANGE = (-18.0, 18.0)
DEFAULT_Q = 0.7
# (value, label) in the order the type buttons show them.
TYPES: tuple[tuple[str, str], ...] = (
    ("bell", "Bell"),
    ("low_shelf", "Low shelf"),
    ("high_shelf", "High shelf"),
    ("low_cut", "Low cut"),
    ("high_cut", "High cut"),
    ("notch", "Notch"),
)
GAINED = ("bell", "low_shelf", "high_shelf")  # the types whose gain does something
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)  # a second-order section that passes audio through unchanged
# The graph's gridlines (the labels the user asked for; 31 and 62 are the usual names for 32 and 64 Hz).
GRID_FREQS = (31.0, 62.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)


def default_type(band: int) -> str:
    """Band 1 is a low cut, band 10 a high cut, the rest bells."""
    return "low_cut" if band == 1 else "high_cut" if band == BANDS else "bell"


def key(band: int, field: str) -> str:
    return f"b{band}_{field}"


@dataclass(frozen=True)
class Band:
    on: bool
    type: str
    freq: float
    gain: float
    q: float


def bands_of(settings: Mapping[str, object]) -> list[Band]:
    """The ten bands from a resolved settings dict."""
    return [
        Band(
            settings[key(i, "on")] is True,
            str(settings[key(i, "type")]),
            float(settings[key(i, "freq")]),  # type: ignore[arg-type]
            float(settings[key(i, "gain")]),  # type: ignore[arg-type]
            float(settings[key(i, "q")]),  # type: ignore[arg-type]
        )
        for i in range(1, BANDS + 1)
    ]


def section(band: Band) -> tuple[float, float, float, float, float, float]:
    """The band's biquad as a second-order section (b0, b1, b2, 1, a1/a0, a2/a0), from the RBJ cookbook. An off band
    is the identity, so every band can be crossfaded to and from "off" the same way."""
    if not band.on:
        return IDENTITY
    w = 2.0 * math.pi * min(band.freq, SAMPLE_RATE * 0.49) / SAMPLE_RATE
    cw, sw = math.cos(w), math.sin(w)
    alpha = sw / (2.0 * band.q)
    amp = 10.0 ** (band.gain / 40.0)
    kind = band.type
    if kind == "bell":
        b = (1.0 + alpha * amp, -2.0 * cw, 1.0 - alpha * amp)
        a = (1.0 + alpha / amp, -2.0 * cw, 1.0 - alpha / amp)
    elif kind in ("low_shelf", "high_shelf"):
        root = 2.0 * math.sqrt(amp) * alpha
        up, down = amp + 1.0, amp - 1.0
        if kind == "low_shelf":
            b = (amp * (up - down * cw + root), 2.0 * amp * (down - up * cw), amp * (up - down * cw - root))
            a = (up + down * cw + root, -2.0 * (down + up * cw), up + down * cw - root)
        else:
            b = (amp * (up + down * cw + root), -2.0 * amp * (down + up * cw), amp * (up + down * cw - root))
            a = (up - down * cw + root, 2.0 * (down - up * cw), up - down * cw - root)
    elif kind == "low_cut":  # a high-pass
        b = ((1.0 + cw) / 2.0, -(1.0 + cw), (1.0 + cw) / 2.0)
        a = (1.0 + alpha, -2.0 * cw, 1.0 - alpha)
    elif kind == "high_cut":  # a low-pass
        b = ((1.0 - cw) / 2.0, 1.0 - cw, (1.0 - cw) / 2.0)
        a = (1.0 + alpha, -2.0 * cw, 1.0 - alpha)
    else:  # notch
        b = (1.0, -2.0 * cw, 1.0)
        a = (1.0 + alpha, -2.0 * cw, 1.0 - alpha)
    return (b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0])


def section_db(sos: tuple[float, ...], freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """One section's gain in dB at ``freqs`` (Hz)."""
    z = np.exp(-2j * np.pi * freqs / SAMPLE_RATE)
    h = (sos[0] + sos[1] * z + sos[2] * z * z) / (sos[3] + sos[4] * z + sos[5] * z * z)
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))


def band_db(band: Band, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return section_db(section(band), freqs)


def response_db(settings: Mapping[str, object], freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The whole EQ's gain in dB at ``freqs``: every band on, plus the output trim."""
    total = np.full(len(freqs), float(settings["trim"]))  # type: ignore[arg-type]
    for band in bands_of(settings):
        if band.on:
            total += band_db(band, freqs)
    return total
