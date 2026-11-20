"""OBS parity of Fine-tune's filters (J10 plan, Tests T1-T9).

Each oracle here is a plain per-sample Python loop written from the algorithm in the plan's §4 (our own code), used to
check the vectorized or optimized class in ``clean_voice.obs_filters``. The expected numbers sit beside the OBS
file:line they come from (``reference/obs-studio/plugins/obs-filters/``).
"""

import math

import numpy as np
import pytest
from speech import looped_noise, mix

from clean_voice import fine_tune, obs_filters
from clean_voice.fine_tune import INT, KINDS, LIST, SLIDER

RATE = 48_000


def db(x: float) -> float:
    return 10.0 ** (x / 20.0)


def run(f, x: np.ndarray, size: int = 480) -> np.ndarray:
    x = x.astype(np.float32)
    return np.concatenate([f.process(x[i : i + size]) for i in range(0, len(x), size)]).astype(np.float64)


def tone(freq: float, seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seconds: float, seed: int, amp: float = 0.2) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(seconds * RATE)) * amp).astype(np.float32)


# ------------------------------------------------------------------ T1 defaults and ranges

# (key, widget, default, min, max, step, suffix), each with its OBS file:line.
EXPECTED = {
    # eq-filter.c: eq_defaults :54-56, eq_properties :65-70 (High, Mid, Low)
    "basic_eq_filter": [
        ("high", SLIDER, 0.0, -20.0, 20.0, 0.1, " dB"),
        ("mid", SLIDER, 0.0, -20.0, 20.0, 0.1, " dB"),
        ("low", SLIDER, 0.0, -20.0, 20.0, 0.1, " dB"),
    ],
    # compressor-filter.c: :44-52, compressor_defaults :440-444, compressor_properties :478-489
    "compressor_filter": [
        ("ratio", SLIDER, 10.0, 1.0, 32.0, 0.5, ":1"),
        ("threshold", SLIDER, -18.0, -60.0, 0.0, 0.1, " dB"),
        ("attack_time", SLIDER, 6, 1, 500, 1, " ms"),
        ("release_time", SLIDER, 60, 1, 1000, 1, " ms"),
        ("output_gain", SLIDER, 0.0, -32.0, 32.0, 0.1, " dB"),
    ],
    # expander-filter.c: :52-62, expander_defaults :159-165, expander_properties :438-462
    "expander_filter": [
        ("presets", LIST, "expander", 0, 0, 0, ""),
        ("ratio", SLIDER, 2.0, 1.0, 20.0, 0.1, ":1"),
        ("threshold", SLIDER, -40.0, -60.0, 0.0, 0.1, " dB"),
        ("attack_time", SLIDER, 10, 1, 100, 1, " ms"),
        ("release_time", SLIDER, 50, 1, 1000, 1, " ms"),
        ("output_gain", SLIDER, 0.0, -32.0, 32.0, 0.1, " dB"),
        ("detector", LIST, "RMS", 0, 0, 0, ""),
    ],
    # gain-filter.c: gain_defaults :70, gain_properties :77-78
    "gain_filter": [("db", SLIDER, 0.0, -30.0, 30.0, 0.1, " dB")],
    # invert-audio-polarity.c: no properties
    "invert_polarity_filter": [],
    # limiter-filter.c: :34-37, limiter_defaults :184-185, limiter_properties :193-197
    "limiter_filter": [
        ("threshold", SLIDER, -6.0, -60.0, 0.0, 0.1, " dB"),
        ("release_time", SLIDER, 60, 1, 1000, 1, " ms"),
    ],
    # noise-gate-filter.c: :47-48, noise_gate_defaults :158-162, noise_gate_properties :170-179
    "noise_gate_filter": [
        ("close_threshold", SLIDER, -32.0, -96.0, 0.0, 1.0, " dB"),
        ("open_threshold", SLIDER, -26.0, -96.0, 0.0, 1.0, " dB"),
        ("attack_time", INT, 25, 0, 10000, 1, " ms"),
        ("hold_time", INT, 200, 0, 10000, 1, " ms"),
        ("release_time", INT, 150, 0, 10000, 1, " ms"),
    ],
    # noise-suppress-filter.c: RNNoise without Speex has no properties (:503-526)
    "noise_suppress_filter": [],
    # expander-filter.c: :54-55, upward_compressor_defaults :170-176, expander_properties :445-465
    "upward_compressor_filter": [
        ("ratio", SLIDER, 0.5, 0.0, 1.0, 0.1, ":1"),
        ("threshold", SLIDER, -20.0, -60.0, 0.0, 0.1, " dB"),
        ("attack_time", SLIDER, 10, 1, 100, 1, " ms"),
        ("release_time", SLIDER, 50, 1, 1000, 1, " ms"),
        ("output_gain", SLIDER, 0.0, -32.0, 32.0, 0.1, " dB"),
        # No suffix on screen: :465 sets a float suffix on this int slider, which libobs ignores
        # (obs-properties.c obs_property_float_set_suffix :1040-1048 finds no float data).
        ("knee_width", SLIDER, 10, 0, 20, 1, ""),
    ],
}

# The display names (en-US.ini :16-19, :101, :108, :111, :129-130; obs-vst en-US.ini :3), in the + menu's order.
MENU = [
    "3-Band Equalizer",
    "Compressor",
    "Expander",
    "Gain",
    "Invert Polarity",
    "Limiter",
    "NVIDIA Audio Effects",  # J12: nv-filters en-US.ini :1; offered only where NVIDIA's runtime is installed
    "Noise Gate",
    "Noise Suppression",
    "Upward Compressor",
    "VST 2.x Plug-in",
]


@pytest.mark.parametrize("kind", sorted(EXPECTED))
def test_t1_every_kind_has_obs_defaults_and_ranges(kind: str) -> None:
    got = [
        (p.key, p.widget, p.default, p.lo, p.hi, p.step, p.suffix)
        for p in KINDS[kind].props
        if p.widget in (SLIDER, INT, LIST)
    ]
    assert got == EXPECTED[kind]
    for p in KINDS[kind].props:  # an int property is OBS's obs_data_get_int, so its default is an int
        if p.is_int:
            assert type(p.default) is int, p.key
    assert fine_tune.defaults(kind) == {k: d for k, _, d, *_ in EXPECTED[kind]}


def test_t1_names_menu_order_and_vst_keys() -> None:
    assert [KINDS[k].name for k in fine_tune.MENU_ORDER] == [
        "Denoise",
        "Voice Style",
        "10-Band EQ",
        *MENU,
    ]  # J13, B012: Clean Voice's own
    vst = KINDS["vst_filter"]
    assert [p.key for p in vst.props] == [
        "plugin_path",
        "open_vst_settings",
        "close_vst_settings",
        "open_when_active_vst_settings",
        "chunk_data",
        "chunk_hash",
    ]
    assert fine_tune.defaults("vst_filter") == {
        "plugin_path": "",
        "open_when_active_vst_settings": False,
        "chunk_data": "",
        "chunk_hash": "",
    }
    assert KINDS["expander_filter"].prop("detector").options == (("RMS", "RMS"), ("peak", "Peak"))
    assert KINDS["noise_suppress_filter"].note == "RNNoise (good quality, more CPU usage)"
    for k in KINDS.values():
        assert k.cite.startswith("Clean Voice's own" if k.id in fine_tune.OWN_KINDS else "plugins/"), k.id


def test_t1_out_of_range_values_clamp() -> None:
    r = fine_tune.resolve("compressor_filter", {"ratio": 99, "threshold": -100.0, "attack_time": 0.4, "x": 1})
    assert r == {"ratio": 32.0, "threshold": -60.0, "attack_time": 1, "release_time": 60, "output_gain": 0.0}
    assert fine_tune.resolve("gain_filter", {"db": 31.5}) == {"db": 30.0}
    assert fine_tune.resolve("noise_gate_filter", {"hold_time": 20000})["hold_time"] == 10000
    assert fine_tune.resolve("noise_gate_filter", {"attack_time": 12.6})["attack_time"] == 13  # an int property
    # A wrong type, a bool for a number, NaN, or a list value OBS doesn't offer, gives the default.
    bad = {"ratio": "x", "threshold": True, "attack_time": float("nan"), "detector": "loud", "presets": 3}
    assert fine_tune.resolve("expander_filter", bad) == fine_tune.defaults("expander_filter")
    assert (
        fine_tune.resolve("vst_filter", {"open_when_active_vst_settings": 1})["open_when_active_vst_settings"] is False
    )


# ------------------------------------------------------------------ T2 Gain, T3 Invert Polarity


def test_t2_gain() -> None:
    ones = np.ones(480, np.float32)
    assert obs_filters.Gain({"db": 6.0}).process(ones) == pytest.approx(np.full(480, 1.99526), abs=1e-5)
    assert float(obs_filters.Gain({"db": 6.0}).process(ones)[0]) == pytest.approx(10 ** (6 / 20), abs=1e-6)
    assert float(obs_filters.Gain({"db": -30.0}).process(ones)[0]) == pytest.approx(db(-30), abs=1e-7)
    assert float(obs_filters.Gain({"db": 30.0}).process(ones)[0]) == pytest.approx(db(30), rel=1e-6)
    assert float(obs_filters.Gain({"db": 45.0}).process(ones)[0]) == pytest.approx(db(30), rel=1e-6)  # clamped
    g = obs_filters.Gain()
    assert np.array_equal(g.process(ones), ones)  # 0 dB by default
    g.update({"db": -6.0})
    assert float(g.process(ones)[0]) == pytest.approx(db(-6), abs=1e-6)


def test_t3_invert_polarity() -> None:
    x = noise(0.5, 1)
    y = obs_filters.InvertPolarity().process(x)
    assert y.dtype == np.float32 and np.array_equal(y, -x)


# ------------------------------------------------------------------ T4 3-Band Equalizer

EPS = 1.0 / 4294967295.0  # eq-filter.c:97


def eq_oracle(x: np.ndarray, low: float, mid: float, high: float) -> np.ndarray:
    """eq-filter.c eq_create :83-85 and eq_process :99-127, one sample at a time."""
    lf = 2.0 * math.sin(math.pi * 800.0 / RATE)
    hf = 2.0 * math.sin(math.pi * 5000.0 / RATE)
    gl, gm, gh = db(low), db(mid), db(high)
    l0 = l1 = l2 = l3 = h0 = h1 = h2 = h3 = 0.0
    d1 = d2 = d3 = 0.0
    out = np.empty(len(x))
    for i, s in enumerate(x.astype(np.float64).tolist()):
        l0 += lf * (s - l0) + EPS
        l1 += lf * (l0 - l1)
        l2 += lf * (l1 - l2)
        l3 += lf * (l2 - l3)
        h0 += hf * (s - h0) + EPS
        h1 += hf * (h0 - h1)
        h2 += hf * (h1 - h2)
        h3 += hf * (h2 - h3)
        lo = l3
        hi = d3 - h3
        mi = d3 - (hi + lo)
        out[i] = lo * gl + mi * gm + hi * gh
        d3, d2, d1 = d2, d1, s
    return out


def test_t4_flat_eq_is_the_input_three_samples_late() -> None:
    x = noise(1.0, 2)
    y = run(obs_filters.ThreeBandEq(), x)
    assert np.max(np.abs(y[3:] - x[:-3])) < 1e-6
    assert np.max(np.abs(y[:3])) < 1e-6


@pytest.mark.parametrize("size", [480, 333])
def test_t4_eq_matches_the_oracle(size: int) -> None:
    rng = np.random.default_rng(5)
    x = noise(1.0, 3)
    for _ in range(3):
        low, mid, high = (float(v) for v in rng.uniform(-20, 20, 3))
        y = run(obs_filters.ThreeBandEq({"low": low, "mid": mid, "high": high}), x, size)
        assert np.max(np.abs(y - eq_oracle(x, low, mid, high))) < 1e-6


def test_t4_low_band_lifts_bass_only() -> None:
    def level(freq: float, low: float) -> float:
        y = run(obs_filters.ThreeBandEq({"low": low}), tone(freq, 1.0, 0.1))
        return 20 * math.log10(float(np.sqrt(np.mean(y[RATE // 2 :] ** 2))))

    assert level(100, 6.0) - level(100, 0.0) == pytest.approx(6.0, abs=0.5)
    assert abs(level(2000, 6.0) - level(2000, 0.0)) <= 1.0
    assert obs_filters.ThreeBandEq.LOW_FREQ == 800.0 and obs_filters.ThreeBandEq.HIGH_FREQ == 5000.0  # eq-filter.c:8-9


def test_t4_update_keeps_the_filter_state() -> None:
    # eq_update :44-50 only sets the gains.
    x = noise(0.5, 4)
    one = obs_filters.ThreeBandEq({"low": 3.0})
    a = run(one, x[:4800])
    one.update({"low": 3.0, "high": -4.0})
    b = run(one, x[4800:])
    want = eq_oracle(x[:4800], 3.0, 0.0, 0.0)
    assert np.max(np.abs(a - want)) < 1e-6
    assert np.max(np.abs(b[:10])) > 0  # carries on, no restart from zero


# ------------------------------------------------------------------ T5 Noise Gate


def gate_oracle(x: np.ndarray, open_db=-26.0, close_db=-32.0, attack=25, hold=200, release=150) -> np.ndarray:
    """noise-gate-filter.c noise_gate_update :84-99 and noise_gate_filter_audio :124-151, mono."""
    open_t, close_t = db(open_db), db(close_db)
    attack_rate = 1.0 / (attack / 1000.0 * RATE) if attack else math.inf
    release_rate = 1.0 / (release / 1000.0 * RATE) if release else math.inf
    decay = (open_t - close_t) / (RATE / 75.0)
    hold_s = hold / 1000.0
    is_open, att, level, held = False, 0.0, 0.0, 0.0
    out = np.empty(len(x))
    for i, s in enumerate(x.astype(np.float64).tolist()):
        cur = abs(s)
        if cur > open_t and not is_open:
            is_open = True
        if level < close_t and is_open:
            held = 0.0
            is_open = False
        level = max(level, cur) - decay
        if is_open:
            att = min(1.0, att + attack_rate)
        else:
            held += 1.0 / RATE
            if held > hold_s:
                att = max(0.0, att - release_rate)
        out[i] = s * att
    return out


def gate_burst() -> tuple[np.ndarray, int, int]:
    """Silence, 0.5 s at -20 dBFS (DC, so every sample is over the open threshold), then silence."""
    start, end = RATE // 10, RATE // 10 + RATE // 2
    x = np.zeros(2 * RATE, np.float32)
    x[start:end] = db(-20)
    return x, start, end


def test_t5_gate_opens_holds_and_releases() -> None:
    x, start, end = gate_burst()
    att = run(obs_filters.NoiseGate(), x) / db(-20)  # the gain, where the input is the burst
    assert np.all(att[:start] == 0.0)
    full = start + int(np.argmax(att[start:end] >= 1.0 - 1e-9))
    assert abs((full - start) - 0.025 * RATE) <= 1  # 25 ms attack (25 ms ± 1 sample)
    # After the burst, a -80 dBFS probe (far under the close threshold) shows the gain. The level decays by
    # (open - close) per 1/75 s until it is under the close threshold, then the gate holds 200 ms, then releases
    # linearly over 150 ms.
    probe = np.where(x == 0, db(-80), x).astype(np.float32)
    g = run(obs_filters.NoiseGate(), probe) / probe.astype(np.float64)
    decay = (db(-26) - db(-32)) / (RATE / 75.0)
    closes = end + math.floor((float(np.float32(db(-20))) - db(-32)) / decay)
    first_drop = end + int(np.argmax(g[end:] < 1.0 - 1e-5))
    assert abs(first_drop - (closes + int(0.2 * RATE))) <= 2
    zero = first_drop + int(np.argmax(g[first_drop:] <= 0.0))
    assert abs((zero - first_drop) - 0.15 * RATE) <= 2
    assert np.max(np.abs(np.diff(g[first_drop : zero - 1]) + 1.0 / (0.15 * RATE))) < 1e-5  # linear


def test_t5_a_level_between_close_and_open_does_not_reopen() -> None:
    x = np.zeros(RATE, np.float32)
    x[: RATE // 4] = db(-20)
    x[RATE // 4 :] = db(-29)  # under open (-26), over close (-32)
    y = run(obs_filters.NoiseGate(), x)
    # It stays open while the level holds above close, so the -29 dB part passes...
    assert np.all(y[RATE // 4 : RATE // 2] > 0)
    # ...but from closed it never opens.
    quiet = np.full(RATE, db(-29), np.float32)
    assert np.all(run(obs_filters.NoiseGate(), quiet) == 0.0)


@pytest.mark.parametrize("size", [480, 333])
def test_t5_gate_matches_the_oracle(size: int) -> None:
    x, _, _ = mix(-26.0, 20.0, seed=3)
    x = x[: 6 * RATE]
    for s in ({}, {"open_threshold": -40.0, "close_threshold": -50.0, "attack_time": 0, "hold_time": 0}):
        y = run(obs_filters.NoiseGate(s), x, size)
        kw = {"open_db": -40.0, "close_db": -50.0, "attack": 0, "hold": 0} if s else {}
        assert np.max(np.abs(y - gate_oracle(x, **kw))) < 1e-6


def test_t5_update_resets_the_gate() -> None:
    # noise_gate_update :96-99: is_open, attenuation, level and held time all start again.
    x, start, end = gate_burst()
    g = obs_filters.NoiseGate()
    run(g, x[: start + RATE // 10])  # open and fully on
    g.update(fine_tune.defaults("noise_gate_filter"))
    y = run(g, x[start + RATE // 10 : end])
    assert np.max(np.abs(y - gate_oracle(x[start + RATE // 10 : end]))) < 1e-6
    assert y[0] == 0.0


# ------------------------------------------------------------------ T6 Compressor


def comp_oracle(x: np.ndarray, threshold=-18.0, ratio=10.0, attack=6, release=60, gain=0.0) -> np.ndarray:
    """compressor-filter.c compressor_update :187-199, analyze_envelope :308-320, process_compression :358-367."""
    a = math.exp(-1.0 / (RATE * attack / 1000.0))
    r = math.exp(-1.0 / (RATE * release / 1000.0))
    slope = 1.0 - 1.0 / ratio
    env = 0.0
    out = np.empty(len(x))
    for i, s in enumerate(x.astype(np.float64).tolist()):
        v = abs(s)
        env = v + (a if env < v else r) * (env - v)
        g = min(0.0, slope * (threshold - 20 * math.log10(max(env, 1e-30))))
        out[i] = s * db(g) * db(gain)
    return out


def test_t6_compressor_static_curve() -> None:
    # A steady -8 dBFS tone is 10 dB over -18: at 10:1, 1 dB is left. A square wave keeps |x| steady, so the follower
    # settles exactly on the static curve.
    square = np.sign(tone(100, 2.0, 1.0)) * db(-8)
    y = run(obs_filters.Compressor(), square)
    assert 20 * math.log10(float(np.max(np.abs(y[RATE:])))) == pytest.approx(-17.0, abs=0.01)
    # On a sine the follower ripples between peaks (6 ms attack, 60 ms release), so the peak comes out a little higher;
    # it is still OBS's curve: the class equals the oracle (below).
    y = run(obs_filters.Compressor(), tone(500, 2.0, db(-8)))
    assert np.max(np.abs(y[RATE:] - comp_oracle(tone(500, 2.0, db(-8)))[RATE:])) < 1e-6
    assert -17.0 < 20 * math.log10(float(np.max(np.abs(y[RATE:])))) < -15.5


def test_t6_compressor_coefficients() -> None:
    c = obs_filters.Compressor({"attack_time": 6, "release_time": 60})
    assert c.attack == pytest.approx(math.exp(-1.0 / (48000 * 0.006)), rel=1e-12)
    assert c.release == pytest.approx(math.exp(-1.0 / (48000 * 0.060)), rel=1e-12)


@pytest.mark.parametrize("size", [480, 333])
def test_t6_compressor_matches_the_oracle(size: int) -> None:
    x, _, _ = mix(-16.0, 30.0, seed=2)
    x = x[: 6 * RATE]
    s = {"threshold": -30.0, "ratio": 4.0, "attack_time": 2, "release_time": 200, "output_gain": 5.0}
    assert np.max(np.abs(run(obs_filters.Compressor(), x, size) - comp_oracle(x))) < 1e-6
    assert np.max(np.abs(run(obs_filters.Compressor(s), x, size) - comp_oracle(x, -30.0, 4.0, 2, 200, 5.0))) < 1e-6


def j06_compressor(x: np.ndarray) -> np.ndarray:
    """dsp.Compressor exactly as J06 left it (-24 dB, 3:1, 5/120 ms, +6 dB), in 480 blocks: the bit-for-bit
    reference for the fixed stage now built on the parameterized class."""
    attack = math.exp(-1.0 / (48000 * 5.0 * 0.001))
    release = math.exp(-1.0 / (48000 * 120.0 * 0.001))
    slope = 1.0 - 1.0 / 3.0
    makeup = 10.0 ** (6.0 / 20.0)
    env = 0.0
    outs = []
    for k in range(0, len(x), 480):
        b = x[k : k + 480]
        env_buf = np.empty(len(b))
        for i, v in enumerate(np.abs(b).astype(np.float64).tolist()):
            env = v + (attack if env < v else release) * (env - v)
            env_buf[i] = env
        env_db = 20.0 * np.log10(np.maximum(env_buf, 1e-30))
        reduction_db = np.minimum(0.0, slope * (-24.0 - env_db))
        outs.append((b * (np.power(10.0, reduction_db / 20.0) * makeup)).astype(np.float32))
    return np.concatenate(outs)


def test_t6_fixed_compressor_is_bit_for_bit_j06() -> None:
    from clean_voice.dsp import Compressor

    x, _, _ = mix(-16.0, 20.0, seed=6)
    x = (x[: 8 * RATE] * 2).astype(np.float32)
    c = Compressor()
    got = np.concatenate([c.process(x[i : i + 480]) for i in range(0, len(x), 480)])
    assert got.dtype == np.float32 and np.array_equal(got, j06_compressor(x))


# ------------------------------------------------------------------ T7 Limiter


def limiter_oracle(x: np.ndarray, threshold=-6.0, release=60) -> np.ndarray:
    """limiter-filter.c limiter_update :88-100 (the 0.001 ms attack divided by 1000 again) and the shared envelope and
    gain of :124-166."""
    a = math.exp(-1.0 / (RATE * (0.001 / 1000.0)))
    r = math.exp(-1.0 / (RATE * release / 1000.0))
    env = 0.0
    out = np.empty(len(x))
    for i, s in enumerate(x.astype(np.float64).tolist()):
        v = abs(s)
        env = v + (a if env < v else r) * (env - v)
        out[i] = s * db(min(0.0, threshold - 20 * math.log10(max(env, 1e-30))))
    return out


def test_t7_limiter_holds_the_threshold_from_the_first_sample() -> None:
    y = run(obs_filters.Limiter(), tone(300, 1.0, 1.0))
    assert np.max(np.abs(y)) <= db(-6) * (1 + 1e-6)
    assert 20 * math.log10(float(np.max(np.abs(y[: RATE // 300])))) == pytest.approx(-6.0, abs=0.01)  # first cycle
    assert obs_filters.Limiter().attack == pytest.approx(math.exp(-1.0 / (48000 * 1e-6)), rel=1e-12)


def test_t7_limiter_release_is_the_60_ms_one_pole() -> None:
    x = np.concatenate([np.full(RATE // 10, 1.0, np.float32), np.full(RATE // 2, db(-40), np.float32)])
    y = run(obs_filters.Limiter(), x)
    r = math.exp(-1.0 / (RATE * 0.060))
    q = db(-40)
    for n in (1, 480, 2400, 4800):
        env = q + r**n * (1.0 - q)
        want = q * db(min(0.0, -6.0 - 20 * math.log10(env)))
        assert y[RATE // 10 + n - 1] == pytest.approx(want, rel=1e-4), n


@pytest.mark.parametrize("size", [480, 333])
def test_t7_limiter_matches_the_oracle(size: int) -> None:
    x, _, _ = mix(-10.0, 20.0, seed=4)
    x = x[: 6 * RATE]
    assert np.max(np.abs(run(obs_filters.Limiter(), x, size) - limiter_oracle(x))) < 1e-6
    lim = obs_filters.Limiter({"threshold": -20.0, "release_time": 5})
    assert np.max(np.abs(run(lim, x, size) - limiter_oracle(x, -20.0, 5))) < 1e-6


# ------------------------------------------------------------------ T8 Expander and Upward Compressor


def expander_oracle(x: np.ndarray, upward: bool, s: dict) -> np.ndarray:
    """expander-filter.c expander_update :198-220, analyze_envelope :284-319 and process_sample :322-374, mono."""
    a = math.exp(-1.0 / (RATE * s["attack_time"] / 1000.0))
    r = math.exp(-1.0 / (RATE * s["release_time"] / 1000.0))
    thr, slope, out_gain = s["threshold"], 1.0 - s["ratio"], db(s["output_gain"])
    knee = float(s["knee_width"]) if upward else 0.0
    rms = upward or s["detector"] == "RMS"
    c = 2.0 ** (-100.0 / RATE)
    runave, prev = 0.0, 0.0
    out = np.empty(len(x))
    for i, v in enumerate(x.astype(np.float64).tolist()):
        if rms:
            runave = c * runave + (1 - c) * v * v
            env = math.sqrt(max(runave, 0.0))
        else:
            env = abs(v)
        env_db = 20 * math.log10(env) if env > 0 else -math.inf
        diff = thr - env_db
        if upward:
            if env_db <= (thr - 60.0) / 2:
                diff = env_db + 60.0 if env_db + 60.0 > 0 else 0.0
            p = max(prev, 0.0)
            gain = 0.0
            if thr - knee / 2 >= env_db:
                gain = slope * diff
            if thr - knee / 2 < env_db < thr + knee / 2:
                gain = slope * (diff + knee / 2) ** 2 / (2.0 * knee)
        else:
            p = prev
            if diff > 0:
                g = slope * diff
                gain = -60.0 if math.isnan(g) else max(g, -60.0)
            else:
                gain = 0.0
        prev = a * p + (1 - a) * gain if gain > p else r * p + (1 - r) * gain
        out[i] = v * db(prev if upward else min(0.0, prev)) * out_gain
    return out


def dc_gain_db(f, level_db: float) -> float:
    y = run(f, np.full(2 * RATE, db(level_db), np.float32))
    return 20 * math.log10(float(y[-1])) - level_db


@pytest.mark.parametrize("detector", ["RMS", "peak"])
def test_t8_expander_static_curve(detector: str) -> None:
    def gain(level: float, **s) -> float:
        return dc_gain_db(obs_filters.Expander({"detector": detector, **s}), level)

    assert gain(-30.0) == pytest.approx(0.0, abs=0.01)  # over the -40 dB threshold: no change
    assert gain(-50.0) == pytest.approx(-10.0, abs=0.01)  # 10 dB under at 2:1: 10 dB more down
    assert gain(-55.0) == pytest.approx(-15.0, abs=0.01)
    assert gain(-59.0, ratio=20.0) == pytest.approx(-60.0, abs=0.01)  # the -60 dB floor (process_sample :353)


@pytest.mark.parametrize("level", [-45.0, -40.0, -30.0, -24.0, -22.0, -16.0, -10.0])
def test_t8_upward_compressor_static_curve(level: float) -> None:
    # Ratio 0.5, threshold -20, knee 10 (upward_compressor_defaults): below the knee slope x diff, in the knee the
    # quadratic, above it nothing; at or below (threshold - 60) / 2 = -40 the difference is env + 60.
    thr, knee, slope = -20.0, 10.0, 0.5
    diff = thr - level if level > (thr - 60) / 2 else level + 60
    if level <= thr - knee / 2:
        want = slope * diff
    elif level < thr + knee / 2:
        want = slope * (diff + knee / 2) ** 2 / (2 * knee)
    else:
        want = 0.0
    assert dc_gain_db(obs_filters.UpwardCompressor(), level) == pytest.approx(want, abs=0.01)


def test_t8_switching_the_preset_resets_to_its_defaults() -> None:
    e = obs_filters.Expander({"ratio": 4.0, "release_time": 300})
    assert e.settings["ratio"] == 4.0
    e.update({"presets": "gate", "ratio": 4.0, "release_time": 300, "threshold": -10.0})
    assert e.settings == fine_tune.defaults("expander_filter", {"presets": "gate"})
    assert (e.settings["ratio"], e.settings["release_time"]) == (10.0, 125)
    e.update({**e.settings, "ratio": 12.0})  # an edit while on the gate preset is kept
    assert e.settings["ratio"] == 12.0
    e.update({**e.settings, "presets": "expander"})
    assert e.settings == fine_tune.defaults("expander_filter")
    # A filter created on the gate preset keeps its own values (compressor_expander_create :242-247).
    assert obs_filters.Expander({"presets": "gate", "ratio": 5.0}).settings["ratio"] == 5.0


@pytest.mark.parametrize("size", [480, 333])
@pytest.mark.parametrize(
    ("upward", "s"),
    [
        (False, {}),
        (False, {"detector": "peak", "ratio": 4.0, "threshold": -30.0, "attack_time": 1, "output_gain": 3.0}),
        (False, {"presets": "gate"}),
        (True, {}),
        (True, {"ratio": 0.2, "threshold": -35.0, "knee_width": 0, "release_time": 400, "output_gain": -2.0}),
    ],
)
def test_t8_expander_and_upward_match_the_oracle(size: int, upward: bool, s: dict) -> None:
    x, _, _ = mix(-30.0, 15.0, seed=8)
    x = x[: 5 * RATE]
    f = obs_filters.UpwardCompressor(s) if upward else obs_filters.Expander(s)
    want = expander_oracle(x, upward, f.settings if upward else {**f.settings})
    assert np.max(np.abs(run(f, x, size) - want)) < 1e-6


# ------------------------------------------------------------------ T9 Noise Suppression


def test_t9_noise_suppression_is_rnnoise_fully_wet() -> None:
    from clean_voice.dsp import NoiseSuppressor

    x, _, _ = mix(-26.0, 10.0, seed=5)
    fan = looped_noise(len(x), seed=2, name="fan.wav") * db(-40)
    x = (x + fan).astype(np.float32)[: 6 * RATE]
    got = run(obs_filters.NoiseSuppression(), x)
    want = run(NoiseSuppressor(block=480), x)
    assert np.array_equal(got, want)
    assert KINDS["noise_suppress_filter"].props == ()


# ------------------------------------------------------------------ every filter


@pytest.mark.parametrize("kind", [k for k in KINDS if k != "vst_filter"])
def test_every_built_in_filter_takes_any_block_and_keeps_its_scratch(kind: str) -> None:
    f = obs_filters.make(kind, None, block=None)
    assert type(f).KIND == kind and f.settings == fine_tune.defaults(kind)
    x = noise(0.3, 9, 0.05)
    for n in (480, 1, 333, 1024, 0):
        y = f.process(x[:n])
        assert y.shape == (n,) and y.dtype == np.float32 and np.all(np.isfinite(y))
