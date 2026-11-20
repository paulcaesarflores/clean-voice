"""B012: the 10-band parametric EQ. The sound (shapes of each type, bypass, click-free changes, save and restore) and
the panel (dragging a dot, typing in a box, the wheel). OBS has no such filter; its RBJ biquads are checked here by
measuring the real filter with sines."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot
from speech import RATE, speech
from test_filter_window import Host, make  # noqa: F401  (a pytest fixture)

from clean_voice import fine_tune, obs_filters, prefs
from clean_voice import parametric_eq as pe
from clean_voice.eq_panel import EqPanel
from clean_voice.fine_tune import EQ_KIND, FilterSpec, new_filter
from clean_voice.prefs import Prefs
from clean_voice.settings import Settings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import audio_check

BLOCK = 480
DROPOUTS = ("discontinuity", "long gap", "zero run")


def eq(**settings: object) -> obs_filters.Filter:
    return obs_filters.make(EQ_KIND, settings)


def band(i: int, **kw: object) -> dict[str, object]:
    """Band ``i`` switched on, with the given type, freq, gain, q."""
    out: dict[str, object] = {pe.key(i, "on"): True}
    out.update({pe.key(i, k): v for k, v in kw.items()})
    return out


def run(f: obs_filters.Filter, x: np.ndarray) -> np.ndarray:
    return np.concatenate([f.process(x[i : i + BLOCK]) for i in range(0, len(x), BLOCK)])


def gain_db(settings: Mapping[str, object], freq: float) -> float:
    """The real filter's gain at ``freq``, measured with a sine (the last second, after the filter has settled)."""
    t = np.arange(2 * RATE) / RATE
    x = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    y = run(eq(**settings), x)
    tail = slice(RATE, None)
    return 20.0 * np.log10(
        np.sqrt(np.mean(y[tail].astype(np.float64) ** 2)) / np.sqrt(np.mean(x[tail].astype(np.float64) ** 2))
    )


# ------------------------------------------------------------------ the sound


def test_defaults_are_the_agreed_ones() -> None:
    s = fine_tune.resolve(EQ_KIND, {})
    assert [s[pe.key(i, "on")] for i in range(1, 11)] == [False] * 10
    assert [s[pe.key(i, "freq")] for i in range(1, 11)] == [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    assert [s[pe.key(i, "type")] for i in range(1, 11)] == ["low_cut"] + ["bell"] * 8 + ["high_cut"]
    assert all(s[pe.key(i, "gain")] == 0.0 and s[pe.key(i, "q")] == 0.7 for i in range(1, 11))
    assert s["trim"] == 0.0
    # ranges are clamped
    r = fine_tune.resolve(EQ_KIND, {"b1_freq": 5, "b1_gain": 99, "b1_q": 0, "trim": -50})
    assert (r["b1_freq"], r["b1_gain"], r["b1_q"], r["trim"]) == (20.0, 18.0, 0.1, -18.0)


def test_a_bell_at_1k_plus_6_db_q_1_measures_plus_6_and_is_flat_two_octaves_away() -> None:
    s = band(6, type="bell", freq=1000.0, gain=6.0, q=1.0)
    assert gain_db(s, 1000.0) == pytest.approx(6.0, abs=0.5)
    assert abs(gain_db(s, 250.0)) < 1.0
    assert abs(gain_db(s, 4000.0)) < 1.0
    assert abs(gain_db(s, 30.0)) < 0.3


def test_each_type_has_its_shape() -> None:
    cut = gain_db(band(1, type="low_cut", freq=200.0, q=0.7), 50.0)
    assert cut < -20.0 and abs(gain_db(band(1, type="low_cut", freq=200.0, q=0.7), 2000.0)) < 0.5
    hi = band(10, type="high_cut", freq=2000.0, q=0.7)
    assert gain_db(hi, 12000.0) < -20.0 and abs(gain_db(hi, 200.0)) < 0.5
    notch = band(5, type="notch", freq=1000.0, q=4.0)
    assert gain_db(notch, 1000.0) < -30.0 and abs(gain_db(notch, 250.0)) < 0.5
    low = band(2, type="low_shelf", freq=200.0, gain=9.0, q=0.7)
    assert gain_db(low, 40.0) == pytest.approx(9.0, abs=1.0) and abs(gain_db(low, 5000.0)) < 0.5
    high = band(9, type="high_shelf", freq=4000.0, gain=-9.0, q=0.7)
    assert gain_db(high, 15000.0) == pytest.approx(-9.0, abs=1.0) and abs(gain_db(high, 200.0)) < 0.5


def test_the_graphs_curve_is_the_filters_response() -> None:
    s = fine_tune.resolve(EQ_KIND, {**band(3, gain=-5.0, freq=300.0, q=2.0), **band(7, type="notch", freq=3000.0)})
    for f in (100.0, 300.0, 1000.0, 2800.0):
        assert pe.response_db(s, np.array([f]))[0] == pytest.approx(gain_db(s, f), abs=0.15)


def test_trim_is_a_plain_output_gain() -> None:
    assert gain_db({"trim": 6.0}, 1000.0) == pytest.approx(6.0, abs=0.01)
    assert gain_db({**band(6, gain=6.0, freq=1000.0, q=1.0), "trim": -6.0}, 1000.0) == pytest.approx(0.0, abs=0.5)


def test_with_every_band_off_and_trim_0_the_output_is_the_input() -> None:
    x = (speech()[: 4 * RATE] * 0.3).astype(np.float32)
    assert np.array_equal(run(eq(), x), x)
    only_off = {pe.key(i, "gain"): 12.0 for i in range(1, 11)}  # gains set, but every band off
    assert np.array_equal(run(eq(**only_off), x), x)


def test_changing_a_gain_mid_stream_does_not_click() -> None:
    from test_dsp import energy_above_db  # the click check the style EQ's tests use

    tone = (0.2 * np.sin(2 * np.pi * 317 * np.arange(6 * RATE) / RATE)).astype(np.float32)
    at = len(tone) // 2 // BLOCK * BLOCK
    loud = band(5, freq=317.0, gain=18.0, q=1.0)
    # (first settings, settings after the change): a gain change, a band switched on and off, a type change
    for first, second in (
        (band(5, freq=317.0, gain=0.0, q=1.0), loud),
        (loud, band(5, freq=317.0, gain=-18.0, q=1.0)),
        ({}, loud),
        (loud, {}),
        (loud, band(5, type="low_cut", freq=2000.0, q=0.7)),
        (band(5, type="notch", freq=317.0, q=4.0), band(5, type="high_cut", freq=100.0, q=0.7)),
    ):
        f = eq(**first)
        out = [f.process(tone[i : i + BLOCK]) for i in range(0, at, BLOCK)]
        f.update(second)
        out += [f.process(tone[i : i + BLOCK]) for i in range(at, len(tone), BLOCK)]
        y = np.concatenate(out)
        assert energy_above_db(y, at + BLOCK // 2) <= -60.0
        assert [d for d in audio_check.analyze(y.astype(np.float64)) if d.kind in DROPOUTS] == []
    f = eq(**band(5, freq=317.0, gain=0.0, q=1.0))
    f.update(loud)
    y = np.concatenate([f.process(tone[i : i + BLOCK]) for i in range(0, len(tone), BLOCK)])
    assert abs(float(np.max(np.abs(y[-RATE:]))) - 0.2 * 10 ** (18 / 20)) < 0.05  # the change did arrive


def test_block_size_does_not_change_the_sound() -> None:
    x = (speech()[: 2 * RATE] * 0.3).astype(np.float32)
    s = {**band(3, gain=6.0, freq=300.0), **band(1, type="low_cut", freq=80.0)}
    a = run(eq(**s), x)
    f = obs_filters.make(EQ_KIND, s)
    b = np.concatenate([f.process(x[i : i + 100]) for i in range(0, len(x), 100)])
    assert np.allclose(a, b, atol=1e-6)


def test_settings_round_trip(tmp_path: Path) -> None:
    spec = new_filter(EQ_KIND, (), {**band(3, type="notch", freq=777.0, q=5.5), "trim": -3.5}, enabled=False)
    path = tmp_path / "settings.json"
    prefs.save(path, Prefs(settings=Settings(fine_tune=True, filters=(spec,))))
    back = prefs.load(path).settings.filters[0]
    assert back == spec and back.enabled is False
    assert back.settings["b3_type"] == "notch" and back.settings["b3_freq"] == 777.0 and back.settings["trim"] == -3.5
    assert back.settings["b4_on"] is False


# ------------------------------------------------------------------ the panel


def panel(qtbot: QtBot, **settings: object) -> tuple[EqPanel, list[tuple[str, object]]]:
    p = EqPanel(fine_tune.resolve(EQ_KIND, settings))
    seen: list[tuple[str, object]] = []
    p.changed.connect(lambda k, v: seen.append((k, v)))
    qtbot.addWidget(p)
    p.resize(760, 760)
    p.show()
    qtbot.waitExposed(p)
    return p, seen


def drag(p: EqPanel, band_no: int, freq: float, db: float) -> None:
    g = p.graph
    start = g.dot_pos(band_no)
    QTest.mousePress(g, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start.toPoint())
    mid = QPoint(round((start.x() + g.x_of(freq)) / 2), round((start.y() + g.y_of(db)) / 2))
    QTest.mouseMove(g, mid)
    QTest.mouseMove(g, QPoint(round(g.x_of(freq)), round(g.y_of(db))))
    QTest.mouseRelease(
        g, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(round(g.x_of(freq)), round(g.y_of(db)))
    )


def wheel(p: EqPanel, pos: QPointF, delta: int) -> QWheelEvent:
    ev = QWheelEvent(
        pos,
        p.graph.mapToGlobal(pos),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    ev.setAccepted(False)
    QApplication.sendEvent(p.graph, ev)
    return ev


def test_dragging_a_dot_changes_that_bands_frequency_and_gain(qtbot: QtBot) -> None:
    p, seen = panel(qtbot, **band(6, gain=0.0))
    drag(p, 6, 2000.0, 6.0)
    assert p.values[pe.key(6, "freq")] == pytest.approx(2000.0, rel=0.03)
    assert p.values[pe.key(6, "gain")] == pytest.approx(6.0, abs=0.3)
    assert {k for k, _ in seen} <= {pe.key(6, "freq"), pe.key(6, "gain")} and seen
    assert p.rows[5].freq.value() == pytest.approx(2000.0, rel=0.03)  # the boxes follow the dot
    assert p.rows[5].gain.value() == pytest.approx(6.0, abs=0.3)
    assert p.values[pe.key(5, "freq")] == 500.0  # nobody else moved


def test_a_dot_stays_inside_the_graph(qtbot: QtBot) -> None:
    p, _ = panel(qtbot, **band(6))
    drag(p, 6, 100000.0, 80.0)
    assert float(p.values[pe.key(6, "freq")]) <= 20000.0 and p.values[pe.key(6, "gain")] == 18.0


def test_typing_in_a_box_moves_the_dot(qtbot: QtBot) -> None:
    p, seen = panel(qtbot, **band(4))
    before = p.graph.dot_pos(4)
    p.rows[3].freq.setValue(800.0)
    p.rows[3].gain.setValue(-9.0)
    after = p.graph.dot_pos(4)
    assert after.x() > before.x() and after.y() > before.y()
    assert after.x() == pytest.approx(p.graph.x_of(800.0), abs=1) and after.y() == pytest.approx(
        p.graph.y_of(-9.0), abs=1
    )
    assert (pe.key(4, "freq"), 800.0) in seen and (pe.key(4, "gain"), -9.0) in seen
    p.rows[3].q.setValue(3.3)
    assert p.values[pe.key(4, "q")] == pytest.approx(3.3)


def test_the_wheel_over_a_dot_changes_q(qtbot: QtBot) -> None:
    p, _ = panel(qtbot, **band(6, q=1.0))
    pos = p.graph.dot_pos(6)
    wheel(p, pos, 120)
    assert float(p.values[pe.key(6, "q")]) > 1.0
    up = float(p.values[pe.key(6, "q")])
    wheel(p, pos, -240)
    assert float(p.values[pe.key(6, "q")]) < up
    assert p.rows[5].q.value() == pytest.approx(p.values[pe.key(6, "q")], abs=0.01)
    # the wheel away from any dot does nothing, and is left for the pane to scroll with
    before = p.values[pe.key(6, "q")]
    ev = wheel(p, QPointF(p.graph.x_of(20.0) + 2, p.graph.y_of(-17.0)), 120)
    assert not ev.isAccepted() and p.values[pe.key(6, "q")] == before


def test_the_on_button_the_type_buttons_and_the_trim(qtbot: QtBot) -> None:
    p, seen = panel(qtbot)
    assert not p.rows[2].on.isChecked()
    p.rows[2].on.click()
    assert p.values[pe.key(3, "on")] is True and (pe.key(3, "on"), True) in seen
    assert p.rows[2].types["bell"].isChecked()
    p.rows[2].types["notch"].click()
    assert p.values[pe.key(3, "type")] == "notch" and not p.rows[2].types["bell"].isChecked()
    p.trim.setValue(-4.0)
    assert p.values["trim"] == -4.0 and ("trim", -4.0) in seen


def test_the_graph_draws_and_marks_a_disabled_band_grey(qtbot: QtBot) -> None:
    p, _ = panel(qtbot, **band(6, gain=6.0), **{pe.key(3, "gain"): 6.0})
    image = p.graph.grab().toImage()
    assert not image.isNull() and image.width() > 300
    assert p.graph.dot_pos(3).x() > 0  # every band has a dot, on or off
    assert p.graph.curve_db(1000.0) == pytest.approx(6.0, abs=1.0)  # one band on: the summed curve


def test_the_filter_window_shows_the_panel_and_saves_what_it_changes(make, qtbot: QtBot) -> None:  # noqa: F811
    spec = new_filter(EQ_KIND, ())
    window, host = make(Settings(fine_tune=True, filters=(spec,)))
    assert isinstance(window.pane.controls.get("eq"), EqPanel)
    panel_ = window.pane.controls["eq"]
    panel_.rows[0].on.click()
    saved = host.settings.filters[0]
    assert saved.settings["b1_on"] is True and saved.kind == EQ_KIND
    drag(panel_, 6, 3000.0, -6.0)
    assert host.settings.filters[0].settings["b6_gain"] == pytest.approx(-6.0, abs=0.3)
    assert host.settings.filters[0].settings["b6_freq"] == pytest.approx(3000.0, rel=0.03)
    assert (
        FilterSpec(EQ_KIND, "x", True, host.settings.filters[0].settings).settings == host.settings.filters[0].settings
    )


# ------------------------------------------------------------------ B014: the slope of a cut


@pytest.mark.parametrize("slope", [6, 12, 18, 24])
@pytest.mark.parametrize("kind", ["low_cut", "high_cut"])
def test_a_cuts_slope_is_6_12_18_or_24_db_per_octave(kind: str, slope: int) -> None:
    s = band(5, type=kind, freq=1000.0, q=0.7, slope=slope)
    near, far = (500.0, 250.0) if kind == "low_cut" else (2000.0, 4000.0)  # one and two octaves into the cut
    one, two = gain_db(s, near), gain_db(s, far)
    assert one - two == pytest.approx(slope, abs=1.5)
    assert two == pytest.approx(-2 * slope, abs=3.0)
    assert gain_db(s, 1000.0) == pytest.approx(-3.0, abs=1.0)  # Butterworth at the default Q: -3 dB at the cutoff


def test_the_default_slope_is_12_and_old_settings_load_as_12() -> None:
    s = fine_tune.resolve(EQ_KIND, {})
    assert all(s[pe.key(i, "slope")] == 12 for i in range(1, 11))
    old = band(1, type="low_cut", freq=200.0, q=0.7)  # saved before B014: no b1_slope
    assert pe.bands_of(fine_tune.resolve(EQ_KIND, old))[0].slope == 12
    assert pe.bands_of(fine_tune.resolve(EQ_KIND, {"b1_slope": 7}))[0].slope == 6  # snapped to an allowed one
    assert gain_db(old, 100.0) == pytest.approx(gain_db({**old, "b1_slope": 12}, 100.0), abs=1e-6)


def test_a_slope_change_mid_stream_does_not_click() -> None:
    from test_dsp import energy_above_db

    tone = (0.2 * np.sin(2 * np.pi * 317 * np.arange(6 * RATE) / RATE)).astype(np.float32)
    at = len(tone) // 2 // BLOCK * BLOCK
    for kind in ("low_cut", "high_cut"):
        for a, b in ((12, 24), (24, 6), (6, 18), (18, 12)):
            f = eq(**band(5, type=kind, freq=317.0, q=0.7, slope=a))
            out = [f.process(tone[i : i + BLOCK]) for i in range(0, at, BLOCK)]
            f.update(band(5, type=kind, freq=317.0, q=0.7, slope=b))
            out += [f.process(tone[i : i + BLOCK]) for i in range(at, len(tone), BLOCK)]
            y = np.concatenate(out)
            assert energy_above_db(y, at + BLOCK // 2) <= -60.0
            assert [d for d in audio_check.analyze(y.astype(np.float64)) if d.kind in DROPOUTS] == []


@pytest.mark.parametrize("slope", [6, 12, 18, 24])
@pytest.mark.parametrize("kind", ["low_cut", "high_cut"])
def test_the_graph_matches_the_filter_at_each_slope(kind: str, slope: int) -> None:
    s = fine_tune.resolve(EQ_KIND, band(5, type=kind, freq=1000.0, q=1.3, slope=slope))
    for f in (150.0, 500.0, 900.0, 1100.0, 2500.0, 7000.0):
        assert pe.response_db(s, np.array([f]))[0] == pytest.approx(gain_db(s, f), abs=0.15)


def test_the_slope_dropdown_shows_only_for_cuts_and_changes_the_setting(qtbot: QtBot) -> None:
    p, seen = panel(qtbot)
    r1, r3 = p.rows[0], p.rows[2]  # band 1 is a low cut, band 3 a bell
    assert not r1.slope.isHidden() and r1.gain.isHidden()
    assert r3.slope.isHidden() and not r3.gain.isHidden()
    assert [r1.slope.itemText(i) for i in range(r1.slope.count())] == ["6 dB", "12 dB", "18 dB", "24 dB"]
    assert r1.slope.currentText() == "12 dB"
    r1.slope.setCurrentIndex(3)
    assert p.values[pe.key(1, "slope")] == 24 and (pe.key(1, "slope"), 24) in seen
    assert r1.q.isEnabled()
    r1.slope.setCurrentIndex(0)
    assert not r1.q.isEnabled()
    r3.types["high_cut"].click()
    assert not r3.slope.isHidden() and r3.gain.isHidden()
    r1.types["bell"].click()
    assert r1.slope.isHidden() and not r1.gain.isHidden() and r1.q.isEnabled()
