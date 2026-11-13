"""The compiled test plug-in through the real loader (J10 plan T23, and the native half of T20's GIL check).

Build it with ``tools/vst_test_plugin/build.py`` (CI does, on Windows and macOS, before the tests); without it these
tests skip."""

from __future__ import annotations

import dataclasses
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from vst_fake import TEST_PLUGIN as PLUGIN
from vst_fake import require_test_plugin

from clean_voice import fine_tune, prefs, vst
from clean_voice.fine_tune import FilterSpec
from clean_voice.settings import Settings


@pytest.fixture
def plugin():
    require_test_plugin()
    loaded = vst.load(str(PLUGIN))
    yield loaded
    loaded.close()


def test_t23_the_real_binary_loads_and_says_what_it_is(plugin: vst.VstPlugin) -> None:
    assert (plugin.effect_name, plugin.vendor) == ("Test Gain", "Clean Voice tests")
    assert (plugin.num_inputs, plugin.num_outputs) == (2, 2)
    assert plugin.flags & vst.FLAG_PROGRAM_CHUNKS
    assert plugin.has_editor is (sys.platform == "win32")
    assert plugin.editor_rect() == (400, 300)


def test_t23_gain_half_halves_a_block(plugin: vst.VstPlugin) -> None:
    eff = plugin._effect.contents
    eff.set_parameter(plugin._effect, 0, 0.5)
    x = (0.8 * np.sin(np.arange(480) / 7)).astype(np.float32)
    np.testing.assert_allclose(plugin.process(x), 0.5 * x, rtol=1e-6)
    long = np.ones(1100, np.float32)  # passes of 512, 512, 76
    np.testing.assert_allclose(plugin.process(long), 0.5 * long)


def test_t23_the_real_binary_reads_obss_tempo_from_get_time(plugin: vst.VstPlugin) -> None:
    # gain_plugin.c asks for the time in every process call and goes silent unless it is 120 BPM, 4/4, 48 kHz with the
    # tempo flagged valid; get_parameter(3) gives the tempo it read.
    x = np.ones(480, np.float32)
    np.testing.assert_allclose(plugin.process(x), x)
    assert plugin._effect.contents.get_parameter(plugin._effect, 3) == 120.0


def test_t23_its_chunk_round_trips_through_settings_json(plugin: vst.VstPlugin, tmp_path: Path) -> None:
    eff = plugin._effect.contents
    eff.set_parameter(plugin._effect, 0, 0.25)
    spec = FilterSpec("vst_filter", "VST 2.x Plug-in", settings={"plugin_path": str(PLUGIN)})
    spec = dataclasses.replace(spec, settings={**spec.settings, **vst.saved_state(plugin)})
    file = tmp_path / "settings.json"
    prefs.save(file, prefs.Prefs(settings=Settings(fine_tune=True, filters=(spec,))))
    loaded = prefs.load(file).settings.filters[0]
    assert loaded.settings["chunk_data"] == spec.settings["chunk_data"]
    assert loaded.settings["chunk_hash"] == vst.file_md5(str(PLUGIN))
    again = vst.create(loaded.settings)  # a relaunch: vst_create with the saved settings
    assert again is not None
    try:
        assert again._effect.contents.get_parameter(again._effect, 0) == pytest.approx(0.25)
        x = np.ones(480, np.float32)
        np.testing.assert_allclose(again.process(x), 0.25 * x)
    finally:
        again.close()
    assert fine_tune.KINDS["vst_filter"].prop("chunk_data").widget == fine_tune.HIDDEN


def test_t20_the_gil_is_released_while_the_plugin_runs(plugin: vst.VstPlugin) -> None:
    eff = plugin._effect.contents
    eff.set_parameter(plugin._effect, 1, 1.0)  # each call sleeps 10 ms in native code
    x = np.ones(480, np.float32)
    done = threading.Event()
    calls = 20

    def worker() -> None:
        for _ in range(calls):
            plugin.process(x)
        done.set()

    # A UI-thread loop that notes how long it was held up: it can only run while the worker doesn't hold the GIL.
    stalled, last = 0.0, time.perf_counter()
    thread = threading.Thread(target=worker)
    thread.start()
    while not done.is_set():
        now = time.perf_counter()
        if now - last > 0.002:
            stalled += now - last
        last = now
    thread.join()
    # 200 ms asleep in the plug-in. Held through the call (measured with the same pointer called as a PYFUNCTYPE),
    # the loop stalls for about all of it; released, it hardly stalls.
    assert stalled < 0.1, stalled


def test_registry_unloads_the_real_binary(plugin: vst.VstPlugin) -> None:
    registry = vst.VstRegistry()
    other = vst.load(str(PLUGIN))
    spec = FilterSpec("vst_filter", "VST 2.x Plug-in", settings={"plugin_path": str(PLUGIN)})
    registry.add(spec.uid, other)
    registry.remove(spec.uid)
    assert not other.loaded and registry.pending == 0
