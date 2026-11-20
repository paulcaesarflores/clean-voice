"""NVIDIA Audio Effects (J12), against OBS's filter (plugins/nv-filters/nvidia-audiofx-filter.c, nvafx-load.h,
nv-filters.c, nv_sdk_versions.h): where the runtime is looked for, the + menu with and without it, OBS's properties and
defaults, the effect's calls in OBS's order on a fake runtime, saving and loading, and a saved filter with no runtime.

Everything here runs without NVIDIA's runtime (CI has none). The last tests load the real one and are skipped where it
is not installed.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from speech import FAN, RATE, active_rms_db, mix, pauses, read_wav
from test_filter_window import Fakes, Host

from clean_voice import fine_tune, nvafx, obs_filters, prefs
from clean_voice.dsp import FineTune
from clean_voice.filter_window import FilterModel, FilterWindow, Pane
from clean_voice.fine_tune import NVIDIA_KIND, FilterSpec, new_filter
from clean_voice.prefs import Prefs
from clean_voice.settings import Settings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import audio_check

BLOCK = 480
SPEECH = Path(__file__).resolve().parent / "data" / "speech.wav"
DROPOUTS = ("discontinuity", "long gap", "zero run")
V1_7 = nvafx.pack_version(1, 7, 0, 0)
STAND_IN = nvafx.Runtime("C:\\NVIDIA", V1_7)  # for the menu and the pane: no API behind it


class FakeApi:
    """The NVAFX C API as a Python object: records every call, scales each frame it is asked to run, and fails on
    request, so the effect's calls can be checked in OBS's order with no GPU."""

    def __init__(self, gain: float = 0.5, fail: dict[str, int] | None = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.gain = gain
        self.fail = dict(fail or {})
        self.params: dict[bytes, Any] = {}
        self.next_handle = 100
        self.destroyed: list[int] = []
        self.load_gate = threading.Event()  # Load waits for this: a slow model
        self.load_gate.set()
        self.channels, self.frame = 1, 480
        self.logger = 0
        self.run_status = 0

    def _do(self, name: str, *args: Any) -> int:
        self.calls.append((name, *args))
        return self.fail.get(name, 0)

    def create_effect(self, code: str) -> tuple[int, int | None]:
        status = self._do("create", code)
        if status:
            return status, None
        self.next_handle += 1
        return 0, self.next_handle

    def destroy_effect(self, handle: int) -> int:
        self.destroyed.append(handle)
        return self._do("destroy", handle)

    def set_u32(self, handle: int, name: bytes, value: int) -> int:
        self.params[name] = value
        return self._do("set_u32", name.decode(), value)

    def set_float(self, handle: int, name: bytes, value: float) -> int:
        self.params[name] = value
        return self._do("set_float", name.decode(), round(value, 4))

    def set_string(self, handle: int, name: bytes, value: str) -> int:
        self.params[name] = value
        return self._do("set_string", name.decode(), value)

    def get_u32(self, handle: int, name: bytes) -> tuple[int, int]:
        status = self._do("get_u32", name.decode())
        return status, self.channels if name == nvafx.PARAM_NUM_INPUT_CHANNELS else self.frame

    def load_effect(self, handle: int) -> int:
        self.load_gate.wait(10)
        return self._do("load")

    def channel_pointers(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def run(self, handle: int, pointers: np.ndarray, samples: int, channels: int) -> int:
        self.calls.append(("run", samples, channels))
        if self.run_status == 0:
            pointers *= self.gain  # the SDK works in place
        return self.run_status

    def init_logger(self) -> int:
        self.logger += 1
        return 0

    def uninit_logger(self) -> None:
        self.logger -= 1

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


def wait_for(condition: Any, seconds: float = 5.0) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("timed out")


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Install a runtime (or None) as the one the check found."""

    def install(rt: nvafx.Runtime | None) -> None:
        monkeypatch.setattr(nvafx, "_probed", True)
        monkeypatch.setattr(nvafx, "_runtime", rt)

    yield install


def fake(version: int = V1_7, **kw: Any) -> tuple[nvafx.Runtime, FakeApi]:
    api = FakeApi(**kw)
    return nvafx.Runtime("C:\\NVIDIA", version, api), api  # type: ignore[arg-type]


def make_nv(settings: dict[str, Any] | None, block: int | None = BLOCK) -> obs_filters.NvidiaAudioEffects:
    f = obs_filters.make(NVIDIA_KIND, settings, block=block)
    assert isinstance(f, obs_filters.NvidiaAudioEffects)
    return f


def blocks(f: Any, x: np.ndarray, size: int = BLOCK) -> np.ndarray:
    return np.concatenate([f.process(x[i : i + size]) for i in range(0, len(x), size)])


# ------------------------------------------------------------------ where OBS looks for the runtime


def test_the_folder_is_found_as_nvafx_load_h_does() -> None:
    pf = {"ProgramFiles": "C:\\Program Files"}
    default = "C:\\Program Files\\NVIDIA Corporation\\NVIDIA Audio Effects SDK"
    assert nvafx.sdk_dir(pf) == default  # nvafx_get_sdk_path :243
    assert nvafx.sdk_dir({**pf, "AFX_SDK_DIR": "D:\\afx"}) == "D:\\afx"  # :234
    assert nvafx.sdk_dir({**pf, "AFX_SDK_DIR": "D:\\afx", "NVAFX_SDK_DIR": "E:\\nv"}) == "E:\\nv"  # :231 first
    assert nvafx.sdk_dir({**pf, "NVAFX_SDK_DIR": ""}) == default  # an empty variable is no variable (:233)
    assert nvafx.sdk_dir({**pf, "NVAFX_SDK_DIR": "x" * 300}) == default  # too long for OBS's buffer (:233)
    assert nvafx.sdk_dir({}) == ""  # not even ProgramFiles (:238-241)


def test_versions_and_model_files_are_nv_sdk_versions_h() -> None:
    assert nvafx.MIN_SDK_VERSION == nvafx.MIN_LOGGER_VERSION == 0x01060102  # 1.6.1.2
    assert nvafx.MIN_ARM_VERSION == 0x03000031  # 3.0.0.49
    assert nvafx.version_text(nvafx.MIN_ARM_VERSION) == "3.0.0.49"
    assert nvafx.MODELS == {
        "denoiser": "\\models\\denoiser_48k.trtpkg",
        "dereverb": "\\models\\dereverb_48k.trtpkg",
        "dereverb_denoiser": "\\models\\dereverb_denoiser_48k.trtpkg",
    }
    assert (nvafx.SAMPLE_RATE, nvafx.FRAME_SIZE) == (48000, 480)


def test_runtime_flags_follow_the_versions() -> None:
    old = nvafx.Runtime("x", nvafx.pack_version(1, 5, 0, 0))
    assert old.outdated and not old.vad_supported  # :846, :821
    assert not nvafx.Runtime("x", 0).outdated  # an unreadable version is not "outdated" (:846 needs a version)
    assert not nvafx.Runtime("x", V1_7).outdated and nvafx.Runtime("x", V1_7).vad_supported


def test_without_the_dll_there_is_no_runtime(tmp_path: Path) -> None:
    assert nvafx.load(str(tmp_path)) is None  # nothing to load: not offered, nothing raised
    assert nvafx.load("") is None
    assert nvafx.dll_version(str(tmp_path / "NVAudioEffects.dll")) == 0


def test_macos_never_looks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert nvafx.load("/anything") is None


# ------------------------------------------------------------------ OBS's data: names, properties, defaults


def test_the_kind_is_obss_names_properties_and_defaults() -> None:
    k = fine_tune.KINDS[NVIDIA_KIND]
    assert (k.id, k.name) == ("nvidia_audiofx_filter", "NVIDIA Audio Effects")  # .id :858, Nvafx en-US.ini :1
    assert [(p.key, p.label, p.widget) for p in k.props] == [
        ("method", "Method", fine_tune.LIST),
        ("intensity", "Suppression Intensity", fine_tune.SLIDER),
        ("vad", "Voice Optimized", fine_tune.CHECKBOX),
    ]
    assert k.prop("method").options == (
        ("denoiser", "NVIDIA Noise Removal"),
        ("dereverb", "NVIDIA Room Echo Removal"),
        ("dereverb_denoiser", "NVIDIA Noise Removal + Room Echo Removal"),
    )
    slider = k.prop("intensity")
    assert (slider.lo, slider.hi, slider.step) == (0.0, 1.0, 0.01)  # obs_properties_add_float_slider :839
    assert fine_tune.defaults(NVIDIA_KIND) == {"method": "denoiser", "intensity": 1.0, "vad": True}  # :809-812


def test_settings_out_of_range_or_wrong_are_cleaned_like_any_filter() -> None:
    r = fine_tune.resolve(NVIDIA_KIND, {"method": "echo", "intensity": 7, "vad": "yes", "extra": 1})
    assert r == {"method": "denoiser", "intensity": 1.0, "vad": True}
    assert fine_tune.resolve(NVIDIA_KIND, {"method": "dereverb", "intensity": 0.25, "vad": False}) == {
        "method": "dereverb",
        "intensity": 0.25,
        "vad": False,
    }


def test_the_menu_position_is_after_limiter_and_before_noise_gate() -> None:
    names = [fine_tune.KINDS[k].name for k in fine_tune.MENU_ORDER]
    assert names[names.index("Limiter") + 1 : names.index("Limiter") + 3] == ["NVIDIA Audio Effects", "Noise Gate"]
    assert fine_tune.MENU_ORDER[: len(fine_tune.OWN_KINDS)] == fine_tune.OWN_KINDS  # below Clean Voice's own, first
    assert NVIDIA_KIND not in fine_tune.menu_order(False)
    assert fine_tune.menu_order(True) == fine_tune.MENU_ORDER
    assert [k for k in fine_tune.MENU_ORDER if k != NVIDIA_KIND] == list(fine_tune.menu_order(False))


# ------------------------------------------------------------------ the + menu shows or hides the filter


@pytest.fixture
def make_window(qtbot: QtBot) -> Iterator[Any]:
    windows: list[FilterWindow] = []

    def make(rt: nvafx.Runtime | None, settings: Settings | None = None) -> tuple[FilterWindow, Host]:
        host = Host(settings)
        model = FilterModel(
            lambda: host.settings,
            host.publish,
            host.registry,
            Fakes(),
            list,
            lambda: [Path("C:/Program Files/VSTPlugins")],
        )
        model.load_saved()
        window = FilterWindow(model, runtime=lambda: rt)
        window.ask_name = lambda default: default  # type: ignore[method-assign]  # the name's question: accept it
        window.confirm = lambda title, text: True  # type: ignore[method-assign]
        qtbot.addWidget(window)
        window.show()
        windows.append(window)
        return window, host

    yield make


def menu_names(window: FilterWindow) -> list[str]:
    return [a.text() for a in window.add_menu.actions() if not a.isSeparator()]


def test_the_menu_offers_the_filter_when_the_runtime_is_found(make_window: Any) -> None:
    window, _ = make_window(STAND_IN)
    names = menu_names(window)
    assert names[:3] == ["Denoise", "Voice Style", "10-Band EQ"] and window.add_menu.actions()[3].isSeparator()
    i = names.index("Limiter")
    assert names[i : i + 3] == ["Limiter", "NVIDIA Audio Effects", "Noise Gate"]
    assert [a.data() for a in window.add_menu.actions() if not a.isSeparator()] == list(fine_tune.MENU_ORDER)


def test_the_menu_has_no_filter_without_the_runtime(make_window: Any) -> None:
    window, _ = make_window(None)
    names = menu_names(window)
    assert "NVIDIA Audio Effects" not in names
    assert len(names) == len(fine_tune.MENU_ORDER) - 1 == 13
    assert [a.data() for a in window.add_menu.actions() if not a.isSeparator()] == list(fine_tune.menu_order(False))


def test_the_menu_asks_the_real_check_by_default(qtbot: QtBot, runtime: Any) -> None:
    for found in (STAND_IN, None):
        runtime(found)
        host = Host()
        model = FilterModel(lambda host=host: host.settings, host.publish, host.registry, Fakes(), list, list)
        window = FilterWindow(model)
        qtbot.addWidget(window)
        assert ("NVIDIA Audio Effects" in menu_names(window)) is (found is not None)


def test_adding_it_names_it_and_uses_obss_defaults(make_window: Any) -> None:
    window, host = make_window(STAND_IN)
    window.add_filter(NVIDIA_KIND)
    window.add_filter(NVIDIA_KIND)
    assert [f.name for f in host.settings.filters] == ["NVIDIA Audio Effects", "NVIDIA Audio Effects 2"]
    assert dict(host.settings.filters[0].settings) == {"method": "denoiser", "intensity": 1.0, "vad": True}


# ------------------------------------------------------------------ the pane: OBS's properties


def test_the_pane_shows_the_three_properties(qtbot: QtBot) -> None:
    pane = Pane(new_filter(NVIDIA_KIND, ()), runtime=STAND_IN)
    qtbot.addWidget(pane)
    pane.show()
    assert set(pane.controls) == {"method", "intensity", "vad"}
    assert [pane.labels[k].text() for k in ("method", "intensity")] == ["Method", "Suppression Intensity"]
    assert pane.controls["vad"].text() == "Voice Optimized"  # type: ignore[attr-defined]
    combo = pane.controls["method"]
    assert [combo.itemText(i) for i in range(combo.count())] == [  # type: ignore[attr-defined]
        "NVIDIA Noise Removal",
        "NVIDIA Room Echo Removal",
        "NVIDIA Noise Removal + Room Echo Removal",
    ]


def test_the_longest_method_name_does_not_push_the_pane_past_the_window(make_window: Any) -> None:
    window, _ = make_window(STAND_IN, Settings(fine_tune=True, filters=(new_filter(NVIDIA_KIND, ()),)))
    window.list.setCurrentRow(0)
    window.show()
    assert window.pane is not None
    assert window.pane.minimumSizeHint().width() <= window.pane_scroll.viewport().width()


def test_voice_optimized_hides_for_room_echo_removal_alone(qtbot: QtBot) -> None:
    # nvidia_audio_method_modified :815-826
    pane = Pane(new_filter(NVIDIA_KIND, ()), runtime=STAND_IN)
    qtbot.addWidget(pane)
    pane.show()
    vad, combo = pane.controls["vad"], pane.controls["method"]
    assert vad.isVisible()
    combo.setCurrentIndex(1)  # type: ignore[attr-defined]  # Room Echo Removal
    assert not vad.isVisible()
    combo.setCurrentIndex(2)  # type: ignore[attr-defined]
    assert vad.isVisible()
    old = nvafx.Runtime("x", nvafx.pack_version(1, 5, 0, 0))
    pane2 = Pane(new_filter(NVIDIA_KIND, ()), runtime=old)  # an SDK before 1.6.1.2 has no Voice Optimized
    qtbot.addWidget(pane2)
    pane2.show()
    assert not pane2.controls["vad"].isVisible()
    assert pane2.status.isVisible() and "outdated" in pane2.status.text()  # :846-851
    assert not pane.status.isVisible()


def test_a_saved_filter_without_the_runtime_has_a_note_and_no_controls(qtbot: QtBot) -> None:
    pane = Pane(new_filter(NVIDIA_KIND, ()), runtime=None)
    qtbot.addWidget(pane)
    assert pane.controls == {}


# ------------------------------------------------------------------ the effect, on a fake runtime


def test_the_effect_loads_on_a_thread_in_obss_order() -> None:
    rt, api = fake()
    api.load_gate.clear()
    effect = nvafx.Effect(rt, "dereverb_denoiser", 0.4, True)
    main = threading.current_thread()
    assert not effect.ready  # it is loading, off this thread: the constructor came back
    frame = np.full(480, 0.5, np.float32)
    assert effect.run(frame) is False and np.all(frame == 0.5)  # loading: the audio passes unchanged (:655, :710)
    api.load_gate.set()
    wait_for(lambda: effect.ready)
    assert threading.current_thread() is main
    assert api.calls == [
        ("create", "dereverb_denoiser"),  # nvidia_audio_initialize_internal :284
        ("set_u32", "input_sample_rate", 48000),  # :290
        ("set_float", "intensity_ratio", 0.4),  # :298
        ("set_u32", "enable_vad", 1),  # :307
        ("set_string", "model_path", "C:\\NVIDIA\\models\\dereverb_denoiser_48k.trtpkg"),  # :316, set_nv_model :424-442
        ("load",),  # :323
        ("get_u32", "num_input_channels"),  # :361
        ("get_u32", "num_input_samples_per_frame"),  # before SDK 3.0.0.49: :371-373
    ]
    assert api.logger == 1  # nvidia_audio_create :614-619
    assert effect.run(frame) is True and np.allclose(frame, 0.25)  # run in place on 480 samples
    assert api.calls[-1] == ("run", 480, 1)


def test_sdk_3_uses_the_renamed_frame_parameter_and_the_denoiser_model() -> None:
    rt, api = fake(nvafx.pack_version(3, 0, 0, 49))
    effect = nvafx.Effect(rt, "denoiser", 1.0, False)
    wait_for(lambda: effect.ready)
    assert ("get_u32", "num_samples_per_input_frame") in api.calls  # updated_string :370-374
    assert ("set_string", "model_path", "C:\\NVIDIA\\models\\denoiser_48k.trtpkg") in api.calls
    assert ("set_u32", "enable_vad", 0) in api.calls


def test_voice_optimized_is_not_set_for_dereverb_or_an_old_sdk() -> None:
    rt, api = fake()
    effect = nvafx.Effect(rt, "dereverb", 1.0, True)
    wait_for(lambda: effect.ready)
    assert "enable_vad" not in [c[1] for c in api.calls if c[0] == "set_u32"]  # :306, :487
    rt2, api2 = fake(nvafx.pack_version(1, 5, 0, 0))
    effect2 = nvafx.Effect(rt2, "denoiser", 1.0, True)
    wait_for(lambda: effect2.ready)
    assert "enable_vad" not in [c[1] for c in api2.calls if c[0] == "set_u32"]
    assert api2.logger == 0  # no logger before SDK 1.6.1.2 (:614)


def test_a_change_of_intensity_or_vad_is_a_set_call_not_a_reload() -> None:
    rt, api = fake()
    effect = nvafx.Effect(rt, "denoiser", 1.0, True)
    wait_for(lambda: effect.ready)
    api.calls.clear()
    effect.set("denoiser", 0.3, False)  # :472-502
    assert api.calls == []  # update is cheap: the next frame applies it
    effect.run(np.zeros(480, np.float32))
    assert api.calls == [("set_float", "intensity_ratio", 0.3), ("set_u32", "enable_vad", 0), ("run", 480, 1)]
    assert "create" not in api.names() and "load" not in api.names()


def test_a_change_of_method_destroys_the_effect_and_loads_the_new_one() -> None:
    rt, api = fake()
    effect = nvafx.Effect(rt, "denoiser", 1.0, True)
    wait_for(lambda: effect.ready)
    api.load_gate.clear()
    effect.set("dereverb", 1.0, True)  # :504-525
    assert not effect.ready  # reinit_done = false: no Run, the audio passes through while it loads
    frame = np.full(480, 0.5, np.float32)
    assert effect.run(frame) is False and np.all(frame == 0.5)
    api.load_gate.set()
    wait_for(lambda: effect.ready)
    assert api.destroyed == [101]
    creates = [c for c in api.calls if c[0] == "create"]
    assert creates == [("create", "denoiser"), ("create", "dereverb")]
    assert ("set_string", "model_path", "C:\\NVIDIA\\models\\dereverb_48k.trtpkg") in api.calls


def test_run_never_waits_for_a_load_in_progress() -> None:
    rt, api = fake()
    effect = nvafx.Effect(rt, "denoiser", 1.0, True)
    wait_for(lambda: effect.ready)
    api.load_gate.clear()
    effect.set("dereverb", 1.0, True)
    wait_for(lambda: effect._lock.locked())  # the load thread holds the lock, inside Load
    start = time.perf_counter()
    effect.run(np.zeros(480, np.float32))
    assert time.perf_counter() - start < 5.0  # OBS's mutex would wait for the whole load (:656), here for ever
    api.load_gate.set()
    wait_for(lambda: effect.ready)


def test_changes_during_a_load_are_applied_when_it_is_done() -> None:
    rt, api = fake()
    api.load_gate.clear()
    effect = nvafx.Effect(rt, "denoiser", 1.0, True)
    effect.set("denoiser", 0.2, True)  # while Load runs: stage 1's values are read at creation (:461-466)
    api.load_gate.set()
    wait_for(lambda: effect.ready)
    effect.run(np.zeros(480, np.float32))
    assert ("set_float", "intensity_ratio", 0.2) in api.calls


def test_the_newest_method_wins_when_changes_come_fast() -> None:
    rt, api = fake()
    api.load_gate.clear()
    effect = nvafx.Effect(rt, "denoiser", 1.0, True)
    wait_for(lambda: effect._lock.locked())
    effect.set("dereverb", 1.0, True)
    effect.set("dereverb_denoiser", 1.0, True)
    api.load_gate.set()
    wait_for(lambda: effect.ready and effect.method == "dereverb_denoiser")
    wait_for(lambda: len([c for c in api.calls if c[0] == "load"]) >= 2)
    time.sleep(0.1)
    assert [c[1] for c in api.calls if c[0] == "create"][-1] == "dereverb_denoiser"
    assert "dereverb" not in [c[1] for c in api.calls if c[0] == "create"]  # the middle one was superseded


@pytest.mark.parametrize(
    "step",
    ["create", "set_u32", "set_float", "set_string", "load", "get_u32"],
)
def test_a_failing_step_ends_in_failure_and_the_audio_passes_through(step: str) -> None:
    rt, _api = fake(fail={step: 3})
    effect = nvafx.Effect(rt, "denoiser", 1.0, True)
    wait_for(lambda: effect.failed)
    assert not effect.ready
    frame = np.full(480, 0.5, np.float32)
    assert effect.run(frame) is False and np.all(frame == 0.5)


def test_a_wrong_channel_count_or_frame_size_is_a_failure() -> None:
    # :366-368 and :380-383: the code is written for one channel and 480 samples
    for attr, value in (("channels", 2), ("frame", 960)):
        rt, api = fake()
        setattr(api, attr, value)
        effect = nvafx.Effect(rt, "denoiser", 1.0, True)
        wait_for(lambda effect=effect: effect.failed)
        assert not effect.ready


def test_run_status_failed_stops_processing_and_other_errors_leave_the_frame() -> None:
    rt, api = fake()
    effect = nvafx.Effect(rt, "denoiser", 1.0, True)
    wait_for(lambda: effect.ready)
    frame = np.full(480, 0.5, np.float32)
    api.run_status = 5  # any other error: logged, the frame kept
    assert effect.run(frame) is False and np.all(frame == 0.5) and effect.ready
    api.run_status = nvafx.STATUS_FAILED  # :663-667: stop all processing until the next load
    assert effect.run(frame) is False and np.all(frame == 0.5)
    assert not effect.ready and not effect.failed
    api.run_status = 0
    assert effect.run(frame) is False  # still stopped
    effect.set("dereverb", 1.0, True)  # a new load restarts it
    wait_for(lambda: effect.ready)
    assert effect.run(frame) is True


def test_closing_destroys_the_effect_and_the_logger() -> None:
    rt, api = fake()
    effect = nvafx.Effect(rt, "denoiser", 1.0, True)
    wait_for(lambda: effect.ready)
    effect.close()  # nvidia_audio_destroy :128-173
    wait_for(lambda: api.destroyed == [101] and api.logger == 0)
    assert not effect.ready
    effect.set("dereverb", 1.0, True)  # nothing more loads
    time.sleep(0.05)
    assert api.names().count("create") == 1


def test_closing_while_loading_destroys_what_loads() -> None:
    rt, api = fake()
    api.load_gate.clear()
    effect = nvafx.Effect(rt, "denoiser", 1.0, True)
    wait_for(lambda: effect._lock.locked())
    effect.close()
    api.load_gate.set()
    wait_for(lambda: api.destroyed == [101] and api.logger == 0)


# ------------------------------------------------------------------ the filter, on a fake runtime


def test_the_filter_passes_audio_through_while_the_effect_loads_then_runs_it(runtime: Any) -> None:
    rt, api = fake()
    api.load_gate.clear()
    runtime(rt)
    f = make_nv(None, block=BLOCK)
    x = (np.random.default_rng(1).standard_normal(BLOCK * 20) * 0.1).astype(np.float32)
    assert np.array_equal(blocks(f, x[: BLOCK * 5]), x[: BLOCK * 5])  # loading: unchanged, no delay
    api.load_gate.set()
    wait_for(lambda: f._effect is not None and f._effect.ready)
    y = blocks(f, x[: BLOCK * 5])
    assert np.allclose(y, x[: BLOCK * 5] * 0.5)  # the fake halves each frame
    assert f.latency == 0


def test_a_block_that_is_not_whole_frames_gets_the_queue_and_its_ten_ms(runtime: Any) -> None:
    rt, _api = fake(gain=1.0)
    runtime(rt)
    f = make_nv(None, block=None)
    wait_for(lambda: f._effect is not None and f._effect.ready)
    x = np.random.default_rng(2).standard_normal(6000).astype(np.float32) * 0.1
    y = blocks(f, x, 1000)
    assert f.latency == 480 and len(y) == len(x)  # OBS's queue: 10 ms (:41, :777-778)
    assert np.array_equal(y[480:], x[:-480])
    assert not np.any(y[:480])


def test_the_filter_follows_setting_changes(runtime: Any) -> None:
    rt, api = fake()
    runtime(rt)
    f = make_nv({"method": "dereverb", "intensity": 0.5}, block=BLOCK)
    wait_for(lambda: f._effect is not None and f._effect.ready)
    assert ("create", "dereverb") in api.calls and ("set_float", "intensity_ratio", 0.5) in api.calls
    f.update({"method": "dereverb", "intensity": 0.9})
    f.process(np.zeros(BLOCK, np.float32))
    assert ("set_float", "intensity_ratio", 0.9) in api.calls
    f.update({"method": "denoiser", "intensity": 0.9})
    wait_for(lambda: ("create", "denoiser") in api.calls)


def test_a_failed_filter_still_passes_audio_through(runtime: Any) -> None:
    rt, _api = fake(fail={"load": 8})
    runtime(rt)
    f = make_nv(None, block=BLOCK)
    wait_for(lambda: f.failed)
    x = np.random.default_rng(3).standard_normal(BLOCK * 4).astype(np.float32)
    assert np.array_equal(blocks(f, x), x)


def test_the_audio_side_never_waits_for_the_runtime_check(runtime: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvafx, "_probed", False)
    monkeypatch.setattr(nvafx, "_probe_started", True)  # the check is running somewhere: never start another
    monkeypatch.setattr(nvafx, "_probe_lock", threading.Lock())
    nvafx._probe_lock.acquire()  # and it holds the lock, as a slow DLL load would
    try:
        start = time.perf_counter()
        f = make_nv(None, block=BLOCK)
        x = np.full(BLOCK, 0.25, np.float32)
        assert np.array_equal(f.process(x), x)
        assert time.perf_counter() - start < 5.0  # the lock is held by the test, so a wait would be for ever
    finally:
        nvafx._probe_lock.release()
    rt, _api = fake()
    monkeypatch.setattr(nvafx, "_runtime", rt)
    monkeypatch.setattr(nvafx, "_probed", True)
    f.process(x)  # the check finished: the next block makes the effect
    wait_for(lambda: f._effect is not None and f._effect.ready)


# ------------------------------------------------------------------ saving, loading, and no runtime


def nvidia_filters() -> list[FilterSpec]:
    filters: list[FilterSpec] = []
    for settings in ({}, {"method": "dereverb_denoiser", "intensity": 0.35, "vad": False}):
        filters.append(new_filter(NVIDIA_KIND, filters, settings, enabled=len(filters) == 0))
    return filters


def test_save_and_restore_round_trip(tmp_path: Path) -> None:
    filters = nvidia_filters()
    p = Prefs(settings=Settings(fine_tune=True, filters=tuple(filters)))
    path = tmp_path / "settings.json"
    prefs.save(path, p)
    back = prefs.load(path)
    assert back == p
    assert [f.name for f in back.settings.filters] == ["NVIDIA Audio Effects", "NVIDIA Audio Effects 2"]
    assert dict(back.settings.filters[1].settings) == {"method": "dereverb_denoiser", "intensity": 0.35, "vad": False}
    assert [f.enabled for f in back.settings.filters] == [True, False]
    saved = fine_tune.filters_to_json(filters)
    assert saved[0] == {
        "id": "nvidia_audiofx_filter",
        "name": "NVIDIA Audio Effects",
        "enabled": True,
        "settings": {"method": "denoiser", "intensity": 1.0, "vad": True},
    }


def test_a_saved_filter_loads_and_keeps_its_place_with_no_runtime(runtime: Any, tmp_path: Path) -> None:
    # OBS keeps a filter whose module is missing in the list, saved as it was, and does not run it
    # (libobs/obs-source.c filter_async_audio :3917); here the audio passes through unchanged.
    runtime(None)
    filters = [new_filter("gain_filter", (), {"db": 6.0}), *nvidia_filters()]
    path = tmp_path / "settings.json"
    prefs.save(path, Prefs(settings=Settings(fine_tune=True, filters=tuple(filters))))
    back = prefs.load(path).settings
    assert [f.kind for f in back.filters] == ["gain_filter", NVIDIA_KIND, NVIDIA_KIND]
    ft = FineTune(back, BLOCK)
    assert isinstance(ft.instances[back.filters[1].uid], obs_filters.NvidiaAudioEffects)
    x = (np.random.default_rng(4).standard_normal(BLOCK * 30) * 0.05).astype(np.float32)
    one_gain = FineTune(Settings(fine_tune=True, filters=(back.filters[0],)), BLOCK)
    assert np.array_equal(blocks(ft, x), blocks(one_gain, x))  # the gain runs; the NVIDIA filters change nothing
    again = tmp_path / "again.json"
    prefs.save(again, prefs.load(path))
    assert prefs.load(again).settings.filters[2].settings == back.filters[2].settings  # kept, not dropped


def test_a_missing_runtime_filter_in_the_window_has_a_note(make_window: Any, qtbot: QtBot) -> None:
    filters = tuple(nvidia_filters())
    window, host = make_window(None, Settings(fine_tune=True, filters=filters))
    window.list.setCurrentRow(0)
    assert window.pane is not None and window.pane.controls == {}
    labels = [lb.text() for lb in window.pane.findChildren(type(window.pane.status)) if lb.text()]
    assert any("not found" in t for t in labels)
    window.model.set_enabled(filters[0].uid, False)  # the eye still works
    assert host.settings.filters[0].enabled is False
    window.model.reset(filters[1].uid)  # so does Defaults
    assert dict(host.settings.filters[1].settings) == fine_tune.defaults(NVIDIA_KIND)


def test_the_eye_reorder_and_defaults_work_with_the_runtime(make_window: Any) -> None:
    window, host = make_window(STAND_IN)
    window.add_filter(NVIDIA_KIND)
    window.add_filter("gain_filter")
    nv, gain = host.settings.filters
    window.model.set_setting(nv.uid, "intensity", 0.2)
    window.model.set_setting(nv.uid, "method", "dereverb")
    assert dict(host.settings.filters[0].settings) == {"method": "dereverb", "intensity": 0.2, "vad": True}
    window.model.set_enabled(nv.uid, False)
    assert host.settings.filters[0].enabled is False
    window.model.move(nv.uid, 1)
    assert [f.kind for f in host.settings.filters] == ["gain_filter", NVIDIA_KIND]
    window.model.reset(nv.uid)
    assert dict(host.settings.filters[1].settings) == {"method": "denoiser", "intensity": 1.0, "vad": True}
    assert gain.uid == host.settings.filters[0].uid


# ------------------------------------------------------------------ the real runtime, where it is installed

REAL = nvafx.runtime() if sys.platform == "win32" else None
needs_runtime = pytest.mark.skipif(REAL is None, reason="NVIDIA's audio effects runtime is not installed here")


def run_real(method: str, x: np.ndarray, intensity: float = 1.0) -> tuple[np.ndarray, obs_filters.NvidiaAudioEffects]:
    f = make_nv({"method": method, "intensity": intensity, "vad": True}, block=BLOCK)
    wait_for(lambda: f.failed or (f._effect is not None and f._effect.ready), 60)
    assert not f.failed
    n = len(x) // BLOCK * BLOCK
    return blocks(f, x[:n]), f


@needs_runtime
def test_real_runtime_is_a_usable_version() -> None:
    assert REAL is not None and REAL.api is not None
    print(f"\nNVIDIA Audio Effects runtime: {REAL.sdk_dir}, version {nvafx.version_text(REAL.version)}")
    assert REAL.version == 0 or REAL.version >= nvafx.MIN_SDK_VERSION


@needs_runtime
@pytest.mark.parametrize("method", ["denoiser", "dereverb", "dereverb_denoiser"])
def test_real_effect_on_speech_and_noisy_speech(method: str) -> None:
    clean = read_wav(SPEECH).astype(np.float32)
    mixed, clean_level, spans = mix(level_db=-26.0, snr_db=6.0, seed=3, noise_name=FAN)
    gaps = pauses(spans, len(mixed), 0.4)
    assert gaps
    # clean speech through the effect: no dropout, nothing broken
    y_clean, _ = run_real(method, clean)
    q = np.round(y_clean.astype(np.float64) * (1 << 23)) / (1 << 23)
    assert np.all(np.isfinite(y_clean)) and np.max(np.abs(y_clean)) > 0.01
    assert [f for f in audio_check.analyze(q, RATE, read_wav(SPEECH)) if f.kind in DROPOUTS] == []
    # noisy speech: how far the noise between the words comes down, and what the words keep
    y, _ = run_real(method, mixed)
    skip = RATE // 10  # past the effect's own settling
    in_noise = np.mean([np.mean(mixed[s + skip : e].astype(np.float64) ** 2) for s, e in gaps])
    out_noise = np.mean([np.mean(y[s + skip : e].astype(np.float64) ** 2) for s, e in gaps])
    reduction = 10 * np.log10(in_noise / max(out_noise, 1e-20))
    words_in, words_out = active_rms_db(mixed, spans), active_rms_db(y, spans)
    q2 = np.round(y.astype(np.float64) * (1 << 23)) / (1 << 23)
    faults = [f for f in audio_check.analyze(q2, RATE, clean_level.astype(np.float64)) if f.kind in DROPOUTS]
    print(
        f"\n{method}: noise between words {reduction:.1f} dB lower; words {words_out - words_in:+.1f} dB; "
        f"dropout faults on clean speech: 0, on noisy speech: {len(faults)}"
    )
    assert faults == []
    if method != "dereverb":  # room echo removal alone is not a noise remover
        assert reduction > 6.0
    assert words_out - words_in > -9.0  # the voice stays


@needs_runtime
def test_real_effect_follows_intensity_and_a_method_change() -> None:
    mixed, _, spans = mix(level_db=-26.0, snr_db=6.0, seed=5, noise_name=FAN)
    gaps = pauses(spans, len(mixed), 0.4)
    skip = RATE // 10

    n = len(mixed) // BLOCK * BLOCK
    half = n // 2 // BLOCK * BLOCK
    late = [(max(s, half) + skip, e) for s, e in gaps if e > half + 2 * skip]  # the pauses after the change

    def noise_db(y: np.ndarray) -> float:
        return float(10 * np.log10(np.mean([np.mean(y[s:e].astype(np.float64) ** 2) for s, e in late])))

    def run_from(f: obs_filters.NvidiaAudioEffects, start: int, end: int) -> np.ndarray:
        return np.concatenate([f.process(mixed[i : i + BLOCK]) for i in range(start, end, BLOCK)])

    def ready(f: obs_filters.NvidiaAudioEffects) -> None:
        wait_for(lambda: f.failed or (f._effect is not None and f._effect.ready), 60)
        assert not f.failed

    # The intensity the effect is created with sets how much it removes: 1.0 most, 0.0 none.
    levels = {}
    for intensity in (1.0, 0.5, 0.0):
        f = make_nv({"intensity": intensity})
        ready(f)
        levels[intensity] = noise_db(run_from(f, 0, n))
        f.close()
    print(f"\nnoise between words, by intensity: {levels} (input {noise_db(mixed):.1f} dB)")
    assert levels[1.0] < levels[0.5] - 3.0 and levels[0.5] < levels[0.0] - 3.0
    assert levels[0.0] == pytest.approx(noise_db(mixed), abs=1.0)  # intensity 0: the noise stays
    # A change while it runs is OBS's set call (:472-485): it succeeds and the audio runs on. (SDK 1.6.1.2 reads the
    # intensity at Load: the same call changes nothing audible there, in OBS too.)
    f = make_nv({"intensity": 1.0})
    ready(f)
    run_from(f, 0, half)
    f.update({"method": "denoiser", "intensity": 0.2, "vad": False})
    assert np.all(np.isfinite(run_from(f, half, n))) and not f.failed and f._effect is not None and f._effect.ready
    # A change of method loads a new effect: the audio passes through meanwhile, then runs the new model.
    f.update({"method": "dereverb_denoiser", "intensity": 1.0, "vad": True})
    ready(f)
    f.process(mixed[:BLOCK])
    again = run_from(f, 0, n)
    assert noise_db(again) < noise_db(mixed) - 6.0
    f.close()
