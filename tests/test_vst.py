"""The VST 2.x host core (J10 plan §5, Tests T18-T22), against the in-test fake plug-in of ``vst_fake``."""

from __future__ import annotations

import base64
import ctypes
import dataclasses
import hashlib
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from vst_fake import FakeLibrary, FakePlugin, load_fake

from clean_voice import fine_tune, vst
from clean_voice.dsp import Chain, FineTune
from clean_voice.fine_tune import FilterSpec
from clean_voice.settings import Settings, SettingsCell, replace

BLOCK = 480

# ------------------------------------------------------------------ the ABI


def test_the_aeffect_layout_is_vst2sdks_on_64_bit() -> None:
    # vst2sdk v0.4.0 include/vst.h struct vst_effect_t :2956-3120 under #pragma pack(push, 8).
    offsets = {name: getattr(vst.AEffect, name).offset for name, *_ in vst.AEffect._fields_}
    assert offsets["magic"] == 0
    assert offsets["dispatcher"] == 8
    assert offsets["num_programs"] == 40
    assert offsets["flags"] == 56
    assert offsets["delay"] == 80
    assert offsets["effect_internal"] == 96
    assert offsets["host_internal"] == 104
    assert offsets["unique_id"] == 112
    assert offsets["process_replacing"] == 120
    assert offsets["process_double"] == 128
    assert ctypes.sizeof(vst.AEffect) == 192
    assert ctypes.sizeof(vst.Rect) == 8
    assert vst.MAGIC == int.from_bytes(b"VstP", "big")


def test_the_plugin_runs_without_the_gil() -> None:
    # CFUNCTYPE (not PYFUNCTYPE): ctypes releases the GIL for the whole call into the plug-in. The timing half of this
    # check runs against the compiled test plug-in (test_vst_plugin.py), whose processing is native code.
    for proto in (vst.ProcessFn, vst.Dispatcher, vst.HostCallback, vst.EntryFn):
        assert not proto._flags_ & ctypes._FUNCFLAG_PYTHONAPI  # type: ignore[attr-defined]


# ------------------------------------------------------------------ T18 scan


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MZ")
    return path


def test_t18_windows_scan_lists_dlls_from_obs_six_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pf, common = tmp_path / "Program Files", tmp_path / "Common Files"
    monkeypatch.setenv("ProgramFiles", str(pf))
    monkeypatch.setenv("CommonProgramFiles", str(common))
    folders = vst.search_folders("win32")
    assert folders == [
        Path(f"{pf}/Steinberg/VstPlugins/"),
        Path(f"{common}/Steinberg/Shared Components/"),
        Path(f"{common}/VST2"),
        Path(f"{common}/Steinberg/VST2"),
        Path(f"{common}/VSTPlugins/"),
        Path(f"{pf}/VSTPlugins/"),
    ]
    touch(pf / "Steinberg" / "VstPlugins" / "zeta.dll")
    touch(pf / "Steinberg" / "VstPlugins" / "Deep" / "er" / "Alpha.DLL")  # recursive, any case
    touch(common / "VST2" / "beta.dll")
    touch(common / "Steinberg" / "VST2" / "my.dll.tool.dll")  # OBS removes every ".dll"
    touch(common / "VSTPlugins" / "notes.txt")
    touch(common / "VSTPlugins" / ".hidden.dll")
    touch(pf / "VSTPlugins" / "gamma.dll")
    touch(tmp_path / "elsewhere" / "outside.dll")  # not one of the six
    names = [name for name, _ in vst.scan(platform="win32")]
    assert names == ["Alpha", "beta", "gamma", "my.tool", "zeta"]  # case-sensitive: capitals first
    paths = dict(vst.scan(platform="win32"))
    assert Path(paths["Alpha"]) == pf / "Steinberg" / "VstPlugins" / "Deep" / "er" / "Alpha.DLL"


def test_t18_mac_scan_lists_vst_bundles_with_home_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(vst, "MAC_SYSTEM_FOLDER", str(tmp_path / "Library" / "Audio" / "Plug-Ins" / "VST"))
    assert vst.search_folders("darwin") == [
        tmp_path / "Library" / "Audio" / "Plug-Ins" / "VST",
        tmp_path / "home" / "Library" / "Audio" / "Plug-ins" / "VST",
    ]
    touch(tmp_path / "Library" / "Audio" / "Plug-Ins" / "VST" / "Sys.vst" / "Contents" / "Info.plist")
    touch(tmp_path / "home" / "Library" / "Audio" / "Plug-ins" / "VST" / "sub" / "user.vst" / "Contents" / "x")
    touch(tmp_path / "home" / "Library" / "Audio" / "Plug-ins" / "VST" / "skip.component" / "Contents" / "x")
    assert [n for n, _ in vst.scan(platform="darwin")] == ["Sys", "user"]


def test_t18_scanning_loads_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    touch(tmp_path / "VSTPlugins" / "a.dll")
    opened: list[str] = []
    monkeypatch.setattr(vst, "open_library", lambda p: opened.append(p))
    assert len(vst.scan(platform="win32")) == 1 and opened == []


# ------------------------------------------------------------------ T19 loading


LOAD_ORDER = [vst.EFF_IDENTIFY, vst.EFF_OPEN, vst.EFF_SET_SAMPLE_RATE, vst.EFF_SET_BLOCK_SIZE, vst.EFF_MAINS_CHANGED]


def test_t19_loading_follows_obs_order() -> None:
    fake = FakePlugin()
    plugin = load_fake(fake)
    # Names first (VSTPlugin.cpp:177-178), then the start-up calls of :187-207.
    assert fake.opcodes == [vst.EFF_GET_EFFECT_NAME, vst.EFF_GET_VENDOR_STRING, *LOAD_ORDER]
    by_op = {c[0]: c for c in fake.calls}
    assert by_op[vst.EFF_SET_SAMPLE_RATE][3] == 48000.0
    assert by_op[vst.EFF_SET_BLOCK_SIZE][2] == 512
    assert by_op[vst.EFF_MAINS_CHANGED][2] == 1
    assert (plugin.effect_name, plugin.vendor) == ("Fake Gain", "Tests Inc")
    plugin.close()


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"magic": 0x1234}, vst.BAD_MAGIC),
        ({"flags": vst.FLAG_CAN_REPLACING | vst.FLAG_IS_SYNTH}, vst.IS_SYNTH),
        ({"flags": 0}, vst.NO_REPLACING),
        ({"inputs": 300}, vst.BAD_CHANNELS),
    ],
)
def test_t19_refused_plugins_say_why_and_are_unloaded(kwargs: dict, reason: str) -> None:
    fake = FakePlugin(**kwargs)
    library = FakeLibrary({"VSTPluginMain": fake})
    with pytest.raises(vst.VstError) as err:
        vst.load("x.dll", lambda p: library, platform="win32")
    assert err.value.reason == reason
    assert library.freed == 1
    if reason == vst.BAD_MAGIC:
        assert fake.calls == []  # an untrustworthy struct is never called
    else:
        assert fake.opcodes[-2:] == [vst.EFF_MAINS_CHANGED, vst.EFF_CLOSE]  # OBS's unloadEffect
        assert vst.EFF_OPEN not in fake.opcodes


@pytest.mark.parametrize(
    ("platform", "entry"),
    [
        ("win32", "VSTPluginMain"),
        ("win32", "VstPluginMain()"),
        ("win32", "main"),
        ("darwin", "VSTPluginMain"),
        ("darwin", "main_macho"),
    ],
)
def test_t19_entry_point_fallbacks(platform: str, entry: str) -> None:
    fake = FakePlugin()
    plugin = vst.load("p", lambda p: FakeLibrary({entry: fake}), platform=platform)
    assert vst.EFF_OPEN in fake.opcodes
    plugin.close()


def test_t19_no_entry_point_or_no_effect() -> None:
    with pytest.raises(vst.VstError) as err:
        vst.load("p", lambda p: FakeLibrary({"main_macho": FakePlugin()}), platform="win32")
    assert err.value.reason == vst.NO_ENTRY_POINT

    class Null(FakePlugin):
        def _main(self, host: object) -> int:
            return 0

    with pytest.raises(vst.VstError) as err:
        vst.load("p", lambda p: FakeLibrary({"VSTPluginMain": Null()}), platform="win32")
    assert err.value.reason == vst.NO_EFFECT


@pytest.mark.skipif(not vst.sys.platform.startswith("win"), reason="LoadLibraryW")
def test_t19_a_wrong_architecture_dll_says_so(tmp_path: Path) -> None:
    # A PE file for another machine: Windows answers ERROR_BAD_EXE_FORMAT, as for a 32-bit plug-in.
    bad = tmp_path / "old.dll"
    pe = bytearray(512)
    pe[0:2] = b"MZ"
    pe[0x3C:0x40] = (0x80).to_bytes(4, "little")
    pe[0x80:0x84] = b"PE\0\0"
    pe[0x84:0x86] = (0x014C).to_bytes(2, "little")  # IMAGE_FILE_MACHINE_I386
    bad.write_bytes(bytes(pe))
    with pytest.raises(vst.VstError) as err:
        vst.load(str(bad))
    assert err.value.reason == vst.WRONG_ARCHITECTURE
    with pytest.raises(vst.VstError) as err:
        vst.load(str(tmp_path / "missing.dll"))
    assert err.value.reason.startswith("Windows couldn't open the file")


def test_the_host_callback_answers_as_obs() -> None:
    fake = FakePlugin()
    plugin = load_fake(fake)
    host = fake.host
    assert host is not None
    eff = ctypes.pointer(fake.effect)
    assert host(eff, vst.HOST_VERSION, 0, 0, None, 0.0) == 2400
    assert host(eff, vst.HOST_PROCESS_LEVEL, 0, 0, None, 0.0) == 1
    assert host(eff, vst.HOST_WILL_REPLACE, 0, 0, None, 0.0) == 1
    assert host(eff, vst.HOST_GET_SAMPLE_RATE, 0, 0, None, 0.0) == 48000
    assert host(None, vst.HOST_GET_SAMPLE_RATE, 0, 0, None, 0.0) == 0
    buf = ctypes.create_string_buffer(64)
    assert host(eff, vst.HOST_VENDOR_STRING, 0, 0, ctypes.addressof(buf), 0.0) == 1
    assert buf.value == b"Clean Voice"
    assert host(None, vst.HOST_GET_TIME, 0, 0, None, 0.0) == 0  # VSTPlugin.cpp:54-57: no plug-in, no time info
    assert host(eff, 999, 0, 0, None, 0.0) == 0
    sizes: list[tuple[int, int]] = []

    class Editor:
        def request_resize(self, width: int, height: int) -> None:
            sizes.append((width, height))

        def close(self) -> bool:
            return True

    assert host(eff, vst.HOST_SIZE_WINDOW, 640, 480, None, 0.0) == 1  # no editor: nothing to resize
    plugin.editor = Editor()
    assert host(eff, vst.HOST_SIZE_WINDOW, 640, 480, None, 0.0) == 1
    assert sizes == [(640, 480)]
    plugin.close()


def test_the_time_info_layout_is_the_vst_2_4_binary_interface() -> None:
    # Eight doubles, then six 32-bit ints: 88 bytes, with tempo at byte 32 and flags at byte 84.
    offsets = {name: getattr(vst.TimeInfo, name).offset for name, *_ in vst.TimeInfo._fields_}
    assert offsets == {
        "sample_pos": 0,
        "sample_rate": 8,
        "nano_seconds": 16,
        "ppq_pos": 24,
        "tempo": 32,
        "bar_start_pos": 40,
        "cycle_start_pos": 48,
        "cycle_end_pos": 56,
        "time_sig_numerator": 64,
        "time_sig_denominator": 68,
        "smpte_offset": 72,
        "smpte_frame_rate": 76,
        "samples_to_next_clock": 80,
        "flags": 84,
    }
    assert ctypes.sizeof(vst.TimeInfo) == 88
    assert (vst.TIME_TRANSPORT_PLAYING, vst.TIME_NANOS_VALID, vst.TIME_TEMPO_VALID) == (1 << 1, 1 << 8, 1 << 10)


def test_get_time_answers_obss_fixed_transport() -> None:
    """VSTPlugin.cpp:53-57 and :194-201: a pointer to the plug-in's own time info, 120 BPM in 4/4 at the sample rate,
    tempo, nanoseconds and playing flagged, sample position 0; only the clock moves (GetTimeInfo :71-75)."""
    fake = FakePlugin()
    plugin = load_fake(fake)
    host = fake.host
    assert host is not None
    eff = ctypes.pointer(fake.effect)
    address = host(eff, vst.HOST_GET_TIME, 0, 0, None, 0.0)
    assert address != 0
    info = vst.TimeInfo.from_address(address)
    assert info.tempo == 120.0
    assert (info.time_sig_numerator, info.time_sig_denominator) == (4, 4)
    assert info.sample_rate == 48000.0
    assert info.sample_pos == 0.0
    assert info.flags == vst.TIME_TEMPO_VALID | vst.TIME_NANOS_VALID | vst.TIME_TRANSPORT_PLAYING
    first = info.nano_seconds
    assert first > 0.0
    time.sleep(0.02)
    assert host(eff, vst.HOST_GET_TIME, 0, 0, None, 0.0) == address  # the same block every time: never reallocated
    assert info.nano_seconds > first  # OBS's clock, os_gettime_ns() / 1000000 (VSTPlugin.cpp:73)
    plugin.close()


def test_get_time_from_the_audio_thread_reads_the_tempo() -> None:
    """A plug-in that asks for the time inside processReplacing, as tempo-synced effects do, reads 120 BPM there."""
    fake = FakePlugin()
    seen: list[float] = []

    def run(effect: object, ins: object, outs: object, frames: int) -> None:
        assert fake.host is not None
        address = fake.host(effect, vst.HOST_GET_TIME, 0, 0, None, 0.0)
        seen.append(vst.TimeInfo.from_address(address).tempo if address else 0.0)

    fake._process = vst.ProcessFn(run)
    fake.effect.process_replacing = fake._process
    plugin = load_fake(fake)
    worker = threading.Thread(target=lambda: [plugin.process(np.zeros(BLOCK, np.float32)) for _ in range(5)])
    worker.start()
    worker.join()
    assert seen == [120.0] * 5
    plugin.close()


# ------------------------------------------------------------------ T20 processing


def test_t20_mono_to_every_input_and_the_mean_of_two_outputs() -> None:
    fake = FakePlugin(inputs=2, outputs=2)
    fake.params[0] = 0.5
    fake.keep_outputs = True
    plugin = load_fake(fake)
    x = np.linspace(-1, 1, BLOCK, dtype=np.float32)
    y = plugin.process(x)
    assert np.array_equal(fake.inputs_seen[0][0], x) and np.array_equal(fake.inputs_seen[0][1], x)
    # output 0 = 0.5 x, output 1 = 2 x 0.5 x: the mean is 0.75 x
    np.testing.assert_allclose(y, 0.75 * x, rtol=1e-6)
    assert not np.any(fake.seen_outputs[0])  # zeroed before the call
    plugin.process(x * 0)  # the previous block's output never shows through
    assert not np.any(fake.seen_outputs[1])
    plugin.close()


@pytest.mark.parametrize(("outputs", "factor"), [(1, 0.5), (0, 1.0), (3, 0.75)])
def test_t20_other_output_counts(outputs: int, factor: float) -> None:
    fake = FakePlugin(inputs=1, outputs=outputs)
    fake.params[0] = 0.5
    plugin = load_fake(fake)
    x = np.linspace(-1, 1, BLOCK, dtype=np.float32)
    np.testing.assert_allclose(plugin.process(x), factor * x, rtol=1e-6)
    plugin.close()


def test_t20_blocks_longer_than_512_go_in_passes() -> None:
    fake = FakePlugin()
    fake.keep_outputs = True
    plugin = load_fake(fake)
    x = np.arange(1200, dtype=np.float32)
    y = plugin.process(x)
    assert [s.shape[1] for s in fake.seen_outputs] == [512, 512, 176]
    np.testing.assert_allclose(y, 1.5 * x)
    plugin.close()


def test_t20_no_buffers_are_made_per_block() -> None:
    fake = FakePlugin()
    plugin = load_fake(fake)
    names = ("_ins", "_outs", "_in_ptrs", "_out_ptrs", "_process", "_effect")
    before = {n: id(getattr(plugin, n)) for n in names}
    x = np.ones(BLOCK, np.float32)
    for _ in range(1000):
        plugin.process(x)
    assert {n: id(getattr(plugin, n)) for n in names} == before
    assert fake.processed == 1000
    plugin.close()


def test_t20_a_closed_plugin_passes_audio_through() -> None:
    fake = FakePlugin()
    plugin = load_fake(fake)
    plugin.close()
    x = np.ones(BLOCK, np.float32)
    assert np.array_equal(plugin.process(x), x) and fake.after_close == 0


# ------------------------------------------------------------------ T21 state


def test_t21_chunk_plugins_round_trip_through_index_1_in_base64() -> None:
    fake = FakePlugin(flags=vst.FLAG_CAN_REPLACING | vst.FLAG_PROGRAM_CHUNKS)
    fake.params = [0.3, 0.7]
    plugin = load_fake(fake)
    data = plugin.get_chunk()
    assert base64.b64decode(data) == np.array([0.3, 0.7], np.float32).tobytes()
    assert [c[1] for c in fake.calls if c[0] == vst.EFF_GET_CHUNK] == [1]
    other = FakePlugin(flags=vst.FLAG_CAN_REPLACING | vst.FLAG_PROGRAM_CHUNKS)
    second = load_fake(other)
    assert second.set_chunk(data)
    assert [(c[1], c[2]) for c in other.calls if c[0] == vst.EFF_SET_CHUNK] == [(1, 8)]
    assert other.params == pytest.approx([0.3, 0.7])
    plugin.close()
    second.close()


def test_t21_parameter_plugins_round_trip_every_float() -> None:
    fake = FakePlugin(params=3)
    fake.params = [0.1, 0.2, 0.9]
    plugin = load_fake(fake)
    data = plugin.get_chunk()
    assert np.frombuffer(base64.b64decode(data), np.float32) == pytest.approx([0.1, 0.2, 0.9])
    other = FakePlugin(params=3)
    second = load_fake(other)
    assert second.set_chunk(data) and other.params == pytest.approx([0.1, 0.2, 0.9])
    wrong = FakePlugin(params=2)  # a different count: nothing is restored (VSTPlugin.cpp:400-402)
    third = load_fake(wrong)
    assert not third.set_chunk(data) and wrong.params == [1.0, 0.25]
    assert not third.set_chunk("%%% not base64")
    for p in (plugin, second, third):
        p.close()


def test_t21_a_hash_mismatch_skips_the_restore(tmp_path: Path) -> None:
    binary = tmp_path / "gain.dll"
    binary.write_bytes(b"version 1")
    digest = hashlib.md5(b"version 1").hexdigest()
    assert vst.file_md5(str(binary)) == digest
    assert vst.file_md5(str(tmp_path)) == ""  # a folder (a macOS bundle) has no hash, as QFile can't open it
    data = base64.b64encode(np.array([0.5, 0.5], np.float32).tobytes()).decode()
    for saved_hash, restored in ((digest, True), ("", True), ("0" * 32, False)):
        fake = FakePlugin()
        plugin = load_fake(fake, path=str(binary))
        assert vst.restore(plugin, data, saved_hash) is restored
        assert (fake.params == [0.5, 0.5]) is restored
        plugin.close()
    fake = FakePlugin()
    plugin = load_fake(fake, path=str(binary))
    assert vst.saved_state(plugin) == {"chunk_data": plugin.get_chunk(), "chunk_hash": digest}
    plugin.close()


def test_create_follows_vst_create_and_update(tmp_path: Path) -> None:
    fake = FakePlugin()
    library = FakeLibrary({"VSTPluginMain": fake})
    data = base64.b64encode(np.array([0.5, 0.25], np.float32).tobytes()).decode()
    settings = fine_tune.resolve(
        "vst_filter", {"plugin_path": "a.dll", "chunk_data": data, "open_when_active_vst_settings": True}
    )
    assert vst.create(fine_tune.resolve("vst_filter", {}), lambda p: library) is None  # no plug-in chosen
    plugin = vst.create(settings, lambda p: library)
    assert plugin is not None and plugin.open_when_active and fake.params == [0.5, 0.25]
    plugin.close()


# ------------------------------------------------------------------ T22 the registry and the disposal handshake


def vst_spec(name: str = "VST 2.x Plug-in") -> FilterSpec:
    return FilterSpec("vst_filter", name, settings={"plugin_path": "fake.dll"})


def test_fine_tune_runs_a_plugin_from_the_registry() -> None:
    registry = vst.VstRegistry()
    fake = FakePlugin()
    fake.params[0] = 0.5
    spec = vst_spec()
    registry.add(spec.uid, load_fake(fake))
    ft = FineTune(Settings(fine_tune=True, filters=(spec,)), block=BLOCK, vst=registry)
    x = np.full(BLOCK, 0.2, np.float32)
    np.testing.assert_allclose(ft.process(x), 0.75 * x, rtol=1e-6)
    assert ft.held == {spec.uid}
    offline = FineTune(Settings(fine_tune=True, filters=(spec,)), block=BLOCK)  # no registry: passes through
    assert np.array_equal(offline.process(x), x)
    ft.close()
    registry.close_all()
    assert fake.closed


def test_t22_removing_unloads_only_after_the_chain_drops_it() -> None:
    registry = vst.VstRegistry()
    fake = FakePlugin()
    spec = vst_spec()
    plugin = load_fake(fake)
    registry.add(spec.uid, plugin)
    cell = SettingsCell(Settings(fine_tune=True, filters=(spec,)))
    chain = Chain(cell, block=BLOCK, vst=registry)
    x = np.full(BLOCK, 0.1, np.float32)
    chain.process(x)
    cell.value = replace(cell.value, filters=())
    registry.remove(spec.uid)
    assert not fake.closed and registry.pending == 1  # still in the chain's list
    assert registry.get(spec.uid) is None  # but no chain can take it any more
    chain.process(x)  # the section fades to its input: the plug-in still runs
    assert registry.collect() == 0 and not fake.closed
    chain.process(x)  # swapped out at this block's start
    assert spec.uid not in {f.uid for f in chain.fine_tune.applied.filters}
    assert registry.collect() == 1 and fake.closed
    # effMainsChanged(0), then effClose, then the library goes (VSTPlugin.cpp unloadEffect :279-286).
    assert fake.calls[-2:] == [(vst.EFF_MAINS_CHANGED, 0, 0, 0.0), (vst.EFF_CLOSE, 0, 0, 0.0)]
    assert plugin.fake_library.freed == 1  # type: ignore[attr-defined]
    assert fake.after_close == 0


def test_t22_with_no_chain_running_a_plugin_goes_at_once() -> None:
    registry = vst.VstRegistry()
    fake = FakePlugin()
    spec = vst_spec()
    registry.add(spec.uid, load_fake(fake))
    chain = Chain(SettingsCell(Settings(fine_tune=True, filters=(spec,))), block=BLOCK, vst=registry)
    chain.process(np.zeros(BLOCK, np.float32))
    chain.fine_tune.close()  # the engine closes a chain's Fine-tune when its streams close
    registry.remove(spec.uid)
    assert fake.closed
    other, spec2 = FakePlugin(), vst_spec()
    registry.add(spec2.uid, load_fake(other))
    Chain(SettingsCell(Settings(fine_tune=True, filters=(spec2,))), block=BLOCK, vst=registry)  # made, then gone
    registry.remove(spec2.uid)
    assert other.closed
    spec3 = vst_spec()
    registry.add(spec3.uid, load_fake(FakePlugin()))
    second = load_fake(FakePlugin())
    with pytest.raises(ValueError):
        registry.add(spec3.uid, second)  # one uid, one instance: a new plug-in needs a new uid
    second.close()
    registry.close_all()


def test_t22_a_new_plugin_for_a_filter_gets_a_new_uid() -> None:
    spec = vst_spec()
    swapped = dataclasses.replace(spec, uid=fine_tune.new_uid(), settings={"plugin_path": "other.dll"})
    assert swapped.uid != spec.uid and swapped == dataclasses.replace(spec, settings={"plugin_path": "other.dll"})


def test_t22_close_all_at_quit() -> None:
    registry = vst.VstRegistry()
    fakes = [FakePlugin() for _ in range(3)]
    specs = [vst_spec(f"V{i}") for i in range(3)]
    for f, s in zip(fakes, specs, strict=True):
        registry.add(s.uid, load_fake(f))
    registry.remove(specs[0].uid)
    registry.close_all()
    assert all(f.closed for f in fakes) and registry.pending == 0


def test_t22_a_thousand_remove_add_cycles_never_call_a_closed_plugin() -> None:
    """The worker runs the chain flat out on its own thread while the UI thread adds a plug-in, publishes it, publishes
    the list without it, removes it and collects, 1000 times. No plug-in is ever called after effClose, and every one
    is unloaded in the end."""
    registry = vst.VstRegistry()
    keep = FilterSpec("gain_filter", "Gain", settings={"db": -1.0})
    cell = SettingsCell(Settings(fine_tune=True, filters=(keep,)))
    chain = Chain(cell, block=BLOCK, vst=registry)
    stop = threading.Event()
    x = (0.1 * np.sin(np.arange(BLOCK) / 10)).astype(np.float32)
    blocks = [0]

    def worker() -> None:
        while not stop.is_set():
            chain.process(x)
            blocks[0] += 1

    thread = threading.Thread(target=worker)
    thread.start()
    fakes: list[FakePlugin] = []
    try:
        for i in range(1000):
            fake = FakePlugin()
            fakes.append(fake)
            spec = vst_spec(f"VST {i}")
            registry.add(spec.uid, load_fake(fake))
            cell.value = replace(cell.value, filters=(keep, spec))
            if i % 2 == 0:  # every other time, wait until the chain runs it: removed while it is in use
                deadline = time.monotonic() + 2.0
                while fake.processed == 0 and time.monotonic() < deadline:
                    time.sleep(0)
            cell.value = replace(cell.value, filters=(keep,))
            registry.remove(spec.uid)
            registry.collect()
    finally:
        stop.set()
        thread.join()
    assert blocks[0] > 100
    assert sum(f.processed > 0 for f in fakes) >= 500  # the chain did run them, and they were removed mid-use
    assert sum(f.after_close for f in fakes) == 0
    chain.process(x)
    chain.process(x)
    registry.collect()
    assert registry.pending == 0 and all(f.closed for f in fakes)


def test_engine_hands_its_registry_to_every_chain(make_engine) -> None:
    from test_engine import CABLE, MIC

    engine, _ = make_engine([MIC, CABLE], record=False)
    assert isinstance(engine.vst, vst.VstRegistry)
    engine.start(None)
    deadline = time.monotonic() + 5
    while engine._path is None and time.monotonic() < deadline:
        time.sleep(0.01)
    path = engine._path
    assert path is not None
    assert path.chain.fine_tune._vst is engine.vst  # type: ignore[attr-defined]
    engine.stop()
    deadline = time.monotonic() + 5
    while engine._path is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.chain.fine_tune.held == frozenset()  # type: ignore[attr-defined]  # closed with its streams
