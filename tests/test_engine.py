"""The engine's device handling with a fake sound system: cable missing, mic unplugged and back, clean stop."""

import sys
import threading
import time
import wave
from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pytest

import clean_voice.engine as engine_module
from clean_voice import devices
from clean_voice.audio_path import AudioPath
from clean_voice.devices import Device, PathFactory, find_cable, is_cable, microphones
from clean_voice.dsp import PassThrough
from clean_voice.engine import State

MIC = Device(1, "Microphone (USB Mic)", 1, 48_000.0)
LAPTOP = Device(2, "Microphone Array (Realtek)", 2, 48_000.0)
CABLE_REC = Device(3, "CABLE Output (VB-Audio Virtual Cable)", 2, 48_000.0)
CABLE = Device(4, "CABLE Input (VB-Audio Virtual Cable)", 2, 48_000.0)
SPEAKERS = Device(5, "Speakers (Realtek)", 2, 48_000.0)
HEADPHONES = Device(6, "Headphones (USB Headset)", 2, 48_000.0)
VOICEMEETER = Device(7, "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)", 2, 48_000.0)
OUTPUT_INDEXES = (4, 5, 6, 7)


class FakeOutput:
    """An output (or loopback) device stream: calls its callback every 10 ms while the device is plugged in."""

    def __init__(self, backend: "FakeBackend", device: Device, callback: Callable[..., None], kind: str) -> None:
        self.backend = backend
        self.device = device
        self.callback = callback
        self.kind = kind  # "output" or "loopback"
        self.closed = False
        self.calls = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        out = np.zeros((480, 2), np.float32)
        block = np.zeros((480, 2), np.float32)
        while not self.closed:
            if self.device in self.backend.present and not (self.kind == "loopback" and self.backend.loopback_stalled):
                if self.kind == "output":
                    self.callback(out, 480, None, None)
                    self.backend.played.setdefault(self.device.name, []).append(out[:, 0].copy())
                else:
                    block[:] = self.backend.loopback_value
                    self.callback(block, 480, None, None)
                self.calls += 1
            time.sleep(0.01)

    @property
    def active(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True
        self._thread.join(timeout=5)


class FakeStreams:
    """Feeds the path like two running devices until one is 'unplugged'."""

    def __init__(self, backend: "FakeBackend", mic: Device, path: AudioPath, output: Device = CABLE) -> None:
        self.backend = backend
        self.mic = mic
        self.output = output  # where the clean voice goes (the cable unless the guest chose another output)
        self.path = path
        self.closed = False
        self.output_closed = False  # only the cable's stream was closed (close_output)
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        block = np.full((480, self.mic.channels), self.backend.mic_value, np.float32)
        out = np.zeros((480, 2), np.float32)
        while not self.closed:
            if self.mic in self.backend.present:
                self.path.input_callback(block, 480, None, None)
            started = time.monotonic() - self._t0 >= self.backend.cable_start_delay_s  # a slow device start
            if self.output in self.backend.present and not self.output_closed and started:
                self.path.output_callback(out, 480, None, self.backend.output_status)
            time.sleep(0.01)

    def close_output(self) -> None:
        self.output_closed = True

    @property
    def input_active(self) -> bool:
        return not self.closed

    @property
    def output_active(self) -> bool:
        return not self.closed and not self.backend.output_inactive

    @property
    def active(self) -> bool:
        return self.input_active and self.output_active

    def close(self) -> None:
        self.closed = True
        self._thread.join(timeout=5)


class FakeBackend:
    def __init__(self, present: list[Device]) -> None:
        self.present = list(present)
        self._listed = list(present)
        self.opened: list[FakeStreams] = []
        self.rescans = 0
        self.rescan_times: list[float] = []  # time.monotonic() of each rescan
        self.rescans_while_open = 0
        self.mic_value = 0.1
        self.fail_open: str | None = None
        self.cable_start_delay_s = 0.0  # the cable's device calls back only this long after its stream opened
        self.output_inactive = False  # the main cable stream reports itself inactive
        self.output_status = False  # what the fake cable reports to each callback (True: a device error)
        self.default_out: Device | None = SPEAKERS
        self.extra_streams: list[FakeOutput] = []  # monitor, silent render and loopback streams
        self.played: dict[str, list[np.ndarray]] = {}  # what each output stream played, block by block
        self.loopback_stalled = False
        self.loopback_value = 0.0
        self.supports_loopback = True
        self.speaker_queries = 0  # default_speakers calls: endpoint queries, never listings
        self.output_latencies: dict[str, list[object]] = {}  # the latency each output stream was opened with

    def rescan(self) -> None:
        if not all(s.closed for s in self.opened) or not all(s.closed for s in self.extra_streams):
            self.rescans_while_open += 1  # the engine catches the assertion below, so the tests check this count
        assert all(s.closed for s in self.opened), "rescan while a stream is open"
        self.rescans += 1
        self.rescan_times.append(time.monotonic())
        self._listed = list(self.present)

    def inputs(self) -> list[Device]:
        return [d for d in self._listed if d.index in (1, 2, 3)]

    def outputs(self) -> list[Device]:
        return [d for d in self._listed if d.index in OUTPUT_INDEXES]

    def default_input(self) -> Device | None:
        return next((d for d in self.inputs() if d is LAPTOP), None)

    def default_output(self) -> Device | None:
        return self.default_out if self.default_out in self._listed else None

    def default_speakers(self) -> Device | None:
        """The default output now, asked for without listing (the real backend asks WASAPI for the default endpoint
        and finds its name in the last scan)."""
        self.speaker_queries += 1
        return self.default_out if self.default_out in self.present else None

    def open_output(
        self, device: Device, make_callback: Callable[[int, int], Callable[..., None]], latency: object = None
    ) -> FakeOutput:
        self.output_latencies.setdefault(device.name, []).append(latency)
        stream = FakeOutput(self, device, make_callback(48_000, device.channels), "output")
        self.extra_streams.append(stream)
        return stream

    def open_loopback(self, device: Device, make_callback: Callable[[int, int], Callable[..., None]]) -> FakeOutput:
        if not self.supports_loopback:
            raise OSError("no loopback on this platform")
        stream = FakeOutput(self, device, make_callback(48_000, 2), "loopback")
        self.extra_streams.append(stream)
        return stream

    def streams_of(self, kind: str, device: Device | None = None) -> list[FakeOutput]:
        return [s for s in self.extra_streams if s.kind == kind and (device is None or s.device == device)]

    def open(self, mic: Device, cable: Device, make_path: PathFactory) -> tuple[AudioPath, FakeStreams]:
        path = make_path(48_000, mic.channels, 48_000, cable.channels)
        if self.fail_open:
            raise OSError(self.fail_open)
        streams = FakeStreams(self, mic, path, cable)
        self.opened.append(streams)
        return path, streams


def wait_for(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_mic_host_buffer_is_large_enough_for_bursty_devices() -> None:
    # At PortAudio's "low" (22 ms) a Focusrite input silently lost 3.4 % of its audio (J05 Step 2a): 46 367 Hz
    # instead of 48 000, far beyond the +-0.2 % correction, so the cable ran dry. See devices.INPUT_LATENCY_S.
    assert isinstance(devices.INPUT_LATENCY_S, float)
    assert devices.INPUT_LATENCY_S >= 0.060


class StubSoundDevice:
    """Just enough of the sounddevice module for SoundDeviceBackend.open; records how each stream was opened."""

    def __init__(self) -> None:
        self.opened: list[tuple[str, dict]] = []

    class WasapiSettings:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    def check_input_settings(self, **kwargs: object) -> None: ...

    def check_output_settings(self, **kwargs: object) -> None: ...

    def _stream(self, kind: str, kwargs: dict) -> object:
        self.opened.append((kind, kwargs))

        class Stream:
            def start(self) -> None: ...
            def abort(self) -> None: ...
            def close(self) -> None: ...

        return Stream()

    def InputStream(self, **kwargs: object) -> object:
        return self._stream("input", kwargs)

    def OutputStream(self, **kwargs: object) -> object:
        return self._stream("output", kwargs)


def test_the_mic_stream_is_opened_with_the_large_host_buffer() -> None:
    # M2 (J05 review r1): the fix for the original 8 underruns is the latency passed where the mic stream opens.
    backend = object.__new__(devices.SoundDeviceBackend)  # no PortAudio: the stub records the open calls
    stub = StubSoundDevice()
    backend._sd = stub  # type: ignore[attr-defined]
    backend._hostapi = 0  # type: ignore[attr-defined]

    def make_path(in_rate: int, in_channels: int, out_rate: int, out_channels: int) -> AudioPath:
        return AudioPath(in_rate, in_channels, out_rate, out_channels, PassThrough())

    backend.open(MIC, CABLE, make_path)
    opened = dict(stub.opened)
    assert opened["input"]["latency"] == devices.INPUT_LATENCY_S >= 0.060
    assert opened["input"]["device"] == MIC.index and opened["output"]["device"] == CABLE.index
    assert opened["output"]["latency"] == devices.CABLE_LATENCY_S


def test_the_cable_opens_with_a_50_ms_host_buffer() -> None:
    # engine-obs r2, Z1: with PortAudio's "low" (22 ms) a render callback 35-38 ms late starved WASAPI, which played
    # silence and ran 10-20 ms later from then on (the 30 min control). OBS's monitor output initialises WASAPI shared
    # mode with a large buffer (libobs/audio-monitoring/win32/wasapi-output.c:197). The cable asks for 50 ms, whether it
    # opens with the mic or alone after a stall; the speakers (Hear myself, the silent render) keep "low".
    assert devices.CABLE_LATENCY_S == 0.050
    backend = object.__new__(devices.SoundDeviceBackend)
    stub = StubSoundDevice()
    backend._sd = stub  # type: ignore[attr-defined]
    backend._hostapi = 0  # type: ignore[attr-defined]
    backend.open_output(CABLE, lambda rate, ch: lambda *a: None, latency=devices.CABLE_LATENCY_S)
    backend.open_output(SPEAKERS, lambda rate, ch: lambda *a: None)
    assert [kw["latency"] for _, kw in stub.opened] == [0.050, devices.OUTPUT_LATENCY]


def test_a_reopened_cable_keeps_the_50_ms_host_buffer(make_engine) -> None:
    engine, backend = make_engine([MIC, CABLE])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    backend.present.remove(CABLE)
    wait_for(lambda: engine.status().state is State.CABLE_LOST)
    backend.present.append(CABLE)
    wait_for(lambda: engine.status().state is State.ON)
    assert backend.output_latencies[CABLE.name] == [devices.CABLE_LATENCY_S]


class ListingStub(StubSoundDevice):
    """sounddevice's last scan (query_devices reads PortAudio's existing list; only a re-initialisation lists again)."""

    def query_devices(self) -> list[dict]:
        found = {d.index: d for d in (CABLE, SPEAKERS, HEADPHONES)}
        return [
            {
                "name": found[i].name if i in found else f"Input {i}",
                "hostapi": 0,
                "max_input_channels": 0 if i in found else 1,
                "max_output_channels": 2 if i in found else 0,
                "default_samplerate": 48e3,
            }
            for i in range(max(found) + 1)
        ]


def test_the_default_speakers_are_found_without_listing_devices(monkeypatch) -> None:
    # engine-obs r1, F1 (report cause 5): echo removal's retry, and Hear myself and the mic test (F2), ask for the
    # default speakers while the mic and cable stream. That used to re-initialise PyAudioWPatch's PortAudio (a full
    # listing of every endpoint, 25-41 ms) on every try, every 3 s. OBS asks for just the default endpoint
    # (IMMDeviceEnumerator::GetDefaultAudioEndpoint, win-wasapi.cpp:607-618), and the name is matched in the last scan.
    backend = object.__new__(devices.SoundDeviceBackend)
    stub = ListingStub()
    backend._sd = stub  # type: ignore[attr-defined]
    backend._hostapi = 0  # type: ignore[attr-defined]
    backend._pa = object()  # type: ignore[attr-defined]
    listings: list[str] = []
    monkeypatch.setattr(backend, "_list_loopbacks", lambda: listings.append("PortAudio initialised"), raising=False)
    monkeypatch.setattr(devices.SoundDeviceBackend, "rescan", lambda self: listings.append("rescan"))
    monkeypatch.setattr(devices.sys, "platform", "win32")
    asked: list[int] = []
    monkeypatch.setattr(devices, "default_render_name", lambda: asked.append(1) or SPEAKERS.name, raising=False)
    found = [backend.default_speakers() for _ in range(10)]
    assert listings == [], listings
    assert len(asked) == 10  # the default endpoint was asked for every time, so a default change is seen at once
    assert all(d is not None and d.name == SPEAKERS.name and d.index == SPEAKERS.index for d in found)
    monkeypatch.setattr(devices, "default_render_name", lambda: "Speakers (Plugged In Since The Scan)")
    assert backend.default_speakers() is None  # unknown to the scan: not found, rather than listing again


class FourChannelStub(StubSoundDevice):
    """A 4-input USB interface whose mix format has 4 channels: WASAPI without auto-convert takes only that count."""

    def query_devices(self) -> list[dict]:
        return [
            {
                "name": "Line (4ch USB Interface)",
                "hostapi": 0,
                "max_input_channels": 4,
                "max_output_channels": 0,
                "default_samplerate": 48e3,
            },
            {
                "name": CABLE.name,
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48e3,
            },
        ]

    def check_input_settings(self, **kwargs: object) -> None:
        if kwargs["channels"] != 4:
            raise ValueError("Invalid number of channels")


def test_a_mic_with_more_than_two_channels_opens_at_its_own_count_and_is_downmixed(monkeypatch) -> None:
    # engine-obs r1, m4: OBS opens a device with its mix format's channel count (InitFormat, win-wasapi.cpp:828) and
    # downmixes to mono by averaging (libobs/obs-source.c:3984-4001). Clean Voice asked for min(2, channels), which a
    # 3+-channel mix format refuses without Windows' converter.
    monkeypatch.setattr(devices.sys, "platform", "win32")
    backend = object.__new__(devices.SoundDeviceBackend)
    stub = FourChannelStub()
    backend._sd = stub  # type: ignore[attr-defined]
    backend._hostapi = 0  # type: ignore[attr-defined]
    mic = backend.inputs()[0]
    assert mic.channels == 4
    paths: list[AudioPath] = []

    def make_path(in_rate: int, in_channels: int, out_rate: int, out_channels: int) -> AudioPath:
        paths.append(AudioPath(in_rate, in_channels, out_rate, out_channels, PassThrough()))
        return paths[-1]

    backend.open(mic, backend.outputs()[0], make_path)
    assert dict(stub.opened)["input"]["channels"] == 4
    frames = np.tile(np.array([[0.4, 0.2, 0.0, -0.2]], np.float32), (480, 1))
    path = paths[0]
    path.input_callback(frames, 480, None, None)
    clean: list[np.ndarray] = []
    path.chain = type("Grab", (), {"process": lambda self, x, raw=None: clean.append(x.copy()) or x})()  # type: ignore[assignment]
    assert path.process() == 1
    assert np.allclose(clean[0], 0.1)  # the four channels averaged


# ------------------------------------------------------------------ the real WASAPI default lookup (engine-obs r2, M1)

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the WASAPI default endpoint is Windows only")


def _portaudio_default_output() -> str | None:
    """PortAudio's WASAPI default output name (its list from start-up: no new listing), or None (none, as on CI)."""
    import sounddevice as sd

    api = next((i for i, a in enumerate(sd.query_hostapis()) if a["name"] == "Windows WASAPI"), None)
    if api is None:
        return None
    index = sd.query_hostapis(api)["default_output_device"]
    return None if index < 0 else str(sd.query_devices(index)["name"])


def _apartment() -> int | None:
    """The calling thread's COM apartment type (CoGetApartmentType), or None when COM is not set up on it."""
    import ctypes

    kind, qual = ctypes.c_int(), ctypes.c_int()
    hr = ctypes.WinDLL("ole32").CoGetApartmentType(ctypes.byref(kind), ctypes.byref(qual))  # type: ignore[attr-defined]
    return kind.value if hr == 0 else None


def _in_thread(setup: Callable[[], None] | None) -> tuple[int | None, list[str | None], int | None]:
    """Run the real lookup 20 times on a new thread after ``setup``: (apartment before, names, apartment after)."""
    from clean_voice import endpoints

    out: dict[str, object] = {}

    def body() -> None:
        if setup is not None:
            setup()
        out["before"] = _apartment()
        out["names"] = [endpoints.default_render_name() for _ in range(20)]
        out["after"] = _apartment()

    t = threading.Thread(target=body)
    t.start()
    t.join()
    return out["before"], out["names"], out["after"]  # type: ignore[return-value]


@windows_only
def test_the_real_default_lookup_names_portaudios_default_output() -> None:
    # engine-obs r2, M1: the F1 lookup is raw COM (vtable indexes, GUIDs, the PROPVARIANT layout), and every failure path
    # returns None, so only a call against the real WASAPI shows it works: the name is PortAudio's default output's.
    from clean_voice import endpoints

    expected = _portaudio_default_output()
    if expected is None:
        pytest.skip("no default output device on this machine")
    assert endpoints.default_render_name() == expected


@windows_only
@pytest.mark.parametrize("where", ["fresh thread", "device thread", "worker"])
def test_the_real_default_lookup_works_on_every_engine_thread(where: str) -> None:
    # A thread with no COM; the engine's device thread (STA, _init_device_thread); the audio worker, where PortAudio
    # was initialised (start-up and every rescan run there) and Hear myself and the mic test ask for the speakers. The
    # lookup must leave each thread's COM apartment as it found it (no unbalanced CoUninitialize).
    import sounddevice as sd

    expected = _portaudio_default_output()
    if expected is None:
        pytest.skip("no default output device on this machine")
    setup = {
        "fresh thread": None,
        "device thread": engine_module._init_device_thread,
        "worker": lambda: (sd._terminate(), sd._initialize()),  # type: ignore[attr-defined]  # as rescan does
    }[where]
    before, names, after = _in_thread(setup)
    assert names == [expected] * 20
    assert after == before


def test_no_default_output_is_handled(monkeypatch) -> None:
    # The lookup's None (no output device at all, or a failed COM call) is "no speakers", never an exception.
    backend = object.__new__(devices.SoundDeviceBackend)
    backend._sd = ListingStub()  # type: ignore[attr-defined]
    backend._hostapi = 0  # type: ignore[attr-defined]
    monkeypatch.setattr(devices.sys, "platform", "win32")
    monkeypatch.setattr(devices, "default_render_name", lambda: None)
    assert backend.default_speakers() is None


def test_on_macos_a_many_input_interface_opens_two_channels(monkeypatch) -> None:
    # engine-obs r2, n3: Core Audio takes any channel count, so a 4-input interface with the mic on input 1 opened at
    # all four and was averaged to -12 dB. Two channels keep it at the stereo case's -6 dB. Windows needs the mix
    # format's own count (WASAPI without auto-convert refuses any other; the 4-channel test above).
    monkeypatch.setattr(devices.sys, "platform", "darwin")
    backend = object.__new__(devices.SoundDeviceBackend)
    backend._sd = FourChannelStub()  # type: ignore[attr-defined]
    backend._hostapi = 0  # type: ignore[attr-defined]
    assert backend.inputs()[0].channels == 2


class MixFormatStub(StubSoundDevice):
    """Like WASAPI without AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM: a device opens only at its own (mix format) rate, while
    Windows' converter, when asked for, accepts anything (the OBS audit measured both on this PC's devices)."""

    def _check(self, kwargs: dict) -> None:
        device = HEADSET if kwargs["device"] == HEADSET.index else CABLE
        converted = kwargs.get("extra_settings") is not None and kwargs["extra_settings"].kwargs.get("auto_convert")
        if not converted and kwargs["samplerate"] != device.default_rate:
            raise ValueError("Invalid sample rate")

    def check_input_settings(self, **kwargs: object) -> None:
        self._check(kwargs)

    def check_output_settings(self, **kwargs: object) -> None:
        self._check(kwargs)


HEADSET = Device(8, "Headset (ATH-M20xBT )", 1, 16_000.0)  # a Bluetooth hands-free mic: its mix format is 16 kHz


def test_each_device_opens_in_its_own_format_and_the_app_converts() -> None:
    # OBS audit I-3: OBS opens a device at its mix format (win-wasapi.cpp:729, :817-831) and resamples in-process
    # (libobs/obs-source.c:4044-4052). Windows' auto-convert made every device pass the 48 kHz check, so the 16 kHz
    # headset mic was upsampled by Windows and the in-process libsamplerate path never ran on Windows.
    backend = object.__new__(devices.SoundDeviceBackend)
    stub = MixFormatStub()
    backend._sd = stub  # type: ignore[attr-defined]
    backend._hostapi = 0  # type: ignore[attr-defined]
    rates: list[tuple[int, int]] = []

    def make_path(in_rate: int, in_channels: int, out_rate: int, out_channels: int) -> AudioPath:
        rates.append((in_rate, out_rate))
        return AudioPath(in_rate, in_channels, out_rate, out_channels, PassThrough())

    path, _ = backend.open(HEADSET, CABLE, make_path)
    assert rates == [(16_000, 48_000)]  # the headset at its own 16 kHz; the app's resampler takes it to 48 kHz
    for _, kwargs in stub.opened:
        settings = kwargs["extra_settings"]
        assert settings is None or not settings.kwargs.get("auto_convert"), "Windows' converter was asked for"
    frames = np.random.default_rng(0).standard_normal((1600, 1)).astype(np.float32) * 0.1  # 100 ms at 16 kHz
    path.input_callback(frames, 1600, None, None)
    assert path.process() >= 8  # converted in the app: 100 ms of 16 kHz became 48 kHz blocks


def test_cable_matching() -> None:
    backend = FakeBackend([MIC, LAPTOP, CABLE_REC, CABLE, SPEAKERS])
    assert find_cable(backend) == CABLE
    assert [d.name for d in microphones(backend)] == [MIC.name, LAPTOP.name]  # never the cable's own side
    assert is_cable("VB-Cable") and not is_cable("Speakers (Realtek)")


def test_lists_mics_and_starts_with_the_default(make_engine) -> None:
    engine, _ = make_engine([MIC, LAPTOP, CABLE_REC, CABLE])
    wait_for(lambda: engine.mics() == [MIC.name, LAPTOP.name])
    assert engine.default_mic() == LAPTOP.name
    engine.start(None)
    wait_for(lambda: engine.status().state is State.ON)
    status = engine.status()
    assert status.mic == LAPTOP.name and status.cable == CABLE.name and status.recording


def test_cable_missing_then_installed(make_engine) -> None:
    engine, backend = make_engine([MIC])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.NO_CABLE)
    assert not engine.recordings_dir.exists() or not any(engine.recordings_dir.iterdir())  # no empty backups
    backend.present.append(CABLE)
    wait_for(lambda: engine.status().state is State.ON)


def test_mic_unplugged_stops_cleanly_and_resumes(make_engine) -> None:
    engine, backend = make_engine([MIC, CABLE])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    first = backend.opened[0]
    backend.present.remove(MIC)
    wait_for(lambda: engine.status().state is State.MIC_LOST)
    assert first.closed
    backend.present.append(MIC)
    wait_for(lambda: engine.status().state is State.ON)
    assert len(backend.opened) == 2
    # The backup continues in the same pair of files across the reconnect.
    files = engine.status().recording_files
    engine.stop()
    wait_for(lambda: engine.status().state is State.OFF)
    with wave.open(str(files[1]), "rb") as w:
        assert w.getnframes() > 0


def test_cable_unplugged_is_reported(make_engine) -> None:
    engine, backend = make_engine([MIC, CABLE])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    backend.present.remove(CABLE)
    wait_for(lambda: engine.status().state is State.CABLE_LOST)


def test_no_mic(make_engine) -> None:
    engine, _ = make_engine([CABLE])
    engine.start(None)
    wait_for(lambda: engine.status().state is State.NO_MIC)


def test_stop_closes_streams_and_recording(make_engine) -> None:
    engine, backend = make_engine([MIC, CABLE])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    wait_for(lambda: engine.status().recording_seconds > 0.2)
    engine.stop()
    wait_for(lambda: engine.status().state is State.OFF)
    assert all(s.closed for s in backend.opened)
    status = engine.status()
    assert not status.recording
    for path in status.recording_files:
        with wave.open(str(path), "rb") as w:
            assert w.getframerate() == 48_000 and w.getsampwidth() == 3


def test_switching_mic_while_on(make_engine) -> None:
    engine, backend = make_engine([MIC, LAPTOP, CABLE])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    engine.select_mic(LAPTOP.name)
    wait_for(lambda: engine.status().mic == LAPTOP.name and len(backend.opened) == 2)
    wait_for(lambda: engine.status().state is State.ON)
    assert backend.opened[0].closed


def test_device_error_is_reported_and_retried_without_empty_backups(make_engine) -> None:
    engine, backend = make_engine([MIC, CABLE])
    backend.fail_open = "The microphone is in use by another app"
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ERROR)
    assert engine.status().error == "The microphone is in use by another app"
    assert not engine.recordings_dir.exists() or not any(engine.recordings_dir.iterdir())
    backend.fail_open = None
    wait_for(lambda: engine.status().state is State.ON)


# ------------------------------------------------------------------ device unplug (J05 Step 4)


def test_no_device_listing_while_audio_runs(make_engine, monkeypatch: pytest.MonkeyPatch) -> None:
    # D1 (docs/simple-clean-mic-audio-report.md, cause 5): listing devices disturbed drivers in the original app.
    # engine-obs r1, F1: also while echo removal retries for speakers every RETRY_S (the default output is the cable,
    # as VB-CABLE's installer often leaves it), each try only asks for the default endpoint. That the real backend's
    # query lists nothing is test_the_default_speakers_are_found_without_listing_devices.
    # Echo removal is Windows only, so on a Mac it never retries (CI run 43368578097: 0 queries); its retry logic is the
    # same everywhere, so run it here on both. The default output is the cable, so no AEC3 is ever created.
    monkeypatch.setattr(engine_module, "echo_platform_reason", lambda: None)
    engine, backend = make_engine([MIC, CABLE, SPEAKERS])
    backend.default_out = CABLE
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    engine.set_echo(True)
    wait_for(lambda: engine.status().echo is engine_module.EchoState.UNAVAILABLE)
    rescans, queries = backend.rescans, backend.speaker_queries
    wait_for(lambda: backend.speaker_queries - queries >= 5)  # the retry path really ran: it asked for the default
    engine.refresh()  # the window asks for a fresh list when the dropdown opens; ignored while running
    time.sleep(0.3)
    assert engine.status().state is State.ON
    assert backend.rescans == rescans
    assert backend.rescans_while_open == 0


def test_a_missing_mic_is_retried_every_retry_interval(make_engine) -> None:
    # D2: OBS reconnects every 3 s (plugins/win-wasapi/win-wasapi.cpp:1089, RECONNECT_INTERVAL).
    assert engine_module.RETRY_S == 3.0
    engine, backend = make_engine([MIC, CABLE])  # retry_s=0.2
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    backend.present.remove(MIC)
    wait_for(lambda: engine.status().state is State.MIC_LOST)
    rescans = backend.rescans
    states = set()
    # Waits for the retries rather than counting them in a fixed second: the macOS runner's timed waits ran late
    # enough that only 2 fitted (CI run 43368578097). What the engine controls is checked: never sooner than retry_s.
    deadline = time.monotonic() + 10.0
    while backend.rescans - rescans < 3 and time.monotonic() < deadline:
        states.add(engine.status().state)
        time.sleep(0.02)
    assert backend.rescans - rescans >= 3
    gaps = [b - a for a, b in pairwise(backend.rescan_times[rescans - 1 :])]
    assert all(0.2 - 1e-6 <= gap < 1.0 for gap in gaps), gaps  # every retry interval: not sooner, and it keeps retrying
    assert states == {State.MIC_LOST}
    assert backend.rescans_while_open == 0


def test_counters_and_backups_survive_a_reconnect(make_engine) -> None:
    # D3: the totals carry across the reconnect, and the backup goes on in the same pair of files.
    engine, backend = make_engine([MIC, CABLE])
    backend.output_status = True  # every cable callback reports a device error, so the counter keeps rising
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    wait_for(lambda: engine.status().recording_seconds > 1.2)  # past the first flush
    files = engine.status().recording_files
    seen: list[int] = []
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            seen.append(engine.status().counters.output_device_errors)
            time.sleep(0.005)

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        before = engine.status().counters.output_device_errors
        frames_before = engine.status().recording_seconds
        backend.present.remove(MIC)
        wait_for(lambda: engine.status().state is State.MIC_LOST)
        backend.present.append(MIC)
        wait_for(lambda: engine.status().state is State.ON and len(backend.opened) == 2)
        wait_for(lambda: engine.status().counters.output_device_errors > before + 20)
        wait_for(lambda: engine.status().recording_seconds > frames_before + 1.2)
    finally:
        stop.set()
        watcher.join()
    assert all(b >= a for a, b in pairwise(seen)), "a counter went down"
    assert engine.status().recording_files == files
    engine.stop()
    wait_for(lambda: engine.status().state is State.OFF)
    for f in files:
        with wave.open(str(f), "rb") as w:
            assert w.getnframes() / 48_000 > frames_before + 1.0


def test_a_cable_stall_keeps_the_mic_and_the_backup(make_engine) -> None:
    # OBS audit I-4a: each WASAPI source has its own capture thread and reconnect (win-wasapi.cpp:1091-1224, :959-993),
    # so one device failing never stops another. A cable stall (its format changed in Sound settings, another app took
    # it exclusively) used to close the mic too, leaving a hole of the stall + watchdog + retry in the backup.
    # The backup is measured against what the mic delivered, not the wall clock: the fake mic's 10 ms sleeps ran ~4x
    # long on the macOS runner, so it delivered 0.27 s in 1.22 s (CI run 43368578097) and the backup had all of it.
    engine, backend = make_engine([MIC, CABLE])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    first = backend.opened[0]
    path = engine._path
    assert path is not None
    wait_for(lambda: engine.status().recording_seconds > 0.5)
    mic0, rec0 = path.in_ring.total_written, engine.status().recording_seconds
    backend.present.remove(CABLE)  # the cable stops calling back; the mic does not
    wait_for(lambda: engine.status().state is State.CABLE_LOST)
    mic_at_stall = path.in_ring.total_written
    backend.present.append(CABLE)
    wait_for(lambda: engine.status().state is State.ON)
    assert path.in_ring.total_written > mic_at_stall, "the mic stopped delivering while the cable was down"
    time.sleep(0.5)
    recorded = engine.status().recording_seconds - rec0
    delivered = (path.in_ring.total_written - mic0) / 48_000
    assert not first.closed, "the mic stream (and so the backup) was stopped by a cable-only stall"
    assert len(backend.opened) == 1 and engine._path is path  # the mic was never reopened
    assert recorded >= delivered - 0.15, f"the backup got {recorded:.2f} s of the {delivered:.2f} s the mic delivered"
    played = engine.status().counters.underruns
    wait_for(lambda: backend.streams_of("output", CABLE) and backend.streams_of("output", CABLE)[-1].calls > 20)
    assert engine.status().counters.underruns <= played + 1  # the reopened cable plays the voice again


def test_a_cable_that_goes_inactive_keeps_the_mic_and_the_backup(make_engine, caplog) -> None:
    # B001 item 8 (E-3): an inactive cable stream used to close everything (MIC_LOST). Each stream's own state decides,
    # as each OBS source has its own reconnect (win-wasapi.cpp:959-993, :1089-1224): only the cable closes and returns.
    caplog.set_level("WARNING", logger="clean_voice")
    engine, backend = make_engine([MIC, CABLE])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    first = backend.opened[0]
    path = engine._path
    assert path is not None
    wait_for(lambda: engine.status().recording_seconds > 0.5)
    mic0, rec0 = path.in_ring.total_written, engine.status().recording_seconds
    backend.output_inactive = True  # stays set: a reopened cable is judged by its own new stream
    seen: list[State] = []
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        seen.append(engine.status().state)
        time.sleep(0.01)
    assert State.MIC_LOST not in seen, set(seen)
    assert not first.closed, "the mic stream was stopped by a cable that went inactive"
    assert State.CABLE_LOST in seen and engine.status().state is State.ON, set(seen)
    assert len(backend.opened) == 1 and len(backend.streams_of("output", CABLE)) == 1  # reopened exactly once
    assert sum("the cable stopped" in r.getMessage() for r in caplog.records) == 1  # no repeated _cable_lost
    delivered = (path.in_ring.total_written - mic0) / 48_000
    assert engine.status().recording_seconds - rec0 >= delivered - 0.15


def test_stop_while_the_cable_is_down_closes_cleanly(tmp_path) -> None:
    backend = BusyCableBackend([MIC, CABLE])
    engine = engine_module.Engine(tmp_path / "Recordings", backend, retry_s=0.2, watchdog_s=0.3)
    try:
        engine.start(MIC.name)
        wait_for(lambda: engine.status().state is State.ON)
        first = backend.opened[0]
        backend.cable_busy = True
        backend.output_inactive = True
        wait_for(lambda: engine.status().state in (State.CABLE_LOST, State.MIC_LOST))
        time.sleep(0.5)
        assert engine.status().state is State.CABLE_LOST and not first.closed  # the mic runs on
        engine.stop()
        wait_for(lambda: engine.status().state is State.OFF)
        assert first.closed
    finally:
        engine.close()


def test_a_slow_starting_cable_does_not_pile_up_voice(make_engine) -> None:
    # B001 item 6 (E-1): the cable calls back only after 0.5 s; nothing may queue before it (wasapi-output.c).
    engine, backend = make_engine([MIC, CABLE])
    backend.cable_start_delay_s = 0.5
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    path = engine._path
    assert path is not None
    wait_for(lambda: path.cable.callbacks > 20)
    assert path.cable.ring.available() <= int(0.060 * 48_000) + 480, path.cable.ring.available()
    delivered = path.in_ring.total_written / 48_000
    assert delivered >= 0.5 and engine.status().recording_seconds >= delivered - 0.15


class BusyCableBackend(FakeBackend):
    """The cable stalls, then refuses to reopen (another app holds it exclusively). Like SoundDeviceBackend.open_output,
    ``make_callback`` runs before the stream can fail to open."""

    def __init__(self, present: list[Device]) -> None:
        super().__init__(present)
        self.cable_busy = False
        self.cable_tries = 0

    def open_output(
        self, device: Device, make_callback: Callable[[int, int], Callable[..., None]], latency: object = None
    ) -> FakeOutput:
        if device == CABLE and self.cable_busy:
            self.cable_tries += 1
            make_callback(48_000, device.channels)
            raise OSError("Device unavailable [PaErrorCode -9985]")
        return super().open_output(device, make_callback, latency)


def test_a_cable_that_will_not_reopen_counts_no_lost_audio(tmp_path) -> None:
    # engine-obs r1, m1: a failed reopen left a fresh sink live that no device read; its 1 s ring filled and
    # "processed samples lost" rose by 100 a second until the next try (92 after 2 s at the real RETRY_S).
    backend = BusyCableBackend([MIC, CABLE])
    engine = engine_module.Engine(tmp_path / "Recordings", backend, retry_s=engine_module.RETRY_S, watchdog_s=0.3)
    try:
        engine.start(MIC.name)
        wait_for(lambda: engine.status().state is State.ON)
        backend.cable_busy = True
        backend.present.remove(CABLE)
        wait_for(lambda: engine.status().state is State.CABLE_LOST)
        backend.present.append(CABLE)  # there, but busy: every reopen fails
        wait_for(lambda: backend.cable_tries >= 1, timeout=10)
        time.sleep(2.0)
        assert engine.status().counters.output_overflows == 0
        assert engine.status().state is State.CABLE_LOST and not backend.opened[0].closed
    finally:
        engine.close()


class SlowStartBackend(FakeBackend):
    """A Bluetooth headset that switches to its hands-free profile: the mic's first callback comes 1.5 s late."""

    def open(self, mic: Device, cable: Device, make_path: PathFactory) -> tuple[AudioPath, FakeStreams]:
        if not self.opened:  # silent from the very start (before the stream's thread can deliver anything)
            self.present.remove(mic)
            threading.Timer(1.5, lambda: self.present.append(mic)).start()
        return super().open(mic, cable, make_path)


def test_a_mic_slow_to_start_is_not_lost(make_engine, tmp_path) -> None:
    # engine-obs r1, m3: on the real headset the first callback took over 1 s, so the 1 s watchdog called the mic lost
    # 2 s after start, closed it and reopened 3 s later: 5 s of silence and a "mic lost" flash at start. A stream gets
    # FIRST_AUDIO_S for its first callback; after that the running watchdog applies.
    assert engine_module.FIRST_AUDIO_S >= 3.0
    backend = SlowStartBackend([MIC, CABLE])
    engine = engine_module.Engine(tmp_path / "Recordings", backend, retry_s=0.2, watchdog_s=0.3)
    try:
        engine.start(MIC.name)
        seen = set()
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            seen.add(engine.status().state)
            time.sleep(0.01)
        assert State.MIC_LOST not in seen and engine.status().state is State.ON, seen
        assert len(backend.opened) == 1  # never closed and reopened
        backend.present.remove(MIC)  # once running, a lost mic is still found by the 0.3 s watchdog
        wait_for(lambda: engine.status().state is State.MIC_LOST, timeout=2.0)
    finally:
        engine.close()


HDMI = Device(7, "DELL U2720Q (NVIDIA High Definition Audio)", 2, 48_000.0)  # listed before the speakers


class NewDefaultBackend(FakeBackend):
    """Windows' default output is headphones plugged in after the last scan: the real backend finds no such name."""

    def outputs(self) -> list[Device]:
        order = {CABLE.index: 0, HDMI.index: 1, SPEAKERS.index: 2, HEADPHONES.index: 3}
        return sorted((d for d in self._listed if d.index in OUTPUT_INDEXES), key=lambda d: order[d.index])

    def default_speakers(self) -> Device | None:
        return None


def test_hear_myself_uses_the_scanned_default_when_the_new_default_is_unknown(tmp_path) -> None:
    # engine-obs r2, n1: with the default unknown to the scan, Hear myself took the first usable output in PortAudio's
    # order (a monitor's HDMI output, no speakers) instead of the default Windows had at the scan.
    backend = NewDefaultBackend([MIC, CABLE, HDMI, SPEAKERS])
    backend.default_out = SPEAKERS  # the scanned default
    engine = engine_module.Engine(tmp_path / "Recordings", backend, retry_s=0.2, watchdog_s=0.3)
    try:
        engine.start(MIC.name)
        wait_for(lambda: engine.status().state is State.ON)
        engine.set_monitor(True)
        wait_for(lambda: engine.status().monitor_device is not None)
        assert engine.status().monitor_device == SPEAKERS.name
    finally:
        engine.close()


class DeadMicStreams(FakeStreams):
    """A mic that opens but never calls back while ``backend.mic_dead``; the cable runs."""

    def _run(self) -> None:
        block = np.full((480, self.mic.channels), 0.1, np.float32)
        out = np.zeros((480, 2), np.float32)
        while not self.closed:
            if not self.backend.mic_dead:  # type: ignore[attr-defined]
                self.path.input_callback(block, 480, None, None)
            self.path.output_callback(out, 480, None, None)
            time.sleep(0.01)


class DeadMicBackend(FakeBackend):
    mic_dead = False

    def open(self, mic: Device, cable: Device, make_path: PathFactory) -> tuple[AudioPath, FakeStreams]:
        path = make_path(48_000, mic.channels, 48_000, cable.channels)
        streams = DeadMicStreams(self, mic, path, cable)
        self.opened.append(streams)
        return path, streams


def test_a_dead_mic_stays_lost_between_retries(tmp_path, monkeypatch) -> None:
    # engine-obs r2, n5: a mic that opens but never delivers. Each retry said ON at once, so with FIRST_AUDIO_S the
    # screen showed "On" for 5 s of every 8. After MIC_LOST, a reopened mic is ON only once it delivers.
    monkeypatch.setattr(engine_module, "FIRST_AUDIO_S", 0.5)
    backend = DeadMicBackend([MIC, CABLE])
    engine = engine_module.Engine(tmp_path / "Recordings", backend, retry_s=0.2, watchdog_s=0.3)
    try:
        engine.start(MIC.name)
        wait_for(lambda: engine.status().state is State.ON)
        backend.mic_dead = True
        wait_for(lambda: engine.status().state is State.MIC_LOST)
        seen = set()
        deadline = time.monotonic() + 3.0  # several retries, each waiting FIRST_AUDIO_S
        while time.monotonic() < deadline:
            seen.add(engine.status().state)
            time.sleep(0.01)
        assert len(backend.opened) >= 3  # it did retry
        assert seen == {State.MIC_LOST}, seen
        backend.mic_dead = False
        wait_for(lambda: engine.status().state is State.ON, timeout=3.0)  # the reopened mic delivers: ON
    finally:
        engine.close()


def test_hear_myself_follows_the_new_default_speakers(tmp_path) -> None:
    # engine-obs r1, F2 (OBS audit I-4b): speaker streams that target the default output follow a default change. The
    # headphones' battery dies and Windows makes the laptop speakers the default; Hear myself turned off ("stopped").
    # Turned on again, it went back to the vanished headphones from the last scan.
    from test_echo import CachedDefaultBackend

    backend = CachedDefaultBackend([MIC, CABLE, SPEAKERS, HEADPHONES])
    engine = engine_module.Engine(tmp_path / "Recordings", backend, retry_s=0.2, watchdog_s=0.3)
    try:
        engine.start(MIC.name)
        wait_for(lambda: engine.status().state is State.ON)
        engine.set_monitor(True)
        wait_for(lambda: engine.status().monitor and engine.status().monitor_device == HEADPHONES.name)
        backend.present.remove(HEADPHONES)
        backend.system_default = SPEAKERS
        wait_for(lambda: not engine.status().monitor)
        engine.set_monitor(True)
        wait_for(lambda: engine.status().monitor and engine.status().monitor_device == SPEAKERS.name, timeout=3)
        engine.start_mic_test()  # the mic test plays on the same, new default
        wait_for(lambda: bool(backend.streams_of("output", SPEAKERS)))
        assert not any(not s.closed for s in backend.streams_of("output", HEADPHONES))
    finally:
        engine.close()


# ------------------------------------------------------------------ what the J07 screen needs (E1-E6)


def test_status_has_the_mic_level_apart_from_the_clean_level(make_engine) -> None:
    # E1: the original's two meters, "Your mic" and "What others hear" (index.html:67-68).
    engine, backend = make_engine([MIC, CABLE])
    backend.mic_value = 0.25
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    seen: list[tuple[float, float]] = []

    def raw_level_seen() -> bool:
        status = engine.status()
        seen.append((status.input_level, status.level))
        return status.input_level > 0

    wait_for(raw_level_seen)
    assert any(abs(raw - 0.25) < 1e-6 for raw, _ in seen)  # the raw mic's own peak, not the clean voice's
    assert any(abs(raw - clean) > 1e-3 for raw, clean in seen if raw > 0)


def test_outputs_and_choosing_where_the_clean_voice_goes(make_engine) -> None:
    # E2: "Send clean voice to" (index.html:115). Choosing while on reopens to it, like choosing a mic.
    engine, backend = make_engine([MIC, CABLE, SPEAKERS, HEADPHONES])
    wait_for(lambda: engine.outputs() == [CABLE.name, SPEAKERS.name, HEADPHONES.name])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    assert engine.status().output == CABLE.name and backend.opened[-1].output == CABLE
    engine.select_output(HEADPHONES.name)
    wait_for(lambda: len(backend.opened) == 2 and engine.status().state is State.ON)
    assert backend.opened[-1].output == HEADPHONES and backend.opened[0].closed
    assert engine.status().output == HEADPHONES.name


def test_a_saved_output_that_is_gone_at_start_uses_the_cable(make_engine) -> None:
    # E3
    engine, backend = make_engine([MIC, CABLE, SPEAKERS])
    engine.select_output("Headphones (Not Plugged In)")
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    assert backend.opened[-1].output == CABLE and engine.status().output == CABLE.name


def test_a_saved_mic_that_is_gone_at_start_uses_the_default_mic(make_engine) -> None:
    # E4: the saved mic is a preference, not a requirement; the original fell back the same way (main.js:435-440).
    engine, backend = make_engine([MIC, LAPTOP, CABLE])
    engine.start("Microphone (Unplugged Since Yesterday)")
    wait_for(lambda: engine.status().state is not State.OFF)
    wait_for(lambda: engine.status().state is State.ON)
    assert engine.status().mic == LAPTOP.name and backend.opened[-1].mic == LAPTOP


def test_hear_myself_never_plays_on_the_chosen_output(make_engine) -> None:
    # E5 (main.js:453-458): the clean voice already goes to S, so Hear myself must use another output.
    engine, backend = make_engine([MIC, CABLE, SPEAKERS, HEADPHONES])
    backend.default_out = SPEAKERS
    engine.select_output(SPEAKERS.name)
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON and engine.status().output == SPEAKERS.name)
    engine.set_monitor(True)
    wait_for(lambda: engine.status().monitor)
    assert engine.status().monitor_device == HEADPHONES.name
    assert backend.streams_of("output", SPEAKERS) == []


def test_the_backup_error_names_any_call_app(tmp_path) -> None:
    # E6: the voice reaches whichever call app the guest uses, not only Meet.
    blocker = tmp_path / "Music"
    blocker.write_text("not a folder")
    backend = FakeBackend([MIC, CABLE])
    engine = engine_module.Engine(blocker / "Clean Voice", backend, retry_s=0.2, watchdog_s=0.3)
    try:
        engine.set_recording(True)  # the backup is off by default (2026-11-11)
        engine.start(MIC.name)
        wait_for(lambda: engine.status().recording_error is not None)
        text = engine.status().recording_error or ""
        assert "the call" in text and "Meet" not in text
    finally:
        engine.close()
