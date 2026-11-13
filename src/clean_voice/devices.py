"""Audio devices through sounddevice (PortAudio): listing, finding VB-CABLE, opening the two streams."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from clean_voice.audio_path import SAMPLE_RATE, AudioPath
from clean_voice.endpoints import default_render_name

log = logging.getLogger(__name__)

# The mic's host buffer. OBS opens WASAPI shared capture with a 5 s buffer and drains it on every event
# (plugins/win-wasapi/win-wasapi.cpp, BUFFER_TIME_100NS). With PortAudio's "low" (22 ms) some devices that deliver in
# 20-30 ms bursts lose samples without any status flag: measured on a Focusrite "Video Call" input, 0.5-0.9 % of the
# audio vanished at 22 ms and none at 60 ms or more. J05 Step 2a measured it again on the same input: at 22 ms it
# delivered 46 367 Hz instead of 48 000 (3.4 % lost, 29 underruns in 60 s, since no +-0.2 % correction can make up for
# it); this was the cause of the 8 underruns in 20 s seen before. At 100 ms: 48 000 Hz within 13 ppm and no underruns
# in 5-minute runs. Capture still arrives every period, so this adds no delay. Never below 60 ms (pinned by a test).
INPUT_LATENCY_S = 0.1
# The speakers' host buffer (Hear myself, the mic test, the silent render for echo removal).
OUTPUT_LATENCY = "low"
# The cable's host buffer. At PortAudio's "low" (22 ms) a render callback 35-38 ms late starved WASAPI, which played
# silence and ran 10-20 ms later from then on (engine-obs r2, Z1: 3 such gaps in a 30 min control). OBS's monitoring
# output initialises WASAPI shared mode with a large buffer (libobs/audio-monitoring/win32/wasapi-output.c:197, 1 s
# for its push model); this pulls with a callback, so 50 ms, beyond those late callbacks. WASAPI rounds it to 60 ms,
# callbacks stay 10 ms (so the 60 ms ring target is unchanged), and the voice reaches CABLE Output about 30-40 ms later
# than at "low" (measured: 94-109 -> 134-137 ms from hand-over to arrival). Orchestrator decision.
CABLE_LATENCY_S = 0.050

# VB-CABLE's playback side: "CABLE Input (VB-Audio Virtual Cable)" on Windows, "VB-Cable" on macOS. Meet then
# listens on the recording side, "CABLE Output" (Windows) or "VB-Cable" (macOS).
_CABLE_OUTPUT = re.compile(r"^CABLE Input\b|\bVB-Cable\b", re.IGNORECASE)
_CABLE_ANY = re.compile(r"\bCABLE(-[A-D])? (Input|Output)\b|\bVB-Cable\b", re.IGNORECASE)
CABLE_DOWNLOAD_URL = "https://vb-audio.com/Cable/"
# Outputs that are not real headphones or speakers (simple-clean-mic ui/main.js:7).
_VIRTUAL = re.compile(r"cable|voicemod|voicemeeter|virtual", re.IGNORECASE)


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    channels: int
    default_rate: float


def is_cable_output(name: str) -> bool:
    return bool(_CABLE_OUTPUT.search(name))


def is_cable(name: str) -> bool:
    return bool(_CABLE_ANY.search(name))


class Streams(Protocol):
    @property
    def active(self) -> bool: ...

    def close(self) -> None: ...


class MainStreams(Streams, Protocol):
    """The mic and cable streams opened together by ``Backend.open``."""

    def close_output(self) -> None:
        """Close only the cable's stream (it stalled); the mic keeps running, and ``active`` then means the mic."""
        ...


PathFactory = Callable[[int, int, int, int], AudioPath]
Callback = Callable[[Any, int, Any, Any], None]
# Given the rate and channels the device opened with, returns the callback (and sets up the sink behind it).
CallbackFactory = Callable[[int, int], Callback]


class Backend(Protocol):
    """What the engine needs from the sound system; tests substitute a fake."""

    def rescan(self) -> None: ...

    def inputs(self) -> list[Device]: ...

    def outputs(self) -> list[Device]: ...

    def default_input(self) -> Device | None: ...

    def default_output(self) -> Device | None: ...

    def default_speakers(self) -> Device | None:
        """The default output as the system has it now, looked up afresh even while the main streams are open (for
        echo removal, Hear myself and the mic test). It lists no devices: one endpoint query (median 0.8 ms, worst
        15.7 ms measured), so the audio worker may call it."""
        ...

    def open(self, mic: Device, cable: Device, make_path: PathFactory) -> tuple[AudioPath, MainStreams]: ...

    def open_output(
        self, device: Device, make_callback: CallbackFactory, latency: float | str | None = None
    ) -> Streams: ...

    def open_loopback(self, device: Device, make_callback: CallbackFactory) -> Streams:
        """Capture what ``device`` plays (WASAPI loopback). Raises OSError where the platform has none."""
        ...


def microphones(backend: Backend) -> list[Device]:
    """Input devices a guest can pick: everything except the cable's own recording side."""
    return [d for d in backend.inputs() if not is_cable(d.name)]


def is_virtual(name: str) -> bool:
    return bool(_VIRTUAL.search(name))


def monitor_device(outputs: list[Device], default: Device | None, cable: Device | None) -> Device | None:
    """Where Hear myself and the mic test play: the default output unless it is the cable or virtual, else the first
    output that is neither, else none (simple-clean-mic ui/main.js:452-458)."""

    def ok(d: Device | None) -> bool:
        return d is not None and d != cable and not is_virtual(d.name)

    if ok(default):
        return default
    return next((d for d in outputs if ok(d)), None)


def find_cable(backend: Backend) -> Device | None:
    outputs = [d for d in backend.outputs() if is_cable_output(d.name)]
    exact = [d for d in outputs if d.name.lower().startswith("cable input")]
    return (exact or outputs or [None])[0]


class _SoundDeviceStreams:
    def __init__(self, streams: list[Any], output: Any = None) -> None:
        self._streams = streams
        self._output = output  # the cable's stream, which can be closed on its own

    @property
    def active(self) -> bool:
        return all(s.active for s in self._streams)

    @staticmethod
    def _close_one(s: Any) -> None:
        # A vanished device may fail to stop or close; carry on so the other stream is closed too.
        try:
            s.abort()
        except Exception:
            log.debug("stream abort failed", exc_info=True)
        try:
            s.close()
        except Exception:
            log.debug("stream close failed", exc_info=True)

    def close(self) -> None:
        for s in self._streams:
            self._close_one(s)

    def close_output(self) -> None:
        out, self._output = self._output, None
        if out is not None and out in self._streams:
            self._streams.remove(out)
            self._close_one(out)


class _Time:
    """PortAudio's callback time info, as sounddevice gives it, for a PyAudio callback (preallocated, reused)."""

    currentTime = 0.0


class _LoopbackStream:
    """A PyAudioWPatch loopback stream behind the ``Streams`` protocol. Its callback turns PyAudio's bytes into a numpy
    view (no copy) and hands it on; the path's callback then copies it into the reference ring."""

    def __init__(self, pa: Any, pyaudio: Any, index: int, rate: int, channels: int, callback: Callback) -> None:
        self._time = _Time()
        self._channels = channels
        self._callback = callback
        self._continue = pyaudio.paContinue
        self._stream = pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=index,
            frames_per_buffer=0,
            stream_callback=self._on_audio,
            start=False,
        )
        self._stream.start_stream()

    def _on_audio(self, in_data: bytes | None, frames: int, time_info: dict[str, float], status: int) -> Any:
        self._time.currentTime = time_info.get("current_time", 0.0)
        if in_data is not None:
            self._callback(np.frombuffer(in_data, np.float32).reshape(-1, self._channels), frames, self._time, status)
        return None, self._continue

    @property
    def active(self) -> bool:
        try:
            return bool(self._stream.is_active())
        except Exception:  # noqa: BLE001 - a vanished device
            return False

    def close(self) -> None:
        for step in (self._stream.stop_stream, self._stream.close):
            try:
                step()
            except Exception:
                log.debug("loopback stream close failed", exc_info=True)


class SoundDeviceBackend:
    """PortAudio through sounddevice, on WASAPI (Windows) or Core Audio (macOS). The speakers' loopback, for echo
    removal, goes through PyAudioWPatch (Windows only): sounddevice's PortAudio has no WASAPI loopback (J06 Step 3)."""

    def __init__(self) -> None:
        import sounddevice as sd

        self._sd = sd
        self._hostapi = self._pick_hostapi()
        self._pa: Any = None
        self._pyaudio: Any = None
        self._loopbacks: dict[str, dict[str, Any]] = {}
        self._list_loopbacks()

    def _list_loopbacks(self) -> None:
        """Make PyAudioWPatch list the loopback devices again. Only while no stream is open (report cause 5)."""
        if sys.platform != "win32":
            return
        try:
            import pyaudiowpatch as pyaudio

            if self._pa is not None:
                self._pa.terminate()
            self._pyaudio = pyaudio
            self._pa = pyaudio.PyAudio()
            self._loopbacks = {
                d["name"].removesuffix(" [Loopback]"): d for d in self._pa.get_loopback_device_info_generator()
            }
        except Exception:
            log.warning("WASAPI loopback unavailable", exc_info=True)
            self._pa, self._loopbacks = None, {}

    def _pick_hostapi(self) -> int:
        wanted = "Windows WASAPI" if sys.platform == "win32" else "Core Audio"
        for i, api in enumerate(self._sd.query_hostapis()):
            if api["name"] == wanted:
                return i
        return self._sd.default.hostapi

    def rescan(self) -> None:
        """Make PortAudio list devices again. Only call while no stream is open."""
        sd = self._sd
        # sounddevice's documented way to make PortAudio enumerate devices again.
        sd._terminate()  # type: ignore[attr-defined]
        sd._initialize()  # type: ignore[attr-defined]
        self._hostapi = self._pick_hostapi()
        self._list_loopbacks()

    def _devices(self, kind: str) -> list[Device]:
        found = []
        for i, d in enumerate(self._sd.query_devices()):
            if d["hostapi"] == self._hostapi and d[kind] > 0:
                found.append(Device(i, d["name"], self._channels(int(d[kind])), float(d["default_samplerate"])))
        return found

    @staticmethod
    def _channels(count: int) -> int:
        """The channel count to open a device with. Windows: the device's own, its mix format's (OBS opens with the mix
        format's nChannels, InitFormat, win-wasapi.cpp:828), since WASAPI without Windows' converter refuses any other.
        macOS: Core Audio takes any count, so at most two, as before (a 4-input interface with the mic on input 1 then
        comes in at -6 dB, as a stereo one does, not at -12 dB). AudioPath averages a mic's channels to mono (OBS's
        downmix averages its output layout's channels, libobs/obs-source.c:3984-4001); a sink writes the voice to
        every channel."""
        return count if sys.platform == "win32" else min(2, count)

    def inputs(self) -> list[Device]:
        return self._devices("max_input_channels")

    def outputs(self) -> list[Device]:
        return self._devices("max_output_channels")

    def default_input(self) -> Device | None:
        index = self._sd.query_hostapis(self._hostapi)["default_input_device"]
        return next((d for d in self.inputs() if d.index == index), None)

    def default_output(self) -> Device | None:
        index = self._sd.query_hostapis(self._hostapi)["default_output_device"]
        return next((d for d in self.outputs() if d.index == index), None)

    def open_output(
        self, device: Device, make_callback: CallbackFactory, latency: float | str | None = None
    ) -> Streams:
        """A plain output stream, started: the speaker sink or the silent render at OUTPUT_LATENCY, or the cable
        reopened on its own with ``latency=CABLE_LATENCY_S``."""
        sd = self._sd
        rate = self._rate(device, False)
        stream = sd.OutputStream(
            device=device.index,
            channels=device.channels,
            samplerate=rate,
            dtype="float32",
            latency=OUTPUT_LATENCY if latency is None else latency,
            callback=make_callback(rate, device.channels),
            extra_settings=self._extra(),
        )
        try:
            stream.start()
        except Exception:
            _SoundDeviceStreams([stream]).close()
            raise
        return _SoundDeviceStreams([stream])

    def default_speakers(self) -> Device | None:
        """Windows' default output now, without listing devices. OBS looks the device up afresh on every (re)connect
        by asking only for the default endpoint (IMMDeviceEnumerator::GetDefaultAudioEndpoint, plugins/win-wasapi/
        win-wasapi.cpp:607-618), so it follows a default change. This asks WASAPI the same (clean_voice.endpoints) and
        finds that name in the last scan: listing again would re-initialise PortAudio while audio streams, which
        disturbs drivers (docs/simple-clean-mic-audio-report.md, cause 5). A default that appeared after the scan is
        not found until the next one (Off and On again). Elsewhere: the scanned default."""
        if sys.platform != "win32":
            return self.default_output()
        name = default_render_name()
        return next((d for d in self.outputs() if d.name == name), None) if name else None

    def open_loopback(self, device: Device, make_callback: CallbackFactory) -> Streams:
        """WASAPI loopback of ``device`` at its mix format (OBS captures output devices the same way,
        plugins/win-wasapi/win-wasapi.cpp:739-742, AUDCLNT_STREAMFLAGS_LOOPBACK)."""
        info = self._loopbacks.get(device.name)
        if self._pa is None or info is None:
            raise OSError(f"no loopback capture for {device.name}")
        rate, channels = int(info["defaultSampleRate"]), int(info["maxInputChannels"])
        return _LoopbackStream(
            self._pa, self._pyaudio, int(info["index"]), rate, channels, make_callback(rate, channels)
        )

    def _extra(self) -> Any:
        # No Windows auto-convert: like OBS, each device opens in its own (mix) format (plugins/win-wasapi/
        # win-wasapi.cpp:729, :817-831, GetMixFormat / InitFormat) and the app converts in-process (libobs/obs-source.c:
        # 4044-4052): the mic through AudioPath's libsamplerate, the cable and speakers through each sink's resampler.
        # PortAudio's WASAPI still maps mono and stereo itself (every device here accepts both).
        return None

    def _rate(self, device: Device, is_input: bool) -> int:
        """48 kHz when the device's own format is 48 kHz, else its mix rate (``default_rate``)."""
        check = self._sd.check_input_settings if is_input else self._sd.check_output_settings
        try:
            check(device=device.index, channels=device.channels, samplerate=SAMPLE_RATE, extra_settings=self._extra())
            return SAMPLE_RATE
        except Exception:  # noqa: BLE001 - PortAudio reports unsupported settings as a generic error
            return int(device.default_rate)

    def open(self, mic: Device, cable: Device, make_path: PathFactory) -> tuple[AudioPath, MainStreams]:
        sd = self._sd
        in_rate = self._rate(mic, True)
        out_rate = self._rate(cable, False)
        path = make_path(in_rate, mic.channels, out_rate, cable.channels)
        opened: list[Any] = []
        try:
            out_stream = sd.OutputStream(
                device=cable.index,
                channels=cable.channels,
                samplerate=out_rate,
                dtype="float32",
                latency=CABLE_LATENCY_S,
                callback=path.output_callback,
                extra_settings=self._extra(),
            )
            opened.append(out_stream)
            in_stream = sd.InputStream(
                device=mic.index,
                channels=mic.channels,
                samplerate=in_rate,
                dtype="float32",
                latency=INPUT_LATENCY_S,
                callback=path.input_callback,
                extra_settings=self._extra(),
            )
            opened.append(in_stream)
            out_stream.start()
            in_stream.start()
        except Exception:
            _SoundDeviceStreams(opened).close()
            raise
        return path, _SoundDeviceStreams(opened, output=out_stream)
