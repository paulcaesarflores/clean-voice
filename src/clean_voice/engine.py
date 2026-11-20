"""The engine: one worker thread that owns the streams, runs the audio path and handles devices coming and going.

The UI talks to it through ``start``, ``stop``, ``select_mic``, ``select_output``, ``refresh``, ``set_recording``,
``set_monitor``, ``set_echo``, ``start_mic_test`` and ``cancel_mic_test`` (queued commands, because they open or close devices), hands it
sound settings with ``set_settings`` (a lock-free reference swap) and reads ``status()``. Devices are listed again only
while no stream is open (at start, on request, and every 3 s while waiting for a device, like OBS's reconnect thread in
plugins/win-wasapi/win-wasapi.cpp), never by polling while audio runs, which disturbed drivers in the original app
(docs/simple-clean-mic-audio-report.md, cause 5). Each device recovers on its own, as each OBS source does
(win-wasapi.cpp:959-993, :1091-1224): the speakers (Hear myself, the mic test, echo removal) have their own streams and
watchdogs, and a stalled cable is reopened on its own while the mic and the backup keep running. Echo removal looks the
default speakers up afresh on every (re)open, so it follows Windows to a new default (win-wasapi.cpp:607-636,
:1237-1265).
"""

from __future__ import annotations

import enum
import logging
import queue
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clean_voice.audio_path import BLOCK, SAMPLE_RATE, AudioPath, Counters, MicTestState
from clean_voice.devices import (
    CABLE_LATENCY_S,
    Backend,
    Device,
    MainStreams,
    Streams,
    find_cable,
    is_cable,
    microphones,
    monitor_device,
)
from clean_voice.dsp import Chain
from clean_voice.permissions import mic_permission
from clean_voice.settings import Settings, SettingsCell
from clean_voice.vst import VstRegistry
from clean_voice.wav import Recorder

log = logging.getLogger(__name__)

RETRY_S = 3.0  # OBS win-wasapi.cpp reconnects every 3 s
WATCHDOG_S = 1.0  # no audio from a device for this long means it is gone
# ...but a stream gets this long for its first callback: a Bluetooth headset switching to its hands-free profile took
# over 1 s, so the mic was called lost 2 s after start and reopened 3 s later (engine-obs r1, m3).
FIRST_AUDIO_S = 5.0
WORKER_WAIT_S = 0.002
MONITOR_CLOSE_S = 0.5  # after the mic test's playback, how long the drained speaker stream stays open at most

# What the screen says (plain words; simple-clean-mic ui/main.js:614-617 for the echo ones).
ECHO_PAUSED = "Paused during Hear myself: it would take your own voice out too."
ECHO_NO_SPEAKERS = "Not working right now: Clean Voice can't reach your speakers to hear the echo."
ECHO_MAC = "Windows only. On a Mac, wear headphones so the call doesn't hear itself."
MONITOR_NO_SPEAKERS = "Clean Voice can't find headphones or speakers to play your voice on."
MONITOR_LOST = "Your headphones or speakers stopped, so Hear myself turned off."
MIC_TEST_NO_SPEAKERS = "Clean Voice can't find speakers or headphones to play the test back."
MIC_TEST_NOT_ON = "Turn Clean Voice on first, then test your mic."


def _init_device_thread() -> None:
    """WASAPI streams can only be opened from a thread that has COM initialised: from a fresh thread both PortAudio
    copies failed with "Unanticipated host error" (J06 Step 10). The device thread initialises it once, like the
    thread that first loads PortAudio."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ole32 = ctypes.WinDLL("ole32")  # type: ignore[attr-defined]
        hr = ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED
        if hr not in (0, 1):  # S_OK, S_FALSE (already initialised)
            log.warning("COM initialisation on the device thread returned 0x%08x", hr & 0xFFFFFFFF)
    except OSError as exc:
        log.warning("COM initialisation on the device thread failed: %s", exc)


def echo_platform_reason() -> str | None:
    """Why echo removal cannot run on this platform, or None. v1: Windows only (macOS would need Core Audio taps or
    ScreenCaptureKit and a permission prompt; orchestrator decision, J06 plan Open question 2)."""
    return None if sys.platform == "win32" else ECHO_MAC


class State(enum.Enum):
    OFF = "off"
    ON = "on"
    NO_CABLE = "no_cable"
    NO_MIC = "no_mic"
    MIC_DENIED = "mic_denied"
    MIC_LOST = "mic_lost"
    CABLE_LOST = "cable_lost"
    ERROR = "error"


class EchoState(enum.Enum):
    OFF = "off"  # the default
    STARTING = "starting"  # asked for: waiting for the audio to start, or for the speakers' streams to open
    ON = "on"
    PAUSED = "paused"  # while Hear myself is on: the speakers carry the guest's own voice (audio.rs:429-431)
    UNAVAILABLE = "unavailable"  # no speakers to listen to; echo_reason says why


@dataclass
class Status:
    state: State = State.OFF
    mic: str | None = None
    cable: str | None = None  # VB-CABLE's playback side, if installed (whether or not the voice goes there)
    output: str | None = None  # where the clean voice goes: the cable unless the guest chose another output
    level: float = 0.0  # peak of the clean voice since the last status() call, 0..1
    input_level: float = 0.0  # peak of the raw mic since the last status() call, 0..1 (the "Your mic" meter)
    recording: bool = False  # a backup pair is being written now
    recording_enabled: bool = False  # the backup switch (off by default, the user's decision of 2026-11-11)
    recording_seconds: float = 0.0
    recording_files: tuple[Path, ...] = ()
    recording_error: str | None = None
    error: str | None = None
    counters: Counters = field(default_factory=Counters)
    monitor: bool = False  # Hear myself (off at start, main.js:19)
    monitor_device: str | None = None
    monitor_error: str | None = None
    echo: EchoState = EchoState.OFF
    echo_reason: str | None = None  # plain words for the screen when PAUSED or UNAVAILABLE
    echo_delay_ms: float = 0.0  # how much the mic is held back so the speaker sound reaches AEC3 first
    mic_test: MicTestState = MicTestState.IDLE
    mic_test_seconds: float = 0.0  # how far into the recording or the playback
    mic_test_error: str | None = None


def _boost_thread_priority() -> bool | None:
    """Ask Windows to schedule the worker like audio (MMCSS "Pro Audio"), as OBS does for its capture thread
    (plugins/win-wasapi/win-wasapi.cpp:1105, AvSetMmThreadCharacteristics). Returns whether it worked (None: not
    Windows); the result is logged instead of ignored."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        avrt = ctypes.WinDLL("avrt", use_last_error=True)  # type: ignore[attr-defined]
        avrt.AvSetMmThreadCharacteristicsW.restype = ctypes.c_void_p
        task = ctypes.c_ulong(0)
        handle = avrt.AvSetMmThreadCharacteristicsW("Pro Audio", ctypes.byref(task))
    except OSError as exc:
        log.warning("MMCSS: could not register the audio thread (%s)", exc)
        return False
    if not handle:
        error = ctypes.get_last_error()  # type: ignore[attr-defined]
        log.warning("MMCSS: could not register the audio thread (error %d)", error)
        return False
    log.info("MMCSS: the audio thread runs as Pro Audio (task %d)", task.value)
    return True


class Engine:
    def __init__(
        self,
        recordings_dir: Path,
        backend: Backend | None = None,
        retry_s: float = RETRY_S,
        watchdog_s: float = WATCHDOG_S,
    ) -> None:
        self.recordings_dir = recordings_dir
        self._backend_arg = backend
        self._backend: Backend | None = None
        self._retry_s = retry_s
        self._watchdog_s = watchdog_s
        self._commands: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self._lock = threading.Lock()  # guards _status and _mics between the worker and the UI
        self._status = Status()
        self._mics: list[str] = []
        self._outputs: list[str] = []
        self._default_mic: str | None = None
        # Worker-thread state.
        self._record = False  # the backup switch, as the worker last applied it (off until switched on)
        self._backup_failed = False  # the last attempt to open a backup pair failed
        self._want_on = False
        self._mic_name: str | None = None
        self._output_name: str | None = None  # None: VB-CABLE
        self._had_mic = False
        self._path: AudioPath | None = None
        self._streams: MainStreams | None = None
        # The cable, reopened on its own after a stall (None while it is inside _streams, or closed for a retry).
        self._cable_stream: Streams | None = None
        self._cable_down = False  # the cable's stream is closed and retried every RETRY_S; the mic goes on
        self._await_mic = False  # a retried mic: MIC_LOST until its first callback
        self._cable_retry_at = 0.0
        self._recorder: Recorder | None = None
        self._backup_gap_from: float | None = None  # when the path closed with a live recorder; None: no gap to fill
        self._retry_at = 0.0
        self._last_in = (0, 0.0)
        self._last_out = (0, 0.0)
        self._totals = Counters()
        self._cell = SettingsCell()  # handed to every chain, so settings survive reconnects
        # Hear myself and the mic test share one stream on the speakers.
        self._monitor_on = False  # the guest's Hear myself choice (off at start, main.js:19)
        self._monitor_volume = 1.0  # Hear myself volume as a gain: the speakers only, never the cable
        # Fine-tune's loaded VST plug-ins, shared with every chain: the UI thread loads and unloads them (J10 §3, §5).
        self.vst = VstRegistry()
        self._monitor_stream: Streams | None = None
        self._monitor_last = (0, 0.0)
        self._monitor_finishing_since: float | None = None
        # Remove speaker echo: the loopback of the speakers and the silent render that keeps it flowing (W1).
        self._echo_on = False  # the guest's choice (off by default)
        self._echo_streams: list[Streams] = []
        self._echo_last = (0, 0.0)
        self._echo_retry_at = 0.0
        # The speakers failed (would not open, or the loopback stalled): Status.echo stays UNAVAILABLE through the
        # retries until a loopback really delivers, so the screen's reason text does not blink (J06 review r2, M2).
        self._echo_failed = False
        self._echo_opening: tuple[Future[list[Streams]], dict[str, Any], AudioPath] | None = None
        self._monitor_opening: tuple[Future[Streams], dict[str, int], Device, AudioPath] | None = None
        self._devices: ThreadPoolExecutor | None = None
        self.mmcss: bool | None = None  # whether the worker got MMCSS "Pro Audio" (None: not Windows, or not yet)
        self._thread = threading.Thread(target=self._run, name="clean-voice-audio", daemon=True)
        self._thread.start()
        self._commands.put(("refresh", None))

    # ------------------------------------------------------------------ UI side

    def mics(self) -> list[str]:
        with self._lock:
            return list(self._mics)

    def default_mic(self) -> str | None:
        with self._lock:
            return self._default_mic

    def outputs(self) -> list[str]:
        """Every output device, for "Send clean voice to" (index.html:115), from the last scan."""
        with self._lock:
            return list(self._outputs)

    def start(self, mic: str | None) -> None:
        self._commands.put(("start", mic))

    def stop(self) -> None:
        self._commands.put(("stop", None))

    def select_mic(self, mic: str) -> None:
        self._commands.put(("select", mic))

    def select_output(self, output: str | None) -> None:
        """Where the clean voice goes (None: VB-CABLE). While on, the streams reopen to it, as for a new mic."""
        self._commands.put(("select_output", output))

    def refresh(self) -> None:
        self._commands.put(("refresh", None))

    def set_recording(self, on: bool) -> None:
        """Switch the backup on or off. Off closes the current pair (final and readable) and the voice keeps going;
        on while running starts a new pair. Reconnects keep the current pair."""
        self._commands.put(("record", "on" if on else "off"))

    def set_settings(self, settings: Settings) -> None:
        """Hand new sound settings to the worker: one reference assignment, no queue and no lock. The chain picks
        them up at its next block and every change is ramped, so it is click-free."""
        self._cell.value = settings

    def settings(self) -> Settings:
        return self._cell.value

    def set_monitor(self, on: bool) -> None:
        """Hear myself: the clean voice on the guest's headphones or speakers. Never reopens the mic or cable."""
        self._commands.put(("monitor", "on" if on else "off"))

    def set_monitor_volume(self, volume: float) -> None:
        """Hear myself volume, 0..100 %: the level of Hear myself on the speakers (ramped over 10 ms). The mic test's
        playback stays at full level, and the voice to the cable never changes. An audio taper, gain = (percent/100)^2: 50 % is
        about -12 dB, roughly half as loud; 0 % is silent and 100 % unity."""
        gain = (min(max(float(volume), 0.0), 100.0) / 100.0) ** 2
        self._commands.put(("monitor_volume", repr(gain)))

    def set_echo(self, on: bool) -> None:
        """Remove speaker echo (off by default). Status.echo says whether it runs, is paused or is unavailable."""
        self._commands.put(("echo", "on" if on else "off"))

    def start_mic_test(self) -> None:
        """Test your mic: record 5 s of what others hear, then play it on the speakers."""
        self._commands.put(("mic_test", "start"))

    def cancel_mic_test(self) -> None:
        self._commands.put(("mic_test", "cancel"))

    def status(self) -> Status:
        path = self._path
        with self._lock:
            status = Status(**vars(self._status))
        if path is not None and status.state is State.ON:
            status.input_level, status.level = path.take_peaks()
        return status

    def close(self) -> None:
        self._commands.put(("quit", None))
        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            self.vst.close_all()  # nothing runs a plug-in any more

    # ------------------------------------------------------------------ worker thread

    def _run(self) -> None:
        self.mmcss = _boost_thread_priority()
        # Hand the GIL over every 1 ms instead of 5 ms, so a device callback never waits long for it.
        sys.setswitchinterval(0.001)
        try:
            self._backend = self._backend_arg
            if self._backend is None:
                from clean_voice.devices import SoundDeviceBackend

                self._backend = SoundDeviceBackend()
        except Exception as exc:  # no sound system at all: show it instead of crashing
            log.exception("sound system unavailable")
            self._update(state=State.ERROR, error=f"The sound system could not start ({exc}).")
            return
        while True:
            cmd, arg = self._next_command()
            if cmd == "quit":
                self._close_streams()
                self._close_recorder()
                if self._devices is not None:
                    self._devices.shutdown(wait=True)
                return
            try:
                self._handle(cmd, arg)
                if self._streams is not None:
                    self._pump()
                elif self._want_on and time.monotonic() >= self._retry_at:
                    self._scan()
                    self._open()
            except Exception as exc:  # keep the worker alive whatever a device does
                log.exception("audio engine error")
                self._close_streams()  # never raises (see there)
                self._retry_at = time.monotonic() + self._retry_s
                self._update(state=State.ERROR, error=str(exc))

    def _next_command(self) -> tuple[str, str | None]:
        if self._streams is None:
            try:
                return self._commands.get(timeout=0.1)
            except queue.Empty:
                return "", None
        # While audio runs: a timed queue wait rounds up to Windows' 15.6 ms timer tick (measured: median 15.7 ms,
        # max 31 ms), while time.sleep uses a high-resolution timer since Python 3.11.
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            time.sleep(WORKER_WAIT_S)
            return "", None

    def _handle(self, cmd: str, arg: str | None) -> None:
        if cmd == "start":
            self._want_on = True
            self._mic_name = arg
            self._had_mic = False
            self._close_streams()
            self._scan()
            # A saved choice that is not plugged in now falls back to the default (main.js:435-450); only a device
            # lost mid-session is waited for (MIC_LOST).
            if self._mic_name not in self.mics():
                self._mic_name = None
            if self._output_name not in self.outputs():
                self._output_name = None
            self._open()
        elif cmd == "select":
            self._mic_name = arg
            self._update(mic=arg)
            if self._want_on:
                self._close_streams()
                self._open()
        elif cmd == "select_output":
            self._output_name = arg
            if self._want_on:
                self._close_streams()
                self._open()
        elif cmd == "stop":
            self._want_on = False
            self._close_streams()
            self._close_recorder()
            self._update(state=State.OFF, error=None, recording=False)
        elif cmd == "refresh" and self._streams is None:
            self._scan()
        elif cmd == "record":
            self._set_recording(arg == "on")
        elif cmd == "monitor":
            self._set_monitor(arg == "on")
        elif cmd == "monitor_volume" and arg is not None:
            self._monitor_volume = float(arg)
            if self._path is not None:
                self._path.monitor_volume = self._monitor_volume
        elif cmd == "echo":
            self._echo_on = arg == "on"
            self._echo_retry_at = 0.0
            self._echo_failed = False
            self._apply_echo()
        elif cmd == "mic_test":
            self._mic_test(arg == "start")

    # ------------------------------------------------------------------ the speakers: Hear myself and the mic test

    def _speakers(self) -> Device | None:
        """Where Hear myself and the mic test play: Windows' default output now (asked for without listing devices,
        as OBS's InitDevice asks for the default endpoint, win-wasapi.cpp:607-618), so they follow a default change
        like echo removal; else another output from the last scan."""
        backend = self._backend
        assert backend is not None
        # Never the output the clean voice goes to (main.js:453-458); monitor_device also skips every cable.
        # A default unknown to the scan (plugged in since): the default Windows had at the scan, before the first
        # listed output, which can be a monitor's HDMI output with no speakers (engine-obs r2, n1).
        default = backend.default_speakers() or backend.default_output()
        return monitor_device(backend.outputs(), default, self._find_output())

    # Opening and closing the speaker streams can take a while (a Bluetooth headset: hundreds of ms). Done on the worker
    # thread, it starved the cable: J06 Step 10's first soak had an underrun when echo removal and Hear myself came on.
    # So the device thread opens and closes them, and the worker only wires the path once a stream is open (so no sink,
    # reference or AEC3 state is ever touched from two threads). The callbacks that run before then play silence and
    # drop the loopback. When the main streams close, the side streams close first, and synchronously.

    def _device_thread(self) -> ThreadPoolExecutor:
        if self._devices is None:
            self._devices = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="clean-voice-devices", initializer=_init_device_thread
            )
        return self._devices

    def _close_later(self, streams: list[Streams], sync: bool) -> None:
        def close() -> None:
            for s in streams:
                try:
                    s.close()
                except Exception:
                    log.exception("closing a speaker stream failed")

        if sync:
            close()
        else:
            self._device_thread().submit(close)

    def _open_monitor(self) -> bool:
        """Start opening the speaker stream (if it is not open or opening). False: no speakers to use."""
        backend, path = self._backend, self._path
        assert backend is not None and path is not None
        if self._monitor_stream is not None or self._monitor_opening is not None:
            return True
        device = self._speakers()
        if device is None:
            return False
        info: dict[str, int] = {}

        def make_callback(rate: int, channels: int) -> object:
            info["rate"], info["channels"] = rate, channels
            return path.monitor_callback  # silence until the worker adds the sink

        job = self._device_thread().submit(backend.open_output, device, make_callback)  # type: ignore[arg-type]
        self._monitor_opening = (job, info, device, path)
        self._update(monitor_device=device.name)
        return True

    def _finish_monitor_open(self, now: float) -> None:
        opening = self._monitor_opening
        if opening is None or not opening[0].done():
            return
        self._monitor_opening = None
        job, info, device, path = opening
        try:
            stream = job.result()
        except Exception as exc:  # noqa: BLE001 - the speakers failing must never stop the voice
            log.warning("could not open %s for Hear myself / the mic test: %s", device.name, exc)
            self._monitor_lost(MONITOR_NO_SPEAKERS)
            return
        wanted = self._monitor_on or (path.mic_test.state is not MicTestState.IDLE)
        if path is not self._path or not wanted:
            self._close_later([stream], sync=False)  # superseded (a reconnect, or switched off meanwhile)
            return
        path.add_monitor(info["rate"], info["channels"])
        path.monitor_live = self._monitor_on
        self._monitor_stream = stream
        self._monitor_last = (0, now)
        self._monitor_finishing_since = None

    def _close_monitor(self, sync: bool = False) -> None:
        stream, self._monitor_stream = self._monitor_stream, None
        if self._path is not None:
            self._path.monitor_live = False
            self._path.remove_monitor()
        opening, self._monitor_opening = self._monitor_opening, None
        if opening is not None:
            self._abandon(opening[0], sync)
        if stream is not None:
            self._close_later([stream], sync)

    def _abandon(self, job: Future[Any], sync: bool) -> None:
        """A stream (or two) still opening that is no longer wanted: close it once it is open."""

        def close(done: Future[Any]) -> None:
            if done.cancelled() or done.exception() is not None:
                return
            result = done.result()
            self._close_later(result if isinstance(result, list) else [result], sync=True)

        if sync:
            try:
                job.result(timeout=10)
            except Exception:
                log.debug("a speaker stream that was opening failed", exc_info=True)
            close(job)
        else:
            job.add_done_callback(close)

    def _monitor_lost(self, error: str) -> None:
        path = self._path
        testing = path is not None and path.mic_test.state is not MicTestState.IDLE
        was_on = self._monitor_on
        if path is not None:
            path.mic_test.cancel()
        self._close_monitor()
        self._monitor_on = False
        # Only say Hear myself stopped if it was on (when just the mic test used the speakers, only the test says so).
        self._update(monitor=False, monitor_error=error if was_on else None, mic_test=MicTestState.IDLE)
        if testing:
            self._update(mic_test_error=MIC_TEST_NO_SPEAKERS)
        self._apply_echo()

    def _set_monitor(self, on: bool) -> None:
        self._monitor_on = on
        path = self._path
        if path is None or self._streams is None:
            self._update(monitor=False, monitor_error=None)
            self._apply_echo()
            return
        if on:
            if not self._open_monitor():
                self._monitor_on = False
                self._update(monitor=False, monitor_error=MONITOR_NO_SPEAKERS)
                return
            path.monitor_live = True
            self._update(monitor=True, monitor_error=None)
        else:
            path.monitor_live = False
            if path.mic_test.state is MicTestState.IDLE:
                self._close_monitor()
            self._update(monitor=False, monitor_error=None)
        self._apply_echo()  # echo removal pauses while Hear myself is on

    def _mic_test(self, start: bool) -> None:
        path = self._path
        if not start:
            if path is not None:
                path.mic_test.cancel()
                if not self._monitor_on:
                    self._close_monitor()
            self._update(mic_test=MicTestState.IDLE, mic_test_seconds=0.0)
            return
        if path is None or self._streams is None:
            self._update(mic_test_error=MIC_TEST_NOT_ON)
            return
        if not self._open_monitor():
            self._update(mic_test_error=MIC_TEST_NO_SPEAKERS, mic_test=MicTestState.IDLE)
            return
        path.mic_test.start()
        self._update(mic_test=MicTestState.RECORDING, mic_test_seconds=0.0, mic_test_error=None)

    def _pump_monitor(self, path: AudioPath, now: float) -> None:
        self._finish_monitor_open(now)
        sink = path.monitor
        test = path.mic_test
        self._update(mic_test=test.state, mic_test_seconds=test.seconds)
        if self._monitor_stream is None or sink is None:
            return
        if sink.callbacks != self._monitor_last[0]:
            self._monitor_last = (sink.callbacks, now)
        if now - self._monitor_last[1] > self._watchdog_s:
            # The headphones or speakers went away: Hear myself and any mic test end; the voice to Meet goes on.
            log.warning("the speaker stream stopped; Hear myself is off")
            self._monitor_lost(MONITOR_LOST)
            return
        if not self._monitor_on and test.state is MicTestState.IDLE and sink.finishing:
            # The test's playback is over: close once its last audio has played (or after MONITOR_CLOSE_S).
            if self._monitor_finishing_since is None:
                self._monitor_finishing_since = now
            if sink.ring.available() == 0 or now - self._monitor_finishing_since > MONITOR_CLOSE_S:
                self._close_monitor()
        else:
            self._monitor_finishing_since = None

    # ------------------------------------------------------------------ the speakers: Remove speaker echo

    @staticmethod
    def _echo_speakers(backend: Backend) -> Device:
        """The speakers to listen to: the default output as Windows has it now (device thread: it may list devices)."""
        device = backend.default_speakers()
        if device is None or is_cable(device.name):
            raise OSError("no speakers to listen to (no default output, or it is the cable)")
        return device

    def _close_echo(self, sync: bool = False) -> None:
        streams, self._echo_streams = self._echo_streams, []
        if self._path is not None:
            self._path.disable_echo()
        opening, self._echo_opening = self._echo_opening, None
        if opening is not None:
            self._abandon(opening[0], sync)
        if streams:
            self._close_later(streams, sync)

    def _apply_echo(self) -> None:
        """Bring echo removal in line with the guest's choice, Hear myself, and what the speakers allow."""
        path, backend = self._path, self._backend
        if not self._echo_on:
            self._close_echo()
            self._update(echo=EchoState.OFF, echo_reason=None, echo_delay_ms=0.0)
            return
        reason = echo_platform_reason()
        if reason is not None:
            self._close_echo()
            self._update(echo=EchoState.UNAVAILABLE, echo_reason=reason)
            return
        if path is None or self._streams is None or backend is None:
            if not self._echo_failed:
                self._update(echo=EchoState.STARTING, echo_reason=None, echo_delay_ms=0.0)  # starts with the audio
            return
        if not self._echo_streams:
            if self._echo_opening is None and time.monotonic() >= self._echo_retry_at:
                self._open_echo(path, backend)
                if not self._echo_failed:  # a retry keeps UNAVAILABLE and its reason (M2)
                    self._update(echo=EchoState.STARTING, echo_reason=None)
            return  # ON or PAUSED once the streams are open
        paused = self._monitor_on
        path.set_echo_paused(paused)
        if self._echo_failed:
            return  # UNAVAILABLE until the reopened loopback has delivered (_pump_echo)
        self._update(echo=EchoState.PAUSED if paused else EchoState.ON, echo_reason=ECHO_PAUSED if paused else None)

    def _open_echo(self, path: AudioPath, backend: Backend) -> None:
        """Start opening the loopback of the speakers and, on the same speakers, the silent render that keeps it
        delivering during silence. The running render is Clean Voice's own, for AEC3's gap-free reference: OBS only
        fills one render buffer with zeros and never starts it (win-wasapi.cpp:751-797, :883-885)."""
        info: dict[str, Any] = {}

        def loopback(rate: int, channels: int) -> object:
            info["rate"], info["channels"] = rate, channels
            return path.loopback_callback  # dropped until the worker creates the echo reference

        def render(rate: int, channels: int) -> object:
            info["render_rate"] = rate
            return path.render_callback  # silence until the worker creates the marker

        def open_both() -> list[Streams]:
            device = self._echo_speakers(backend)  # afresh on every try, as OBS's InitDevice
            info["device"] = device.name
            loop = backend.open_loopback(device, loopback)  # type: ignore[arg-type]
            try:
                return [loop, backend.open_output(device, render)]  # type: ignore[arg-type]
            except Exception:
                loop.close()
                raise

        self._echo_opening = (self._device_thread().submit(open_both), info, path)

    def _finish_echo_open(self, now: float) -> None:
        opening = self._echo_opening
        if opening is None or not opening[0].done():
            return
        self._echo_opening = None
        job, info, path = opening
        try:
            streams = job.result()
        except Exception as exc:  # noqa: BLE001 - echo removal failing must never stop the voice
            log.warning("echo removal could not listen to %s: %s", info.get("device", "the speakers"), exc)
            self._echo_retry_at = now + self._retry_s
            self._echo_failed = True
            self._update(echo=EchoState.UNAVAILABLE, echo_reason=ECHO_NO_SPEAKERS)
            return
        if path is not self._path or not self._echo_on:
            self._close_later(streams, sync=False)
            return
        path.enable_echo(info["rate"], info["channels"])  # on the worker thread: AEC3 is created here
        if path.render is not None:
            path.render.set_rate(info.get("render_rate", info["rate"]))
        self._echo_streams = streams
        self._echo_last = (0, now)
        log.info("echo removal on: listening to %s", info["device"])
        self._apply_echo()

    def _pump_echo(self, path: AudioPath, now: float) -> None:
        self._finish_echo_open(now)
        if not self._echo_on:
            return
        if not self._echo_streams:
            if self._echo_opening is None and now >= self._echo_retry_at and echo_platform_reason() is None:
                self._apply_echo()  # retry every RETRY_S (the loopback was lost, or would not open)
            return
        ref = path.reference
        count = ref.stamps.count if ref is not None else 0
        if count != self._echo_last[0]:
            self._echo_last = (count, now)
            if self._echo_failed:  # the retried loopback really delivers: ON (or PAUSED) again
                self._echo_failed = False
                self._apply_echo()
        if now - self._echo_last[1] > self._watchdog_s:
            # The loopback stalled even with the silent render running: close only these two streams and retry;
            # the mic and cable streams are untouched.
            log.warning("the speakers' loopback stopped; echo removal retries every %g s", self._retry_s)
            self._close_echo()
            self._echo_retry_at = now + self._retry_s
            self._echo_failed = True
            self._update(echo=EchoState.UNAVAILABLE, echo_reason=ECHO_NO_SPEAKERS)
            return
        stage = path.echo
        self._update(echo_delay_ms=stage.delay_ms if stage is not None else 0.0)

    def _scan(self) -> None:
        backend = self._backend
        assert backend is not None
        backend.rescan()
        mics = [d.name for d in microphones(backend)]
        outputs = [d.name for d in backend.outputs()]
        default = backend.default_input()
        cable = find_cable(backend)
        with self._lock:
            self._mics = mics
            self._outputs = outputs
            self._default_mic = default.name if default and default.name in mics else (mics[0] if mics else None)
            self._status.cable = cable.name if cable else None

    def _find_mic(self) -> Device | None:
        backend = self._backend
        assert backend is not None
        mics = microphones(backend)
        name = self._mic_name or self.default_mic()
        return next((d for d in mics if d.name == name), None)

    def _find_output(self) -> Device | None:
        """The guest's chosen output if it is listed, else VB-CABLE."""
        backend = self._backend
        assert backend is not None
        chosen = next((d for d in backend.outputs() if d.name == self._output_name), None)
        return chosen or find_cable(backend)

    def _open(self) -> None:
        backend = self._backend
        assert backend is not None
        now = time.monotonic()
        self._retry_at = now + self._retry_s
        cable = self._find_output()
        if cable is None:
            self._update(state=State.CABLE_LOST if self._had_mic else State.NO_CABLE, output=None)
            return
        mic = self._find_mic()
        if mic is None:
            self._update(state=State.MIC_LOST if self._had_mic else State.NO_MIC, output=cable.name)
            return
        # Check the OS's microphone permission before opening, as OBS's CheckPermission (frontend/utility/platform-osx.mm:208-231).
        if mic_permission() == "denied":
            self._update(state=State.MIC_DENIED, output=cable.name)
            return

        def make_path(in_rate: int, in_channels: int, out_rate: int, out_channels: int) -> AudioPath:
            chain = Chain(self._cell, block=BLOCK, vst=self.vst)
            path = AudioPath(in_rate, in_channels, out_rate, out_channels, chain)
            path.monitor_volume = self._monitor_volume
            return path

        self._path, self._streams = backend.open(mic, cable, make_path)
        # The backup files are made only once the devices are open, and kept across reconnects. Only this thread
        # runs the path's worker step, so attaching the recorder after the streams started is safe.
        if self._recorder is None and self._record:
            self._recorder = self._new_recorder()
        elif self._recorder is not None and self._backup_gap_from is not None:
            # The time the mic was away is silence in the file: OBS's mix stays on the clock and a source that is gone
            # adds nothing (libobs/media-io/audio-io.c audio_thread, input_and_output; libobs/obs-audio.c).
            self._recorder.write_silence(round((time.monotonic() - self._backup_gap_from) * SAMPLE_RATE))
        self._backup_gap_from = None
        recorder = self._recorder
        self._path.recorder = recorder
        self._had_mic = True
        self._mic_name = mic.name
        self._last_in = (0, now)
        self._last_out = (0, now)
        self._cable_down = False
        # A retry after the mic was lost says ON only once the reopened mic delivers: a dead mic that opens but never
        # calls back would otherwise show "On" for FIRST_AUDIO_S of every retry (engine-obs r2, n5). A retry after an
        # error keeps showing the error the same way: an audio step that fails every time showed "On" from here until
        # it failed again (CI run 43368578097).
        prior = self._status.state
        retrying = prior in (State.MIC_LOST, State.ERROR)
        self._await_mic = retrying
        log.info("audio on: %s -> %s", mic.name, cable.name)
        self._update(
            state=prior if retrying else State.ON,
            mic=mic.name,
            output=cable.name,
            error=self._status.error if prior is State.ERROR else None,
            recording=recorder is not None and recorder.active,
            recording_error=self._recording_error(),
            **({"recording_files": recorder.files} if recorder else {}),
        )
        # Hear myself and echo removal carry on across reconnects.
        if self._monitor_on:
            self._set_monitor(True)
        self._echo_retry_at = 0.0
        self._apply_echo()

    def _pump(self) -> None:
        path, streams = self._path, self._streams
        assert path is not None and streams is not None
        path.process()
        now = time.monotonic()
        written, callbacks = path.in_ring.total_written, path.output_callbacks
        if written != self._last_in[0]:
            self._last_in = (written, now)
            if self._await_mic:
                self._await_mic = False
                self._update(state=State.CABLE_LOST if self._cable_down else State.ON, error=None)
        if callbacks != self._last_out[0]:
            self._last_out = (callbacks, now)
        recorder = self._recorder
        self._update(
            counters=self._combined_counters(path),
            recording=recorder is not None and recorder.active,
            recording_error=self._recording_error(),
            recording_seconds=recorder.frames_total / SAMPLE_RATE if recorder else 0.0,
            # A long session rolls over to a new pair before the 4 GiB WAV limit.
            **({"recording_files": recorder.files} if recorder else {}),
        )
        self._pump_monitor(path, now)
        self._pump_echo(path, now)
        mic_stalled = now - self._last_in[1] > self._allowed(self._last_in[0])
        # Each stream's own state decides (OBS win-wasapi.cpp:959-993, :1089-1224): a dead cable closes only the cable.
        cable_stream = self._cable_stream
        cable_active = cable_stream.active if cable_stream is not None else streams.output_active
        cable_dead = not self._cable_down and (
            now - self._last_out[1] > self._allowed(self._last_out[0]) or not cable_active
        )
        if mic_stalled or not streams.input_active:
            self._lost()
        elif cable_dead:
            self._cable_lost(now)
        elif self._cable_down and now >= self._cable_retry_at:
            self._reopen_cable(path, now)

    def _allowed(self, seen: int) -> float:
        """How long a stream may stay silent: the watchdog once it has delivered, FIRST_AUDIO_S before its first
        callback (a Bluetooth headset's switch to hands-free)."""
        return self._watchdog_s if seen else max(self._watchdog_s, FIRST_AUDIO_S)

    def _cable_lost(self, now: float) -> None:
        """Only the cable stopped: close its stream and retry it on its own; the mic and the backup go on (OBS: one
        source failing never stops another, win-wasapi.cpp:1091-1224)."""
        path, streams = self._path, self._streams
        assert path is not None and streams is not None
        cable, self._cable_stream = self._cable_stream, None
        try:
            if cable is not None:
                cable.close()
            else:
                streams.close_output()
        except Exception:
            log.exception("closing the cable's stream failed")
        path.detach_cable()
        self._cable_down = True
        self._cable_retry_at = now + self._retry_s
        log.warning("the cable stopped; the mic and backup go on, and the cable is retried every %.0f s", self._retry_s)
        self._update(state=State.CABLE_LOST)

    def _reopen_cable(self, path: AudioPath, now: float) -> None:
        """Open the cable again (the device from the last scan: sounddevice cannot list devices while the mic's stream
        is open, and the cable keeps its endpoint) with a fresh sink on the same path."""
        backend = self._backend
        assert backend is not None
        self._cable_retry_at = now + self._retry_s
        cable = self._find_output()
        if cable is None:
            return
        try:
            self._cable_stream = backend.open_output(
                cable,
                lambda rate, channels: path.attach_cable(rate, channels).callback,  # type: ignore[arg-type]
                latency=CABLE_LATENCY_S,  # the cable's own host buffer, as when it opened with the mic
            )
        except Exception as exc:  # noqa: BLE001 - the cable still busy: keep the mic going and retry
            path.detach_cable()  # the sink made for the failed stream: nothing reads it, so nothing may feed it
            log.warning("could not reopen %s: %s", cable.name, exc)
            return
        self._cable_down = False
        self._last_out = (path.output_callbacks, time.monotonic())
        log.info("the cable is back: %s", cable.name)
        self._update(state=State.ON, output=cable.name)

    def _lost(self) -> None:
        """A device stopped: close cleanly, find out which, and retry every few seconds."""
        self._close_streams()
        self._scan()
        backend = self._backend
        assert backend is not None
        state = State.CABLE_LOST if self._find_output() is None else State.MIC_LOST
        log.warning("audio device lost (%s); retrying every %.0f s", state.value, self._retry_s)
        self._retry_at = time.monotonic() + self._retry_s
        self._update(state=state)

    def _combined_counters(self, path: AudioPath) -> Counters:
        return self._totals + path.status_counters()  # by field name: a new counter is never left out

    def _close_streams(self) -> None:
        """Close the streams and drop the path. Never raises: it also runs after an error, and an exception here
        would end the worker thread (J05 review r1, F1)."""
        for close in (self._close_monitor, self._close_echo):
            try:
                close(sync=True)
            except Exception:
                log.exception("closing the speaker streams failed")
        if self._path is not None and self._monitor_stream is None:
            self._update(monitor=False)
        streams, self._streams = self._streams, None
        cable, self._cable_stream = self._cable_stream, None
        self._cable_down = False
        for s in (cable, streams):
            if s is None:
                continue
            try:
                s.close()
            except Exception:
                log.exception("closing the audio streams failed")
        path, self._path = self._path, None
        if path is not None:
            self._totals = self._combined_counters(path)
            try:
                path.process()  # what was already captured still reaches the backup
            except Exception:
                log.exception("the last audio step before closing failed")
            if self._recorder is not None:
                # The gap starts when the mic last delivered, not when the stall was noticed (see _open).
                self._backup_gap_from = self._last_in[1]
            fine = getattr(path.chain, "fine_tune", None)
            if fine is not None:
                fine.close()  # this chain runs no more: its VST plug-ins may be unloaded (vst.VstRegistry)
            log.info("audio stopped; totals %s", self._totals)
        try:
            self._apply_echo()  # nothing listens to the speakers now: echo says STARTING, never ON (J06 r2, M1)
        except Exception:
            log.exception("updating the echo state failed")

    def _new_recorder(self) -> Recorder | None:
        """A new backup pair, or None when the folder cannot be written: the voice goes on without a backup."""
        try:
            recorder = Recorder(self.recordings_dir)
        except Exception as exc:  # noqa: BLE001 - a missing, protected or full folder must never stop the voice (F2)
            log.warning("cannot save backups to %s: %s", self.recordings_dir, exc)
            self._backup_failed = True
            return None
        self._backup_failed = False
        return recorder

    def _recording_error(self) -> str | None:
        """What the screen says when the backup cannot be saved, in plain words."""
        recorder = self._recorder
        if not self._backup_failed and (recorder is None or recorder.error is None):
            return None
        if recorder is not None and recorder.error is not None:
            log.debug("backup error: %s", recorder.error)
        return (
            f"Can't save backups to {self.recordings_dir}. Your voice still reaches the call. Check that the disk has "
            "space and that the folder can be written."
        )

    def _set_recording(self, on: bool) -> None:
        self._record = on
        if not on:
            if self._path is not None:
                self._path.recorder = None  # this thread runs the path's worker step: no block is half written
            self._close_recorder()
            self._backup_failed = False
            self._update(recording_enabled=False, recording=False, recording_error=None)
            return
        update: dict[str, object] = {"recording_enabled": True}
        if self._streams is not None and self._path is not None and self._recorder is None:
            self._recorder = self._new_recorder()
            self._path.recorder = self._recorder
            if self._recorder is not None:
                update.update(recording=True, recording_files=self._recorder.files)
        self._update(recording_error=self._recording_error(), **update)

    def _close_recorder(self) -> None:
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None

    def _update(self, **changes: object) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self._status, key, value)
