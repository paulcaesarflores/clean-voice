"""Soak the real audio path on this computer's default mic and VB-CABLE, and report how it held up (J05).

Runs the real ``Engine`` with the real ``Chain`` for ``--seconds``. A subclass of ``AudioPath`` stamps every mic
callback, cable callback, worker step and chain block into preallocated arrays, which are saved as
``logs/audio/<stamp>.npz`` and summarized on stdout. The backup recordings go to ``logs/audio/<stamp> rec/``.

    python tools/audio_soak.py --seconds 300                  # engine only
    python tools/audio_soak.py --seconds 300 --with-ui        # the window open too (Qt on the GIL)
    python tools/audio_soak.py --seconds 300 --with-ui --fine-tune  # ...with Fine-tune's gate, EQ, compressor, gain
    python tools/audio_soak.py --seconds 120 --mic "Headset (USB)"   # another microphone than the default
    python tools/audio_soak.py --seconds 300 --listen-cable   # also record CABLE Output and count exact-zero runs
    python tools/audio_soak.py --seconds 30 --toggle-recording   # the backup switched off and on twice
    python tools/audio_soak.py --seconds 300 --listen-cable --echo      # with Remove speaker echo on
    python tools/audio_soak.py --seconds 300 --listen-cable --monitor   # with Hear myself on (muted: nothing plays)
    python tools/audio_soak.py --seconds 60 --settings preset=radio strength=100   # other sound settings
    python tools/audio_soak.py --analyze logs/audio/<stamp>.npz  # summarize a saved trace again
    python tools/audio_soak.py --export-fixture logs/audio/<stamp>.npz   # write tests/data/real_pattern.npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import clean_voice.engine as engine_mod
from clean_voice.audio_path import FADE_S, AudioPath, Processor
from clean_voice.engine import EchoState, Engine, State

_now = time.perf_counter
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "audio"
FIXTURE = ROOT / "tests" / "data" / "real_pattern.npz"
STARTUP_FIXTURE = ROOT / "tests" / "data" / "real_startup.npz"
SETTLE_S = 10.0  # the summary's "settled" figures leave out the first 10 s
SWING_FROM_S = 60.0  # also shown: the correction after the first minute
JUDGE_FROM_S = 10.0  # Step 2d is judged from 10 s, so the start-up is covered (J05 review r1, F4)
MAX_SPREAD_PPM = 300.0
# A mic measured losing audio (its estimated rate this far off nominal; a Bluetooth headset lost audio at -248 to
# -503 ppm) makes the correction follow it, so the spread is then information, not a failure (orchestrator
# decision, engine-obs r1). Underruns, overflows and unexplained zero runs stay hard checks.
MIC_LOSS_PPM = 300.0
# A zero run on the cable is quantization when the lined-up clean voice peaks at this many cable steps or less:
# VB-CABLE truncates toward zero, and the drift resampler's interpolation takes peaks just over one step below it
# (measured: zeros where the clean voice peaked at 1.00-1.75 steps, lined up at r = 0.89-1.00). Orchestrator decision.
QUIET_MAX_STEPS = 2.0
MAX_CHAIN_P99_MS = 3.0  # J06 Step 10
MAX_LOOP_GAP_MS = 100.0  # J06 Step 10: the loopback keeps delivering during silence (the silent render)
MAX_MEAN_OFF_PPM = 100.0
FIXTURE_S = 60.0
ZERO_RUN_S = 0.001  # exact-zero runs on CABLE Output at least this long are reported
PRINT_RUNS = 40  # the explained runs printed (every run is judged; every fault is printed)
MATCH_WINDOW_S = 0.25  # a zero run this close to a counted event is matched to it
QUIET_SPAN_S = 0.25  # how far back from a zero run on the cable the backup is searched (cushion + resampler)
ALIGN_S = 1.0  # the cable is lined up with the backup on this much audio before each zero run
# The start-up allowance is only the cable's own leading silence (the stream's start, the ring filling to its
# target, the chain's ~120 ms of warm-up zeros): the first zero run, LEAD_MIN_S or longer, starting within LEAD_RING_S
# of the cable's first non-zero sample (the resampler's ringing puts a few tiny samples ahead of it; measured on every
# trace: 252-292 ms, or 1.2 s on a Bluetooth headset that starts late). A hole after it is judged like any other
# (fine-tune-updates r1, Z2: a 1 s allowance hid real runs at 0.48 and 0.69 s).
LEAD_RING_S = 0.1
LEAD_MIN_S = 0.1
# VB-CABLE (or the Windows audio engine) inserts one period of silence within the first seconds after a render stream
# into CABLE Input starts, and runs that period later from then on. A bare tone shows it too, with no Clean Voice
# (docs/evidence/Z2-vbcable-start-gap.md). Labelled, as information, only when all of these hold (orchestrator
# decision): within VBCABLE_GAP_S of the cable stream's start or reopen; one period long; the hand-over count
# constant (nothing of ours was late); the lined-up audio after it later by the run's length, where alignment
# exists. At most one per stream start; any other zero run fails.
VBCABLE_GAP_S = 5.0
VBCABLE_PERIOD_MS = (9.0, 13.0)
VBCABLE_COUNT_MS = 3.0
VBCABLE_LAG_SAMPLES = 2
ALIGN_MIN_R = 0.5  # the lined-up cable and backup correlate at least this well on both sides: alignment exists
STREAM_GAP_S = 0.5  # a cable callback this long after the one before is a new stream (a reopen)

F64 = npt.NDArray[np.float64]

# Trace columns.
MIC_COLS = ("perf", "adc", "now", "frames", "status")
CABLE_COLS = ("perf", "dac", "now", "frames", "fill", "underruns", "target")
WORK_COLS = ("start", "end", "blocks", "correction", "target", "fill")


class Trace:
    """Preallocated arrays the callbacks and the worker write one row into each time they run."""

    def __init__(self, seconds: float) -> None:
        callbacks = int(seconds * 500) + 5_000  # callbacks as short as 2 ms
        self.mic = np.zeros((callbacks, len(MIC_COLS)))
        self.cable = np.zeros((callbacks, len(CABLE_COLS)))
        self.work = np.zeros((int(seconds * 2_000) + 20_000, len(WORK_COLS)))
        self.chain = np.zeros((int(seconds * 120) + 2_000, 2))  # start, seconds
        self.aec = np.zeros((int(seconds * 120) + 2_000, 2))  # start, seconds of the echo stage (delay + AEC3)
        self.loop = np.zeros((callbacks, 2))  # perf at entry, frames (the speakers' loopback)
        self.monitor = np.zeros((callbacks, 3))  # perf at entry, frames, fill before the read
        self.n_mic = self.n_cable = self.n_work = self.n_chain = 0
        self.n_aec = self.n_loop = self.n_monitor = 0
        self.full = 0  # rows that did not fit
        self.paths: list[AudioPath] = []
        self.worker_priority: int | None = None


TRACE = Trace(1.0)


class _TimedChain:
    def __init__(self, chain: Processor) -> None:
        self._chain = chain

    def process(
        self, x: npt.NDArray[np.float32], raw: npt.NDArray[np.float32] | None = None
    ) -> npt.NDArray[np.float32]:
        t0 = time.perf_counter()
        y = self._chain.process(x, raw)
        tr = TRACE
        i = tr.n_chain
        if i < len(tr.chain):
            tr.chain[i, 0] = t0
            tr.chain[i, 1] = time.perf_counter() - t0
            tr.n_chain = i + 1
        return y


def _thread_priority() -> int | None:
    if sys.platform != "win32":
        return None
    import ctypes

    k32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    k32.GetCurrentThread.restype = ctypes.c_void_p
    k32.GetThreadPriority.argtypes = [ctypes.c_void_p]
    return int(k32.GetThreadPriority(k32.GetCurrentThread()))


class StampedPath(AudioPath):
    """The real path, plus one trace row per callback, worker step and chain block."""

    def __init__(self, in_rate: int, in_channels: int, out_rate: int, out_channels: int, chain: Processor) -> None:
        super().__init__(in_rate, in_channels, out_rate, out_channels, _TimedChain(chain))
        TRACE.paths.append(self)

    def input_callback(self, indata: Any, frames: int, time: Any, status: Any) -> None:
        t = _now()
        tr = TRACE
        i = tr.n_mic
        if i < len(tr.mic):
            row = tr.mic[i]
            row[0] = t
            if time is not None:
                row[1] = time.inputBufferAdcTime
                row[2] = time.currentTime
            row[3] = frames
            row[4] = 1.0 if status else 0.0
            tr.n_mic = i + 1
        else:
            tr.full += 1
        super().input_callback(indata, frames, time, status)

    def output_callback(self, outdata: Any, frames: int, time: Any, status: Any) -> None:
        t = _now()
        fill = self.out_ring.available()
        super().output_callback(outdata, frames, time, status)
        tr = TRACE
        i = tr.n_cable
        if i < len(tr.cable):
            row = tr.cable[i]
            row[0] = t
            if time is not None:
                row[1] = time.outputBufferDacTime
                row[2] = time.currentTime
            row[3] = frames
            row[4] = fill
            row[5] = self.counters.underruns
            row[6] = self.controller.target
            tr.n_cable = i + 1
        else:
            tr.full += 1

    def monitor_callback(self, outdata: Any, frames: int, time: Any, status: Any) -> None:
        t = _now()
        sink = self.monitor
        fill = sink.ring.available() if sink is not None else 0
        super().monitor_callback(outdata, frames, time, status)
        # The real callback ran (real stream timing, real counters); the speakers get silence, so the soak's mic can
        # never hear itself.
        outdata.fill(0.0)
        tr = TRACE
        i = tr.n_monitor
        if i < len(tr.monitor):
            tr.monitor[i] = (t, frames, fill)
            tr.n_monitor = i + 1

    def loopback_callback(self, indata: Any, frames: int, time: Any, status: Any) -> None:
        t = _now()
        super().loopback_callback(indata, frames, time, status)
        tr = TRACE
        i = tr.n_loop
        if i < len(tr.loop):
            tr.loop[i] = (t, frames)
            tr.n_loop = i + 1

    def enable_echo(self, ref_rate: int, ref_channels: int) -> None:
        super().enable_echo(ref_rate, ref_channels)
        stage = self.echo
        if stage is not None and not getattr(stage, "_timed", False):
            real = stage.cancel

            def cancel(near: Any) -> Any:
                t0 = time.perf_counter()
                y = real(near)
                tr = TRACE
                i = tr.n_aec
                if i < len(tr.aec):
                    tr.aec[i] = (t0, time.perf_counter() - t0)
                    tr.n_aec = i + 1
                return y

            stage.cancel = cancel  # type: ignore[method-assign]
            stage._timed = True  # type: ignore[attr-defined]

    def process(self) -> int:
        tr = TRACE
        if tr.worker_priority is None:
            tr.worker_priority = _thread_priority()
        t0 = time.perf_counter()
        ran = super().process()
        t1 = time.perf_counter()
        i = tr.n_work
        if i < len(tr.work):
            row = tr.work[i]
            row[0] = t0
            row[1] = t1
            row[2] = ran
            row[3] = self.controller.correction
            row[4] = self.controller.target
            row[5] = self.out_ring.available()
            tr.n_work = i + 1
        else:
            tr.full += 1
        return ran


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(f"{record.created:.3f} {record.levelname} {record.name}: {record.getMessage()}")


class CableListener:
    """Records CABLE Output (what Meet hears) in a child process, so its PortAudio and its callbacks stay out of the
    engine's process (PortAudio there belongs to the engine's thread). ``perf_counter`` is system-wide on Windows
    (QueryPerformanceCounter), so the child's stamps line up with the trace."""

    def __init__(self, seconds: float, out: Path) -> None:
        import subprocess

        self.out = out
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--listen-child", str(seconds + 30), str(out)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline().strip()
        if line != "listening":
            raise SystemExit(f"the CABLE Output listener did not start: {line!r}")

    def close(self) -> tuple[npt.NDArray[np.float32], F64]:
        assert self.proc.stdin is not None
        self.proc.stdin.write("stop\n")
        self.proc.stdin.flush()
        self.proc.wait(timeout=60)
        d = np.load(self.out)
        audio, stamps = d["audio"], d["stamps"]
        d.close()
        self.out.unlink()
        return audio, stamps


class _ListenChild:
    """The child side: a preallocated buffer that the callback only copies into."""

    def __init__(self, seconds: float) -> None:
        import sounddevice as sd

        apis = sd.query_hostapis()
        wanted = "Windows WASAPI" if sys.platform == "win32" else "Core Audio"
        api = next(i for i, a in enumerate(apis) if a["name"] == wanted)
        device = next(
            i
            for i, d in enumerate(sd.query_devices())
            if d["hostapi"] == api and d["max_input_channels"] > 0 and d["name"].lower().startswith("cable output")
        )
        info = sd.query_devices(device)
        self.rate = 48_000
        self.audio = np.zeros(int((seconds + 30) * self.rate), np.float32)
        self.stamps = np.zeros((int((seconds + 30) * 500), 2))  # perf at entry, first sample index
        self.pos = 0
        self.n = 0
        extra = sd.WasapiSettings(auto_convert=True) if sys.platform == "win32" else None
        self.stream = sd.InputStream(
            device=device,
            channels=min(2, int(info["max_input_channels"])),
            samplerate=self.rate,
            dtype="float32",
            callback=self._callback,
            extra_settings=extra,
        )
        self.stream.start()

    def _callback(self, indata: Any, frames: int, time_: Any, status: Any) -> None:
        t = time.perf_counter()
        n = min(frames, len(self.audio) - self.pos)
        self.audio[self.pos : self.pos + n] = indata[:n, 0]
        if self.n < len(self.stamps):
            self.stamps[self.n, 0] = t
            self.stamps[self.n, 1] = self.pos
            self.n += 1
        self.pos += n


def _listen_child(seconds: float, out: Path) -> int:
    child = _ListenChild(seconds)
    print("listening", flush=True)
    sys.stdin.readline()  # the parent writes "stop" (or exits)
    child.stream.abort()
    child.stream.close()
    np.savez(out, audio=child.audio[: child.pos], stamps=child.stamps[: child.n])
    return 0


# ---------------------------------------------------------------------- the run


def run(args: argparse.Namespace) -> Path:
    global TRACE
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    TRACE = Trace(args.seconds + 30)
    records = _Records()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger().addHandler(records)
    engine_mod.AudioPath = StampedPath  # type: ignore[misc]

    result: dict[str, Any] = {}
    if args.listen_cable:
        result["listener"] = CableListener(args.seconds, LOG_DIR / f"{stamp} listen.npz")
    engine = Engine(LOG_DIR / f"{stamp} rec")
    if args.settings:
        from harness import parse_settings

        engine.set_settings(parse_settings(args.settings))
    if args.with_ui:
        _run_with_ui(engine, args, result)
    else:
        _run_plain(engine, args, result)

    trace = TRACE
    path = trace.paths[-1] if trace.paths else None
    status = result["status"]
    print(f"mic: {status.mic!a}  cable: {status.cable!a}", flush=True)
    if path is not None:
        print(f"in {path.in_rate} Hz x{path.in_channels}, out {path.out_rate} Hz x{path.out_channels}")
    out = LOG_DIR / f"{stamp}.npz"
    meta = {
        "t_on": result["t_on"],
        "t_end": result["t_end"],
        "in_rate": path.in_rate if path else 48_000,
        "out_rate": path.out_rate if path else 48_000,
        "fade": max(1, int(FADE_S * (path.out_rate if path else 48_000))),
        "underruns": status.counters.underruns,
        "input_overflows": status.counters.input_overflows,
        "output_overflows": status.counters.output_overflows,
        "input_device_errors": status.counters.input_device_errors,
        "output_device_errors": status.counters.output_device_errors,
        "buffering_changes": path.cable.buffering_changes if path else 0,
        "final_target": path.controller.target if path else 0.0,
        "paths": len(trace.paths),
        "rows_dropped": trace.full,
        "worker_priority": -999 if trace.worker_priority is None else trace.worker_priority,
        "with_ui": int(args.with_ui),
        "toggles": result.get("toggles", 0),
        # What the engine's own clock estimators concluded (J05 Step 2c; absent on older code).
        "est_mic_rate": float(getattr(getattr(path, "mic_clock", None), "rate", 0.0)),
        "est_cable_rate": float(getattr(getattr(getattr(path, "cable", None), "clock", None), "rate", 0.0)),
        "est_resets": int(getattr(getattr(path, "mic_clock", None), "resets", 0))
        + int(getattr(getattr(getattr(path, "cable", None), "clock", None), "resets", 0)),
        "est_outliers": int(getattr(getattr(path, "mic_clock", None), "outliers", 0))
        + int(getattr(getattr(getattr(path, "cable", None), "clock", None), "outliers", 0)),
        # J06: Hear myself and Remove speaker echo.
        "want_echo": int(bool(args.echo)),
        "want_monitor": int(bool(args.monitor)),
        "monitor_underruns": status.counters.monitor_underruns,
        "monitor_overflows": status.counters.monitor_overflows,
        "monitor_on_at_end": int(status.monitor),
        "monitor_device": str(status.monitor_device or ""),
        "reference_gaps": status.counters.reference_gaps,
        "reference_overflows": status.counters.reference_overflows,
        "echo_state": status.echo.value,
        "echo_reason": str(status.echo_reason or ""),
        "echo_delay_ms": float(status.echo_delay_ms),
        "echo_age_ms": float(result.get("echo_age_ms", float("nan"))),
        "echo_device": str(result.get("echo_device", "")),
        "settings": " ".join(args.settings or []),
    }
    arrays: dict[str, Any] = {
        "mic": trace.mic[: trace.n_mic],
        "cable": trace.cable[: trace.n_cable],
        "work": trace.work[: trace.n_work],
        "chain": trace.chain[: trace.n_chain],
        "aec": trace.aec[: trace.n_aec],
        "loop": trace.loop[: trace.n_loop],
        "monitor": trace.monitor[: trace.n_monitor],
        "log": np.array(records.lines if records.lines else [""]),
    }
    arrays.update({f"meta_{k}": np.array(v) for k, v in meta.items()})
    if "listen" in result:
        arrays["listen_audio"], arrays["listen_stamps"] = result["listen"]
    if status.recording_files and not result.get("toggles"):
        arrays["clean_wav"] = np.array(str(status.recording_files[0]))
    np.savez_compressed(out, **arrays)
    print(f"trace: {out}")
    for f in status.recording_files:
        print(f"recording: {f}")
    return out


def _wait_on(engine: Engine, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while engine.status().state is not State.ON:
        if time.monotonic() > deadline:
            raise SystemExit(f"the engine did not turn on: {engine.status()}")
        time.sleep(0.05)


def _check_mic(engine: Engine, args: argparse.Namespace) -> None:
    """--mic must be the mic that runs: the engine falls back to the default for a name it does not list."""
    mic = engine.status().mic
    if args.mic is not None and mic != args.mic:
        engine.close()
        raise SystemExit(f"--mic {args.mic!a} is not listed; the engine opened {mic!a}. Listed: {engine.mics()!a}")


def _toggle_times(seconds: float) -> list[tuple[float, bool]]:
    return [(seconds * k / 5, k % 2 == 0) for k in range(1, 5)]  # off, on, off, on


def _set_recording(engine: Engine, on: bool) -> None:
    setter = getattr(engine, "set_recording", None)
    if setter is None:
        raise SystemExit("--toggle-recording needs Engine.set_recording")
    setter(on)


def _side_streams(engine: Engine, args: argparse.Namespace) -> None:
    """Switch on what the run asks for once the engine is on, and wait until it took."""
    if args.echo:
        engine.set_echo(True)
    if args.monitor:
        engine.set_monitor(True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        st = engine.status()
        settled = st.echo not in (EchoState.OFF, EchoState.STARTING)
        if (not args.echo or settled) and (not args.monitor or st.monitor or st.monitor_error):
            break
        time.sleep(0.05)
    st = engine.status()
    if args.echo:
        print(f"echo removal: {st.echo.value} {st.echo_reason or ''}", flush=True)
    if args.monitor:
        print(
            f"Hear myself: {'on' if st.monitor else 'off'} on {st.monitor_device!a} {st.monitor_error or ''}",
            flush=True,
        )


def _echo_facts(result: dict[str, Any]) -> None:
    path = TRACE.paths[-1] if TRACE.paths else None
    stage = getattr(path, "echo", None)
    meter = getattr(stage, "meter", None)
    age = getattr(meter, "age_s", None)
    result["echo_age_ms"] = age * 1000 if age is not None else float("nan")


def _run_plain(engine: Engine, args: argparse.Namespace, result: dict[str, Any]) -> None:
    time.sleep(0.5)
    engine.start(args.mic)
    _wait_on(engine)
    _check_mic(engine, args)
    _side_streams(engine, args)
    result["t_on"] = t_on = time.perf_counter()
    toggles = _toggle_times(args.seconds) if args.toggle_recording else []
    result["toggles"] = 0
    while (elapsed := time.perf_counter() - t_on) < args.seconds:
        if toggles and elapsed >= toggles[0][0]:
            _set_recording(engine, toggles.pop(0)[1])
            result["toggles"] += 1
        time.sleep(0.25)
        engine.status()  # what the window's timer does, at a lower rate
    _echo_facts(result)
    _finish(engine, result)


def _run_with_ui(engine: Engine, args: argparse.Namespace, result: dict[str, Any]) -> None:
    import tempfile

    from PySide6.QtWidgets import QApplication

    from clean_voice import prefs
    from clean_voice.app import MainWindow

    app = QApplication.instance() or QApplication([sys.argv[0]])
    with tempfile.TemporaryDirectory() as tmp:
        # The run's own saved choices (its --settings, or the defaults), never the guest's settings file.
        prefs_path = Path(tmp) / prefs.FILE_NAME
        # The backup on (it is off by default): the zero-run check needs the clean voice to tell quiet from fault.
        prefs.save(prefs_path, prefs.Prefs(settings=engine.settings(), mic=args.mic, recording=True))
        window = MainWindow(engine, prefs_path)  # starts the engine itself, as at launch
        window.show()
        if getattr(args, "fine_tune", False):
            _fine_tune_chain(window, result)
        _soak_window(app, window, engine, args, result)


FINE_TUNE_CHAIN = ("noise_gate_filter", "basic_eq_filter", "compressor_filter", "gain_filter")  # J10's done-when


def _fine_tune_chain(window: Any, result: dict[str, Any]) -> None:
    """Fine-tune on with the done-when's four filters at OBS's defaults, switched on in Step 2 as a guest would, and
    the filter window opened over them (Show plugin), so the soak runs with it on screen."""
    from clean_voice import fine_tune
    from clean_voice.settings import replace

    filters: list[fine_tune.FilterSpec] = []
    for kind in FINE_TUNE_CHAIN:
        filters.append(fine_tune.new_filter(kind, filters))
    window._send(replace(window.prefs.settings, filters=tuple(filters)))
    window.fine_tune_toggle.click()  # Step 2's Fine-tune switch: on
    window.show_plugin_button.click()
    sent = window.prefs.settings
    result["fine_tune_filters"] = len(sent.filters) if sent.fine_tune else 0
    print(f"Fine-tune on with {', '.join(f.name for f in sent.filters)}", flush=True)


def _soak_window(app: Any, window: Any, engine: Engine, args: argparse.Namespace, result: dict[str, Any]) -> None:
    from PySide6.QtCore import QTimer

    toggles = _toggle_times(args.seconds) if args.toggle_recording else []
    result["toggles"] = 0

    def started() -> None:
        if engine.status().state is not State.ON:
            QTimer.singleShot(50, started)
            return
        _check_mic(engine, args)
        _side_streams(engine, args)
        result["t_on"] = time.perf_counter()
        QTimer.singleShot(int(args.seconds * 1000), app.quit)
        for at, on in toggles:
            QTimer.singleShot(int(at * 1000), lambda on=on: _toggle(on))

    def _toggle(on: bool) -> None:
        _set_recording(engine, on)
        result["toggles"] += 1

    QTimer.singleShot(50, started)  # wait for the engine the window started
    app.exec()
    _echo_facts(result)
    _finish(engine, result)
    window.close()


def _finish(engine: Engine, result: dict[str, Any]) -> None:
    result["t_end"] = time.perf_counter()
    result["status"] = engine.status()
    listener: CableListener | None = result.get("listener")
    if listener is not None:
        result["listen"] = listener.close()
    engine.stop()
    deadline = time.monotonic() + 5
    while engine.status().state is not State.OFF and time.monotonic() < deadline:
        time.sleep(0.05)
    engine.close()


# ---------------------------------------------------------------------- the summary


def _stats(x: F64, scale: float = 1000.0) -> str:
    if len(x) == 0:
        return "n/a"
    return (
        f"median {np.median(x) * scale:.2f}, p99 {np.percentile(x, 99) * scale:.2f}, max {np.max(x) * scale:.2f} "
        f"(min {np.min(x) * scale:.2f})"
    )


def _rate(t: F64, frames: F64) -> float:
    """Frames per second from a least-squares line through (time, frames so far)."""
    if len(t) < 3:
        return float("nan")
    cum = np.cumsum(frames)
    tc = t - t.mean()
    return float(np.dot(tc, cum - cum.mean()) / np.dot(tc, tc))


def _sizes(frames: F64) -> str:
    values, counts = np.unique(frames.astype(int), return_counts=True)
    order = np.argsort(-counts)[:6]
    return ", ".join(f"{values[i]} x{counts[i]}" for i in order)


def analyze(trace_file: Path) -> dict[str, float]:
    """Print the summary of a saved trace; return the numbers the Step 2d criteria use."""
    d = np.load(trace_file)
    meta = {k[5:]: d[k].item() for k in d.files if k.startswith("meta_")}
    mic, cable, work, chain = d["mic"], d["cable"], d["work"], d["chain"]
    t_on, t_end = float(meta["t_on"]), float(meta["t_end"])
    in_rate, out_rate, fade = int(meta["in_rate"]), int(meta["out_rate"]), int(meta["fade"])
    to_ms = 1000.0 / out_rate
    out: dict[str, float] = {}

    def after(a: F64, s: float, col: int = 0) -> F64:
        return a[a[:, col] >= t_on + s]

    print(f"\n=== {trace_file.name}: {t_end - t_on:.0f} s{' with the window open' if meta['with_ui'] else ''} ===")
    print(
        f"counters: underruns={meta['underruns']} input_overflows={meta['input_overflows']} "
        f"output_overflows={meta['output_overflows']} input_device_errors={meta['input_device_errors']} "
        f"output_device_errors={meta['output_device_errors']}  (reconnects: {meta['paths'] - 1}, "
        f"trace rows dropped: {meta['rows_dropped']}, recording toggles: {meta['toggles']})"
    )
    out["underruns"] = float(meta["underruns"])
    out["input_overflows"] = float(meta["input_overflows"])
    out["output_overflows"] = float(meta["output_overflows"])
    und = np.diff(np.concatenate(([0.0], cable[:, 5])))
    und_times = cable[und > 0, 0]
    for s in (SETTLE_S, SWING_FROM_S):
        print(f"underruns after the first {s:.0f} s: {int(np.count_nonzero(und_times >= t_on + s))}")

    c10 = after(cable, SETTLE_S)
    if len(c10):
        fill = c10[:, 4]
        spare = fill - c10[:, 3] - fade
        out["min_spare_ms"] = float(spare.min() * to_ms)
        print(
            f"fill before each cable read, after {SETTLE_S:.0f} s: "
            f"{fill.min() * to_ms:.1f}..{fill.max() * to_ms:.1f} ms "
            f"(median {np.median(fill) * to_ms:.1f}); spare beyond the callback and fade: "
            f"{spare.min() * to_ms:.1f}..{spare.max() * to_ms:.1f} ms"
        )
    first = work[(work[:, 0] < t_on + SWING_FROM_S) & (work[:, 2] > 0)]
    if len(first):
        print(
            f"correction in the first {SWING_FROM_S:.0f} s (start-up): {first[:, 3].min() * 1e6:+.0f}.."
            f"{first[:, 3].max() * 1e6:+.0f} ppm"
        )
    for s in (SETTLE_S, SWING_FROM_S):
        w = after(work, s)
        w = w[w[:, 2] > 0]
        if len(w):
            corr = w[:, 3] * 1e6
            print(
                f"correction after {s:.0f} s: {corr.min():+.0f}..{corr.max():+.0f} ppm "
                f"(mean {corr.mean():+.0f}, spread {corr.max() - corr.min():.0f})"
            )
            if s == JUDGE_FROM_S:
                out["corr_spread_ppm"] = float(corr.max() - corr.min())
                out["corr_mean_ppm"] = float(corr.mean())
    if len(work):
        targets = work[:, 4] * to_ms
        print(
            f"target cushion: start {targets[0]:.1f} ms, end {float(meta['final_target']) * to_ms:.1f} ms, "
            f"buffering changes {meta['buffering_changes']}"
        )
    for line in d["log"]:
        if "buffering" in str(line) or "MMCSS" in str(line) or "WARNING" in str(line):
            print(f"  log: {line}")

    print(f"mic callback frames: {_sizes(mic[:, 3])}")
    print(f"cable callback frames: {_sizes(cable[:, 3])}")
    m10, k10 = after(mic, SETTLE_S), after(cable, SETTLE_S)
    print(f"mic callback interval ms: {_stats(np.diff(m10[:, 0]))}")
    print(f"cable callback interval ms: {_stats(np.diff(k10[:, 0]))}")

    r_mic, r_cable = _rate(m10[:, 0], m10[:, 3]), _rate(k10[:, 0], k10[:, 3])
    gap = ((r_cable / out_rate) / (r_mic / in_rate) - 1.0) * 1e6
    out["gap_ppm"] = gap
    print(
        f"rates (least squares on perf_counter at entry): mic {r_mic:.2f} Hz ({(r_mic / in_rate - 1) * 1e6:+.0f} ppm), "
        f"cable {r_cable:.2f} Hz ({(r_cable / out_rate - 1) * 1e6:+.0f} ppm); gap cable/mic {gap:+.0f} ppm"
    )
    if float(meta.get("est_mic_rate", 0.0)) > 0:
        em, ec = float(meta["est_mic_rate"]), float(meta["est_cable_rate"])
        out["mic_rate_off_ppm"] = (em / in_rate - 1.0) * 1e6
        # The estimator keeps the nominal rate exactly until its fit is sure (audio_path.ClockEstimator).
        out["mic_clock_trusted"] = 1.0 if em != in_rate else 0.0
        print(
            f"engine's clock estimators at the end: mic {em:.2f} Hz, cable {ec:.2f} Hz; feed-forward "
            f"{((ec / out_rate) / (em / in_rate) - 1) * 1e6:+.0f} ppm; resets {meta['est_resets']}, "
            f"outliers left out {meta['est_outliers']}"
        )
    if np.any(m10[:, 1]) and np.any(k10[:, 1]):
        ra, rd = _rate(m10[:, 1], m10[:, 3]), _rate(k10[:, 1], k10[:, 3])
        gap2 = ((rd / out_rate) / (ra / in_rate) - 1.0) * 1e6
        print(f"rates (on PortAudio ADC/DAC times): mic {ra:.2f} Hz, cable {rd:.2f} Hz; gap {gap2:+.0f} ppm")

    for name, a in (("mic", m10), ("cable", k10)):
        if not np.any(a[:, 2]):
            continue
        late = a[:, 0] - a[:, 2]
        tt = a[:, 0] - a[0, 0]
        slope = float(np.polyfit(tt, late, 1)[0]) * 1e6 if len(a) > 2 else 0.0
        print(
            f"{name} callback lateness (entry - time.currentTime) ms: {_stats(late)}; offset drift {slope:+.2f} ppm "
            f"=> currentTime {'follows' if abs(slope) < 5 and abs(np.median(late)) < 0.005 else 'does NOT follow'} "
            "perf_counter"
        )
        out[f"{name}_late_max_ms"] = float(late.max() * 1000)
    if len(m10) and np.any(m10[:, 1]):
        print(f"mic ADC time before entry ms: {_stats(m10[:, 0] - m10[:, 1])}")

    w10 = after(work, SETTLE_S)
    if len(w10) > 1:
        steps = w10[:, 1] - w10[:, 0]
        gaps = np.diff(w10[:, 0])
        print(f"worker step gaps (start to start) ms: {_stats(gaps)}")
        print(f"worker step length ms: {_stats(steps)}")
        out["worker_step_max_ms"] = float(steps.max() * 1000)
        out["worker_gap_max_ms"] = float(gaps.max() * 1000)
    ch = chain[chain[:, 0] >= t_on + SETTLE_S]
    if len(ch):
        print(f"chain time per 10 ms block ms: {_stats(ch[:, 1])}")
        out["chain_p99_ms"] = float(np.percentile(ch[:, 1], 99) * 1000)
    _side_summary(d, meta, out, t_on)
    prio = int(meta["worker_priority"])
    print(f"worker thread priority (GetThreadPriority): {prio if prio != -999 else 'n/a'}")

    if len(und_times):
        print("underruns (time since on; last mic callback; last worker step that ran blocks):")
        busy = work[work[:, 2] > 0]
        for t in und_times[:40]:
            prev_mic = mic[mic[:, 0] <= t]
            prev_work = busy[busy[:, 0] <= t]
            last_mic = 1000 * (t - prev_mic[-1, 0]) if len(prev_mic) else float("nan")
            last_work = 1000 * (t - prev_work[-1, 1]) if len(prev_work) else float("nan")
            print(f"  t={t - t_on:8.3f} s  mic {last_mic:6.1f} ms ago  worker block {last_work:6.1f} ms ago")

    if "listen_audio" in d.files:
        clean = None
        wavs = [Path(str(d["clean_wav"]))] if "clean_wav" in d.files else []
        if not wavs and not meta["toggles"]:  # traces saved before the key: the one backup next to the trace
            wavs = sorted(trace_file.parent.glob(f"{trace_file.stem} rec/*clean voice.wav"))
        if len(wavs) == 1 and wavs[0].exists():
            first_block = float(work[work[:, 2] > 0][0, 0])  # the backup holds every block from the first one
            clean = (_read_wav24(wavs[0]), first_block)
        # A buffering raise adds its silence to the cable at once (faded; engine-obs r1, B1): the target steps up.
        raises = cable[1:, 0][np.diff(cable[:, 6]) > 0] if len(cable) > 1 else np.zeros(0)
        info: dict[str, float] = {}
        out["zero_runs_unmatched"] = float(
            _zero_runs(d["listen_audio"], d["listen_stamps"], und_times, t_on, clean, raises, cable, info)
        )
        out["vbcable_gaps"] = info.get("vbcable_gaps", 0.0)
    _verdict(out)
    return out


def _side_summary(d: Any, meta: dict[str, Any], out: dict[str, float], t_on: float) -> None:
    """J06: the speaker sink (Hear myself) and the echo reference."""
    if meta.get("settings"):
        print(f"settings: {meta['settings']}")
    if int(meta.get("want_monitor", 0)):
        mon = d["monitor"] if "monitor" in d.files else np.zeros((0, 3))
        m10 = mon[mon[:, 0] >= t_on + SETTLE_S]
        out["monitor_underruns"] = float(meta["monitor_underruns"])
        out["monitor_overflows"] = float(meta["monitor_overflows"])
        out["monitor_callbacks"] = float(len(mon))
        print(
            f"Hear myself (muted) on {meta.get('monitor_device', '')!a}: underruns={meta['monitor_underruns']} "
            f"overflows={meta['monitor_overflows']}, {len(mon)} callbacks, on at the end: {bool(meta['monitor_on_at_end'])}"
        )
        if len(m10):
            fill_ms = m10[:, 2] / 48.0
            print(
                f"monitor fill before each read, after {SETTLE_S:.0f} s: {fill_ms.min():.1f}..{fill_ms.max():.1f} ms "
                f"(median {np.median(fill_ms):.1f}); callback interval ms: {_stats(np.diff(m10[:, 0]))}"
            )
    if int(meta.get("want_echo", 0)):
        loop = d["loop"] if "loop" in d.files else np.zeros((0, 2))
        aec = d["aec"] if "aec" in d.files else np.zeros((0, 2))
        l10 = loop[loop[:, 0] >= t_on + SETTLE_S]
        out["reference_gaps"] = float(meta["reference_gaps"])
        out["reference_overflows"] = float(meta["reference_overflows"])
        out["loop_callbacks"] = float(len(loop))
        print(
            f"echo removal: {meta['echo_state']} {meta.get('echo_reason', '')}; reference gaps={meta['reference_gaps']} "
            f"overflows={meta['reference_overflows']}; reference age {float(meta['echo_age_ms']):.1f} ms, "
            f"mic delay {float(meta['echo_delay_ms']):.1f} ms"
        )
        if len(l10) > 1:
            iv = np.diff(l10[:, 0])
            out["loop_interval_max_ms"] = float(iv.max() * 1000)
            print(f"loopback callback interval ms (after {SETTLE_S:.0f} s): {_stats(iv)}; frames {_sizes(l10[:, 1])}")
        a10 = aec[aec[:, 0] >= t_on + SETTLE_S]
        if len(a10):
            print(f"echo stage (delay + AEC3) time per 10 ms block ms: {_stats(a10[:, 1])}")


def _verdict(out: dict[str, float]) -> list[str]:
    """The J05 Step 2d acceptance, judged from 10 s so the start-up is covered (J05 review r1, F4). Returns the failed
    checks."""
    spread_name = f"correction spread from {JUDGE_FROM_S:.0f} s under {MAX_SPREAD_PPM:.0f} ppm"
    spread_ok = out.get("corr_spread_ppm", 1e9) < MAX_SPREAD_PPM
    off = out.get("mic_rate_off_ppm", 0.0)
    if not spread_ok and abs(off) > MIC_LOSS_PPM:
        print(
            f"correction spread {out.get('corr_spread_ppm', 0.0):.0f} ppm: information only, not a failure: the mic was "
            f"measured losing audio (its estimated rate {off:+.0f} ppm off nominal, over {MIC_LOSS_PPM:.0f}), and the "
            "correction followed it"
        )
        spread_ok = True
    mean_ok = abs(out.get("corr_mean_ppm", 1e9) - out.get("gap_ppm", 0.0)) <= MAX_MEAN_OFF_PPM
    if not mean_ok and out.get("mic_clock_trusted", 1.0) == 0.0:
        # Orchestrator decision (fine-tune-updates r1): too short for the mic clock estimator, so the correction's
        # mean cannot follow the rate gap yet. Hard again whenever the estimate was trusted (every 300 s soak).
        print(
            f"correction mean {out.get('corr_mean_ppm', 0.0):+.0f} ppm against a rate gap of {out.get('gap_ppm', 0.0):+.0f} "
            "ppm: information only, not a failure: the mic clock estimate was never trusted in this run (too short for "
            "the estimator)"
        )
        mean_ok = True
    checks = [
        ("underruns 0", out.get("underruns", 1) == 0),
        ("input overflows 0", out.get("input_overflows", 1) == 0),
        ("output overflows 0", out.get("output_overflows", 1) == 0),
        (spread_name, spread_ok),
        (f"correction mean from {JUDGE_FROM_S:.0f} s within {MAX_MEAN_OFF_PPM:.0f} ppm of the rate gap", mean_ok),
        ("spare above 0 ms", out.get("min_spare_ms", -1.0) > 0.0),
    ]
    if "zero_runs_unmatched" in out:
        checks.append(("every exact-zero run on CABLE Output explained", out["zero_runs_unmatched"] == 0))
    # J06 Step 10.
    if "chain_p99_ms" in out:
        checks.append((f"chain p99 under {MAX_CHAIN_P99_MS:.0f} ms", out["chain_p99_ms"] <= MAX_CHAIN_P99_MS))
    if "monitor_underruns" in out:
        checks.append(("monitor underruns 0", out["monitor_underruns"] == 0))
        checks.append(("monitor overflows 0", out["monitor_overflows"] == 0))
        checks.append(("the monitor ran", out["monitor_callbacks"] > 100))
    if "reference_gaps" in out:
        checks.append(("reference gaps 0", out["reference_gaps"] == 0))
        checks.append(("reference overflows 0", out["reference_overflows"] == 0))
        checks.append(("the loopback ran", out["loop_callbacks"] > 100))
        checks.append(
            (
                f"loopback interval max under {MAX_LOOP_GAP_MS:.0f} ms",
                out.get("loop_interval_max_ms", 1e9) < MAX_LOOP_GAP_MS,
            )
        )
    if out.get("vbcable_gaps"):
        print(
            f"VB-CABLE inserted gaps: {out['vbcable_gaps']:.0f} (information: VB-CABLE or the Windows engine, outside the "
            "app; docs/evidence/Z2-vbcable-start-gap.md)"
        )
    failed = [name for name, ok in checks if not ok]
    print(
        f"Step 2d verdict (judged from {JUDGE_FROM_S:.0f} s): {'PASS' if not failed else 'FAIL: ' + '; '.join(failed)}"
    )
    return failed


def _read_wav24(path: Path) -> npt.NDArray[np.float64]:
    import wave

    with wave.open(str(path), "rb") as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), np.uint8).reshape(-1, 3).astype(np.int32)
    x = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
    return np.where(x >= 1 << 23, x - (1 << 24), x) / float(1 << 23)


def _zero_runs(
    audio: npt.NDArray[np.float32],
    stamps: F64,
    events: F64,
    t_on: float,
    clean: tuple[npt.NDArray[np.float64], float] | None = None,
    raises: F64 | None = None,
    cable_rows: F64 | None = None,
    info: dict[str, float] | None = None,
) -> int:
    """Report runs of exact zeros on CABLE Output; return how many match neither a counted underrun, nor a
    buffering raise (the silence it adds), nor a stretch where the lined-up clean voice is too small for the cable,
    nor (at most once per cable stream start) VB-CABLE's start-up gap. ``info`` gets the VB-CABLE gap count."""
    nonzero = np.abs(audio[audio != 0])
    step = float(nonzero.min()) if len(nonzero) else 0.0  # the cable's smallest step (2^-18 on VB-CABLE, measured)
    if cable_rows is not None and len(cable_rows):
        # Only what CABLE Output carried from Clean Voice's cable stream on is judged: before that, a reader opening it
        # can get VB-CABLE's stale buffer from whatever played last (measured: 14 ms of an earlier test tone).
        k0 = int(np.searchsorted(stamps[:, 0], cable_rows[0, 0], side="right")) - 1
        before = int(stamps[max(0, k0), 1] + (cable_rows[0, 0] - stamps[max(0, k0), 0]) * 48_000) if k0 >= 0 else 0
        if before > 0 and np.any(audio[:before]):
            print(
                f"CABLE Output before the cable stream started: {np.count_nonzero(audio[:before])} non-zero samples "
                "(not Clean Voice's: VB-CABLE's stale buffer), not judged"
            )
            audio = audio.copy()
            audio[:before] = 0.0
    nz = np.flatnonzero(audio)
    if len(nz) == 0:
        print("CABLE Output: all zeros (nothing reached the cable)")
        return 1
    body = audio[nz[0] : nz[-1] + 1]
    zero = np.concatenate(([0], (body == 0).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(zero))
    starts, ends = edges[::2], edges[1::2]
    long = (ends - starts) >= int(ZERO_RUN_S * 48_000)
    starts, ends = starts[long] + nz[0], ends[long] + nz[0]
    unmatched = 0
    # The cable's own leading silence: the first long zero run, right after the resampler's ringing.
    lead = (
        0
        if len(starts) and starts[0] - nz[0] <= LEAD_RING_S * 48_000 and ends[0] - starts[0] >= LEAD_MIN_S * 48_000
        else -1
    )
    first_audio = int(ends[0]) if lead == 0 else int(nz[0])
    stream_starts = _stream_starts(cable_rows) if cable_rows is not None and len(cable_rows) else np.zeros(0)
    gap_used: set[int] = set()  # the stream starts that already had their one VB-CABLE gap
    print(
        f"CABLE Output: {len(audio) / 48_000:.1f} s recorded, exact-zero runs of {ZERO_RUN_S * 1000:.0f} ms or more: "
        f"{len(starts)}"
    )
    # Every run is judged (fine-tune-updates r2, S1: judging only the first 40 let a later hole pass); only the
    # printing is capped, with the faults always printed.
    printed = 0
    for n_run, (s, e) in enumerate(zip(starts, ends, strict=True)):
        # Map the sample index to the time of the listener callback that delivered it.
        k = max(0, int(np.searchsorted(stamps[:, 1], s, side="right")) - 1)
        t = stamps[k, 0] + (s - stamps[k, 1]) / 48_000
        near = events[np.abs(events - t) < MATCH_WINDOW_S] if len(events) else events
        if n_run == lead:
            # The cable's leading silence: the stream's start, the ring filling to its target, and the chain's own
            # warm-up (the clean voice WAV starts with ~120 ms of exact zeros); the resampler's ringing puts a few tiny
            # samples ahead of the first audio.
            verdict = "start-up (the cable's leading silence, before its first audio)"
        elif len(near):
            verdict = "matches an underrun"
        elif raises is not None and len(raises) and np.min(np.abs(raises - t)) < MATCH_WINDOW_S and (e - s) <= 48 * 160:
            verdict = "matches a buffering raise (the silence it adds, faded)"
        elif clean is not None and (quiet := _quiet(clean, audio, s, e, t, step, first_audio))[0]:
            verdict = (
                f"quiet: the clean voice, lined up (r={quiet[1]:.2f}), peaks at {QUIET_MAX_STEPS:.0f} cable steps or "
                f"less ({step:.1e} each)"
            )
        elif (
            cable_rows is not None
            and (gap := _vbcable_gap(audio, stamps, cable_rows, stream_starts, s, e, t, clean))[0]
        ):
            unmatched_before = gap_used.copy()
            if gap[1] in unmatched_before:
                verdict = "NO: a second VB-CABLE-like gap after the same stream start"
                unmatched += 1
            else:
                gap_used.add(gap[1])
                verdict = f"VB-CABLE inserted gap ({gap[2]})"
        else:
            verdict = "NO counted event nearby"
            unmatched += 1
        if printed < PRINT_RUNS or verdict.startswith("NO"):
            print(f"  t={t - t_on:8.3f} s  {(e - s) / 48:.1f} ms  {verdict}")
            printed += 1
    if info is not None:
        info["vbcable_gaps"] = float(len(gap_used))
    return unmatched


def _stream_starts(cable_rows: F64) -> F64:
    """When each cable stream started: its first callback, and any callback STREAM_GAP_S or more after the one before
    (a reopen)."""
    t = cable_rows[:, 0]
    return np.concatenate((t[:1], t[1:][np.diff(t) >= STREAM_GAP_S]))


def _handover_step_ms(cable_rows: F64, stamps: F64, t: float, start: float) -> float | None:
    """How much the hand-over count moved across time t, in ms: (when the listener got its samples less how many it
    got) minus (when the cable callbacks ran less how many frames they handed over), just before and just after. A
    callback that came late (render starvation) makes it step by the silence Windows played; VB-CABLE inserting a
    period downstream leaves it unchanged. None when there are too few callbacks on either side to tell."""
    handed = np.cumsum(cable_rows[:, 3]) - cable_rows[:, 3]
    oc = cable_rows[:, 0] - handed / 48_000
    ends = np.append(stamps[1:, 1], stamps[-1, 1] + 480)
    ol = stamps[:, 0] - ends / 48_000

    def offset(lo: float, hi: float) -> float | None:
        a = oc[(cable_rows[:, 0] >= lo) & (cable_rows[:, 0] < hi)]
        b = ol[(stamps[:, 0] >= lo) & (stamps[:, 0] < hi)]
        return float(b.min() - a.min()) if len(a) >= 3 and len(b) >= 3 else None

    before = offset(max(start + 0.05, t - 0.4), t - 0.03)
    after = offset(t + 0.03, t + 0.4)
    return None if before is None or after is None else (after - before) * 1000


def _lag(
    clean: tuple[npt.NDArray[np.float64], float], cable: npt.NDArray[np.float32], c0: int, c1: int, t0: float
) -> tuple[int, float]:
    """(cable index - backup index, r) for the cable stretch cable[c0:c1], heard at time t0."""
    from scipy import signal

    audio, first_block = clean
    ref = np.asarray(cable[c0:c1], np.float64)
    at = int((t0 - first_block) * 48_000)
    lo = max(0, at - int((QUIET_SPAN_S + 0.2) * 48_000))
    seg = audio[lo : max(lo, at + int(0.05 * 48_000) + len(ref))]
    if len(seg) < len(ref) or not np.any(ref):
        return 0, 0.0
    corr = signal.correlate(seg, ref, mode="valid", method="fft")
    k = int(np.argmax(corr))
    r = float(corr[k] / (np.linalg.norm(seg[k : k + len(ref)]) * np.linalg.norm(ref) + 1e-30))
    return c0 - (lo + k), r


def _vbcable_gap(
    cable: npt.NDArray[np.float32],
    stamps: F64,
    cable_rows: F64,
    stream_starts: F64,
    s: int,
    e: int,
    t: float,
    clean: tuple[npt.NDArray[np.float64], float] | None,
) -> tuple[bool, int, str]:
    """Whether the zero run cable[s:e] (heard at time t) is VB-CABLE's start-up gap; (yes, which stream start, why)."""
    since = t - stream_starts
    k = int(np.argmin(np.where(since >= 0, since, np.inf))) if len(stream_starts) and np.any(since >= 0) else -1
    if k < 0 or since[k] > VBCABLE_GAP_S:
        return False, -1, ""
    ms = (e - s) / 48
    if not VBCABLE_PERIOD_MS[0] <= ms <= VBCABLE_PERIOD_MS[1]:
        return False, -1, ""
    step = _handover_step_ms(cable_rows, stamps, t, float(stream_starts[k]))
    if step is None or abs(step) > VBCABLE_COUNT_MS:
        return False, -1, ""
    detail = f"{since[k]:.2f} s after the stream start, one period, hand-over count moved {step:+.1f} ms"
    if clean is None:
        return True, k, detail + ", unconfirmed alignment (no backup)"
    w = int(0.4 * 48_000)
    before = _lag(clean, cable, max(0, s - w), s - int(0.01 * 48_000), t - 0.4)
    after = _lag(clean, cable, e + int(0.01 * 48_000), e + w, t + (e - s) / 48_000 + 0.01)
    if min(before[1], after[1]) < ALIGN_MIN_R:
        return True, k, detail + f", unconfirmed alignment (r {before[1]:.2f} / {after[1]:.2f})"
    moved = after[0] - before[0]
    if abs(moved - (e - s)) > VBCABLE_LAG_SAMPLES:
        return False, -1, ""
    return True, k, detail + f", the audio after it {moved} samples later (the run is {e - s}; r {after[1]:.2f})"


def _quiet(
    clean: tuple[npt.NDArray[np.float64], float],
    cable: npt.NDArray[np.float32],
    s: int,
    e: int,
    t: float,
    step: float,
    first_audio: int = 0,
) -> tuple[bool, float]:
    """Whether the zero run cable[s:e] (heard at time t) is the clean voice itself too small for the cable, and how
    well the two lined up (r).

    VB-CABLE truncates toward zero (engine-obs r1: clean values of 1.3-2.0 steps came out as 1), so it shows exact
    zeros where the clean voice is under one step, and the drift resampler's interpolation can take a peak of a little
    more below it. The cable is lined up with the backup by cross-correlation on the ALIGN_S before the run, over every
    lag the cushion allows (QUIET_SPAN_S); the run is quiet only if the backup peaks at QUIET_MAX_STEPS or less over the
    whole lined-up run. Any other zero run is a fault. Right after the cable's first audio (``first_audio``) there is
    too little before the run, so it is lined up on the ALIGN_S after it instead."""
    audio, first_block = clean
    from scipy import signal

    n = e - s
    if s - max(first_audio, s - int(ALIGN_S * 48_000)) >= 4_800:
        ref = np.asarray(cable[max(first_audio, s - int(ALIGN_S * 48_000)) : s], np.float64)
        at = int((t - first_block) * 48_000)  # the backup's sample for the run's start if the cable had no delay
        lo = max(0, at - int(QUIET_SPAN_S * 48_000) - len(ref))
        seg = audio[lo : max(lo, at + int(0.05 * 48_000))]
        after = False
    else:
        ref = np.asarray(cable[e : e + int(ALIGN_S * 48_000)], np.float64)
        at = int((t - first_block) * 48_000) + n  # the backup's sample for the run's end if the cable had no delay
        lo = max(0, at - int(QUIET_SPAN_S * 48_000))
        seg = audio[lo : max(lo, at + int(0.05 * 48_000) + len(ref))]
        after = True
    if len(ref) < 4_800 or len(seg) < len(ref) or not np.any(ref):
        return False, 0.0
    corr = signal.correlate(seg, ref, mode="valid", method="fft")
    k = int(np.argmax(corr))
    window = seg[k : k + len(ref)]
    r = float(corr[k] / (np.linalg.norm(window) * np.linalg.norm(ref) + 1e-30))
    j = lo + k - n if after else lo + k + len(ref)  # the backup's sample the cable played at the run's start
    span = audio[max(0, j) : max(0, j) + n]
    return bool(len(span) == e - s and np.max(np.abs(span)) <= QUIET_MAX_STEPS * step), r


# ---------------------------------------------------------------------- the regression fixture


def export_fixture(trace_file: Path, dest: Path = FIXTURE) -> Path:
    """Write the callback pattern of the trace's worst 60 s as a test fixture: the most underruns, and between
    windows with as many, the least spare at a cable callback.

    Times and frame counts only: no device names, no audio.
    """
    d = np.load(trace_file)
    mic, cable, work, chain = d["mic"], d["cable"], d["work"], d["chain"]
    t_on = float(d["meta_t_on"])
    t_end = float(d["meta_t_end"])
    fade = int(d["meta_fade"])
    und = np.diff(np.concatenate(([0.0], cable[:, 5])))
    und_times = cable[und > 0, 0]
    playing = cable[cable[:, 0] >= t_on + 2.0]  # past the first prime
    spare = playing[:, 4] - playing[:, 3] - fade
    best, best_key = t_on, (-1, 0.0)
    start = t_on
    while start + FIXTURE_S <= t_end + 1e-9:
        count = int(np.count_nonzero((und_times >= start) & (und_times < start + FIXTURE_S)))
        inside = (playing[:, 0] >= start) & (playing[:, 0] < start + FIXTURE_S)
        key = (count, -float(spare[inside].min()) if np.any(inside) else 0.0)
        if key > best_key:
            best, best_key = start, key
        start += 1.0
    best_count = best_key[0]
    print(f"worst window: {best_count} underruns, least spare {-best_key[1] / 48:.1f} ms")

    def window(a: F64, col: int = 0) -> F64:
        return a[(a[:, col] >= best) & (a[:, col] < best + FIXTURE_S)]

    w = window(work)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        seconds=np.array(FIXTURE_S),
        in_rate=d["meta_in_rate"],
        out_rate=d["meta_out_rate"],
        mic_time=window(mic)[:, 0] - best,
        mic_frames=window(mic)[:, 3].astype(np.int32),
        cable_time=window(cable)[:, 0] - best,
        cable_frames=window(cable)[:, 3].astype(np.int32),
        worker_start=w[:, 0] - best,
        worker_end=w[:, 1] - best,
        chain_cost=window(chain)[:, 1],
        source_underruns=np.array(best_count),
    )
    print(
        f"fixture: {dest} ({dest.stat().st_size / 1024:.0f} KB), "
        f"window {best - t_on:.0f}..{best - t_on + FIXTURE_S:.0f} s "
        f"with {best_count} underruns"
    )
    return dest


def export_startup(trace_files: list[Path], dest: Path = STARTUP_FIXTURE, seconds: float = 150.0) -> Path:
    """Write the first ``seconds`` of each trace's callback stamps, from the stream start, as the clock estimator's
    start-up fixture: PortAudio's currentTime (what the engine stamps) and frame counts only."""
    arrays: dict[str, Any] = {"patterns": np.array(len(trace_files))}
    for i, trace_file in enumerate(trace_files):
        d = np.load(trace_file)
        for name in ("mic", "cable"):
            a = d[name]
            t = np.where(a[:, 2] > 0, a[:, 2], a[:, 0])
            keep = t - t[0] < seconds
            arrays[f"p{i}_{name}_time"] = t[keep] - t[0]
            arrays[f"p{i}_{name}_frames"] = a[keep, 3].astype(np.int32)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **arrays)
    print(f"start-up fixture: {dest} ({dest.stat().st_size / 1024:.0f} KB), {len(trace_files)} patterns")
    return dest


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(prog="audio_soak", description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--with-ui", action="store_true", help="also open the main window")
    parser.add_argument(
        "--fine-tune", action="store_true", help="with --with-ui: Fine-tune on with a gate, EQ, compressor, gain"
    )
    parser.add_argument("--mic", help="this microphone instead of the default (its exact name)")
    parser.add_argument("--listen-cable", action="store_true", help="record CABLE Output and count exact-zero runs")
    parser.add_argument("--toggle-recording", action="store_true", help="switch the backup off and on twice")
    parser.add_argument(
        "--settings", nargs="*", default=None, metavar="KEY=VALUE", help="sound settings (harness keys)"
    )
    parser.add_argument("--echo", action="store_true", help="switch Remove speaker echo on")
    parser.add_argument("--monitor", action="store_true", help="switch Hear myself on (its output muted)")
    parser.add_argument("--analyze", type=Path, help="summarize a saved trace")
    parser.add_argument("--export-fixture", type=Path, help="write tests/data/real_pattern.npz from a trace")
    parser.add_argument("--export-startup", type=Path, nargs="+", help="write tests/data/real_startup.npz from traces")
    parser.add_argument("--listen-child", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.export_startup:
        export_startup(args.export_startup)
        return 0
    if args.listen_child:
        return _listen_child(float(args.listen_child[0]), Path(args.listen_child[1]))
    if args.export_fixture:
        export_fixture(args.export_fixture)
        return 0
    if args.analyze:
        analyze(args.analyze)
        return 0
    trace_file = run(args)
    analyze(trace_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
