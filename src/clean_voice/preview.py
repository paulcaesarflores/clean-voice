"""A stand-in engine that holds one fixed state, for rendering and testing the screen without audio devices.

Previews never invent device names: every state is built from a ``DeviceNames`` value, which ``machine_devices()``
fills from this computer's real list (enumeration only, no stream) and tests fill with their own names.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, cast

from clean_voice import fine_tune, vst
from clean_voice.devices import is_cable
from clean_voice.engine import EchoState, MicTestState, State, Status
from clean_voice.prefs import Prefs
from clean_voice.settings import Settings, replace, style_bands
from clean_voice.vst import VstError, VstPlugin, VstRegistry

if TYPE_CHECKING:
    from clean_voice.filter_window import FilterModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceNames:
    mics: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    cable: str | None = None  # VB-CABLE's playback side, if listed
    default_mic: str | None = None
    speakers: str | None = None  # where Hear myself would play


def machine_devices() -> DeviceNames:
    """This computer's devices as the engine lists them (WASAPI or Core Audio), without opening any stream. No sound
    system (CI): empty lists, and the screens say so."""
    try:
        from clean_voice.devices import SoundDeviceBackend, find_cable, is_virtual, microphones

        backend = SoundDeviceBackend()
        mics = tuple(d.name for d in microphones(backend))
        outputs = tuple(d.name for d in backend.outputs())
        cable = find_cable(backend)
        default = backend.default_input()
        speaker = backend.default_output()
    except Exception:
        log.warning("no sound system: the previews show no devices", exc_info=True)
        return DeviceNames()
    return DeviceNames(
        mics=mics,
        outputs=outputs,
        cable=cable.name if cable else None,
        default_mic=default.name if default and default.name in mics else (mics[0] if mics else None),
        speakers=speaker.name if speaker and not is_virtual(speaker.name) else None,
    )


def guest_home(platform: str) -> PurePath:
    """A made-up guest's home, in ``platform``'s path form on any computer: the renders are published, so they never
    show this computer's user name."""
    return PureWindowsPath("C:/Users/Guest") if platform == "win32" else PurePosixPath("/Users/guest")


class PreviewEngine:
    def __init__(
        self,
        status: Status,
        mics: list[str] | None = None,
        recordings_dir: Path | None = None,
        outputs: list[str] | None = None,
        default_mic: str | None = None,
    ) -> None:
        self._status = status
        self._mics = list(mics or [])
        self._outputs = list(outputs or [])
        self._default_mic = default_mic
        self.recordings_dir = recordings_dir or Path.home() / "Music" / "Clean Voice"
        self.commands: list[tuple[str, str | None]] = []
        self.sent: list[Settings] = []  # every set_settings, in order
        self._settings = Settings()
        self.monitor_volume = 100.0
        self.vst = VstRegistry()  # no chain runs here: a removed plug-in is unloaded at once

    def set_status(self, status: Status) -> None:
        self._status = status

    def mics(self) -> list[str]:
        return list(self._mics)

    def outputs(self) -> list[str]:
        return list(self._outputs)

    def default_mic(self) -> str | None:
        return self._default_mic or (self._mics[0] if self._mics else None)

    def start(self, mic: str | None) -> None:
        self.commands.append(("start", mic))

    def stop(self) -> None:
        self.commands.append(("stop", None))

    def select_mic(self, mic: str) -> None:
        self.commands.append(("select", mic))
        self._status = dataclasses.replace(self._status, mic=mic)

    def select_output(self, output: str | None) -> None:
        self.commands.append(("select_output", output))
        if output is not None:
            self._status = dataclasses.replace(self._status, output=output)

    def refresh(self) -> None:
        self.commands.append(("refresh", None))

    def set_recording(self, on: bool) -> None:
        self.commands.append(("record", "on" if on else "off"))
        self._status = dataclasses.replace(self._status, recording_enabled=on, recording=on and self._status.recording)

    def set_settings(self, settings: Settings) -> None:
        self.commands.append(("settings", None))
        self.sent.append(settings)
        self._settings = settings

    def settings(self) -> Settings:
        return self._settings

    def set_monitor(self, on: bool) -> None:
        self.commands.append(("monitor", "on" if on else "off"))
        self._status = dataclasses.replace(self._status, monitor=on)

    def set_monitor_volume(self, volume: float) -> None:
        self.commands.append(("monitor_volume", str(float(volume))))
        self.monitor_volume = float(volume)

    def set_echo(self, on: bool) -> None:
        self.commands.append(("echo", "on" if on else "off"))
        if not on:
            self._status = dataclasses.replace(self._status, echo=EchoState.OFF, echo_reason=None)
        elif self._status.echo is EchoState.OFF:
            self._status = dataclasses.replace(self._status, echo=EchoState.ON, echo_reason=None)
        # else: a sample that already shows echo removal asked for (paused, unavailable) keeps its state.

    def start_mic_test(self) -> None:
        self.commands.append(("mic_test", "start"))
        self._status = dataclasses.replace(self._status, mic_test=MicTestState.RECORDING)

    def cancel_mic_test(self) -> None:
        self.commands.append(("mic_test", "cancel"))
        self._status = dataclasses.replace(self._status, mic_test=MicTestState.IDLE)

    def status(self) -> Status:
        return dataclasses.replace(self._status)

    def close(self) -> None:
        self.commands.append(("close", None))
        self.vst.close_all()


class PreviewMicLevel:
    """A stand-in for the mic's OS level (Step 1's Mic level): every mic at ``value`` (None: no level, as for an
    unplugged mic), reported as ``db`` (None: the OS gives no dB, so the slider shows its percent); never this
    computer's own."""

    def __init__(self, value: float | None = 0.8, db: float | None = None) -> None:
        self.value = value
        self.db = db

    def level(self, mic: str) -> float | None:
        return self.value

    def level_db(self, mic: str) -> float | None:
        return self.db if self.value is not None else None

    def set_level(self, mic: str, level: float) -> bool:
        if self.value is None:
            return False
        self.value = level
        return True

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class Sample:
    """One screen state: the engine's status and the saved choices the window starts from."""

    status: Status
    prefs: Prefs = field(default_factory=lambda: Prefs(recording=True))  # the backup on, unless a state says
    mics: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    default_mic: str | None = None
    menu_open: bool = False
    mic_level: float | None = 0.8  # the mic's OS level in Step 1; None when the mic is gone
    mic_db: float | None = -4.0  # the dB the OS reports for it
    size: tuple[int, int] | None = None  # the window's size, when not its opening size
    # The window's control that has the keyboard focus (its attribute name), drawn with its ring.
    focus: str | None = None
    theme_menu_open: bool = False  # the footer's Theme dropdown, open

    def mic_levels(self) -> PreviewMicLevel:
        return PreviewMicLevel(self.mic_level, self.mic_db)

    def engine(self, recordings_dir: Path | None = None) -> PreviewEngine:
        return PreviewEngine(
            self.status, list(self.mics), recordings_dir, list(self.outputs), default_mic=self.default_mic
        )

    def saved(self) -> Prefs:
        return self.prefs


def sample_states(devices: DeviceNames, platform: str = sys.platform) -> dict[str, Sample]:
    """Every state the one window can show (J10 plan §7, States and their look), as the real engine reports them, with
    ``platform``'s folders (a Mac render made on Windows shows the Mac's)."""
    from clean_voice.engine import ECHO_NO_SPEAKERS, ECHO_PAUSED, MONITOR_LOST

    mic = devices.default_mic or (devices.mics[0] if devices.mics else None)
    output = devices.cable or (devices.outputs[0] if devices.outputs else None)
    speakers_out = next((o for o in devices.outputs if not is_cable(o)), None)
    # The made-up guest's folder (Status holds Paths; only the error shows it, in the platform's form).
    folder = guest_home(platform) / "Music" / "Clean Voice"
    files = tuple(Path(folder, f"2026-11-10 14-03-12 {name}.wav") for name in ("clean voice", "raw mic"))
    on = Status(
        state=State.ON,
        mic=mic,
        cable=devices.cable,
        output=output,
        level=0.42,
        input_level=0.3,
        recording=True,
        recording_seconds=754.0,
        recording_files=files,
    )
    lists = {"mics": devices.mics, "outputs": devices.outputs, "default_mic": devices.default_mic}

    def sample(status: Status, **kw: object) -> Sample:
        return Sample(status, **{**lists, **kw})  # type: ignore[arg-type]

    def saved(**settings: object) -> Prefs:
        return Prefs(recording=True, settings=replace(Settings(), **settings))

    echo_on = Prefs(echo=True, recording=True)
    # Fine-tune on with the done-when's chain (gate, EQ, compressor, gain) at OBS's defaults.
    chain: list[fine_tune.FilterSpec] = []
    for kind in ("noise_gate_filter", "basic_eq_filter", "compressor_filter", "gain_filter"):
        chain.append(fine_tune.new_filter(kind, chain))
    warm = Prefs(preset="warm", recording=True, settings=Settings(bands=style_bands("warm", 0.5)))
    return {
        "running": sample(on, prefs=warm),
        "power-off": sample(on, prefs=saved(cleaning=False)),
        "starting": sample(Status(state=State.OFF, cable=devices.cable)),
        "cable-missing": sample(Status(state=State.NO_CABLE, mic=mic, cable=None)),
        # A cable lost mid-session, and the voice sent to the speakers (the first output that is not a cable).
        "cable-lost": sample(dataclasses.replace(on, state=State.CABLE_LOST, level=0.0, input_level=0.0)),
        "voice-to-speakers": sample(dataclasses.replace(on, output=speakers_out)),
        "no-mic": sample(
            Status(state=State.NO_MIC, cable=devices.cable, output=output), mics=(), default_mic=None, mic_level=None
        ),
        "mic-unplugged": sample(
            dataclasses.replace(on, state=State.MIC_LOST, level=0.0, input_level=0.0), mic_level=None
        ),
        "error": sample(
            Status(
                state=State.ERROR,
                mic=mic,
                cable=devices.cable,
                output=output,
                error="The microphone is in use by another app",
            )
        ),
        "mic-test-recording": sample(dataclasses.replace(on, mic_test=MicTestState.RECORDING, mic_test_seconds=1.6)),
        "mic-test-playing": sample(dataclasses.replace(on, mic_test=MicTestState.PLAYING, mic_test_seconds=2.0)),
        "hear-myself-on": sample(
            dataclasses.replace(on, monitor=True, monitor_device=devices.speakers),
            prefs=Prefs(recording=True, monitor_volume=70.0),
        ),
        "hear-myself-error": sample(dataclasses.replace(on, monitor=False, monitor_error=MONITOR_LOST)),
        "recording-off": sample(
            dataclasses.replace(on, recording=False, recording_enabled=False, recording_files=()),
            prefs=Prefs(recording=False),
        ),
        "backup-error": sample(
            dataclasses.replace(
                on,
                recording=False,
                recording_error=f"Can't save backups to {folder}. Your voice still reaches the call. Check that the "
                "disk has space and that the folder can be written.",
            )
        ),
        "echo-on": sample(dataclasses.replace(on, echo=EchoState.ON, echo_delay_ms=80.0), prefs=echo_on),
        "echo-paused": sample(
            dataclasses.replace(on, echo=EchoState.PAUSED, echo_reason=ECHO_PAUSED, monitor=True), prefs=echo_on
        ),
        "echo-unavailable": sample(
            dataclasses.replace(on, echo=EchoState.UNAVAILABLE, echo_reason=ECHO_NO_SPEAKERS), prefs=echo_on
        ),
        "denoise-off": sample(on, prefs=saved(denoise=False)),
        "style-off": sample(on, prefs=saved(style=False)),
        "fine-tune-on": sample(on, prefs=saved(fine_tune=True, filters=tuple(chain))),
        "no-clean-up": sample(on, prefs=saved(denoise=False, style=False)),
        "mic-menu-open": sample(on, menu_open=True),
        "minimum-size": sample(on, size=(680, 520)),
        "keyboard-focus": sample(on, focus="test_button"),
        # The footer's Theme choice (DESIGN "Theme choice", 2026-11-13) on "running": its keyboard ring, its list open.
        "theme-focus": sample(on, prefs=warm, focus="theme_combo"),
        "theme-menu-open": sample(on, prefs=warm, theme_menu_open=True),
    }


# ------------------------------------------------------------------ the filter window's states (J10 plan §7)

# Made-up plug-ins: the renders are published, so they never list this computer's own.
PREVIEW_NAMES = ("Room Tamer", "Test Gain", "Voice Shaper")


def preview_folders(platform: str = sys.platform) -> list[PurePath]:
    """The folders the no-plug-ins line names: one of OBS's on Windows, OBS's two on the Mac (``vst.search_folders``)
    under the made-up guest's home."""
    if platform == "darwin":
        home = guest_home(platform)
        return [PurePosixPath(vst.MAC_SYSTEM_FOLDER), home / vst.MAC_USER_FOLDER]
    return [PureWindowsPath("C:/Program Files/VSTPlugins")]


def preview_plugins(platform: str = sys.platform) -> tuple[tuple[str, str], ...]:
    suffix = ".vst" if platform == "darwin" else ".dll"
    folder = preview_folders(platform)[0]
    return tuple((name, (folder / f"{name}{suffix}").as_posix()) for name in PREVIEW_NAMES)


class PreviewPlugin:
    """A stand-in for a loaded VST plug-in, for the filter window's renders: its names, and an editor it never
    opens. Nothing is loaded and no audio runs through it."""

    def __init__(self, path: str, effect_name: str = "Test Gain", vendor: str = "Clean Voice tests") -> None:
        self.path = path
        self.effect_name = effect_name
        self.vendor = vendor
        self.loaded = True
        self.has_editor = True
        self.editor = None
        self.open_when_active = False

    def get_chunk(self) -> str:
        return ""

    def close(self) -> None:
        self.loaded = False


@dataclass(frozen=True)
class FilterSample:
    """One filter-window state: the list, the selected row, the plug-ins found, and what the render opens on top."""

    filters: tuple[fine_tune.FilterSpec, ...] = ()
    select: int = 0
    found: tuple[tuple[str, str], ...] | None = None  # None: ``platform``'s made-up plug-ins
    error: str | None = None  # the chosen plug-in won't load, for this reason
    menu_open: bool = False  # the + menu
    confirm_remove: bool = False  # the remove question
    rename: bool = False  # the selected row's name being edited
    platform: str = sys.platform  # whose folders and plug-in files

    def plugins(self) -> tuple[tuple[str, str], ...]:
        return preview_plugins(self.platform) if self.found is None else self.found

    def model(self) -> FilterModel:
        """A ``filter_window.FilterModel`` over these filters, with stand-in plug-ins (imported here: Qt)."""
        from clean_voice.filter_window import FilterModel

        holder = {"settings": Settings(fine_tune=bool(self.filters), filters=self.filters)}
        error = self.error

        def load(settings: Mapping[str, object]) -> VstPlugin:
            if error is not None:
                raise VstError(error)
            return cast("VstPlugin", PreviewPlugin(str(settings.get("plugin_path") or "")))

        def publish(settings: Settings) -> None:
            holder["settings"] = settings

        model = FilterModel(
            lambda: holder["settings"],
            publish,
            VstRegistry(),
            load,
            lambda: list(self.plugins()),
            lambda: preview_folders(self.platform),
        )
        model.load_saved()
        return model


def filter_samples(platform: str = sys.platform) -> dict[str, FilterSample]:
    """Every state of the filter window (§7 "filter window"): empty; each of the ten kinds selected; the VST filter
    with its plug-in loaded, failing to load and with none found; the + menu open; the remove question; a rename."""
    from clean_voice.vst import WRONG_ARCHITECTURE

    def spec(kind: str, name: str | None = None, enabled: bool = True, **settings: object) -> fine_tune.FilterSpec:
        return fine_tune.FilterSpec(kind, name or fine_tune.KINDS[kind].name, enabled, settings)

    # The done-when's chain, one of it switched off so the crossed eye shows.
    base = (
        spec("noise_gate_filter"),
        spec("basic_eq_filter"),
        spec("compressor_filter"),
        spec("gain_filter", enabled=False),
    )

    def sample(filters: tuple[fine_tune.FilterSpec, ...] = (), select: int = 0, **kw: object) -> FilterSample:
        return FilterSample(filters, select, platform=platform, **kw)  # type: ignore[arg-type]

    out: dict[str, FilterSample] = {"filters-empty": sample()}
    for kind in fine_tune.MENU_ORDER:
        at = next((i for i, f in enumerate(base) if f.kind == kind), None)
        filters = base if at is not None else (*base, spec(kind))
        out[f"filters-{kind.removesuffix('_filter').replace('_', '-')}"] = sample(
            filters, select=at if at is not None else len(base)
        )
    chosen = preview_plugins(platform)[1][1]
    with_vst = (*base, spec("vst_filter", plugin_path=chosen))
    out["filters-vst-loaded"] = sample(with_vst, select=len(base))
    out["filters-vst-failed"] = sample(with_vst, select=len(base), error=WRONG_ARCHITECTURE)
    out["filters-vst-no-plugins"] = sample((*base, spec("vst_filter")), select=len(base), found=())
    out["filters-add-menu"] = sample(base, select=1, menu_open=True)
    out["filters-remove-confirm"] = sample(base, select=2, confirm_remove=True)
    out["filters-rename"] = sample(base, select=1, rename=True)
    return out
