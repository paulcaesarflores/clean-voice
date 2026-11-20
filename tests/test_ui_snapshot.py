"""The ui-snapshot render tool writes every screen state, from real (or no) device names only (J07 S1, S2)."""

import inspect
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtGui import QImage

from clean_voice import filter_window, preview
from clean_voice.app import MIN_SIZE
from clean_voice.preview import DeviceNames, filter_samples, sample_states

ROOT = Path(__file__).resolve().parents[1]
STATES = [  # J10 plan §7, States and their look
    "running",
    "power-off",
    "starting",
    "cable-missing",
    "cable-lost",
    "voice-to-speakers",
    "no-mic",
    "mic-unplugged",
    "error",
    "mic-test-recording",
    "mic-test-playing",
    "hear-myself-on",
    "hear-myself-error",
    "recording-off",
    "backup-error",
    "echo-on",
    "echo-paused",
    "echo-unavailable",
    "denoise-off",
    "style-off",
    "fine-tune-on",
    "no-clean-up",
    "mic-menu-open",
    "minimum-size",
    "keyboard-focus",  # round 15: the focus ring, on Test your mic
    "theme-focus",  # the footer's Theme choice (DESIGN "Theme choice", 2026-11-13): its focus ring
    "theme-menu-open",  # and its list open
]


# The filter window's (§7 "filter window"): empty, each of the ten kinds selected, the VST filter's plug-in loaded,
# failing and none found, the + menu, the remove question and a rename.
FILTER_STATES = [
    "filters-empty",
    "filters-clean-voice-denoise",
    "filters-clean-voice-style",
    "filters-basic-eq",
    "filters-compressor",
    "filters-expander",
    "filters-gain",
    "filters-invert-polarity",
    "filters-limiter",
    "filters-nvidia-audiofx",
    "filters-noise-gate",
    "filters-noise-suppress",
    "filters-upward-compressor",
    "filters-vst",
    "filters-vst-loaded",
    "filters-vst-failed",
    "filters-vst-no-plugins",
    "filters-add-menu",
    "filters-remove-confirm",
    "filters-rename",
]


def test_every_planned_state_is_sampled() -> None:
    assert list(sample_states(DeviceNames())) == STATES
    assert list(filter_samples()) == FILTER_STATES


def test_renders_every_filter_window_state_light_and_dark(tmp_path: Path) -> None:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    command = [sys.executable, str(ROOT / "tools" / "ui_snapshot.py"), "--out", str(tmp_path)]
    for name in FILTER_STATES:
        command += ["--state", name]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr
    for name in FILTER_STATES:
        for stem in (name, f"{name}-dark"):
            one, two = QImage(str(tmp_path / f"{stem}.png")), QImage(str(tmp_path / f"{stem}@2x.png"))
            assert one.size().toTuple() == filter_window.SIZE, stem
            assert two.size().toTuple() == (2 * filter_window.SIZE[0], 2 * filter_window.SIZE[1]), stem
        light, dark = QImage(str(tmp_path / f"{name}.png")), QImage(str(tmp_path / f"{name}-dark.png"))
        assert light.pixelColor(5, 5).lightness() > dark.pixelColor(5, 5).lightness()
    # The + menu and the question are drawn over the window where they open.
    plain, menu = QImage(str(tmp_path / "filters-basic-eq.png")), QImage(str(tmp_path / "filters-add-menu.png"))
    assert plain != menu


def test_renders_every_state_light_and_dark_at_1x_and_2x(tmp_path: Path) -> None:
    # S1
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "ui_snapshot.py"), "--out", str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for name in sample_states(DeviceNames()):  # the filter window's states have their own test below
        for stem in (name, f"{name}-dark"):  # the light and the dark theme
            one, two = QImage(str(tmp_path / f"{stem}.png")), QImage(str(tmp_path / f"{stem}@2x.png"))
            assert not one.isNull() and not two.isNull(), stem
            assert one.width() >= 680 and one.height() >= 520, stem  # MIN_SIZE (J10 open question 9)
            assert (two.width(), two.height()) == (2 * one.width(), 2 * one.height()), stem
        light, dark = QImage(str(tmp_path / f"{name}.png")), QImage(str(tmp_path / f"{name}-dark.png"))
        assert light.pixelColor(5, light.height() - 5).lightness() > dark.pixelColor(5, dark.height() - 5).lightness()


def test_renders_another_platforms_text(tmp_path: Path) -> None:
    # Release r1, m4: the download page's Mac screenshot. Only the platform's text differs ("Pick VB-Cable").
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    images = {}
    for platform in ("win32", "darwin"):
        out = tmp_path / platform
        command = [sys.executable, str(ROOT / "tools" / "ui_snapshot.py"), "--out", str(out), "--state", "running"]
        result = subprocess.run(
            [*command, "--platform", platform], env=env, capture_output=True, text=True, timeout=120, check=False
        )
        assert result.returncode == 0, result.stderr
        images[platform] = QImage(str(out / "running.png"))
        assert not images[platform].isNull()
    assert images["win32"].size() == images["darwin"].size()
    assert images["win32"] != images["darwin"]


def test_a_mac_render_shows_the_macs_folders_and_plugins_from_any_computer() -> None:
    # Review J10 r1 m2: rendered on Windows with --platform darwin, the backup folder and the VST folders and files
    # read as Windows ones. The Mac's are made up too (no user name), in the Mac's form.
    for platform, folder, suffix in (
        ("darwin", "/Users/guest/Music/Clean Voice", ".vst"),
        ("win32", "C:\\Users\\Guest\\Music\\Clean Voice", ".dll"),
    ):
        error = sample_states(DeviceNames(), platform)["backup-error"].status.recording_error
        assert error is not None and f"Can't save backups to {folder}." in error, platform
        samples = filter_samples(platform)
        assert all(path.endswith(suffix) for _, path in samples["filters-vst"].plugins()), platform
        model = samples["filters-vst-no-plugins"].model()
        text = filter_window.no_plugins_text(model.folders())
        if platform == "darwin":
            assert text.splitlines()[1:] == ["/Library/Audio/Plug-Ins/VST", "/Users/guest/Library/Audio/Plug-ins/VST"]
        else:
            assert text.splitlines()[1:] == ["C:\\Program Files\\VSTPlugins"]
        model.registry.close_all()


def test_previews_use_only_the_device_names_they_are_given() -> None:
    # S2
    given = {"Mic X", "Out Y"}
    for name, sample in sample_states(DeviceNames(mics=("Mic X",), outputs=("Out Y",), cable=None)).items():
        s = sample.status
        names = {s.mic, s.cable, s.output, s.monitor_device, *sample.mics, *sample.outputs} - {None}
        assert names <= given, (name, names - given)
    source = inspect.getsource(preview)
    assert not hasattr(preview, "SAMPLE_MICS") and not hasattr(preview, "SAMPLE_CABLE")
    for invented in ("Yeti", "Realtek", "Jabra", "VB-Audio Virtual Cable"):
        assert invented not in source


def test_the_minimum_size_state_renders_at_the_minimum(tmp_path: Path) -> None:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    command = [
        sys.executable,
        str(ROOT / "tools" / "ui_snapshot.py"),
        "--out",
        str(tmp_path),
        "--state",
        "minimum-size",
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    assert QImage(str(tmp_path / "minimum-size.png")).size().toTuple() == MIN_SIZE
    assert QImage(str(tmp_path / "minimum-size@2x.png")).size().toTuple() == (2 * MIN_SIZE[0], 2 * MIN_SIZE[1])


def test_the_theme_menu_is_drawn_over_the_window_where_it_opens(tmp_path: Path) -> None:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    command = [sys.executable, str(ROOT / "tools" / "ui_snapshot.py"), "--out", str(tmp_path)]
    for name in ("running", "theme-menu-open"):
        command += ["--state", name]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    plain, menu = QImage(str(tmp_path / "running.png")), QImage(str(tmp_path / "theme-menu-open.png"))
    assert plain.size() == menu.size() and plain != menu
    # Only the footer's corner changes: the list opens from the dropdown, over the bottom right of the window.
    changed = [
        (x, y)
        for y in range(0, plain.height(), 2)
        for x in range(0, plain.width(), 2)
        if plain.pixelColor(x, y) != menu.pixelColor(x, y)
    ]
    assert changed
    assert min(x for x, _ in changed) > plain.width() // 3 and min(y for _, y in changed) > plain.height() // 2
