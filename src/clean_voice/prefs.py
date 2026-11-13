"""The guest's saved choices: what the original kept in localStorage (simple-clean-mic ui/main.js:9-16, 145-146,
200, 407, 416, 436, 447, 688), saved as one JSON file.

A missing, unreadable or corrupt file, or any one bad value in it, gives that value's default: saved choices can never
block the launch. Numbers out of range are clamped by ``Settings`` and ``Band``.

Since the 2026-11-11 decisions the style's bands are not saved: they follow from the style and its Strength. A file from
before (a "custom" curve, "bands", or Bass/Mid/Treble) loads without an error and those values are ignored.

J10 adds the Step 2 switches ("denoise", "style", "fine_tune") and Fine-tune's "filters": a list of OBS's four fields
per filter, {"id", "name", "enabled", "settings"} (``fine_tune.filters_from_json`` keeps the valid entries, clamped
and renamed). A file from before J10 has none of them and loads with Fine-tune off and empty. Its "user_eq" (the
deleted 7-band EQ) and "tab" (the deleted sidebar's page) are ignored, and are not written back.

"theme" is the window's Light / Dark / Automatic choice (DESIGN "Theme choice", 2026-11-13): "auto", "light" or
"dark". A file without it, or with any other value, loads as Automatic.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clean_voice import fine_tune, theme
from clean_voice.settings import PRESETS, STYLE_STRENGTH, Settings, style_bands

FILE_NAME = "settings.json"
CUSTOM = "custom"  # an older file's own curve (the old Fine-tune): loads as the default style


@dataclass(frozen=True)
class Prefs:
    preset: str = "natural"  # a PRESETS key
    style_strength: float = STYLE_STRENGTH  # the style's Strength, 0..1
    settings: Settings = field(default_factory=Settings)  # the style's bands at that strength, the Step 2 switches, ...
    echo: bool = False  # DESIGN Features: echo removal is off by default (the original's was on, main.js:416)
    recording: bool = False  # DESIGN Features: the backup is off by default (user decision, 2026-11-11)
    mic: str | None = None  # None: the default microphone
    output: str | None = None  # None: VB-CABLE
    monitor_volume: float = 100.0  # Hear myself's volume, 0..100 %
    theme: str = "auto"  # a theme.MODES value: Automatic follows the system


def default_path() -> Path:
    """settings.json in the app's config folder (QStandardPaths.AppConfigLocation; needs the application name set)."""
    from PySide6.QtCore import QStandardPaths

    folder = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
    return Path(folder) / FILE_NAME


def to_json(prefs: Prefs) -> dict[str, Any]:
    s = prefs.settings
    return {
        "preset": prefs.preset,
        "style_strength": prefs.style_strength,
        "loudness": s.loudness,
        "strength": s.strength,
        "cleaning": s.cleaning,
        "denoise": s.denoise,
        "style": s.style,
        "fine_tune": s.fine_tune,
        "filters": fine_tune.filters_to_json(s.filters),
        "echo": prefs.echo,
        "recording": prefs.recording,
        "mic": prefs.mic,
        "output": prefs.output,
        "monitor_volume": prefs.monitor_volume,
        "theme": prefs.theme,
    }


def save(path: Path, prefs: Prefs) -> None:
    """Write atomically: a temp file in the same folder, then ``os.replace``, so a crash leaves the old file or the
    new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_json(prefs), indent=2, ensure_ascii=False)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load(path: Path) -> Prefs:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # missing, a folder, unreadable, not UTF-8 or not JSON
        return Prefs()
    if not isinstance(data, dict):
        return Prefs()
    return from_json(data)


def _number(data: dict[str, Any], key: str, default: float) -> float:
    v = data.get(key)
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        return default
    return float(v)


def _flag(data: dict[str, Any], key: str, default: bool) -> bool:
    v = data.get(key)
    return v if isinstance(v, bool) else default


def _name(data: dict[str, Any], key: str) -> str | None:
    v = data.get(key)
    return v if isinstance(v, str) and v else None


def from_json(data: dict[str, Any]) -> Prefs:
    d = Prefs()
    preset = data.get("preset")
    if not isinstance(preset, str) or preset not in PRESETS:
        preset = d.preset  # also an older file's "custom" curve
    strength = min(1.0, max(0.0, _number(data, "style_strength", d.style_strength)))
    settings = Settings(
        strength=_number(data, "strength", d.settings.strength),
        loudness=_number(data, "loudness", d.settings.loudness),
        bands=style_bands(preset, strength),
        cleaning=_flag(data, "cleaning", d.settings.cleaning),
        denoise=_flag(data, "denoise", d.settings.denoise),
        style=_flag(data, "style", d.settings.style),
        fine_tune=_flag(data, "fine_tune", d.settings.fine_tune),
        filters=fine_tune.filters_from_json(data.get("filters")),
    )
    return Prefs(
        preset=preset,
        style_strength=strength,
        settings=settings,
        echo=_flag(data, "echo", d.echo),
        recording=_flag(data, "recording", d.recording),
        mic=_name(data, "mic"),
        output=_name(data, "output"),
        monitor_volume=min(100.0, max(0.0, _number(data, "monitor_volume", d.monitor_volume))),
        theme=mode if isinstance(mode := data.get("theme"), str) and mode in theme.MODES else d.theme,
    )
