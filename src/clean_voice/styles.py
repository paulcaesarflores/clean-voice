"""The six voice styles: their names and band gains. Its own module so that both ``settings`` (the style's bands) and
``fine_tune`` (the Voice Style filter's list) read the one table, with no import cycle between them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    gains: tuple[float, float, float, float, float]


# main.js:136-143: the names and gains. The styles carry no description (user decision, J07 round 4).
PRESETS: dict[str, Preset] = {
    "natural": Preset("Natural", (0.0, -2.0, 0.0, 2.5, 1.5)),
    "warm": Preset("Warm", (3.0, 0.5, -1.0, 1.0, -1.0)),
    "crisp": Preset("Crisp", (-1.5, -3.0, 0.0, 4.0, 4.0)),
    "radio": Preset("Radio", (4.0, -3.0, 1.0, 4.0, 2.5)),
    "deep": Preset("Deep", (6.0, 1.5, -1.0, 1.0, 0.0)),
    "flat": Preset("Flat", (0.0, 0.0, 0.0, 0.0, 0.0)),
}
