# Janus — an engine for building specialized AI agents.
# Copyright (C) 2026 Pantheos Forge
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. This program is distributed WITHOUT ANY WARRANTY;
# see the GNU AGPL <https://www.gnu.org/licenses/> for details.
#
# A persona exception applies — see LICENSE-EXCEPTION.

"""Stateful amber "Crush" spinner — the TUI running indicator.

A scramble-only Crush spinner: cycling runes with a phosphorescent amber glow and a
staggered fade-in birth. ``render()`` returns a Rich ``Text`` (per-character truecolor)
so it drops straight into a Textual ``Static``. Colors come from ``theme.py``.
"""

from __future__ import annotations

import math
import random

from rich.text import Text

from janus.interface.theme import ACCENT as FORGE_AMBER
from janus.interface.theme import GLOW as FORGE_GLOW

_AVAILABLE_RUNES = list("0123456789abcdefABCDEF~!@#$%^&*()+=_")
_INITIAL_CHAR = "."
_MAX_BIRTH_STEPS = 20          # staggered entrance window
_PRERENDERED_FRAMES = 20       # one full glow pulse cycle
_DEFAULT_WIDTH = 10
_DEEP_EMBER_FACTOR = 0.30      # deep-ember ramp start = dimmed FORGE_AMBER


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))


def _rgb_to_hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def _dim(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, round(v * factor))) for v in color)  # type: ignore[return-value]


def _blend(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    # Linear RGB interpolation — preserves per-channel ordering across the ramp.
    return tuple(round(c1[k] + (c2[k] - c1[k]) * t) for k in range(3))  # type: ignore[return-value]


def _make_ramp(size: int, *stops: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    num_segments = len(stops) - 1
    base, remainder = divmod(size, num_segments)
    ramp: list[tuple[int, int, int]] = []
    for i in range(num_segments):
        seg_size = base + (1 if i < remainder else 0)
        for j in range(seg_size):
            ramp.append(_blend(stops[i], stops[i + 1], j / seg_size))
    return ramp


class CrushSpinner:
    """Stateful scramble-only amber spinner rendering to a Rich ``Text``."""

    def __init__(self, width: int = _DEFAULT_WIDTH, seed: int = 0) -> None:
        self.width = max(1, width)
        self._step = 0
        self._frames_since_start = 0
        rng = random.Random(seed)

        deep_ember = _dim(_hex_to_rgb(FORGE_AMBER), _DEEP_EMBER_FACTOR)
        amber = _hex_to_rgb(FORGE_AMBER)
        glow = _hex_to_rgb(FORGE_GLOW)
        ramp = _make_ramp(self.width, deep_ember, amber, glow)

        num_frames = _PRERENDERED_FRAMES
        # Phosphorescent breathing: brightness pulses per frame in [0.55, 1.0].
        pulses = [
            0.55 + 0.45 * (0.5 - 0.5 * math.cos(2 * math.pi * f / num_frames))
            for f in range(num_frames)
        ]

        # Pre-render styled cells: birth (dots) and cycling (runes) layers, one row per frame.
        self._initial_frames: list[list[tuple[str, str]]] = []
        self._cycling_frames: list[list[tuple[str, str]]] = []
        for f in range(num_frames):
            init_row: list[tuple[str, str]] = []
            cyc_row: list[tuple[str, str]] = []
            for j in range(self.width):
                init_row.append((_INITIAL_CHAR, _rgb_to_hex(_dim(ramp[j], pulses[f]))))
                phase = (f + j) % num_frames
                cyc_row.append(
                    (rng.choice(_AVAILABLE_RUNES), _rgb_to_hex(_dim(ramp[j], pulses[phase])))
                )
            self._initial_frames.append(init_row)
            self._cycling_frames.append(cyc_row)

        # Staggered birth schedule: per-column entrance offset.
        self._birth_steps = [rng.randrange(_MAX_BIRTH_STEPS) for _ in range(self.width)]

    @property
    def initialized(self) -> bool:
        return self._frames_since_start >= _MAX_BIRTH_STEPS

    def advance(self) -> None:
        self._step = (self._step + 1) % _PRERENDERED_FRAMES
        self._frames_since_start += 1

    def render(self) -> Text:
        text = Text()
        for i in range(self.width):
            if not self.initialized and self._frames_since_start < self._birth_steps[i]:
                char, style = self._initial_frames[self._step][i]
            else:
                char, style = self._cycling_frames[self._step][i]
            text.append(char, style=style)
        return text
