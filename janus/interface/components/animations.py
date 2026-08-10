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

"""Braille/Unicode animation frame registry for the Janus TUI.

Provides a centralized collection of spinner and multi-character animations
used across splash screen, header, activity feed, and status bar.
"""

from enum import Enum, auto
from typing import NamedTuple


class AnimationType(Enum):
    """Available animation types."""

    # Single-char Braille spinners
    BRAILLE_SPIN = auto()
    BRAILLE_CHASE = auto()
    BRAILLE_ZIGZAG = auto()
    BRAILLE_BOUNCE = auto()
    BRAILLE_CLIMBER = auto()
    BRAILLE_SAND = auto()

    # Multi-char Braille animations
    PENDULUM = auto()
    COMPRESS = auto()
    SORT = auto()
    WAVE = auto()
    RADAR = auto()
    DNA = auto()

    # Unicode block animations
    BLOCK_SPIN = auto()
    BLOCK_WORM = auto()
    BAR_GROW = auto()
    BAR_BOUNCE = auto()
    NOISE_FADE = auto()
    ARROWS = auto()


class AnimationDef(NamedTuple):
    """Definition of an animation sequence."""

    frames: tuple[str, ...]
    interval_ms: int
    width: int


ANIMATIONS: dict[AnimationType, AnimationDef] = {
    # ---- Single-char Braille spinners ----
    AnimationType.BRAILLE_SPIN: AnimationDef(
        frames=("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"),
        interval_ms=80,
        width=1,
    ),
    AnimationType.BRAILLE_CHASE: AnimationDef(
        frames=("⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"),
        interval_ms=100,
        width=1,
    ),
    AnimationType.BRAILLE_ZIGZAG: AnimationDef(
        frames=("⠋", "⠙", "⠚", "⠞", "⠖", "⠦", "⠴", "⠲", "⠳", "⠓"),
        interval_ms=80,
        width=1,
    ),
    AnimationType.BRAILLE_BOUNCE: AnimationDef(
        frames=("⣤", "⠶", "⠛", "⠛", "⠶"),
        interval_ms=120,
        width=1,
    ),
    AnimationType.BRAILLE_CLIMBER: AnimationDef(
        frames=(
            "⠁",
            "⠂",
            "⠄",
            "⡀",
            "⡈",
            "⡐",
            "⡠",
            "⣀",
            "⣁",
            "⣂",
            "⣄",
            "⣌",
        ),
        interval_ms=100,
        width=1,
    ),
    AnimationType.BRAILLE_SAND: AnimationDef(
        frames=(
            "⠁",
            "⠂",
            "⠄",
            "⡀",
            "⡈",
            "⡐",
            "⡠",
            "⣀",
            "⣁",
            "⣂",
            "⣄",
            "⣌",
            "⣔",
            "⣤",
            "⣥",
            "⣦",
            "⣮",
            "⣶",
            "⣷",
            "⣿",
            "⡿",
            "⠿",
            "⢟",
            "⠟",
            "⡛",
            "⠛",
            "⠫",
            "⢋",
            "⠋",
            "⠍",
            "⡉",
            "⠉",
            "⠑",
            "⠒",
            "⠂",
        ),
        interval_ms=60,
        width=1,
    ),
    # ---- Multi-char Braille animations ----
    AnimationType.PENDULUM: AnimationDef(
        frames=(
            "⠁      ",
            " ⠂     ",
            "  ⠄    ",
            "   ⡀   ",
            "    ⠄  ",
            "     ⠂ ",
            "      ⠁",
            "     ⠂ ",
            "    ⠄  ",
            "   ⡀   ",
            "  ⠄    ",
            " ⠂     ",
            "⠁      ",
            " ⠂     ",
        ),
        interval_ms=100,
        width=7,
    ),
    AnimationType.COMPRESS: AnimationDef(
        frames=(
            "⠁     ⠁",
            " ⠂   ⠂ ",
            "  ⠄ ⠄  ",
            "   ⡀   ",
            "  ⠄ ⠄  ",
            " ⠂   ⠂ ",
            "⠁     ⠁",
            "⡀     ⡀",
            " ⣀ ⣀  ",
            "  ⣤⣤   ",
        ),
        interval_ms=120,
        width=8,
    ),
    AnimationType.SORT: AnimationDef(
        frames=(
            "⣀ ⡀ ⠄ ⠂ ⠁",
            "⡀ ⣀ ⠄ ⠂ ⠁",
            "⡀ ⠄ ⣀ ⠂ ⠁",
            "⡀ ⠄ ⠂ ⣀ ⠁",
            "⡀ ⠄ ⠂ ⠁ ⣀",
            "⠄ ⡀ ⠂ ⠁ ⣀",
            "⠄ ⠂ ⡀ ⠁ ⣀",
            "⠄ ⠂ ⠁ ⡀ ⣀",
            "⠂ ⠄ ⠁ ⡀ ⣀",
            "⠂ ⠁ ⠄ ⡀ ⣀",
            "⠁ ⠂ ⠄ ⡀ ⣀",
            "⠁ ⠂ ⠄ ⡀ ⣀",
        ),
        interval_ms=150,
        width=9,
    ),
    AnimationType.WAVE: AnimationDef(
        frames=(
            "⠁⠂⠄⡀⠄⠂⠁⠀",
            "⠀⠁⠂⠄⡀⠄⠂⠁",
            "⠁⠀⠁⠂⠄⡀⠄⠂",
            "⠂⠁⠀⠁⠂⠄⡀⠄",
            "⠄⠂⠁⠀⠁⠂⠄⡀",
            "⡀⠄⠂⠁⠀⠁⠂⠄",
            "⠄⡀⠄⠂⠁⠀⠁⠂",
            "⠂⠄⡀⠄⠂⠁⠀⠁",
            "⠁⠂⠄⡀⠄⠂⠁⠀",
            "⠀⠁⠂⠄⡀⠄⠂⠁",
            "⠁⠀⠁⠂⠄⡀⠄⠂",
            "⠂⠁⠀⠁⠂⠄⡀⠄",
            "⠄⠂⠁⠀⠁⠂⠄⡀",
            "⡀⠄⠂⠁⠀⠁⠂⠄",
            "⠄⡀⠄⠂⠁⠀⠁⠂",
            "⠂⠄⡀⠄⠂⠁⠀⠁",
        ),
        interval_ms=80,
        width=8,
    ),
    AnimationType.RADAR: AnimationDef(
        frames=(
            "⠀⠀⠀⠀⠀",
            "⠁⠀⠀⠀⠀",
            "⠁⠂⠀⠀⠀",
            "⠁⠂⠄⠀⠀",
            "⠁⠂⠄⡀⠀",
            "⠁⠂⠄⡀⠄",
            "⠀⠂⠄⡀⠄",
            "⠀⠀⠄⡀⠄",
            "⠀⠀⠀⡀⠄",
            "⠀⠀⠀⠀⠄",
            "⠀⠀⠀⠀⠀",
        ),
        interval_ms=100,
        width=5,
    ),
    AnimationType.DNA: AnimationDef(
        frames=(
            "⠋⠉⠙⠹⠸⠼",
            "⠙⠋⠉⠙⠹⠸",
            "⠹⠙⠋⠉⠙⠹",
            "⠸⠹⠙⠋⠉⠙",
            "⠼⠸⠹⠙⠋⠉",
            "⠴⠼⠸⠹⠙⠋",
            "⠦⠴⠼⠸⠹⠙",
            "⠧⠦⠴⠼⠸⠹",
            "⠇⠧⠦⠴⠼⠸",
            "⠏⠇⠧⠦⠴⠼",
            "⠋⠏⠇⠧⠦⠴",
            "⠉⠋⠏⠇⠧⠦",
        ),
        interval_ms=100,
        width=6,
    ),
    # ---- Unicode block animations ----
    AnimationType.BLOCK_SPIN: AnimationDef(
        frames=("▄", "▌", "▀", "▐"),
        interval_ms=100,
        width=1,
    ),
    AnimationType.BLOCK_WORM: AnimationDef(
        frames=("▘", "▀", "▝", "▐", "▗", "▄", "▖", "▌"),
        interval_ms=100,
        width=1,
    ),
    AnimationType.BAR_GROW: AnimationDef(
        frames=(" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"),
        interval_ms=100,
        width=1,
    ),
    AnimationType.BAR_BOUNCE: AnimationDef(
        frames=(
            "▏       ",
            " ▎      ",
            "  ▍     ",
            "   ▌    ",
            "    ▋   ",
            "     ▊  ",
            "      ▉ ",
            "       █",
            "      ▉ ",
            "     ▊  ",
            "    ▋   ",
            "   ▌    ",
            "  ▍     ",
            " ▎      ",
        ),
        interval_ms=100,
        width=8,
    ),
    AnimationType.NOISE_FADE: AnimationDef(
        frames=("█", "▓", "▒", "░", " ", "░", "▒", "▓"),
        interval_ms=120,
        width=1,
    ),
    AnimationType.ARROWS: AnimationDef(
        frames=("←", "↖", "↑", "↗", "→", "↘", "↓", "↙"),
        interval_ms=100,
        width=1,
    ),
}


# --- Helper functions ---


def get_frame(anim_type: AnimationType, step: int) -> str:
    """Return the frame for the given type at *step*, wrapping via modulo."""
    anim = ANIMATIONS[anim_type]
    return anim.frames[step % len(anim.frames)]


# --- Context presets — which animations go where ---

SPLASH_SPINNER = AnimationType.BRAILLE_ZIGZAG
SPLASH_BAR = AnimationType.COMPRESS
FEED_SPINNERS: list[AnimationType] = [
    AnimationType.BRAILLE_SPIN,
    AnimationType.BRAILLE_CHASE,
    AnimationType.BRAILLE_ZIGZAG,
    AnimationType.BRAILLE_BOUNCE,
    AnimationType.BLOCK_WORM,
]
STATUS_ANIMATION = AnimationType.SORT
