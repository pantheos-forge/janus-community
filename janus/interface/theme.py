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

"""Generic dark color palette — single source of truth for TUI colors.

TCSS (styles.tcss) duplicates these values because Textual CSS cannot
import Python. If you change a color here, update styles.tcss too.

This module is pure hex-string constants and must never import ``textual``:
it is used by headless/CLI rendering paths that do not require the
optional ``[tui]`` extra.
"""

# --- Foundation / Dark System ---
OBSIDIAN = "#0A0C10"  # Deepest background
GUNMETAL = "#161B22"  # Panels, headers
DARK_STEEL = "#1C2128"  # Elevated surfaces, inputs
STEEL = "#2D333B"  # Borders, dividers

# --- Accent / Primary ---
ACCENT = "#E8913A"  # Primary brand color
ACCENT_SECONDARY = "#F5B84C"  # Secondary accent
GLOW = "#FFCF70"  # Highlight / glow effects

# --- Secondary / Functional ---
ELECTRIC_BLUE = "#3B82F6"
CYBER_TEAL = "#14B8A6"
PLASMA_VIOLET = "#8B5CF6"
DANGER_RED = "#EF4444"
SUCCESS_GREEN = "#22C55E"

# --- Text ---
TEXT_PRIMARY = "#E6EDF3"  # Main body text
TEXT_SECONDARY = "#8B949E"  # Muted / subtitle text
TEXT_DIM = "#484F58"  # Timestamps, disabled
TEXT_BRIGHT = "#C9D1D9"  # Button text, emphasis

# --- Derived / Tinted Backgrounds ---
BG_SURFACE = "#0F1218"  # Slightly lighter than Obsidian
BORDER_TOOL = "#242930"  # Tool block borders
BG_ERROR_TINT = "#1A0E0E"  # Error-tinted background
BG_ERROR_BUTTON = "#2A1818"  # Error button background
BG_WARNING_TINT = "#1A1508"  # Warning-tinted background

# --- Semantic Aliases (what the UI references) ---
PRIMARY = ACCENT
SECONDARY = ACCENT_SECONDARY
HIGHLIGHT = GLOW
SUCCESS = SUCCESS_GREEN
WARNING = ACCENT_SECONDARY
ERROR = DANGER_RED
THINKING = PLASMA_VIOLET

# Neutral monospace/command accent, kept for components that render
# machine-generated output (e.g. an activity feed) in a fixed-width style.
MONO_CMD = SUCCESS_GREEN
