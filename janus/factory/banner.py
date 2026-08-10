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

"""Commons-sourced braille banner generation for factory-built agents.

Pillow is an optional dependency (``janus[art]``): the module imports
without it, and ``convert_to_braille`` raises ``BannerError`` when it is
missing so callers degrade to a bannerless build.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _PIL_AVAILABLE = False

_COMMONS_API = "https://commons.wikimedia.org/w/api.php"
_ALLOWED_LICENSES = ("Public domain", "CC0")
_THUMB_WIDTH = 960
_MAX_DOT_COLS = 140  # 70 braille chars
# (dx, dy, bit) for the 8 dots of a braille cell.
_DOTS = ((0, 0, 0x01), (0, 1, 0x02), (0, 2, 0x04), (1, 0, 0x08),
         (1, 1, 0x10), (1, 2, 0x20), (0, 3, 0x40), (1, 3, 0x80))

Fetcher = Callable[..., bytes]


class BannerError(Exception):
    """A banner could not be produced; message is safe to show the model."""


def default_fetcher(url: str, params: dict | None = None) -> bytes:
    import httpx

    from janus import __version__

    # Wikimedia's User-Agent policy 403s requests with no UA or a generic
    # library UA, so identify the tool + version + a contact URL.
    # https://meta.wikimedia.org/wiki/User-Agent_policy
    user_agent = (
        f"Janus-Factory/{__version__} "
        "(https://github.com/pantheosforge/janus; banner art)"
    )
    resp = httpx.get(
        url,
        params=params,
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": user_agent},
    )
    resp.raise_for_status()
    return resp.content


def fetch_commons_image(fetcher: Fetcher, commons_file: str) -> tuple[bytes, str]:
    """Resolve a Commons file title to image bytes; enforce the license gate.

    Returns ``(image_bytes, license_short_name)``. Raises ``BannerError``
    for missing files, disallowed licenses, or a URL-less API response.
    """
    body = fetcher(_COMMONS_API, {
        "action": "query", "titles": commons_file, "prop": "imageinfo",
        "iiprop": "url|extmetadata", "iiurlwidth": _THUMB_WIDTH,
        "format": "json",
    })
    data = json.loads(body)
    pages = (data.get("query") or {}).get("pages") or {}
    page = next(iter(pages.values()), None)
    if not page or "imageinfo" not in page:
        raise BannerError(f"Commons file not found: {commons_file}")
    info = page["imageinfo"][0]
    meta = info.get("extmetadata") or {}
    license_name = (meta.get("LicenseShortName") or {}).get("value", "")
    if license_name not in _ALLOWED_LICENSES:
        raise BannerError(
            f"license {license_name or 'unknown'!r} is not allowed "
            "(need Public domain or CC0)")
    thumb = info.get("thumburl") or info.get("url")
    if not thumb:
        raise BannerError("Commons API returned no downloadable URL")
    return fetcher(thumb), license_name


def convert_to_braille(
    image_bytes: bytes,
    *,
    rows: int = 16,
    crop: list[float] | None = None,
    invert: bool = False,
    threshold: int = 128,
) -> str:
    """1-bit-threshold ``image_bytes`` and map 2x4 dot blocks to braille.

    Dots are lit for dark pixels by default (``invert`` flips). Width
    follows the (cropped) aspect ratio; if it would exceed 70 chars the
    whole banner shrinks to fit rather than cropping.
    """
    if not _PIL_AVAILABLE:
        raise BannerError("banner art unavailable: install janus[art] (Pillow)")
    im = Image.open(io.BytesIO(image_bytes))
    if "A" in im.getbands():
        white = Image.new("L", im.size, 255)
        white.paste(im.convert("L"), mask=im.getchannel("A"))
        im = white
    else:
        im = im.convert("L")
    if crop:
        left, top, right, bottom = crop
        w, h = im.size
        box = (int(left * w), int(top * h), int(right * w), int(bottom * h))
        if box[0] >= box[2] or box[1] >= box[3]:
            raise BannerError(f"empty crop box: {crop}")
        im = im.crop(box)
    rows = max(8, min(17, int(rows)))
    dots_h = rows * 4
    w, h = im.size
    dots_w = max(2, round(dots_h * (w / h)))
    dots_w += dots_w % 2
    if dots_w > _MAX_DOT_COLS:
        dots_h = max(32, int(dots_h * (_MAX_DOT_COLS / dots_w)) // 4 * 4)
        dots_w = _MAX_DOT_COLS
    small = im.resize((dots_w, dots_h), Image.LANCZOS)
    px = small.load()
    lines: list[str] = []
    for row in range(dots_h // 4):
        chars: list[str] = []
        for col in range(dots_w // 2):
            bits = 0
            for dx, dy, bit in _DOTS:
                lit = px[col * 2 + dx, row * 4 + dy] < threshold
                if invert:
                    lit = not lit
                if lit:
                    bits |= bit
            chars.append(chr(0x2800 + bits))
        lines.append("".join(chars))
    return "\n".join(lines)
