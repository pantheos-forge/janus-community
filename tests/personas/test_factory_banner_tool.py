"""Hermetic tests for the set_persona_banner factory tool. No network."""

import io
import json
from pathlib import Path

import pytest

from janus.core.persona import Persona, banner_errors
from janus.core.tools.registry import ToolContext

FACTORY_DIR = Path(__file__).resolve().parents[2] / "personas" / "factory"


@pytest.fixture
def persona():
    return Persona.load(FACTORY_DIR)


def make_png(w=40, h=40) -> bytes:
    from PIL import Image

    im = Image.new("L", (w, h), 255)
    for x in range(10, 30):
        for y in range(10, 30):
            im.putpixel((x, y), 0)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def fake_fetcher(license_name="Public domain", png: bytes | None = None):
    def fetch(url, params=None):
        if params and params.get("action") == "query":
            return json.dumps({"query": {"pages": {"1": {"imageinfo": [{
                "url": "https://upload.example/x.png",
                "thumburl": "https://upload.example/thumb.png",
                "extmetadata": {"LicenseShortName": {"value": license_name}},
            }]}}}}).encode()
        return png if png is not None else make_png()
    return fetch


def ctx_for(tmp_path, **extra) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), extra=extra)


@pytest.mark.asyncio
async def test_preview_returns_valid_braille(persona, tmp_path):
    ctx = ctx_for(tmp_path, banner_fetcher=fake_fetcher())
    result = await persona.registry.dispatch(
        "set_persona_banner", {"commons_file": "File:X.png"}, ctx)
    assert result.startswith("Preview (Public domain):")
    art = result.split("\n", 1)[1]
    assert banner_errors(art) == []


@pytest.mark.asyncio
async def test_non_pd_license_is_refused(persona, tmp_path):
    ctx = ctx_for(tmp_path, banner_fetcher=fake_fetcher("CC BY-SA 4.0"))
    result = await persona.registry.dispatch(
        "set_persona_banner", {"commons_file": "File:X.png"}, ctx)
    assert result.startswith("Error:")
    assert "CC BY-SA 4.0" in result


@pytest.mark.asyncio
async def test_non_commons_title_is_refused(persona, tmp_path):
    ctx = ctx_for(tmp_path, banner_fetcher=fake_fetcher())
    result = await persona.registry.dispatch(
        "set_persona_banner",
        {"commons_file": "https://example.com/img.png"}, ctx)
    assert result.startswith("Error:")
    assert "File:" in result


@pytest.mark.asyncio
async def test_write_requires_scaffold_first(persona, tmp_path):
    ctx = ctx_for(tmp_path, banner_fetcher=fake_fetcher())
    result = await persona.registry.dispatch(
        "set_persona_banner",
        {"commons_file": "File:X.png", "write": True, "name": "demo_agent"}, ctx)
    assert result.startswith("Error:")
    # Bite on the guard's actual error text — a loose "scaffold" match would
    # also hit pytest's tmp_path (which embeds this test's name).
    assert "call scaffold_persona first" in result
    assert not (tmp_path / "build" / "demo_agent" / "banner.txt").exists()


@pytest.mark.asyncio
async def test_write_creates_banner_and_provenance(persona, tmp_path):
    build = tmp_path / "build" / "demo_agent"
    build.mkdir(parents=True)
    (build / "manifest.toml").write_text('[persona]\nname = "demo_agent"\n')
    ctx = ctx_for(tmp_path, banner_fetcher=fake_fetcher())
    result = await persona.registry.dispatch(
        "set_persona_banner",
        {"commons_file": "File:X.png", "write": True, "name": "demo_agent"}, ctx)
    assert not result.startswith("Error:")
    art = (build / "banner.txt").read_text().rstrip("\n")
    assert banner_errors(art) == []
    src = (build / "banner_source.txt").read_text()
    assert "File:X.png" in src and "Public domain" in src


@pytest.mark.asyncio
async def test_pillow_missing_degrades_gracefully(persona, tmp_path, monkeypatch):
    import janus.factory.banner as banner_mod

    monkeypatch.setattr(banner_mod, "_PIL_AVAILABLE", False)
    ctx = ctx_for(tmp_path, banner_fetcher=fake_fetcher())
    result = await persona.registry.dispatch(
        "set_persona_banner", {"commons_file": "File:X.png"}, ctx)
    assert result.startswith("Error:")
    assert "janus[art]" in result
