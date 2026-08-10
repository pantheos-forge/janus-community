"""Persona.load banner.txt handling + the shared banner validator."""

import pytest

from janus.core.persona import Persona, banner_errors

VALID_BANNER = "\n".join(["⣿" * 10] * 8)  # 8 rows x 10 cols, braille-only


def make_persona_dir(tmp_path, banner: str | None = None):
    d = tmp_path / "p"
    d.mkdir()
    (d / "manifest.toml").write_text(
        '[persona]\nname = "p"\n\n[prompt]\nfile = "prompt.md"\n\n'
        '[task]\ntemplate = "do {subject}"\n'
    )
    (d / "prompt.md").write_text("You are p.")
    if banner is not None:
        (d / "banner.txt").write_text(banner + "\n")
    return d


def test_banner_absent_is_none(tmp_path):
    p = Persona.load(make_persona_dir(tmp_path))
    assert p.banner is None


def test_valid_banner_exposed_without_trailing_newline(tmp_path):
    p = Persona.load(make_persona_dir(tmp_path, banner=VALID_BANNER))
    assert p.banner == VALID_BANNER


@pytest.mark.parametrize(
    "bad, fragment",
    [
        ("\n".join(["⣿⣿"] * 3), "rows"),          # too few rows
        ("\n".join(["⣿⣿"] * 20), "rows"),         # too many rows
        ("\n".join(["⣿" * 71] * 8), "columns"),   # too wide
        ("\n".join(["X" * 10] * 8), "braille"),   # non-braille chars
    ],
)
def test_invalid_banner_raises_descriptive_valueerror(tmp_path, bad, fragment):
    d = make_persona_dir(tmp_path, banner=bad)
    with pytest.raises(ValueError) as exc:
        Persona.load(d)
    assert "banner.txt" in str(exc.value)
    assert fragment in str(exc.value)


def test_banner_errors_valid_is_empty():
    assert banner_errors(VALID_BANNER) == []


def test_banner_errors_reports_all_violations():
    errs = banner_errors("X" * 80)  # 1 row, 80 cols, non-braille
    assert len(errs) == 3
