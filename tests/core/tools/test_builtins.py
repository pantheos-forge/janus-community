import pytest
from janus.core.tools.builtins import BUILTINS, builtin_registry
from janus.core.tools.registry import ToolContext


def test_exposes_the_builtin_toolset():
    assert set(BUILTINS) == {
        "bash", "write_file", "read_file", "edit_file",
        "grep", "glob", "web_fetch", "ask_user", "update_plan",
    }


def test_no_report_finding_tool():
    assert "report_finding" not in BUILTINS


@pytest.mark.asyncio
async def test_bash_runs_in_ctx_cwd(tmp_path):
    ctx = ToolContext(cwd=tmp_path)
    out = await BUILTINS["bash"].handler(ctx, command="pwd")
    assert str(tmp_path) in out


@pytest.mark.asyncio
async def test_write_then_read(tmp_path):
    ctx = ToolContext(cwd=tmp_path)
    await BUILTINS["write_file"].handler(ctx, path="note.txt", content="hello janus")
    assert (tmp_path / "note.txt").read_text() == "hello janus"
    out = await _maybe_await(BUILTINS["read_file"].handler(ctx, path="note.txt"))
    assert "hello janus" in out


@pytest.mark.asyncio
async def test_edit_file_replaces_unique_substring(tmp_path):
    ctx = ToolContext(cwd=tmp_path)
    (tmp_path / "f.txt").write_text("alpha beta gamma")
    await _maybe_await(BUILTINS["edit_file"].handler(ctx, path="f.txt", old_string="beta", new_string="BETA"))
    assert (tmp_path / "f.txt").read_text() == "alpha BETA gamma"


def test_builtin_registry_selects_subset():
    reg = builtin_registry(["bash", "write_file"])
    assert sorted(reg.names()) == ["bash", "write_file"]


def test_builtin_registry_unknown_name_raises():
    with pytest.raises(KeyError):
        builtin_registry(["bash", "nmap"])


async def _maybe_await(v):
    import inspect
    return await v if inspect.isawaitable(v) else v


@pytest.mark.asyncio
async def test_ask_user_fails_open_without_a_bridge(tmp_path):
    from janus.core.tools.builtins import BUILTINS
    from janus.core.tools.registry import ToolContext

    ctx = ToolContext(cwd=tmp_path)  # await_user_reply defaults to None
    result = await BUILTINS["ask_user"].handler(ctx, question="Proceed?")
    assert result == (
        "No user is available in this run; proceed with reasonable "
        "assumptions and complete the task."
    )


@pytest.mark.asyncio
async def test_ask_user_returns_the_bridged_reply(tmp_path):
    from janus.core.tools.builtins import BUILTINS
    from janus.core.tools.registry import ToolContext

    async def bridge(question, choices=None):
        assert question == "Which domain?"
        return "healthcare"

    ctx = ToolContext(cwd=tmp_path, await_user_reply=bridge)
    result = await BUILTINS["ask_user"].handler(ctx, question="Which domain?")
    assert result == "healthcare"


@pytest.mark.asyncio
async def test_ask_user_requires_a_question(tmp_path):
    from janus.core.tools.builtins import BUILTINS
    from janus.core.tools.registry import ToolContext

    result = await BUILTINS["ask_user"].handler(ToolContext(cwd=tmp_path), question="  ")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_ask_user_passes_choices_through_the_bridge(tmp_path):
    from janus.core.tools.builtins import BUILTINS
    from janus.core.tools.registry import ToolContext

    seen = {}

    async def bridge(question, choices):
        seen["q"], seen["c"] = question, choices
        return "Approve the spec"

    ctx = ToolContext(cwd=tmp_path, await_user_reply=bridge)
    result = await BUILTINS["ask_user"].handler(
        ctx, question="Approve?", choices=["Approve the spec", "Request changes"])
    assert result == "Approve the spec"
    assert seen["c"] == ["Approve the spec", "Request changes"]


@pytest.mark.asyncio
async def test_ask_user_fail_open_ignores_choices(tmp_path):
    from janus.core.tools.builtins import BUILTINS
    from janus.core.tools.registry import ToolContext

    result = await BUILTINS["ask_user"].handler(
        ToolContext(cwd=tmp_path), question="Approve?", choices=["A", "B"])
    assert result == (
        "No user is available in this run; proceed with reasonable "
        "assumptions and complete the task."
    )


@pytest.mark.asyncio
async def test_web_fetch_rejects_urls_with_whitespace(tmp_path):
    """The live crash input: a model-emitted URL with an internal space.
    Must return an actionable error string, never raise."""
    from janus.core.tools.builtins import BUILTINS
    from janus.core.tools.registry import ToolContext

    result = BUILTINS["web_fetch"].handler(
        ToolContext(cwd=tmp_path), url="https://www.thekelleyf irmlaw.com"
    )
    assert isinstance(result, str)
    assert result.startswith("Error:")
    assert "whitespace" in result or "control" in result
