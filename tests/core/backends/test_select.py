import pytest
from janus.core.config import load_config
from janus.core.tools.registry import ToolRegistry
from janus.core.backends.select import build_backend
from janus.core.backends.ds4 import Ds4Backend
from janus.core.backends.openrouter import OpenRouterBackend
from janus.core.backends.ollama import OllamaBackend
from janus.core.backends.anthropic_api import AnthropicAPIBackend
from janus.core.backends.claude_sdk import ClaudeSDKBackend


@pytest.fixture(autouse=True)
def _isolate_from_repo_dotenv(tmp_path, monkeypatch):
    """load_config() reads ./.env (pydantic-settings env_file). A developer's
    real .env in the repo root (e.g. an OpenRouter key for live runs) must not
    leak provider selection into these tests — they assert on precedence from
    a clean slate."""
    monkeypatch.chdir(tmp_path)


def _b(cfg):
    return build_backend(cfg, system_prompt="sys", registry=ToolRegistry())


def test_ds4_precedence_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config(ds4_url="http://127.0.0.1:8000", openrouter_model="anthropic/claude-sonnet-5",
                      local_model="qwen")
    assert isinstance(_b(cfg), Ds4Backend)


def test_openrouter_over_local(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config(openrouter_model="anthropic/claude-sonnet-5", openrouter_api_key="sk-or-1", local_model="qwen")
    assert isinstance(_b(cfg), OpenRouterBackend)


def test_local_model_selects_ollama(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config(local_model="qwen:latest")
    assert isinstance(_b(cfg), OllamaBackend)


def test_anthropic_default_from_config_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config(anthropic_api_key="sk-ant-cfg")
    b = build_backend(cfg, system_prompt="sys", registry=ToolRegistry())
    assert isinstance(b, AnthropicAPIBackend)
    assert b._api_key == "sk-ant-cfg"


def test_anthropic_default_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    cfg = load_config()
    b = build_backend(cfg, system_prompt="sys", registry=ToolRegistry())
    assert isinstance(b, AnthropicAPIBackend)
    assert b._api_key == "sk-ant-env"


def test_no_provider_and_no_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config()
    with pytest.raises(NotImplementedError, match="No provider configured"):
        build_backend(cfg, system_prompt="sys", registry=ToolRegistry())


def test_use_claude_agent_sdk_selects_sdk_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config(use_claude_agent_sdk=True)
    b = build_backend(cfg, system_prompt="sys", registry=ToolRegistry())
    assert isinstance(b, ClaudeSDKBackend)


def test_sdk_flag_beats_anthropic_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config(use_claude_agent_sdk=True, anthropic_api_key="sk-ant-1")
    assert isinstance(
        build_backend(cfg, system_prompt="s", registry=ToolRegistry()), ClaudeSDKBackend
    )


def test_build_backend_for_persona_uses_persona_prompt_and_registry(tmp_path, monkeypatch):
    from pathlib import Path
    from janus.core.persona import Persona
    from janus.core.backends.select import build_backend_for_persona
    from janus.core.backends.anthropic_api import AnthropicAPIBackend

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    fixture = Path(__file__).parent.parent.parent / "fixtures" / "personas" / "echo_brief"
    persona = Persona.load(fixture)
    cfg = load_config()   # no explicit provider -> anthropic default via env key
    backend = build_backend_for_persona(cfg, persona)
    assert isinstance(backend, AnthropicAPIBackend)
    assert backend._system_prompt == persona.system_prompt
    assert backend._registry is persona.registry
