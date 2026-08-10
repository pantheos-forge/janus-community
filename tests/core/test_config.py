# tests/core/test_config.py
from janus.core.config import JanusConfig, load_config


def test_defaults():
    cfg = load_config()
    assert cfg.persona is None
    assert cfg.permission_mode == "bypassPermissions"
    assert cfg.llm_model.startswith("claude-")
    assert len(cfg.session_id) == 8


def test_overrides_win():
    cfg = load_config(persona="market_research", local_model="qwen:latest")
    assert cfg.persona == "market_research"
    assert cfg.local_model == "qwen:latest"


def test_no_pentest_fields():
    fields = set(JanusConfig.model_fields)
    for banned in ("target", "prompt_type", "aggressive", "no_research", "s3_bucket"):
        assert banned not in fields


def test_aliased_fields_settable_by_python_name():
    # populate_by_name=True: both the env-var alias AND the Python field name must work as kwargs.
    cfg = load_config(openrouter_api_key="sk-or-xyz", anthropic_api_key="sk-ant-xyz")
    assert cfg.openrouter_api_key == "sk-or-xyz"
    assert cfg.anthropic_api_key == "sk-ant-xyz"


def test_empty_string_bool_coerces_to_default():
    # Defense in depth for the compose env bug: a container that receives
    # USE_CLAUDE_AGENT_SDK="" (unset var expanded to empty) must not crash
    # pydantic's bool parser — "" coerces back to the False default.
    cfg = load_config(use_claude_agent_sdk="")
    assert cfg.use_claude_agent_sdk is False


def test_empty_string_bool_from_env_coerces_to_default(monkeypatch):
    # The real failure path: the env var is present but empty.
    monkeypatch.setenv("USE_CLAUDE_AGENT_SDK", "")
    cfg = load_config()
    assert cfg.use_claude_agent_sdk is False


def test_real_bool_string_still_parses():
    assert load_config(use_claude_agent_sdk="true").use_claude_agent_sdk is True
    assert load_config(use_claude_agent_sdk="false").use_claude_agent_sdk is False
