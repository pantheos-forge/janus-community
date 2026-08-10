import abc

import pytest
from janus.core.backend import AgentBackend, AgentMessage, MessageType


def test_message_types():
    assert {m.name for m in MessageType} == {
        "TEXT", "TOOL_START", "TOOL_RESULT", "OUTPUT", "RESULT", "ERROR", "AWAITING_INPUT",
    }


def test_agentmessage_defaults():
    msg = AgentMessage(type=MessageType.TEXT, content="hi")
    assert msg.tool_name is None
    assert msg.tool_args is None
    assert msg.metadata == {}


def test_backend_is_abstract():
    assert issubclass(AgentBackend, abc.ABC)
    with pytest.raises(TypeError):
        AgentBackend()  # abstract, cannot instantiate


def test_concrete_subclass_can_be_built():
    class Dummy(AgentBackend):
        async def connect(self): ...
        async def disconnect(self): ...
        async def query(self, prompt): ...
        async def receive_messages(self):
            yield AgentMessage(MessageType.TEXT, "ok")
        @property
        def session_id(self): return "s1"
        async def resume(self, session_id): return False

    d = Dummy()
    assert d.session_id == "s1"
    assert d.supports_resume is False
