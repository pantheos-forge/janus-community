import asyncio
import json
from pathlib import Path

import pytest
from janus.core.backend import AgentBackend, AgentMessage, MessageType
from janus.core.config import load_config
from janus.core.controller import AgentController
from janus.core.events import EventType
from janus.core.persona import Persona
from janus.core.session import SessionStore
from janus.core.backends.generic import GenericBackend

FIXTURE = Path(__file__).parent.parent / "fixtures" / "personas" / "echo_brief"


class _PersonaScriptedBackend(GenericBackend):
    """A GenericBackend whose _chat_completion scripts a call to emit_output then finishes."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._calls = 0

    async def _chat_completion(self):
        self._calls += 1
        if self._calls == 1:
            return {"message": {"content": "here is the brief", "tool_calls": [
                {"id": "c1", "function": {"name": "emit_output",
                                          "arguments": json.dumps({"summary": "EVs are growing"})}}]}}
        return {"message": {"content": "done"}}

    def _tool_result_message(self, tool_call, result):
        return {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result}


@pytest.mark.asyncio
async def test_persona_run_writes_deliverable_and_emits_output(tmp_path):
    p = Persona.load(FIXTURE)
    wd = p.prepare_workspace(tmp_path / "wd")
    backend = _PersonaScriptedBackend(working_directory=wd, system_prompt=p.system_prompt,
                                      model=p.provider_model or "m", registry=p.registry)
    ctrl = AgentController(load_config(persona=p.name), backend=backend,
                           session_store=SessionStore(sessions_dir=tmp_path / "sessions"))
    outputs = []
    ctrl.events.subscribe(EventType.OUTPUT, lambda e: outputs.append(e.data))

    result = await ctrl.run(p.build_task("EV charging"))

    assert result["status"] == "completed"
    # the schema-valid deliverable was written to the workspace
    deliverable = json.loads((wd / "output.json").read_text())
    assert deliverable == {"summary": "EVs are growing"}
    # and the live OUTPUT event fired with the same payload
    assert outputs == [{"summary": "EVs are growing"}]
