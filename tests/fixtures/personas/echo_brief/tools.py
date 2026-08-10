from janus.core.tools.registry import ToolContext, tool


@tool("echo_note", "Echo a note", {"type": "object", "properties": {"text": {"type": "string"}}})
def echo_note(ctx: ToolContext, text=""):
    return f"note:{text}"


TOOLS = [echo_note]
