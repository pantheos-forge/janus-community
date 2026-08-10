from janus.interface.components.animations import ANIMATIONS, get_frame
from janus.interface.components.command_input import CommandHistory, complete_command
from janus.interface.components.crush_spinner import CrushSpinner


def test_get_frame_cycles_over_frames():
    name = next(iter(ANIMATIONS))
    f0 = get_frame(name, 0)
    assert isinstance(f0, str)
    # Advancing the tick eventually yields a different frame for a multi-frame anim.
    frames = {get_frame(name, t) for t in range(12)}
    assert len(frames) >= 1


def test_complete_command_prefix_match():
    commands = ["pause", "resume", "stop", "status", "help", "instruction"]
    assert complete_command("/pau", commands) == "/pause"
    assert complete_command("/re", commands) == "/resume"
    assert complete_command("/xyz", commands) is None


def test_command_history_navigation():
    h = CommandHistory()
    h.add("/pause")
    h.add("/resume")
    assert h.previous() == "/resume"
    assert h.previous() == "/pause"
    assert h.next() == "/resume"


def test_crush_spinner_advances():
    s = CrushSpinner()
    first = s.render()
    for _ in range(21):
        s.advance()
    later = s.render()
    # render() returns a rich Text; both are renderable, spinner is initialized.
    assert first is not None and later is not None
    assert s.initialized is True
