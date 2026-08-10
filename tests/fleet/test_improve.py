import subprocess

from janus.fleet.improve import sync_and_commit
from tests.personas.factory_samples import GOOD_MANIFEST, GOOD_PROMPT, GOOD_RUBRIC, GOOD_SCHEMA


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _make_agent_repo(tmp_path):
    """A fleet agent dir that is a git repo with an initial persona/."""
    agent = tmp_path / "agent"
    persona = agent / "persona"
    persona.mkdir(parents=True)
    for fn, content in (("manifest.toml", GOOD_MANIFEST), ("prompt.md", GOOD_PROMPT),
                        ("output_schema.json", GOOD_SCHEMA), ("rubric.toml", GOOD_RUBRIC)):
        (persona / fn).write_text(content)
    _git(["init", "-q"], agent)
    _git(["add", "-A"], agent)
    _git(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], agent)
    return agent


def test_sync_and_commit_updates_persona_and_commits(tmp_path):
    agent = _make_agent_repo(tmp_path)
    # a build dir with a tightened prompt
    build = tmp_path / "build"
    build.mkdir()
    for fn, content in (("manifest.toml", GOOD_MANIFEST), ("prompt.md", GOOD_PROMPT + "\nCite everything.\n"),
                        ("output_schema.json", GOOD_SCHEMA), ("rubric.toml", GOOD_RUBRIC)):
        (build / fn).write_text(content)

    sha = sync_and_commit(build, agent, summary="require citations")

    # persona/ updated in the repo
    assert "Cite everything." in (agent / "persona" / "prompt.md").read_text()
    # a new commit exists with the improve message
    log = _git(["log", "--oneline", "-1"], agent).stdout
    assert "improve: require citations" in log
    assert sha and sha in log


def test_sync_and_commit_git_inits_a_non_repo(tmp_path):
    # agent dir with persona files but NO .git (e.g. adopted from a zip)
    agent = tmp_path / "agent"
    (agent / "persona").mkdir(parents=True)
    for fn, content in (("manifest.toml", GOOD_MANIFEST), ("prompt.md", GOOD_PROMPT),
                        ("output_schema.json", GOOD_SCHEMA), ("rubric.toml", GOOD_RUBRIC)):
        (agent / "persona" / fn).write_text(content)

    build = tmp_path / "build"
    build.mkdir()
    for fn, content in (("manifest.toml", GOOD_MANIFEST), ("prompt.md", GOOD_PROMPT + "\nx\n"),
                        ("output_schema.json", GOOD_SCHEMA), ("rubric.toml", GOOD_RUBRIC)):
        (build / fn).write_text(content)

    sha = sync_and_commit(build, agent, summary="first improvement")
    assert (agent / ".git").exists()
    assert sha


def _write_four(d, prompt=GOOD_PROMPT):
    for fn, content in (("manifest.toml", GOOD_MANIFEST), ("prompt.md", prompt),
                        ("output_schema.json", GOOD_SCHEMA), ("rubric.toml", GOOD_RUBRIC)):
        (d / fn).write_text(content)


def test_sync_and_commit_carries_banner_when_present(tmp_path):
    """A banner produced during the improve run must reach the fleet repo —
    otherwise set_persona_banner(write=True) is silently dropped on export."""
    agent = _make_agent_repo(tmp_path)
    build = tmp_path / "build"
    build.mkdir()
    _write_four(build)
    (build / "banner.txt").write_text("⣿⣿⣿\n⣿⣿⣿\n")
    (build / "banner_source.txt").write_text("File:X.png · Public domain · Wikimedia Commons")

    sync_and_commit(build, agent, summary="add banner")

    persona = agent / "persona"
    assert (persona / "banner.txt").read_text() == "⣿⣿⣿\n⣿⣿⣿\n"
    assert "Public domain" in (persona / "banner_source.txt").read_text()
    # both banner files are committed, not just left in the worktree
    tracked = _git(["ls-files", "persona"], agent).stdout
    assert "persona/banner.txt" in tracked
    assert "persona/banner_source.txt" in tracked


def test_sync_and_commit_without_banner_preserves_existing(tmp_path):
    """An improve run that doesn't regenerate art must not crash on the
    missing files, nor delete a banner the fleet agent already had."""
    agent = _make_agent_repo(tmp_path)
    persona = agent / "persona"
    (persona / "banner.txt").write_text("⠿⠿⠿\n")
    _git(["add", "-A"], agent)
    _git(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "banner"], agent)

    build = tmp_path / "build"
    build.mkdir()
    _write_four(build, prompt=GOOD_PROMPT + "\nz\n")  # no banner files in the build dir

    sync_and_commit(build, agent, summary="tighten, no art")  # must not raise

    assert (persona / "banner.txt").read_text() == "⠿⠿⠿\n"  # untouched


def test_sync_and_commit_preserves_prior_history(tmp_path):
    agent = _make_agent_repo(tmp_path)
    build = tmp_path / "build"
    build.mkdir()
    for fn, content in (("manifest.toml", GOOD_MANIFEST), ("prompt.md", GOOD_PROMPT + "\ny\n"),
                        ("output_schema.json", GOOD_SCHEMA), ("rubric.toml", GOOD_RUBRIC)):
        (build / fn).write_text(content)
    sync_and_commit(build, agent, summary="second")
    count = _git(["rev-list", "--count", "HEAD"], agent).stdout.strip()
    assert count == "2"  # original init + the improvement, history intact


def test_sync_and_commit_carries_container_toml(tmp_path):
    agent = _make_agent_repo(tmp_path)
    build = tmp_path / "build"
    build.mkdir()
    _write_four(build)
    (build / "container.toml").write_text('[install]\napt = ["ripgrep"]\n')

    sync_and_commit(build, agent, summary="add tools")

    persona = agent / "persona"
    assert (persona / "container.toml").read_text().startswith("[install]")
    assert "persona/container.toml" in _git(["ls-files", "persona"], agent).stdout
