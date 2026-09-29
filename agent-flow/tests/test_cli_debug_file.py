"""Opt-in `--debug-file` for the bundled CLI.

Six campaigns failed with a synthetic assistant message reading `API Error: The
operation timed out.`, `error: "unknown"`, and every usage counter zero. That
record is identical whether the CLI's own idle watchdog gave up or the server
closed the connection -- and those have opposite fixes: one is a client setting
we control, the other is not. Every hypothesis tried against the transcripts
alone (a request-total timeout, a provider-specific watchdog gate, a broken
environment hand-off, a context-size limit) was falsified, because the evidence
that separates them is only in the CLI's own log, which it does not write unless
asked.

These tests pin the asking: off unless a directory is named, one file per
client, and never a reason the campaign fails to start.
"""

from __future__ import annotations

from pathlib import Path

from agent_flow.backends.claude_code import CLI_DEBUG_DIR_ENV, _cli_debug_args


def test_off_unless_a_directory_is_named(monkeypatch, tmp_path):
    """Default off. 16 KB per short request is tens of MB over a campaign."""
    monkeypatch.delenv(CLI_DEBUG_DIR_ENV, raising=False)

    assert _cli_debug_args(tmp_path) == {}


def test_blank_is_off_too(monkeypatch, tmp_path):
    """An operator clearing the value means off, not a file called ""."""
    monkeypatch.setenv(CLI_DEBUG_DIR_ENV, "   ")

    assert _cli_debug_args(tmp_path) == {}


def test_it_names_a_file_under_the_directory(monkeypatch, tmp_path):
    monkeypatch.setenv(CLI_DEBUG_DIR_ENV, str(tmp_path / "logs"))

    args = _cli_debug_args(Path("/runs/workspace-20260930-gemma"))

    path = Path(args["debug-file"])
    assert path.parent == tmp_path / "logs"
    assert path.parent.is_dir(), "the directory is created, not merely named"


def test_the_name_carries_the_workspace_so_a_file_traces_to_its_campaign(monkeypatch, tmp_path):
    """Otherwise a shared directory is a pile of files nobody can attribute.

    `flow_env` is service-wide, so every concurrent campaign writes here.
    """
    monkeypatch.setenv(CLI_DEBUG_DIR_ENV, str(tmp_path))

    args = _cli_debug_args(Path("/runs/workspace-20260930-gemma"))

    assert "workspace-20260930-gemma" in Path(args["debug-file"]).name


def test_each_client_gets_its_own_file(monkeypatch, tmp_path):
    """Parallel roles share ONE `perf-optimize` process.

    Up to three optimizer/evaluator pairs run concurrently, so a path fixed per
    process would interleave several agents into one unreadable file -- and the
    stall being investigated is a property of a single stream.
    """
    monkeypatch.setenv(CLI_DEBUG_DIR_ENV, str(tmp_path))
    same_workspace = Path("/runs/workspace-20260930-gemma")

    first = _cli_debug_args(same_workspace)["debug-file"]
    second = _cli_debug_args(same_workspace)["debug-file"]

    assert first != second


def test_an_unusable_directory_does_not_fail_the_campaign(monkeypatch, tmp_path):
    """Diagnostics are never worth losing a run over.

    A path under a file rather than a directory is the plausible typo, and it
    raises from `mkdir`. The campaign proceeds without the log.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    monkeypatch.setenv(CLI_DEBUG_DIR_ENV, str(blocker / "logs"))

    assert _cli_debug_args(tmp_path) == {}


def test_a_missing_cwd_still_produces_a_name(monkeypatch, tmp_path):
    """`cwd` is optional on `create_client`, so it can be None here."""
    monkeypatch.setenv(CLI_DEBUG_DIR_ENV, str(tmp_path))

    args = _cli_debug_args(None)

    assert args["debug-file"].endswith(".log")
