# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The staged agent home, which is what makes the orchestrator able to spawn agents.

The failure this module exists for produced a turn with zero tokens and
``stop_reason='stop_sequence'``, i.e. something indistinguishable from a model
refusal, when in fact no agent had started. These tests are about the parts
that decide whether that happens, not about the copying.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.agent_home import (
    CLAUDE_CONFIG_DIRNAME,
    CREDENTIAL_SUBPATH,
    ensure_agent_home,
    home_is_writable,
    refresh_credentials,
    stage_agent_home,
)


@pytest.fixture
def fake_home(tmp_path: Path) -> Path:
    """A writable stand-in for ``$HOME`` with a credential store in it."""
    home = tmp_path / "realhome"
    creds = home / CREDENTIAL_SUBPATH
    creds.mkdir(parents=True)
    (creds / "active_config").write_text("default\n", encoding="utf-8")
    (creds / "credentials").mkdir()
    (creds / "credentials" / "default.json").write_text('{"token": "one"}', encoding="utf-8")
    return home


def test_a_writable_home_stages_nothing(tmp_path, fake_home):
    """An ordinary workstation must be left completely alone."""
    assert home_is_writable(fake_home)
    assert ensure_agent_home(tmp_path / "ws", home=fake_home) is None


def test_the_write_probe_removes_itself(fake_home):
    """A probe that leaves a directory behind is a probe that pollutes the store."""
    assert home_is_writable(fake_home)
    assert not (fake_home / CREDENTIAL_SUBPATH / ".agent_home_write_probe").exists()


def test_an_unwritable_home_is_detected_without_relying_on_permission_bits(tmp_path, fake_home):
    """A sandbox is not a permission bit, so the probe must actually write.

    Simulated here by pointing at a path whose parent is a file: ``mkdir``
    fails with ``ENOTDIR`` exactly as it fails with ``EROFS``, and
    ``os.access`` on the nonexistent path would tell us nothing either way.
    """
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    assert not home_is_writable(blocked)


def test_staging_selects_the_new_home_through_all_three_variables(tmp_path, fake_home):
    """HOME alone is not enough: the credential store resolves through XDG.

    Setting only ``HOME`` leaves the child reading the real, read-only store
    and the failure is byte-identical to having done nothing at all.
    """
    staged = stage_agent_home(tmp_path / "agent-home", home=fake_home)
    assert staged.env["HOME"] == str(staged.root)
    assert staged.env["XDG_CONFIG_HOME"] == str(staged.root / ".config")
    assert staged.env["CLAUDE_CONFIG_DIR"] == str(staged.root / CLAUDE_CONFIG_DIRNAME)


def test_the_claude_config_dir_is_not_named_dot_claude(tmp_path, fake_home):
    """A sandbox that denies `**/.claude` writes would silently cripple the child.

    The symptom is a warning, not an error, so a run can lose its config
    directory and still appear to work.
    """
    staged = stage_agent_home(tmp_path / "agent-home", home=fake_home)
    assert ".claude" not in Path(staged.env["CLAUDE_CONFIG_DIR"]).name
    assert (staged.root / CLAUDE_CONFIG_DIRNAME / ".claude.json").is_file()


def test_credentials_are_copied_not_referenced(tmp_path, fake_home):
    staged = stage_agent_home(tmp_path / "agent-home", home=fake_home)
    copied = staged.root / CREDENTIAL_SUBPATH / "credentials" / "default.json"
    assert copied.read_text(encoding="utf-8") == '{"token": "one"}'
    assert not copied.is_symlink()


def test_refresh_replaces_a_stale_copy(tmp_path, fake_home):
    """A stale copy fails as an auth error minutes into a turn, which is expensive.

    Replacement rather than merge, because a rotated store can have FEWER
    files than the one it replaces and a merge would leave the old ones
    readable.
    """
    staged = stage_agent_home(tmp_path / "agent-home", home=fake_home)
    (fake_home / CREDENTIAL_SUBPATH / "credentials" / "default.json").write_text(
        '{"token": "two"}', encoding="utf-8"
    )
    (staged.root / CREDENTIAL_SUBPATH / "credentials" / "stale.json").write_text("x", "utf-8")

    assert refresh_credentials(staged.root, home=fake_home)
    creds = staged.root / CREDENTIAL_SUBPATH / "credentials"
    assert (creds / "default.json").read_text(encoding="utf-8") == '{"token": "two"}'
    assert not (creds / "stale.json").exists()


def test_refresh_reports_failure_rather_than_raising(tmp_path):
    """There is nothing to copy from; the caller decides what that means."""
    assert refresh_credentials(tmp_path / "agent-home", home=tmp_path / "nowhere") is False


def test_git_identity_is_written_because_redirecting_home_removes_the_real_one(tmp_path, fake_home):
    """Otherwise every smith commit dies with "Please tell me who you are".

    That is a new infrastructure failure traded for the one just fixed, and
    it surfaces at the first commit rather than at startup.
    """
    staged = stage_agent_home(
        tmp_path / "agent-home", home=fake_home, git_identity=("A Name", "a@example.com")
    )
    text = (staged.root / ".gitconfig").read_text(encoding="utf-8")
    assert "name = A Name" in text
    assert "email = a@example.com" in text


def test_apply_exports_into_the_process_so_every_backend_inherits_it(
    tmp_path, fake_home, monkeypatch
):
    """The alternative is per-backend plumbing, and Codex would need its own.

    Claude Code takes ``ClaudeAgentOptions.env``; Codex does not expose one.
    Exporting once is what makes this backend-agnostic.
    """
    monkeypatch.setenv("HOME", str(fake_home))
    staged = stage_agent_home(tmp_path / "agent-home", home=fake_home)
    staged.apply()
    assert os.environ["HOME"] == str(staged.root)
    assert os.environ["XDG_CONFIG_HOME"] == str(staged.root / ".config")


def test_staging_is_idempotent(tmp_path, fake_home):
    """A resume re-enters the same code path and must not destroy state."""
    root = tmp_path / "agent-home"
    stage_agent_home(root, home=fake_home)
    (root / CLAUDE_CONFIG_DIRNAME / ".claude.json").write_text('{"k": 1}', encoding="utf-8")
    stage_agent_home(root, home=fake_home)
    assert (root / CLAUDE_CONFIG_DIRNAME / ".claude.json").read_text(encoding="utf-8") == '{"k": 1}'
