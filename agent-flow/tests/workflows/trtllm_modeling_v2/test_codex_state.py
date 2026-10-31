# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A run's Codex state: its own home in the workspace, its databases on the host's local disk."""

from __future__ import annotations

import os

from agent_flow.agent_runtime import AgentConfig
from agent_flow.workflows.trtllm_modeling_v2.agents import codex_state
from agent_flow.workflows.trtllm_modeling_v2.run.startup.agent_probe import make_probe_layer


def _mountinfo(path, mounts):
    path.write_text(
        "".join(
            f"{i} 1 0:{i} / {point} rw - {kind} dev rw\n" for i, (point, kind) in enumerate(mounts)
        )
    )
    return str(path)


def test_a_run_keeps_its_codex_home_in_its_workspace_and_its_databases_on_local_disk(tmp_path):
    user = tmp_path / "user-home"
    user.mkdir()
    (user / "auth.json").write_text("login")
    (user / "config.toml").write_text("model = 'm'\n")
    (user / "state_5.sqlite").write_text("shared database")
    workspace, local = tmp_path / "net" / "ws", tmp_path / "local"
    workspace.mkdir(parents=True)
    local.mkdir()
    env = {"CODEX_HOME": str(user), "TMPDIR": str(local)}
    mounts = _mountinfo(tmp_path / "mountinfo", [("/", "nfs"), (str(local), "xfs")])

    state = codex_state.prepare(workspace, environ=env, mountinfo=mounts)
    assert state.home == workspace / "codex-home" and (state.home_fs, state.sqlite_fs) == (
        "nfs",
        "xfs",
    )
    assert state.sqlite_home.parent.parent == local and state.local
    # The login and configuration come along, private; the user's databases never do.
    assert sorted(p.name for p in state.home.iterdir()) == ["auth.json", "config.toml"]
    assert oct((state.home / "auth.json").stat().st_mode & 0o777) == "0o600"
    # A newer login replaces the copy; the source is never written.
    (user / "auth.json").write_text("new login")
    os.utime(user / "auth.json", (2e9, 2e9))
    codex_state.prepare(workspace, environ=env, mountinfo=mounts)
    assert (state.home / "auth.json").read_text() == "new login"
    assert (user / "state_5.sqlite").read_text() == "shared database"

    # With no local disk, the databases stay in the run's own home: never the user's.
    shared = codex_state.prepare(
        workspace, environ=env, mountinfo=_mountinfo(tmp_path / "m2", [("/", "lustre")])
    )
    assert shared.sqlite_home == shared.home and not shared.local

    # The agent probe runs a Codex agent on the same state, a Claude Code one on none.
    backend = make_probe_layer(AgentConfig("codex", "m"), workspace).config.backend
    assert backend.codex_home == state.home and backend.codex_sqlite_home is not None
    backend = make_probe_layer(AgentConfig("claude-code", "m"), workspace).config.backend
    assert backend.codex_home is None and backend.codex_sqlite_home is None
