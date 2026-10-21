# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The composed role prompts are snapshotted into the workspace at every launch.

Same contract as the perf workflows (`agent_flow.dump_prompt_bundle`). What
is specific here is *when*: a role prompt carries the launch environment
that preflight resolves, so the snapshot is taken after preflight has composed
the bundle, and a dry run already leaves one behind.
"""

from __future__ import annotations

import dataclasses

from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import DEFAULT_PROMPTS, PROMPTS_DIRNAME

from .test_workflow import Script, _install, _workflow


def _snapshot(tmp_path) -> dict[str, str]:
    directory = tmp_path / "ws" / PROMPTS_DIRNAME
    return {path.stem: path.read_text(encoding="utf-8") for path in directory.glob("*.md")}


def test_a_dry_run_snapshots_every_role_as_composed(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path, dry_run=True)
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)

    snapshot = _snapshot(tmp_path)
    roles = {field.name for field in dataclasses.fields(workflow.prompts)}
    assert set(snapshot) == roles
    for role in roles:
        assert snapshot[role] == getattr(workflow.prompts, role), role
    # Composed, not the bare defaults: the resolved launch environment is in it.
    assert snapshot["assembler"] != DEFAULT_PROMPTS.assembler
    assert "/opt/venv/bin/python" in snapshot["assembler"]


def test_an_injected_bundle_is_the_one_snapshotted(tmp_path, task_file, monkeypatch):
    """The snapshot records what the agents are built from, whoever composed it."""
    bundle = DEFAULT_PROMPTS.with_extensions(reviewer="INJECTED FOR THIS RUN")
    workflow = _workflow(tmp_path, dry_run=True, prompts=bundle)
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)

    assert _snapshot(tmp_path)["reviewer"].endswith("INJECTED FOR THIS RUN")
