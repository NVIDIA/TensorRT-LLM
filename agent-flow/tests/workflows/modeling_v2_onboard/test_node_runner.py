# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-task worker callback."""

from __future__ import annotations

import anyio
import pytest

from agent_flow.workflows.modeling_v2_onboard.files import Layout, parse_task_result
from agent_flow.workflows.modeling_v2_onboard.node_runner import make_run_task
from agent_flow.workflows.modeling_v2_onboard.plan import parse_plan
from agent_flow.workflows.modeling_v2_onboard.prompts import PLAN_EXAMPLE

from ._fakes import FakeFactory, result_path_in, write_result

_PLAN = parse_plan(PLAN_EXAMPLE)


def _run(tmp_path, factory, node_id="join-core", *, slurm=False):
    layout = Layout(tmp_path / "ws")
    run_task = make_run_task(layout=layout, agent_factory=factory, slurm=slurm)
    node = _PLAN.graph.by_id[node_id]
    is_final = node_id == _PLAN.sink_id()
    result = anyio.run(run_task, node, tmp_path / "wt", _PLAN.tags[node_id], is_final)
    return layout, result


@pytest.mark.parametrize("status", ["DONE", "FAILED"])
def test_status_in_the_result_file_is_the_outcome(tmp_path, status):
    async def worker(agent, prompt):
        write_result(result_path_in(prompt), status, "did it")

    factory = FakeFactory(worker=worker)
    layout, result = _run(tmp_path, factory)
    assert (result.status, result.summary) == (status, "did it")

    (agent,) = factory.agents
    assert agent.closed and agent.session_mode == "persistent"
    assert agent.cwd == tmp_path / "wt"
    assert "## Tag `merge`" in agent.system_prompt and "## Tag `target`" in agent.system_prompt
    # Upstream results are named for the worker to read.
    for dep in _PLAN.graph.by_id["join-core"].depends_on:
        assert str(layout.result_path(dep)) in agent.prompts[0]


def test_a_missing_result_gets_one_follow_up_turn(tmp_path):
    async def worker(agent, prompt):
        if len(agent.prompts) == 2:
            assert "is not valid" in prompt
            write_result(result_path_in(prompt), "DONE")

    factory = FakeFactory(worker=worker)
    _, result = _run(tmp_path, factory)
    assert result.status == "DONE"
    assert len(factory.agents[0].prompts) == 2


def test_no_result_after_the_follow_up_fails_the_task_with_a_written_reason(tmp_path):
    async def worker(agent, prompt):
        result_path_in(prompt).write_text("no front matter\n")

    layout, result = _run(tmp_path, FakeFactory(worker=worker))
    assert result.status == "FAILED"
    text = layout.result_path("join-core").read_text()
    assert parse_task_result(text).status == "FAILED"
    assert "must start with" in text


def test_a_crashing_session_fails_only_the_task(tmp_path):
    async def worker(agent, prompt):
        raise RuntimeError("backend died")

    factory = FakeFactory(worker=worker)
    layout, result = _run(tmp_path, factory)
    assert result.status == "FAILED"
    assert "backend died" in layout.result_path("join-core").read_text()
    assert factory.agents[0].closed


def test_a_stale_result_from_an_interrupted_turn_is_not_reused(tmp_path):
    layout = Layout(tmp_path / "ws")
    layout.results_dir.mkdir(parents=True)
    write_result(layout.result_path("join-core"), "DONE")

    async def worker(agent, prompt):
        return None

    _, result = _run(tmp_path, FakeFactory(worker=worker))
    assert result.status == "FAILED"


def test_final_task_and_slurm_are_announced(tmp_path):
    async def worker(agent, prompt):
        write_result(result_path_in(prompt), "DONE")

    factory = FakeFactory(worker=worker)
    layout, _ = _run(tmp_path, factory, node_id="land", slurm=True)
    (agent,) = factory.agents
    assert "final task" in agent.prompts[0]
    assert str(layout.test_commands_path) in agent.prompts[0]
    assert "## Running GPU work on Slurm" in agent.system_prompt
