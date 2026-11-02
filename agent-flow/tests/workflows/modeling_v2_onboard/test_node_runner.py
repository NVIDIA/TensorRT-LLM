# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-task worker callback."""

from __future__ import annotations

import anyio
import pytest

from agent_flow.workflows.modeling_v2_onboard.files import Layout, parse_task_result
from agent_flow.workflows.modeling_v2_onboard.node_runner import Primer, make_run_task
from agent_flow.workflows.modeling_v2_onboard.plan import parse_plan
from agent_flow.workflows.modeling_v2_onboard.prompts import PLAN_EXAMPLE

from ._fakes import FakeFactory, result_path_in, write_result

_PLAN = parse_plan(PLAN_EXAMPLE)


def _run(tmp_path, factory, node_id="join-core", *, slurm=False, side=False, primer=None):
    layout = Layout(tmp_path / "ws")
    run_task = make_run_task(layout=layout, agent_factory=factory, slurm=slurm)
    node = _PLAN.graph.by_id[node_id]
    is_final = node_id == _PLAN.sink_id()
    result = anyio.run(run_task, node, tmp_path / "wt", _PLAN.tags[node_id], is_final, side, primer)
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


def test_a_side_task_is_told_its_branch_is_not_merged(tmp_path):
    async def worker(agent, prompt):
        write_result(result_path_in(prompt), "DONE")

    factory = FakeFactory(worker=worker)
    _run(tmp_path, factory, node_id="perf-moe-gemm", side=True)
    _run(tmp_path / "main", factory, node_id="perf-fuse-add-norm")
    side, main = factory.agents
    assert "**side task**" in side.prompts[0]
    assert "side task" not in main.prompts[0]


def test_a_primed_worker_is_a_fork_told_where_its_reading_came_from(tmp_path):
    async def worker(agent, prompt):
        write_result(result_path_in(prompt), "DONE")

    factory = FakeFactory(worker=worker)
    primer = Primer(session_id="s-1", repo=tmp_path / "repo", commit="abc123")
    _, result = _run(tmp_path, factory, primer=primer)
    assert result.status == "DONE"
    (agent,) = factory.agents
    assert agent.fork_from == "s-1"
    assert agent.prompts[0].startswith("**You are now a worker, not the briefing session.**")
    assert "git diff abc123 -- <file>" in agent.prompts[0]


def test_a_fork_that_cannot_start_falls_back_to_a_fresh_worker(tmp_path):
    knowledge = Layout(tmp_path / "ws").knowledge_path("op")
    knowledge.parent.mkdir(parents=True)
    knowledge.write_text("summary\n")

    async def worker(agent, prompt):
        if agent.fork_from is not None:
            raise RuntimeError("No conversation found with session ID: s-1")
        write_result(result_path_in(prompt), "DONE")

    factory = FakeFactory(worker=worker)
    primer = Primer(session_id="s-1", repo=tmp_path / "repo", commit="abc123")
    _, result = _run(tmp_path, factory, primer=primer)
    assert result.status == "DONE"
    forked, fresh = factory.agents
    assert forked.fork_from == "s-1" and forked.closed
    assert fresh.fork_from is None
    assert "You are now a worker" not in fresh.prompts[0]
    assert f"read it first: `{knowledge}`" in fresh.prompts[0]
