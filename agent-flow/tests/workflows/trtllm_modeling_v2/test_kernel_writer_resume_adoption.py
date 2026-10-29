# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A kernel writer's turn is dispatched, and adopted on resume, as every task's turn is."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run.execution import kernel_task as worker
from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import (
    current_decision,
    decision_dispatch_matches,
    decision_path,
    mark_decision_dispatch,
)
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage
from agent_flow.workflows.trtllm_modeling_v2.run.graph import Graph, Task

from .test_workflow import Script, _install, _workflow


def _decision(workspace, n, decision="LANDED"):
    path = decision_path(workspace, "kernel_writer", n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"decision": decision, "summary": "landed", "entry": "norm/x"}),
        encoding="utf-8",
    )


def _planned_writer(tmp_path, monkeypatch, decision="LANDED"):
    wf = _workflow(tmp_path)
    wf._turns.keep_records(wf.workspace / "turns")
    execution = ExecutionStage(wf)
    item = QueueItem(n=7, mode="onboard", target="norm/x")
    monkeypatch.setattr(execution, "_queue_item", lambda task: item)
    monkeypatch.setattr(
        worker,
        "_kernel_writer_prompt",
        lambda execution, item, note, task=None: f"`norm/x`\n{note}",
    )
    calls = _install(
        monkeypatch,
        wf,
        Script(None, kernel_writer=lambda turn, prompt: {"decision": decision, "entry": "norm/x"}),
    )
    return execution, calls


@pytest.mark.parametrize("decision", ["LANDED", "FAILED_RECEIPT"])
def test_a_relaunch_adopts_the_same_attempts_answer_but_a_rebase_dispatches_anew(
    tmp_path, monkeypatch, decision
):
    task = Task(id="kernel-x", kind="kernel", goal="Verify norm/x", attempts=3, ref="7")
    wf, calls = _planned_writer(tmp_path, monkeypatch, decision)
    Graph(tasks=[replace(task, status="running")]).save(wf.workflow.workspace / "graph.yaml")
    assert wf._work_task_body(task).decision == decision
    assert len(calls) == 1
    answer = current_decision(wf.workflow.workspace, "kernel_writer", 7)
    assert decision_dispatch_matches(answer, {"task_id": "kernel-x", "task_attempt": 3})

    resumed, resumed_calls = _planned_writer(tmp_path, monkeypatch, decision)
    assert resumed._work_task_body(task).decision == decision
    assert not resumed_calls
    assert current_decision(wf.workflow.workspace, "kernel_writer", 7) == answer
    # An explicit rebase requires a new dispatch, even for the same attempt.
    rebased, rebased_calls = _planned_writer(tmp_path, monkeypatch, decision)
    rebase = "Rebase onto the integrated dependency."
    assert rebased._work_in_tree(task, "Worktree: test", rebase).decision == decision
    assert len(rebased_calls) == 1 and rebase in rebased_calls[0][1]
    assert current_decision(wf.workflow.workspace, "kernel_writer", 7) != answer


def test_a_retry_is_told_so_and_never_handed_the_previous_attempts_decision(tmp_path, monkeypatch):
    wf, calls = _planned_writer(tmp_path, monkeypatch)
    old = mark_decision_dispatch(
        wf.workflow.workspace,
        "kernel_writer",
        7,
        identity={"task_id": "kernel-x", "task_attempt": 2},
    )
    _decision(wf.workflow.workspace, 7, "FAILED_RECEIPT")

    task = Task(id="kernel-x", kind="kernel", goal="Verify norm/x", attempts=3, ref="7")
    assert wf._work_task_body(task).decision == "LANDED"
    assert len(calls) == 1
    assert "Attempt 3 of task kernel-x" in calls[0][1]
    assert current_decision(wf.workflow.workspace, "kernel_writer", 7) != old
    assert "FAILED_RECEIPT" in old.read_text()


def test_failed_receipt_integrates_only_entry_changes(tmp_path, monkeypatch):
    execution, _ = _planned_writer(tmp_path, monkeypatch)
    item = QueueItem(n=7, mode="onboard", target="norm/x")
    task = Task(id="kernel-x", kind="kernel", goal="Certify norm/x", ref="7")
    paths = ["tensorrt_llm/_torch/_experimental/modeling_v2/catalog/norm/x.py"]
    integrated, closed = [], []
    execution._integration = SimpleNamespace(
        changed_paths=lambda task_id: paths,
        branch=lambda task_id: "task/kernel-x",
        integrate=lambda task_id: (
            integrated.append(task_id) or SimpleNamespace(merged=True, commits=1, detail="")
        ),
        close=closed.append,
    )
    assert "are integrated" in worker._land_failed_receipt(execution, item, task, "FAILED_RECEIPT")
    assert integrated == closed == [task.id]
    paths.append("tensorrt_llm/_torch/pyexecutor/executor.py")
    assert "change more" in worker._land_failed_receipt(execution, item, task, "FAILED_RECEIPT")
    assert integrated == closed == [task.id]
    assert worker._land_failed_receipt(execution, item, task, "LANDED") == ""
