# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A kernel writer's turn is dispatched, and adopted on resume, as every task's turn is."""

from __future__ import annotations

from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run.execution import kernel_task as worker
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage
from agent_flow.workflows.trtllm_modeling_v2.run.graph import Task

from .test_workflow import Script, _install, _workflow


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
