# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A resumed kernel writer phase collects answers the dead process never read."""

from __future__ import annotations

import yaml

from agent_flow.workflows.trtllm_modeling_v2.onboard.vocabulary import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run.progress import (
    decision_path,
    mark_decision_dispatch,
)

from .test_workflow import Script, _install, _workflow


def _decision(workspace, n, decision="LANDED"):
    path = decision_path(workspace, "kernel_writer", n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"decision": decision, "summary": "landed", "entry": "norm/x"}),
        encoding="utf-8",
    )


def test_a_decision_written_after_the_last_dispatch_is_adopted(tmp_path, monkeypatch):
    wf = _workflow(tmp_path)
    calls = _install(monkeypatch, wf, Script(None))
    mark_decision_dispatch(wf.workspace, "kernel_writer", 7)  # the dead process's dispatch
    _decision(wf.workspace, 7)  # answered after it

    result = wf._run_one_kernel_writer(QueueItem(n=7, mode="onboard", target="norm/x"))

    assert result.startswith("adopted LANDED")
    assert not [c for c in calls if c[0] == "kernel_writer"]
    assert (
        wf._kernel_writer_decision_for(QueueItem(n=7, mode="onboard", target="norm/x")) == "LANDED"
    )
