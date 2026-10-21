# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A resumed smith phase collects answers the dead process never read.

A process can die with many smith turns in flight and some of their LANDED
decisions already on disk, while the state file still has every item
`running`. Adopting those decisions keeps a restart from re-dispatching work
that already landed.
"""

from __future__ import annotations

import yaml

from agent_flow.workflows.trtllm_modeling_v2.onboard.vocabulary import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run.progress import (
    decision_path,
    mark_decision_dispatch,
)

from .test_workflow import Script, _install, _workflow


def _decision(workspace, n, decision="LANDED"):
    path = decision_path(workspace, "smith", n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"decision": decision, "summary": "landed", "entry": "norm/x"}),
        encoding="utf-8",
    )


def test_a_decision_written_after_the_last_dispatch_is_adopted(tmp_path, monkeypatch):
    wf = _workflow(tmp_path)
    calls = _install(monkeypatch, wf, Script(None))
    wf._decision_files = True
    mark_decision_dispatch(wf.workspace, "smith", 7)  # the dead process's dispatch
    _decision(wf.workspace, 7)  # answered after it

    result = wf._run_one_smith(QueueItem(n=7, mode="onboard", target="norm/x"))

    assert result.startswith("adopted LANDED")
    assert not [c for c in calls if c[0] == "smith"]
    assert wf._smith_decision_for(QueueItem(n=7, mode="onboard", target="norm/x")) == "LANDED"


def test_a_retry_never_adopts_the_failure_it_is_retrying(tmp_path, monkeypatch):
    wf = _workflow(tmp_path)
    calls = _install(monkeypatch, wf, Script(None))
    wf._decision_files = True
    monkeypatch.setattr(
        type(wf), "_smith_prompt", lambda self, item, note="", task=None: "Dispatch 7"
    )
    mark_decision_dispatch(wf.workspace, "smith", 7)
    _decision(wf.workspace, 7, decision="FAILED")

    wf._run_one_smith(QueueItem(n=7, mode="onboard", target="norm/x"), note="retry")

    assert [c for c in calls if c[0] == "smith"]


def test_no_decision_on_disk_dispatches_the_smith(tmp_path, monkeypatch):
    wf = _workflow(tmp_path)
    calls = _install(monkeypatch, wf, Script(None))
    wf._decision_files = True
    monkeypatch.setattr(
        type(wf), "_smith_prompt", lambda self, item, note="", task=None: "Dispatch 3"
    )

    wf._run_one_smith(QueueItem(n=3, mode="onboard", target="norm/y"))

    assert [c for c in calls if c[0] == "smith"]
