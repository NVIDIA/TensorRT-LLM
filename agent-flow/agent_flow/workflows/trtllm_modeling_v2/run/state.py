# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run's state: ``<workspace>/.trtllm_modeling_v2_state.json``.

What preflight resolved (the slug, the branch, the claimed devices) and the onboard
record, written once and never taken back. Where the run stands is in the files the run
keeps anyway: the task list (``tasks.yaml``), the verdicts (``verdicts.jsonl``) and
the decision files; the same command against the same workspace goes on from them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..infra.files import read_json, write_text_atomic

STATE_FILENAME = ".trtllm_modeling_v2_state.json"
SCHEMA_VERSION = 4


@dataclass
class WorkflowState:
    task_path: str
    slug: str = ""
    branch: str = ""
    devices: list[int] = field(default_factory=list)
    #: Set when core's boot and benchmarks passed on the commit that then landed:
    #: ``commit``, ``ref``, ``libraries`` (their content hash), ``time`` and ``accuracy``
    #: (``judged`` or ``unjudged (boot only)``). Empty until then, and never changed after.
    onboarded: dict[str, str] = field(default_factory=dict)


class StateError(ValueError):
    """The workspace holds a run state this version does not read."""


def load_state(path: Path) -> WorkflowState:
    data = read_json(path)
    if not isinstance(data, dict) or data.get("version") != SCHEMA_VERSION:
        version = data.get("version") if isinstance(data, dict) else None
        raise StateError(
            f"{path} is not this version's run state (version {version!r}, expected "
            f"{SCHEMA_VERSION}): an earlier version of the flow ran in this workspace; start "
            f"over with --clean, or take another --workspace"
        )
    known = {k: v for k, v in data.items() if k in WorkflowState.__dataclass_fields__}
    return WorkflowState(**known)


def save_state(path: Path, state: WorkflowState) -> None:
    write_text_atomic(path, json.dumps({"version": SCHEMA_VERSION, **asdict(state)}, indent=2))
