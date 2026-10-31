# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The keep rule: a gain against the tree's own spread, never at a member it does not change."""

from __future__ import annotations

import json
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.tune.inert import inert_members
from agent_flow.workflows.trtllm_modeling_v2.tune.records import (
    PARTS_FILE,
    Measurement,
    NoiseFloor,
    compare,
)

RULE = {"min_gain": 0.02, "noise_factor": 3.0}
#: Two members: one rank, and two ranks (where a collective change runs).
MEMBERS = {"tp1": {"tensor_parallel_size": 1}, "tp2": {"tensor_parallel_size": 2}}
CONS = (1, 16)
MARKER = "partial sums reduce through a workspace"


def _bind(slots: int, ranks: int) -> str:
    return f"modeling_v2 x: bound Pool #2: {slots} recurrent slots, 32 states per slot\n" * ranks


def _write(root: Path, label: str, scale: dict, logs: dict, host: str = "h1") -> Measurement:
    """A measurement over :data:`MEMBERS`: ``scale[site]`` multiplies both metrics at that site."""
    for name, config in MEMBERS.items():
        part = root / label / name
        part.mkdir(parents=True, exist_ok=True)
        rows = []
        for con in CONS:
            g = scale.get(f"{name}.con{con}", 1.0)
            rows.append({"concurrency": con, "tok_s_user": 100.0 * g, "output_tok_s_gpu": 1000.0 * g,
                         "mean_tpot_ms": 10.0, "mean_ttft_ms": 100.0})  # fmt: skip
        (part / "summary.json").write_text(json.dumps(rows))
        meta = {"label": name, "config": {"modeling_v2": "require", **config}, "hostname": host}
        (part / "meta.json").write_text(json.dumps(meta))
        (part / "server.log").write_text(logs.get(name, ""))
    (root / label / PARTS_FILE).write_text(json.dumps({"parts": list(MEMBERS), "constraints": {}}))
    (root / label / "meta.json").write_text(json.dumps({"label": label, "hostname": host}))
    (root / label / "summary.json").write_text("[]")
    return Measurement.load(root / label)


def test_a_member_without_the_marker_and_with_the_controls_pools_is_inert(tmp_path):
    ctrl = _write(tmp_path, "a", {}, {"tp1": _bind(1334, 1), "tp2": _bind(3461, 2)})
    cand = _write(
        tmp_path, "b", {"tp1.con16": 0.988, "tp2.con1": 1.15},
        {"tp1": _bind(1334, 1), "tp2": f"{MARKER}\n" * 2 + _bind(3450, 2)},
    )  # fmt: skip
    inert = inert_members([cand], [ctrl], marker=MARKER)
    assert list(inert) == ["tp1"] and "recurrent slots [1334] as the control's" in inert["tp1"]
    noise = NoiseFloor(spread={k: 0.0001 for k in ctrl.headline()})
    cmp = compare(cand, ctrl, noise, inert=inert, **RULE)
    assert cmp.verdict == "WIN" and cmp.inert_losses == ("tp1.con16",)
    assert "tp1.con16" not in str(cmp.judged())
