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
    remeasure_reason,
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


def test_a_pair_of_one_tree_judges_its_own_spread_as_noise(tmp_path):
    a = _write(tmp_path, "a", {}, {})
    a2 = _write(tmp_path, "a2", {"tp2.con16": 1.0638}, {})
    old = NoiseFloor(spread={k: 0.0008 for k in a.headline()})
    assert compare(a2, a, old, **RULE).verdict == "WIN"  # the floor of an older tree
    assert compare(a, a2, old, **RULE).verdict == "LOSS"
    own = NoiseFloor.from_measurements([a, a2])
    assert compare(a2, a, own, **RULE).verdict == compare(a, a2, own, **RULE).verdict == "NEUTRAL"


def test_a_loss_one_disturbed_run_explains_is_measured_again_and_a_far_one_stands(tmp_path):
    a = _write(tmp_path, "a", {}, {})
    noise = NoiseFloor(spread={k: 0.0001 for k in a.headline()})
    near = compare(_write(tmp_path, "near", {"tp1.con16": 0.988, "tp2.con1": 1.1}, {}), a, noise,
                   **RULE)  # fmt: skip
    assert near.verdict == "LOSS" and near.losses == (
        "tp1.con16.output_tok_s_gpu",
        "tp1.con16.tok_s_user",
    )
    assert remeasure_reason(near, noise, **RULE).startswith("every loss is within 2x its floor")
    far = compare(_write(tmp_path, "far", {"tp1.con16": 0.93}, {}), a, noise, **RULE)
    assert far.verdict == "LOSS" and remeasure_reason(far, noise, **RULE) == ""


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
    assert remeasure_reason(cmp, noise, **RULE) == "it lost where it changes nothing: tp1.con16"
