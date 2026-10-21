# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A campaign over several perf configs: a broken constraint, and the goal."""

from __future__ import annotations

import json
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.tune.records import (
    PARTS_FILE,
    Measurement,
    NoiseFloor,
    compare,
    goal_reached,
)


def _curve(path: Path, label: str, lat: float, thr: float, tpot: float = 20.0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        {"concurrency": 1, "tok_s_user": lat, "output_tok_s_gpu": lat, "mean_tpot_ms": tpot},
        {"concurrency": 64, "tok_s_user": thr / 64, "output_tok_s_gpu": thr, "mean_tpot_ms": tpot},
    ]
    (path / "summary.json").write_text(json.dumps(rows))
    (path / "meta.json").write_text(json.dumps({"label": label, "hostname": "h1"}))


def _composite(root: Path, label: str, parts: dict, constraints=None) -> Measurement:
    for name, (lat, thr, tpot) in parts.items():
        _curve(root / label / name, name, lat, thr, tpot)
    (root / label / PARTS_FILE).write_text(
        json.dumps({"parts": list(parts), "constraints": constraints or {}})
    )
    (root / label / "meta.json").write_text(json.dumps({"label": label, "hostname": "h1"}))
    return Measurement.load(root / label)


def test_breaking_a_constraint_the_control_kept_is_a_loss(tmp_path):
    bound = {"chat": {"tpot_ms_max": 30.0}}
    ctrl = _composite(tmp_path, "ctrl", {"chat": (100, 6400, 25)}, bound)
    cand = _composite(tmp_path, "cand", {"chat": (130, 9000, 35)}, bound)
    result = compare(cand, ctrl, NoiseFloor(), min_gain=0.03, noise_factor=2.0)
    assert result.verdict == "LOSS"
    assert "breaks a constraint" in result.reason and "chat@con1" in result.reason


def test_the_goal_is_every_headline_against_the_reference(tmp_path):
    ref = _composite(tmp_path, "ref", {"chat": (100, 6400, 20)})
    near = _composite(tmp_path, "near", {"chat": (112, 6900, 20)})
    there = _composite(tmp_path, "there", {"chat": (112, 7100, 20)})
    assert not goal_reached(near, ref, 1.1)
    assert goal_reached(there, ref, 1.1)
