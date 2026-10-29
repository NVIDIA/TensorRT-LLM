# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A micro-bench scaled to e2e by its share of the step; outputs and first tokens held roughly."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.tune.campaign import (
    _share_problems,
    perf_points,
    project,
)
from agent_flow.workflows.trtllm_modeling_v2.tune.harness.microbench import diverged
from agent_flow.workflows.trtllm_modeling_v2.tune.logits_check import LogitsCheck, compare


def test_a_speedup_projects_through_the_share_of_each_point_it_takes():
    points = perf_points(SimpleNamespace(parts=[], concurrencies=[1, 32, 128]))
    assert points == ["con1", "con32", "con128"]
    share = {"moe": {"con1": 0.5, "con128": 0.25}, "norm": {"con1": 0.1}}
    # 1.25x on half the step saves 10 % of it: 1/0.9 - 1; a quarter of it: 1/0.95 - 1.
    gains = project({"moe": 0.25, "norm": 0.0}, share)
    assert gains == {"con1": pytest.approx(1 / 0.9 - 1), "con128": pytest.approx(1 / 0.95 - 1)}
    assert project({"moe": 0.25}, share, calibration=0.5)["con1"] == pytest.approx(0.5 / 0.9 - 0.5)
    assert _share_problems(share, ["moe", "norm"], points) == ""
    for bad, why in [
        ({}, "names no case"),
        ({"other": {"con1": 0.5}}, "'other' is no declared case"),
        ({"moe": {"con64": 0.5}}, "'con64' is not one of"),
        ({"moe": {"con1": 0.7}, "norm": {"con1": 0.4}}, "more than the whole step at con1"),
        ({"moe": {"con1": 0.0}}, "every share is 0"),
    ]:
        assert why in _share_problems(bad, ["moe", "norm"], points)


def test_outputs_and_first_tokens_are_held_roughly_not_bit_exact():
    assert diverged({"a": {"exact": False, "error": 0.03}, "b": {"missing": "no tensor"}}) == ""
    assert diverged({"a": {"error": 0.2}, "b": {"error": float("nan")}}) == (
        "a: error 0.2 past 0.05; b: error nan past 0.05"
    )
    check = LogitsCheck(prompts=("x",) * 4)
    control = {f"p{i}": {7: -0.1, 9: -0.2, 3: -4.0} for i in range(4)}
    # A flip between near-tied tokens (0.1 nats apart) agrees; a small shift in mass is close.
    tied = {**control, "p0": {9: -0.1, 7: -0.2, 3: -4.0}}
    ok, why, numbers = compare(control, tied, check)
    assert ok and numbers["top1"] == 1.0 and numbers["mean_kl"] < 0.01
    broken = {**control, "p0": {3: -0.1, 7: -3.0}}
    ok, why, numbers = compare(control, broken, check)
    assert not ok and numbers["top1"] == 0.75 and "agree on 75% of 4 prompts" in why
    ok, why, _ = compare(control, {"p0": control["p0"]}, check)
    assert not ok and "3 unanswered" in why
