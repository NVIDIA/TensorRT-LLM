# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A requested cell the contract leaves uncovered stops the run at preflight."""

from __future__ import annotations

from types import SimpleNamespace

import yaml

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import requirements as r


def _data(**extra):
    """The request, with witnesses on: the disaggregated witness covers ``disagg``."""
    return {
        "parallel": "unified",
        "members": ["tp1", "tep4"],
        "features": ["disagg", "multimodal"],
        "perf": {"workload": {"isl": 8192, "osl": 1024}},
        "accuracy": {"witness": {"enabled": True}},
        **extra,
    }


def test_preflight_stops_on_a_gap_and_writes_the_matrix(tmp_path, monkeypatch):
    from agent_flow.workflows.trtllm_modeling_v2.run.stages.preflight import PreflightStage

    sections: list[tuple[str, str]] = []

    class Stage(PreflightStage):
        task_data = {**_data(), "features": ["disagg"]}
        workspace = tmp_path
        journal = SimpleNamespace(
            open=lambda **kw: None, section=lambda t, b: sections.append((t, b))
        )

        def _gate_env(self):
            return SimpleNamespace(world_size=4)

    contract = c.Contract(checks=(c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),))
    monkeypatch.setattr(c, "compile_contract", lambda data, world: contract)
    assert Stage()._check_coverage(log=None) is False
    written = yaml.safe_load((tmp_path / r.COVERAGE_FILENAME).read_text())
    assert written["members"] == ["tp1", "tep4"]
    assert sections and "left uncovered" in sections[0][0]
