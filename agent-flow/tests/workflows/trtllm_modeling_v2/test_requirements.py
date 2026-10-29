# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A requested cell the contract leaves uncovered stops the run at preflight."""

from __future__ import annotations

from types import SimpleNamespace

import yaml

from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run.validation import requirements as r


def test_a_smoke_unit_is_left_unjudged_by_the_specs_own_choice():
    data = {**_data(), "features": ["multimodal"], "profile": {"multimodal": "smoke"}}
    matrix = r.requirements_matrix(data, c.compile_contract(data, world_size=4), 4)
    assert not matrix.gaps()
    assert "smoke" in matrix.cell("multimodal", "tep4").reason


def _data(**extra):
    """The request: the disaggregated-serving test covers ``disagg``."""
    return {
        "parallel": "unified",
        "members": ["tp1", "tep4"],
        "features": ["disagg", "multimodal"],
        "perf": {"workload": {"isl": 8192, "osl": 1024}},
        "accuracy": {"disagg": {"test": "tests/accuracy/test_disagg.py::TestM::test_auto_dtype"}},
        **extra,
    }


def test_preflight_stops_on_a_gap_and_writes_the_matrix(tmp_path, monkeypatch):
    from agent_flow.workflows.trtllm_modeling_v2.run.startup.bootstrap import _check_coverage

    sections: list[tuple[str, str]] = []

    class Stage:
        task_data = {**_data(), "features": ["disagg"]}
        workspace = tmp_path
        journal = SimpleNamespace(
            open=lambda **kw: None, section=lambda t, b: sections.append((t, b))
        )

        def _gate_env(self):
            return SimpleNamespace(world_size=4)

    contract = c.Contract(checks=(c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),))
    monkeypatch.setattr(c, "compile_contract", lambda data, world: contract)
    assert _check_coverage(Stage(), log=None) is False
    written = yaml.safe_load((tmp_path / r.COVERAGE_FILENAME).read_text())
    assert written["members"] == ["tp1", "tep4"]
    assert sections and "left uncovered" in sections[0][0]


def test_disaggs_test_runs_once_and_a_pair_of_features_is_proven_only_where_named():
    data = {
        **_data(),
        "features": ["disagg", "multimodal"],
        "profile": {"multimodal": "smoke"},
        "members": {"default": ["tp1", "tep4"], "disagg": ["dep2"]},
    }
    matrix = r.requirements_matrix(data, c.compile_contract(data, world_size=4), 4)
    assert not matrix.gaps()
    assert "runs once, at dep2" in matrix.cell("disagg", "tp1").reason
    # dep2 is the test's place, not a column of the request.
    assert matrix.cell("core", "dep2") is None and "| core | certified | certified |  |" in (
        matrix.render()
    )
    assert matrix.cell("disagg+multimodal", "tep4").accepted
