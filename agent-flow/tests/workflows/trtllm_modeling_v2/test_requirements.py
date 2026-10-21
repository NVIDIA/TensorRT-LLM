# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The requirements matrix: what the contract proves of the request, cell by cell."""

from __future__ import annotations

from types import SimpleNamespace

import yaml

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import requirements as r
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema


def _data(**extra):
    return {
        "parallel": "unified",
        "members": ["tp1", "tep4"],
        "features": ["disagg", "multimodal"],
        "perf": {"workload": {"isl": 8192, "osl": 1024}},
        **extra,
    }


def _contract():
    return c.Contract(
        checks=(
            c.Check("core", "boot", "tp1", c.TIER_INNER),
            c.Check("core", "boot", "tep4", c.TIER_INNER),
            c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
            c.Check("disagg", "boot", "tp1", c.TIER_INNER),
            c.Check("disagg", "equivalence", "tp1", c.TIER_PROOF),
            c.Check("multimodal", "boot", "tp1", c.TIER_INNER),
            c.Check("multimodal", "boot", "tep4", c.TIER_INNER),
            c.Check("multimodal", "accuracy[gsm8k]", "tep4", c.TIER_INNER),
            c.Check("multimodal", "envelope", "tep4", c.TIER_DELIVERY),
        )
    )


def test_each_cell_says_how_the_contract_covers_it():
    matrix = r.requirements_matrix(_data(), _contract(), world_size=4)
    assert matrix.members == ("tp1", "tep4")
    assert matrix.features == ("core", "disagg", "multimodal", "disagg+multimodal")
    assert matrix.cell("core", "tp1").status == r.CERTIFIED
    sampled = matrix.cell("core", "tep4")
    assert sampled.status == r.SAMPLED and "judged at tp1" in sampled.reason
    assert matrix.cell("disagg", "tp1").status == r.CERTIFIED  # its witness judges it
    gap = matrix.cell("disagg", "tep4")
    assert gap.status == r.NOT_COVERED and gap.checks == ()
    # gsm8k sends no image: nothing judges the vision path, which the keyword boot does not.
    vision = matrix.cell("multimodal", "tp1")
    assert vision.status == r.NOT_COVERED and "image" in vision.reason
    pair = matrix.cell("disagg+multimodal", "tp1")
    assert pair.status == r.NOT_COVERED and "text prompts only" in pair.reason
    assert matrix.workload == "ISL 8192 / OSL 1024"


def test_a_gap_is_a_decision_the_spec_states():
    matrix = r.requirements_matrix(_data(), _contract(), world_size=4)
    assert {(g.feature, g.member) for g in matrix.gaps()} == {
        ("disagg", "tep4"),
        ("multimodal", "tp1"),
        ("multimodal", "tep4"),
        ("disagg+multimodal", "tp1"),
        ("disagg+multimodal", "tep4"),
    }
    message = r.gap_message(matrix)
    assert "disagg@tep4" in message and "coverage:" in message
    accepting = _data(coverage={"accept": ["disagg@tep4", "multimodal@*", "disagg+multimodal@*"]})
    matrix = r.requirements_matrix(accepting, _contract(), world_size=4)
    assert matrix.gaps() == []
    assert matrix.cell("disagg", "tep4").accepted
    assert "not covered (accepted)" in matrix.render()


def test_a_multimodal_unit_is_judged_only_by_a_check_that_sends_images(monkeypatch):
    with_logits = _data(accuracy={"logits": {"reference": "/ref.json.gz"}})
    contract = c.Contract(
        checks=(
            c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
            c.Check("multimodal", "boot", "tp1", c.TIER_INNER),
            c.Check("multimodal", "logits", "tp1", c.TIER_INNER),
            c.Check("multimodal", "boot", "tep4", c.TIER_INNER),
            c.Check("multimodal", "accuracy[gsm8k]", "tep4", c.TIER_INNER),
        )
    )
    reference = SimpleNamespace(prompts=[SimpleNamespace(images=()), SimpleNamespace(images=())])
    monkeypatch.setattr(r, "load_reference", lambda path: reference)
    matrix = r.requirements_matrix(with_logits, contract, world_size=4)
    assert matrix.cell("multimodal", "tp1").status == r.NOT_COVERED  # text prompts only
    reference.prompts.append(SimpleNamespace(images=("img/merlion.png",)))
    matrix = r.requirements_matrix(with_logits, contract, world_size=4)
    assert matrix.cell("multimodal", "tp1").status == r.CERTIFIED
    sampled = matrix.cell("multimodal", "tep4")
    assert sampled.status == r.SAMPLED and "judged at tp1" in sampled.reason
    benchmark = c.Contract(checks=(c.Check("multimodal", "accuracy[mmmu]", "tep4", c.TIER_INNER),))
    matrix = r.requirements_matrix(_data(), benchmark, world_size=4)
    assert matrix.cell("multimodal", "tep4").status == r.CERTIFIED


def test_the_matrix_renders_as_a_table_with_why_each_cell_falls_short():
    text = r.requirements_matrix(_data(), _contract(), world_size=4).render()
    assert text.startswith("Workload: ISL 8192 / OSL 1024.")
    assert "| feature | tp1 | tep4 |" in text
    assert "| core | certified | sampled |" in text
    assert "- disagg@tep4: not covered: disagg is gated at" in text


def test_the_spec_validates_what_it_accepts():
    errors: list[str] = []
    task_schema._validate_coverage({"accept": ["mtp7@tp4", "x@*"]}, errors)
    assert errors == []
    task_schema._validate_coverage({"accept": ["mtp7"]}, errors)
    task_schema._validate_coverage({"allow": []}, errors)
    assert len(errors) == 2


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
