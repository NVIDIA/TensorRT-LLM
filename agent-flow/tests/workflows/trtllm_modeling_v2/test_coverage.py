# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``vocabulary.yaml`` — the typed dispatch queue."""

from __future__ import annotations

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as vocab


def _ok_step(**overrides):
    step = {
        "step": "rmsnorm_q",
        "values": {"dtype": "bf16", "head_dim": 128},
        "state": vocab.STATE_OK,
        "entry": "catalog/norm/flashinfer_rmsnorm.py",
        "contract_line": "certified: sm_103, bf16, head_dim in {64, 128}",
    }
    step.update(overrides)
    return step


def _doc(**overrides):
    doc = {
        "verdict": vocab.VERDICT_ASSEMBLABLE,
        "steps": [_ok_step()],
        "queue": [],
    }
    doc.update(overrides)
    return doc


def test_ok_row_without_contract_line_is_rejected():
    doc = _doc(steps=[_ok_step(contract_line="")])
    with pytest.raises(vocab.VocabularyError, match="contract_line"):
        vocab.parse_vocabulary(doc)


def test_a_writing_tier_must_say_what_it_rejected():
    doc = _doc(
        verdict=vocab.VERDICT_AFTER_ADDITIONS,
        steps=[{"step": "exotic_norm", "state": vocab.STATE_GAP, "tier": vocab.TIER_TRITON}],
        queue=[{"n": 1, "mode": vocab.MODE_ONBOARD, "target": "exotic_norm"}],
    )
    with pytest.raises(vocab.VocabularyError, match="tried"):
        vocab.parse_vocabulary(doc)


def test_a_ceiling_that_ignores_the_writing_tiers_is_rejected():
    doc = _doc(
        verdict=vocab.VERDICT_CEILING,
        ceiling={
            "step": "exotic_norm",
            "searched": ["inventory"],
            "exhausted": {1: "no trtllm op", 2: "nothing in modules/**"},
        },
    )
    with pytest.raises(vocab.VocabularyError, match=r"tier\(s\) \[3, 4, 5\]"):
        vocab.parse_vocabulary(doc)


def test_assemblable_with_a_queue_is_contradictory():
    doc = _doc(queue=[{"n": 1, "mode": vocab.MODE_ONBOARD, "target": "a"}])
    with pytest.raises(vocab.VocabularyError, match="assemblable_after_additions"):
        vocab.parse_vocabulary(doc)


def test_every_problem_is_batched():
    doc = {
        "verdict": "maybe",
        "steps": [{"step": "", "state": "NOPE"}],
        "queue": [{"n": 1, "mode": "guess", "target": ""}],
    }
    with pytest.raises(vocab.VocabularyError) as excinfo:
        vocab.parse_vocabulary(doc)
    message = str(excinfo.value)
    assert "verdict" in message and "state" in message and "mode" in message


def test_load_from_disk(tmp_path):
    doc = _doc(
        verdict=vocab.VERDICT_AFTER_ADDITIONS,
        steps=[_ok_step(state=vocab.STATE_CERT)],
        queue=[
            {
                "n": 1,
                "mode": vocab.MODE_VERIFY,
                "target": "flashinfer_rmsnorm",
                "surface": "head_dim=72",
            }
        ],
    )
    path = tmp_path / "vocabulary.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    parsed = vocab.load_vocabulary(path)
    assert parsed[0].surface == "head_dim=72"
    assert parsed[0].target == "flashinfer_rmsnorm"


# ------------------------------------------------------------------ unit


_UNITS = ("core", "dspark", "multimodal", "kv_quant")


def _queue_doc(*rows):
    return _doc(verdict=vocab.VERDICT_AFTER_ADDITIONS, queue=list(rows))


def test_a_queue_row_may_name_the_unit_that_first_needs_it():
    parsed = vocab.parse_vocabulary(
        _queue_doc(
            {"n": 1, "mode": vocab.MODE_ONBOARD, "target": "moe/drafter_moe", "unit": "dspark"},
            {"n": 2, "mode": vocab.MODE_ONBOARD, "target": "norm/core_only"},
        ),
        units=_UNITS,
    )
    assert [item.unit for item in parsed] == ["dspark", vocab.UNIT_CORE]


def test_a_unit_the_run_does_not_build_is_rejected():
    doc = _queue_doc({"n": 1, "mode": vocab.MODE_ONBOARD, "target": "x", "unit": "dspakr"})
    with pytest.raises(vocab.VocabularyError, match="not one of this run's assembly units"):
        vocab.parse_vocabulary(doc, units=_UNITS)
