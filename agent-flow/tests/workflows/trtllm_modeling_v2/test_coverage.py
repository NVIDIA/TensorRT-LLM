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


@pytest.mark.parametrize(
    "doc,match",
    [
        (_doc(steps=[_ok_step(contract_line="")]), "contract_line"),
        # A writing tier must say what it rejected.
        (
            _doc(
                verdict=vocab.VERDICT_AFTER_ADDITIONS,
                steps=[
                    {"step": "exotic_norm", "state": vocab.STATE_GAP, "tier": vocab.TIER_TRITON}
                ],
                queue=[{"n": 1, "mode": vocab.MODE_ONBOARD, "target": "exotic_norm"}],
            ),
            "tried",
        ),
        # A ceiling that ignores the writing tiers.
        (
            _doc(
                verdict=vocab.VERDICT_CEILING,
                ceiling={
                    "step": "exotic_norm",
                    "searched": ["inventory"],
                    "exhausted": {1: "no trtllm op", 2: "nothing in modules/**"},
                },
            ),
            r"tier\(s\) \[3, 4, 5\]",
        ),
        # Assemblable with a queue is contradictory.
        (
            _doc(queue=[{"n": 1, "mode": vocab.MODE_ONBOARD, "target": "a"}]),
            "assemblable_after_additions",
        ),
    ],
    ids=["ok-without-contract-line", "writing-tier-without-tried", "ceiling", "queue"],
)
def test_an_incoherent_vocabulary_is_rejected(doc, match):
    with pytest.raises(vocab.VocabularyError, match=match):
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


def test_a_row_covers_each_axis_not_their_product():
    def doc(count):
        row = {
            "n": 1,
            "mode": vocab.MODE_VERIFY,
            "target": "gemm/cublas_mm",
            "surface": "k=7168",
            "cells": [{"m": m, "k": 7168, "n": 1536} for m in range(1, count + 1)],
        }
        return _doc(
            verdict=vocab.VERDICT_AFTER_ADDITIONS,
            steps=[_ok_step(state=vocab.STATE_CERT)],
            queue=[row],
        )

    vocab.parse_vocabulary(doc(vocab.MAX_ROW_CELLS))
    with pytest.raises(vocab.VocabularyError, match=f"{vocab.MAX_ROW_CELLS + 1} cells"):
        vocab.parse_vocabulary(doc(vocab.MAX_ROW_CELLS + 1))


# ------------------------------------------------------------------ unit


def test_a_queue_row_may_name_a_unit_the_run_builds_that_first_needs_it():
    def queue(*rows):
        return _doc(verdict=vocab.VERDICT_AFTER_ADDITIONS, queue=list(rows))

    units = ("core", "dspark", "multimodal", "kv_quant")
    parsed = vocab.parse_vocabulary(
        queue(
            {"n": 1, "mode": vocab.MODE_ONBOARD, "target": "moe/drafter_moe", "unit": "dspark"},
            {"n": 2, "mode": vocab.MODE_ONBOARD, "target": "norm/core_only"},
        ),
        units=units,
    )
    assert [item.unit for item in parsed] == ["dspark", vocab.UNIT_CORE]
    typo = queue({"n": 1, "mode": vocab.MODE_ONBOARD, "target": "x", "unit": "dspakr"})
    with pytest.raises(vocab.VocabularyError, match="not one of this run's assembly units"):
        vocab.parse_vocabulary(typo, units=units)
