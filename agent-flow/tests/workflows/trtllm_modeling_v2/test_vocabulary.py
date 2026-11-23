# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 vocabulary parsing and engine sizing."""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as vocab
from agent_flow.workflows.trtllm_modeling_v2.spec import sizing


def test_an_engine_takes_the_fewest_devices_its_weights_fit_on(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 300 * 2**30}})
    )
    weights = sizing.checkpoint_weight_bytes(checkpoint)
    # 300 GiB over 60 GiB a device (75% of 80): one engine needs 8 devices.
    assert sizing.engine_width(weights, 80) == 8
    # Unknown device memory, or no weights to read: no estimate.
    assert sizing.engine_width(weights, None) is None
    assert sizing.checkpoint_weight_bytes(tmp_path / "absent") is None


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


@pytest.mark.parametrize("tier", ["tier 2", 0])
def test_bad_queue_tiers_are_batched_as_vocabulary_errors(tier):
    doc = _doc(
        verdict=vocab.VERDICT_AFTER_ADDITIONS,
        steps=[_ok_step(state=vocab.STATE_CERT)],
        queue=[
            {"n": 1, "mode": "verify", "target": "gemm/cublas_mm", "surface": "k=64", "tier": tier}
        ],
    )
    with pytest.raises(vocab.VocabularyError, match=r"queue\[0\].tier"):
        vocab.parse_vocabulary(doc)
