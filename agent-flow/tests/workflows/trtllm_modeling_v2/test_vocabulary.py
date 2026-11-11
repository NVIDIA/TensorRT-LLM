# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The survey's vocabulary: what a ``vocabulary.yaml`` must say to be dispatched from."""

from __future__ import annotations

import pytest

from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as vocab


def _ok_step(**overrides):
    step = {
        "step": "rmsnorm_q",
        "values": {"dtype": "bf16", "head_dim": 128},
        "state": "OK",
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
        "queue": [{"n": 1, "mode": "guess", "target": ""}, {"n": "x"}],
    }
    with pytest.raises(vocab.VocabularyError) as excinfo:
        vocab.parse_vocabulary(doc)
    message = str(excinfo.value)
    assert "`verdict` is 'maybe'" in message and "queue[0].mode is 'guess'" in message
    assert "queue[0].target must name" in message and "queue[1] must be a mapping whose" in message


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
            steps=[_ok_step(state="CERT")],
            queue=[row],
        )

    vocab.parse_vocabulary(doc(vocab.MAX_ROW_CELLS))
    with pytest.raises(vocab.VocabularyError, match=f"{vocab.MAX_ROW_CELLS + 1} cells"):
        vocab.parse_vocabulary(doc(vocab.MAX_ROW_CELLS + 1))


@pytest.mark.parametrize("tier", ["tier 2", 0])
def test_bad_queue_tiers_are_batched_as_vocabulary_errors(tier):
    doc = _doc(
        verdict=vocab.VERDICT_AFTER_ADDITIONS,
        steps=[_ok_step(state="CERT")],
        queue=[
            {"n": 1, "mode": "verify", "target": "gemm/cublas_mm", "surface": "k=64", "tier": tier}
        ],
    )
    with pytest.raises(vocab.VocabularyError, match=r"queue\[0\].tier"):
        vocab.parse_vocabulary(doc)


def _rows(*targets):
    rows = [
        {"n": n, "mode": vocab.MODE_VERIFY, "target": target, "surface": "k=64"}
        for n, target in enumerate(targets, 1)
    ]
    return _doc(verdict=vocab.VERDICT_AFTER_ADDITIONS, queue=rows)


@pytest.mark.parametrize(
    "target",
    ["ops/../escape", "ops/norm/..", "../escape", "ops/norm/../../escape", "norm/./rms", "ops/rms"],
)
def test_a_row_lands_inside_the_catalog_or_the_target_by_path_component(target):
    with pytest.raises(vocab.VocabularyError, match="lands inside the catalog or the target"):
        vocab.parse_vocabulary(_rows(target))


def test_one_row_writes_one_landing_and_a_shared_rows_variant_is_the_targets_own():
    # The same entry spelled twice is one landing: the second row is refused, not run.
    with pytest.raises(vocab.VocabularyError) as refused:
        vocab.parse_vocabulary(_rows("ops/norm/rms", "/ops/norm/rms/"))
    assert "queue[1] lands at ops/norm/rms, as row 1: one row writes one entry" in str(
        refused.value
    )
    [local, shared] = vocab.parse_vocabulary(_rows("ops/norm/rms", "norm/rms"))
    home = "tensorrt_llm/_torch/_experimental/modeling_v2/models/family/target"
    assert vocab.entry_paths(local, home)[0] == f"{home}/ops/norm/rms.py"
    assert vocab.entry_paths(shared, home)[0] == f"{home}/ops/norm/rms__r2.py"
    # Every file a row writes is under the target or its mirrored tests.
    tests = vocab.model_ops_tests_dir(home)
    for item in (local, shared):
        assert all(
            p.startswith((f"{home}/ops/", f"{tests}/")) for p in vocab.entry_paths(item, home)
        )
