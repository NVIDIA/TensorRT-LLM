# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The trunk verify's early index scores (`DP_SCORES_EARLY_ENV`, opt-in, CPU): where the switch
engages, and that only the trunk call behind the attention decode prologue takes the early
launch, with its anchor convert on PDL.

The early launch's bits (the flat grid, eager and under replay behind a PDL convert that writes
the anchors it reads) are covered by
`ops/attention/test_modeling_v2_qsa_index_scores_fp8.py::test_early_is_bitwise_the_default_flat_launch`.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)


def _core(**over):
    core = SimpleNamespace(index_key_fp8=True, dp_rank=1, dp_size=8)
    for k, v in over.items():
        setattr(core, k, v)
    return core


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "value, dp, fp8, pdl, engaged, warned",
    [
        (None, True, True, True, False, 0),
        ("0", True, True, True, False, 0),
        ("1", True, True, True, True, 0),
        ("1", False, True, True, False, 1),
        ("1", True, False, True, False, 1),
        ("1", True, True, False, False, 1),
    ],
)
def test_build(monkeypatch, value, dp, fp8, pdl, engaged, warned):
    if value is None:
        monkeypatch.delenv(M.DP_SCORES_EARLY_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_SCORES_EARLY_ENV, value)
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: dp)
    monkeypatch.setattr(M, "_INDEX_SCORES_PDL", pdl)
    warnings = []
    monkeypatch.setattr(M.logger, "warning", warnings.append)
    assert M._dp_scores_early(_core(index_key_fp8=fp8), torch.device("cpu")) is engaged
    assert len(warnings) == warned
    assert all(M.DP_SCORES_EARLY_ENV in w for w in warnings)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "behind, engaged, early", [(True, True, True), (False, True, False), (True, False, False)]
)
def test_only_the_call_behind_the_prologue_is_early(monkeypatch, behind, engaged, early):
    calls = []
    monkeypatch.setattr(M, "qsa_index_scores_fp8", lambda *a, early=False: calls.append(early))
    core = _core(dp_scores_early=engaged, compress_ratio=4, tep_fast=False)
    q = torch.zeros(48, 4, 128)
    M._decode_index_scores(core, q, None, None, None, None, 32, 4, behind_prologue=behind)
    assert calls == [early]


@pytest.mark.cpu_only
def test_convert_takes_pdl_only_when_asked(monkeypatch):
    calls = []
    monkeypatch.setattr(M, "index_anchors_to_fp8", lambda *a, use_pdl=False: calls.append(use_pdl))
    core = _core(compress_ratio=4)
    M._index_anchors_fp8(core, None, None, None, None, 32)
    M._index_anchors_fp8(core, None, None, None, None, 32, use_pdl=True)
    M._index_anchors_fp8(_core(index_key_fp8=False), None, None, None, None, 32, use_pdl=True)
    assert calls == [False, True]


@pytest.mark.cpu_only
def test_the_prologue_branch_converts_on_pdl_under_the_switch():
    """The decode attention block's prologue branch hands the switch to its anchor convert, and
    its scores call says it is behind the prologue (the only early caller)."""
    src = inspect.getsource(M.DecodeTarget._attention_block)
    branch = src[src.index("qwen4_exp_attn_decode_prologue(") : src.index("_attn_in_heads(")]
    assert 'use_pdl=getattr(core, "dp_scores_early", False)' in branch
    assert "behind_prologue=prologue" in src
    others = [
        name
        for name, fn in inspect.getmembers(M, inspect.isfunction)
        if name != "_decode_index_scores" and "behind_prologue=True" in inspect.getsource(fn)
    ]
    assert not others
