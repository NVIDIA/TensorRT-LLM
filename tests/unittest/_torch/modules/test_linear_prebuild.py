# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linear GEMM prebuild next to weight loading (modules/linear_prebuild.py)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.modules.linear_prebuild as lp
import tensorrt_llm._torch.pyexecutor.warmup_overlap as warmup_overlap
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.utils import deep_gemm_jit_warmup_buckets
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

FP8 = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
# Wider than one wave of 128-row tiles on SM100 / SM103: no low-M range.
NO_LOW_M = (32768, 512)


class _Mapping:
    def __init__(self, helix: bool):
        self.helix = helix

    def has_cp_helix(self) -> bool:
        return self.helix


def _llm_args(max_num_tokens, max_batch_size, max_seq_len=None, draft=0):
    spec = SimpleNamespace(max_total_draft_tokens=draft, max_draft_len=draft) if draft else None
    return SimpleNamespace(
        max_num_tokens=max_num_tokens,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        speculative_config=spec,
        cuda_graph_config=None,
    )


def _sweep(*token_counts):
    return tuple(sorted({m for t in token_counts for m in (t, *deep_gemm_jit_warmup_buckets(t))}))


def test_tuning_token_counts():
    # A context forward of max_num_tokens (capped by the sequence budget) and one
    # generation request, as DecoderWarmup._run_autotuner_warmup tunes them.
    assert lp._tuning_token_counts(_llm_args(32768, 16, 996672), _Mapping(False)) == (32768, 1)
    assert lp._tuning_token_counts(_llm_args(8192, 4, 1025, draft=3), _Mapping(False)) == (4096, 4)
    # Helix CP generation workers tune one batch of generation requests.
    assert lp._tuning_token_counts(_llm_args(14, 2, draft=6), _Mapping(True)) == (14,)
    assert lp._tuning_token_counts(_llm_args(192, 32, draft=5), _Mapping(True)) == (192,)


class _TextModel(nn.Module):
    """A model that builds an FP8 block-scale GEMM while it loads."""

    def linear_prebuild_fp8_block_scale_shapes(self):
        return [(4096, 7168)]


def _model():
    with torch.device("meta"):
        model = nn.Module()
        model.fp8 = Linear(7168, 2048, bias=False, dtype=torch.bfloat16, quant_config=FP8)
        model.wide = Linear(
            NO_LOW_M[1], NO_LOW_M[0], bias=False, dtype=torch.bfloat16, quant_config=FP8
        )
        model.bf16 = Linear(1024, 1024, bias=False, dtype=torch.bfloat16)
        # The hook is found on a wrapped submodule, as on a multimodal model's text model.
        model.llm = _TextModel()
    return model


@pytest.mark.skipif(not is_sm_100f(), reason="DeepGEMM swap-AB GEMMs run on SM100f")
def test_collect_plans_the_swept_shapes():
    model = _model()
    ctx = lp.collect(model, _llm_args(8192, 8), _Mapping(False))
    shapes = {(n, k): sweep for n, k, sweep in ctx.fp8}
    assert set(shapes) == {(2048, 7168), NO_LOW_M, (4096, 7168)}
    # The 8192-token forward runs fp8_swap_ab_gemm for every shape; the one-request
    # forward only where the low-M GEMM does not serve one row.
    for (n, k), sweep in shapes.items():
        expected = _sweep(8192, 1) if lp._low_m_max(n, k) < 1 else _sweep(8192)
        assert sweep == expected, (n, k)
    assert not ctx.flashinfer_mxfp8

    # 14-token generation steps: the shapes whose low-M GEMM serves them are not swept.
    gen = lp.collect(model, _llm_args(14, 2, draft=6), _Mapping(True))
    assert lp._low_m_max(*NO_LOW_M) < 14
    swept = {(n, k) for n, k in shapes if lp._low_m_max(n, k) < 14}
    assert {(n, k): sweep for n, k, sweep in gen.fp8} == {s: _sweep(14) for s in swept}


@pytest.mark.skipif(not is_sm_100f(), reason="DeepGEMM swap-AB GEMMs run on SM100f")
def test_start_linear_prebuild_builds_the_sweep_before_wait(monkeypatch):
    monkeypatch.setenv("TLLM_OVERLAP_PREWARM", "1")
    monkeypatch.delenv("TLLM_OVERLAP_PREWARM_LINEAR", raising=False)
    monkeypatch.setattr(warmup_overlap, "_linear_groups", [])
    monkeypatch.setattr(warmup_overlap, "_linear_threads", [])
    monkeypatch.setattr(warmup_overlap, "_released", False)
    monkeypatch.setattr(warmup_overlap, "_thread", None)
    with torch.device("meta"):
        model = nn.Module()
        model.wide = Linear(
            NO_LOW_M[1], NO_LOW_M[0], bias=False, dtype=torch.bfloat16, quant_config=FP8
        )

    warmup_overlap.start_linear_prebuild(model, _llm_args(14, 2, draft=6), _Mapping(True))
    warmup_overlap.wait()

    groups = {group.name: group for group in warmup_overlap._linear_groups}
    deep_gemm = groups["deep_gemm"]
    assert (deep_gemm.count, deep_gemm.failed) == (1, 0)
    assert deep_gemm.launches == len(_sweep(14))
    assert groups["flashinfer_mxfp8"].count == 0
    assert not any(thread.is_alive() for thread in warmup_overlap._linear_threads)
