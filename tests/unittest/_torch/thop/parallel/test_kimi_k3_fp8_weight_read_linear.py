# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Coverage for Kimi K3's CuTe-only FP8 weight-read linear.

This module is hand-built rather than a ``Linear`` + ``FP8BlockScalesLinearMethod``,
so the shipping FP8-block-scale tests never reach it. These tests pin the
single-scale loader contract and the specialized quant + CuTe path.
"""

import pytest
import torch
from _torch.helpers import calc_diff, per_block_cast_to_fp8
from utils.util import getSMVersion

from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
    _get_kimi_k3_mxfp8_tuning_buckets,
    _kimi_k3_mxfp8_tuning_bucket,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.models.modeling_kimi_linear import (
    _Fp8BlockScaleWeightReadLinear as K3Fp8Linear,
)

# K3-representative (out, in) projections, all 128-aligned.
SHAPES = [(512, 1024), (2048, 1024)]
MS = [1, 32, 128]

# The Rubin MXFP8 weight-read path needs both SM107 and the internal CuTe DSL
# build; keep the predicate local so the gate does not depend on a helper that
# lives outside this change.
RUBIN = pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="needs SM107 with Rubin CuTe DSL support",
)


def _ref(x, w):
    return (x.float() @ w.float().t()).to(x.dtype)


def _check(out, expected, tag):
    assert out.dtype == expected.dtype, tag
    assert out.shape == expected.shape, tag
    assert torch.isfinite(out).all(), f"{tag}: non-finite output"
    diff = calc_diff(out, expected)
    assert diff < 5e-3, f"{tag}: calc_diff={diff}"


def _make(out_features, in_features, seed=0):
    torch.random.manual_seed(seed)
    w = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / in_features**0.5
    )
    weight, weight_scale = K3Fp8Linear.quantize_weight(w)
    return w, K3Fp8Linear(weight, weight_scale, out_features)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@RUBIN
@pytest.mark.parametrize("out_features, in_features", SHAPES)
@pytest.mark.parametrize("m", MS)
def test_forward_matches_bf16(m, out_features, in_features):
    """The CuTe-only path must preserve the FP8 block-scale numerics."""
    w, mod = _make(out_features, in_features)
    x = torch.randn((m, in_features), device="cuda", dtype=torch.bfloat16)
    out = mod(x)
    _check(out, _ref(x, w), f"m={m} n={out_features} k={in_features}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@RUBIN
def test_forward_prequantized_matches_bf16():
    """D-Spark's fused KDA output must use the same CuTe scale ABI."""
    w, mod = _make(512, 1024, seed=9)
    x = torch.randn((16, 1024), device="cuda", dtype=torch.bfloat16)
    activation, activation_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x)

    out = mod.forward_prequantized(activation, activation_scale)

    _check(out, _ref(x, w), "prequantized")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@RUBIN
def test_forward_uses_specialized_quant():
    """Every call must use the specialized quantizer and fine-M runner."""
    _, mod = _make(512, 1024, seed=8)
    x = torch.randn((64, 1024), device="cuda", dtype=torch.bfloat16)
    mod(x)

    quant_calls = []
    fine_grained_m = []
    real_quant = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0
    real_gemm = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin

    class _QuantSpy:
        def __call__(self, *args, **kwargs):
            quant_calls.append(args[0].shape[0])
            return real_quant(*args, **kwargs)

    class _GemmSpy:
        def __call__(self, *args, **kwargs):
            fine_grained_m.append(kwargs["fine_grained_m"])
            return real_gemm(*args, **kwargs)

    torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0 = _QuantSpy()
    torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin = _GemmSpy()
    try:
        mod(x)
        mod(x)
    finally:
        torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0 = real_quant
        torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin = real_gemm

    assert quant_calls == [64, 64]
    assert fine_grained_m == [True, True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@RUBIN
def test_single_cute_scale_is_built_on_rubin():
    """The CuTe layout must be prepared at load time, not during capture."""
    _, mod = _make(2048, 1024, seed=5)
    assert mod.weight_scale.numel() > 0
    assert mod.weight_scale.dtype is torch.uint8
    assert not hasattr(mod, "weight_scale_mx")
    assert not hasattr(mod, "gemm_alpha")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@RUBIN
def test_placeholder_load_carries_cute_scale():
    """A loader-filled placeholder must receive the CuTe scale layout."""
    in_features, parts = 1024, [512, 512]
    torch.random.manual_seed(6)
    pairs = []
    for p in parts:
        wp = torch.randn((p, in_features), device="cuda", dtype=torch.bfloat16) / in_features**0.5
        q, s = per_block_cast_to_fp8(wp)
        pairs.append((q, s.float()))

    mod = K3Fp8Linear.empty_placeholder(sum(parts), in_features)
    mod.load_checkpoint_pair(pairs)
    assert not mod.is_placeholder
    assert mod.weight_scale.numel() > 0
    assert mod.weight_scale.dtype is torch.uint8
    assert not hasattr(mod, "gemm_alpha")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_weight_preparation_returns_only_cute_pair():
    """Both construction routes return exactly FP8 weight + CuTe scale."""
    torch.random.manual_seed(7)
    w = torch.randn((512, 1024), device="cuda", dtype=torch.bfloat16) / 32
    assert len(K3Fp8Linear.quantize_weight(w)) == 2

    q, s = per_block_cast_to_fp8(w)
    assert len(K3Fp8Linear.prepare_checkpoint_scale(q, s.float())) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_unfilled_placeholder_raises():
    mod = K3Fp8Linear.empty_placeholder(256, 256)
    x = torch.randn((2, 256), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="never filled"):
        mod(x)


def test_kimi_k3_fine_m_tuning_uses_hybrid_buckets():
    low_m = (1, 2, 4, 8, *range(16, 193, 16))
    assert _get_kimi_k3_mxfp8_tuning_buckets(17) == low_m[:5]
    assert _get_kimi_k3_mxfp8_tuning_buckets(65) == (*low_m[:8], 80)
    assert _get_kimi_k3_mxfp8_tuning_buckets(192) == low_m
    assert _get_kimi_k3_mxfp8_tuning_buckets(193) == (*low_m, 256)
    assert _get_kimi_k3_mxfp8_tuning_buckets(4096) == (
        *low_m,
        256,
        512,
        1024,
        2048,
        4096,
    )
    assert tuple(
        _kimi_k3_mxfp8_tuning_bucket(m)
        for m in (
            1,
            3,
            5,
            8,
            9,
            15,
            16,
            17,
            31,
            32,
            64,
            65,
            79,
            80,
            128,
            129,
            191,
            192,
            193,
            255,
            256,
            511,
            512,
        )
    ) == (
        1,
        2,
        4,
        8,
        8,
        8,
        16,
        16,
        16,
        32,
        64,
        80,
        80,
        80,
        128,
        144,
        192,
        192,
        256,
        256,
        256,
        512,
        512,
    )
