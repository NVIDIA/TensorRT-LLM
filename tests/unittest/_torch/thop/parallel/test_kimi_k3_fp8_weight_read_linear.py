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
"""Coverage for Kimi K3's FP8 block-scale weight-read linear.

This module is hand-built rather than a ``Linear`` + ``FP8BlockScalesLinearMethod``,
so the shipping FP8-block-scale tests never reach it. These tests pin its two
construction routes and the deferred scale-preparation contract.
"""

import pytest
import torch
from _torch.helpers import calc_diff, per_block_cast_to_fp8
from utils.util import getSMVersion, isSM100Family

from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
    _get_kimi_k3_mxfp8_tuning_buckets,
    _kimi_k3_mxfp8_tuning_bucket,
)
from tensorrt_llm._torch.models.modeling_kimi_linear import (
    _Fp8BlockScaleWeightReadLinear as K3Fp8Linear,
)

# K3-representative (out, in) projections, all 128-aligned.
SHAPES = [(512, 1024), (2048, 1024)]
MS = [1, 32, 128]

# The weight read serves every GEMM through deep_gemm's fp8_swap_ab_gemm, which
# ships for the SM100 family only.
DEEP_GEMM = pytest.mark.skipif(
    not isSM100Family(),
    reason="FP8 block-scale weight read needs SM100family. Current SM is %d." % getSMVersion(),
)


def _ref(x, w):
    return (x.float() @ w.float().t()).to(x.dtype)


def _check(out, expected, tag):
    assert out.dtype == expected.dtype, tag
    assert out.shape == expected.shape, tag
    assert torch.isfinite(out).all(), f"{tag}: non-finite output"
    diff = calc_diff(out, expected)
    assert diff < 5e-3, f"{tag}: calc_diff={diff}"


def _bf16_weight(out_features, in_features, seed=0):
    torch.random.manual_seed(seed)
    return (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / in_features**0.5
    )


def _make(out_features, in_features, seed=0):
    """Build via ``quantize_weight`` — the BF16 conversion route."""
    w = _bf16_weight(out_features, in_features, seed)
    weight, weight_scale = K3Fp8Linear.quantize_weight(w)
    return w, K3Fp8Linear(weight, weight_scale, out_features)


def _make_deferred(out_features, in_features, seed=0):
    """Build the way the checkpoint branch of ``from_linear`` leaves a module:
    raw checkpoint FP8 codes plus the FP32 128x128 grid, scales unprepared."""
    w = _bf16_weight(out_features, in_features, seed)
    weight, weight_scale = per_block_cast_to_fp8(w)
    mod = K3Fp8Linear(weight, weight_scale.float(), out_features)
    mod._weights_transformed = False
    return w, mod


@DEEP_GEMM
@pytest.mark.parametrize("out_features, in_features", SHAPES)
@pytest.mark.parametrize("m", MS)
def test_forward_matches_bf16(m, out_features, in_features):
    """The FP8 weight read must preserve the block-scale numerics."""
    w, mod = _make(out_features, in_features)
    x = torch.randn((m, in_features), device="cuda", dtype=torch.bfloat16)
    out = mod(x)
    _check(out, _ref(x, w), f"m={m} n={out_features} k={in_features}")


@DEEP_GEMM
def test_checkpoint_route_matches_bf16_after_transform():
    """A deferred checkpoint pair must match BF16 once transformed."""
    w, mod = _make_deferred(512, 1024, seed=9)
    mod.transform_weights()
    x = torch.randn((16, 1024), device="cuda", dtype=torch.bfloat16)

    _check(mod(x), _ref(x, w), "checkpoint route")


@DEEP_GEMM
def test_transform_weights_is_idempotent():
    """The loader walks post_load_weights over every module, so a second pass
    must not resmooth already-prepared codes into garbage."""
    w, mod = _make_deferred(512, 1024, seed=4)
    mod.post_load_weights()
    prepared_weight = mod.weight.clone()
    prepared_scale = mod.weight_scale.clone()

    mod.post_load_weights()

    torch.testing.assert_close(
        mod.weight.view(torch.uint8), prepared_weight.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(mod.weight_scale, prepared_scale, rtol=0, atol=0)
    x = torch.randn((32, 1024), device="cuda", dtype=torch.bfloat16)
    _check(mod(x), _ref(x, w), "second transform")


@DEEP_GEMM
def test_forward_uses_the_swap_ab_gemm(monkeypatch):
    """Every call must route through deep_gemm's swap-AB block-scale GEMM."""
    _, mod = _make(512, 1024, seed=8)
    x = torch.randn((64, 1024), device="cuda", dtype=torch.bfloat16)
    mod(x)

    observed_m = []
    real_gemm = torch.ops.trtllm.fp8_swap_ab_gemm

    def spy(*args, **kwargs):
        observed_m.append(args[0].shape[0])
        return real_gemm(*args, **kwargs)

    monkeypatch.setattr(torch.ops.trtllm, "fp8_swap_ab_gemm", spy)
    mod(x)
    mod(x)

    assert observed_m == [64, 64]


@DEEP_GEMM
def test_prepared_scale_is_the_packed_deep_gemm_layout():
    """``quantize_weight`` must return the packed scale the GEMM consumes.

    ``fp8_swap_ab_gemm`` runs with ``disable_ue8m0_cast=True``, so it reads a
    packed UE8M0 scale; the checkpoint's FP32 grid would be misread.
    """
    _, mod = _make(2048, 1024, seed=5)
    assert mod.weight.dtype is torch.float8_e4m3fn
    assert mod.weight_scale.numel() > 0
    assert mod.weight_scale.dtype is torch.int32
    assert mod._weights_transformed


@DEEP_GEMM
def test_deferred_module_reports_untransformed_scales():
    """The checkpoint route must stay flagged until transform_weights runs."""
    _, mod = _make_deferred(512, 1024, seed=6)
    assert not mod._weights_transformed
    assert mod.weight_scale.dtype is torch.float32

    mod.transform_weights()

    assert mod._weights_transformed
    assert mod.weight_scale.dtype is torch.int32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_weight_preparation_returns_only_cute_pair():
    """Both construction routes return exactly FP8 weight + prepared scale."""
    w = _bf16_weight(512, 1024, seed=7)
    assert len(K3Fp8Linear.quantize_weight(w)) == 2

    q, s = per_block_cast_to_fp8(w)
    assert len(K3Fp8Linear._prepare_weights(q, s.float())) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_unfilled_placeholder_raises():
    """An unprepared scale must fail loudly, not silently produce NaN."""
    _, mod = _make_deferred(256, 256, seed=2)
    x = torch.randn((2, 256), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="before its scales were prepared"):
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
