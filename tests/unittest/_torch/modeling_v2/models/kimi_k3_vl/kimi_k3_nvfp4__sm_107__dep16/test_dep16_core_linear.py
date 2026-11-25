# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's FP8 weight read against the family ``ops/gemm/sm107_linear.py``: bitwise.

The checkpoint codes and grid loaded into an ``Fp8BlockScaleLinear``, swapped into dep16's FP8
weight-read projection (``_Fp8BlockScaleWeightReadLinear``, as dep16 swaps its attention
projections) and post-loaded; the primitive loads the same full tensors itself. dep16's weight read
serves the low-M rows with the K3 low-M GEMM (``fp8w_lowm_gemm``), so the two agree bitwise where
the weight read runs ``layers.fp8_block_scale_gemm``.

The primitive's own checks against main's ``Linear`` (shards, BF16, parameter names) are in
``ops/gemm/test_core_linear.py``.
"""

import pytest
import torch
from utils.util import getSMVersion

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.gemm import sm107_linear as linear

# Row counts across the low-M / MXFP8 (DeepGEMM) cut-overs and the K3 tuning buckets.
ROWS = (1, 3, 8, 16, 31, 64, 100, 128, 129, 192, 256, 300, 512, 1000)

requires_block_fp8 = pytest.mark.skipif(
    getSMVersion() not in (100, 103, 107)
    or (getSMVersion() == 107 and not IS_CUTLASS_DSL_RUBIN_AVAILABLE),
    reason="needs a supported block-scaled FP8 GPU and toolchain",
)


def _checkpoint_fp8(n, k, seed):
    """E4M3 codes and FP32 (non power-of-two) 128x128 block scales of a random ``[n, k]`` weight."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(n, k, device="cuda", generator=gen) * 0.02
    blocks = torch.nn.functional.pad(weight, (0, -k % 128, 0, -n % 128)).view(
        -(-n // 128), 128, -(-k // 128), 128
    )
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-8) / 448.0
    expanded = scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    return (weight / expanded).to(torch.float8_e4m3fn), scale


def _weight_read(codes, scale):
    """dep16's decode chain: the checkpoint loaded into an unprepared ``Fp8BlockScaleLinear``,
    swapped to the weight-read projection, which prepares the codes and scales itself."""
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__dep16.modeling import (  # noqa: E501
        _Fp8BlockScaleWeightReadLinear,
    )

    n, k = codes.shape
    loader = linear.Fp8BlockScaleLinear(k, n, device="cuda")
    loader.load(codes, scale)
    proj = _Fp8BlockScaleWeightReadLinear.from_linear(loader)
    proj.post_load_weights()
    return proj


def _runs_k3_low_m(proj, x):
    """Whether dep16's weight read serves ``x`` with the K3 low-M GEMM (its ``forward`` dispatch)."""
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__dep16.modeling import (  # noqa: E501
        _rubin_mxfp8_is_available,
    )

    scale = proj.lowm_weight_scale if _rubin_mxfp8_is_available() else proj.weight_scale
    return scale is not None and proj._lowm_gemm_applies(x, scale)


# K3's checkpoint FP8 projection shapes (q_a + kv_a fused, q_b / kv_b / o / g heads, KDA q/k/v/g/o).
FP8_SHAPES = [(2112, 7168), (1152, 1536), (2304, 512), (7168, 1024), (512, 7168), (4096, 7168)]


@requires_block_fp8
@pytest.mark.parametrize("n,k", FP8_SHAPES)
@torch.inference_mode()
def test_fp8_block_linear_matches_k3_weight_read(n, k):
    codes, scale = _checkpoint_fp8(n, k, seed=n + k)
    ref = _weight_read(codes.clone(), scale.clone())
    prim = linear.Fp8BlockScaleLinear(k, n, device="cuda")
    prim.load(codes, scale)
    # Without the generic low-M GEMM: dep16's weight read never prepares it.
    prim.prepare(low_m=False)
    assert torch.equal(prim.weight.view(torch.uint8), ref.weight.view(torch.uint8))
    assert torch.equal(prim.weight_scale, ref.weight_scale)
    assert prim.low_m_max == 0
    compared = 0
    for m in ROWS:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        if _runs_k3_low_m(ref, x):
            continue
        assert torch.equal(prim(x), ref(x)), m
        compared += 1
    assert compared > 0
