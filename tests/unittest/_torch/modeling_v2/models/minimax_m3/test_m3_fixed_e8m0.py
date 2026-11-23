# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The UE8M0 zero code in the family library's fixed low-M MXFP8 GEMM (``minimax_m3_mxfp8_low_m_gemm``).

An activation block whose largest magnitude is at most 448 * 2^-127 takes the scale byte 0 (2^-127, an fp32
denormal). Its reciprocal must not be the flush-to-zero +inf, which turns the block into NaN or +-448 values: the
block quantizes to zero, exactly like an all-zero block, so the GEMM output is bit for bit the one with that block
zeroed, and finite.

The family's cuBLASLt MXFP8 configuration queries: ``minimax_m31_mxfp8_cublaslt_heuristic``'s configuration for
(M, N, K) passes ``minimax_m31_mxfp8_cublaslt_algo_check`` with the same workspace bytes and runs in
``minimax_m31_mxfp8_cublaslt_gemm`` within the FP32-accumulation band of the dequantized product.
"""

import pytest
import torch
from _m3_mxfp8_operands import mxfp8

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._utils import get_sm_version

_LOW_M_GEMM = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="the family library's low-M MXFP8 GEMM runs on SM100 and SM103 only",
)

# (N, K) -> (warps, cluster_k, row_tiles): configurations the op is built for.
_CONFIGS = {(2560, 6144): (4, 4, 2), (6144, 3072): (4, 2, 2)}
# Values whose block maximum is below 448 * 2^-127 (~2.6e-36): a BF16 normal and a BF16 subnormal.
_TINY = (1.0e-37, 2.0e-39)


@_LOW_M_GEMM
@pytest.mark.parametrize("shape", sorted(_CONFIGS), ids=lambda s: f"N{s[0]}_K{s[1]}")
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("tiny", _TINY)
def test_an_all_tiny_block_quantizes_to_zero(shape, rows, tiny):
    n, k = shape
    warps, cluster_k, row_tiles = _CONFIGS[shape]
    g = torch.Generator(device="cuda").manual_seed(n + k + rows)
    w, wsf = mxfp8(n, k, g)
    x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
    # Blocks 0 and 5 of every row, and the last block of row 0, are all tiny (both signs).
    signs = torch.where(torch.rand(rows, 32, generator=g, device="cuda") < 0.5, -1.0, 1.0)
    tiny_x = x.clone()
    zero_x = x.clone()
    for block in (0, 5):
        tiny_x[:, block * 32 : (block + 1) * 32] = (signs * tiny).to(torch.bfloat16)
        zero_x[:, block * 32 : (block + 1) * 32] = 0
    tiny_x[0, -32:] = tiny
    zero_x[0, -32:] = 0
    assert bool((tiny_x[:, :32] != 0).all()), "the tiny values must be representable in BF16"
    out_tiny = family_ops().minimax_m3_mxfp8_low_m_gemm(
        tiny_x, w, wsf, warps, cluster_k, row_tiles, False
    )
    out_zero = family_ops().minimax_m3_mxfp8_low_m_gemm(
        zero_x, w, wsf, warps, cluster_k, row_tiles, False
    )
    assert bool(torch.isfinite(out_tiny).all())
    assert torch.equal(out_tiny.view(torch.int16), out_zero.view(torch.int16))


#: The serving workspace limit of the pinned cuBLASLt GEMM (kWorkspaceBytes).
WORKSPACE_BYTES = 1 << 20


def _dequant(values, swizzled_sf, rows, k):
    """values [rows, k] E4M3 times their UE8M0 block scales (un-swizzling mxfp8's 128x4 layout)."""
    cols = k // 32
    padded_cols = (cols + 3) // 4 * 4
    r = torch.arange(rows, device=values.device).view(-1, 1)
    c = torch.arange(cols, device=values.device).view(1, -1)
    index = (
        (c % 4)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )
    sf = swizzled_sf.view(torch.uint8).reshape(-1)[index].to(torch.int32)
    scale = torch.exp2((sf - 127).to(torch.float32)).repeat_interleave(32, dim=1)
    return values.to(torch.float32) * scale


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the MXFP8 cuBLASLt GEMM runs on SM10x",
)
@pytest.mark.parametrize("m,n,k", [(1024, 6144, 4096), (4096, 6144, 1536)], ids=str)
def test_the_cublaslt_heuristic_configuration_checks_and_runs(m, n, k) -> None:
    ops = family_ops()
    config = list(ops.minimax_m31_mxfp8_cublaslt_heuristic(m, n, k))
    assert len(config) == 10, config
    algo, workspace = config[:9], config[9]
    assert 0 <= workspace <= WORKSPACE_BYTES
    assert ops.minimax_m31_mxfp8_cublaslt_algo_check(m, n, k, algo) == workspace
    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    w, w_sf = mxfp8(n, k, g)
    act, act_sf = mxfp8(m, k, g)
    out = ops.minimax_m31_mxfp8_cublaslt_gemm(act, act_sf, w, w_sf, algo)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    ref = _dequant(act, act_sf, m, k) @ _dequant(w, w_sf, n, k).t()
    # One bf16 rounding of an FP32 accumulation: within 2^-7 of each row's largest magnitude.
    row_max = ref.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    assert bool(((out.float() - ref).abs() <= row_max * 2.0**-7).all())
