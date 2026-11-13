# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the activation quantize of torch.ops.trtllm.minimax_m3_mxfp8_low_m_gemm (SM100 / SM103): a 32-column
block with 0 < amax <= 448 * 2^-127 takes UE8M0 byte 0 and contributes zeros, as an all-zero block does, never a
NaN."""

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["."]
from _m3_mxfp8_operands import mxfp8  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="minimax_m3_mxfp8_low_m_gemm runs on SM100 / SM103 only",
)

# (N, K, warps, cluster_k, row_tiles): the configurations the op is built for, each for 1..8 rows.
CONFIGS = [
    (2560, 6144, 4, 4, 2),
    (6144, 2048, 8, 2, 4),
    (6144, 6144, 4, 4, 1),
    (6144, 3072, 4, 2, 2),
]


@pytest.mark.parametrize("n, k, warps, cluster_k, row_tiles", CONFIGS)
@pytest.mark.parametrize("amax", [1e-40, 1e-38, 2e-36])
def test_a_byte_0_block_contributes_zeros(n, k, warps, cluster_k, row_tiles, amax):
    g = torch.Generator(device="cuda").manual_seed(n + k)
    w, wsf = mxfp8(n, k, g)
    x = torch.randn(4, k, generator=g, device="cuda").to(torch.bfloat16)
    tiny, zeroed = x.clone(), x.clone()
    tiny[1, 64:96] = ((torch.rand(32, generator=g, device="cuda") * 2 - 1) * amax).to(
        torch.bfloat16
    )
    tiny[1, 64] = amax
    zeroed[1, 64:96] = 0
    op = torch.ops.trtllm.minimax_m3_mxfp8_low_m_gemm
    out = op(tiny, w, wsf, warps, cluster_k, row_tiles, False)
    assert torch.isfinite(out).all()
    assert torch.equal(out, op(zeroed, w, wsf, warps, cluster_k, row_tiles, False))
