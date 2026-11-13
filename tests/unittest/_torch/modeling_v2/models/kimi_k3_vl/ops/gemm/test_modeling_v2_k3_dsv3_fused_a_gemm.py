# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3 family's dsv3_fused_a_gemm entry: every cell against the fp32 reference, and bit for bit
the shared op it wraps."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm.dsv3_fused_a_gemm import (
    dsv3_fused_a_gemm,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402

assert torch.cuda.is_available(), "dsv3_fused_a_gemm requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)


@pytest.mark.parametrize(
    "cell", dsv3_fused_a_gemm.CELLS, ids=[c.why[:48] for c in dsv3_fused_a_gemm.CELLS]
)
def test_certified_cells(cell):
    spec = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = torch.randn(spec["m"], spec["k"], generator=gen, device="cuda").to(torch.bfloat16)
    w = (0.02 * torch.randn(spec["n"], spec["k"], generator=gen, device="cuda")).to(torch.bfloat16)
    with validating(dsv3_fused_a_gemm):
        y = dsv3_fused_a_gemm(x, w)
    assert y.shape == (spec["m"], spec["n"]) and y.dtype == torch.bfloat16
    ref = dsv3_fused_a_gemm.reference(x, w)
    # One bf16 rounding of an fp32 sum taken in another order: at most one bf16 ulp apart (2^-7 relative).
    scale = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
    assert ((y.float() - ref.float()).abs() / scale).max().item() <= 2.0**-7
    raw = torch.ops.trtllm.dsv3_fused_a_gemm_op(x, w.t(), None, None)
    assert torch.equal(y, raw)
