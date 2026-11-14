# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3 family's dsv3_router_gemm entry: every cell against the fp64 reference, and bit for bit
the shared op it wraps."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm.dsv3_router_gemm import (
    dsv3_router_gemm,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402

assert torch.cuda.is_available(), "dsv3_router_gemm requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)


@pytest.mark.parametrize(
    "cell", dsv3_router_gemm.CELLS, ids=[c.why[:48] for c in dsv3_router_gemm.CELLS]
)
def test_certified_cells(cell):
    spec = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = torch.randn(spec["m"], spec["k"], generator=gen, device="cuda").to(torch.bfloat16)
    w = (0.02 * torch.randn(spec["n"], spec["k"], generator=gen, device="cuda")).to(torch.bfloat16)
    with validating(dsv3_router_gemm):
        y = dsv3_router_gemm(x, w)
    assert y.shape == (spec["m"], spec["n"]) and y.dtype == torch.float32
    ref = dsv3_router_gemm.reference(x, w)
    # The products are exact in fp32; only the fp32 summation order differs (cuBLAS picks its own split above 16
    # rows). Rounding error of a K-term fp32 sum grows like sqrt(K) * 2^-24 * sum |x_k w_k|; the gate allows 8x that
    # per element. Measured worst on sm_100: 3.2e-5 absolute at M=2048, about 1% of the gate.
    with torch.no_grad():
        magnitude = x.double().abs() @ w.double().abs().t()
    bound = 8.0 * math.sqrt(spec["k"]) * 2.0**-24 * magnitude
    assert ((y.double() - ref.double()).abs() <= bound).all()
    raw = torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)
    assert torch.equal(y, raw)


def test_rejects_strided_weight():
    x = torch.randn(4, 7168, device="cuda").to(torch.bfloat16)
    w = torch.randn(896, 2 * 7168, device="cuda").to(torch.bfloat16)[:, :7168]
    with pytest.raises(AssertionError), validating(dsv3_router_gemm):
        dsv3_router_gemm(x, w)
