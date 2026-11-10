# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the situ_and_mul catalog entry."""

import zlib
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.activation.situ_and_mul import (
    situ_and_mul,
)

__extra_import_path__ = ["../.."]
from _validating import validating

assert torch.cuda.is_available(), "situ_and_mul requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

# Kimi K3's activation_situ_beta and activation_situ_linear_beta.
K3_BETA, K3_LINEAR_BETA = 4.0, 25.0


def _check(x: torch.Tensor, beta: float, linear_beta: Optional[float]) -> torch.Tensor:
    with validating(situ_and_mul):
        out = situ_and_mul(x, beta, linear_beta)
    assert out.shape == (x.shape[0], x.shape[1] // 2) and out.dtype == x.dtype
    assert out.is_contiguous() and out.device == x.device
    ref = situ_and_mul.reference(x, beta, linear_beta)
    # Both sides compute in fp32 and round once to x.dtype. The kernel's libdevice tanh can differ from
    # torch.tanh in the last fp32 bits: for a 16-bit dtype that moves the rounded result by at most one
    # ulp, for fp32 output it shows directly as a few fp32 ulps.
    rtol = 1e-5 if x.dtype == torch.float32 else 2 * torch.finfo(x.dtype).eps
    torch.testing.assert_close(out, ref, rtol=rtol, atol=1e-5)
    return out


@pytest.mark.parametrize("cell", situ_and_mul.CELLS, ids=[c.why[:40] for c in situ_and_mul.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows, width = spec["shape"]
    x = torch.randn(
        rows, spec.get("row_width", width), generator=gen, dtype=spec["dtype"], device="cuda"
    ) * spec.get("scale", 4.0)
    x = x[:, :width]
    if spec.get("saturated"):
        # Far beyond the caps both halves saturate: gate -> beta * sigmoid(gate) -> beta (large positive)
        # or 0 (large negative), up -> +-linear_beta.
        x[0, :256], x[0, 256:] = 1e4, 1e4
        x[1, :256], x[1, 256:] = 1e4, -1e4
        x[2, :256], x[2, 256:] = -1e4, 1e4
        x[3, :256], x[3, 256:] = 0.0, 1e4
    if "row_width" in spec:
        assert not x.is_contiguous() and x.stride(-1) == 1
    out = _check(x, spec["beta"], spec["linear_beta"])
    if spec.get("saturated"):
        expected = torch.tensor(
            [K3_BETA * K3_LINEAR_BETA, -K3_BETA * K3_LINEAR_BETA, 0.0, 0.0], device="cuda"
        )
        torch.testing.assert_close(out, expected.view(4, 1).expand(4, 256))


def test_cuda_graph_replay() -> None:
    torch.manual_seed(4)
    x = torch.randn(8, 768, dtype=torch.bfloat16, device="cuda")
    situ_and_mul(x, K3_BETA, K3_LINEAR_BETA)  # Triton compiles on the first call, outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = situ_and_mul(x, K3_BETA, K3_LINEAR_BETA)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        assert torch.equal(out, situ_and_mul(x, K3_BETA, K3_LINEAR_BETA))


def test_rejects_out_of_contract() -> None:
    x = torch.randn(8, 1024, dtype=torch.bfloat16, device="cuda")
    bad_calls = {
        "strided last dim": (x[:, ::2], ValueError),
        "odd width": (x[:, :1023].contiguous(), AssertionError),
        "3-D input": (x.view(2, 4, 1024), ValueError),
    }
    for name, (arg, error) in bad_calls.items():
        try:
            situ_and_mul(arg, K3_BETA, K3_LINEAR_BETA)
        except error:
            continue
        raise AssertionError(f"{name}: the op accepted an out-of-contract call")
