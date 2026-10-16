# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the situ_and_mul catalog entry."""

from typing import Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.situ_and_mul import (
    situ_and_mul,
)

assert torch.cuda.is_available(), "situ_and_mul requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
)

# Kimi K3's activation_situ_beta and activation_situ_linear_beta.
K3_BETA, K3_LINEAR_BETA = 4.0, 25.0


def _ref(x: torch.Tensor, beta: float, linear_beta: Optional[float]) -> torch.Tensor:
    """fp32 reference from native torch ops: SiTU on the gate half times the (soft-capped) up half."""
    d = x.shape[-1] // 2
    gate = x[:, :d].float()
    up = x[:, d:].float()
    situ = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ * up).to(x.dtype)


def _check(x: torch.Tensor, beta: float, linear_beta: Optional[float]) -> torch.Tensor:
    out = situ_and_mul(x, beta, linear_beta)
    assert out.shape == (x.shape[0], x.shape[1] // 2) and out.dtype == x.dtype
    assert out.is_contiguous() and out.device == x.device
    ref = _ref(x, beta, linear_beta)
    # Both sides compute in fp32 and round once to x.dtype. The kernel's libdevice tanh can differ from
    # torch.tanh in the last fp32 bits: for a 16-bit dtype that moves the rounded result by at most one
    # ulp, for fp32 output it shows directly as a few fp32 ulps.
    rtol = 1e-5 if x.dtype == torch.float32 else 2 * torch.finfo(x.dtype).eps
    torch.testing.assert_close(out, ref, rtol=rtol, atol=1e-5)
    return out


def test_kimi_k3_shapes() -> None:
    # Per-rank [gate | up] widths of Kimi K3's two SiTU MLPs: the shared experts (intermediate
    # 2 x 3072) and the dense layer (33792), at TP16 and TP4; decode- through prefill-sized rows.
    torch.manual_seed(0)
    for width in (768, 3072, 4224, 16896):
        for num_tokens in (1, 8, 64, 2048):
            x = torch.randn(num_tokens, width, dtype=torch.bfloat16, device="cuda") * 4.0
            _check(x, K3_BETA, K3_LINEAR_BETA)


def test_betas() -> None:
    # Asymmetric caps, and linear_beta=None (the up half passes through unchanged).
    torch.manual_seed(1)
    x = torch.randn(33, 6144, dtype=torch.bfloat16, device="cuda") * 4.0
    for beta, linear_beta in ((2.5, 7.0), (1.0, None), (K3_BETA, None)):
        _check(x, beta, linear_beta)


def test_soft_caps_saturate() -> None:
    # Far beyond the caps both halves saturate: gate -> beta * sigmoid(gate) -> beta (large positive)
    # or 0 (large negative), up -> +-linear_beta.
    x = torch.zeros(4, 512, dtype=torch.float32, device="cuda")
    x[0, :256], x[0, 256:] = 1e4, 1e4
    x[1, :256], x[1, 256:] = 1e4, -1e4
    x[2, :256], x[2, 256:] = -1e4, 1e4
    x[3, :256], x[3, 256:] = 0.0, 1e4
    out = situ_and_mul(x, K3_BETA, K3_LINEAR_BETA)
    expected = torch.tensor(
        [K3_BETA * K3_LINEAR_BETA, -K3_BETA * K3_LINEAR_BETA, 0.0, 0.0], device="cuda"
    )
    torch.testing.assert_close(out, expected.view(4, 1).expand(4, 256))
    _check(x, K3_BETA, K3_LINEAR_BETA)


def test_dtypes_and_strided_rows() -> None:
    # The output dtype follows the input's; rows may be strided (a column slice of a wider buffer)
    # as long as the last dimension is dense.
    torch.manual_seed(3)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        x = torch.randn(17, 1536, dtype=dtype, device="cuda") * 3.0
        _check(x, K3_BETA, K3_LINEAR_BETA)
    wide = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    rows = wide[:, :768]
    assert not rows.is_contiguous() and rows.stride(-1) == 1
    _check(rows, K3_BETA, K3_LINEAR_BETA)


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
