# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rms_norm_gated_token_major catalog entry."""

from typing import Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.rms_norm_gated_token_major import (
    rms_norm_gated_token_major,
)

assert torch.cuda.is_available(), "rms_norm_gated_token_major requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
)

# Kimi K3's KDA: head_dim 128, rms_norm_eps 1e-5, 96 heads (6 / 12 / 24 per rank at TP16 / TP8 / TP4).
HEAD_DIM = 128
EPS = 1e-5


def _ref(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_scale: Optional[torch.Tensor],
    gate_activation: str,
) -> torch.Tensor:
    """fp32 reference: norm before gate, the gate read token-major, one final rounding."""
    rows, n = x.shape
    zf = z.float().reshape(rows, n)
    xf = x.float()
    y = xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    gate = torch.sigmoid(zf)
    if gate_activation == "silu":
        gate = gate * zf
    y = y * gate
    if fp8_scale is None:
        return y.to(x.dtype)
    # The output is first rounded to x.dtype, then scaled by the rounded reciprocal of the scale.
    return (y.to(x.dtype).float() * (1.0 / fp8_scale.float())).to(torch.float8_e4m3fn)


def _gate_view(tokens: int, heads: int, n: int, dtype, extra: int = 0) -> torch.Tensor:
    """[tokens, heads, n] whose (heads, n) block is dense per token; `extra` columns widen the token stride."""
    wide = torch.randn(tokens, heads * n + extra, dtype=dtype, device="cuda")
    return wide[:, : heads * n].view(tokens, heads, n)


def _check(x, z, weight, eps, fp8_scale=None, gate_activation="silu") -> torch.Tensor:
    out = rms_norm_gated_token_major(x, z, weight, eps, fp8_scale, gate_activation)
    expected_dtype = torch.float8_e4m3fn if fp8_scale is not None else x.dtype
    assert out.shape == x.shape and out.dtype == expected_dtype and out.is_contiguous()
    ref = _ref(x, z, weight, eps, fp8_scale, gate_activation)
    if fp8_scale is None:
        # fp32 math and one rounding on both sides; the kernel's reduction order and sigmoid can move
        # an output by one ulp of x.dtype.
        torch.testing.assert_close(out, ref, rtol=2 * torch.finfo(x.dtype).eps, atol=1e-5)
    else:
        # A one-ulp difference before quantization can move an e4m3 code by one step (2^-3 relative).
        torch.testing.assert_close(out.float(), ref.float(), rtol=2.0**-3 + 2.0**-8, atol=2.0**-9)
    return out


def test_kimi_k3_kda_output_gate() -> None:
    # Kimi K3's KDA output gate: o rows [tokens * heads, 128], the full-rank gate projection as a
    # column slice of a wider per-token buffer, sigmoid gate, bf16.
    torch.manual_seed(0)
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    for heads in (6, 12, 24, 96):
        for tokens in (1, 7, 64, 1024, 8192):
            x = torch.randn(tokens * heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            z = _gate_view(tokens, heads, HEAD_DIM, torch.bfloat16, extra=3 * heads * HEAD_DIM)
            _check(x, z, weight, EPS, gate_activation="sigmoid")


def test_silu_gate_and_row_lengths() -> None:
    # The GDN form of the gate, and the multi-row kernel's other row lengths (powers of two up to
    # 256); a plain [rows, 1, N] gate (heads = 1).
    torch.manual_seed(1)
    for n in (32, 64, 128, 256):
        weight = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(48 * 4, n, dtype=torch.bfloat16, device="cuda")
        _check(x, _gate_view(48, 4, n, torch.bfloat16), weight, 1e-6, gate_activation="silu")
        _check(x, _gate_view(192, 1, n, torch.bfloat16), weight, 1e-6, gate_activation="sigmoid")


def test_fallback_paths() -> None:
    # Shapes outside the token-level multi-row path, same result: a row length that is not a power of
    # two or is above 256 runs the generic kernel; a gate whose heads are strided runs the multi-row
    # kernel one head per row, on the gate viewed as [tokens * heads, N].
    torch.manual_seed(2)
    for n, heads in ((96, 4), (512, 2)):
        weight = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(32 * heads, n, dtype=torch.bfloat16, device="cuda")
        _check(x, _gate_view(32, heads, n, torch.bfloat16), weight, EPS, gate_activation="sigmoid")
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(16 * 6, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    strided_heads = torch.randn(16, 6, 2 * HEAD_DIM, dtype=torch.bfloat16, device="cuda")[
        :, :, :HEAD_DIM
    ]
    _check(x, strided_heads, weight, EPS, gate_activation="sigmoid")


def test_fp8_output() -> None:
    # With fp8_scale the output is quantized to e4m3 in the same kernel: rounded to x.dtype first, then
    # multiplied by the rounded reciprocal of the scale.
    torch.manual_seed(3)
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    for scale in (0.5, 1.0, 3.0):
        fp8_scale = torch.tensor(scale, dtype=torch.float32, device="cuda")
        x = torch.randn(64 * 6, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        _check(x, _gate_view(64, 6, HEAD_DIM, torch.bfloat16), weight, EPS, fp8_scale, "sigmoid")


def test_cuda_graph_replay() -> None:
    torch.manual_seed(4)
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(8 * 6, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    z = _gate_view(8, 6, HEAD_DIM, torch.bfloat16, extra=HEAD_DIM)
    rms_norm_gated_token_major(
        x, z, weight, EPS, None, "sigmoid"
    )  # Triton compiles outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = rms_norm_gated_token_major(x, z, weight, EPS, None, "sigmoid")
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        z.copy_(torch.randn_like(z))
        graph.replay()
        assert torch.equal(out, rms_norm_gated_token_major(x, z, weight, EPS, None, "sigmoid"))


def test_rejects_out_of_contract() -> None:
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(8 * 6, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    z = _gate_view(8, 6, HEAD_DIM, torch.bfloat16)
    bad_calls = {
        "unknown gate": ((x, z, weight, EPS, None, "relu"), ValueError),
        "row count": ((x[:47], z, weight, EPS, None, "sigmoid"), AssertionError),
        "row length": (
            (x[:, :64].contiguous(), z, weight[:64], EPS, None, "sigmoid"),
            AssertionError,
        ),
    }
    for name, (args, error) in bad_calls.items():
        try:
            rms_norm_gated_token_major(*args)
        except error:
            continue
        raise AssertionError(f"{name}: the op accepted an out-of-contract call")
