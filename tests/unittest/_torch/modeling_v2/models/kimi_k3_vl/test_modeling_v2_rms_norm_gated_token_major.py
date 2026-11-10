# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rms_norm_gated_token_major catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.rms_norm_gated_token_major import (
    rms_norm_gated_token_major,
)

__extra_import_path__ = ["../.."]
from _validating import validating

assert torch.cuda.is_available(), "rms_norm_gated_token_major requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

# Kimi K3's KDA: head_dim 128, rms_norm_eps 1e-5, 96 heads (6 / 12 / 24 per rank at TP16 / TP8 / TP4).
HEAD_DIM = 128
EPS = 1e-5


def _gate_view(tokens: int, heads: int, n: int, dtype, extra: int = 0) -> torch.Tensor:
    """[tokens, heads, n] whose (heads, n) block is dense per token; `extra` columns widen the token stride."""
    wide = torch.randn(tokens, heads * n + extra, dtype=dtype, device="cuda")
    return wide[:, : heads * n].view(tokens, heads, n)


def _check(x, z, weight, eps, fp8_scale=None, gate_activation="silu") -> torch.Tensor:
    with validating(rms_norm_gated_token_major):
        out = rms_norm_gated_token_major(x, z, weight, eps, fp8_scale, gate_activation)
    expected_dtype = torch.float8_e4m3fn if fp8_scale is not None else x.dtype
    assert out.shape == x.shape and out.dtype == expected_dtype and out.is_contiguous()
    ref = rms_norm_gated_token_major.reference(x, z, weight, eps, fp8_scale, gate_activation)
    if fp8_scale is None:
        # fp32 math and one rounding on both sides; the kernel's reduction order and sigmoid can move
        # an output by one ulp of x.dtype.
        torch.testing.assert_close(out, ref, rtol=2 * torch.finfo(x.dtype).eps, atol=1e-5)
    else:
        # A one-ulp difference before quantization can move an e4m3 code by one step (2^-3 relative).
        torch.testing.assert_close(out.float(), ref.float(), rtol=2.0**-3 + 2.0**-8, atol=2.0**-9)
    return out


@pytest.mark.parametrize(
    "cell",
    rms_norm_gated_token_major.CELLS,
    ids=[c.why[:40] for c in rms_norm_gated_token_major.CELLS],
)
def test_certified_cells(cell) -> None:
    spec = cell.spec
    torch.manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    tokens, heads, n, dtype = spec["tokens"], spec["heads"], spec["width"], spec["dtype"]
    weight = torch.randn(n, dtype=dtype, device="cuda")
    x = torch.randn(tokens * heads, n, dtype=dtype, device="cuda")
    if spec.get("head_strided"):
        z = torch.randn(tokens, heads, 2 * n, dtype=dtype, device="cuda")[:, :, :n]
    else:
        z = _gate_view(tokens, heads, n, dtype, extra=spec.get("extra", 0))
    fp8_scale = None
    if "fp8_scale" in spec:
        fp8_scale = torch.tensor(spec["fp8_scale"], dtype=torch.float32, device="cuda")
    _check(x, z, weight, spec["eps"], fp8_scale, spec["gate_activation"])


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


def test_guard_refuses_per_column_fp8_scale() -> None:
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(8 * 6, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    z = _gate_view(8, 6, HEAD_DIM, torch.bfloat16)
    fp8_scale = torch.linspace(0.5, 3.0, HEAD_DIM, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="exactly one element"):
        with validating(rms_norm_gated_token_major):
            rms_norm_gated_token_major(x, z, weight, EPS, fp8_scale, "sigmoid")
