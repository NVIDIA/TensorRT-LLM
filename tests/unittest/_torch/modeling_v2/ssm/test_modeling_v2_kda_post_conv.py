# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the kda_post_conv catalog entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.kda_post_conv import kda_post_conv

assert torch.cuda.is_available(), "kda_post_conv requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
)

HEAD_DIM = 128  # Kimi K3's KDA head_dim
EPS = 1e-6


def _ref(packed: torch.Tensor, heads: int, eps: float = EPS):
    """fp32 reference: q / sqrt(sum(q^2) + eps) per head (k alike), v as is; token-major [1, T, H, D]."""
    tokens = packed.shape[1]
    q, k, v = packed.float().view(3, heads, HEAD_DIM, tokens).permute(0, 3, 1, 2)

    def norm(t):
        return t / torch.sqrt(t.pow(2).sum(-1, keepdim=True) + eps)

    return (
        norm(q).to(packed.dtype).unsqueeze(0),
        norm(k).to(packed.dtype).unsqueeze(0),
        v.to(packed.dtype).unsqueeze(0),
    )


def _check(packed: torch.Tensor, heads: int) -> None:
    q, k, v = kda_post_conv(packed, heads, HEAD_DIM)
    rq, rk, rv = _ref(packed, heads)
    shape = (1, packed.shape[1], heads, HEAD_DIM)
    for t in (q, k, v):
        assert t.shape == shape and t.dtype == packed.dtype and t.is_contiguous()
    # fp32 normalization and one rounding on both sides; the reduction order can move a value by
    # one ulp of the dtype.
    rtol = 1e-5 if packed.dtype == torch.float32 else 2 * torch.finfo(packed.dtype).eps
    torch.testing.assert_close(q, rq, rtol=rtol, atol=1e-6)
    torch.testing.assert_close(k, rk, rtol=rtol, atol=1e-6)
    assert torch.equal(v, rv), "v must be copied, not computed"


def test_kimi_k3_prefill_shapes() -> None:
    # Kimi K3's KDA prefill: 6 heads (TP16) and 24 (TP4), head_dim 128, bf16, decode- through
    # prefill-sized token counts (16-token blocks and a ragged tail).
    torch.manual_seed(0)
    for heads in (6, 24):
        rows = 3 * heads * HEAD_DIM
        for tokens in (1, 15, 16, 300, 4096):
            packed = torch.randn(rows, tokens, dtype=torch.bfloat16, device="cuda") * 3
            _check(packed, heads)


def test_other_dtypes_and_heads() -> None:
    torch.manual_seed(1)
    for dtype in (torch.float16, torch.float32):
        packed = torch.randn(3 * 4 * HEAD_DIM, 77, dtype=dtype, device="cuda")
        _check(packed, 4)


def test_zero_and_tiny_rows() -> None:
    # An all-zero q / k head stays zero (eps keeps the norm finite); a tiny one is dominated by eps.
    packed = torch.zeros(3 * 6 * HEAD_DIM, 4, dtype=torch.bfloat16, device="cuda")
    packed[:, 1] = 1e-4
    q, k, _ = kda_post_conv(packed, 6, HEAD_DIM)
    assert torch.equal(q[0, 0], torch.zeros_like(q[0, 0]))
    assert torch.equal(k[0, 0], torch.zeros_like(k[0, 0]))
    assert bool(torch.isfinite(q).all() and torch.isfinite(k).all())
    _check(packed, 6)


def test_zero_tokens() -> None:
    packed = torch.empty(3 * 6 * HEAD_DIM, 0, dtype=torch.bfloat16, device="cuda")
    q, k, v = kda_post_conv(packed, 6, HEAD_DIM)
    assert q.shape == k.shape == v.shape == (1, 0, 6, HEAD_DIM)


def test_rejects_out_of_contract() -> None:
    packed = torch.randn(3 * 6 * HEAD_DIM, 32, dtype=torch.bfloat16, device="cuda")
    bad = {
        "row count": (packed[:-1], 6),
        "heads": (packed, 5),
        "transposed": (packed.t().contiguous().t(), 6),
        "3-D": (packed.unsqueeze(0), 6),
    }
    for name, (arg, heads) in bad.items():
        try:
            kda_post_conv(arg, heads, HEAD_DIM)
        except ValueError:
            continue
        raise AssertionError(f"{name}: the call accepted an out-of-contract input")
