# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the kda_post_conv catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.kda_post_conv import (
    kda_post_conv,
)

__extra_import_path__ = ["../.."]
from _validating import validating

assert torch.cuda.is_available(), "kda_post_conv requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

HEAD_DIM = 128  # Kimi K3's KDA head_dim


def _check(packed: torch.Tensor, heads: int) -> None:
    with validating(kda_post_conv):
        q, k, v = kda_post_conv(packed, heads, HEAD_DIM)
    rq, rk, rv = kda_post_conv.reference(packed, heads, HEAD_DIM)
    shape = (1, packed.shape[1], heads, HEAD_DIM)
    for t in (q, k, v):
        assert t.shape == shape and t.dtype == packed.dtype and t.is_contiguous()
    # fp32 normalization and one rounding on both sides; the reduction order can move a value by
    # one ulp of the dtype.
    rtol = 1e-5 if packed.dtype == torch.float32 else 2 * torch.finfo(packed.dtype).eps
    torch.testing.assert_close(q, rq, rtol=rtol, atol=1e-6)
    torch.testing.assert_close(k, rk, rtol=rtol, atol=1e-6)
    assert torch.equal(v, rv), "v must be copied, not computed"


@pytest.mark.parametrize("cell", kda_post_conv.CELLS, ids=[c.why[:40] for c in kda_post_conv.CELLS])
@pytest.mark.parametrize("token_major", [False, True])
def test_kimi_k3_prefill_shapes(cell, token_major: bool) -> None:
    # Kimi K3's KDA prefill: 6 heads (TP16) and 24 (TP4), head_dim 128, bf16, decode- through
    # prefill-sized token counts (16-token blocks and a ragged tail).
    spec = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    packed = (
        torch.randn(
            3 * spec["heads"] * HEAD_DIM,
            spec["tokens"],
            dtype=spec["dtype"],
            generator=gen,
            device="cuda",
        )
        * 3
    )
    if spec["kind"] == "tiny":
        packed.zero_()
        packed[:, 1] = 1e-4
    if token_major:
        packed = packed.t().contiguous().t()
    _check(packed, spec["heads"])


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
        "non-unit strides": (
            torch.empty_strided(packed.shape, (128, 2), dtype=packed.dtype, device="cuda"),
            6,
        ),
        "3-D": (packed.unsqueeze(0), 6),
    }
    for name, (arg, heads) in bad.items():
        try:
            kda_post_conv(arg, heads, HEAD_DIM)
        except ValueError:
            continue
        raise AssertionError(f"{name}: the call accepted an out-of-contract input")
