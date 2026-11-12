# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the fused_kda_post_conv entry: every cell of ``fused_kda_post_conv.CELLS`` against
its fp64 reference (Q/K within one bf16 ulp of the per-head scale, V bitwise), plus the
chunk-padded form with beta."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.fused_kda_post_conv import (  # noqa: E501
    fused_kda_post_conv,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating

HEAD_DIM = 128


def _driven_here() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in {arch.value for arch in fused_kda_post_conv.ARCHS}


pytestmark = pytest.mark.skipif(not _driven_here(), reason="fused_kda_post_conv.ARCHS")


def _packed(heads: int, tokens: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(3 * heads * HEAD_DIM, tokens, generator=gen, device="cuda").bfloat16()


def _check(out, ref, context):
    assert len(out) == len(ref), context
    for name, got, want in zip("qkvb", out, ref):
        assert got.shape == want.shape and got.dtype == want.dtype, (context, name)
        assert got.is_contiguous(), (context, name)
        if name in "qk":
            # fp32 norm in the kernel vs fp64 here, each rounded once to bf16: an element can
            # land one bf16 ulp of its own value away, at most 2**-7 of the head's largest
            # magnitude, i.e. two ULP units of the row scale (1.27 measured on GB300).
            assert_within_ulp(got, want, element_ulp=2.0, rms_ulp=0.25)
        else:
            assert torch.equal(got, want), (context, name)


@pytest.mark.parametrize(
    "cell", fused_kda_post_conv.CELLS, ids=[c.why[:60] for c in fused_kda_post_conv.CELLS]
)
@torch.no_grad()
def test_cells(cell) -> None:
    heads, tokens = cell.spec["heads"], cell.spec["tokens"]
    packed = _packed(heads, tokens, zlib.crc32(cell.why.encode()) % 2**31)
    with validating(fused_kda_post_conv):
        out = fused_kda_post_conv(packed, heads, HEAD_DIM)
    _check(out, fused_kda_post_conv.reference(packed, heads, HEAD_DIM), cell.why)


@pytest.mark.parametrize("tokens", [1, 63, 65, 193])
@torch.no_grad()
def test_chunk_padded_with_beta(tokens: int) -> None:
    """``kda_prepad_prefill``'s form: Q/K/V and beta (a column view of the [f_a | b]
    output) zero-padded to the 64-token chunk."""
    heads = 6
    packed = _packed(heads, tokens, tokens)
    projection = torch.randn(1, tokens, HEAD_DIM + 8, device="cuda").bfloat16()
    beta = projection[..., HEAD_DIM : HEAD_DIM + heads]
    padded = -(-tokens // 64) * 64
    with validating(fused_kda_post_conv):
        out = fused_kda_post_conv(packed, heads, HEAD_DIM, padded_tokens=padded, beta=beta)
    ref = fused_kda_post_conv.reference(packed, heads, HEAD_DIM, padded_tokens=padded, beta=beta)
    _check(out, ref, f"tokens={tokens}")
