# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the inkling2_residual_mix entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.inkling2.ops.residual.inkling2_residual_mix import (
    inkling2_residual_mix as op,
)
from tensorrt_llm._torch.modeling_v2.models.inkling2.ops.residual.inkling2_residual_mix import (
    normalize_block,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "inkling2_residual_mix requires a CUDA device"

HIDDEN = 4608
EPS = 1e-6


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, s, q = spec["tokens"], spec["blocks"], spec["queries"]
    # Block states at different scales: the normalization, not the raw scale, sets their weight.
    raw = torch.randn(s, t, HIDDEN, generator=g, device="cuda") * torch.logspace(
        -1, 1, s, device="cuda"
    ).view(s, 1, 1)
    blocks = torch.stack([normalize_block(raw[i].to(torch.bfloat16)) for i in range(s)])
    partial = (3.0 * torch.randn(t, HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    # Queries with norms like the checkpoint's, so the softmax is neither flat nor one-hot.
    queries = 0.05 * torch.randn(q, HIDDEN, generator=g, device="cuda")
    weights = tuple(
        (1.0 + 0.2 * torch.randn(HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
        for _ in range(q)
    )
    return blocks, partial, queries, weights, EPS


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = op(*args)
    ref = op.reference(*args)
    assert out.shape == (cell.spec["queries"], cell.spec["tokens"], HIDDEN)
    # fp32 against fp64, one bf16 rounding at the end: at most one bf16 ulp per element.
    assert_within_ulp(
        out.reshape(-1, HIDDEN), ref.reshape(-1, HIDDEN), element_ulp=1.0, rms_ulp=0.1
    )


def test_values_are_the_normalized_states() -> None:
    """Scaling the partial block does not change the mix: it enters normalized."""
    args = list(_build(dict(blocks=3, queries=1, tokens=5), 7))
    out = op(*args)
    args[1] = (args[1].float() * 8.0).to(torch.bfloat16)
    torch.testing.assert_close(op(*args), out, rtol=0, atol=0)


def test_fp32_partial_block_keeps_its_low_bits() -> None:
    """The decoder hands the partial block over in fp32; the mix must not round it to bf16."""
    blocks, partial, queries, weights, eps = _build(dict(blocks=3, queries=2, tokens=5), 11)
    g = torch.Generator(device="cuda").manual_seed(12)
    partial = partial.float() + 1e-3 * torch.randn(partial.shape, generator=g, device="cuda")
    rounded = partial.to(torch.bfloat16)
    assert not torch.equal(partial, rounded.float()), "values bf16 cannot hold"
    out = op(blocks, partial, queries, weights, eps)
    ref = op.reference(blocks, partial, queries, weights, eps)
    assert_within_ulp(
        out.reshape(-1, HIDDEN), ref.reshape(-1, HIDDEN), element_ulp=1.0, rms_ulp=0.1
    )
    # Had the op rounded the partial block first, both inputs would give the same mix.
    assert not torch.equal(out, op(blocks, rounded, queries, weights, eps))


def test_guard_refuses_bf16_queries() -> None:
    blocks, partial, queries, weights, eps = _build(dict(blocks=2, queries=1, tokens=3), 3)
    with pytest.raises(AssertionError, match="fp32"):
        with validating(op):
            op(blocks, partial, queries.to(torch.bfloat16), weights, eps)
