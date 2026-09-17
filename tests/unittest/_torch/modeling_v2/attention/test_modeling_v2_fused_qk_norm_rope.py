# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_qk_norm_rope catalog entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    fused_qk_norm_rope as op,
)

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "fused_qk_norm_rope requires a CUDA device"


def _build(spec, seed):
    """Materialize one cell as (kwargs, qkv), with qkv separate because it is
    the buffer the op writes and the caller has to keep a copy of."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens = spec["tokens"]
    heads = spec["num_heads_q"] + spec["num_heads_k"] + spec["num_heads_v"]
    width = heads * spec["head_dim"]
    qkv = torch.randn(tokens, width, generator=g, device="cuda").to(torch.bfloat16)
    weight = torch.randn(spec["head_dim"], generator=g, device="cuda").to(torch.bfloat16)
    kwargs = {k: v for k, v in spec.items() if k != "tokens"}
    kwargs.update(
        eps=1e-6,
        q_weight=weight,
        k_weight=weight,
        position_ids=torch.arange(tokens, dtype=torch.int32, device="cuda"),
    )
    return qkv, kwargs


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    qkv, kwargs = _build(cell.spec, seed=abs(hash(cell.why)) % 2**31)
    ref = op.reference(qkv, **kwargs)
    with validating(op):
        assert op(qkv, **kwargs) is None, "the op writes in place"
    op.compare(qkv, ref)


def test_the_v_heads_are_left_alone() -> None:
    """The op's boundary, and nothing about the output shape shows it.

    A kernel that rotated the v heads too would still return a buffer of the
    right size, and every cell compares against a reference that makes the same
    claim -- so this pins it against the input instead.
    """
    spec = dict(op.CELLS[1].spec)
    qkv, kwargs = _build(spec, seed=11)
    before = qkv.clone()
    op(qkv, **kwargs)
    v_start = (kwargs["num_heads_q"] + kwargs["num_heads_k"]) * kwargs["head_dim"]
    assert torch.equal(qkv[:, v_start:], before[:, v_start:])
    assert not torch.equal(qkv[:, :v_start], before[:, :v_start]), "q and k were not touched"


def _ok(**over):
    spec = dict(op.CELLS[0].spec)
    spec.update(over)
    return _build(spec, seed=3)


def test_guard_refuses_a_width_that_disagrees_with_the_head_counts() -> None:
    """Read as whatever the counts say; the heads come out shifted."""
    qkv, kwargs = _ok()
    with pytest.raises(AssertionError, match="last dim must be"):
        with validating(op):
            op(qkv[:, :-64], **kwargs)


def test_guard_refuses_a_rotary_dim_past_the_head() -> None:
    qkv, kwargs = _ok()
    kwargs["rotary_dim"] = 128
    with pytest.raises(AssertionError, match="exceeds head_dim"):
        with validating(op):
            op(qkv, **kwargs)


def test_guard_refuses_a_position_id_count_that_is_not_the_token_count() -> None:
    qkv, kwargs = _ok()
    kwargs["position_ids"] = torch.arange(3, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="one position per token"):
        with validating(op):
            op(qkv, **kwargs)
