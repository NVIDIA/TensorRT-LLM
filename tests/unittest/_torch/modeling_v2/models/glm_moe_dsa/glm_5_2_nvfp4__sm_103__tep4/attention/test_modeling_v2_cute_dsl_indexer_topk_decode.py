# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_indexer_topk_decode entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel is built for sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_indexer_topk_decode import (  # noqa: E402, E501
    build,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_indexer_topk_decode import (  # noqa: E402, E501
    cute_dsl_indexer_topk_decode as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

TOP_K = 2048
#: Written past each row's causal end. A kernel that read past the end would
#: select these before any real score.
POISON = 1.0e9

_COMPILED: dict = {}


def _compiled(max_columns: int, next_n: int):
    key = (max_columns, next_n)
    if key not in _COMPILED:
        _COMPILED[key] = build(
            torch.cuda.current_device(), max_columns=max_columns, top_k=TOP_K, next_n=next_n
        )
    return _COMPILED[key]


def _build(spec, seed):
    """`(logits, seq_lens, indices)` for one cell.

    Each row's causal part is a permutation of distinct integers, or, with
    `levels`, integers drawn from that many values so the k-th value is tied
    many times over. Columns from the causal end on hold `POISON`. `indices`
    starts at -7 so an unwritten slot shows.
    """
    g = torch.Generator().manual_seed(seed)
    next_n, width = spec["next_n"], spec["width"]
    lens = spec["seq_lens"]
    rows = len(lens) * next_n
    pad = spec.get("row_pad", 0)
    parent = POISON + torch.arange(width + pad, dtype=torch.float32).repeat(rows, 1)
    for r in range(rows):
        end = max(0, min(width, lens[r // next_n] - next_n + r % next_n + 1))
        if "levels" in spec:
            vals = torch.randint(0, spec["levels"], (end,), generator=g).float()
        else:
            vals = torch.randperm(end, generator=g).float()
        parent[r, :end] = vals
    logits = parent.cuda()[:, :width]
    assert pad == 0 or not logits.is_contiguous()
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    indices = torch.full((rows, TOP_K), -7, dtype=torch.int32, device="cuda")
    return logits, seq_lens, indices


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Exact set equality per row, -1 padding included.

    A top-k picks the same keys or it does not; there is no rounding to
    absorb. The reference applies the kernel's tie rule (fp32 bit order, lower
    column first), so even tied rows have one right answer. The kernel does
    not order its winners, so each row is sorted before comparing.
    """
    assert out.dtype == ref.dtype == torch.int32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    got, want = out.sort(dim=1).values, ref.sort(dim=1).values
    bad = (got != want).any(dim=1)
    assert not bool(bad.any()), (
        f"{int(bad.sum())} of {out.shape[0]} rows select a different key set; "
        f"first at row {int(bad.nonzero()[0])}"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    logits, seq_lens, indices = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    kw = dict(
        top_k=TOP_K, next_n=spec["next_n"], compiled=_compiled(spec["max_columns"], spec["next_n"])
    )
    with validating(op):
        out = op(logits, seq_lens, indices, **kw)
    assert out is indices, "the op writes the caller's buffer and returns it"
    _compare(out, op.reference(logits, seq_lens, indices, **kw))


def test_short_rows_list_their_columns_in_order() -> None:
    """A row no longer than top-k is the identity, in order, then -1."""
    spec = dict(next_n=2, max_columns=8192, seq_lens=[5, 40], width=8192)
    logits, seq_lens, indices = _build(spec, seed=11)
    out = op(logits, seq_lens, indices, top_k=TOP_K, next_n=2, compiled=_compiled(8192, 2))
    assert out[0].tolist() == [0, 1, 2, 3] + [-1] * (TOP_K - 4)
    assert out[1].tolist() == [0, 1, 2, 3, 4] + [-1] * (TOP_K - 5)


def test_top_k_and_next_n_cannot_be_passed_positionally() -> None:
    """The stock op's order is (next_n, index_topk); this entry refuses positions outright."""
    logits, seq_lens, indices = _build(
        dict(next_n=1, max_columns=8192, seq_lens=[3000], width=8192), seed=12
    )
    with pytest.raises(TypeError):
        op(logits, seq_lens, indices, 1, TOP_K, _compiled(8192, 1))


def test_the_handle_refuses_a_swapped_call() -> None:
    """Passing the stock op's values by the wrong names is caught by the handle's key."""
    logits, seq_lens, indices = _build(
        dict(next_n=2, max_columns=8192, seq_lens=[3000], width=8192), seed=13
    )
    with pytest.raises(ValueError, match="specialization mismatch"):
        op(logits, seq_lens, indices, top_k=2, next_n=TOP_K, compiled=_compiled(8192, 2))


def test_guard_refuses_a_causal_end_past_the_row() -> None:
    """The kernel would clamp it to the row width and answer for the wrong range."""
    logits, _, indices = _build(
        dict(next_n=1, max_columns=8192, seq_lens=[3000], width=4096), seed=14
    )
    long_lens = torch.tensor([4097], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="exceeds the 4096-column logits"):
        with validating(op):
            op(logits, long_lens, indices, top_k=TOP_K, next_n=1, compiled=_compiled(8192, 1))


def test_build_refuses_a_width_no_longer_than_top_k() -> None:
    """The call site keys its stock-op fallback on this: max_seq_len <= index_topk."""
    with pytest.raises(ValueError):
        build(torch.cuda.current_device(), max_columns=TOP_K, top_k=TOP_K, next_n=1)
