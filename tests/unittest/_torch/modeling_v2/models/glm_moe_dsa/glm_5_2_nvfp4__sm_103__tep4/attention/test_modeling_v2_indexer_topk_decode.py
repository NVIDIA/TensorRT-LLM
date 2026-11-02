# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's indexer_topk_decode entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.indexer_topk_decode import (  # noqa: E501
    indexer_topk_decode as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "indexer_topk_decode requires a CUDA device"

#: Written past each row's causal end. A kernel that read past the end would
#: select these before any real score.
POISON = 1.0e9


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Exact set equality per row, -1 padding included. A top-k either picks
    the same keys or it does not; there is no rounding to absorb. The op does
    not order its winners, so each row is sorted before comparing."""
    assert out.dtype == ref.dtype == torch.int32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    got, want = out.sort(dim=1).values, ref.sort(dim=1).values
    bad = (got != want).any(dim=1)
    assert not bool(bad.any()), (
        f"{int(bad.sum())} of {out.shape[0]} rows select a different key set; "
        f"first at row {int(bad.nonzero()[0])}"
    )


def _build(spec, seed):
    """`(logits, seq_lens, indices, next_n, index_topk, aux_idx, aux_val)` for one cell.

    Each row's scores are a random permutation of distinct integers, so the
    top-k set is unique and `_compare`'s set equality is exact. Columns at and
    past the row's causal end hold `POISON`. Scratch is sized for the widest
    split the op can choose (10 blocks per row), as the engine's arena is.
    """
    g = torch.Generator().manual_seed(seed)
    next_n, width, k = spec["next_n"], spec["width"], spec["index_topk"]
    lens = spec["seq_lens"]
    rows = len(lens) * next_n
    pad = spec.get("row_pad", 0)
    parent = torch.empty(rows, width + pad, dtype=torch.float32)
    for r in range(rows):
        end = lens[r // next_n] - next_n + r % next_n + 1
        parent[r] = POISON + torch.arange(width + pad, dtype=torch.float32)
        parent[r, :end] = torch.randperm(end, generator=g).float()
    logits = parent.cuda()[:, :width]
    assert pad == 0 or not logits.is_contiguous()
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    indices = torch.full((rows, k), -7, dtype=torch.int32, device="cuda")
    aux_idx = torch.empty(rows * 10 * k, dtype=torch.int32, device="cuda")
    aux_val = torch.empty(rows * 10 * k, dtype=torch.float32, device="cuda")
    return logits, seq_lens, indices, next_n, k, aux_idx, aux_val


def _call(logits, seq_lens, indices, next_n, k, aux_idx, aux_val):
    # As the GLM target calls it: keyword scratch, compress_ratio left at 1.
    return op(
        logits,
        seq_lens,
        indices,
        next_n,
        k,
        radix_aux_indices=aux_idx,
        radix_aux_logits=aux_val,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    logits, seq_lens, indices, next_n, k, aux_idx, aux_val = _build(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31
    )
    with validating(op):
        out = _call(logits, seq_lens, indices, next_n, k, aux_idx, aux_val)
    assert out is indices, "the op writes the caller's buffer and returns it"
    _compare(out, op.reference(logits, seq_lens, indices, next_n, k))


def test_short_rows_list_their_columns_in_order() -> None:
    """A row no wider than top-k is the identity, in order, then -1."""
    spec = dict(next_n=2, seq_lens=[5, 40], width=64, index_topk=16)
    logits, seq_lens, indices, next_n, k, aux_idx, aux_val = _build(spec, seed=11)
    out = _call(logits, seq_lens, indices, next_n, k, aux_idx, aux_val)
    assert out[0].tolist() == [0, 1, 2, 3] + [-1] * 12
    assert out[1].tolist() == [0, 1, 2, 3, 4] + [-1] * 11


def _ok():
    return _build(dict(next_n=1, seq_lens=[100, 80], width=128, index_topk=16), seed=7)


def test_guard_refuses_a_wider_indices_buffer() -> None:
    """The kernel writes at a row stride of index_topk, not of indices.size(1)."""
    logits, seq_lens, _, next_n, k, aux_idx, aux_val = _ok()
    wide = torch.full((2, 32), -7, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="exactly 16 columns wide"):
        with validating(op):
            _call(logits, seq_lens, wide, next_n, k, aux_idx, aux_val)


def test_guard_refuses_int64_seq_lens() -> None:
    logits, seq_lens, indices, next_n, k, aux_idx, aux_val = _ok()
    with pytest.raises(AssertionError, match="seq_lens must be int32"):
        with validating(op):
            _call(logits, seq_lens.long(), indices, next_n, k, aux_idx, aux_val)


def test_guard_refuses_a_causal_end_past_the_row() -> None:
    logits, _, indices, next_n, k, aux_idx, aux_val = _ok()
    long_lens = torch.tensor([129, 80], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="exceeds the 128-column logits"):
        with validating(op):
            _call(logits, long_lens, indices, next_n, k, aux_idx, aux_val)


def test_guard_refuses_a_column_strided_view() -> None:
    logits, seq_lens, indices, next_n, k, aux_idx, aux_val = _ok()
    strided = torch.empty(128, 2, dtype=torch.float32, device="cuda").t()
    strided.copy_(logits)
    with pytest.raises(AssertionError, match="unit-stride"):
        with validating(op):
            _call(strided, seq_lens, indices, next_n, k, aux_idx, aux_val)


def test_guard_refuses_top_k_past_the_launch_ceiling() -> None:
    logits, seq_lens, _, next_n, _, aux_idx, aux_val = _ok()
    k = 8189
    indices = torch.full((2, k), -7, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="index_topk must be in"):
        with validating(op):
            _call(logits, seq_lens, indices, next_n, k, aux_idx, aux_val)
