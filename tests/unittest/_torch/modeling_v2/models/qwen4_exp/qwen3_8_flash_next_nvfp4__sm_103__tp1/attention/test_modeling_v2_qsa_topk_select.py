# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_topk_select catalog entry.

The op selects; it computes nothing. `_compare` therefore checks the output's
structure exactly and the *multiset of selected values* bit for bit, never the
permutation, which the op leaves unspecified.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_topk_select import (  # noqa: E501
    TopkExpectation,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_topk_select import (  # noqa: E501
    qsa_topk_select as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_topk_select requires a CUDA device"


def _compare(out: torch.Tensor, ref: TopkExpectation) -> None:
    """Structure exactly, then the selected values bit for bit.

    A selection does no arithmetic, so nothing is tolerated. Values rather
    than offsets are compared so that a tied k-th score -- common here: the
    relu cells are half zeros -- is a pass rather than a coin flip.
    """
    assert out.dtype == torch.int32, out.dtype
    assert out.shape == ref.values.shape, (out.shape, ref.values.shape)
    index_topk = out.shape[1]
    valid = out >= 0
    assert not bool((valid[:, 1:] & ~valid[:, :-1]).any()), "a valid offset follows a -1 pad"
    assert bool((out >= -1).all()), "padding must be exactly -1"
    assert torch.equal(valid.sum(1), ref.window.clamp(max=index_topk)), (
        "each row must select exactly min(index_topk, window) offsets"
    )
    assert bool((out < ref.window[:, None]).all()), "an offset reaches past row_end"
    descending = torch.sort(out, dim=1, descending=True).values
    assert not bool(((descending[:, :-1] == descending[:, 1:]) & (descending[:, 1:] >= 0)).any()), (
        "duplicate offset within a row"
    )
    absolute = (out.long() + ref.row_starts[:, None]).clamp(0, ref.logits.shape[1] - 1)
    picked = torch.gather(ref.logits, 1, absolute).masked_fill(~valid, float("-inf"))
    got = torch.sort(picked, dim=1, descending=True).values
    assert torch.equal(got, ref.values), "selected values are not the window's top-k"


DEV = "cuda"
COMPRESS_RATIO = 4
BLOCK_TOPK = 512
#: Pre-fill for every output buffer, so a slot the op failed to write shows up
#: as an offset past `row_end` in `_compare`.
SENTINEL = 12345


def _logits(kind: str, rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(rows, cols, generator=gen, device=DEV)
    # QSA's scores are relu(q.k) summed over heads: non-negative with ties at 0.
    return torch.relu(x) * 3.0 if kind == "relu" else x


def _build(spec: dict, seed: int):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows, cols, index_topk = spec["rows"], spec["cols"], spec["index_topk"]
    if spec.get("strided"):
        backing = _logits(spec["logits"], rows, 2 * cols, gen).view(rows, cols, 2)
        logits = backing[:, :, 0]
        assert logits.stride() == (2 * cols, 2)
    else:
        logits = _logits(spec["logits"], rows, cols, gen)

    kind = spec["windows"][0]
    if kind in ("positions", "chunk"):
        if kind == "positions":
            positions = torch.tensor(spec["windows"][1], device=DEV)
        else:
            positions = spec["windows"][1] + torch.arange(rows, device=DEV)
        assert positions.numel() == rows
        # QSA's per-row causal bound: complete compressed blocks at the position.
        row_ends = ((positions + 1) // COMPRESS_RATIO).clamp_max(cols).to(torch.int32)
        row_starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    elif kind == "explicit":
        row_starts = torch.tensor(spec["windows"][1], dtype=torch.int32, device=DEV)
        row_ends = torch.tensor(spec["windows"][2], dtype=torch.int32, device=DEV)
    else:
        row_starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
        row_ends = torch.randint(0, cols + 1, (rows,), generator=gen, device=DEV).to(torch.int32)

    if spec["tail"] == "nan":
        columns = torch.arange(cols, device=DEV)[None, :]
        outside = (columns < row_starts[:, None]) | (columns >= row_ends[:, None])
        logits.masked_fill_(outside, float("nan"))
    indices = torch.full((rows, index_topk), SENTINEL, dtype=torch.int32, device=DEV)
    return logits, row_starts, row_ends, indices, index_topk


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=sum(map(ord, cell.why)))
    out = op(*args)
    assert out is args[3], "the op writes `indices` and returns it"
    _compare(out, op.reference(*args))


def test_selection_honours_row_ends_and_is_not_a_prefix() -> None:
    """The two wrong answers that would still look plausible.

    The global maxima sit outside every window, and the in-window winners are
    the *last* visible columns: a selector that ignored `row_ends` would take
    the former, one that returned the first k columns would miss the latter.
    """
    gen = torch.Generator(device=DEV).manual_seed(2)
    rows, cols, top_k = 6, 1024, 64
    logits = torch.rand(rows, cols, generator=gen, device=DEV)
    logits[:, 256:] += 100.0
    logits[:, 192:256] += 10.0
    row_starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    row_ends = torch.full((rows,), 256, dtype=torch.int32, device=DEV)
    indices = torch.full((rows, top_k), SENTINEL, dtype=torch.int32, device=DEV)
    out = op(logits, row_starts, row_ends, indices, top_k)
    _compare(out, op.reference(logits, row_starts, row_ends, indices, top_k))
    assert bool((out < 256).all()), "selected a column beyond row_ends"
    assert bool((out >= 192).all()), "did not find the planted in-window maxima"


def test_nan_inside_the_window_is_selected() -> None:
    """Why the note makes finiteness a precondition: NaN outranks every value."""
    gen = torch.Generator(device=DEV).manual_seed(11)
    logits = torch.randn(1, 32, generator=gen, device=DEV)
    logits[0, :4] = float("nan")
    row_starts = torch.zeros(1, dtype=torch.int32, device=DEV)
    row_ends = torch.full((1,), 32, dtype=torch.int32, device=DEV)
    out = op(logits, row_starts, row_ends, torch.empty(1, 8, dtype=torch.int32, device=DEV), 8)
    assert {0, 1, 2, 3} <= set(out[0].tolist()), "NaN is no longer selected; update the note"


def test_cuda_graph_capture_and_replay() -> None:
    """Captured, then replayed against new scores and a new causal bound."""
    gen = torch.Generator(device=DEV).manual_seed(8)
    rows, cols = 8, 2048
    logits = _logits("relu", rows, cols, gen)
    row_starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    row_ends = torch.full((rows,), cols, dtype=torch.int32, device=DEV)
    indices = torch.empty(rows, BLOCK_TOPK, dtype=torch.int32, device=DEV)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(logits, row_starts, row_ends, indices, BLOCK_TOPK)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(logits, row_starts, row_ends, indices, BLOCK_TOPK)

    logits.copy_(_logits("relu", rows, cols, gen))
    row_ends.fill_(1500)
    indices.fill_(SENTINEL)
    graph.replay()
    torch.cuda.synchronize()
    _compare(indices, op.reference(logits, row_starts, row_ends, indices, BLOCK_TOPK))


def test_rejects_out_of_contract() -> None:
    """The op's own loud refusals -- the reason `is_valid` does not restate them."""
    rows, cols, top_k = 4, 128, 8
    logits = torch.randn(rows, cols, device=DEV)
    row_starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    row_ends = torch.full((rows,), cols, dtype=torch.int32, device=DEV)
    out = torch.empty(rows, top_k, dtype=torch.int32, device=DEV)
    with pytest.raises(RuntimeError, match="expected scalar type Float"):
        op(logits.to(torch.bfloat16), row_starts, row_ends, out, top_k)
    with pytest.raises(RuntimeError, match="expected scalar type Int"):
        op(logits, row_starts, row_ends, out.long(), top_k)
    with pytest.raises(RuntimeError, match="one entry per row"):
        op(logits, row_starts[:2], row_ends[:2], out, top_k)


def _layout_case():
    torch.manual_seed(10)
    rows, cols, top_k = 4, 128, 8
    logits = torch.randn(rows, cols, device=DEV)
    row_starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    row_ends = torch.full((rows,), cols, dtype=torch.int32, device=DEV)
    return logits, row_starts, row_ends, top_k


def test_guard_refuses_a_wider_output_buffer() -> None:
    """Accepted by the op, which packs the rows at a stride of `index_topk`
    regardless: row 1's answer starts inside row 0's slots and the last row is
    never written. Shown unguarded first, then refused."""
    logits, row_starts, row_ends, top_k = _layout_case()
    rows = logits.shape[0]
    wide = torch.full((rows, top_k + 4), -99, dtype=torch.int32, device=DEV)
    op(logits, row_starts, row_ends, wide, top_k)
    torch.cuda.synchronize()
    packed = wide.flatten()[: rows * top_k].view(rows, top_k)
    _compare(packed, op.reference(logits, row_starts, row_ends, packed, top_k))
    assert not bool((wide[:, top_k:] == -99).all()), "the hazard stopped reproducing"
    assert bool((wide[-1] == -99).all()), "the hazard stopped reproducing"

    with pytest.raises(AssertionError, match=r"indices must be \[num_rows, index_topk\]"):
        with validating(op):
            op(logits, row_starts, row_ends, wide, top_k)


def test_guard_refuses_the_token_budget_default() -> None:
    """The default `index_topk` is 2048, the token budget; the target's buffer
    is 512 wide. That is the wider-buffer fault the other way round, and the
    guard catches it at the call that forgot to pass the block budget."""
    logits, row_starts, row_ends, _ = _layout_case()
    indices = torch.empty(logits.shape[0], BLOCK_TOPK, dtype=torch.int32, device=DEV)
    with pytest.raises(AssertionError, match="index_topk=2048"):
        with validating(op):
            op(logits, row_starts, row_ends, indices)


def test_guard_refuses_a_non_contiguous_output() -> None:
    """A row slice of a wider arena: the same flat addressing, the same fault."""
    logits, row_starts, row_ends, top_k = _layout_case()
    arena = torch.full((logits.shape[0], top_k + 4), -99, dtype=torch.int32, device=DEV)
    view = arena[:, :top_k]
    assert not view.is_contiguous()
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(logits, row_starts, row_ends, view, top_k)
