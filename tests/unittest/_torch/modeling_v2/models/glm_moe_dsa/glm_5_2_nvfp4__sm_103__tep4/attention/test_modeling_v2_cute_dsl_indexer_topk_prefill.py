# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_indexer_topk_prefill entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel is built for sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.attention.self_attn_indexer_topk_prefill import (  # noqa: E402, E501
    compile_indexer_topk_prefill_general,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_indexer_topk_prefill import (  # noqa: E402, E501
    build,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_indexer_topk_prefill import (  # noqa: E402, E501
    cute_dsl_indexer_topk_prefill as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

TOP_K = 2048
#: Written outside each row's window. A kernel that read outside would select
#: these before any real score.
POISON = 1.0e9

_COMPILED: dict = {}


def _compiled(max_columns: int):
    if max_columns not in _COMPILED:
        _COMPILED[max_columns] = build(
            torch.cuda.current_device(), top_k=TOP_K, max_columns=max_columns
        )
    return _COMPILED[max_columns]


def _windows(requests):
    """`(row_starts, row_ends, width)` for requests packed back to back in one K block.

    Request `(cached, new, taken)` owns `cached + new` keys starting where the
    previous one ended; its `j`-th new token sees `[offset, offset + cached + j
    + 1)`, as TRT-LLM's prefill chunk metadata builds it. `taken` keeps only the
    request's last rows, like one row tile of a long chunk.
    """
    starts, ends, offset = [], [], 0
    for cached, new, taken in requests:
        first = 0 if taken is None else new - taken
        for j in range(first, new):
            starts.append(offset)
            ends.append(offset + cached + j + 1)
        offset += cached + new
    as_t = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    return as_t(starts), as_t(ends), offset


def _build(spec, seed):
    """`(logits, row_starts, row_ends, indices)` for one cell.

    Inside its window each row holds distinct values (a random permutation of
    the block width), so its top-k set is unique; outside it holds `POISON`.
    `indices` starts at -7 so an unwritten slot shows.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    row_starts, row_ends, width = _windows(spec["requests"])
    rows = row_starts.shape[0]
    perm = torch.rand(rows, width, generator=g, device="cuda").argsort(dim=1).float()
    cols = torch.arange(width, device="cuda")
    inside = (cols >= row_starts[:, None]) & (cols < row_ends[:, None])
    logits = torch.where(inside, perm, POISON + cols.float()).contiguous()
    indices = torch.full((rows, TOP_K), -7, dtype=torch.int32, device="cuda")
    return logits, row_starts, row_ends, indices


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Exact set equality per row, -1 padding included: every window holds
    distinct values, so its top-k set is unique and there is nothing to round.
    The kernel does not order its winners, so each row is sorted first."""
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
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    kw = dict(top_k=TOP_K, compiled=_compiled(cell.spec["max_columns"]))
    with validating(op):
        out = op(*args, **kw)
    assert out is args[3], "the op writes the caller's buffer and returns it"
    _compare(out, op.reference(*args, **kw))


def test_short_rows_of_a_later_request_stay_window_local() -> None:
    """The regression the general arm's localize pass once had.

    In a K block of packed requests, a short row of the second or later request
    (window start > 0, length <= top-k) is already window-local when the
    general arm writes it. Subtracting the window start again turned every such
    row into negative indices. Here each short row must read 0..len-1 then -1.
    """
    spec = dict(requests=[(2900, 100, None), (0, 1500, None), (2000, 600, None)])
    logits, starts, ends, indices = _build(spec, seed=21)
    assert logits.shape[1] > TOP_K, "the cell must take the general arm"
    out = op(logits, starts, ends, indices, top_k=TOP_K, compiled=_compiled(16384))
    lengths = (ends - starts).long()
    short_later = ((lengths <= TOP_K) & (starts > 0)).nonzero().flatten()
    long_rows = (lengths > TOP_K).nonzero().flatten()
    assert short_later.numel() and long_rows.numel(), "the cell must mix both kinds"
    ident = torch.arange(TOP_K, device="cuda")
    want = torch.where(ident < lengths[short_later, None], ident, -1).to(torch.int32)
    assert torch.equal(out[short_later], want), "a short row of a later request is not local"
    assert bool((out[long_rows] >= 0).all()), "a long row has an unfilled or negative slot"
    assert bool((out[long_rows] < lengths[long_rows, None]).all()), "a long row is not local"


def test_a_width_past_the_build_has_no_launcher() -> None:
    """A narrower build refuses a wider block loudly; the call site must widen it first."""
    narrow = _compiled(8192)
    assert narrow.max_columns == 8192 and sorted(narrow.general) == [4096, 8192]
    args = _build(dict(requests=[(9000, 64, None)]), seed=22)
    with pytest.raises(KeyError):
        op(*args, top_k=TOP_K, compiled=narrow)


def test_guard_refuses_a_window_past_the_block() -> None:
    logits, starts, ends, indices = _build(dict(requests=[(3000, 8, None)]), seed=23)
    ends = ends.clone()
    ends[-1] = logits.shape[1] + 1
    with pytest.raises(AssertionError, match="past the"):
        with validating(op):
            op(logits, starts, ends, indices, top_k=TOP_K, compiled=_compiled(16384))


def test_guard_refuses_an_inverted_window() -> None:
    logits, starts, ends, indices = _build(dict(requests=[(3000, 8, None)]), seed=24)
    starts = starts.clone()
    starts[0] = ends[0] + 1
    with pytest.raises(AssertionError, match="row_start <= row_end"):
        with validating(op):
            op(logits, starts, ends, indices, top_k=TOP_K, compiled=_compiled(16384))


def test_guard_refuses_a_top_k_the_launchers_were_not_built_for() -> None:
    logits, starts, ends, _ = _build(dict(requests=[(3000, 8, None)]), seed=25)
    indices = torch.full((8, 1024), -7, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="compiled for top_k"):
        with validating(op):
            op(logits, starts, ends, indices, top_k=1024, compiled=_compiled(16384))


@pytest.mark.parametrize(
    "coarse_radix_bits, num_threads_per_cta",
    [(11, 1024), (10, 512), (10, 256)],
    ids=["radix2048-cta1024", "radix1024-cta512", "radix1024-cta256"],
)
def test_a_radix_wider_than_the_cta_is_refused(coarse_radix_bits, num_threads_per_cta) -> None:
    """The histogram clear writes one bin per thread; more bins than threads would
    leave some uninitialized, so the kernel refuses the configuration."""
    with pytest.raises(ValueError, match="histogram bins"):
        compile_indexer_topk_prefill_general(
            torch.cuda.current_device(),
            torch.float32,
            max_num_columns=4 * TOP_K,
            top_k=TOP_K,
            coarse_radix_bits=coarse_radix_bits,
            num_threads_per_cta=num_threads_per_cta,
        )
