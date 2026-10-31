# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_sparse_gqa catalog entry.

The paged state is real: one `[pages, slot_pages, kv, tpb, D]` HND pool whose
layer planes are the call's K and V -- tp1's cells one layer's adjacent pair,
the unified target's cells the engine's 24-role-page trunk, sized so the
requests' pages straddle 2^31 elements -- and the engine's slot table: a row
per request slot, assigned by a seeded permutation, `-1` past each sequence.
Selection rows are built the way QSA's top-k + expansion lay them out:
complete groups expanded to tokens, then the trailing partial group, then
`-1`, all packed at the front.

The call has two kernels behind it (and a merge). The `launches` fixture
records which ones a call indexed with a grid, and whether each launched with
PDL, so every cell asserts its branch (and, where it names `splits`, the
split count). A cell marked `capture` is also captured in a CUDA graph,
refilled in place and replayed, as the engine's decode step is.
"""

import math
import re

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_sparse_gqa import (  # noqa: E501
    qsa_paged_sparse_gqa as op,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels

__extra_import_path__ = [".."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_paged_sparse_gqa requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 ulp of each (row, head)'s magnitude, 2 ulp relative RMS, in q's dtype.

    The kernel rounds its online-softmax probabilities to the cache dtype
    before P.V, so the error tracks the magnitude of the values averaged,
    not of the output: against an fp32 reference it measured 1.2e-3 on rows
    averaging a thousand tokens (outputs 0.14 to 0.4) and 1.2e-2 on rows
    averaging a handful (outputs to 5). Scaled by each head row's own
    magnitude that is one band. Against this reference (fp32, rounded once)
    every ungated cell measured at most 2.0 element ulp -- a single final-rounding
    flip, since bf16's step is 2 of these units at the bottom of a binade --
    and at most 0.62 ulp RMS: 2x and 3.2x margin. Six wrong references (no
    scale, interleaved heads, padding attended, identity pages, K/V
    swapped, page/slot transposed) were each measured more than 0.1 off.

    The unified target's five cells (6/1 heads at 1, 128, 1024 and 8192 rows,
    24/2 over an e4m3 pool at 32) measured at most 2.00 element ulp (p99.9
    1.31) and 0.572 ulp RMS, cosine at least 0.9999975 -- the same band. Three
    wrong references sit outside it at every one of them: the table shifted
    by one page (at least 414 ulp), the softmax scale dropped (at least 254),
    and one selected token dropped a row (at least 12.5, the single-row cell).

    A gated output is rounded twice, the attention and then the gated
    product, each by at most one bf16 ulp of the value rounded, so its bound
    is near the band's 4 element ulp. The seven gated cells, 10 seeds each,
    measured at most 3.32 element ulp and 0.667 ulp RMS; the same calls
    ungated at most 2.00. Two gate variants sit outside the band at every
    gated cell: the gate not applied (at least 207.8 ulp) and read one head
    early (at least 234.8). One selected token dropped at the 1-row gated
    cell measures 3.82 element and 2.89 RMS ulp, refused on the RMS gate.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    if out.numel() == 0:
        return
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


DEV = "cuda"

_KERNELS = {
    "_qsa_paged_sparse_gqa_kernel": "prefill",
    "_qsa_paged_sparse_gqa_splitk_kernel": "splitk",
    "_qsa_merge_splitk_kernel": "merge",
}


class _LaunchRecorder:
    """Records a Triton kernel's grid indexing and whether it launched with PDL, then
    delegates to the kernel."""

    def __init__(self, inner, name, log):
        self._inner, self._name, self._log = inner, name, log

    def __getitem__(self, grid):
        entry = [self._name, tuple(grid), False]
        self._log.append(entry)
        launch = self._inner[grid]

        def run(*args, **kwargs):
            entry[2] = bool(kwargs.get("launch_pdl", False))
            return launch(*args, **kwargs)

        return run


@pytest.fixture
def launches(monkeypatch):
    """Which branch kernels the next call launches, in order: [name, grid, PDL]."""
    log: list = []
    for name, tag in _KERNELS.items():
        inner = getattr(_qsa_kernels, name)
        monkeypatch.setattr(_qsa_kernels, name, _LaunchRecorder(inner, tag, log))
    return log


# --------------------------------------------------------------------------
# building real state from a cell spec
# --------------------------------------------------------------------------


def _selected_tokens(positions, seed, compress_ratio, token_topk):
    """QSA's expansion layout, built in one pass.

    A random `min(available, block_topk)`-subset of the complete visible
    groups, each expanded to its `compress_ratio` tokens, then the incomplete
    trailing group, then `-1` -- packed at the front, as the producer emits.
    """
    block_topk = token_topk // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows = positions.numel()
    visible = positions.long() + 1
    available = visible // compress_ratio
    taken = available.clamp(max=block_topk)
    width = max(block_topk, int(available.max().item()) if rows else 0)

    scores = torch.rand(rows, width, generator=gen, device=DEV)
    group_ids = torch.arange(width, device=DEV)[None, :]
    scores = torch.where(group_ids < available[:, None], scores, torch.inf)
    chosen = scores.argsort(dim=1)[:, :block_topk]

    column = torch.arange(final_topk, device=DEV)[None, :]
    group_of_column = (column // compress_ratio).clamp(max=block_topk - 1).expand(rows, -1)
    body = torch.gather(chosen, 1, group_of_column) * compress_ratio + column % compress_ratio
    in_body = column < (taken * compress_ratio)[:, None]
    tail_column = column - (taken * compress_ratio)[:, None]
    tail = (available * compress_ratio)[:, None] + tail_column
    in_tail = ~in_body & (tail_column >= 0) & (tail < visible[:, None])
    selected = torch.where(in_body, body, torch.where(in_tail, tail, torch.full_like(body, -1)))
    return selected.to(torch.int32)


#: NaN's bit pattern per pool dtype: filled through an integer view, since not every fill
#: kernel takes an e4m3 tensor.
_NAN_BITS = {torch.bfloat16: (torch.int16, 0x7FC0), torch.float8_e4m3fn: (torch.uint8, 0x7F)}


def _nan_pool(shape, dtype):
    bits_dtype, nan = _NAN_BITS[dtype]
    return torch.full(shape, nan, dtype=bits_dtype, device=DEV).view(dtype)


def _build(spec, seed):
    """Real paged state for one spec, keyed by `__call__`'s parameter names.

    The pool is `[pool_pages, slot_pages, kv, tpb, D]` and the call's K and V are layer
    `layer`'s two role planes: tp1's cells keep one layer's adjacent pair (2 role pages a
    page), the unified target's cells the engine's trunk (24). The requests sit on the
    pool's last pages; every other page, and every other plane, is NaN, so an address that
    strayed could not come back finite.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d, q_heads = (
        spec["tokens_per_block"],
        spec["kv_heads"],
        spec["head_dim"],
        spec["q_heads"],
    )
    segments = list(spec["segments"])
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in segments])
    req = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    rows, slots = positions.numel(), spec["slots"]
    assert len(segments) <= slots

    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(segments)]
    mapped = [math.ceil((s + n) / tpb) for s, n in segments]
    assert max(mapped) <= spec["pages"]
    needed = sum(mapped) + 7
    pool_pages = spec.get("pool_pages", needed)
    assert pool_pages >= needed
    first = pool_pages - needed
    page_ids = first + torch.randperm(needed, generator=gen, device=DEV).to(torch.int32)
    block_table = torch.full((slots, spec["pages"]), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count

    slot_pages, layer = spec.get("slot_pages", 2), spec.get("layer", 0)
    planes = slice(2 * layer, 2 * layer + 2)
    pool = _nan_pool((pool_pages, slot_pages, kv, tpb, d), spec["cache_dtype"])
    pool[first:, planes] = torch.randn(needed, 2, kv, tpb, d, generator=gen, device=DEV).to(
        spec["cache_dtype"]
    )
    k_cache, v_cache = pool[:, 2 * layer], pool[:, 2 * layer + 1]

    q_dtype = spec["q_dtype"]
    fused = spec["q_layout"] == "fused_row"
    # The row q is read off: the packed `[q | k | v]` buffer, or the target's fused input
    # projection row `[q | k | v | gate | index q, k | pad]` (`row_heads` heads of D).
    row_heads = spec.get("row_heads", 55 if fused else q_heads + 2 * kv)
    row = torch.randn(rows, row_heads * d, generator=gen, device=DEV).to(q_dtype)
    q = row.view(rows, row_heads, d)[:, :q_heads]

    selected = _selected_tokens(positions, seed + 3, spec["compress_ratio"], spec["token_topk"])
    request_indices = slot_of.to(DEV)[req].to(spec["request_dtype"])

    gate = None
    if spec.get("gate") == "fused_row":
        assert fused, "the gate is read off the fused row"
        # The gate columns of the same row, after q, k and v.
        start = (q_heads + 2 * kv) * d
        gate = row[:, start : start + q_heads * d]

    args = dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        selected_tokens=selected,
        request_indices=request_indices,
        tokens_per_block=tpb,
        softmax_scale=d**-0.5,
    )
    if spec["prefill_pair"]:
        args.update(query_positions=positions, compress_ratio=spec["compress_ratio"])
    if gate is not None:
        args.update(output_gate=gate)
    return args


def _bound(positions, spec):
    """The prefill kernel's per-row scan bound, from the positions alone."""
    cr = spec["compress_ratio"]
    visible = positions.long() + 1
    return (visible // cr).clamp(max=spec["token_topk"] // cr) * cr + visible % cr


def _assert_straddles(args, edge: int) -> None:
    """The pages the cell's rows map put the K and V planes' element offsets on both sides
    of `edge`: what the cell claims about its pool."""
    table = args["block_table"]
    pages = table[table >= 0].long()
    for name in ("k_cache", "v_cache"):
        plane = args[name]
        ends = plane.storage_offset() + pages * plane.stride(0) + plane[0].numel() - 1
        assert bool((ends >= edge).any()), f"{name}: no mapped page lies past {edge}"
        assert bool((ends < edge).any()), f"{name}: every mapped page lies past {edge}"


def _is_prefix_packed(table):
    valid = table >= 0
    columns = torch.arange(table.shape[1], device=table.device)[None, :]
    return (valid == (columns < valid.sum(1)[:, None])).all(1)


def _refill(args, spec, seed) -> None:
    """The engine's next step, written into the call's own buffers: new q and gate values
    (the fused row's columns), a new selection at the same positions, and new K/V in every
    mapped page."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for name in ("q", "output_gate"):
        if name in args:
            view = args[name]
            view.copy_(torch.randn(view.shape, generator=gen, device=DEV).to(view.dtype))
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in spec["segments"]])
    args["selected_tokens"].copy_(
        _selected_tokens(positions, seed + 3, spec["compress_ratio"], spec["token_topk"])
    )
    table = args["block_table"]
    pages = table[table >= 0].long()
    for name in ("k_cache", "v_cache"):
        plane = args[name]
        bits = _NAN_BITS[plane.dtype][0]
        fresh = torch.randn((pages.numel(), *plane.shape[1:]), generator=gen, device=DEV)
        plane.view(bits)[pages] = fresh.to(plane.dtype).view(bits)


def _replay_in_a_graph(args, spec, seed, launches, first) -> None:
    """The decode site's capture: the call captured into a CUDA graph (its split-K partials
    allocated from the graph's pool), the inputs refilled in place, the graph replayed.
    The replay must take the eager call's launches (`first` is that call's output), answer
    the new inputs within the band, and match an eager call on them bitwise."""
    eager = [tuple(entry) for entry in launches]
    del launches[:]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(**args)
    assert [tuple(entry) for entry in launches] == eager, launches
    _refill(args, spec, seed)
    graph.replay()
    torch.cuda.synchronize()
    replayed = captured.clone()
    assert not torch.equal(replayed, first), "the replay did not read the refilled inputs"
    _compare(replayed, op.reference(**args))
    assert torch.equal(replayed, op(**args)), "the replay differs from an eager call"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying, launches) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    args = _build(spec, seed=sum(map(ord, cell.why)))
    rows = args["q"].shape[0]
    assert rows == spec.get("rows", rows)
    if "row_heads" in spec:
        # q, and the gate when fused, are views of one fused row `row_heads` heads wide.
        q = args["q"]
        assert q.stride(0) == spec["row_heads"] * spec["head_dim"]
        if "output_gate" in args:
            gate = args["output_gate"]
            assert gate.stride(0) == q.stride(0)
            assert gate.untyped_storage().data_ptr() == q.untyped_storage().data_ptr()
    if "rows_per_request" in spec:
        assert {n for _, n in spec["segments"]} == {spec["rows_per_request"]}
    if "offset_edge" in spec:
        _assert_straddles(args, spec["offset_edge"])
    if spec["prefill_pair"]:
        # The precondition the op cannot check: front-packed, and exactly as
        # many valid entries as the prefill kernel will scan.
        selected = args["selected_tokens"]
        assert bool(_is_prefix_packed(selected).all())
        valid = (selected >= 0).sum(1)
        assert torch.equal(valid, _bound(args["query_positions"], spec))

    out = op(**args)
    assert [name for name, _, _ in launches] == list(spec["launches"])
    assert launches[0][1][:2] == (rows, spec["kv_heads"])
    if "splits" in spec:
        assert launches[0][1][2] == spec["splits"], launches
    assert all(pdl == spec.get("pdl", False) for _, _, pdl in launches), launches
    if "output_gate" in args:
        gated = (args["q"], args["k_cache"], args["selected_tokens"], args["output_gate"])
        assert op.fuses_output_gate(*gated)
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == args["q"].dtype and out.is_contiguous()
    assert not bool(out.isnan().any())
    _compare(out, op.reference(**args))
    if spec.get("capture"):
        seed = sum(map(ord, cell.why)) + 1
        _replay_in_a_graph(args, spec, seed=seed, launches=launches, first=out)


#: The first cell is this checkpoint's own decode geometry; the tests below
#: start from it.
CKPT = op.CELLS[0].spec
assert (CKPT["q_heads"], CKPT["kv_heads"], CKPT["head_dim"]) == (24, 2, 256)


def _with(**overrides):
    return {**CKPT, **overrides}


def test_calls_are_bitwise_deterministic() -> None:
    """Including a 32-split launch whose merge is a second kernel."""
    spec = _with(segments=tuple((4000 + 37 * i, 2) for i in range(4)), gate=None)
    args = _build(spec, seed=8800)
    assert torch.equal(op(**args), op(**args))


# --------------------------------------------------------------------------
# what the op refuses
# --------------------------------------------------------------------------


def test_rejects_positions_without_a_compress_ratio() -> None:
    """The op's own loud refusal, verbatim -- the reason `is_valid` is empty: half of
    the prefill pair."""
    args = _build(_with(segments=((600, 1),) * 4, gate=None), seed=9100)
    args["selected_tokens"] = args["selected_tokens"][:, :64].contiguous()
    args["query_positions"] = torch.zeros(4, dtype=torch.int64, device=DEV)
    message = "QSA causal-prefix bounds require positions and compression ratio"
    with pytest.raises(ValueError, match=re.escape(message)):
        op(**args)
