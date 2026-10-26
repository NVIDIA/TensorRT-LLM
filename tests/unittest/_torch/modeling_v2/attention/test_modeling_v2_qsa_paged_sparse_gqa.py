# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_sparse_gqa catalog entry.

The paged state is real: one `[pages, 2, kv, tpb, D]` HND pool, and the
engine's slot table -- a row per request slot, more slots than requests,
assigned by a seeded permutation, `-1` past each sequence. The QSA cells size
the pool from the pages their requests map and lay selection rows out the way
QSA's top-k + expansion do: complete groups expanded to tokens, then the
trailing partial group, then `-1`, all packed at the front.

The Step-5 cells hold a fixed pool of `pool_pages` pages with 4 (attention DP)
or 1 (TP4) KV heads, map each request onto scattered pages of it, and hand the
op the single-head view `pool[:, i, g:g+1]` the target passes per KV group; q
is the group's 16-head slice of a `[T, 16 * heads, 256]` projection. Dims
192..255 are zero in q and in the pool, as the target pads them. Their
selection is the target's layout: the top-512 complete regions of 8 tokens in
score order (`-1` padded when fewer are visible), each expanded to its 8
logical positions, and the current region's 8 columns, `-1` past the query.
The cell's `layout` puts the current region last (`regions_then_tail`, so a
short row's `-1`s sit between its regions and its tail) or first
(`tail_first`).

The Step-5 cells with `slot_pages` hold the engine's coalesced pool instead:
`pool_pages` slots of `slot_pages` pages each, `[pool_pages, slot_pages,
heads, tpb, D]` in the cache dtype, and hand the op layer `layer`'s K and V
pages of every slot (pages `2 * layer` and `2 * layer + 1`) as `[slots, kv,
tpb, D]` strided views, head-sliced as above. Only those pages of the mapped
slots hold data; every other page is NaN (the e4m3 NaN code in an fp8 pool),
so a read at a wrong offset shows: the kernel turns a NaN tile into a split
of zeros, which lands 256 ulp off the reference (measured with the
neighbouring layer's view), far outside the band. `include_last_page` maps the
pool's last slot under the last row's current region, so the highest offset
is read.

The call has two kernels behind it (and a merge). The `launches` fixture
records which ones a call indexed with a grid, so every cell asserts its
branch, and the Step-5 cells their split count.
"""

import math
import re
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_sparse_gqa import (
    qsa_paged_sparse_gqa as op,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels

__extra_import_path__ = [".."]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_paged_sparse_gqa requires a CUDA device"

#: The band `_compare` holds, in ulps of q's dtype.
ELEMENT_ULP, RMS_ULP = 4.0, 2.0


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 ulp of each (row, head)'s magnitude, 2 ulp relative RMS, in q's dtype.

    The kernel rounds its online-softmax probabilities to the cache dtype
    before P.V, so the error tracks the magnitude of the values averaged,
    not of the output: against an fp32 reference it measured 1.2e-3 on rows
    averaging a thousand tokens (outputs 0.14 to 0.4) and 1.2e-2 on rows
    averaging a handful (outputs to 5). Scaled by each head row's own
    magnitude that is one band. Against this reference (fp32, rounded once)
    every QSA cell measured at most 2.0 element ulp -- a single final-rounding
    flip, since bf16's step is 2 of these units at the bottom of a binade --
    and at most 0.62 ulp RMS: 2x and 3.2x margin. Six wrong references (no
    scale, interleaved heads, padding attended, identity pages, K/V
    swapped, page/slot transposed) were each measured more than 0.1 off.
    The Step-5 cells' own measurement and wrong variants are in
    `test_step5_band_rejects_wrong_variants`.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    if out.numel() == 0:
        return
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


def _ulps(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """`assert_within_ulp`'s two statistics, measured rather than asserted."""
    ulp = ULP[ref.dtype]
    flat_out = out.float().reshape(-1, out.shape[-1])
    flat_ref = ref.float().reshape(-1, ref.shape[-1])
    row_scale = flat_ref.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    element = ((flat_out - flat_ref).abs() / row_scale).max().item() / ulp
    rms = (
        (flat_out - flat_ref).pow(2).mean().sqrt() / flat_ref.pow(2).mean().sqrt().clamp_min(1e-9)
    ).item() / ulp
    return element, rms


DEV = "cuda"

_KERNELS = {
    "_qsa_paged_sparse_gqa_kernel": "prefill",
    "_qsa_paged_sparse_gqa_splitk_kernel": "splitk",
    "_qsa_merge_splitk_kernel": "merge",
}


class _LaunchRecorder:
    """Records a Triton kernel's grid indexing, then delegates to the kernel."""

    def __init__(self, inner, name, log):
        self._inner, self._name, self._log = inner, name, log

    def __getitem__(self, grid):
        self._log.append((self._name, tuple(grid)))
        return self._inner[grid]


@pytest.fixture
def launches(monkeypatch):
    """Which branch kernels the next call launches, in order, with grids."""
    log: list = []
    for name, tag in _KERNELS.items():
        inner = getattr(_qsa_kernels, name)
        monkeypatch.setattr(_qsa_kernels, name, _LaunchRecorder(inner, tag, log))
    return log


# --------------------------------------------------------------------------
# building real state from a cell spec
# --------------------------------------------------------------------------


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


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


def _step5_selection(positions, seed, spec):
    """Step-5's selection: top-512 history regions in score order, and the tail.

    History is the complete regions before the query's own, `[0, pos // 8)`;
    the current region's columns are `8 * (pos // 8) + j`, `-1` where past
    `pos`. Width `512 * 8 + 8` = 4104. `regions_then_tail` puts the current
    region in the last 8 columns, `tail_first` in the first 8.
    """
    region, topk = spec["region_size"], spec["region_topk"]
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows = positions.numel()
    pos = positions.long()
    history = pos // region
    width = max(topk, int(history.max().item()))
    scores = torch.rand(rows, width, generator=gen, device=DEV)
    ids = torch.arange(width, device=DEV)[None, :]
    scores = torch.where(ids < history[:, None], scores, -1.0)
    values, regions = scores.topk(topk, dim=1)
    regions = torch.where(values >= 0, regions, -1)
    offsets = torch.arange(region, device=DEV)
    body = torch.where(regions[:, :, None] >= 0, regions[:, :, None] * region + offsets, -1)
    tail = (history * region)[:, None] + offsets[None, :]
    tail = torch.where(tail <= pos[:, None], tail, -1)
    body = body.reshape(rows, topk * region)
    if spec["layout"] == "tail_first":
        selected = torch.cat([tail, body], dim=1)
    else:
        assert spec["layout"] == "regions_then_tail", spec["layout"]
        selected = torch.cat([body, tail], dim=1)
    assert selected.shape[1] == spec["width"]
    return selected.to(torch.int32)


def _check_step5_selection(selected, positions, spec):
    """The layout's invariants: count, causality, no token twice."""
    region, topk = spec["region_size"], spec["region_topk"]
    pos = positions.long()
    expected = (pos // region).clamp(max=topk) * region + pos % region + 1
    assert torch.equal((selected >= 0).sum(1), expected)
    assert bool((selected.long() <= pos[:, None]).all())
    ordered = selected.long().sort(dim=1).values
    assert not bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any())


def _positions(spec):
    return torch.cat([torch.arange(s, s + n, device=DEV) for s, n in spec["segments"]])


#: NaN in each pool dtype, as raw bits: what every unmapped page holds.
_NAN_BITS = {torch.bfloat16: (torch.int16, 0x7FC0), torch.float8_e4m3fn: (torch.uint8, 0x7F)}


def _slot_pool(spec, mapped_slots, pool_heads, real, gen):
    """The engine's coalesced pool, `[pool_pages, slot_pages, heads, tpb, D]`.

    Built in the cache dtype with no whole-pool temporary: every page is NaN
    except layer `layer`'s K and V pages of the mapped slots, which hold
    data, zero from dim `real` on. A read at any other offset reads NaN.
    """
    dtype, layer = spec["cache_dtype"], spec["layer"]
    bits_dtype, nan = _NAN_BITS[dtype]
    shape = (
        spec["pool_pages"],
        spec["slot_pages"],
        pool_heads,
        spec["tokens_per_block"],
        spec["head_dim"],
    )
    pool = torch.full(shape, nan, dtype=bits_dtype, device=DEV).view(dtype)
    data = torch.randn(mapped_slots.numel(), 2, *shape[2:], generator=gen, device=DEV)
    data[..., real:] = 0
    pool.view(bits_dtype)[mapped_slots, 2 * layer : 2 * layer + 2] = data.to(dtype).view(bits_dtype)
    return pool


def _last_page_offset(args) -> int:
    """Element offset, from the pool's base, of the K page of the highest page
    the rows' requests map."""
    page = int(args["block_table"][args["request_indices"].long()].max())
    k_cache = args["k_cache"]
    return k_cache.storage_offset() + page * k_cache.stride(0)


def _build(spec, seed):
    """Real paged state for one spec, keyed by `__call__`'s parameter names."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d, q_heads = (
        spec["tokens_per_block"],
        spec["kv_heads"],
        spec["head_dim"],
        spec["q_heads"],
    )
    step5 = spec.get("selection") == "step5_regions"
    segments = list(spec["segments"])
    positions = _positions(spec)
    req = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    rows, slots = positions.numel(), spec["slots"]
    assert len(segments) <= slots

    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(segments)]
    mapped = [math.ceil((s + n) / tpb) for s, n in segments]
    assert max(mapped) <= spec["pages"]
    pool_pages = spec.get("pool_pages", sum(mapped) + 7)
    assert sum(mapped) <= pool_pages
    page_ids = torch.randperm(pool_pages, generator=gen, device=DEV).to(torch.int32)
    used = sum(mapped)
    if spec.get("include_last_page"):
        # The last request's last page -- its last row's current region,
        # which every selection reads -- is the pool's last page.
        last = int((page_ids == pool_pages - 1).nonzero()[0, 0])
        page_ids[[last, used - 1]] = page_ids[[used - 1, last]]
    block_table = torch.full((slots, spec["pages"]), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count

    pool_heads = spec.get("pool_kv_heads", kv)
    real = spec.get("real_head_dim", d)
    if "slot_pages" in spec:
        pool = _slot_pool(spec, page_ids[:used].long(), pool_heads, real, gen)
        planes = pool[:, 2 * spec["layer"]], pool[:, 2 * spec["layer"] + 1]
    else:
        pool = torch.randn(pool_pages, 2, pool_heads, tpb, d, generator=gen, device=DEV)
        pool[..., real:] = 0
        pool = pool.to(spec["cache_dtype"])
        planes = pool[:, 0], pool[:, 1]
    view = spec.get("head_view")
    if view is None:
        k_cache, v_cache = planes
    else:
        k_cache, v_cache = (plane[:, view : view + 1] for plane in planes)
    assert k_cache.shape[1] == kv

    q_dtype = spec["q_dtype"]
    if step5:
        projection = torch.randn(rows, q_heads * pool_heads, d, generator=gen, device=DEV)
        projection[..., real:] = 0
        first = q_heads * (view or 0)
        q = projection.to(q_dtype)[:, first : first + q_heads]
        selected = _step5_selection(positions, seed + 3, spec)
    else:
        row_heads = q_heads + 2 * kv if spec["q_layout"] == "qkv_slice" else 55
        qkv = torch.randn(rows, row_heads, d, generator=gen, device=DEV).to(q_dtype)
        q = qkv[:, :q_heads]
        selected = _selected_tokens(positions, seed + 3, spec["compress_ratio"], spec["token_topk"])
    request_indices = slot_of.to(DEV)[req].to(spec["request_dtype"])

    gate = None
    if spec.get("gate") == "fused_row":
        # The gate columns of the target's fused projection row (55 heads of
        # D): heads 28..51, after q, k and v.
        fused = torch.randn(rows, 55 * d, generator=gen, device=DEV).to(q_dtype)
        gate = fused[:, 28 * d : (28 + q_heads) * d]

    args = dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        selected_tokens=selected,
        request_indices=request_indices,
        tokens_per_block=tpb,
        softmax_scale=spec.get("softmax_scale", d**-0.5),
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


def _is_prefix_packed(table):
    valid = table >= 0
    columns = torch.arange(table.shape[1], device=table.device)[None, :]
    return (valid == (columns < valid.sum(1)[:, None])).all(1)


_CELL_PARAMS = [
    pytest.param(c, id=c.why[:44], marks=(pytest.mark.large,) if c.spec.get("large") else ())
    for c in op.CELLS
]


@pytest.mark.parametrize("cell", _CELL_PARAMS)
def test_certified_cells(cell, certifying, launches) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    args = _build(spec, seed=_seed(cell.why))
    rows = args["q"].shape[0]
    if spec.get("include_last_page"):
        # The pool's last slot is mapped, its page offset is past 2^33
        # elements, and the last row's current region (always read) is on it.
        assert int(args["block_table"].max()) == args["k_cache"].shape[0] - 1
        assert _last_page_offset(args) > 2**33, _last_page_offset(args)
        last_row_request = int(args["request_indices"][-1])
        last_position = int(_positions(spec)[-1])
        last_page = args["block_table"][last_row_request, last_position // spec["tokens_per_block"]]
        assert int(last_page) == args["k_cache"].shape[0] - 1
    if spec["prefill_pair"]:
        # The precondition the op cannot check: front-packed, and exactly as
        # many valid entries as the prefill kernel will scan.
        selected = args["selected_tokens"]
        assert bool(_is_prefix_packed(selected).all())
        valid = (selected >= 0).sum(1)
        assert torch.equal(valid, _bound(args["query_positions"], spec))
    if "splits" in spec:
        _check_step5_selection(args["selected_tokens"], _positions(spec), spec)
        pdl = spec["splits"] > 1 and _qsa_kernels._qsa_pdl_enabled(rows)
        assert pdl == spec.get("pdl", False)

    out = op(**args)
    assert [name for name, _ in launches] == list(spec["launches"])
    assert launches[0][1][:2] == (rows, spec["kv_heads"])
    if "splits" in spec:
        assert launches[0][1] == (rows, spec["kv_heads"], spec["splits"])
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == args["q"].dtype and out.is_contiguous()
    assert not bool(out.isnan().any())
    if "real_head_dim" in spec:
        # Zero-padded V columns average to exact zeros.
        assert not bool(out[..., spec["real_head_dim"] :].any())
    _compare(out, op.reference(**args))


#: The first cell is the QSA checkpoint's own decode geometry; the two QSA
#: tests below start from it.
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


# --------------------------------------------------------------------------
# Step-5-Preview
# --------------------------------------------------------------------------


def _step5_cell(prefix: str) -> dict:
    (spec,) = [c.spec for c in op.CELLS if c.why.startswith(prefix)]
    return spec


STEP5_DECODE = _step5_cell("Step-5 tail-first dp 128 decode rows")
STEP5_ONE_ROW = _step5_cell("Step-5 dp 1 row")
STEP5_TWO_ROWS = _step5_cell("Step-5 dp 2 decode rows")
STEP5_EIGHT_ROWS = _step5_cell("Step-5 dp 8 decode rows")
STEP5_120_ROWS = _step5_cell("Step-5 dp 120 decode rows")


def _capture_id(spec: dict) -> str:
    rows = sum(n for _, n in spec["segments"])
    pdl = ", PDL" if spec.get("pdl") else ""
    return f"{rows} rows tail-first{pdl}, {spec['splits']} splits"


#: Every Step-5 cell the graph test captures: the regions-then-tail capture
#: batches 1, 2, 8 and 120, the tail-first 128-row decode cell, and every cell
#: whose spec says `capture` (all tail-first).
_TAGGED = [c.spec for c in op.CELLS if c.spec.get("capture")]
assert all(spec["layout"] == "tail_first" for spec in _TAGGED)
STEP5_CAPTURED = [
    STEP5_ONE_ROW,
    STEP5_TWO_ROWS,
    STEP5_EIGHT_ROWS,
    STEP5_120_ROWS,
    STEP5_DECODE,
    *_TAGGED,
]
STEP5_CAPTURED_IDS = [
    "1 row, PDL, 64 splits",
    "2 rows, PDL, 64 splits",
    "8 rows, 32 splits",
    "120 rows, 8 splits",
    "128 rows, 8 splits",
    *(_capture_id(spec) for spec in _TAGGED),
]


def _next_head(cache: torch.Tensor) -> torch.Tensor:
    """The same single-head view, one KV head further into the pool."""
    return cache.as_strided(cache.shape, cache.stride(), cache.storage_offset() + cache.stride(1))


def test_step5_band_rejects_wrong_variants() -> None:
    """The 4/2-ulp band separates the mistakes a Step-5 call site can make.

    On the tail-first dp 128-row decode cell (head view 0, so the
    neighbouring view is group 1) the op measured 2.0 element / 0.44 RMS ulp
    against `reference`. Each wrong reference lands outside the band, in
    element / RMS ulp: the neighbouring KV group's view 689 / 343; the
    padded head's scale 256^-0.5 instead of 192^-0.5 155 / 25.4; the current
    region dropped 8.4e11 / 422 (a row at position 1 then has a zero
    reference); K and V swapped 710 / 366; `-1` read as token 0 316 / 252.
    """
    spec = STEP5_DECODE
    args = _build(spec, seed=_seed("wrong variants"))
    out = op(**args)
    element, rms = _ulps(out, op.reference(**args))
    assert element <= ELEMENT_ULP and rms <= RMS_ULP, (element, rms)

    no_tail = args["selected_tokens"].clone()
    no_tail[:, : spec["region_size"]] = -1
    wrong = {
        "neighbouring KV group": dict(
            args, k_cache=_next_head(args["k_cache"]), v_cache=_next_head(args["v_cache"])
        ),
        "softmax scale 256^-0.5": dict(args, softmax_scale=256**-0.5),
        "current region dropped": dict(args, selected_tokens=no_tail),
        "K and V swapped": dict(args, k_cache=args["v_cache"], v_cache=args["k_cache"]),
        "padding read as token 0": dict(args, selected_tokens=args["selected_tokens"].clamp_min(0)),
    }
    for name, wrong_args in wrong.items():
        element, rms = _ulps(out, op.reference(**wrong_args))
        assert element > ELEMENT_ULP or rms > RMS_ULP, (name, element, rms)
        with pytest.raises(AssertionError):
            _compare(out, op.reference(**wrong_args))


@pytest.mark.parametrize("spec", STEP5_CAPTURED, ids=STEP5_CAPTURED_IDS)
def test_step5_cuda_graph_capture_and_replay(spec) -> None:
    """The decode target captures this call; a replay must read refilled state.

    Regions then the current region, captured at 1 and 2 rows (64-way
    split-K launched with PDL, then the merge), 8 rows (32-way, BLOCK_N 16,
    merged) and 120 rows (8-way, merged). Current region first, captured at
    2 and 4 rows (64-way, PDL), 16 and 24 rows (32-way), 32, 64 and 128 rows
    (8-way): every launch class of the padded capture batches in the layout
    the target emits, bf16 and fp8 pools. The split-K partials and the output are allocated inside the
    capture, so each replay writes the same tensors. After refilling q, the
    pool, the table, the request slots and the selection in place, every
    replay must equal an eager call on the same tensors bit for bit and pass
    the band.
    """
    args = _build(spec, seed=_seed("graph"))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(**args)  # compile and warm up outside the capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(**args)
    refilled = ("q", "k_cache", "v_cache", "block_table", "request_indices", "selected_tokens")
    for seed in (1, 2):
        fresh = _build(spec, seed=_seed(f"graph {seed}"))
        for name in refilled:
            args[name].copy_(fresh[name])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, op(**args))
        _compare(captured, op.reference(**args))


def test_step5_prefill_pair_is_refused_or_ignored() -> None:
    """Step-5 must call without the prefill pair; passing it is loud or inert.

    At compress ratio 8 a 4104-wide selection implies token_topk 4097, not a
    multiple of 8, so above 512 programs (where the prefill kernel would run)
    the op refuses it. At or below 512 programs the split-K kernel ignores the
    pair: the output is bit-identical to the call without it.
    """
    chunk = dict(STEP5_DECODE, segments=((4000, 600),), splits=1, launches=("splitk",))
    args = _build(chunk, seed=_seed("pair refused"))
    message = "QSA selected-token width is incompatible with its compression ratio"
    with pytest.raises(ValueError, match=re.escape(message)):
        op(**args, query_positions=_positions(chunk), compress_ratio=8)

    args = _build(STEP5_DECODE, seed=_seed("pair ignored"))
    with_pair = op(**args, query_positions=_positions(STEP5_DECODE), compress_ratio=8)
    assert torch.equal(with_pair, op(**args))
