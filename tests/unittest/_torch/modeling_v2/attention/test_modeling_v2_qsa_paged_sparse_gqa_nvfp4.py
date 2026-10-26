# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_sparse_gqa_nvfp4 catalog entry.

The paged state is real: one NVFP4 pool as the cache manager serves it --
e2m1 data `[pages, planes, kv, tpb, D/2]` and e4m3 block scales `[pages,
planes, kv, tpb, D/16]` -- and the engine's slot table, a row per request
slot, more slots than requests, assigned by a seeded permutation, `-1` past
each sequence. The requests sit on scattered pages at the end of the pool and
every scale outside them is NaN, so a stray page address cannot come back
finite.

The QSA cells (24/2 x 256) lay selection rows out the way QSA's top-k +
expansion do, packed at the front. The Step-5 cells hold 4 (attention DP) or 1
(TP4) KV heads per pool and hand the op the single-head views `planes[:, 2l +
{0, 1}, g:g+1]` the target passes per KV group; q is the group's 16-head slice
of a `[T, 16 * heads, 256]` projection. Dims 192..255 are zero in q, and in
the pool they are all-zero 16-value blocks (code 0 under scale 0), as the
NVFP4 append writes a zero block. Their selection is the top-512 complete
regions of 8 tokens in score order (`-1` padded when fewer are visible), each
expanded to its 8 logical positions, plus the current region's 8 columns, `-1`
past the query: after the regions ("regions_then_tail", so a short row's `-1`s
sit between its regions and its tail) or before them ("tail_first").

A cell's `pool_pages` is the pool's slot count, each slot `slot_pages` role pages
of which the call reads layer `layer`'s K and V. The measured-pool cells hold
the slot count the kv_nvfp4 tep4 engine allocated in the trunk layout, and the
dep4 slot pool cells the kv_nvfp4 dep4 engine's four-head slot layout at half
its slot count; both map the last slot under a page the selection reads
(`read_last_slot`), so the call addresses past 2^31 bytes into both the data
and the scale pool; the pools are built in uint8 with no whole-pool temporary
of a wider dtype.

Beside `_compare` against the reference, every cell but the capture sweeps
runs QSA's bf16 sparse GQA over a bf16 pool holding the dequantized values and
holds the two outputs to the same band; a cell with `capture=True` then also
replays its call from a CUDA graph on refilled state.
"""

import math
import re

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention import (
    qsa_paged_sparse_gqa_nvfp4 as _entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_sparse_gqa_nvfp4 import (  # noqa: E501
    _SPLIT_K,
    _spread,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_sparse_gqa_nvfp4 import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4 as op,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import (
    _qsa_pdl_enabled,
    _qsa_splitk_launch_config,
    triton_qsa_paged_sparse_gqa,
)

__extra_import_path__ = [".."]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_paged_sparse_gqa_nvfp4 requires a CUDA device"

DEV = "cuda"
ENGINE_MAX_SEQ_LEN = 8192
#: The e4m3fn NaN bit pattern.
E4M3_NAN = 0x7F
#: The band `_compare` holds, in ulps of q's dtype.
ELEMENT_ULP, RMS_ULP = 4.0, 2.0

_KERNELS = {
    "_nvfp4_sparse_gqa_prefill_kernel": "prefill",
    "_nvfp4_sparse_gqa_splitk_kernel": "splitk",
    "_nvfp4_merge_splitk_kernel": "merge",
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
        monkeypatch.setattr(_entry, name, _LaunchRecorder(getattr(_entry, name), tag, log))
    return log


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 ulp of each (row, head)'s magnitude, 2 ulp relative RMS, in q's dtype.

    After the dequantize the kernels are the bf16 sparse GQA kernels, whose band
    this is (they round the online-softmax probabilities to the value dtype
    before P.V); the dequantize itself is exact at unit scale and rounded once,
    as `reference` rounds it, otherwise. The Step-5 cells' measurement and the
    wrong variants this band rejects are in `test_step5_band_rejects_wrong_variants`.
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


def _seed(text: str) -> int:
    return sum(map(ord, text))


# --------------------------------------------------------------------------
# building real state from a cell spec
# --------------------------------------------------------------------------


def _selected_tokens(positions, seed, compress_ratio, token_topk):
    """QSA's expansion layout: chosen complete groups, the trailing partial group, -1."""
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
    `pos`. Width `512 * 8 + 8` = 4104.
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
    body = body.reshape(rows, topk * region)
    tail = (history * region)[:, None] + offsets[None, :]
    tail = torch.where(tail <= pos[:, None], tail, -1)
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


def _random_pool(pages, kv, tpb, d, real, gen):
    """Pool bytes: any e2m1 nibble, and positive e4m3 scales from 2^-6 to 0.5, for
    dims below `real`; dims from `real` on are all-zero blocks (code 0, scale 0)."""
    data = torch.randint(0, 256, (pages, 2, kv, tpb, d // 2), generator=gen, device=DEV)
    codes = torch.randint(0x08, 0x31, (pages, 2, kv, tpb, d // 16), generator=gen, device=DEV)
    data, codes = data.to(torch.uint8), codes.to(torch.uint8)
    data[..., real // 2 :] = 0
    # K's scales are token-major [tpb, D/16]; V's sit in four-token groups [tpb/4, D/16, 4].
    codes[:, 0, ..., real // 16 :] = 0
    codes[:, 1].view(pages, kv, tpb // 4, d // 16, 4)[..., real // 16 :, :] = 0
    return data, codes


def _build(spec, seed):
    """Real paged state for one spec, keyed by the op's parameter names."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d, q_heads = (spec[k] for k in ("tokens_per_block", "kv_heads", "head_dim", "q_heads"))
    step5 = spec.get("selection") == "step5_regions"
    pool_heads = spec.get("pool_kv_heads", kv)
    view = spec.get("head_view")
    real = spec.get("real_head_dim", d)
    segments = list(spec["segments"])
    positions = _positions(spec)
    req = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    rows, slots = positions.numel(), spec["slots"]
    assert len(segments) <= slots
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(segments)]
    mapped = [math.ceil((s + n) / tpb) for s, n in segments]
    assert not mapped or max(mapped) <= spec["pages"]
    needed = sum(mapped) + 7
    slot_pages, layer = spec["slot_pages"], spec["layer"]
    planes = slice(2 * layer, 2 * layer + 2)
    # The requests sit on the pool's last `needed` pages, and every scale outside them is
    # NaN, so a page address that strayed could not come back finite.
    pool_pages = spec.get("pool_pages", needed)
    assert pool_pages >= needed
    first = pool_pages - needed
    page_ids = first + torch.randperm(needed, generator=gen, device=DEV).to(torch.int32)
    if spec.get("read_last_slot"):
        # The last request's last page holds its last row's own position, which every
        # selection layout reads (the current region): give it the pool's last slot.
        last = sum(mapped) - 1
        at = int((page_ids == pool_pages - 1).nonzero())
        page_ids[[at, last]] = page_ids[[last, at]]
    # Built in the pool's own uint8: no whole-pool temporary in a wider dtype.
    data = torch.zeros(
        pool_pages, slot_pages, pool_heads, tpb, d // 2, dtype=torch.uint8, device=DEV
    )
    scales = torch.full(
        (pool_pages, slot_pages, pool_heads, tpb, d // 16), E4M3_NAN, dtype=torch.uint8, device=DEV
    )
    data[first:, planes], scales[first:, planes] = _random_pool(
        needed, pool_heads, tpb, d, real, gen
    )
    block_table = torch.full((slots, spec["pages"]), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count

    if step5:
        projection = torch.randn(rows, q_heads * pool_heads, d, generator=gen, device=DEV)
        projection[..., real:] = 0
        first_head = q_heads * (view or 0)
        q = projection.to(spec["q_dtype"])[:, first_head : first_head + q_heads]
        selected = _step5_selection(positions, seed + 3, spec)
    else:
        assert spec["q_layout"] == "qkv_slice", spec["q_layout"]
        qkv = torch.randn(rows, q_heads + 2 * kv, d, generator=gen, device=DEV)
        q = qkv.to(spec["q_dtype"])[:, :q_heads]
        selected = _selected_tokens(positions, seed + 3, spec["compress_ratio"], spec["token_topk"])
    request_indices = slot_of.to(DEV)[req].to(spec["request_dtype"])

    def plane(pool, role):
        role_plane = pool[:, 2 * layer + role]
        return role_plane if view is None else role_plane[:, view : view + 1]

    args = dict(
        q=q,
        k_cache=plane(data, 0),
        v_cache=plane(data, 1),
        k_scale_cache=plane(scales, 0),
        v_scale_cache=plane(scales, 1),
        kv_scale_quant_orig=torch.tensor(
            [1.0, spec["global_k"], spec["global_v"]], dtype=torch.float32, device=DEV
        ),
        block_table=block_table,
        selected_tokens=selected,
        request_indices=request_indices,
        tokens_per_block=tpb,
        softmax_scale=spec.get("softmax_scale", d**-0.5),
    )
    assert args["k_cache"].shape[1] == kv
    if spec["prefill_pair"]:
        args.update(query_positions=positions, compress_ratio=spec["compress_ratio"])
    return args


#: Pages dequantized per step when the bf16 twin is filled.
_TWIN_CHUNK_PAGES = 2048


def _bf16_twin(args):
    """QSA's bf16 sparse GQA over a bf16 pool holding the dequantized values.

    The twin is a contiguous `[pages, H, tpb, D]` pool at the same page ids; the
    pages the table maps hold the dequantized values, every other page NaN (as the
    NVFP4 pool's NaN scales dequantize), filled a chunk of pages at a time.
    """
    pages, heads, tpb = args["k_cache"].shape[:3]
    head_dim = args["q"].shape[2]
    table = args["block_table"]
    mapped = table[(table >= 0) & (table < pages)].long().unique()
    slots = torch.arange(tpb, device=DEV)[None, :]
    planes = []
    for data, scales, index, grouped in (
        (args["k_cache"], args["k_scale_cache"], 1, False),
        (args["v_cache"], args["v_scale_cache"], 2, True),
    ):
        twin = torch.full(
            (pages, heads, tpb, head_dim), float("nan"), dtype=args["q"].dtype, device=DEV
        )
        for start in range(0, mapped.numel(), _TWIN_CHUNK_PAGES):
            chunk = mapped[start : start + _TWIN_CHUNK_PAGES]
            values = _entry._dequantize_tokens(
                data,
                scales,
                chunk[:, None].expand(-1, tpb),
                slots.expand(chunk.numel(), -1),
                args["kv_scale_quant_orig"][index],
                grouped,
            )  # [chunk, tpb, H, D]
            twin[chunk] = values.to(twin.dtype).transpose(1, 2)
        planes.append(twin)
    twin = {
        name: args[name]
        for name in ("q", "block_table", "selected_tokens", "request_indices", "tokens_per_block")
    }
    twin.update(softmax_scale=args["softmax_scale"], k_cache=planes[0], v_cache=planes[1])
    if "query_positions" in args:
        twin.update(query_positions=args["query_positions"], compress_ratio=args["compress_ratio"])
    return triton_qsa_paged_sparse_gqa(**twin)


def _split_columns(rows: int, kv_heads: int, width: int):
    """Per split of the split-K launch for this call: (its leading tile's columns,
    all its columns). Split `s` takes tiles `[s * tiles // splits, (s + 1) *
    tiles // splits)` of `block_n` columns."""
    block_n, splits, _ = _qsa_splitk_launch_config(rows, kv_heads, width)
    tiles = math.ceil(width / block_n)
    for split in range(splits):
        first, last = split * tiles // splits, (split + 1) * tiles // splits
        start = first * block_n
        yield slice(start, min(start + block_n, width)), slice(start, min(last * block_n, width))


def _empty_leading_tile_rows(selected: torch.Tensor, kv_heads: int) -> torch.Tensor:
    """Rows where a split opens on a tile naming no token and names one later."""
    rows, width = selected.shape
    named = selected >= 0
    hazard = torch.zeros(rows, dtype=torch.bool, device=selected.device)
    for lead, body in _split_columns(rows, kv_heads, width):
        hazard |= ~named[:, lead].any(1) & named[:, body].any(1)
    return hazard


def _bound(positions, spec):
    cr = spec["compress_ratio"]
    visible = positions.long() + 1
    return (visible // cr).clamp(max=spec["token_topk"] // cr) * cr + visible % cr


def _is_prefix_packed(table):
    valid = table >= 0
    columns = torch.arange(table.shape[1], device=table.device)[None, :]
    return (valid == (columns < valid.sum(1)[:, None])).all(1)


def _assert_reads_last_slot(args) -> None:
    """The call reads the pool's last slot, past 2^31 bytes into the data and the
    scale pools both, so a page offset formed in 32 bits could not come back right."""
    selected, table = args["selected_tokens"].long(), args["block_table"]
    named = selected >= 0
    logical = selected.clamp_min(0) // args["tokens_per_block"]
    requests = args["request_indices"].long()[:, None].expand_as(logical)
    physical = table[requests, logical.clamp(max=table.shape[1] - 1)]
    last = args["k_cache"].shape[0] - 1
    assert bool((named & (physical == last)).any()), "no selected token sits in the last slot"
    for name in ("k_cache", "v_cache", "k_scale_cache", "v_scale_cache"):
        view = args[name]
        offset = last * view.stride(0) + view.storage_offset()
        print(f"{name}: the last slot starts {offset} bytes into its pool")
        assert offset >= 2**31, (name, offset)


_OPERANDS = (
    "q",
    "k_cache",
    "v_cache",
    "k_scale_cache",
    "v_scale_cache",
    "block_table",
    "selected_tokens",
    "request_indices",
)


# --------------------------------------------------------------------------
# CUDA graphs
# --------------------------------------------------------------------------


def _capture_sweep(spec, seed, launches):
    """The decode site under a CUDA graph, at each batch the cell names: capture,
    write a second state's operands in place, replay, and require the eager call
    on that second state bitwise."""
    sweep = spec["capture"]
    if "batches" in sweep:
        for rows, splits in sweep["batches"]:
            shape = {**spec, "segments": _spread(rows), "splits": splits, "launches": _SPLIT_K}
            _capture_once(shape, seed + rows, launches, where=f"{rows} rows")
        return
    for per_request in sweep["rows_per_request"]:
        for groups in sweep["groups"]:
            _capture_once(
                _decode_step(spec, groups, per_request),
                seed + 100 * per_request + groups,
                launches,
                where=f"G={groups} x {per_request} rows",
            )


def _start(request: int, rows: int) -> int:
    """A request's first position, across the budget."""
    return min(100 + (499 * request) % 8000, ENGINE_MAX_SEQ_LEN - rows)


def _decode_step(spec, groups, per_request):
    """The decode site's step: `groups` requests x `per_request` rows."""
    segments = tuple((_start(g, per_request), per_request) for g in range(groups))
    return {**spec, "segments": segments}


def _capture_once(shape, seed, launches, where) -> None:
    """One call of `shape` captured, a second state's operands written in place,
    replayed: the replay must be bitwise the eager call on the second state."""
    where = f"{where}, [{shape['slots']}, {shape['pages']}] table"
    args = _build(shape, seed)
    fresh = _build(shape, seed + 50)
    launches.clear()
    op(**args)
    assert [name for name, _ in launches] == list(shape["launches"]), where
    if "splits" in shape:
        assert launches[0][1] == (args["q"].shape[0], shape["kv_heads"], shape["splits"]), where
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = op(**args)
        for name in _OPERANDS:
            args[name].copy_(fresh[name])
        graph.replay()
        torch.cuda.synchronize()
        eager = op(**fresh)
        assert torch.equal(captured, eager), f"replay differs from eager at {where}"
        _compare(captured, op.reference(**fresh))
    finally:
        del graph
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# the cells
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying, launches) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    sweep = spec.get("capture")
    if isinstance(sweep, dict):
        _capture_sweep(spec, seed=_seed(cell.why), launches=launches)
        return
    args = _build(spec, seed=_seed(cell.why))
    rows = args["q"].shape[0]
    if spec["prefill_pair"]:
        selected = args["selected_tokens"]
        assert bool(_is_prefix_packed(selected).all())
        valid = (selected >= 0).sum(1)
        assert torch.equal(valid, _bound(args["query_positions"], spec))
    if "splits" in spec:
        _check_step5_selection(args["selected_tokens"], _positions(spec), spec)
        pdl = spec["splits"] > 1 and _qsa_pdl_enabled(rows)
        assert pdl == spec.get("pdl", False)
    if spec.get("read_last_slot"):
        _assert_reads_last_slot(args)

    out = op(**args)
    assert [name for name, _ in launches] == list(spec["launches"])
    assert launches[0][1][:2] == (rows, spec["kv_heads"])
    if "splits" in spec:
        assert launches[0][1] == (rows, spec["kv_heads"], spec["splits"])
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == args["q"].dtype and out.is_contiguous()
    assert not bool(out.isnan().any())
    if "real_head_dim" in spec:
        # The zero pad blocks of V average to exact zeros.
        assert not bool(out[..., spec["real_head_dim"] :].any())
    _compare(out, op.reference(**args))
    # QSA's bf16 kernels over the same values: inside the same band, not bitwise
    # (the dequantizing load changes the compiled kernel). Rows that open a split
    # on an empty leading tile depend on the guarded correction (see the entry's
    # note); `test_step5_split_after_an_empty_leading_tile` covers them, and they
    # are left out of this comparison.
    twin_rows = ~_empty_leading_tile_rows(args["selected_tokens"], spec["kv_heads"])
    _compare(out[twin_rows], _bf16_twin(args)[twin_rows])
    if sweep:
        # The decode target captures this call at this padded batch.
        _capture_once(spec, _seed(cell.why) + 1, launches, where=f"{rows} rows")


CKPT = op.CELLS[0].spec
assert (CKPT["q_heads"], CKPT["kv_heads"], CKPT["head_dim"]) == (24, 2, 256)


def _with(**overrides):
    return {**CKPT, **overrides}


def test_calls_are_bitwise_deterministic() -> None:
    args = _build(_with(segments=tuple((4000 + 37 * i, 2) for i in range(4))), seed=8800)
    assert torch.equal(op(**args), op(**args))


def test_rejects_positions_without_a_compress_ratio() -> None:
    """The op's own loud refusal -- the reason the entry carries no `is_valid`: half of
    the prefill pair."""
    args = _build(_with(segments=((600, 1),) * 4), seed=9100)
    args["selected_tokens"] = args["selected_tokens"][:, :64].contiguous()
    args["query_positions"] = torch.zeros(4, dtype=torch.int64, device=DEV)
    message = "pass both query_positions and compress_ratio, or neither"
    with pytest.raises(ValueError, match=re.escape(message)):
        with validating(op):
            op(**args)


# --------------------------------------------------------------------------
# Step-5-Preview
# --------------------------------------------------------------------------


def _step5_cell(prefix: str):
    (cell,) = [c for c in op.CELLS if c.why.startswith(prefix)]
    return cell


STEP5_DECODE = _step5_cell("Step-5 dp 128 decode rows")
STEP5_PREFILL = _step5_cell("Step-5 dp 8192-row prefill from 0")


def _next_head(cache: torch.Tensor) -> torch.Tensor:
    """The same single-head view, one KV head further into the pool."""
    return cache.as_strided(cache.shape, cache.stride(), cache.storage_offset() + cache.stride(1))


def _swap_nibbles(data: torch.Tensor) -> torch.Tensor:
    """The pool read with the high nibble as the even element."""
    wide = data.int()
    return (((wide & 0xF) << 4) | (wide >> 4)).to(torch.uint8)


def _token_major(scales: torch.Tensor) -> torch.Tensor:
    """V's scale bytes re-laid so `reference`'s grouped read sees them token-major,
    as K's are: what a reader that ignored V's four-token groups would use."""
    pages, heads, tpb, groups = scales.shape
    flat = scales.contiguous().view(pages, heads, tpb // 4, 4, groups)
    return flat.transpose(-1, -2).contiguous().view(pages, heads, tpb, groups)


def test_step5_band_rejects_wrong_variants() -> None:
    """The 4/2-ulp band separates the mistakes a Step-5 call site can make.

    On the regions-then-tail dp 128-row decode cell (head view 1, so the
    neighbouring view is group 2) each wrong reference must land outside the
    band; the measured element / RMS ulps are in the entry's note.
    """
    spec = STEP5_DECODE.spec
    args = _build(spec, seed=_seed(STEP5_DECODE.why))
    out = op(**args)
    element, rms = _ulps(out, op.reference(**args))
    print(f"op vs reference: {element:.3g} element ulp, {rms:.3g} RMS ulp")
    assert element <= ELEMENT_ULP and rms <= RMS_ULP, (element, rms)

    no_tail = args["selected_tokens"].clone()
    no_tail[:, -spec["region_size"] :] = -1
    wrong = {
        "neighbouring KV group": dict(
            args,
            k_cache=_next_head(args["k_cache"]),
            v_cache=_next_head(args["v_cache"]),
            k_scale_cache=_next_head(args["k_scale_cache"]),
            v_scale_cache=_next_head(args["v_scale_cache"]),
        ),
        "softmax scale 256^-0.5": dict(args, softmax_scale=256**-0.5),
        "current region dropped": dict(args, selected_tokens=no_tail),
        "K and V swapped": dict(
            args,
            k_cache=args["v_cache"],
            v_cache=args["k_cache"],
            k_scale_cache=args["v_scale_cache"],
            v_scale_cache=args["k_scale_cache"],
        ),
        "padding read as token 0": dict(args, selected_tokens=args["selected_tokens"].clamp_min(0)),
        "nibbles swapped": dict(
            args, k_cache=_swap_nibbles(args["k_cache"]), v_cache=_swap_nibbles(args["v_cache"])
        ),
        "V scales token-major": dict(args, v_scale_cache=_token_major(args["v_scale_cache"])),
    }
    for name, wrong_args in wrong.items():
        element, rms = _ulps(out, op.reference(**wrong_args))
        print(f"wrong variant {name}: {element:.3g} element ulp, {rms:.3g} RMS ulp")
        assert element > ELEMENT_ULP or rms > RMS_ULP, (name, element, rms)
        with pytest.raises(AssertionError):
            _compare(out, op.reference(**wrong_args))


@pytest.mark.parametrize(
    "cell", [STEP5_DECODE, STEP5_PREFILL], ids=["128 decode rows, 8 splits", "8192 rows, 1 split"]
)
def test_step5_split_after_an_empty_leading_tile(cell) -> None:
    """Detector: a split whose leading tile names no token keeps the tokens after it.

    In the regions-then-tail layout a row with fewer than 512 regions has `-1`
    region slots between its regions and its current region, so a split can
    open on a tile with nothing resident and hold the current region (the
    query's own token included) further on. This finds those rows from the
    split-K launch's own split boundaries, requires the cell to have them, holds
    the op to the band on exactly them, and shows that the reference without
    those splits' tokens lands outside the band: an op that dropped them could
    not pass.
    """
    spec = cell.spec
    args = _build(spec, seed=_seed(cell.why))
    selected = args["selected_tokens"]
    rows, width = selected.shape
    hazard = _empty_leading_tile_rows(selected, spec["kv_heads"])
    count = int(hazard.sum())
    print(f"{cell.why[:44]}: {count} of {rows} rows open a split on an empty tile")
    assert count > 0

    out = op(**args)
    _compare(out[hazard], op.reference(**args)[hazard])

    named = selected >= 0
    dropped = selected.clone()
    for lead, body in _split_columns(rows, spec["kv_heads"], width):
        empty_lead = ~named[:, lead].any(1)
        dropped[:, body] = torch.where(empty_lead[:, None], -1, dropped[:, body])
    without = op.reference(**dict(args, selected_tokens=dropped))
    element, rms = _ulps(out[hazard], without[hazard])
    print(f"op vs reference without those splits: {element:.3g} element ulp, {rms:.3g} RMS ulp")
    with pytest.raises(AssertionError):
        _compare(out[hazard], without[hazard])
