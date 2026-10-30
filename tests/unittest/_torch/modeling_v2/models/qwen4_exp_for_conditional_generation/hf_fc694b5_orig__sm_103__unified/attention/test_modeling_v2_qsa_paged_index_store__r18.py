# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_index_store catalog entry.

The paged state is real: an index-key plane and a position plane of `pages`
pages, and the engine's `[slots, width]` table -- more slots than requests,
each live row's pages drawn from a seeded permutation of the pool, `-1` past
its allocation, rows past the batch holding a previous step's plausible page
ids. Both planes are pre-filled with a sentinel so a stray write shows.
A `hole` cell appends one row that resolves to no slot.

The unified target's cells store into the stock manager's bf16 layout -- one
layer's plane of the 12 sparse layers' coalesced index pool, the other 11
checked untouched -- at the measured tep4 pool, with the batch's last row on the
pool's last page. `test_the_stock_managers_planes_are_the_cells_layout` builds
the stock manager itself (through the target's pool-layout test) and stores
through its own views.
"""

import importlib.util
import zlib
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.attention.qsa_paged_index_store__r18 import (  # noqa: E501
    qsa_paged_index_store as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_paged_index_store requires a CUDA device"


def _compare(out, ref) -> None:
    """Bitwise over both whole planes, as raw bits.

    A store is a move with one conversion that `reference` performs too, so
    there is nothing to spend a tolerance on; comparing every slot rather
    than the written ones is the stray-write check.
    """
    for name, got, want in zip(("index_cache", "position_cache"), out, ref):
        assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
        assert got.shape == want.shape, (name, got.shape, want.shape)
        torch.testing.assert_close(
            _bits(got), _bits(want), rtol=0, atol=0, msg=lambda m, n=name: f"{n}: {m}"
        )


DEV = "cuda"
KEY_FILL = -7.0
POS_FILL = -7
ENGINE_MAX_NUM_TOKENS = 16384


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    as_int = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.contiguous().view(as_int)


def _hole(spec, table, allocated_cols):
    """One row that resolves to no slot: its (request, position)."""
    assert spec["hole"] == "unallocated_page", spec["hole"]
    assert int(table[0, allocated_cols[0]]) == -1
    return 0, allocated_cols[0] * spec["tokens_per_block"]


def _shared_page_planes(spec, extras: dict):
    """The NVFP4-KV pool's side planes: each physical page holds the `layers`
    index-K pages and then the position page, so the index cache (the last
    role) and the position cache are strided views of one byte allocation.
    Every role holds `KEY_FILL` and the position page `POS_FILL`; the other
    roles are recorded for the untouched check."""
    layers, tpb, hkv, head_dim, axes, pages = (
        spec[k] for k in ("layers", "tokens_per_block", "hkv", "head_dim", "axes", "pages")
    )
    index_bytes = tpb * hkv * head_dim * 2
    page = torch.empty(pages, layers * index_bytes + tpb * axes * 4, dtype=torch.uint8, device=DEV)
    page[:, : layers * index_bytes].view(torch.bfloat16).fill_(KEY_FILL)
    page[:, layers * index_bytes :].view(torch.int32).fill_(POS_FILL)
    index_cache = page[:, (layers - 1) * index_bytes : layers * index_bytes].view(torch.bfloat16)
    index_cache = index_cache.view(pages, tpb, hkv, head_dim)
    position_cache = page[:, layers * index_bytes :].view(torch.int32).view(pages, tpb, axes)
    assert index_cache.stride() == (spec["page_stride"], hkv * head_dim, head_dim, 1)
    assert position_cache.stride() == spec["position_strides"]
    extras["roles"] = page[:, : (layers - 1) * index_bytes]
    return index_cache, position_cache


def _coalesced_planes(spec, extras: dict):
    """The bf16 index pool of the unified target: the `layers` sparse layers' INDEX_KEY buffers
    coalesced into one allocation, so the cell's plane (layer `layer`) is a strided view, and the
    position plane its own contiguous pool. The whole allocation holds `KEY_FILL`; the pool is
    recorded for the untouched check."""
    layers, layer, tpb, hkv, head_dim, axes, pages = (
        spec[k] for k in ("layers", "layer", "tokens_per_block", "hkv", "head_dim", "axes", "pages")
    )
    pool = torch.full(
        (pages, layers, tpb, hkv, head_dim), KEY_FILL, dtype=torch.bfloat16, device=DEV
    )
    index_cache = pool[:, layer]
    assert index_cache.stride() == (spec["page_stride"], hkv * head_dim, head_dim, 1)
    position_cache = torch.full((pages, tpb, axes), POS_FILL, dtype=torch.int32, device=DEV)
    extras["coalesced"] = (pool, layer)
    return index_cache, position_cache


def _other_layers_untouched(pool: torch.Tensor, layer: int) -> None:
    """Every other layer's plane in the coalesced pool still holds `KEY_FILL`, checked a band of
    pages at a time so the check stays a fraction of the pool's size."""
    fill = int(_bits(torch.tensor([KEY_FILL], dtype=torch.bfloat16))[0])
    bits = pool.view(torch.int16)
    for start in range(0, pool.shape[0], 16384):
        band = bits[start : start + 16384]
        for other in range(pool.shape[1]):
            if other != layer:
                assert bool((band[:, other] == fill).all()), (
                    f"layer {other}'s index plane moved in pages {start}..{start + band.shape[0]}"
                )


def _build(spec, seed: int, planes=None):
    """Real planes, table and rows for one cell; returns the op's kwargs and extras.

    `planes`, if given, is an `(index_cache, position_cache)` pair to store into instead of the
    cell's own (a stock manager's views)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tpb, hkv, head_dim, axes = (spec[k] for k in ("tokens_per_block", "hkv", "head_dim", "axes"))
    slots, width, pages = spec["slots"], spec["width"], spec["pages"]
    segments = spec["segments"]
    requests = [r for r, (_, n) in enumerate(segments) for _ in range(n)]
    positions = [p for start, n in segments for p in range(start, start + n)]
    rows = len(positions)

    table = torch.randint(0, pages, (slots, width), generator=gen, dtype=torch.int32)
    order = torch.randperm(pages, generator=gen).to(torch.int32)
    used, allocated_cols = 0, []
    for r, (start, n) in enumerate(segments):
        cols = (start + n - 1) // tpb + 1
        table[r] = -1
        table[r, :cols] = order[used : used + cols]
        used += cols
        allocated_cols.append(cols)
    assert used <= pages and len(segments) <= slots, "pool or table too small for the batch"
    if spec.get("straddle"):
        # The batch's last row stores into the pool's last page; every other row keeps its page
        # from the permutation, so the rows land on both sides of 2^31 elements of the plane.
        last_start, last_rows = segments[-1]
        column = (last_start + last_rows - 1) // tpb
        live = table[: len(segments)]
        holder = (live == pages - 1).nonzero()
        if len(holder):
            live[holder[0, 0], holder[0, 1]] = live[len(segments) - 1, column]
        live[len(segments) - 1, column] = pages - 1
    if spec["hole"] is not None:
        request, position = _hole(spec, table, allocated_cols)
        requests.append(request)
        positions.append(position)
        rows += 1
    table = table.to(DEV)

    projection = torch.randn(
        rows, spec["index_heads"] + hkv, head_dim, generator=gen, dtype=torch.float32
    ).to(device=DEV, dtype=spec["key_dtype"])
    token_k = projection[:, spec["index_heads"] :]
    if spec["key_layout"] == "fused_row":
        # The target's fused input projection row, [q | k | v | gate | index q, k |
        # pad of one index head], in heads of 128: the raw index key is the second to
        # last (108 of 110 at 14080 wide; 32 of 34 at the unified target's 4352).
        heads = spec.get("row_width", 14080) // head_dim
        row = torch.randn(rows, heads, head_dim, generator=gen, dtype=torch.float32)
        token_k = row.to(device=DEV, dtype=spec["key_dtype"])[:, heads - 2 : heads - 2 + hkv]
        assert token_k.stride(0) == heads * head_dim

    request_indices = torch.tensor(requests, dtype=spec["request_dtype"], device=DEV)
    logical_positions = torch.tensor(positions, dtype=spec["position_dtype"], device=DEV)

    # Three genuinely different axes, as an image span has: t, t + 7, t + 14.
    coords = logical_positions.to(torch.int32)[:, None] + 7 * torch.arange(
        3, dtype=torch.int32, device=DEV
    )
    # The engine's `[3, max_num_tokens]` position buffer, transposed.
    buffer = torch.full((3, ENGINE_MAX_NUM_TOKENS), -1, dtype=torch.int32, device=DEV)
    buffer[:, :rows] = coords.t()
    coords = buffer[:, :rows].t()

    extras = {}
    position_cache = None
    if planes is not None:
        index_cache, position_cache = planes
    elif spec["cache_layout"] == "shared_page":
        index_cache, position_cache = _shared_page_planes(spec, extras)
    elif spec["cache_layout"] == "coalesced":
        index_cache, position_cache = _coalesced_planes(spec, extras)
    else:
        index_cache = torch.full((pages, tpb, hkv, head_dim), KEY_FILL, device=DEV).to(
            torch.bfloat16
        )
    if position_cache is None:
        position_cache = torch.full((pages, tpb, axes), POS_FILL, dtype=torch.int32, device=DEV)
    args = dict(
        token_k=token_k,
        position_coordinates=coords,
        index_cache=index_cache,
        position_cache=position_cache,
        request_indices=request_indices,
        logical_positions=logical_positions,
        block_table=table,
        tokens_per_block=tpb,
    )
    return args, extras


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args, extras = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    if cell.spec.get("straddle"):
        # The last row's page starts past 2^31 elements of the plane it is stored through.
        assert (cell.spec["pages"] - 1) * args["index_cache"].stride(0) >= 2**31
    ref = op.reference(**args)
    out = op(**args)
    assert out[0] is args["index_cache"] and out[1] is args["position_cache"]
    _compare(out, ref)
    if "roles" in extras:
        fill = int(_bits(torch.tensor([KEY_FILL], dtype=torch.bfloat16))[0])
        assert bool((extras["roles"].view(torch.int16) == fill).all()), (
            "another index-K role of the shared page moved"
        )
    if "coalesced" in extras:
        _other_layers_untouched(*extras["coalesced"])


#: The cell of 32 MTP7 verify blocks (256 rows): the segments, pool and table it runs over.
_MTP7_32 = next(c.spec for c in op.CELLS if c.why.startswith("decode MTP7: 32 requests"))
_MTP7_BATCH = {key: _MTP7_32[key] for key in ("segments", "pages", "slots")}


def _small(seed: int, **overrides):
    """Four decode rows at the checkpoint's call-site layouts."""
    spec = {**_MTP7_32, "segments": ((0, 1), (7, 1), (32, 1), (40, 1)), "pages": 8}
    return _build({**spec, **overrides}, seed=seed)[0]


def test_cuda_graph_capture_and_replay() -> None:
    """Captured once at the 32 MTP7 verify blocks (256 rows), replayed against new
    positions written into the same buffer."""
    args = _small(seed=52, **_MTP7_BATCH)
    positions = args["logical_positions"]
    replayed = positions.clone()
    positions.zero_()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(**args)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**args)

    positions.copy_(replayed)
    args["index_cache"].fill_(KEY_FILL)
    args["position_cache"].fill_(POS_FILL)
    ref = op.reference(**args)
    graph.replay()
    torch.cuda.synchronize()
    _compare((args["index_cache"], args["position_cache"]), ref)


#: The unified target's 128-request decode cell, the largest batch its decode graphs capture.
_UNIFIED_DECODE_128 = next(
    c.spec for c in op.CELLS if c.why.startswith("unified tep4 decode: 128 requests")
)


def test_cuda_graph_capture_and_replay_unified_decode() -> None:
    """Captured once at the unified target's 128-request decode on its coalesced plane, replayed
    against new positions written into the same buffer."""
    args, extras = _build(_UNIFIED_DECODE_128, seed=53)
    positions = args["logical_positions"]
    replayed = positions.clone()
    positions.zero_()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(**args)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**args)

    positions.copy_(replayed)
    pool, layer = extras["coalesced"]
    pool.fill_(KEY_FILL)
    args["position_cache"].fill_(POS_FILL)
    ref = op.reference(**args)
    graph.replay()
    torch.cuda.synchronize()
    _compare((args["index_cache"], args["position_cache"]), ref)
    _other_layers_untouched(pool, layer)


def _pool_layout():
    """The target's pool-layout test, loaded by path: its `_manager` builds the stock manager the
    engine builds on a rank of a member (core cache config, bf16 KV), and `ATTN_LAYERS` names the
    sparse layers. Loaded rather than restated so the two cannot disagree on what the stock pool
    is."""
    path = Path(__file__).resolve().parents[1] / "test_pool_layout.py"
    spec = importlib.util.spec_from_file_location("_r18_pool_layout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("member", ("tp4", "tep4", "dep4"))
def test_the_stock_managers_planes_are_the_cells_layout(member) -> None:
    """The stock manager's own index and position planes at rank 0 of `member` have the layout
    the member's cells model, and the op stores through them with the member's row width: four
    rows, the last on the pool's last page, compared against `reference` over the whole plane,
    every other sparse layer's plane untouched."""
    layout = _pool_layout()
    cell = next(c.spec for c in op.CELLS if c.spec.get("member") == member)
    tpb, hkv, head_dim, axes = (cell[k] for k in ("tokens_per_block", "hkv", "head_dim", "axes"))
    with layout._manager(member, 0) as cm:
        planes = [cm.get_index_k_buffer(i) for i in layout.ATTN_LAYERS]
        index_cache, position_cache = planes[cell["layer"]], cm.get_qsa_position_buffer()
        assert len(planes) == cell["layers"]
        assert index_cache.stride() == (cell["page_stride"], hkv * head_dim, head_dim, 1)
        assert position_cache.stride() == (tpb * axes, axes, 1)
        assert position_cache.shape[0] == index_cache.shape[0]
        plane_bytes = tpb * hkv * head_dim * index_cache.element_size()
        offsets = [p.data_ptr() - planes[0].data_ptr() for p in planes]
        assert offsets == [j * plane_bytes for j in range(len(planes))]

        for plane in planes:
            plane.fill_(KEY_FILL)
        position_cache.fill_(POS_FILL)
        spec = dict(
            cell,
            segments=((0, 1), (31, 1), (33, 2)),
            pages=index_cache.shape[0],
            slots=8,
        )
        args, _ = _build(spec, seed=71, planes=(index_cache, position_cache))
        assert int(args["block_table"][2, 1]) == index_cache.shape[0] - 1
        ref = op.reference(**args)
        op(**args)
        _compare((index_cache, position_cache), ref)
        fill = int(_bits(torch.tensor([KEY_FILL], dtype=torch.bfloat16))[0])
        for j, plane in enumerate(planes):
            if j != cell["layer"]:
                assert bool((_bits(plane) == fill).all()), f"sparse layer {j}'s plane moved"


def test_rejects_metadata_of_unequal_length() -> None:
    """A shape the kernel would otherwise read past; the entry refuses it before
    launching, so `is_valid` has nothing left to guard."""
    args = _small(seed=61)
    args["logical_positions"] = args["logical_positions"][:1]
    before = (args["index_cache"].clone(), args["position_cache"].clone())
    with pytest.raises(ValueError, match="equally long"):
        op(**args)
    assert torch.equal(_bits(args["index_cache"]), _bits(before[0]))
    assert torch.equal(args["position_cache"], before[1])
