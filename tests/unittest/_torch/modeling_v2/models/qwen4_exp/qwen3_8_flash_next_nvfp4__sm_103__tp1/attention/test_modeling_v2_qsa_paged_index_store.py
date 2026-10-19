# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_index_store catalog entry.

The paged state is real: an index-key plane and a position plane of `pages`
pages, and the engine's `[slots, width]` table -- more slots than requests,
each live row's pages drawn from a seeded permutation of the pool, `-1` past
its allocation, rows past the batch holding a previous step's plausible page
ids. Both planes are pre-filled with a sentinel so a stray write shows.
A `hole` cell appends one row that resolves to no slot.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_paged_index_store import (  # noqa: E501
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


def _hole(spec, table, live_requests, allocated_cols):
    """One row that resolves to no slot, and the table edit it needs."""
    tpb, width, slots, pages = (spec[k] for k in ("tokens_per_block", "width", "slots", "pages"))
    kind = spec["hole"]
    if kind == "unallocated_page":
        assert int(table[0, allocated_cols[0]]) == -1
        return 0, allocated_cols[0] * tpb
    if kind == "page_past_pool":
        table[live_requests] = -1
        table[live_requests, 0] = pages + 7
        return live_requests, 0
    if kind == "request_past_table":
        return slots, 0
    if kind == "negative_request":
        return -1, 0
    if kind == "negative_position":
        return 0, -1
    assert kind == "column_past_table"
    return 0, width * tpb


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


def _build(spec, seed: int):
    """Real planes, table and rows for one cell; returns the op's kwargs and extras."""
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
    assert used <= pages and len(segments) < slots, "pool or table too small for the batch"
    if spec["hole"] is not None:
        request, position = _hole(spec, table, len(segments), allocated_cols)
        requests.append(request)
        positions.append(position)
        rows += 1
    table = table.to(DEV)
    if spec["table_layout"] == "column_slice":
        wide = torch.full((slots, width + 4), -1, dtype=torch.int32, device=DEV)
        wide[:, 2 : 2 + width] = table
        table = wide[:, 2 : 2 + width]
        assert not table.is_contiguous()
    elif spec["table_layout"] == "transposed":
        table = table.t().contiguous().t()
        assert table.stride(1) != 1

    projection = torch.randn(
        rows, spec["index_heads"] + hkv, head_dim, generator=gen, dtype=torch.float32
    ).to(device=DEV, dtype=spec["key_dtype"])
    token_k = projection[:, spec["index_heads"] :]
    if spec["key_layout"] == "fused_row":
        # The target's fused input projection row, [q | k | v | gate | index q, k |
        # pad] = 110 heads of 128: the raw index key is head 108.
        row = torch.randn(rows, 110, head_dim, generator=gen, dtype=torch.float32)
        token_k = row.to(device=DEV, dtype=spec["key_dtype"])[:, 108 : 108 + hkv]
    if spec["key_layout"] == "contiguous":
        token_k = token_k.contiguous()

    request_indices = torch.tensor(requests, dtype=spec["request_dtype"], device=DEV)
    logical_positions = torch.tensor(positions, dtype=spec["position_dtype"], device=DEV)
    if spec["meta_layout"] == "strided":
        wide_requests = torch.zeros(rows, 2, dtype=spec["request_dtype"], device=DEV)
        wide_positions = torch.zeros(rows, 3, dtype=spec["position_dtype"], device=DEV)
        wide_requests[:, 0] = request_indices
        wide_positions[:, 1] = logical_positions
        request_indices, logical_positions = wide_requests[:, 0], wide_positions[:, 1]
        assert not logical_positions.is_contiguous()

    # Three genuinely different axes, as an image span has: t, t + 7, t + 14.
    coords = logical_positions.to(torch.int32)[:, None] + 7 * torch.arange(
        3, dtype=torch.int32, device=DEV
    )
    if spec["coords_layout"] == "transposed":
        # The engine's `[3, max_num_tokens]` position buffer, transposed.
        buffer = torch.full((3, ENGINE_MAX_NUM_TOKENS), -1, dtype=torch.int32, device=DEV)
        buffer[:, :rows] = coords.t()
        coords = buffer[:, :rows].t()
    elif spec["coords_layout"] == "stride0":
        coords = logical_positions[:, None].expand(-1, 3)
        assert coords.stride(1) == 0
    else:
        coords = coords.contiguous()

    extras = {}
    position_cache = None
    if spec["cache_layout"] == "shared_page":
        index_cache, position_cache = _shared_page_planes(spec, extras)
    elif spec["cache_layout"] == "role_view":
        pool = torch.full((pages, 2, tpb, hkv, head_dim), KEY_FILL, device=DEV).to(torch.bfloat16)
        index_cache = pool[:, 0]
        extras["neighbour"] = (pool[:, 1], pool[:, 1].clone())
        assert not index_cache.is_contiguous()
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
    before = args["position_cache"].clone()
    ref = op.reference(**args)
    out = op(**args)
    assert out[0] is args["index_cache"] and out[1] is args["position_cache"]
    _compare(out, ref)
    if "neighbour" in extras:
        view, saved = extras["neighbour"]
        assert torch.equal(_bits(view), _bits(saved)), "the neighbouring role moved"
    if "roles" in extras:
        fill = int(_bits(torch.tensor([KEY_FILL], dtype=torch.bfloat16))[0])
        assert bool((extras["roles"].view(torch.int16) == fill).all()), (
            "another index-K role of the shared page moved"
        )
    if cell.spec["axes"] > 3:
        assert torch.equal(out[1][..., 3:], before[..., 3:]), "axes past the coordinates moved"
    if cell.spec["coords_layout"] == "stride0":
        for axis in (1, 2):
            assert torch.equal(out[1][..., axis], out[1][..., 0])


def _small(seed: int, **overrides):
    """Four decode rows at the checkpoint's call-site layouts."""
    spec = {**op.CELLS[0].spec, "segments": ((0, 1), (7, 1), (32, 1), (40, 1)), "pages": 8}
    return _build({**spec, **overrides}, seed=seed)[0]


def test_only_the_named_slots_change() -> None:
    """The exact set of (page, slot) pairs written, independent of the reference."""
    args = _small(seed=13)
    before_index = args["index_cache"].clone()
    before_position = args["position_cache"].clone()
    op(**args)
    table = args["block_table"]
    written = {(int(table[r, p // 32]), p % 32) for r, p in enumerate((0, 7, 32, 40))}
    changed_index = {
        (int(p), int(s))
        for p, s in torch.nonzero(
            (_bits(args["index_cache"]) != _bits(before_index)).flatten(2).any(-1)
        )
    }
    changed_position = {
        (int(p), int(s))
        for p, s in torch.nonzero((args["position_cache"] != before_position).any(-1))
    }
    assert changed_index == written, f"index plane touched {changed_index}, expected {written}"
    assert changed_position == written, f"position plane touched {changed_position}"


def test_coordinates_wider_than_int32_wrap_rather_than_raise() -> None:
    """int64 coordinates are truncated into the int32 plane with nothing raised."""
    args = _small(seed=39)
    args["position_coordinates"] = torch.full((4, 3), 2**31 + 5, dtype=torch.int64, device=DEV)
    op(**args)
    page = int(args["block_table"][0, 0])
    assert int(args["position_cache"][page, 0, 0]) == -2147483643


def test_two_rows_on_one_slot_store_one_of_them() -> None:
    """Nothing orders duplicate destinations; the slot holds one of the two rows."""
    args = _small(seed=47)
    args["logical_positions"][1] = args["logical_positions"][0]
    args["request_indices"][1] = args["request_indices"][0]
    args["token_k"] = torch.stack(
        [torch.full((1, 128), float(v), dtype=torch.bfloat16, device=DEV) for v in (1, 2, 3, 4)]
    )
    op(**args)
    stored = float(args["index_cache"][int(args["block_table"][0, 0]), 0, 0, 0])
    assert stored in (1.0, 2.0), f"a duplicate slot held {stored}, neither row's value"


def test_repeats_are_stream_ordered_and_idempotent() -> None:
    """A read enqueued after the store sees it; repeating the call repeats it."""
    args = _small(seed=49)
    op(**args)
    first = (args["index_cache"].clone(), args["position_cache"].clone())
    for _ in range(7):
        op(**args)  # no synchronize: the clone below is on the same stream
        assert torch.equal(_bits(args["index_cache"]), _bits(first[0]))
        assert torch.equal(args["position_cache"], first[1])


#: The cell of 32 MTP7 verify blocks (256 rows): the segments, pool and table it runs over.
_MTP7_32 = next(c.spec for c in op.CELLS if c.why.startswith("decode MTP7: 32 requests"))
_MTP7_BATCH = {key: _MTP7_32[key] for key in ("segments", "pages", "slots")}


@pytest.mark.parametrize("batch", ({}, _MTP7_BATCH), ids=("decode", "mtp7-32x8"))
def test_cuda_graph_capture_and_replay(batch) -> None:
    """Captured once, replayed against new positions written into the same buffer."""
    args = _small(seed=52, **batch)
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


def test_the_stored_key_is_raw_not_normalized() -> None:
    """Discrimination against a Gemma-normed key, the decode kernel's form.

    A key stored normed double-normalizes every anchor `qsa_prefill_compress`
    builds. Old receipt at its seeds: 3.0625 max abs apart; the floor states a
    bound rather than one draw.
    """
    args = _small(seed=66)
    out = op(**args)
    gen = torch.Generator(device=DEV).manual_seed(68)
    weight = torch.randn(128, generator=gen, device=DEV) * 0.5
    x = args["token_k"].float()
    normed = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + weight)).to(
        torch.bfloat16
    )
    page = int(args["block_table"][0, 0])
    stored = out[0][page, 0]
    assert torch.equal(_bits(stored), _bits(args["token_k"][0]))
    distance = (stored.float() - normed[0].float()).abs().max().item()
    assert distance > 0.25, f"a normed key differs from the raw one by only {distance}"


# ------------------------------------------------------- loud rejections
#
# Each is a shape the kernel would otherwise read past or write wrongly; the
# entry refuses it before launching, so `is_valid` has nothing left to guard.


def _rejected(args, match: str) -> None:
    before = (args["index_cache"].clone(), args["position_cache"].clone())
    with pytest.raises(ValueError, match=match):
        op(**args)
    assert torch.equal(_bits(args["index_cache"]), _bits(before[0]))
    assert torch.equal(args["position_cache"], before[1])


def test_rejects_two_dimensional_request_indices() -> None:
    args = _small(seed=60)
    args["request_indices"] = args["request_indices"][:, None]
    _rejected(args, "one-dimensional")


def test_rejects_metadata_of_unequal_length() -> None:
    args = _small(seed=61)
    args["logical_positions"] = args["logical_positions"][:1]
    _rejected(args, "equally long")


def test_rejects_float_metadata() -> None:
    args = _small(seed=62)
    args["logical_positions"] = args["logical_positions"].float()
    _rejected(args, "int32 or int64")


def test_rejects_a_single_token_k_row() -> None:
    args = _small(seed=63)
    args["token_k"] = args["token_k"][:1]
    _rejected(args, "token_k")


def test_rejects_a_single_coordinate_row() -> None:
    args = _small(seed=64)
    args["position_coordinates"] = args["position_coordinates"][:1]
    _rejected(args, "position_coordinates")


def test_rejects_a_page_size_that_is_not_the_cache_s() -> None:
    args = _small(seed=65)
    args["tokens_per_block"] = 16
    _rejected(args, "index_cache")


def test_rejects_a_position_plane_with_a_different_page_size() -> None:
    args = _small(seed=66)
    args["position_cache"] = torch.full((8, 16, 3), POS_FILL, dtype=torch.int32, device=DEV)
    _rejected(args, "position_cache")


def test_rejects_a_position_plane_narrower_than_the_coordinates() -> None:
    args = _small(seed=67)
    args["position_cache"] = torch.full((8, 32, 2), POS_FILL, dtype=torch.int32, device=DEV)
    _rejected(args, "position_cache")


def test_rejects_a_cache_with_more_heads_than_the_key() -> None:
    args = _small(seed=68)
    args["index_cache"] = torch.full((8, 32, 2, 128), KEY_FILL, device=DEV).to(torch.bfloat16)
    _rejected(args, "index_cache")


def test_rejects_a_one_dimensional_table() -> None:
    args = _small(seed=69)
    args["block_table"] = args["block_table"][0]
    _rejected(args, "two-dimensional")


def test_huge_int64_position_is_skipped_not_truncated() -> None:
    """`2^32 + 5` truncated to int32 would be position 5 of column 0; the
    kernel keeps int64 and finds it past the table."""
    args = _small(seed=70)
    args["logical_positions"][0] = 2**32 + 5
    before = (args["index_cache"].clone(), args["position_cache"].clone())
    op(**args)
    page = int(args["block_table"][0, 0])
    assert torch.equal(_bits(args["index_cache"][page, 5]), _bits(before[0][page, 5]))
    assert torch.equal(args["position_cache"][page, 5], before[1][page, 5])
