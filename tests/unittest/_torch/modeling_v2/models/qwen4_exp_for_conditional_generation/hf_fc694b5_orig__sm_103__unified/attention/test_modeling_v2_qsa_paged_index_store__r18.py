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
        # pad] = 110 heads of 128: the raw index key is head 108.
        row = torch.randn(rows, 110, head_dim, generator=gen, dtype=torch.float32)
        token_k = row.to(device=DEV, dtype=spec["key_dtype"])[:, 108 : 108 + hkv]

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
    if spec["cache_layout"] == "shared_page":
        index_cache, position_cache = _shared_page_planes(spec, extras)
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
    ref = op.reference(**args)
    out = op(**args)
    assert out[0] is args["index_cache"] and out[1] is args["position_cache"]
    _compare(out, ref)
    if "roles" in extras:
        fill = int(_bits(torch.tensor([KEY_FILL], dtype=torch.bfloat16))[0])
        assert bool((extras["roles"].view(torch.int16) == fill).all()), (
            "another index-K role of the shared page moved"
        )


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
