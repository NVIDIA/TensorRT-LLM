# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_kv_store catalog entry.

The paged state is real: one `[pages, 2, Hkv, tpb, D]` HND pool whose K and V
role views are the caches, and the engine's `[slots, width]` table -- a row
per request slot, more slots than requests, each live row's pages drawn from a
seeded permutation of the pool, `-1` past its allocation, and the rows past
the batch holding a previous step's plausible page ids. Pool sizes are the old
receipts'; the pool is pre-filled with a sentinel so a stray write shows.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.attention.qsa_paged_kv_store__r25 import (  # noqa: E501
    qsa_paged_kv_store as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_paged_kv_store requires a CUDA device"


def _compare(out, ref) -> None:
    """Bitwise over both role planes, compared as raw bits.

    A store is a move with one conversion, and `reference` performs the
    same conversion, so there is no rounding to spend a tolerance on. In
    the target both planes together are the whole pool allocation, so this
    is also the stray-write check.
    """
    for name, got, want in zip(("k_cache", "v_cache"), out, ref):
        assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
        assert got.shape == want.shape, (name, got.shape, want.shape)
        torch.testing.assert_close(
            _bits(got), _bits(want), rtol=0, atol=0, msg=lambda m, n=name: f"{n}: {m}"
        )


DEV = "cuda"
FILL = -2.0


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    as_int = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.contiguous().view(as_int)


def _hole(spec, table, allocated_cols):
    """One row that resolves to no slot: its (request, position)."""
    assert spec["hole"] == "unallocated_page", spec["hole"]
    assert int(table[0, allocated_cols[0]]) == -1
    return 0, allocated_cols[0] * spec["tokens_per_block"]


def _build(spec, seed: int):
    """Real pool, table and rows for one cell; returns the op's kwargs and the pool."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tpb, hkv, head_dim = spec["tokens_per_block"], spec["hkv"], spec["head_dim"]
    slots, width, pages = spec["slots"], spec["width"], spec["pages"]
    segments = spec["segments"]
    requests = [r for r, (_, n) in enumerate(segments) for _ in range(n)]
    positions = [p for start, n in segments for p in range(start, start + n)]

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

    rows = len(positions)
    qkv = torch.randn(
        rows,
        spec["heads_q"] + 2 * hkv + spec.get("row_pad_heads", 0),
        head_dim,
        generator=gen,
        dtype=torch.float32,
    ).to(device=DEV, dtype=spec["row_dtype"])
    k = qkv[:, spec["heads_q"] : spec["heads_q"] + hkv]
    v = qkv[:, spec["heads_q"] + hkv : spec["heads_q"] + 2 * hkv]
    table = table.to(DEV)

    pool = torch.full((pages, 2, hkv, tpb, head_dim), FILL, dtype=torch.bfloat16, device=DEV).to(
        spec["cache_dtype"]
    )
    args = dict(
        k=k,
        v=v,
        k_cache=pool[:, 0],
        v_cache=pool[:, 1],
        request_indices=torch.tensor(requests, dtype=spec["request_dtype"], device=DEV),
        logical_positions=torch.tensor(positions, dtype=spec["position_dtype"], device=DEV),
        block_table=table,
        tokens_per_block=tpb,
    )
    return args, pool


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args, pool = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    before = pool.clone()
    ref = op.reference(**args)
    out = op(**args)
    assert out[0] is args["k_cache"] and out[1] is args["v_cache"]
    _compare(out, ref)
    wrote = not torch.equal(_bits(pool), _bits(before))
    assert wrote == bool(cell.spec["segments"]), (
        "a cell with rows must write, an empty one must not"
    )


def _decode_args(seed: int, **overrides):
    """The decode cell's layout over a bf16 pool, without its skipped row."""
    spec = {**op.CELLS[0].spec, "hole": None}
    return _build({**spec, **overrides}, seed=seed)


#: The cell of 32 MTP7 verify blocks (256 rows): the segments, pool and table it runs over.
_MTP7_32 = next(c.spec for c in op.CELLS if c.why.startswith("fp8 decode MTP7: 32 requests"))
_MTP7_BATCH = {key: _MTP7_32[key] for key in ("segments", "pages", "slots")}


def test_cuda_graph_replay_follows_updated_positions() -> None:
    """Captured once at 32 MTP7 verify blocks, replayed against new positions written
    into the same buffer."""
    args, pool = _decode_args(seed=9, **_MTP7_BATCH)
    positions = args["logical_positions"]
    saved = positions.clone()
    positions.zero_()

    def call():
        op(**args)

    for _ in range(3):  # compile the Triton kernel before capture
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()

    pool.fill_(FILL)
    positions.copy_(saved)
    ref = op.reference(**args)
    graph.replay()
    torch.cuda.synchronize()
    _compare((args["k_cache"], args["v_cache"]), ref)


def test_guard_refuses_strided_row_metadata() -> None:
    """`is_valid`'s one guard, after the hazard it exists for.

    The kernel receives strides for k, v, both caches and the table, but reads
    the two metadata vectors as flat unit-stride buffers. A column of a 2-wide
    buffer is read through its base pointer: accepted, and rows land elsewhere.
    """
    args, pool = _decode_args(seed=10)
    requests = args["request_indices"]
    rows = requests.numel()
    wide_requests = torch.full((rows, 2), rows - 1, dtype=requests.dtype, device=DEV)
    wide_requests[:, 0] = requests
    ref = op.reference(**args)
    strided = {**args, "request_indices": wide_requests[:, 0]}

    pool.fill_(FILL)
    out = op(**strided)  # unguarded: accepted
    assert not all(torch.equal(_bits(got), _bits(want)) for got, want in zip(out, ref)), (
        "strided metadata was expected to place rows elsewhere"
    )

    pool.fill_(FILL)
    before = pool.clone()
    with pytest.raises(AssertionError, match="must be a contiguous 1-D tensor"):
        with validating(op):
            op(**strided)
    assert torch.equal(_bits(pool), _bits(before)), "a refused call still wrote"
