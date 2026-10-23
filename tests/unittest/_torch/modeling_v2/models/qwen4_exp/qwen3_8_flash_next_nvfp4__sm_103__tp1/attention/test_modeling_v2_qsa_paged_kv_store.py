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

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_paged_kv_store import (  # noqa: E501
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
        request, position = _hole(spec, table, len(segments), allocated_cols)
        requests.append(request)
        positions.append(position)

    rows = len(positions)
    qkv = torch.randn(
        rows, spec["heads_q"] + 2 * hkv, head_dim, generator=gen, dtype=torch.float32
    ).to(device=DEV, dtype=spec["row_dtype"])
    k = qkv[:, spec["heads_q"] : spec["heads_q"] + hkv]
    v = qkv[:, spec["heads_q"] + hkv :]
    if spec["k_layout"] == "head_major":
        k = k.transpose(1, 2).contiguous().transpose(1, 2)
        v = v.transpose(1, 2).contiguous().transpose(1, 2)
        assert k.stride(2) != 1

    table = table.to(DEV)
    if spec["table_layout"] == "role_slice":
        padded = torch.full((slots, 2, width), -1, dtype=torch.int32, device=DEV)
        padded[:, 0] = table
        table = padded[:, 0]
    elif spec["table_layout"] == "transposed":
        table = table.t().contiguous().t()
        assert table.stride(1) != 1

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
    spec = op.CELLS[0].spec
    return _build({**spec, **overrides}, seed=seed)


def test_fp8_pool_saturates_an_out_of_range_row() -> None:
    """1000.0 lands as e4m3's finite maximum, matching torch's own conversion."""
    args, _ = _decode_args(seed=4, cache_dtype=torch.float8_e4m3fn)
    args["k"] = torch.full_like(args["k"], 1000.0)
    out = op(**args)
    page, slot = int(args["block_table"][0, 0]), 0
    stored = out[0][page, :, slot, :]
    assert torch.equal(_bits(stored), _bits(args["k"][0].to(torch.float8_e4m3fn)))
    assert float(stored[0, 0].float()) == 448.0


def test_cuda_graph_replay_follows_updated_positions() -> None:
    """Captured once, replayed against new positions written into the same buffer."""
    args, pool = _decode_args(seed=9)
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


def test_op_rejects_operands_it_cannot_serve() -> None:
    """The callable's own loud errors: why `is_valid` carries only the stride guard."""
    args, _ = _decode_args(seed=5)
    row, cache = args["k"], args["k_cache"]
    narrow = torch.zeros(row.shape[0], row.shape[1], 64, dtype=row.dtype, device=DEV)
    cases = {
        "QSA K/V inputs must use BF16 or FP16": dict(k=row.float(), v=row.float()),
        "QSA K/V cache must use BF16, FP16, or FP8 E4M3": dict(
            k_cache=cache.float(), v_cache=cache.float()
        ),
        "QSA K/V pairs must use matching dtypes": dict(v=args["v"].half()),
        "QSA K/V cache token dimension does not match tokens_per_block": dict(tokens_per_block=16),
        "QSA K/V inputs and caches have incompatible head geometry": dict(k=narrow, v=narrow),
        "QSA K/V store request metadata must use integer storage": dict(
            logical_positions=args["logical_positions"].float()
        ),
        "QSA K/V store metadata must have one entry per input row": dict(
            request_indices=args["request_indices"][:1].contiguous()
        ),
        "QSA K/V store requires a two-dimensional int32 block table": dict(
            block_table=args["block_table"].long()
        ),
        "QSA paged K/V store requires CUDA tensors": {
            name: (t.cpu() if isinstance(t, torch.Tensor) else t) for name, t in args.items()
        },
    }
    for message, override in cases.items():
        with pytest.raises(ValueError, match=message):
            op(**{**args, **override})


def test_guard_refuses_strided_row_metadata() -> None:
    """`is_valid`'s one guard, after the hazard it exists for.

    The kernel receives strides for k, v, both caches and the table, but reads
    the two metadata vectors as flat unit-stride buffers. A column of a 2-wide
    buffer is read through its base pointer: accepted, and rows land elsewhere.
    """
    args, pool = _decode_args(seed=10)
    requests, positions = args["request_indices"], args["logical_positions"]
    rows = requests.numel()
    wide_requests = torch.full((rows, 2), rows - 1, dtype=requests.dtype, device=DEV)
    wide_requests[:, 0] = requests
    wide_positions = torch.full((rows, 2), 31, dtype=positions.dtype, device=DEV)
    wide_positions[:, 0] = positions
    ref = op.reference(**args)

    for override in (
        dict(request_indices=wide_requests[:, 0]),
        dict(logical_positions=wide_positions[:, 0]),
    ):
        pool.fill_(FILL)
        out = op(**{**args, **override})  # unguarded: accepted
        assert not all(torch.equal(_bits(got), _bits(want)) for got, want in zip(out, ref)), (
            "strided metadata was expected to place rows elsewhere"
        )

        pool.fill_(FILL)
        before = pool.clone()
        with pytest.raises(AssertionError, match="must be a contiguous 1-D tensor"):
            with validating(op):
                op(**{**args, **override})
        assert torch.equal(_bits(pool), _bits(before)), "a refused call still wrote"


def test_int64_position_past_int32_is_truncated_not_skipped() -> None:
    """Hard-limit probe: the kernel casts each position to int32.

    `2^32 + 5` should resolve to a page column far past the table and be
    skipped; truncated, it is position 5, a real slot, and it is written.
    """
    args, pool = _decode_args(seed=11)
    args["logical_positions"][0] = 2**32 + 5
    ref = op.reference(**args)
    out = op(**args)
    page = int(args["block_table"][0, 0])
    assert torch.equal(_bits(out[0][page, :, 5, :]), _bits(args["k"][0].to(out[0].dtype))), (
        "position 2^32 + 5 was expected to land on slot 5 of page column 0"
    )
    assert not torch.equal(_bits(out[0]), _bits(ref[0])), "the reference skips that row"
