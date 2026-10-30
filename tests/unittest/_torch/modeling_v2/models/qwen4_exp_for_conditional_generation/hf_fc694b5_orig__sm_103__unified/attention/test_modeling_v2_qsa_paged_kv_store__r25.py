# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_kv_store entry.

The paged state is real: an HND pool whose K and V role views are the caches, and the
engine's `[slots, width]` table -- a row per request slot, each live row's pages drawn from a
seeded permutation of the pool, `-1` past its allocation, and the rows past the batch holding a
previous step's plausible page ids. The pool is pre-filled with a sentinel so a stray write
shows. A cell's pool is one of two:

- a contiguous `[pages, 2, Hkv, tpb, D]` pool, compared whole;
- with `pool="engine"`, the stock cache manager's layout at an engine's measured page count:
  every attention layer's K and V planes in one page (24 role pages), the views at layer 47's
  planes. At ~257k pages that spans ~100 GB, so it is a reserved virtual range with physical
  memory mapped under the pages the rows touch only (`_EnginePool`); the pool's last page is
  always one of them. The reference runs on a dense copy of those pages, and every other mapped
  byte is checked unchanged.
"""

import zlib

import cuda.bindings.driver as cuda_drv
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


def _check(result):
    err, *value = result
    assert err == cuda_drv.CUresult.CUDA_SUCCESS, err
    return value[0] if len(value) == 1 else value


class _ArrayInterface:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = dict(
            shape=(nbytes,), strides=None, typestr="|u1", data=(ptr, False), version=2
        )


class _EnginePool:
    """One attention layer's K and V views of the stock manager's QSA pool at `pages` pages.

    A page holds `role_pages` role planes of `[hkv, tpb, D]` (K then V of every attention
    layer); the views are layer `layer_slot`'s. The range is reserved whole and physical memory
    is mapped under the K and V planes of `touched` pages only, and under page 0's (a tensor's
    base pointer). Reading an unmapped address faults, so only those pages are ever read.
    """

    def __init__(self, spec, touched):
        dtype = spec["cache_dtype"]
        elem = torch.empty((), dtype=dtype).element_size()
        hkv, tpb, dim = spec["hkv"], spec["tokens_per_block"], spec["head_dim"]
        self.pages, self.touched = spec["pages"], sorted(set(touched))
        self.role = hkv * tpb * dim
        self.stride = spec["role_pages"] * self.role
        self.own = spec["layer_slot"] * 2 * self.role

        def planes(page):
            """This layer's K and V planes of `page`, in bytes of the range."""
            lo = (page * self.stride + self.own) * elem
            return lo, lo + 2 * self.role * elem

        self.written = [planes(p) for p in self.touched]

        dev = torch.cuda.current_device()
        prop = cuda_drv.CUmemAllocationProp()
        prop.type = cuda_drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = dev
        gran = _check(
            cuda_drv.cuMemGetAllocationGranularity(
                prop,
                cuda_drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )
        self.size = -(-self.pages * self.stride * elem // gran) * gran
        self.va = int(_check(cuda_drv.cuMemAddressReserve(self.size, gran, 0, 0)))
        merged = []
        for lo, hi in sorted({planes(0), *self.written}):
            lo, hi = lo // gran * gran, min(-(-hi // gran) * gran, self.size)
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        access = cuda_drv.CUmemAccessDesc()
        access.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = dev
        access.flags = cuda_drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.maps, self.spans = [], merged
        for lo, hi in merged:
            handle = _check(cuda_drv.cuMemCreate(hi - lo, prop, 0))
            ptr = cuda_drv.CUdeviceptr(self.va + lo)
            _check(cuda_drv.cuMemMap(ptr, hi - lo, 0, handle, 0))
            self.maps.append((ptr, hi - lo, handle))
            _check(cuda_drv.cuMemSetAccess(ptr, hi - lo, [access], 1))

        self.raw = torch.as_tensor(_ArrayInterface(self.va, self.size), device=DEV)
        pool = self.raw.view(dtype).as_strided(
            (self.pages, 2, hkv, tpb, dim),
            (self.stride, self.role, tpb * dim, dim, 1),
            self.own,
        )
        self.k, self.v = pool[:, 0], pool[:, 1]
        self.index = torch.tensor(self.touched, dtype=torch.long, device=DEV)
        self.fill()

    def fill(self) -> None:
        """Every mapped byte to the sentinel's pattern, and remember it."""
        dtype = self.k.dtype
        pattern = _bits(torch.full((2,), FILL, dtype=torch.bfloat16).to(dtype))
        for lo, hi in self.spans:
            self.raw[lo:hi].view(pattern.dtype).fill_(int(pattern[0]))
        torch.cuda.synchronize()
        self.before = [self.raw[lo:hi].clone() for lo, hi in self.spans]

    def dense(self, table: torch.Tensor):
        """`(k, v, table)`: copies of the touched pages, and `table` renumbered into them.

        An entry naming no touched page becomes `-1`. No row resolves to one, so the
        reference reads the same pages it would over the whole pool.
        """
        remap = torch.full((self.pages,), -1, dtype=torch.int32, device=DEV)
        remap[self.index] = torch.arange(len(self.touched), dtype=torch.int32, device=DEV)
        renamed = torch.where(table >= 0, remap[table.clamp_min(0).long()], -1).to(torch.int32)
        return self.k[self.index].clone(), self.v[self.index].clone(), renamed.contiguous()

    def gathered(self):
        return self.k[self.index], self.v[self.index]

    def assert_rest_unchanged(self) -> None:
        """Every mapped byte outside the touched pages' K and V planes is what it was."""
        torch.cuda.synchronize()
        for (lo, hi), before in zip(self.spans, self.before):
            keep = torch.ones(hi - lo, dtype=torch.bool, device=DEV)
            for p_lo, p_hi in self.written:
                if p_hi > lo and p_lo < hi:
                    keep[max(p_lo, lo) - lo : min(p_hi, hi) - lo] = False
            assert torch.equal(self.raw[lo:hi][keep], before[keep]), (
                f"bytes [{lo}, {hi}) of the pool changed outside the touched pages' planes"
            )

    def close(self) -> None:
        torch.cuda.synchronize()
        del self.k, self.v, self.raw
        for ptr, size, handle in self.maps:
            _check(cuda_drv.cuMemUnmap(ptr, size))
            _check(cuda_drv.cuMemRelease(handle))
        _check(cuda_drv.cuMemAddressFree(cuda_drv.CUdeviceptr(self.va), self.size))


def _hole(spec, table, allocated_cols):
    """One row that resolves to no slot: its (request, position)."""
    assert spec["hole"] == "unallocated_page", spec["hole"]
    assert int(table[0, allocated_cols[0]]) == -1
    return 0, allocated_cols[0] * spec["tokens_per_block"]


def _build(spec, seed: int):
    """Real pool, table and rows for one cell: `(op kwargs, pool)`.

    `pool` is the contiguous pool tensor, or the `_EnginePool` under a `pool="engine"` cell.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tpb, hkv, head_dim = spec["tokens_per_block"], spec["hkv"], spec["head_dim"]
    slots, width, pages = spec["slots"], spec["width"], spec["pages"]
    engine = spec.get("pool") == "engine"
    segments = spec["segments"]
    requests = [r for r, (_, n) in enumerate(segments) for _ in range(n)]
    positions = [p for start, n in segments for p in range(start, start + n)]

    table = torch.randint(0, pages, (slots, width), generator=gen, dtype=torch.int32)
    order = torch.randperm(pages, generator=gen).to(torch.int32)
    if engine:
        # The first row's page is the pool's last one, whose planes are its highest bytes.
        first = segments[0][0] // tpb
        last = int((order == pages - 1).nonzero()[0])
        order[[first, last]] = order[[last, first]]
    used, allocated_cols = 0, []
    for r, (start, n) in enumerate(segments):
        cols = (start + n - 1) // tpb + 1
        assert cols <= width, f"request {r} ends past the table's {width} columns"
        table[r] = -1
        table[r, :cols] = order[used : used + cols]
        used += cols
        allocated_cols.append(cols)
    assert used <= pages and len(segments) <= slots, "pool or table too small for the batch"
    if spec["hole"] is not None:
        request, position = _hole(spec, table, allocated_cols)
        requests.append(request)
        positions.append(position)

    rows = len(positions)
    assert rows == spec.get("rows", rows), (rows, spec["rows"])
    qkv = torch.randn(
        rows,
        spec["heads_q"] + 2 * hkv + spec.get("row_pad_heads", 0),
        head_dim,
        generator=gen,
        dtype=torch.float32,
    ).to(device=DEV, dtype=spec["row_dtype"])
    assert qkv.stride(0) == spec.get("row_stride", qkv.stride(0)), qkv.stride(0)
    k = qkv[:, spec["heads_q"] : spec["heads_q"] + hkv]
    v = qkv[:, spec["heads_q"] + hkv : spec["heads_q"] + 2 * hkv]
    touched = [int(table[r, p // tpb]) for r, p in zip(requests, positions)]
    table = table.to(DEV)

    if engine:
        pool = _EnginePool(spec, [t for t in touched if t >= 0])
        k_cache, v_cache = pool.k, pool.v
    else:
        pool = torch.full(
            (pages, 2, hkv, tpb, head_dim), FILL, dtype=torch.bfloat16, device=DEV
        ).to(spec["cache_dtype"])
        k_cache, v_cache = pool[:, 0], pool[:, 1]
    args = dict(
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        request_indices=torch.tensor(requests, dtype=spec["request_dtype"], device=DEV),
        logical_positions=torch.tensor(positions, dtype=spec["position_dtype"], device=DEV),
        block_table=table,
        tokens_per_block=tpb,
    )
    return args, pool


def _engine_reference(args, pool: _EnginePool):
    """The entry's reference over a dense copy of the touched pages."""
    dense_k, dense_v, renamed = pool.dense(args["block_table"])
    return op.reference(**{**args, "k_cache": dense_k, "v_cache": dense_v, "block_table": renamed})


def _assert_reaches_past_int32(pool: _EnginePool) -> None:
    """The highest touched page's planes lie past 2^32 elements of the view's base: a page
    offset computed in 32 bits would land elsewhere."""
    top = max(pool.touched)
    assert top == pool.pages - 1, (top, pool.pages)
    assert top * pool.k.stride(0) >= 2**32, top * pool.k.stride(0)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args, pool = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    if isinstance(pool, _EnginePool):
        try:
            _assert_reaches_past_int32(pool)
            before = pool.gathered()
            before = (before[0].clone(), before[1].clone())
            ref = _engine_reference(args, pool)
            out = op(**args)
            assert out[0] is args["k_cache"] and out[1] is args["v_cache"]
            got = pool.gathered()
            _compare(got, ref)
            pool.assert_rest_unchanged()
            wrote = any(not torch.equal(_bits(g), _bits(b)) for g, b in zip(got, before))
            assert wrote, "a cell with rows must write"
            print(
                f"[{cell.why[:44]}] {len(pool.touched)} of {pool.pages} pages touched; "
                f"top plane at byte {pool.written[-1][0]} of {pool.size}; "
                f"{sum(hi - lo for lo, hi in pool.spans)} bytes mapped"
            )
        finally:
            pool.close()
        return
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

#: The tep4 decode cell of 128 rows, over its engine pool.
_TEP4_128 = next(c.spec for c in op.CELLS if c.why.startswith("tep4 decode, 128 rows"))


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


def test_cuda_graph_replay_on_the_tep4_engine_pool() -> None:
    """The tep4 decode site as the engine captures it: 128 rows of one KV head into a
    257109-page pool. Captured with every row on its page's first slot, replayed after the
    real positions are written into the same buffer."""
    args, pool = _build(_TEP4_128, seed=11)
    try:
        positions = args["logical_positions"]
        saved = positions.clone()
        tpb = args["tokens_per_block"]
        positions.copy_(saved - saved % tpb)  # the same pages, so only mapped memory is written

        def call():
            op(**args)

        for _ in range(3):  # compile the Triton kernel before capture
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()

        pool.fill()
        positions.copy_(saved)
        ref = _engine_reference(args, pool)
        graph.replay()
        torch.cuda.synchronize()
        _compare(pool.gathered(), ref)
        pool.assert_rest_unchanged()
    finally:
        pool.close()


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
