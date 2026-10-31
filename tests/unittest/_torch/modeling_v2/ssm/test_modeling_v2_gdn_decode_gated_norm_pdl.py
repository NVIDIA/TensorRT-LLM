# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gdn_decode_gated_norm_pdl entry.

Cells against the composition's fp64 reference (the gated delta rule's recurrence,
then the gated norm entry's) in torch's default band for the bf16 output and the fp32
pool. A cell's pool is one of three:

- a contiguous 461-slot pool, compared whole so slots the call did not name are gated too;
- the tp1 coalesced page (36 layers' states a slot, 102 slots), the view at layer 0;
- with `pool_slots`, the stock cache manager's page at an engine's measured slot count:
  36 GDN layers' states a slot, the view at the last layer's sub-page. Such a page spans
  53 GB (tep4) or 96 GB (dep4) in fp32, so it is a reserved virtual range with physical
  memory mapped under the touched slots only (`_SparsePage`). The reference runs on a dense
  copy of those slots, and every other byte of their pages is checked unchanged.

Beyond the cells: a stale-read check under graph replay at both head layouts (a producer
rewrites the projection inside the graph right before the op), and the row guard.
"""

import zlib

import cuda.bindings.driver as cuda_drv
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.gdn_decode_gated_norm_pdl import (  # noqa: E501
    gdn_decode_gated_norm_pdl as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gdn_decode_gated_norm_pdl requires a CUDA device"

DEV = "cuda"
K, V = 128, 128
SLOTS, EPS = 461, 1e-6
#: GDN layers whose states share one slot page in the stock manager (48 layers, every 4th
#: full attention); the view is at the last one's sub-page, the page's highest addresses.
PAGE_LAYERS = 36


def _heads(spec) -> tuple[int, int]:
    """(k heads, v heads): tp1's 16 / 48 where the cell names none."""
    return spec.get("k_heads", 16), spec.get("v_heads", 48)


def _width(k_heads: int, v_heads: int) -> int:
    """`[q | k | v | z | a | b]`: 16480 at 16 / 48, 4120 at a tp4/tep4 rank's 4 / 12."""
    return 2 * k_heads * K + 2 * v_heads * V + 2 * v_heads


def _scattered(n: int, slots: int) -> tuple[int, ...]:
    """`n` distinct slots of a `slots`-row pool, both ends first, then a stride-37 walk
    over the interior (37 is coprime to every pool size here)."""
    rest = tuple((i * 37 + 4) % (slots - 2) + 1 for i in range(max(n - 2, 0)))
    return ((slots - 1, 0) + rest)[:n]


def _check(result):
    err, *value = result
    assert err == cuda_drv.CUresult.CUDA_SUCCESS, err
    return value[0] if len(value) == 1 else value


class _ArrayInterface:
    def __init__(self, ptr: int, shape, strides_bytes):
        self.__cuda_array_interface__ = dict(
            shape=tuple(shape),
            strides=tuple(strides_bytes),
            typestr="<f4",
            data=(ptr, False),
            version=2,
        )


class _SparsePage:
    """A reserved fp32 virtual range of `elems` elements with physical memory mapped under
    `regions` (element ranges) only. Views of it are torch tensors over that range; reading
    an unmapped address faults, so only mapped ranges are read."""

    def __init__(self, elems: int, regions):
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
        self.size = -(-elems * 4 // gran) * gran
        self.va = int(_check(cuda_drv.cuMemAddressReserve(self.size, gran, 0, 0)))
        spans = sorted(
            ((lo * 4) // gran * gran, min(-(-hi * 4 // gran) * gran, self.size))
            for lo, hi in regions
        )
        merged = []
        for lo, hi in spans:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        access = cuda_drv.CUmemAccessDesc()
        access.location.type = cuda_drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = dev
        access.flags = cuda_drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.maps = []
        for lo, hi in merged:
            handle = _check(cuda_drv.cuMemCreate(hi - lo, prop, 0))
            ptr = cuda_drv.CUdeviceptr(self.va + lo)
            _check(cuda_drv.cuMemMap(ptr, hi - lo, 0, handle, 0))
            self.maps.append((ptr, hi - lo, handle))
            _check(cuda_drv.cuMemSetAccess(ptr, hi - lo, [access], 1))

    def view(self, offset: int, shape, strides) -> torch.Tensor:
        """fp32 tensor at element `offset`, element `strides`."""
        return torch.as_tensor(
            _ArrayInterface(self.va + 4 * offset, shape, [4 * s for s in strides]), device=DEV
        )

    def close(self) -> None:
        torch.cuda.synchronize()
        for ptr, size, handle in self.maps:
            _check(cuda_drv.cuMemUnmap(ptr, size))
            _check(cuda_drv.cuMemRelease(handle))
        _check(cuda_drv.cuMemAddressFree(cuda_drv.CUdeviceptr(self.va), self.size))


class _EnginePool:
    """The stock manager's coalesced state page at `slots` slots (measured slot stride: 36
    layers' `[HV, V, K]` states), this layer's view the last layer's sub-page, mapped under
    the touched slots: each one's whole page and the next slot's first layer. The view's own
    base (slot 0's sub-page) is mapped too: torch reads the device off a tensor's base
    pointer."""

    def __init__(self, slots: int, used, v_heads: int, gen):
        layer = v_heads * V * K
        stride = PAGE_LAYERS * layer
        total = slots * stride
        base = (PAGE_LAYERS - 1) * layer
        self.used = sorted(set(used))
        self.spans = [(s * stride, min((s + 1) * stride + layer, total)) for s in self.used]
        self.page = _SparsePage(total, [*self.spans, (base, base + layer)])
        # This layer's state within each mapped span: every span starts at a slot's page.
        self.own = base
        self.layer = layer
        self.pool = self.page.view(base, (slots, v_heads, V, K), (stride, V * K, K, 1))
        self.flat = [self.page.view(lo, (hi - lo,), (1,)) for lo, hi in self.spans]
        for f in self.flat:
            f.normal_(0.0, 0.05, generator=gen)
        torch.cuda.synchronize()
        self.before = [f.clone() for f in self.flat]

    def dense(self, slots: torch.Tensor):
        """A dense copy of the touched slots and `slots` renumbered into it (-1 stays)."""
        index = {s: i for i, s in enumerate(self.used)}
        compact = torch.stack([self.pool[s] for s in self.used]).contiguous()
        renamed = [index[s] if s >= 0 else -1 for s in slots.tolist()]
        return compact, torch.tensor(renamed, dtype=torch.int32, device=DEV)

    def touched(self) -> torch.Tensor:
        return torch.stack([self.pool[s] for s in self.used])

    def assert_rest_unchanged(self) -> None:
        """Every byte of a touched slot's page but this layer's state, and the next slot's
        first layer, is what it was."""
        torch.cuda.synchronize()
        for f, b, s in zip(self.flat, self.before, self.used):
            keep = torch.ones(f.numel(), dtype=torch.bool, device=DEV)
            keep[self.own : self.own + self.layer] = False
            assert torch.equal(f[keep], b[keep]), f"slot {s}: the call wrote outside its state"

    def close(self) -> None:
        del self.pool, self.flat
        self.page.close()


def _build(spec, seed):
    """`(args, engine)`: the call's arguments, and the `_EnginePool` behind them (or None)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows = spec["rows"]
    k_heads, v_heads = _heads(spec)
    proj = torch.randn(rows, _width(k_heads, v_heads), generator=g, device=DEV).to(torch.bfloat16)
    engine = None
    if spec.get("pool_slots"):
        n = spec["pool_slots"]
        picked = (n - 1, n - 2) if rows == 2 else _scattered(rows, n)
        engine = _EnginePool(n, picked, v_heads, g)
        pool = engine.pool
        slots = torch.tensor(picked, dtype=torch.int32, device=DEV)
    elif spec.get("pool") == "coalesced":
        # The KV-cache-v2 manager's view: every GDN layer's state of a slot in one page,
        # this layer's state one sub-page of it, so the slot stride is 36 layers' states
        # and slot 100 sits ~2.8G elements past the base (past int32).
        layers, num = PAGE_LAYERS, 102
        page = torch.empty(num * layers * v_heads * V * K, device=DEV)
        pool = page.as_strided((num, v_heads, V, K), (layers * v_heads * V * K, V * K, K, 1))
        # The pool is compared whole: no slot of the view is left uninitialized.
        pool.zero_()
        pool[0].normal_(0.0, 0.05, generator=g)
        for s in (100, 101):
            pool[s].copy_(torch.randn(v_heads, V, K, generator=g, device=DEV) * 0.05)
        slots = torch.tensor([100, 101][:rows], dtype=torch.int32, device=DEV)
    else:
        pool = torch.randn(SLOTS, v_heads, V, K, generator=g, device=DEV) * 0.05
        slots = torch.tensor(_scattered(rows, SLOTS), dtype=torch.int32, device=DEV)
    if spec.get("pad") and rows > 1:
        slots[1] = -1
    A_log = torch.empty(v_heads, device=DEV).uniform_(-5.5, 1.0, generator=g)
    dt_bias = torch.randn(v_heads, generator=g, device=DEV) * 4.0
    w = (1.0 + 0.2 * torch.randn(V, generator=g, device=DEV)).to(torch.bfloat16)
    return [proj, pool, slots, A_log, dt_bias, w, EPS, k_heads, K], engine


def _compare(out, pool, ref_out, ref_pool) -> None:
    assert out.dtype == torch.bfloat16 and out.shape == ref_out.shape and out.is_contiguous()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(pool, ref_pool)


def _stats(out, ref_out, pool, ref_pool) -> str:
    """What `note` quotes: the output against its bf16-rounded reference, the state against
    the fp32-rounded one."""
    o, r = out.float().flatten(), ref_out.float().flatten()
    cos = torch.nn.functional.cosine_similarity(o, r, dim=0).item()
    rel = ((o - r).norm() / r.norm().clamp_min(1e-30)).item()
    differ = (out.view(torch.int16) != ref_out.view(torch.int16)).sum().item()
    p, q = pool.flatten(), ref_pool.flatten()
    p_rel = ((p - q).norm() / q.norm().clamp_min(1e-30)).item()
    return (
        f"out max|d| {(o - r).abs().max().item():.3e}, {differ}/{o.numel()} elements differ, "
        f"cos {cos:.8f}, rel L2 {rel:.3e}; state max|d| {(p - q).abs().max().item():.3e}, "
        f"rel L2 {p_rel:.3e}"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    spec = cell.spec
    args, engine = _build(spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    try:
        if engine is None:
            ref_out, ref_pool = op.reference(*args)
            out = op(*args)
            pool = args[1]
        else:
            dense, renamed = engine.dense(args[2])
            ref_out, ref_pool = op.reference(args[0], dense, renamed, *args[3:])
            out = op(*args)
            torch.cuda.synchronize()
            pool = engine.touched()
        _compare(out, pool, ref_out, ref_pool)
        if engine is not None:
            engine.assert_rest_unchanged()
        print(f"[{cell.why}] {_stats(out, ref_out, pool, ref_pool)}")
        if spec.get("pad"):
            if spec["rows"] > 1:
                assert (out[1] == 0).all(), "a pad row's output is zero"
            else:
                # One row: a second call over the pad slot (-1) writes zeros and touches no state.
                state = engine.touched().clone() if engine else args[1].clone()
                pad = torch.full_like(args[2], -1)
                out_pad = op(args[0], args[1], pad, *args[3:])
                torch.cuda.synchronize()
                assert (out_pad == 0).all(), "a pad row's output is zero"
                after = engine.touched() if engine else args[1]
                assert torch.equal(after, state), "a pad row touches no state"
                if engine:
                    engine.assert_rest_unchanged()
    finally:
        if engine is not None:
            engine.close()


@pytest.mark.parametrize("k_heads, v_heads", [(16, 48), (4, 12)], ids=["16x48", "4x12"])
def test_reads_after_the_wait_under_graph_replay(k_heads, v_heads) -> None:
    """A producer rewrites the projection inside the graph right before the op; with PDL
    a read of it before `griddepcontrol.wait` would see the previous replay's row. The
    pool advances one step per replay, as in decode."""
    spec = dict(rows=1, k_heads=k_heads, v_heads=v_heads)
    args, _ = _build(spec, seed=41)
    src = args[0].clone()

    def step():
        args[0].copy_(src * 1.0)
        return op(*args)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step()
    for trial in range(3):
        src.copy_(_build(spec, seed=100 + trial)[0][0])
        before = args[1].clone()
        graph.replay()
        torch.cuda.synchronize()
        after = args[1].clone()
        args[1].copy_(before)
        want = op(src.clone(), *args[1:])
        assert torch.equal(graph_out, want), f"replay {trial} output"
        assert torch.equal(after, args[1]), f"replay {trial} pool"


def test_guards() -> None:
    """More rows than `MAX_ROWS` are refused."""
    args, _ = _build(dict(rows=1), seed=5)
    with validating(op), pytest.raises(AssertionError):
        op(args[0].expand(5, -1).contiguous(), args[1], args[2].expand(5).contiguous(), *args[3:])
