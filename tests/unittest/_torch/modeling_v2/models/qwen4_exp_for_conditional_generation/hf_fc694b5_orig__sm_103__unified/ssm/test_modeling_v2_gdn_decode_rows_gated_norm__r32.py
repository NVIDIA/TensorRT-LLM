# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gdn_decode_rows_gated_norm entry.

Cells against the composition's fp64 reference (the gated delta rule's recurrence,
then the gated norm entry's) in torch's default band for the bf16 output and the pool's
dtype (fp32 or bf16). A cell's pool is one of three:

- a contiguous 461-slot pool, compared whole so slots the call did not name are gated too;
- the tp1 coalesced page (36 layers' states a slot, 102 slots), the view at layer 0;
- with `pool_slots`, the stock cache manager's page at an engine's slot count: 36 GDN
  layers' states a slot, the view at the last layer's sub-page. Such a page spans 26 GB
  (tep4) or 48 GB (dep4) in bf16, so it is a reserved virtual range with physical memory
  mapped under the touched slots only (`_SparsePage`). The reference runs on a dense copy
  of those slots, and every other byte of their pages is checked unchanged.

Beyond the cells: a fixed row bitwise across batch sizes over a bf16 pool, a stale-read
check under graph replay (a producer rewrites the projection inside the graph right
before the op), the two launch variants (state rows by bulk copy into shared memory, or
loaded into registers) bitwise each other, and the guards, each at tp1's 16 / 48 heads
and a tp4/tep4 rank's 4 / 12. The `gdn_decode_rows` entry (the recurrence alone) has its
own cells below.
"""

import zlib

import cuda.bindings.driver as cuda_drv
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.ssm import (  # noqa: E501
    gdn_decode_rows_gated_norm__r32 as rows_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.ssm.gdn_decode_rows_gated_norm__r32 import (  # noqa: E501
    gdn_decode_rows as rows_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.ssm.gdn_decode_rows_gated_norm__r32 import (  # noqa: E501
    gdn_decode_rows_gated_norm as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gdn_decode_rows_gated_norm requires a CUDA device"

DEV = "cuda"
H, HV, K, V = 16, 48, 128, 128
SLOTS, EPS = 461, 1e-6
WIDTH = 2 * H * K + 2 * HV * V + 2 * HV
#: GDN layers whose states share one slot page in the stock manager (48 layers, every 4th
#: full attention); the view is at the last one's sub-page, the page's highest addresses.
PAGE_LAYERS = 36
#: (k heads, v heads) of a tp4/tep4 rank.
RANK_HEADS = (4, 12)


def _heads(spec) -> tuple[int, int]:
    """(k heads, v heads): tp1's 16 / 48 where the cell names none."""
    return spec.get("k_heads", H), spec.get("v_heads", HV)


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


#: Array-interface type of each pool dtype; bf16 has none, so it is read as int16 and viewed.
_TYPESTR = {torch.float32: ("<f4", torch.float32), torch.bfloat16: ("<i2", torch.int16)}


class _ArrayInterface:
    def __init__(self, ptr: int, shape, strides_bytes, typestr: str):
        self.__cuda_array_interface__ = dict(
            shape=tuple(shape),
            strides=tuple(strides_bytes),
            typestr=typestr,
            data=(ptr, False),
            version=2,
        )


class _SparsePage:
    """A reserved virtual range of `elems` elements of `dtype` with physical memory mapped
    under `regions` (element ranges) only. Views of it are torch tensors over that range;
    reading an unmapped address faults, so only mapped ranges are read."""

    def __init__(self, elems: int, regions, dtype: torch.dtype):
        self.dtype, self.es = dtype, torch.empty((), dtype=dtype).element_size()
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
        es = self.es
        self.size = -(-elems * es // gran) * gran
        self.va = int(_check(cuda_drv.cuMemAddressReserve(self.size, gran, 0, 0)))
        spans = sorted(
            ((lo * es) // gran * gran, min(-(-hi * es // gran) * gran, self.size))
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
        """Tensor of the page's dtype at element `offset`, element `strides`."""
        typestr, carrier = _TYPESTR[self.dtype]
        es = self.es
        t = torch.as_tensor(
            _ArrayInterface(self.va + es * offset, shape, [es * s for s in strides], typestr),
            device=DEV,
        )
        assert t.dtype == carrier
        return t.view(self.dtype)

    def close(self) -> None:
        torch.cuda.synchronize()
        for ptr, size, handle in self.maps:
            _check(cuda_drv.cuMemUnmap(ptr, size))
            _check(cuda_drv.cuMemRelease(handle))
        _check(cuda_drv.cuMemAddressFree(cuda_drv.CUdeviceptr(self.va), self.size))


class _EnginePool:
    """The stock manager's coalesced state page at `slots` slots (slot stride: 36 layers'
    `[HV, V, K]` states), this layer's view the last layer's sub-page, mapped under the
    touched slots: each one's whole page and the next slot's first layer. The view's own
    base (slot 0's sub-page) is mapped too: torch reads the device off a tensor's base
    pointer."""

    def __init__(self, slots: int, used, v_heads: int, dtype: torch.dtype, gen):
        layer = v_heads * V * K
        stride = PAGE_LAYERS * layer
        total = slots * stride
        base = (PAGE_LAYERS - 1) * layer
        self.used = sorted(set(used))
        self.spans = [(s * stride, min((s + 1) * stride + layer, total)) for s in self.used]
        self.page = _SparsePage(total, [*self.spans, (base, base + layer)], dtype)
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
        lo, hi = self.own, self.own + self.layer
        for f, b, s in zip(self.flat, self.before, self.used):
            assert torch.equal(f[:lo], b[:lo]) and torch.equal(f[hi:], b[hi:]), (
                f"slot {s}: the call wrote outside its state"
            )

    def close(self) -> None:
        del self.pool, self.flat
        self.page.close()


def _build(spec, seed):
    """`(args, engine)`: the call's arguments, and the `_EnginePool` behind them (or None)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows = spec["rows"]
    k_heads, v_heads = _heads(spec)
    width = _width(k_heads, v_heads)
    assert spec.get("proj_width", width) == width, (spec, width)
    state = torch.bfloat16 if spec.get("state") == "bf16" else torch.float32
    proj = torch.randn(rows, width, generator=g, device=DEV).to(torch.bfloat16)
    engine = None
    if spec.get("pool_slots"):
        n = spec["pool_slots"]
        picked = _scattered(rows, n)
        engine = _EnginePool(n, picked, v_heads, state, g)
        pool = engine.pool
        slots = torch.tensor(picked, dtype=torch.int32, device=DEV)
    elif spec.get("pool") == "coalesced":
        # The KV-cache-v2 manager's view: every GDN layer's state of a slot in one page,
        # this layer's state one sub-page of it, so the slot stride is 36 layers' states
        # and slot 100 sits ~2.8G elements past the base (past int32).
        layers, num = PAGE_LAYERS, 102
        page = torch.empty(num * layers * v_heads * V * K, device=DEV, dtype=state)
        page[: v_heads * V * K * 4].normal_(0.0, 0.05, generator=g)
        pool = page.as_strided((num, v_heads, V, K), (layers * v_heads * V * K, V * K, K, 1))
        for s in (100, 101):
            pool[s].copy_(torch.randn(v_heads, V, K, generator=g, device=DEV) * 0.05)
        slots = torch.tensor([100, 101][:rows], dtype=torch.int32, device=DEV)
    else:
        pool = (torch.randn(SLOTS, v_heads, V, K, generator=g, device=DEV) * 0.05).to(state)
        slots = torch.tensor(_scattered(rows, SLOTS), dtype=torch.int32, device=DEV)
    if spec.get("pad"):
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
    the reference rounded to the pool's dtype. Elements are counted by value (a pad row's
    -0.0 equals the reference's 0.0)."""
    o, r = out.float().flatten(), ref_out.float().flatten()
    cos = torch.nn.functional.cosine_similarity(o, r, dim=0).item()
    rel = ((o - r).norm() / r.norm().clamp_min(1e-30)).item()
    differ = (o != r).sum().item()
    p, q = pool.float().flatten(), ref_pool.float().flatten()
    p_rel = ((p - q).norm() / q.norm().clamp_min(1e-30)).item()
    p_differ = (p != q).sum().item()
    return (
        f"out max|d| {(o - r).abs().max().item():.3e}, {differ}/{o.numel()} elements differ, "
        f"cos {cos:.8f}, rel L2 {rel:.3e}; state max|d| {(p - q).abs().max().item():.3e}, "
        f"{p_differ}/{p.numel()} differ, rel L2 {p_rel:.3e}"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args, engine = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
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
        if cell.spec.get("pad"):
            assert (out[1] == 0).all(), "a pad row's output is zero"
    finally:
        if engine is not None:
            engine.close()


#: Decode batch sizes up to the engine's largest CUDA-graph batch.
_BATCHES = (1, 2, 4, 5, 16, 33, 128, 256)
_FIXED_SLOT = 301


@pytest.mark.parametrize("k_heads, v_heads", [(H, HV), RANK_HEADS], ids=["16x48", "4x12"])
def test_bf16_pool_is_batch_invariant(k_heads, v_heads) -> None:
    """The property the bf16 decode step relies on: a row's output and state are bitwise
    independent of the batch it rides in. One fixed row (its projection and slot 301 of a
    bf16 pool) sits first and last in random batches of every size, across the batch where
    the launch switches to the bulk-copy variant (7 rows at 48 v heads, 26 at 12)."""
    width = _width(k_heads, v_heads)
    gen = torch.Generator(device=DEV).manual_seed(3)
    cpu = torch.Generator().manual_seed(3)
    pool0 = (torch.randn(SLOTS, v_heads, V, K, generator=gen, device=DEV) * 0.05).to(torch.bfloat16)
    row = torch.randn(width, generator=gen, device=DEV).to(torch.bfloat16)
    rest = _build(dict(rows=1, k_heads=k_heads, v_heads=v_heads), seed=3)[0][3:]
    first = None
    for n in _BATCHES:
        for pos in sorted({0, n - 1}):
            g = torch.Generator(device=DEV).manual_seed(1000 * n + pos)
            proj = torch.randn(n, width, generator=g, device=DEV).to(torch.bfloat16)
            proj[pos] = row
            others = [s for s in torch.randperm(SLOTS, generator=cpu).tolist() if s != _FIXED_SLOT]
            others = others[:n]
            others[pos] = _FIXED_SLOT
            slots = torch.tensor(others, dtype=torch.int32, device=DEV)
            pool = pool0.clone()
            out = op(proj, pool, slots, *rest)
            got = (out[pos].clone(), pool[_FIXED_SLOT].clone())
            first = first or got
            assert torch.equal(got[0], first[0]) and torch.equal(got[1], first[1]), (
                f"the fixed row differs at batch {n}, position {pos}"
            )


@pytest.mark.parametrize(
    "spec",
    (
        dict(rows=16),
        dict(rows=3, k_heads=4, v_heads=12, state="bf16"),
        dict(rows=128, k_heads=4, v_heads=12, state="bf16"),
    ),
    ids=["16x48-16rows", "4x12-3rows-bf16", "4x12-128rows-bf16"],
)
def test_reads_after_the_wait_under_graph_replay(spec) -> None:
    """A producer rewrites the projection inside the graph right before the op; with PDL
    a read of it before `griddepcontrol.wait` would see the previous replay's row. The
    pool advances one step per replay, as in decode. At 4 / 12 heads 3 rows take the
    register variant, 128 the bulk copy."""
    args = _build(spec, seed=41)[0]
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


_VARIANT_SPECS = (
    dict(rows=1, state="bf16"),
    dict(rows=8, state="bf16", pad=True),
    dict(rows=33, state="bf16"),
    dict(rows=3),
    dict(rows=33, pad=True),
    dict(rows=2, pool="coalesced", state="bf16", pad=True),
    dict(rows=2, pool="coalesced"),
    dict(rows=3, k_heads=4, v_heads=12, state="bf16", pad=True),
    dict(rows=128, k_heads=4, v_heads=12, state="bf16"),
    dict(rows=3, k_heads=4, v_heads=12, state="bf16", pool_slots=rows_mod.TEP4_SLOTS),
)


@pytest.mark.parametrize("norm", [True, False], ids=["norm", "rows"])
@pytest.mark.parametrize("spec", _VARIANT_SPECS, ids=lambda s: "-".join(map(str, s.values())))
def test_bulk_and_register_variants_agree(spec, norm) -> None:
    """The bulk-copy and the register variants: outputs and state bitwise equal.

    The bulk-copy variant stages the state rows in shared memory; the register variant holds
    them in registers; both compute the same instructions. Over contiguous bf16 and fp32 pools,
    the tp1 coalesced page (slot 100, ~2.8G elements past the base, by the bulk copy) and
    tep4's engine page (slot 1873, ~1.3e10 elements past the base), with pad rows, at 16 / 48
    and 4 / 12 heads.
    """
    args, engine = _build(spec, seed=zlib.crc32(repr(spec).encode()) % (2**31))
    try:
        proj, pool, slots, k_heads = args[0], args[1], args[2], args[7]
        assert rows_mod._bulk_eligible(pool, K)
        live = slots[slots >= 0].long()
        before = pool[live].clone()
        got = []
        for bulk in (False, True):
            pool[live] = before
            w = args[5] if norm else None
            out = rows_mod.launch(
                proj, pool, slots, args[3], args[4], w, EPS, k_heads, K, bulk=bulk
            )
            got.append((out.clone(), pool[live].clone()))
        assert torch.equal(got[0][0], got[1][0]), "outputs differ between the variants"
        assert torch.equal(got[0][1], got[1][1]), "states differ between the variants"
        if engine is not None:
            engine.assert_rest_unchanged()
    finally:
        if engine is not None:
            engine.close()


@pytest.mark.parametrize("v_heads", [HV, RANK_HEADS[1]], ids=["48", "12"])
def test_variant_choice_and_unaligned_pool(v_heads) -> None:
    """The variant choice, and the fallback for a pool the bulk copy cannot take.

    The bulk variant from the batch the register variant needs a second wave for; a pool 8-
    but not 16-byte aligned stays on the register variant whatever is asked, and matches the
    aligned pool's result.
    """
    sms = torch.cuda.get_device_properties(DEV).multi_processor_count
    edge = (2 * sms) // v_heads
    assert not rows_mod._use_bulk(edge, v_heads, DEV) and rows_mod._use_bulk(edge + 1, v_heads, DEV)
    print(f"[{v_heads} v heads] {sms} SMs: the bulk-copy variant from {edge + 1} rows")
    k_heads = v_heads // 3
    args = _build(dict(rows=33, k_heads=k_heads, v_heads=v_heads, state="bf16"), seed=19)[0]
    pool = args[1]
    flat = torch.empty(pool.numel() + 4, dtype=pool.dtype, device=DEV)
    shifted = flat[4:].view(pool.shape)
    shifted.copy_(pool)
    assert not rows_mod._bulk_eligible(shifted, K)
    want = rows_mod.launch(args[0], pool, *args[2:], bulk=True)
    got = rows_mod.launch(args[0], shifted, *args[2:], bulk=True)
    assert torch.equal(got, want) and torch.equal(shifted, pool)


@pytest.mark.parametrize("k_heads, v_heads", [(H, HV), RANK_HEADS], ids=["16x48", "4x12"])
def test_guards(k_heads, v_heads) -> None:
    """An fp16 pool, int64 slot indices and a projection of the wrong width are refused."""
    args = _build(dict(rows=4, k_heads=k_heads, v_heads=v_heads), seed=5)[0]
    with validating(op), pytest.raises(AssertionError):
        op(args[0], args[1].to(torch.float16), *args[2:])
    with validating(op), pytest.raises(AssertionError):
        op(args[0], args[1], args[2].long(), *args[3:])
    with validating(op), pytest.raises(AssertionError):
        op(args[0][:, :-2], *args[1:])


# ---- gdn_decode_rows: the recurrence alone, into a row range of a step's output ----------


def _rows_args(spec, seed):
    """`(args, buf)`: the entry's arguments, `out` rows 5.. of a `[1, rows + 9, HV, V]` buffer
    (the step's output, the call's rows a range of it), and that buffer."""
    (proj, pool, slots, A_log, dt_bias, _, _, h, k), _ = _build(spec, seed)
    rows, v_heads = proj.shape[0], pool.shape[1]
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    buf = torch.randn(1, rows + 9, v_heads, V, generator=g, device=DEV).to(torch.bfloat16)
    out = buf[0, 5 : 5 + rows].view(rows, v_heads * V)
    return [proj, pool, slots, A_log, dt_bias, h, k, out], buf


@pytest.mark.parametrize("cell", rows_op.CELLS, ids=[c.why[:44] for c in rows_op.CELLS])
def test_rows_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(rows_op)
    args, buf = _rows_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    outside = torch.cat([buf[0, :5], buf[0, 5 + args[0].shape[0] :]]).clone()
    ref_o, ref_pool = rows_op.reference(*args)
    got = rows_op(*args)
    assert got.data_ptr() == args[-1].data_ptr(), "written into out"
    torch.testing.assert_close(got, ref_o)
    torch.testing.assert_close(args[1], ref_pool)
    print(f"[{cell.why}] {_stats(got, ref_o, args[1], ref_pool)}")
    assert torch.equal(torch.cat([buf[0, :5], buf[0, 5 + args[0].shape[0] :]]), outside), (
        "rows outside the range untouched"
    )
    if cell.spec.get("pad"):
        assert (got[1] == 0).all(), "a pad row's output is zero"
