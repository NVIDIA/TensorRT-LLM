# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_chunk_gated_delta_rule entry (FlashInfer GDN prefill).

A cell's pool is one of two:

- a contiguous pool (tp1's cells), compared whole so slots the call did not name are
  gated too;
- with `page_layers`, the stock cache manager's bf16 state page at a member's measured
  slot count: `page_layers` GDN layers' states a slot, the view at the last layer's
  sub-page. A tep4 page spans 26.5 GB and a dep4 page 48 GB, so it is a reserved virtual
  range with physical memory mapped under the named slots only (`_SparsePage`). The
  reference runs on a dense copy of those slots, and every other byte of their pages is
  checked unchanged; a write to any other slot would fault.

Beyond the cells: the has_initial_state equivalence, and a negative control that the band
refuses a value head reading the wrong key head at 4/12.
"""

import warnings
import zlib

import cuda.bindings.driver as cuda_drv
import pytest
import torch
from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import should_use_cp_host
from flashinfer.utils import get_compute_capability, get_device_name, get_device_sm_count

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.ssm.flashinfer_chunk_gated_delta_rule__r28 import (  # noqa: E501
    RANK4_GEOMETRY,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.ssm.flashinfer_chunk_gated_delta_rule__r28 import (  # noqa: E501
    flashinfer_chunk_gated_delta_rule as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "flashinfer_chunk_gated_delta_rule requires a CUDA device"

DEV = "cuda"
#: The band the Triton chunk op is certified at (rtol bf16 default; atol 2e-3
#: on `o`, 2e-2 on states), derived for a chunked WY reformulation with bf16
#: operands. This op is the same class of sum: measured against fp64 at
#: 1 x 8192 tokens, o max abs 1.26e-3 and state max abs 8.6e-3 (Triton op:
#: 1.13e-3 / 7.7e-3), inside it. At a rank's pool every cell measured o max
#: abs 4.9e-4 and state max abs 3.9e-3, CP pipeline and chunked kernel alike;
#: a value head reading the wrong key head at 4/12 is o max abs 7.6e-2 (o
#: rel-L2 1.41 against 3.8e-3), which `test_band_refuses_a_misgrouped_key_head`
#: holds the band refuses.
RTOL, ATOL_OUTPUT, ATOL_STATE = 1.6e-2, 2e-3, 2e-2
#: Rows past the call's own in the output buffer of a `slice` cell: the target passes
#: `out[:, :num_tokens]` of a wider buffer, and the rows past it must stay as they were.
_OUTPUT_TAIL = 16


def _compare(out: tuple, ref: tuple) -> None:
    """`(o, final_state, pool)` against `reference`'s triple; `None` must match."""
    for got, want, atol in zip(out, ref, (ATOL_OUTPUT, ATOL_STATE, ATOL_STATE)):
        assert (got is None) == (want is None), (got is None, want is None)
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape, (
            (got.dtype, got.shape),
            (want.dtype, want.shape),
        )
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got.float(), want.float(), rtol=RTOL, atol=atol)


def _stats(name: str, got: torch.Tensor, want: torch.Tensor) -> str:
    """Relative L2, cosine, p99.9 and max absolute error of `got` against `want`."""
    g, w = got.double().flatten(), want.double().flatten()
    err = (g - w).abs()
    rel = (err.norm() / w.norm().clamp_min(1e-30)).item()
    cos = torch.nn.functional.cosine_similarity(g, w, dim=0).item()
    p999 = torch.quantile(err[:: max(1, err.numel() // 2_000_000)].float(), 0.999).item()
    return f"{name} rel-L2 {rel:.3e} cos {cos:.6f} p99.9 {p999:.3e} max {err.max().item():.3e}"


def _flashinfer_path(spec) -> str:
    """Which FlashInfer kernel the call reaches: `cp` (the context-parallel pipeline) when
    FlashInfer's own heuristic picks it for `num_seqs x num_v_heads` on this device."""
    device = torch.device(DEV)
    cp = should_use_cp_host(
        len(spec["seqlens"]) * spec["geometry"][1],
        get_device_sm_count(device),
        get_device_name(device),
        device_capability=get_compute_capability(device),
    )
    return "cp" if cp else "chunked"


def _check(result):
    err, *value = result
    assert err == cuda_drv.CUresult.CUDA_SUCCESS, err
    return value[0] if len(value) == 1 else value


class _ArrayInterface:
    def __init__(self, ptr: int, shape, strides_bytes):
        self.__cuda_array_interface__ = dict(
            shape=tuple(shape),
            strides=tuple(strides_bytes),
            typestr="<i2",
            data=(ptr, False),
            version=2,
        )


class _SparsePage:
    """A reserved bf16 virtual range of `elems` elements with physical memory mapped under
    `regions` (element ranges) only. Views of it are torch tensors over that range; reading
    an unmapped address faults, so only mapped ranges are read."""

    ELEM = 2

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
        self.size = -(-elems * self.ELEM // gran) * gran
        self.va = int(_check(cuda_drv.cuMemAddressReserve(self.size, gran, 0, 0)))
        spans = sorted(
            (
                (lo * self.ELEM) // gran * gran,
                min(-(-hi * self.ELEM // gran) * gran, self.size),
            )
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
        """bf16 tensor at element `offset`, element `strides`."""
        raw = torch.as_tensor(
            _ArrayInterface(self.va + self.ELEM * offset, shape, [self.ELEM * s for s in strides]),
            device=DEV,
        )
        return raw.view(torch.bfloat16)

    def close(self) -> None:
        torch.cuda.synchronize()
        for ptr, size, handle in self.maps:
            _check(cuda_drv.cuMemUnmap(ptr, size))
            _check(cuda_drv.cuMemRelease(handle))
        _check(cuda_drv.cuMemAddressFree(cuda_drv.CUdeviceptr(self.va), self.size))


class _EnginePool:
    """The stock manager's coalesced bf16 state page at `slots` slots (slot stride:
    `layers` `[HV, V, K]` states), this layer's view the last layer's sub-page, mapped
    under the named slots: each one's whole page and the next slot's first layer. The
    view's own base (slot 0's sub-page) is mapped too: torch reads the device off a
    tensor's base pointer."""

    def __init__(self, slots: int, layers: int, used, v_heads: int, head_v: int, head_k: int, gen):
        layer = v_heads * head_v * head_k
        stride = layers * layer
        total = slots * stride
        base = (layers - 1) * layer
        self.used = sorted(set(used))
        self.spans = [(s * stride, min((s + 1) * stride + layer, total)) for s in self.used]
        self.page = _SparsePage(total, [*self.spans, (base, base + layer)])
        self.own = base
        self.layer = layer
        self.pool = self.page.view(
            base, (slots, v_heads, head_v, head_k), (stride, head_v * head_k, head_k, 1)
        )
        self.flat = [self.page.view(lo, (hi - lo,), (1,)) for lo, hi in self.spans]
        for f in self.flat:
            f.copy_(torch.randn(f.numel(), generator=gen, device=DEV) * 0.1)
        torch.cuda.synchronize()
        self.before = [f.clone() for f in self.flat]

    def dense(self, slots: torch.Tensor):
        """A dense copy of the named slots and `slots` renumbered into it."""
        index = {s: i for i, s in enumerate(self.used)}
        compact = torch.stack([self.pool[s] for s in self.used]).contiguous()
        renamed = [index[s] for s in slots.tolist()]
        return compact, torch.tensor(renamed, dtype=slots.dtype, device=DEV)

    def touched(self) -> torch.Tensor:
        return torch.stack([self.pool[s] for s in self.used])

    def assert_rest_unchanged(self) -> None:
        """Every byte of a named slot's page but this layer's state, and the next slot's
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
    """`(args, engine)`: keyword arguments in the domain a GDN post-conv step produces
    (unit-norm q/k, g < 0 in log space, beta a sigmoid), and the `_EnginePool` behind the
    pool (or None)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = spec["geometry"]
    seqlens = spec["seqlens"]
    total = sum(seqlens)
    q = torch.randn(1, total, num_k_heads, head_k_dim, generator=gen, device=DEV)
    k = torch.randn(1, total, num_k_heads, head_k_dim, generator=gen, device=DEV)
    q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    q, k = q.bfloat16(), k.bfloat16()
    v = torch.randn(
        1, total, num_v_heads, head_v_dim, generator=gen, device=DEV, dtype=torch.bfloat16
    )
    a_log = torch.log(torch.rand(num_v_heads, generator=gen, device=DEV) * 15 + 1)
    x = torch.randn(1, total, num_v_heads, generator=gen, device=DEV) + torch.randn(
        num_v_heads, generator=gen, device=DEV
    )
    g = (-torch.exp(a_log) * torch.nn.functional.softplus(x)).float()
    beta = torch.rand(1, total, num_v_heads, generator=gen, device=DEV).sigmoid().float()
    engine = None
    if spec.get("page_layers"):
        engine = _EnginePool(
            spec["slots"],
            spec["page_layers"],
            spec["indices"],
            num_v_heads,
            head_v_dim,
            head_k_dim,
            gen,
        )
        pool = engine.pool
    else:
        pool = (
            torch.randn(
                spec["slots"], num_v_heads, head_v_dim, head_k_dim, generator=gen, device=DEV
            )
            * 0.1
        ).to(spec.get("pool_dtype", torch.float32))
    cu = torch.tensor([0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.long, device=DEV)
    indices = torch.tensor(spec["indices"], dtype=torch.int32, device=DEV)
    output = None
    if spec.get("output") == "buffer":
        output = torch.full(
            (1, total, num_v_heads, head_v_dim), -7.0, device=DEV, dtype=torch.bfloat16
        )
    elif spec.get("output") == "slice":
        output = torch.full(
            (1, total + _OUTPUT_TAIL, num_v_heads, head_v_dim),
            -7.0,
            device=DEV,
            dtype=torch.bfloat16,
        )[:, :total]
    extra = {}
    if "initial" in spec:
        extra["has_initial_state"] = torch.tensor(spec["initial"], dtype=torch.bool, device=DEV)
    if spec.get("linear_gate"):
        extra["linear_gate"] = True
        g = torch.exp(g)
    args = dict(
        **extra,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=None,
        initial_state=pool,
        initial_state_indices=indices,
        inplace_indexed_state_update=True,
        output_final_state=False,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
        output=output,
        state_workspace=None,
    )
    return args, engine


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args, engine = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    try:
        if engine is None:
            pool = args["initial_state"]
            pool_before = pool.clone()
            ref = op.reference(**args)
        else:
            compact, renamed = engine.dense(args["initial_state_indices"])
            ref = op.reference(**dict(args, initial_state=compact, initial_state_indices=renamed))
        output = args["output"]
        # A `slice` cell's output is a prefix view of the wider buffer the target owns.
        full = None if output is None else output._base
        tail_before = None if full is None else full[:, output.shape[1] :].clone()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            o, final_state = op(**args)
        # FlashInfer falls back from a CP heuristic match with a RuntimeWarning, so a CP
        # cell certifies the CP pipeline only if none was raised.
        fallback = [str(w.message) for w in caught if "falling back to non-CP" in str(w.message)]
        path = "chunked" if fallback else _flashinfer_path(cell.spec)

        got_pool = pool if engine is None else engine.touched()
        _compare((o, final_state, got_pool), ref)
        print(
            f"\n[{cell.why}] path={path} sms={get_device_sm_count(torch.device(DEV))}; "
            f"{_stats('o', o, ref[0])}; {_stats('state', got_pool, ref[2])}"
        )
        if "fi_path" in cell.spec:
            assert path == cell.spec["fi_path"], (path, fallback)
        if output is not None:
            assert o.data_ptr() == output.data_ptr(), "output= must be written in place"
        if tail_before is not None:
            assert torch.equal(full[:, output.shape[1] :], tail_before), "rows past output written"
        if engine is None:
            named = set(args["initial_state_indices"].tolist())
            unnamed = [s for s in range(pool.shape[0]) if s not in named]
            assert torch.equal(pool[unnamed], pool_before[unnamed]), "unnamed slots written"
        else:
            engine.assert_rest_unchanged()
    finally:
        if engine is not None:
            engine.close()


_CARRIED = next(c for c in op.CELLS if "initial" in c.spec)


def test_has_initial_state_is_the_reset_then_the_plain_call() -> None:
    """The flags replace `reset_recurrent_state_rows` before the call: zeroing the fresh
    slots and running the plain in-place call gives the same output and pool, bitwise; a
    linear gate that is `torch.exp` of the plain call's log gate changes nothing."""
    args, _ = _build(_CARRIED.spec, seed=5)
    flags = args.pop("has_initial_state")
    linear_gate = args.pop("linear_gate", False)
    pool = args["initial_state"]
    fresh = args["initial_state_indices"][~flags].long()
    pool[fresh] = torch.randn_like(pool[fresh])  # stale rows the flags must not read
    plain = dict(args, initial_state=pool.clone(), output=args["output"].clone())
    plain["initial_state"][fresh] = 0.0
    if linear_gate:
        plain["g"] = torch.log(args["g"])
        # The linear gate of the target call is torch.exp of the log gate, bitwise.
        args["g"] = torch.exp(plain["g"])

    o_plain, _ = op(**plain)
    o, final_state = op(**args, has_initial_state=flags, linear_gate=linear_gate)

    assert final_state is None
    assert torch.equal(o, o_plain)
    assert torch.equal(pool, plain["initial_state"])


def test_band_refuses_a_misgrouped_key_head() -> None:
    """The band is wider than torch's bf16 default, so it has to refuse the error a 4/12
    rank can make and 16/48 shares: a value head reading the wrong key head. Rolling the
    key heads by one makes value head i read key head (i // 3 - 1) mod 4; the op's answer
    then has to fall outside the band around the true reference."""
    spec = dict(
        geometry=RANK4_GEOMETRY,
        seqlens=(200, 37, 91),
        slots=64,
        indices=(63, 0, 17),
        pool_dtype=torch.bfloat16,
        output="buffer",
        initial=(True, False, True),
        linear_gate=True,
    )
    args, _ = _build(spec, seed=11)
    pool = args["initial_state"]
    ref = op.reference(**args)
    rolled = dict(args, q=args["q"].roll(1, dims=2), k=args["k"].roll(1, dims=2))
    o, _ = op(**rolled)
    print(f"\n[misgrouped key head] {_stats('o', o, ref[0])}; {_stats('state', pool, ref[2])}")
    with pytest.raises(AssertionError):
        _compare((o, None, pool), ref)
