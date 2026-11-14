# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the FP8 unified target's Gated-DeltaNet entries over their certified cells, and
for launches no cell reaches. Engine-sized state pools are reserved virtual ranges mapped under
the touched slots only (`_SparsePage`); every other byte of their pages is checked unchanged."""

from __future__ import annotations

import math
import types
import warnings
import zlib

import cuda.bindings.driver as cuda_drv
import pytest
import torch
import torch.nn.functional as F
from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import should_use_cp_host
from flashinfer.utils import get_compute_capability, get_device_name, get_device_sm_count

import tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm.causal_conv1d_token_major__verify as ctm  # noqa: E501
import tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm.gdn_cached_replay_cute__verify as cr  # noqa: E501
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.norm.rms_norm_gated_token_major import (  # noqa: E501
    rms_norm_gated_token_major,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.gdn_prefill_conv_post import (  # noqa: E501
    HEAD_K_DIM,
    HEAD_V_DIM,
    NUM_K_HEADS,
    NUM_V_HEADS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    modeling as _m,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.gemm.low_m_bf16_mm__decode import (  # noqa: E501
    low_m_bf16_mm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm.causal_conv1d_token_major__verify import (  # noqa: E501
    causal_conv1d_token_major,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm.flashinfer_chunk_gated_delta_rule import (  # noqa: E501
    INPLACE_MAX_TOKENS,
    flashinfer_chunk_gated_delta_rule,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.ssm.gdn_in_proj_conv__verify import (  # noqa: E501
    fused_tactic,
    gdn_in_proj_conv,
)
from tensorrt_llm._torch.modules.fla.cache_manager import (
    MIN_REPLAY_HISTORY_SIZE,
    ReplayStateUpdateMetadata,
    _advance_replay_state,
)
from tensorrt_llm._torch.modules.fla.cached_replay import (
    CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
    commit_gdn_cached_replay_history_layers,
)
from tensorrt_llm._torch.modules.fla.cached_replay import (
    fused_recurrent_gated_delta_rule_cached_replay_update as stock_replay,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import (
    REPLAY_WORK_ITEM_WIDTH,
    _build_replay_work_items_triton,
)

__extra_import_path__ = ["../../.."]
from _validating import ULP, certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "the Gated-DeltaNet entries require a CUDA device"

DEV = "cuda"
#: Key heads, value heads, key and value head dims of the whole layer (tp1, a dep4 rank).
H, HV, K, V = NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM


# ------------------------------------------------- reserved pages, mapped where touched ---


def _scattered(n: int, slots: int) -> list[int]:
    """`n` distinct slots: both ends, then a stride-37 walk (coprime to every `slots - 2` here)."""
    rest = [(i * 37 + 4) % (slots - 2) + 1 for i in range(max(n - 2, 0))]
    picked = ([slots - 1, 0] + rest)[:n]
    assert len(set(picked)) == n, (n, slots)
    return picked


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
    """A reserved virtual range of `elems` `dtype` elements with physical memory mapped under
    `regions` (element ranges) only, so a stray access faults."""

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
    """The stock manager's coalesced state page (`layers` `[v_heads, head_v, head_k]` states a
    slot), viewed at the last layer's sub-page. Mapped under each touched slot's page plus the next
    slot's first layer, and under the view's base (torch reads the device off it)."""

    def __init__(self, slots, layers, used, v_heads, head_v, head_k, dtype, init):
        layer = v_heads * head_v * head_k
        stride = layers * layer
        total = slots * stride
        base = (layers - 1) * layer
        self.used = sorted(set(used))
        self.spans = [(s * stride, min((s + 1) * stride + layer, total)) for s in self.used]
        self.page = _SparsePage(total, [*self.spans, (base, base + layer)], dtype)
        self.own, self.layer = base, layer
        self.pool = self.page.view(
            base, (slots, v_heads, head_v, head_k), (stride, head_v * head_k, head_k, 1)
        )
        self.flat = [self.page.view(lo, (hi - lo,), (1,)) for lo, hi in self.spans]
        for f in self.flat:
            init(f)
        torch.cuda.synchronize()
        self.before = [f.clone() for f in self.flat]

    def dense(self, slots: torch.Tensor):
        """A dense copy of the touched slots and `slots` renumbered into it (-1 stays)."""
        index = {s: i for i, s in enumerate(self.used)}
        compact = torch.stack([self.pool[s] for s in self.used]).contiguous()
        renamed = [index[s] if s >= 0 else -1 for s in slots.tolist()]
        return compact, torch.tensor(renamed, dtype=slots.dtype, device=DEV)

    def touched(self) -> torch.Tensor:
        return torch.stack([self.pool[s] for s in self.used])

    def assert_rest_unchanged(self) -> None:
        """Every mapped byte outside this layer's state of each touched slot is unchanged."""
        torch.cuda.synchronize()
        lo, hi = self.own, self.own + self.layer
        for f, b, s in zip(self.flat, self.before, self.used):
            assert torch.equal(f[:lo], b[:lo]) and torch.equal(f[hi:], b[hi:]), (
                f"slot {s}: the call wrote outside its state"
            )

    def close(self) -> None:
        del self.pool, self.flat
        self.page.close()


# ------------------------------------------------ flashinfer_chunk_gated_delta_rule ---

#: The Triton chunk op's band (rtol bf16 default; atol 2e-3 on `o`, 2e-2 on states): this op is
#: the same class of sum against fp64; a value head reading the wrong key head lands far outside.
CHUNK_RTOL, CHUNK_ATOL_OUTPUT, CHUNK_ATOL_STATE = 1.6e-2, 2e-3, 2e-2
#: Rows past `out[:, :num_tokens]` in a `slice` cell's wider output buffer; they must not change.
_CHUNK_OUTPUT_TAIL = 16


def _chunk_compare(out: tuple, ref: tuple) -> None:
    """`(o, final_state, pool)` against `reference`'s triple; `None` must match."""
    for got, want, atol in zip(out, ref, (CHUNK_ATOL_OUTPUT, CHUNK_ATOL_STATE, CHUNK_ATOL_STATE)):
        assert (got is None) == (want is None), (got is None, want is None)
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape, (
            (got.dtype, got.shape),
            (want.dtype, want.shape),
        )
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got.float(), want.float(), rtol=CHUNK_RTOL, atol=atol)


def _flashinfer_path(spec) -> str:
    """The FlashInfer kernel the call reaches: `inplace` for a `has_initial_state` call of at most
    `INPLACE_MAX_TOKENS` tokens, else `cp` when FlashInfer's heuristic picks it, else `chunked`."""
    if "initial" in spec and sum(spec["seqlens"]) <= INPLACE_MAX_TOKENS:
        return "inplace"
    device = torch.device(DEV)
    cp = should_use_cp_host(
        len(spec["seqlens"]) * spec["geometry"][1],
        get_device_sm_count(device),
        get_device_name(device),
        device_capability=get_compute_capability(device),
    )
    return "cp" if cp else "chunked"


def _chunk_build(spec, seed):
    """`(args, engine)`: inputs in a GDN post-conv step's domain (unit-norm q/k, g < 0 in log
    space, beta a sigmoid), and the `_EnginePool` behind the pool or None."""
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
            torch.bfloat16,
            lambda f: f.copy_(torch.randn(f.numel(), generator=gen, device=DEV) * 0.1),
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
            (1, total + _CHUNK_OUTPUT_TAIL, num_v_heads, head_v_dim),
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


def _check_flashinfer_chunk_gated_delta_rule(cell) -> None:
    args, engine = _chunk_build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    try:
        if engine is None:
            pool = args["initial_state"]
            pool_before = pool.clone()
            ref = flashinfer_chunk_gated_delta_rule.reference(**args)
        else:
            compact, renamed = engine.dense(args["initial_state_indices"])
            ref = flashinfer_chunk_gated_delta_rule.reference(
                **dict(args, initial_state=compact, initial_state_indices=renamed)
            )
        output = args["output"]
        full = None if output is None else output._base
        tail_before = None if full is None else full[:, output.shape[1] :].clone()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            o, final_state = flashinfer_chunk_gated_delta_rule(**args)
        # FlashInfer falls back from a CP heuristic match with a RuntimeWarning, so a CP
        # cell certifies the CP pipeline only if none was raised.
        fallback = [str(w.message) for w in caught if "falling back to non-CP" in str(w.message)]
        path = "chunked" if fallback else _flashinfer_path(cell.spec)

        got_pool = pool if engine is None else engine.touched()
        _chunk_compare((o, final_state, got_pool), ref)
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


# ------------------------------------------------------- the cached-replay test harness ---

#: `replay_history_size` = max(MIN_REPLAY_HISTORY_SIZE, T): the minimum at every T <= 8.
HIST = MIN_REPLAY_HISTORY_SIZE
BF16_ULP = ULP[torch.bfloat16]

#: This checkpoint's GDN `A_log` / `dt_bias` statistics: mean, standard deviation, range.
A_LOG_STATS = (1.006, 1.487, (-7.125, 5.0625))
DT_BIAS_STATS = (-4.411, 1.732, (-11.4375, 2.53125))

#: Element gate of the bf16 members (output, old_u, old_k), in bf16 ulps of their block's absmax:
#: the kernel rounds q, k and the history-mixing coefficients to bf16 before its dots.
_ELEMENT_ULPS = 4.0
#: Relative-RMS gate of the output, in bf16 ulps.
_OUTPUT_RMS_ULPS = 1.5
#: Element gate of an fp32 checkpoint state, relative to its (slot, head) absmax: the commit
#: splits each history term into bf16 hi and lo parts (~16 bits), up to 16 terms within 2^-17.
_STATE_TOL = 2.0**-13
#: Element gate of a bf16 checkpoint state: that fp32 commit rounded once, as the reference's is,
#: so the two are equal or adjacent, at most 2 ulps of the head's absmax apart.
_BF16_STATE_ULPS = 2.0
#: Gate of the log-decays, |dG| <= atol + rtol * |G|: G enters only as exp(G_t - G_j), atol is an
#: eighth of a bf16 ulp of the u it scales, rtol covers the cumsum where |G| reaches thousands.
_G_ATOL, _G_RTOL = 2.0**-11, 2.0**-18


def _gate(name, got, want, element, rms=None):
    """Every element of `got` within `element` of its block's absmax in `want` (a block is the
    last two dims, so a near-silent element's rounding does not read as relative noise)."""
    block = got.shape[-2] * got.shape[-1]
    g, w = got.float().reshape(-1, block), want.float().reshape(-1, block)
    keep = ~torch.isnan(w).any(1)
    g, w = g[keep], w[keep]
    scale = w.abs().amax(1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    ratio = ((g - w).abs() / scale).nan_to_num(nan=float("inf"))
    worst = ratio.max().item() if ratio.numel() else 0.0
    assert worst <= element, (
        f"{name}: an element is {worst:.3e} of its block's absmax from the reference "
        f"(gate {element:.3e})"
    )
    if rms is not None and w.numel():
        rel = ((g - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt().clamp_min(1e-30)).item()
        assert rel <= rms, f"{name}: relative RMS {rel:.3e} exceeds {rms:.3e}"


def _replay_compare(out: tuple, ref: tuple) -> None:
    """`(returned, ssm_states, old_u, old_k, old_G)` against `reference`'s five by the gates above;
    a member `None` on both sides is skipped (the exact twin compares the output alone)."""
    names = ("output", "ssm_states", "old_u", "old_k", "old_G")
    assert len(out) == len(ref) == len(names)
    for name, got, want in zip(names, out, ref):
        assert (got is None) == (want is None), f"{name}: present on one side only"
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape, (
            name,
            (got.dtype, tuple(got.shape)),
            (want.dtype, tuple(want.shape)),
        )
        if name == "old_G":
            torch.testing.assert_close(got, want, rtol=_G_RTOL, atol=_G_ATOL, msg=name)
        elif name == "ssm_states":
            bf16 = got.dtype == torch.bfloat16
            _gate(name, got, want, _BF16_STATE_ULPS * BF16_ULP if bf16 else _STATE_TOL)
        else:
            rms = _OUTPUT_RMS_ULPS * BF16_ULP if name == "output" else None
            _gate(name, got, want, _ELEMENT_ULPS * BF16_ULP, rms)


def _seed(why):
    return zlib.crc32(why.encode())


def _gating(gen, layers, hv=HV):
    """Per-layer A_log / dt_bias around the checkpoint's statistics; head 0 pinned to the corner
    that forgets everything every token (g ~ -400), head 1 to the one that hardly decays."""
    (a_mean, a_std, a_range), (d_mean, d_std, d_range) = A_LOG_STATS, DT_BIAS_STATS
    a_log = torch.randn(layers, hv, generator=gen, device=DEV) * a_std + a_mean
    dt_bias = torch.randn(layers, hv, generator=gen, device=DEV) * d_std + d_mean
    a_log, dt_bias = a_log.clamp(*a_range), dt_bias.clamp(*d_range)
    a_log[:, 0], dt_bias[:, 0] = a_range[1], d_range[1]
    a_log[:, 1], dt_bias[:, 1] = a_range[0], d_range[0]
    return a_log.contiguous(), dt_bias.contiguous()


def _unit(x):
    return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)


class _ReplayEnginePage:
    """The stock page with the bench's `layers` on top, mapped under each touched slot's page, the
    next slot's first sub-page (where an overrun lands) and slot 0's page (the views' base)."""

    def __init__(self, slots, page_layers, layers, used, hv, dtype, gen):
        state = hv * V * K
        stride = page_layers * state
        total = slots * stride
        self.state, self.stride, self.layers = state, stride, layers
        self.top = (page_layers - layers) * state
        self.used = sorted(set(used))
        self.mapped = sorted(set(self.used) | {0})
        self.spans = [(s * stride, min((s + 1) * stride + state, total)) for s in self.mapped]
        self.page = _SparsePage(total, self.spans, dtype)
        self.pools = [
            self.page.view(self.top + j * state, (slots, hv, V, K), (stride, V * K, K, 1))
            for j in range(layers)
        ]
        self.flat = [self.page.view(lo, (hi - lo,), (1,)) for lo, hi in self.spans]
        for f in self.flat:
            f.normal_(0.0, 0.5, generator=gen)
        torch.cuda.synchronize()
        self.before = [f.clone() for f in self.flat]

    def assert_rest_unchanged(self) -> None:
        """Every mapped byte outside the touched slots' own sub-pages is what it was."""
        torch.cuda.synchronize()
        lo, hi = self.top, self.top + self.layers * self.state
        for s, f, b in zip(self.mapped, self.flat, self.before):
            if s in self.used:
                same = torch.equal(f[:lo], b[:lo]) and torch.equal(f[hi:], b[hi:])
            else:
                same = torch.equal(f, b)
            assert same, f"slot {s}: the page was written outside the cell's sub-pages"

    def close(self) -> None:
        del self.pools, self.flat, self.before
        self.page.close()


class _ReplayBench:
    """`layers` GDN layers' state as `MambaHybridCacheManagerV2` lays it out, driven as the engine
    drives it (work items, op per layer, all-layer commit, `_advance_replay_state`). `exact=True`
    adds the exact twin: fp32 `reference` over a never-wrapping history advanced by `p += accepted`
    alone. Over an engine page both see a dense copy of the touched slots (`position` maps them)."""

    def __init__(self, spec, seed, exact=True):
        self.spec = spec
        self.path, self.n, self.t = spec["path"], spec["n"], spec["t"]
        self.h, self.hv = spec.get("H", H), spec.get("HV", HV)
        assert spec.get("history_size", HIST) == HIST, spec
        self.width = 2 * self.h * K + self.hv * V
        self.layers = spec.get("layers", 2)
        self.slots = spec.get("slots", 64)
        self.steps = spec.get("steps", 8)
        self.gen = torch.Generator(device=DEV).manual_seed(seed)
        self.cpu = torch.Generator().manual_seed(seed)
        n, t, L, S = self.n, self.t, self.layers, self.slots
        h, hv = self.h, self.hv
        page_layers = spec.get("page_layers")

        if page_layers:
            chosen = torch.tensor(_scattered(n, S))
        else:
            chosen = torch.randperm(S, generator=self.cpu)[:n]
        chosen = chosen.to(torch.int32)
        live_rows = torch.ones(n, dtype=torch.bool)
        dup = tuple(spec.get("dup", ()))
        if dup:
            chosen[list(dup)] = int(chosen[dup[0]])
            # The padding rows' shared slot is nobody's state: the bookkeeping never names it.
            live_rows[list(dup)] = False
        self.state_indices = chosen.to(DEV)
        self.live = self.state_indices[live_rows.to(DEV)].clone()

        state_dtype = spec.get("state_dtype", torch.float32)
        self.page = self.idx = self.position = None
        if page_layers:
            self.page = _ReplayEnginePage(
                S, page_layers, L, chosen.tolist(), hv, state_dtype, self.gen
            )
            self.pool = self.page.pools
            self.layer_stride, self.slot_stride = self.page.state, self.page.stride
            self.idx = torch.tensor(self.page.used, dtype=torch.long, device=DEV)
            self.position = torch.full((S,), -1, dtype=torch.long, device=DEV)
            self.position[self.idx] = torch.arange(self.idx.numel(), device=DEV)
        else:
            # [slots, layers, HV, V, K]: the affine layout whose strides the all-layer commit takes.
            self.backing = (self._randn(S, L, hv, V, K) * 0.5).to(state_dtype)
            self.pool = [self.backing[:, i] for i in range(L)]
            self.layer_stride, self.slot_stride = self.backing.stride(1), self.backing.stride(0)
        self.old_x = torch.zeros(L, S, 2, HIST, hv, V, device=DEV, dtype=torch.bfloat16)
        self.old_B = torch.zeros(L, S, 2, HIST, h, K, device=DEV, dtype=torch.bfloat16)
        self.old_dt = torch.zeros(L, S, 2, hv, HIST, device=DEV)
        self.old_dA = self._randn(L, S, 2, hv, HIST)  # handed over as old_beta; never read
        self.pnat = torch.zeros(S, dtype=torch.int32, device=DEV)
        self.buf = torch.zeros(S, dtype=torch.int32, device=DEV)
        self.a_log, self.dt_bias = _gating(self.gen, L, hv)
        if spec.get("start", "warm") == "warm":
            self.old_x.copy_(self._randn(L, S, 2, HIST, hv, V) * 0.5)
            self.old_B.copy_(_unit(self._randn(L, S, 2, HIST, h, K)))
            raw_a = self._randn(L, S, 2, hv, HIST)
            g = -torch.exp(self.a_log)[:, None, None, :, None] * F.softplus(
                raw_a + self.dt_bias[:, None, None, :, None]
            )
            self.old_dt.copy_(g.cumsum(-1))
            pnat = torch.randint(0, HIST + 1, (S,), generator=self.cpu, dtype=torch.int32)
            live = self.live.long().cpu()
            pnat[live[0]] = HIST  # a wrap at step 0
            if live.numel() > 1:
                pnat[live[1]] = 0  # an append at step 0
            self.pnat.copy_(pnat)
            self.buf.copy_(torch.randint(0, 2, (S,), generator=self.cpu, dtype=torch.int32))
        self.meta = ReplayStateUpdateMetadata(self.pnat, self.buf, t, HIST)
        self.work_items = self.n_writes = None
        if self.path == "all_layer":
            # The engine hands out work items exactly from the all-layer commit's minimum batch.
            assert n >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE, n
            self.work_items = torch.zeros(n, REPLAY_WORK_ITEM_WIDTH, dtype=torch.int32, device=DEV)
            self.n_writes = torch.zeros(1, dtype=torch.int32, device=DEV)

        self.exact = exact
        if exact:
            self.depth = HIST + (self.steps + 1) * t
            self.x_pool = [self.compact(p).clone() for p in self.pool]
            rows = self.x_pool[0].shape[0]
            self.x_u = torch.zeros(L, rows, 1, self.depth, hv, V, device=DEV)
            self.x_k = torch.zeros(L, rows, 1, self.depth, h, K, device=DEV)
            self.x_G = torch.zeros(L, rows, 1, hv, self.depth, device=DEV)
            self.x_pnat = self.compact(self.pnat).clone()
            self.x_buf = torch.zeros_like(self.x_pnat)
            for s in self.live.tolist():
                p, b = int(self.pnat[s]), int(self.buf[s])
                c = s if self.position is None else int(self.position[s])
                self.x_u[:, c, 0, :p] = self.old_x[:, s, b, :p].float()
                self.x_k[:, c, 0, :p] = self.old_B[:, s, b, :p].float()
                self.x_G[:, c, 0, :, :p] = self.old_dt[:, s, b, :, :p]

    def _randn(self, *shape, dtype=torch.float32):
        return torch.randn(*shape, generator=self.gen, device=DEV, dtype=dtype)

    def compact(self, per_slot: torch.Tensor) -> torch.Tensor:
        """A per-slot tensor as the reference sees it: over a page, the touched slots' rows."""
        return per_slot if self.idx is None else per_slot[self.idx]

    def compact_indices(self, slots: torch.Tensor) -> torch.Tensor:
        """Slot ids as rows of `compact`'s output (negative ids stay negative)."""
        if self.position is None:
            return slots
        renamed = torch.where(slots >= 0, self.position[slots.long().clamp(min=0)], -1)
        return renamed.to(slots.dtype)

    def close(self) -> None:
        if self.page is not None:
            self.pool = []
            self.page.close()

    def inputs(self):
        """One layer's post-conv `[q|k|v]` rows and raw `a`/`b`, as the verify step passes them."""
        n, t, h, hv = self.n, self.t, self.h, self.hv
        packed = self._randn(n * t, self.width, dtype=torch.bfloat16)
        rows = packed.view(n, t, self.width)
        q = rows[..., : h * K].view(n, t, h, K)
        k = rows[..., h * K : 2 * h * K].view(n, t, h, K)
        v = rows[..., 2 * h * K :].view(n, t, hv, V)
        a = self._randn(n, t, hv, dtype=torch.bfloat16)
        b = self._randn(n, t, hv, dtype=torch.bfloat16)
        return dict(q=q, k=k, v=v, a=a, b=b, packed=packed)

    def kwargs(self, layer, act):
        return dict(
            q=act["q"],
            k=act["k"],
            v=act["v"],
            g=act["a"],
            beta=act["b"],
            ssm_states=self.pool[layer],
            state_indices=self.state_indices,
            old_u=self.old_x[layer],
            old_k=self.old_B[layer],
            old_G=self.old_dt[layer],
            old_beta=self.old_dA[layer],
            cache_buf_idx=self.buf,
            prev_num_accepted_tokens=self.pnat,
            history_size=HIST,
            scale=None,
            use_qk_l2norm_in_kernel=True,
            A_log=self.a_log[layer],
            dt_bias=self.dt_bias[layer],
            launch_with_pdl=False,
            replay_work_items=self.work_items,
            n_writes=self.n_writes,
            packed_qkv=act["packed"],
            use_all_layer_commit=self.path == "all_layer",
            output=None,
        )

    def exact_kwargs(self, layer, kw):
        return dict(
            kw,
            ssm_states=self.x_pool[layer],
            state_indices=self.compact_indices(kw["state_indices"]),
            old_u=self.x_u[layer],
            old_k=self.x_k[layer],
            old_G=self.x_G[layer],
            cache_buf_idx=self.x_buf,
            prev_num_accepted_tokens=self.x_pnat,
            history_size=self.depth,
            replay_work_items=None,
            n_writes=None,
            use_all_layer_commit=False,
            output=None,
        )

    def adopt_exact(self, layer, ref):
        assert torch.equal(ref[1], self.x_pool[layer]), "the twin must never wrap"
        self.x_u[layer].copy_(ref[2])
        self.x_k[layer].copy_(ref[3])
        self.x_G[layer].copy_(ref[4])

    def state(self, layer):
        return self.pool[layer], self.old_x[layer], self.old_B[layer], self.old_dt[layer]

    def observed(self, layer):
        """`state(layer)` through `compact`, as `reference`'s results are laid out."""
        return tuple(self.compact(x) for x in self.state(layer))

    def prepare(self):
        if self.path == "all_layer":
            self.work_items.zero_()
            _build_replay_work_items_triton(
                self.live, self.pnat, self.buf, self.work_items, self.n_writes, self.t, HIST
            )

    def commit(self):
        if self.path == "all_layer":
            commit_gdn_cached_replay_history_layers(
                ssm_states=self.pool[0],
                ssm_state_num_layers=self.layers,
                ssm_state_layer_stride=self.layer_stride,
                ssm_state_slot_stride=self.slot_stride,
                old_u=self.old_x,
                old_k=self.old_B,
                old_G=self.old_dt,
                replay_work_items=self.work_items,
                n_writes=self.n_writes,
                history_size=HIST,
            )

    def draw_acceptance(self):
        """Accepted tokens per live row, bonus token included, by `spec["accept"]`."""
        mode = self.spec.get("accept", "random")
        rows = self.live.numel()
        if mode == "full":
            return torch.full((rows,), self.t, dtype=torch.int32, device=DEV)
        assert mode == "random", mode
        return torch.randint(1, self.t + 1, (rows,), generator=self.cpu, dtype=torch.int32).to(DEV)

    def accept(self, accepted):
        _advance_replay_state(self.meta, self.live, accepted)
        if self.exact:
            self.x_pnat[self.compact_indices(self.live).long()] += accepted

    def writes(self):
        """This step's append mask `[slots, 2, HIST]`, wrapping slot ids, and exact-fill count."""
        s = self.live.long()
        p, b = self.pnat[s].long(), self.buf[s].long()
        wrap = p + self.t > HIST
        half, start = torch.where(wrap, 1 - b, b), torch.where(wrap, 0, p)
        mask = torch.zeros(self.slots, 2, HIST, dtype=torch.bool, device=DEV)
        for t in range(self.t):
            mask[s, half, start + t] = True
        return mask, s[wrap], int((p + self.t == HIST).sum())


# -------------------------------------------------------------- gdn_cached_replay_cute__verify ---
# Each cell drives consecutive verify steps through `_ReplayBench`, holding every call to the
# entry's `reference` on its own inputs and every output to the exact twin.


_grc_op = cr.gdn_cached_replay_cute
GRC_K, GRC_V = cr.K, cr.V


def _grc_gates(bench, act, strided):
    """The raw a / b `[N, T, HV]`: contiguous, or the same values as column views of the rank's
    `[N*T, conv + z + 2 HV]` input projection, as the target hands them over."""
    if not strided:
        return act["a"], act["b"]
    n, t, hv = bench.n, bench.t, bench.hv
    conv, z = 2 * bench.h * GRC_K + hv * GRC_V, hv * GRC_V
    proj = torch.randn(n * t, conv + z + 2 * hv, device=DEV, dtype=torch.bfloat16)
    a = proj[:, conv + z : conv + z + hv]
    b = proj[:, conv + z + hv :]
    a.copy_(act["a"].reshape(n * t, hv))
    b.copy_(act["b"].reshape(n * t, hv))
    a, b = a.reshape(n, t, hv), b.reshape(n, t, hv)
    assert a.stride() == (t * proj.shape[1], proj.shape[1], 1), a.stride()
    return a, b


def _grc_reference(bench, kw):
    """The entry's `reference` on a call's arguments, every per-slot buffer through `compact`."""
    if bench.idx is None:
        return _grc_op.reference(**kw)
    per_slot = (
        "ssm_states",
        "old_u",
        "old_k",
        "old_G",
        "old_beta",
        "cache_buf_idx",
        "prev_num_accepted_tokens",
    )
    compact = {name: bench.compact(kw[name]) for name in per_slot}
    return _grc_op.reference(
        **dict(kw, **compact, state_indices=bench.compact_indices(kw["state_indices"]))
    )


def _grc_drive(cell, bench) -> None:
    strided = cell.spec.get("strided_gates", False)
    wraps = appends = 0
    for _ in range(bench.steps):
        mask, wrapped, _ = bench.writes()
        wraps += wrapped.numel()
        appends += bench.live.numel() - wrapped.numel()
        bench.prepare()
        after_commit = []
        for layer in range(bench.layers):
            act = bench.inputs()
            kw = bench.kwargs(layer, act)
            kw["g"], kw["beta"] = _grc_gates(bench, act, strided)
            _, u, k, G = bench.state(layer)
            before = dict(u=u.clone(), k=k.clone(), G=G.clone(), dA=bench.old_dA[layer].clone())
            ref = _grc_reference(bench, kw)
            exact = _grc_op.reference(**bench.exact_kwargs(layer, kw))
            committed = None
            if bench.path == "all_layer":
                committed = _grc_reference(
                    bench,
                    dict(kw, replay_work_items=None, n_writes=None, use_all_layer_commit=False),
                )
            got = _grc_op(**kw)
            _replay_compare((got, *bench.observed(layer)), ref)
            _, u, k, G = bench.state(layer)
            assert torch.equal(bench.old_dA[layer], before["dA"]), "old_beta is never written"
            assert torch.equal(u[~mask], before["u"][~mask]), "old_u written outside the append"
            assert torch.equal(k[~mask], before["k"][~mask]), "old_k written outside the append"
            Gt, Gt0 = G.transpose(-1, -2), before["G"].transpose(-1, -2)
            assert torch.equal(Gt[~mask], Gt0[~mask]), "old_G written outside the append"
            bench.adopt_exact(layer, exact)
            after_commit.append((got, exact, committed))
        bench.commit()
        for layer, (got, exact, committed) in enumerate(after_commit):
            if committed is not None:
                # The partitioned call plus the manager's commit equals the in-kernel commit.
                _replay_compare((got, *bench.observed(layer)), committed)
            _replay_compare((got, None, None, None, None), (exact[0], None, None, None, None))
        if bench.page is not None:
            bench.page.assert_rest_unchanged()
        bench.accept(bench.draw_acceptance())
    assert wraps > 0 and appends > 0, f"the cell never wrapped or appended ({wraps}, {appends})"


def _grc_drive_padding(cell) -> None:
    """Trailing rows sharing one slot, as CUDA-graph padding does: the unique rows and every
    other slot match the reference run on the unique rows alone."""
    spec = {k_: v_ for k_, v_ in cell.spec.items() if k_ not in ("page_layers",)}
    spec["slots"] = 24
    bench = _ReplayBench(spec, seed=_seed(cell.why), exact=False)
    dup = spec["dup"]
    unique = bench.n - len(dup)
    shared = int(bench.state_indices[unique])
    others = torch.ones(bench.slots, dtype=torch.bool, device=DEV)
    others[shared] = False
    for _ in range(bench.steps):
        for layer in range(bench.layers):
            act = bench.inputs()
            kw = bench.kwargs(layer, act)
            kw["g"], kw["beta"] = _grc_gates(bench, act, spec.get("strided_gates", False))
            alone = dict(
                kw,
                q=kw["q"][:unique],
                k=kw["k"][:unique],
                v=kw["v"][:unique],
                g=kw["g"][:unique],
                beta=kw["beta"][:unique],
                state_indices=kw["state_indices"][:unique],
                packed_qkv=act["packed"][: unique * bench.t],
            )
            ref = _grc_op.reference(**alone)
            got = _grc_op(**kw)
            pool, u, k, G = bench.state(layer)
            _replay_compare(
                (got[:unique], pool[others], u[others], k[others], G[others]),
                (ref[0], *(x[others] for x in ref[1:])),
            )
        bench.accept(bench.draw_acceptance())


def _check_gdn_cached_replay_cute(cell) -> None:
    if "dup" in cell.spec:
        _grc_drive_padding(cell)
    else:
        bench = _ReplayBench(cell.spec, seed=_seed(cell.why))
        try:
            _grc_drive(cell, bench)
        finally:
            bench.close()


#: The geometries the target calls the entry at: a tp4/tep4 rank's heads, a dep4 rank's.
_GRC_GEOMETRIES = {"tep4": (cr.RANK_K_HEADS, cr.RANK_V_HEADS), "dep4": (16, 48)}


def _grc_rank_spec(geometry, **spec):
    h, hv = _GRC_GEOMETRIES[geometry]
    return dict(state_dtype=torch.bfloat16, H=h, HV=hv, **spec)


def _variant_replay_graph(n, path, geometry, pdl) -> None:
    """Two replay layers (and the all-layer commit when partitioned) captured once and replayed on
    new activations with acceptance between: bitwise the same steps run eagerly on twin state."""
    t = 4
    spec = _grc_rank_spec(geometry, path=path, n=n, t=t, layers=2, slots=40)
    seed = 9100 + n
    eager, graph, warm = (_ReplayBench(spec, seed, exact=False) for _ in range(3))
    acts = [warm.inputs() for _ in range(2)]
    gates = [_grc_gates(warm, act, True) for act in acts]

    def step(b):
        b.prepare()
        outs = []
        for layer in range(2):
            kw = b.kwargs(layer, acts[layer])
            kw["g"], kw["beta"] = gates[layer]
            kw["launch_with_pdl"] = pdl
            outs.append(_grc_op(**kw))
        b.commit()
        return outs

    step(warm)
    torch.cuda.synchronize()
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        outs_graph = step(graph)
    refill = torch.Generator(device=DEV).manual_seed(seed + 1)
    for _ in range(3):
        for act, (a, b) in zip(acts, gates):
            for x in (act["packed"], a, b):
                x.copy_(torch.randn(x.shape, generator=refill, device=DEV, dtype=x.dtype))
        outs_eager = step(eager)
        captured.replay()
        torch.cuda.synchronize()
        for e, g in zip(outs_eager, outs_graph):
            assert torch.equal(e, g), "replayed output differs from eager"
        for name in ("backing", "old_x", "old_B", "old_dt", "pnat", "buf"):
            assert torch.equal(getattr(eager, name), getattr(graph, name)), f"{name} differs"
        accepted = eager.draw_acceptance()
        eager.accept(accepted)
        graph.accept(accepted)


def _variant_ht_verify(n) -> None:
    """A tp2/tep2 rank's HT verify step as `_gdn_verify` launches it (`_GDN_REPLAY_HT_LAUNCH`,
    gated norm folded, the conv too at few requests) bitwise the plain chain on twin state. A
    folded conv leaves the pool to the acceptance promote (the plain advance: the last window)."""
    core = types.SimpleNamespace(num_k_heads=8, num_v_heads=24)
    t, h, hv = 3, 8, 24
    assert _m._gdn_verify_folds_norm(core, n, t)
    conv_fold = _m._gdn_verify_folds_conv(core, n)
    pdl, mapping = _m._gdn_replay_launch(core, n)
    assert (pdl, mapping) != (False, None), "every HT bucket changes the launch"
    spec = dict(
        state_dtype=torch.bfloat16, H=h, HV=hv, path="all_layer", n=n, t=t, layers=1, slots=n + 8
    )
    width = 2 * h * GRC_K + hv * GRC_V
    gen = torch.Generator(device=DEV).manual_seed(5100 + n)
    proj = torch.randn(n * t, width + hv * GRC_V + 2 * hv, generator=gen, device=DEV).to(
        torch.bfloat16
    )
    x = proj[:, :width]
    z = proj[:, width : width + hv * GRC_V].reshape(n * t, hv, GRC_V)
    weight = (torch.randn(width, 4, generator=gen, device=DEV) * 0.3).to(torch.bfloat16)
    norm_w = (1 + 0.1 * torch.randn(GRC_V, generator=gen, device=DEV)).to(torch.bfloat16)
    pool0 = (torch.randn(n + 8, width, 3, generator=gen, device=DEV) * 0.5).to(torch.bfloat16)
    rows = torch.arange(n, dtype=torch.int32, device=DEV).flip(0).contiguous()
    results = []
    for target in (False, True):
        bench = _ReplayBench(spec, 7700 + n, exact=False)
        act = bench.inputs()
        a, b = _grc_gates(bench, act, True)
        pool = pool0.clone()
        window = torch.zeros(n, t, width, 3, dtype=torch.bfloat16, device=DEV)
        bench.prepare()
        kw = bench.kwargs(0, act)
        kw.update(g=a, beta=b)
        if target and conv_fold:
            fold = dict(x=x, weight=weight, pool=pool, window=window, window_rows=rows)
            qkv = x
            kw.update(packed_qkv=None, conv_fold=fold)
        else:
            qkv = causal_conv1d_token_major(
                x, n, pool, weight, bench.state_indices, window, rows, -1
            )
            kw.update(packed_qkv=qkv)
        q, k, v = torch.split(qkv, [h * GRC_K, h * GRC_K, hv * GRC_V], 1)
        kw.update(
            q=q.reshape(n, t, h, GRC_K), k=k.reshape(n, t, h, GRC_K), v=v.reshape(n, t, hv, GRC_V)
        )
        if target:
            kw.update(launch_with_pdl=pdl, gated_norm=dict(z=z, weight=norm_w, eps=1e-6))
            if mapping is not None:
                kw["config"] = mapping
        out = _grc_op(**kw)
        if not target:
            out = rms_norm_gated_token_major(
                torch.reshape(out, [-1, GRC_V]),
                z,
                weight=norm_w,
                eps=1e-6,
                fp8_scale=None,
                gate_activation="sigmoid",
            )
        bench.commit()
        torch.cuda.synchronize()
        out = torch.reshape(out, [n * t, hv * GRC_V])
        results.append(([out.clone()] + [y.clone() for y in bench.state(0)], window, pool, bench))
    (ref, win_ref, pool_ref, bench_ref), (got, win_got, pool_got, bench_got) = results
    for name, y, w in zip(("normed out", "state", "u", "k", "G"), got, ref):
        assert torch.equal(y, w), f"{name}: n={n} launch {pdl, mapping}, conv fold {conv_fold}"
    assert torch.equal(win_got, win_ref), "snapshot windows"
    if conv_fold:
        assert torch.equal(pool_got, pool0), "the fold must not advance the pool"
        live = bench_ref.state_indices.long()
        assert torch.equal(pool_ref[live], win_ref[rows.long(), t - 1]), "advance = last window"
    else:
        assert torch.equal(pool_got, pool_ref), "the conv pool"
    for bench in (bench_ref, bench_got):
        bench.close()


# ------------------------------------------ DecodeTarget._gdn_verify's replay by rank geometry ---


GVR_K = GVR_V = 128
GVR_HIST, GVR_T, GVR_CONV_W, GVR_PAGE_LAYERS, GVR_LAYERS = 16, 4, 4, 36, 2
#: A rank's GDN (key, value) heads per member.
GVR_GEOM = {"tep4": (4, 12), "tep2": (8, 24), "dep4": (16, 48)}


def _gvr_build(member: str, n: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    bf = torch.bfloat16
    h, hv = GVR_GEOM[member]
    key_dim, value_dim = h * GVR_K, hv * GVR_V
    conv_dim = 2 * key_dim + value_dim
    split = [conv_dim, value_dim, hv, hv]
    slots = n + 3
    page = (
        torch.randn(slots, GVR_PAGE_LAYERS, hv, GVR_V, GVR_K, generator=g, device=DEV) * 0.05
    ).to(bf)
    conv_page = torch.randn(
        slots, GVR_PAGE_LAYERS, conv_dim, GVR_CONV_W - 1, generator=g, device=DEV
    )
    conv_page = conv_page.to(bf)
    old, int_conv, layers, projs = {}, {}, [], []
    for i in range(GVR_LAYERS):
        u = (torch.randn(slots, 2, GVR_HIST, hv, GVR_V, generator=g, device=DEV) * 0.1).to(bf)
        k = torch.randn(slots, 2, GVR_HIST, h, GVR_K, generator=g, device=DEV)
        k = (k / k.norm(dim=-1, keepdim=True)).to(bf)
        G = -torch.cumsum(torch.rand(slots, 2, hv, GVR_HIST, generator=g, device=DEV) * 0.3, -1)
        old[i] = (u, k, G.contiguous(), torch.zeros_like(G))
        int_conv[i] = torch.zeros(n, GVR_T, conv_dim, GVR_CONV_W - 1, device=DEV, dtype=bf)
        layers.append(
            types.SimpleNamespace(
                conv_w=(torch.randn(conv_dim, GVR_CONV_W, generator=g, device=DEV) * 0.3).to(bf),
                a_log=(torch.randn(hv, generator=g, device=DEV) * 1.5 + 1.0).clamp(-7, 5),
                dt_bias=(torch.randn(hv, generator=g, device=DEV) * 1.7 - 4.4).clamp(-11, 2.5),
            )
        )
        projs.append(torch.randn(n * GVR_T, sum(split), generator=g, device=DEV).to(bf))
    gc = torch.Generator().manual_seed(seed)
    cache_buf_idx = torch.randint(0, 2, (slots,), generator=gc, dtype=torch.int32).to(DEV)
    # Odd slots hold 13..15 pending tokens (they wrap: p + 4 > 16), even ones 1..8 (they
    # append); the step's rows start at slot 1, so every batch has a wrapping row.
    pending = torch.tensor(
        [(13 + (s // 2) % 3) if s % 2 else (1 + s % 8) for s in range(slots)], dtype=torch.int32
    ).to(DEV)
    replay = _m._Replay(old, cache_buf_idx, pending, GVR_HIST, GVR_T)
    pools = types.SimpleNamespace(
        ssm={i: page[:, i] for i in range(GVR_LAYERS)},
        conv={i: conv_page[:, i] for i in range(GVR_LAYERS)},
    )
    core = types.SimpleNamespace(
        conv_dim=conv_dim,
        key_dim=key_dim,
        value_dim=value_dim,
        num_k_heads=h,
        head_k_dim=GVR_K,
        num_v_heads=hv,
        head_v_dim=GVR_V,
        _bound=types.SimpleNamespace(pools=pools, int_conv=int_conv, replay=replay),
    )
    rows = torch.arange(1, n + 1, dtype=torch.int32, device=DEV)  # slot 0 untouched
    verify = (torch.arange(n, dtype=torch.int32, device=DEV), None, None, False, None)

    def run():
        outs = []
        for i in range(GVR_LAYERS):
            mq, _, a, b = torch.split(projs[i], split, 1)
            outs.append(_m._gdn_verify(core, mq, a, b, i, layers[i], rows, n, GVR_T, verify))
        torch.cuda.synchronize()
        return outs

    state = dict(page=page, conv=conv_page, old=old, int_conv=int_conv)
    return run, state


def _gvr_rel(x, y):
    x, y = x.float(), y.float()
    return float((x - y).norm() / y.norm().clamp_min(1e-30))


def _gvr_stock_replay(**kw):
    """The engine's stock Triton cached replay in the CuTe entry's place: contiguous gate
    columns, no CuTe launch mapping (`_gdn_verify` folds nothing at these batches)."""
    assert "conv_fold" not in kw and "gated_norm" not in kw
    kw.pop("config", None)
    return stock_replay(**dict(kw, g=kw["g"].contiguous(), beta=kw["beta"].contiguous()))


@pytest.mark.parametrize("member,n", [("tep4", 1), ("tep2", 4), ("dep4", 8)])
def test_gdn_verify_agrees_with_the_engines_stock_replay(member, n, monkeypatch):
    """`_gdn_verify` over two MTP-3 layers at every rank geometry against the same layers over the
    engine's stock Triton replay on twin state: outputs, checkpoint page, history and conv agree
    to the replays' arithmetic (one fp32 recurrence, another dot order); other slots stay put."""
    run_c, st_c = _gvr_build(member, n, seed=500 + n)
    page0 = st_c["page"].clone()
    outs_c = run_c()
    monkeypatch.setattr(_m, "gdn_cached_replay_cute", _gvr_stock_replay)
    run_t, st_t = _gvr_build(member, n, seed=500 + n)
    outs_t = run_t()
    for oc, o_t in zip(outs_c, outs_t):
        assert torch.isfinite(oc).all()
        assert _gvr_rel(oc, o_t) < 4e-3, _gvr_rel(oc, o_t)
    live = slice(1, n + 1)
    assert _gvr_rel(st_c["page"][live], st_t["page"][live]) < 4e-3
    assert torch.equal(st_c["page"][0], page0[0]) and torch.equal(
        st_c["page"][n + 1 :], page0[n + 1 :]
    )
    assert not torch.equal(st_c["page"][live], page0[live]), "no row wrapped and committed"
    assert torch.equal(st_c["conv"], st_t["conv"]), "the conv is the same kernel on both paths"
    for i in range(GVR_LAYERS):
        assert torch.equal(st_c["int_conv"][i], st_t["int_conv"][i])
        for x, y in zip(st_c["old"][i][:3], st_t["old"][i][:3]):
            assert _gvr_rel(x[live], y[live]) < 4e-3
            assert torch.equal(x[0], y[0])


#: Requests the precompile test walks: past every `_GDN_REPLAY_HT_LAUNCH` bucket and the partition.
GVR_MAX_REQUESTS = 96


@pytest.mark.parametrize("separate", [False, True], ids=["coupled", "replay_slots"])
@pytest.mark.parametrize("pdl", [True, False], ids=["pdl", "no_pdl"])
@pytest.mark.parametrize("member", list(GVR_GEOM))
def test_gdn_replay_precompile_is_every_key_a_verify_step_launches(
    member, pdl, separate, monkeypatch
):
    """The CuTe replay kernels the bind precompiles (`_gdn_replay_launches`) are exactly those
    `_gdn_verify` launches over 1..`GVR_MAX_REQUESTS` requests, from a generation-only step
    (the gated norm folded where `_gdn_verify_folds_norm` says) and from a mixed step's verify
    rows (never folded), the partitioned ones with work items of the metadata's width. With
    `separate` the manager keeps the history at replay slots of its own: precompiled with
    `separate_replay_slots`, every step passes `replay_indices`. Kernels are recorded, not
    compiled or run."""
    monkeypatch.setattr(_m, "get_env_enable_pdl", lambda: pdl)
    monkeypatch.setattr(
        _m,
        "_GDN_REPLAY_HT_LAUNCH",
        tuple((bound, p and pdl, mapping) for bound, p, mapping in _m._GDN_REPLAY_HT_LAUNCH),
    )
    launched = []
    # An empty cache: every launch reaches `_compile`, which records the key instead.
    monkeypatch.setattr(cr, "_COMPILED", {})
    monkeypatch.setattr(cr, "_compile", lambda key, kern: launched.append(key) or (lambda *a: None))
    # The conv runs as its own kernel unless folded; the fold's operands are the engine's layout.
    monkeypatch.setattr(_m, "causal_conv1d_token_major", lambda mq, *a, **k: mq.contiguous())
    monkeypatch.setattr(_m, "gdn_conv_fold_supported", lambda *a: True)
    h, hv = GVR_GEOM[member]
    key_dim, value_dim = h * GVR_K, hv * GVR_V
    conv_dim = 2 * key_dim + value_dim
    bf, nmax, t = torch.bfloat16, GVR_MAX_REQUESTS, GVR_T
    slots = nmax + 1
    old = {
        0: (
            torch.zeros(slots, 2, GVR_HIST, hv, GVR_V, dtype=bf, device=DEV),
            torch.zeros(slots, 2, GVR_HIST, h, GVR_K, dtype=bf, device=DEV),
            torch.zeros(slots, 2, hv, GVR_HIST, device=DEV),
            torch.zeros(slots, 2, hv, GVR_HIST, device=DEV),
        )
    }
    zeros_i32 = torch.zeros(slots, dtype=torch.int32, device=DEV)
    work_items = torch.zeros(slots, REPLAY_WORK_ITEM_WIDTH, dtype=torch.int32, device=DEV)
    replay_slots = torch.arange(slots, 2 * slots, dtype=torch.int32, device=DEV)
    pools = types.SimpleNamespace(
        ssm={0: torch.zeros(slots, hv, GVR_V, GVR_K, dtype=bf, device=DEV)},
        conv={0: torch.zeros(slots, conv_dim, GVR_CONV_W - 1, dtype=bf, device=DEV)},
    )
    core = types.SimpleNamespace(
        conv_dim=conv_dim,
        key_dim=key_dim,
        value_dim=value_dim,
        num_k_heads=h,
        head_k_dim=GVR_K,
        num_v_heads=hv,
        head_v_dim=GVR_V,
        eps=1e-6,
        _bound=types.SimpleNamespace(
            pools=pools,
            int_conv={0: torch.zeros(nmax, t, conv_dim, GVR_CONV_W - 1, dtype=bf, device=DEV)},
            replay=_m._Replay(old, zeros_i32, zeros_i32, GVR_HIST, t),
        ),
    )
    layer = types.SimpleNamespace(
        conv_w=torch.zeros(conv_dim, GVR_CONV_W, dtype=bf, device=DEV),
        a_log=torch.zeros(hv, device=DEV),
        dt_bias=torch.zeros(hv, device=DEV),
        gnorm=torch.ones(GVR_V, dtype=bf, device=DEV),
    )
    proj = torch.zeros(nmax * t, conv_dim + value_dim + 2 * hv, dtype=bf, device=DEV)
    for n in range(1, nmax + 1):
        mq, z, a, b = torch.split(proj[: n * t], [conv_dim, value_dim, hv, hv], 1)
        rows = torch.arange(1, n + 1, dtype=torch.int32, device=DEV)
        partitioned = n >= _m.CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
        work = work_items[:n] if partitioned else None
        n_writes = zeros_i32[:1] if partitioned else None
        replay_indices = replay_slots[:n] if separate else None
        int_idx = torch.arange(n, dtype=torch.int32, device=DEV)
        verify = (int_idx, work, n_writes, partitioned, replay_indices)
        _m._gdn_verify(core, mq, a, b, 0, layer, rows, n, t, verify)  # a mixed step's rows
        fold_norm = _m._gdn_verify_folds_norm(core, n, t)
        _m._gdn_verify(core, mq, a, b, 0, layer, rows, n, t, verify, z=z if fold_norm else None)
    want = set(launched)
    got = cr.precompile_keys(
        t,
        h,
        hv,
        nmax,
        **_m._gdn_replay_launches(core, t),
        work_item_width=REPLAY_WORK_ITEM_WIDTH,
        separate_replay_slots=separate,
    )
    assert all(dict(key)["rsep"] == separate for key in got)
    assert {dict(key)["wi_w"] for key in got if dict(key)["use_wi"]} == {REPLAY_WORK_ITEM_WIDTH}
    assert got == want, (
        f"precompiled but never launched: {len(got - want)}, launched but not precompiled: "
        f"{len(want - got)}"
    )
    if pdl and hv * 2 <= _m._GDN_REPLAY_PDL_MAX_HEAD_ROWS:
        assert not any(not dict(key)["pdl"] and dict(key)["tpc"] == 2 for key in got), (
            "with PDL on, a batch of up to two requests never launches without it"
        )


# ------------------------------------------------ the replay history at slots of its own ---


def _replay_pages(slots, layers, h, hv):
    """The history and its bookkeeping in pages of their own, as a manager keeps them in its
    KVCM pool: one page per replay slot holding every layer's `old_u`, then every layer's
    `old_k`, then every layer's `old_G`, then the pending count and the half (int32), and 8
    bytes nobody writes; every byte 0xFF (NaN history, counts of -1) until written. The
    `[layers, slots, ...]` history views have the page as their slot stride, the bookkeeping
    columns the page as their element stride; `data` is the `[slots, page]` bytes, `tail`
    the offset of the unwritten bytes."""
    regions = (
        (torch.bfloat16, (2, HIST, hv, V)),
        (torch.bfloat16, (2, HIST, h, K)),
        (torch.float32, (2, hv, HIST)),
    )
    sizes = [math.prod(shape) * dtype.itemsize for dtype, shape in regions]
    page = layers * sum(sizes) + 16
    data = torch.full((slots, page), 0xFF, dtype=torch.uint8, device=DEV)
    flat = data.view(-1)
    views, offset = [], 0
    for (dtype, shape), size in zip(regions, sizes):
        es = dtype.itemsize
        rows = torch.empty(shape, device="meta").stride()
        views.append(
            torch.as_strided(
                flat.view(dtype),
                (layers, slots, *shape),
                (size // es, page // es, *rows),
                offset // es,
            )
        )
        offset += layers * size
    pnat, buf = (
        torch.as_strided(flat.view(torch.int32), (slots,), (page // 4,), offset // 4 + field)
        for field in (0, 1)
    )
    return types.SimpleNamespace(
        data=data, u=views[0], k=views[1], G=views[2], pnat=pnat, buf=buf, tail=offset + 8
    )


class _ReplaySlotBench(_ReplayBench):
    """`_ReplayBench` (the same draws for the same seed) with the history and its bookkeeping
    moved to replay slots of their own (`_replay_pages`): row `i`'s at `replay_indices[i]`, a
    permutation of slots past the checkpoint pool's, so no row's replay slot is a checkpoint
    slot. The work items carry the replay slot as their fifth field; the kernel is handed their
    first `width` fields (4: it reads the replay slot from `replay_indices`), the manager's
    all-layer commit the full items. `writes` is the coupled bench's alone (it reads the
    bookkeeping at the checkpoint slots)."""

    def __init__(self, spec, seed, width):
        super().__init__(spec, seed, exact=False)
        assert "dup" not in spec and "page_layers" not in spec, spec
        n, slots = self.n, self.state_indices.long()
        cpu = torch.Generator().manual_seed(seed + 1)
        replay = torch.randperm(2 * n, generator=cpu)[:n] + self.slots
        self.replay_indices = replay.to(DEV, torch.int32)
        rows = self.replay_indices.long()
        self.pages = _replay_pages(self.slots + 2 * n, self.layers, self.h, self.hv)
        pages = self.pages
        for paged, coupled in zip(
            (pages.u, pages.k, pages.G), (self.old_x, self.old_B, self.old_dt)
        ):
            paged[:, rows] = coupled[:, slots]
        pages.pnat[rows] = self.pnat[slots]
        pages.buf[rows] = self.buf[slots]
        self.old_x, self.old_B, self.old_dt = pages.u, pages.k, pages.G
        self.pnat, self.buf = pages.pnat, pages.buf
        self.meta = ReplayStateUpdateMetadata(self.pnat, self.buf, self.t, HIST)
        self.item_width = width
        self.kernel_items = None

    def prepare(self):
        if self.path == "all_layer":
            self.work_items.zero_()
            _build_replay_work_items_triton(
                self.live,
                self.pnat,
                self.buf,
                self.work_items,
                self.n_writes,
                self.t,
                HIST,
                self.replay_indices,
            )
            full = self.work_items.shape[1]
            self.kernel_items = (
                self.work_items
                if self.item_width == full
                else self.work_items[:, : self.item_width].contiguous()
            )

    def kwargs(self, layer, act):
        kw = super().kwargs(layer, act)
        kw["replay_indices"] = self.replay_indices
        if self.kernel_items is not None:
            kw["replay_work_items"] = self.kernel_items
        return kw

    def accept(self, accepted):
        _advance_replay_state(self.meta, self.replay_indices, accepted)


_SLOT_STATE = ("checkpoint", "old_u", "old_k", "old_G", "pending count", "half")


def _slot_state(bench):
    """A bench's checkpoints, and its rows' history and bookkeeping at their history slots
    (the replay slots, when it has them)."""
    rows = getattr(bench, "replay_indices", bench.state_indices).long()
    return (
        bench.backing,
        bench.old_x[:, rows],
        bench.old_B[:, rows],
        bench.old_dt[:, rows],
        bench.pnat[rows],
        bench.buf[rows],
    )


def _rs_spec(heads, path, n, t):
    h, hv = heads
    return dict(
        state_dtype=torch.bfloat16,
        H=h,
        HV=hv,
        path=path,
        n=n,
        t=t,
        layers=2,
        steps=HIST // t + 3,
    )


#: Replay-slot runs: requests and the work-item width the kernel is handed (None: no work
#: items, the small path, the kernel commits wrapping rows itself).
_REPLAY_SLOT_RUNS = {
    "small_n1": (1, None),
    "small_n4": (4, None),
    "w4_all_layer": (CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE, 4),
    "w5_all_layer": (32, REPLAY_WORK_ITEM_WIDTH),
}


@pytest.mark.parametrize("heads", cr.GEOMETRIES, ids=[f"{h}x{hv}" for h, hv in cr.GEOMETRIES])
@pytest.mark.parametrize("n,width", list(_REPLAY_SLOT_RUNS.values()), ids=list(_REPLAY_SLOT_RUNS))
def test_separate_replay_slots_match_coupled(n, width, heads, certifying) -> None:  # noqa: F811
    """Replay slots that differ from the checkpoint slots give, bit for bit, the coupled run.

    The coupled side is `_ReplayBench`: history and bookkeeping at the checkpoint slots. The
    other (`_ReplaySlotBench`) keeps the same history and bookkeeping at a permuted set of
    replay slots past the checkpoint pool's, in pages holding every layer's rows (a slot stride
    of the whole page, the bookkeeping as strided int32 fields of it), 0xFF elsewhere. Both run
    verify steps through this entry -- strided a / b and PDL, as the target hands them over --
    with checkpoint wraps committed in-kernel (no work items) or by the manager's all-layer
    commit, the kernel handed the work items at `width` fields (4: the replay slot from
    `replay_indices`; 5: from the items' fifth field), then advance their bookkeeping with the
    same acceptance. Each replay-slot call is also held to this entry's `reference` given the
    same `replay_indices`, and no byte of the pages outside the rows' replay slots is written.
    """
    certifying(_grc_op)
    t = 4
    path = "small" if width is None else "all_layer"
    spec = _rs_spec(heads, path, n, t)
    seed = 31 + n
    coupled = _ReplayBench(spec, seed, exact=False)
    paged = _ReplaySlotBench(spec, seed, width or REPLAY_WORK_ITEM_WIDTH)
    sides = (coupled, paged)
    rows = paged.replay_indices.long()
    assert not torch.isin(rows, coupled.state_indices.long()).any()
    wraps = appends = 0
    for step in range(coupled.steps):
        wrapped = coupled.writes()[1].numel()
        wraps += wrapped
        appends += n - wrapped
        for side in sides:
            side.prepare()
        if width is not None:
            assert torch.equal(coupled.n_writes, paged.n_writes)
        for layer in range(coupled.layers):
            act = coupled.inputs()
            a, b = _grc_gates(coupled, act, True)
            outs = []
            for side in sides:
                kw = dict(side.kwargs(layer, act), g=a, beta=b, launch_with_pdl=True)
                if side is paged:
                    ref = _grc_op.reference(**kw)
                outs.append(_grc_op(**kw))
            pool, u, k, G = paged.state(layer)
            _replay_compare(
                (outs[1], pool, u[rows], k[rows], G[rows]),
                (ref[0], ref[1], ref[2][rows], ref[3][rows], ref[4][rows]),
            )
            assert torch.equal(outs[0], outs[1]), f"step {step} layer {layer}: outputs differ"
        for side in sides:
            side.commit()
        accepted = coupled.draw_acceptance()
        for side in sides:
            side.accept(accepted)
        for name, x, y in zip(_SLOT_STATE, _slot_state(coupled), _slot_state(paged)):
            assert torch.equal(x, y), f"step {step}: {name} differs"
    assert wraps > 0 and appends > 0, f"the run never wrapped or appended ({wraps}, {appends})"
    pages = paged.pages
    elsewhere = torch.ones(pages.data.shape[0], dtype=torch.bool, device=DEV)
    elsewhere[rows] = False
    assert (pages.data[elsewhere] == 0xFF).all(), "a page outside the rows' replay slots moved"
    assert (pages.data[:, pages.tail :] == 0xFF).all(), "a page's bytes past its rows moved"


#: Context rows ahead of the generation rows in the dispatch test's mixed-step metadata.
_DISPATCH_PREFILLS = 3


def _dispatch_core(bench, separate):
    """What `_verify_state` and `_gdn_verify` read, over `bench`'s pool, history and
    bookkeeping: the GDN core's geometry and bind (the pools, the replay buffers, a manager
    with the all-layer commit on and, with `separate`, replay slots of its own), and the mamba
    metadata of a mixed step whose generation rows follow `_DISPATCH_PREFILLS` context rows
    (`bench.prepare` fills the work items)."""
    replay = _m._Replay(
        {
            layer: (
                bench.old_x[layer],
                bench.old_B[layer],
                bench.old_dt[layer],
                bench.old_dA[layer],
            )
            for layer in range(bench.layers)
        },
        bench.buf,
        bench.pnat,
        HIST,
        bench.t,
    )
    cm = types.SimpleNamespace(
        use_gdn_cached_replay_all_layer_commit=True,
        replay_layer_group_id=1 if separate else None,
    )
    core = types.SimpleNamespace(
        conv_dim=bench.width,
        key_dim=bench.h * K,
        value_dim=bench.hv * V,
        num_k_heads=bench.h,
        head_k_dim=K,
        num_v_heads=bench.hv,
        head_v_dim=V,
        _bound=types.SimpleNamespace(
            pools=types.SimpleNamespace(ssm=dict(enumerate(bench.pool))),
            replay=replay,
            cm=cm,
            int_idx=torch.arange(bench.n, dtype=torch.int32, device=DEV),
        ),
    )
    context = torch.full((_DISPATCH_PREFILLS,), -1, dtype=torch.int32, device=DEV)
    rows = bench.replay_indices if separate else bench.state_indices
    mm = types.SimpleNamespace(
        replay_work_items=bench.work_items,
        replay_n_writes=bench.n_writes,
        replay_state_indices=torch.cat([context, rows]),
    )
    return core, mm


@pytest.mark.parametrize("heads", cr.GEOMETRIES, ids=[f"{h}x{hv}" for h, hv in cr.GEOMETRIES])
@pytest.mark.parametrize("separate", [False, True], ids=["coupled", "replay_slots"])
@pytest.mark.parametrize("n,t", [(1, 4), (4, 4), (32, 4), (5, 3)])
def test_target_dispatch_matches_the_plain_call(n, t, separate, heads) -> None:
    """The target's verify dispatch over a bf16 pool (`_verify_state` for a mixed step's
    generation rows, then `_gdn_verify` per layer: `a` / `b` column views of the input
    projection, the conv output as `packed_qkv`, the launch `_gdn_replay_launch` picks) against
    the call it replaces (contiguous `a` / `b`, no PDL, the default mapping), each on its own
    copy of the pool, the history and the bookkeeping, over consecutive verify steps with
    random partial acceptance in between: outputs, checkpoints, history and bookkeeping bitwise
    equal at every step, with the entry's guard on. With `separate` the dispatched side's
    manager keeps the history and its bookkeeping at replay slots of its own
    (`_ReplaySlotBench`), which `_verify_state` takes from the metadata's `replay_state_indices`
    behind the context rows and `_gdn_verify` hands this entry as `replay_indices`. From
    `CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE` requests both sides take the work items and the
    manager's all-layer commit; below it the kernel commits wraps."""
    path = "all_layer" if n >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE else "small"
    spec = _rs_spec(heads, path, n, t)
    seed = 77 + n
    plain = _ReplayBench(spec, seed, exact=False)
    disp = (
        _ReplaySlotBench(spec, seed, REPLAY_WORK_ITEM_WIDTH)
        if separate
        else _ReplayBench(spec, seed, exact=False)
    )
    core, mm = _dispatch_core(disp, separate)
    wraps = 0
    with validating(_grc_op):
        for step in range(plain.steps):
            wraps += plain.writes()[1].numel()
            plain.prepare()
            disp.prepare()
            verify = _m._verify_state(core, mm, n, _DISPATCH_PREFILLS)
            assert verify[3] == (path == "all_layer"), (
                "the all-layer commit exactly when partitioned"
            )
            if separate:
                assert torch.equal(verify[4], disp.replay_indices), "the rows' replay slots"
            else:
                assert verify[4] is None, "no replay slots of its own: the checkpoint slots"
            for layer in range(plain.layers):
                act = plain.inputs()
                a, b = _grc_gates(plain, act, True)
                want = _grc_op(**plain.kwargs(layer, act))
                gating = types.SimpleNamespace(a_log=disp.a_log[layer], dt_bias=disp.dt_bias[layer])
                got = _m._gdn_verify(
                    core,
                    None,
                    a.reshape(n * t, -1),
                    b.reshape(n * t, -1),
                    layer,
                    gating,
                    disp.state_indices,
                    n,
                    t,
                    verify,
                    mq_out=act["packed"],
                )
                assert torch.equal(got.view_as(want), want), f"step {step} layer {layer}"
            plain.commit()
            disp.commit()
            accepted = plain.draw_acceptance()
            plain.accept(accepted)
            disp.accept(accepted)
            for name, x, y in zip(_SLOT_STATE, _slot_state(disp), _slot_state(plain)):
                assert torch.equal(x, y), f"step {step}: {name} differs"
    assert wraps > 0, "no history wrapped: no checkpoint commit was exercised"


# ------------------------------------------- causal_conv1d_token_major__verify ---


def _ctm_build(rows, p, seed):
    """A tep4 rank's verify conv operands: the q|k|v column view of a [rows * p, 4128]
    projection, one layer's view of a 36-layer conv page, the snapshot buffer."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    bf = torch.bfloat16
    conv_dim, width, slots = 2560, 4, rows + 5
    proj = torch.randn(rows * p, 4128, generator=g, device=DEV).to(bf)
    page = torch.randn(slots, 36, conv_dim, width - 1, generator=g, device=DEV).to(bf)
    weight = (torch.randn(conv_dim, width, generator=g, device=DEV) * 0.3).to(bf)
    window = torch.zeros(rows + 2, p, conv_dim, width - 1, device=DEV, dtype=bf)
    slot_ids = torch.arange(2, rows + 2, dtype=torch.int32, device=DEV)
    window_rows = torch.arange(rows, dtype=torch.int32, device=DEV)
    return proj, page, weight, window, slot_ids, window_rows


def _check_causal_conv1d_token_major(cell) -> None:
    rows, p, wait = cell.spec["rows"], cell.spec["p"], cell.spec.get("wait", False)
    proj, page, weight, window, slot_ids, window_rows = _ctm_build(rows, p, _seed(cell.why))
    page_ref, window_ref = page.clone(), window.clone()
    x = proj[:, :2560]
    causal_conv1d_token_major.is_valid(x, rows, page[:, 7], weight, slot_ids, window, window_rows)
    out = causal_conv1d_token_major(
        x, rows, page[:, 7], weight, slot_ids, window, window_rows, -1, True, wait
    )
    ref = causal_conv1d_token_major.reference(
        x, rows, page_ref[:, 7], weight, slot_ids, window_ref, window_rows, -1
    )
    torch.cuda.synchronize()
    assert out.is_contiguous() and out.shape == (rows * p, 2560)
    assert torch.equal(out, ref), "output"
    assert torch.equal(page, page_ref), "the advanced pool"
    assert torch.equal(window, window_ref), "the snapshots"


def _variant_conv_seqs(rows, p, monkeypatch) -> None:
    """The verify conv's multi-sequence kernel (`_AUTO_SEQS` sequences a program, a tep2 rank's
    large batches) bitwise the one-sequence launch, with a pad slot and a ragged sequence count."""
    seqs = ctm._AUTO_SEQS
    g = torch.Generator(device=DEV).manual_seed(rows * 31 + p * 7 + seqs)
    bf = torch.bfloat16
    conv_dim, slots_n = 5120, rows + 7
    proj = torch.randn(rows * p, 8240, generator=g, device=DEV).to(bf)
    page = torch.randn(slots_n, 12, conv_dim, 3, generator=g, device=DEV).to(bf)
    weight = (torch.randn(conv_dim, 4, generator=g, device=DEV) * 0.3).to(bf)
    window = torch.randn(rows + 3, p, conv_dim, 3, generator=g, device=DEV).to(bf)
    slot_ids = torch.randperm(slots_n, generator=g, device=DEV)[:rows].to(torch.int32)
    slot_ids[rows // 2] = -1  # a padded sequence: nothing of it is read or written
    window_rows = torch.randperm(rows + 3, generator=g, device=DEV)[:rows].to(torch.int32)
    x = proj[:, :conv_dim]

    def run(n_seqs, pg, win):
        monkeypatch.setattr(ctm, "_SEQS", n_seqs)
        return ctm.causal_conv1d_token_major(
            x, rows, pg[:, 5], weight, slot_ids, win, window_rows, -1, True, _m._GDN_CONV_PDL_WAIT
        )

    # a padded sequence's output rows are never written (torch.empty): compare the live ones
    keep = torch.repeat_interleave(slot_ids != -1, p)
    page_ref, win_ref = page.clone(), window.clone()
    ref = run(1, page_ref, win_ref)
    page_new, win_new = page.clone(), window.clone()
    out = run(seqs, page_new, win_new)
    torch.cuda.synchronize()
    assert torch.equal(out[keep], ref[keep]), "output"
    assert torch.equal(page_new, page_ref), "the advanced pool"
    assert torch.equal(win_new, win_ref), "the snapshots"


# ------------------------------------------------- gdn_in_proj_conv__verify / norm_out_proj ---


def _ipc_operands(n: int, p: int, seed: int):
    """A tep4 verify step's rows, input projection, conv taps, strided conv pool and snapshots."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows = n * p
    x = (0.5 * torch.randn(rows, 2560, device=DEV, generator=g)).to(torch.bfloat16)
    w = (0.03 * torch.randn(4128, 2560, device=DEV, generator=g)).to(torch.bfloat16)
    cw = (0.4 * torch.randn(2560, 4, device=DEV, generator=g)).to(torch.bfloat16)
    page = torch.randn(9, 3, 2560, 3, device=DEV, generator=g).to(torch.bfloat16)
    slots = torch.tensor([7 - 2 * i for i in range(n)], dtype=torch.int32, device=DEV)
    win = torch.randn(8, p, 2560, 3, device=DEV, generator=g).to(torch.bfloat16)
    wrows = torch.tensor([i + 1 for i in range(n)], dtype=torch.int32, device=DEV)
    return x, w, cw, page, slots, win, wrows


def _check_gdn_in_proj_conv(cell) -> None:
    """Projection, conv, snapshots and pool bitwise the two-launch chain and within the entry's
    fp32 reference; a captured launch replays on new rows."""
    n, p = cell.spec["requests"], cell.spec["p"]
    x, w, cw, page, slots, win, wrows = _ipc_operands(n, p, zlib.crc32(cell.why.encode()))
    tac = fused_tactic(n * p)
    assert tac is not None

    def two_launch(x, page, win):
        proj = low_m_bf16_mm(x, w.t(), True)
        conv = causal_conv1d_token_major(
            proj[:, :2560], n, page[:, 1], cw, slots, win, wrows, -1, True, True
        )
        return proj, conv

    page_ref, win_ref = page.clone(), win.clone()
    proj_ref, conv_ref = two_launch(x, page_ref, win_ref)
    page_new, win_new = page.clone(), win.clone()
    proj, conv = gdn_in_proj_conv(
        x, w.t(), cw, page_new[:, 1], slots, win_new, wrows, -1, p, tac, True
    )
    torch.cuda.synchronize()
    for got, want, what in (
        (proj, proj_ref, "projection"),
        (conv, conv_ref, "conv"),
        (win_new, win_ref, "snapshots"),
        (page_new, page_ref, "pool page"),
    ):
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), what
    # The entry's own fp32 reference: the projection within bf16 rounding of a 2560-term
    # sum, the conv within a bf16 ulp of the activations it reads.
    rp, rc = gdn_in_proj_conv.reference(
        x, w.t(), cw, page.clone()[:, 1], slots, win.clone(), wrows, -1, p, tac, True
    )
    torch.testing.assert_close(proj.float(), rp.float(), atol=0.06, rtol=0.02)
    torch.testing.assert_close(conv.float(), rc.float(), atol=0.06, rtol=0.02)
    if not cell.spec["capture"]:
        return
    sx = x.clone()
    graph = torch.cuda.CUDAGraph()
    page_g, win_g = page.clone(), win.clone()
    with torch.cuda.graph(graph):
        gp, gc = gdn_in_proj_conv(
            sx, w.t(), cw, page_g[:, 1], slots, win_g, wrows, -1, p, tac, True
        )
    x2 = (0.5 * torch.randn_like(x.float())).to(torch.bfloat16)
    sx.copy_(x2)
    page_g.copy_(page)
    win_g.copy_(win)
    graph.replay()
    torch.cuda.synchronize()
    page_r, win_r = page.clone(), win.clone()
    rp, rc = two_launch(x2, page_r, win_r)
    assert torch.equal(gp, rp) and torch.equal(gc, rc) and torch.equal(page_g, page_r)


# ------------------------------------------------------------------------------ the tests ---

#: Every Gated-DeltaNet entry of the target with its checker: `CASES` is each (entry, cell).
_ENTRIES = (
    (flashinfer_chunk_gated_delta_rule, _check_flashinfer_chunk_gated_delta_rule),
    (_grc_op, _check_gdn_cached_replay_cute),
    (causal_conv1d_token_major, _check_causal_conv1d_token_major),
    (gdn_in_proj_conv, _check_gdn_in_proj_conv),
)
CASES = [(op, check, cell) for op, check in _ENTRIES for cell in op.CELLS]


@pytest.mark.parametrize(
    "entry,check,cell",
    CASES,
    ids=[f"{type(op).__name__}-{i}" for op, _ in _ENTRIES for i in range(len(op.CELLS))],
)
def test_certified_cells(entry, check, cell, certifying) -> None:  # noqa: F811 — fixture by name
    """Every certified cell of every entry against the entry's own reference."""
    certifying(entry)
    check(cell)


@pytest.mark.parametrize("t", range(2, _m._MAX_VERIFY_WIDTH + 1), ids=lambda t: f"T{t}")
def test_gdn_cached_replay_every_verify_width_is_the_recurrence(t) -> None:
    """The cached replay at every verify width over consecutive steps from warm histories with
    random acceptance, against its reference and the exact twin."""
    spec = _grc_rank_spec(
        "tep4", path="small", n=2, t=t, steps=6, layers=1, slots=24, strided_gates=True
    )
    cell = types.SimpleNamespace(why=f"tep4 small n=2 T={t}", spec=spec)
    bench = _ReplayBench(spec, seed=_seed(cell.why))
    try:
        _grc_drive(cell, bench)
    finally:
        bench.close()


#: `(id, helper, args)`: a launch the cells do not reach, against its plain launch.
_VARIANTS = (
    [("replay_graph-tep4-n1-pdl", _variant_replay_graph, (1, "small", "tep4", True))]
    + [("replay_graph-dep4-n16", _variant_replay_graph, (16, "all_layer", "dep4", False))]
    + [(f"ht_verify-n{n}", _variant_ht_verify, (n,)) for n in (16, 24, 32, 64, 128)]
    + [(f"conv_seqs-{r}x{p}", _variant_conv_seqs, (r, p)) for r, p in ((37, 4), (128, 3))]
)


@pytest.mark.parametrize("helper,args", [v[1:] for v in _VARIANTS], ids=[v[0] for v in _VARIANTS])
def test_launch_variants_are_bitwise_the_plain_launch(helper, args, monkeypatch) -> None:
    """Launches no cell reaches, each bitwise its plain counterpart."""
    if helper is _variant_conv_seqs:
        helper(*args, monkeypatch)
    else:
        helper(*args)
