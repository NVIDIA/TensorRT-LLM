# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_recurrent_gated_delta_rule_cached_replay_update entry.

The op is stateful across verify steps: what one call leaves in the checkpoint
pool and the history double buffer is what the next call reads, after the
engine's own bookkeeping ran in between. So every cell drives several
consecutive verify steps the way the engine does -- work items before the
forward (partitioned batches), the op once per layer, then the all-layer
commit and `_advance_replay_state` with a random partial acceptance -- and
checks two things at every step:

* each call against the entry's `reference` on that call's own inputs (the
  output and all four state buffers), plus bitwise locks on everything the
  call must not touch;
* every output against an *exact twin*: the entry's `reference` driven on the
  same checkpoint and history held in fp32 and deep enough never to wrap,
  advanced by `p += accepted` alone. The twin's state is the plain
  token-by-token recurrence over every accepted token, independent of the
  wrap, the double buffer and the commit, so a wrong commit or wrong
  bookkeeping surfaces as an output that drifts off it.

A cell with `page_layers` runs over the stock manager's coalesced state page at an
engine's slot count (36 layers' states a slot): 26.5 GB at tep4's 1874 bf16 slots, so a
reserved virtual range with memory mapped under the touched slots only (`_SparsePage`).
The reference and the twin run on a dense copy of those slots, and every byte of their
pages outside the cell's sub-pages is checked unchanged.
"""

from __future__ import annotations

import zlib

import cuda.bindings.driver as cuda_drv
import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.fused_recurrent_gated_delta_rule_cached_replay_update import (  # noqa: E501
    ALL_LAYER_COMMIT_MIN_BATCH,
    HEAD_K_DIM,
    HEAD_V_DIM,
    HISTORY_SIZE,
    NUM_K_HEADS,
    NUM_V_HEADS,
    PACKED_QKV_WIDTH,
    RANK_V_HEADS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.fused_recurrent_gated_delta_rule_cached_replay_update import (  # noqa: E501
    fused_recurrent_gated_delta_rule_cached_replay_update as op,
)
from tensorrt_llm._torch.modules.fla.cache_manager import (
    ReplayStateUpdateMetadata,
    _advance_replay_state,
)
from tensorrt_llm._torch.modules.fla.cached_replay import (
    CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
    commit_gdn_cached_replay_history_layers,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import _build_replay_work_items_triton

__extra_import_path__ = [".."]
from _validating import ULP, certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), (
    "fused_recurrent_gated_delta_rule_cached_replay_update requires a CUDA device"
)

DEV = "cuda"
H, HV, K, V = NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM
W = PACKED_QKV_WIDTH
HIST = HISTORY_SIZE
BF16_ULP = ULP[torch.bfloat16]

#: This checkpoint's 36 GDN layers of `linear_attn.A_log` / `linear_attn.dt_bias`
#: (bf16, 48 each): mean, standard deviation, range.
A_LOG_STATS = (1.006, 1.487, (-7.125, 5.0625))
DT_BIAS_STATS = (-4.411, 1.732, (-11.4375, 2.53125))

#: Element gate of the bf16 members, in bf16 ulps of the block they belong to.
_ELEMENT_ULPS = 4.0
#: Relative-RMS gate of the output, in bf16 ulps.
_OUTPUT_RMS_ULPS = 1.5
#: Element gate of an fp32 checkpoint state, relative to its (slot, head) scale.
_STATE_TOL = 2.0**-13
#: Element gate of a bf16 checkpoint state, in bf16 ulps of its (slot, head) scale.
_BF16_STATE_ULPS = 2.0
#: Gate of the stored cumulative log-decays: |dG| <= atol + rtol * |G|.
_G_ATOL, _G_RTOL = 2.0**-11, 2.0**-18

#: The largest deviation each gated member reached in the running cell, in units of
#: its gate's measure (printed per cell, so the margins are on record).
_MEASURED: dict[str, float] = {}


def _record(key: str, value: float) -> None:
    _MEASURED[key] = max(_MEASURED.get(key, 0.0), value)


def _gate(name, got, want, element, rms=None):
    """`got` against `want` per block of the last two dims (a token's
    `[heads, dim]`, or one head's `[V, K]` state): every element within
    `element` of its block's absmax, NaN blocks of `want` excluded."""
    block = got.shape[-2] * got.shape[-1]
    g, w = got.float().reshape(-1, block), want.float().reshape(-1, block)
    keep = ~torch.isnan(w).any(1)
    g, w = g[keep], w[keep]
    scale = w.abs().amax(1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    ratio = ((g - w).abs() / scale).nan_to_num(nan=float("inf"))
    worst = ratio.max().item() if ratio.numel() else 0.0
    _record(f"{name} element/gate", worst / element)
    assert worst <= element, (
        f"{name}: an element is {worst:.3e} of its block's absmax from the reference "
        f"(gate {element:.3e})"
    )
    if rms is not None and w.numel():
        rel = ((g - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt().clamp_min(1e-30)).item()
        _record(f"{name} rms/gate", rel / rms)
        assert rel <= rms, f"{name}: relative RMS {rel:.3e} exceeds {rms:.3e}"


def _compare(out: tuple, ref: tuple) -> None:
    """`(returned, ssm_states, old_u, old_k, old_G)` as the call left them,
    against `reference`'s five; a member `None` on both sides is skipped (the
    exact twin is compared on the output alone).

    The kernel is not fp32 throughout, and the bands say where. It rounds the
    L2-normalized q and k to bf16 before every product (2^-9 relative each),
    rounds the history-mixing coefficients to bf16 for its tensor-core dots, and
    stores the output and the history's u and k in bf16. So the output, `old_u`
    and `old_k` are gated per block -- one token's `[heads, dim]` -- at 4 bf16
    ulps of the block's absmax, which keeps a near-silent head from turning
    rounding into relative noise, and the output's relative RMS at 1.5 ulps. An fp32
    checkpoint state is fp32 by construction: the commit splits the decayed
    history update into a bf16 hi and lo part, ~16 significant bits per term, and
    up to 16 terms each within 2^-17 give the 2^-13 gate, relative to one (slot,
    head) state's absmax. A bf16 checkpoint is that fp32 commit rounded once, as the
    reference's is, so the two stored values are equal or adjacent, and adjacent bf16
    values are at most 2 bf16 ulps of their head's absmax apart: the bf16 gate is one
    representable step. `G` only ever enters as exp(G_t - G_j), so an absolute
    error is a relative decay error: the kernel's softplus runs on approximate
    transcendentals whose fp32 sum drops exp(x)'s low bits; atol 2^-11 is an
    eighth of a bf16 ulp of the u it scales, and rtol 2^-18 covers the cumulative
    sum's rounding where |G| reaches thousands on fast-forgetting heads.
    """
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
            bound = _G_ATOL + _G_RTOL * want.abs()
            _record("old_G |dG|/gate", ((got - want).abs() / bound).max().item())
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
    """Per-layer A_log / dt_bias drawn around the checkpoint's own statistics;
    head 0 pinned to the corner that forgets everything every token
    (g ~ -400), head 1 to the one that hardly decays (g ~ -1e-8)."""
    (a_mean, a_std, a_range), (d_mean, d_std, d_range) = A_LOG_STATS, DT_BIAS_STATS
    a_log = torch.randn(layers, hv, generator=gen, device=DEV) * a_std + a_mean
    dt_bias = torch.randn(layers, hv, generator=gen, device=DEV) * d_std + d_mean
    a_log, dt_bias = a_log.clamp(*a_range), dt_bias.clamp(*d_range)
    a_log[:, 0], dt_bias[:, 0] = a_range[1], d_range[1]
    a_log[:, 1], dt_bias[:, 1] = a_range[0], d_range[0]
    return a_log.contiguous(), dt_bias.contiguous()


def _unit(x):
    return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)


def _scattered(n: int, slots: int) -> list[int]:
    """`n` distinct slots of a `slots`-slot pool: both ends first, then a stride-37 walk
    over the interior (37 is coprime to `slots - 2` for every pool here)."""
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
    """A reserved virtual range of `elems` elements of `dtype` with physical memory mapped
    under `regions` (element ranges) only. Views of it are torch tensors over that range;
    an access to an unmapped address faults, so a stray write is loud."""

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


class _EnginePage:
    """The stock manager's coalesced state page at `slots` slots: a slot's page holds
    `page_layers` layers' `[HV, V, K]` states, and the bench's `layers` are its top
    sub-pages, the page's highest addresses. Memory is mapped under each touched slot's
    page plus the next slot's first sub-page (where a write past the page would land),
    and under slot 0's page, which holds the views' base."""

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


class _Bench:
    """The state the cache manager keeps for `layers` GDN layers, laid out as
    `MambaHybridCacheManagerV2` lays it out, driven the way the engine drives
    it: work items before the forward (`_build_replay_work_items_triton`, what
    mamba metadata's prepare runs for a partitioned batch), the op per layer,
    then `update_mamba_states`'s order -- the all-layer commit, then
    `_advance_replay_state`.

    With `exact=True` it also carries the exact twin (see the module docstring).
    Over an engine page (`page_layers`) the reference and the twin see a dense copy
    of the touched slots: `idx` the slots in order, `position` a slot's row in it.
    """

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
        elif spec.get("edge_slots"):
            inner = torch.randperm(S - 2, generator=self.cpu)[: n - 2] + 1
            chosen = torch.cat([torch.tensor([S - 1]), inner, torch.tensor([0])])
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
            self.page = _EnginePage(S, page_layers, L, chosen.tolist(), hv, state_dtype, self.gen)
            self.pool = self.page.pools
            self.layer_stride, self.slot_stride = self.page.state, self.page.stride
            self.idx = torch.tensor(self.page.used, dtype=torch.long, device=DEV)
            self.position = torch.full((S,), -1, dtype=torch.long, device=DEV)
            self.position[self.idx] = torch.arange(self.idx.numel(), device=DEV)
        else:
            # [slots, layers, HV, V, K]: each layer's state is a slot-strided view,
            # the affine layout whose strides the all-layer commit is handed.
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
            self.work_items = torch.zeros(n, 4, dtype=torch.int32, device=DEV)
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
        """A per-slot tensor as the reference sees it: whole over a dense pool, the touched
        slots' rows (a copy, in `idx` order) over an engine page."""
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
        """One layer's fresh activations: post-conv `[q|k|v]` rows and the raw
        `a`/`b` projections, as the verify step hands them over."""
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

    def reference(self, kw):
        """The entry's `reference` on a call's arguments, every per-slot buffer through
        `compact`: the touched slots' rows, the rows' slots renumbered into them."""
        if self.idx is None:
            return op.reference(**kw)
        per_slot = (
            "ssm_states",
            "old_u",
            "old_k",
            "old_G",
            "old_beta",
            "cache_buf_idx",
            "prev_num_accepted_tokens",
        )
        compact = {name: self.compact(kw[name]) for name in per_slot}
        return op.reference(
            **dict(kw, **compact, state_indices=self.compact_indices(kw["state_indices"]))
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

    def unwritten(self, wrapped):
        """Rows of `observed`'s pool the call must leave as they were: all of them on the
        partitioned path, all but the wrapping slots' when the call commits in-kernel."""
        rows = self.slots if self.idx is None else self.idx.numel()
        keep = torch.ones(rows, dtype=torch.bool, device=DEV)
        if self.path == "small":
            keep[self.compact_indices(wrapped).long()] = False
        return keep

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
        """Where this step's calls append, by the documented rule, which slots wrap, and how
        many append exactly up to the history's end: `[slots, 2, HIST]` mask, wrapping slot
        ids, the `pnat + t == HIST` count."""
        s = self.live.long()
        p, b = self.pnat[s].long(), self.buf[s].long()
        wrap = p + self.t > HIST
        half, start = torch.where(wrap, 1 - b, b), torch.where(wrap, 0, p)
        mask = torch.zeros(self.slots, 2, HIST, dtype=torch.bool, device=DEV)
        for t in range(self.t):
            mask[s, half, start + t] = True
        return mask, s[wrap], int((p + self.t == HIST).sum())


def _check_duplicate_padding_slot(cell) -> None:
    """The last rows of a batch share one slot, as CUDA-graph padding repeats the manager's
    dummy slot: the rows on unique slots, and every slot but the shared one, are held to
    the reference run on those rows alone -- the padding rows change nothing about them.
    The shared slot's state and the padding rows' outputs are unspecified."""
    bench = _Bench(cell.spec, seed=_seed(cell.why), exact=False)
    dup = cell.spec["dup"]
    unique = bench.n - len(dup)
    assert dup == tuple(range(unique, bench.n)), "the padding rows trail the batch"
    shared = int(bench.state_indices[unique])
    assert (bench.state_indices[unique:] == shared).all()
    assert shared not in bench.live.tolist()
    others = torch.ones(bench.slots, dtype=torch.bool, device=DEV)
    others[shared] = False
    for _ in range(bench.steps):
        for layer in range(bench.layers):
            act = bench.inputs()
            kw = bench.kwargs(layer, act)
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
            ref = op.reference(**alone)
            got = op(**kw)
            pool, u, k, G = bench.state(layer)
            _compare(
                (got[:unique], pool[others], u[others], k[others], G[others]),
                (ref[0], *(x[others] for x in ref[1:])),
            )
        bench.accept(bench.draw_acceptance())


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    _MEASURED.clear()
    if "dup" in cell.spec:
        return _check_duplicate_padding_slot(cell)
    bench = _Bench(cell.spec, seed=_seed(cell.why))
    try:
        _drive_cell(cell, bench)
    finally:
        bench.close()
    print(f"\n[{cell.why}] " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(_MEASURED.items())))


def _drive_cell(cell, bench) -> None:
    wraps = appends = to_the_end = 0
    for _ in range(bench.steps):
        mask, wrapped, at_end = bench.writes()
        wraps += wrapped.numel()
        to_the_end += at_end
        appends += bench.live.numel() - wrapped.numel()
        bench.prepare()
        after_commit = []
        for layer in range(bench.layers):
            act = bench.inputs()
            kw = bench.kwargs(layer, act)
            _, u, k, G = bench.state(layer)
            before = dict(
                pool=bench.observed(layer)[0].clone(),
                u=u.clone(),
                k=k.clone(),
                G=G.clone(),
                old_beta=bench.old_dA[layer].clone(),
            )
            ref = bench.reference(kw)
            exact = op.reference(**bench.exact_kwargs(layer, kw))
            committed = None
            if bench.path == "all_layer":
                committed = bench.reference(
                    dict(kw, replay_work_items=None, n_writes=None, use_all_layer_commit=False)
                )
            got = op(**kw)
            _compare((got, *bench.observed(layer)), ref)
            # The locks: in place, and nothing outside the documented writes.
            _, u, k, G = bench.state(layer)
            assert torch.equal(bench.old_dA[layer], before["old_beta"]), "old_beta is never written"
            assert torch.equal(u[~mask], before["u"][~mask]), "old_u written outside the append"
            assert torch.equal(k[~mask], before["k"][~mask]), "old_k written outside the append"
            Gt, Gt0 = G.transpose(-1, -2), before["G"].transpose(-1, -2)
            assert torch.equal(Gt[~mask], Gt0[~mask]), "old_G written outside the append"
            kept = bench.unwritten(wrapped)
            assert torch.equal(bench.observed(layer)[0][kept], before["pool"][kept]), (
                "the pool moved where the call commits nothing (a partitioned call commits none)"
            )
            bench.adopt_exact(layer, exact)
            after_commit.append((got, exact, committed))
        bench.commit()
        for layer, (got, exact, committed) in enumerate(after_commit):
            if committed is not None:
                # The partitioned call plus the manager's commit is the call
                # that commits in-kernel.
                _compare((got, *bench.observed(layer)), committed)
            _compare((got, None, None, None, None), (exact[0], None, None, None, None))
        if bench.page is not None:
            bench.page.assert_rest_unchanged()
        bench.accept(bench.draw_acceptance())
    assert wraps > 0 and appends > 0, (
        f"the cell never wrapped or never appended ({wraps}, {appends})"
    )
    if cell.spec.get("accept") == "full":
        live = bench.live.numel()
        assert wraps >= 3 * live and to_the_end >= 3 * live, (
            f"full acceptance should wrap every row at least 3 times and append up to the "
            f"history's end as often ({wraps}, {to_the_end} over {live} rows)"
        )


def test_the_band_refuses_a_wrong_head_pairing() -> None:
    """The output gate is wider than torch's per-element default near a block's small
    elements, so it has to refuse a plausible wrong answer. At a tp4/tep4 rank's 4 / 12
    heads value head `hv` reads key head `hv // 3`; the same call's reference with the
    key heads reversed (value heads 0..2 on key head 3) must fall outside it, by a margin
    printed with the passing reference's."""
    cell = next(
        c
        for c in op.CELLS
        if c.spec.get("HV") == RANK_V_HEADS and c.spec["n"] == 128 and c.spec["t"] == 8
    )
    bench = _Bench(dict(cell.spec, start="fresh", steps=1), seed=4646, exact=False)
    try:
        kw = bench.kwargs(0, bench.inputs())
        reversed_heads = dict(
            kw, q=kw["q"].flip(2).contiguous(), k=kw["k"].flip(2).contiguous(), packed_qkv=None
        )
        right, wrong = bench.reference(kw)[0], bench.reference(reversed_heads)[0]
        bench.prepare()
        got = op(**kw)
        _MEASURED.clear()
        _compare((got, None, None, None, None), (right, None, None, None, None))
        passing = dict(_MEASURED)
        _MEASURED.clear()
        with pytest.raises(AssertionError, match="output"):
            _compare((got, None, None, None, None), (wrong, None, None, None, None))
        print(f"\n[wrong head pairing] passing {passing}, wrong {dict(_MEASURED)}")
    finally:
        bench.close()


# ── CUDA graphs ───────────────────────────────────────────────────────────


def test_cuda_graph_replay_on_new_data_is_bitwise_eager() -> None:
    """Two layers of 32 sequences x 8 tokens over an fp32 checkpoint captured with the
    all-layer commit after them; replayed three times on new activations, with acceptance
    in between, against the same steps run eagerly on a twin state."""
    n, t = 32, 8
    assert ALL_LAYER_COMMIT_MIN_BATCH == CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
    assert n >= ALL_LAYER_COMMIT_MIN_BATCH
    spec = dict(path="all_layer", n=n, t=t, layers=2, slots=40, state_dtype=torch.float32)
    seed = 7000 + 10 * n + t
    eager, graph, warm = (_Bench(spec, seed, exact=False) for _ in range(3))
    acts = [warm.inputs() for _ in range(2)]
    warm.prepare()
    for layer in range(2):
        op(**warm.kwargs(layer, acts[layer]))
    warm.commit()
    torch.cuda.synchronize()

    graph.prepare()
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        outs_graph = [op(**graph.kwargs(layer, acts[layer])) for layer in range(2)]
        graph.commit()
    assert torch.equal(graph.backing, eager.backing) and torch.equal(graph.old_x, eager.old_x)

    refill = torch.Generator(device=DEV).manual_seed(seed + 1)
    for _ in range(3):
        for act in acts:
            for key in ("packed", "a", "b"):
                act[key].copy_(
                    torch.randn(act[key].shape, generator=refill, device=DEV, dtype=act[key].dtype)
                )
        eager.prepare()
        outs_eager = [op(**eager.kwargs(layer, acts[layer])) for layer in range(2)]
        eager.commit()
        graph.prepare()
        captured.replay()
        torch.cuda.synchronize()
        for e, g in zip(outs_eager, outs_graph):
            assert torch.equal(e, g), "replayed output differs from eager"
        for name in ("backing", "old_x", "old_B", "old_dt", "pnat", "buf"):
            assert torch.equal(getattr(eager, name), getattr(graph, name)), f"{name} differs"
        accepted = eager.draw_acceptance()
        eager.accept(accepted)
        graph.accept(accepted)


# ── guards: accepted and silently wrong ──────────────────────────────────


def test_a_short_state_indices_reads_past_its_end_and_is_refused() -> None:
    """Unguarded, three indices for four rows: the fourth row reads whatever
    follows the view -- here a slot the call never named -- and writes its
    history."""
    bench = _Bench(dict(path="small", n=4, t=2, layers=1), seed=101, exact=False)
    stranger = next(s for s in range(bench.slots) if s not in bench.live.tolist())
    parent = torch.cat(
        [bench.state_indices[:3], torch.tensor([stranger], dtype=torch.int32, device=DEV)]
    )
    kw = dict(bench.kwargs(0, bench.inputs()), state_indices=parent[:3])
    before = bench.old_x[0, stranger].clone()
    op(**kw)
    assert not torch.equal(bench.old_x[0, stranger], before), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="state_indices has shape"):
        with validating(op):
            op(**kw)
