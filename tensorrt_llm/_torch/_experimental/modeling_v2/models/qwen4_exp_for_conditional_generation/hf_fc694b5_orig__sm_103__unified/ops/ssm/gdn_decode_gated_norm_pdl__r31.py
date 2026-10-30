# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One Gated-DeltaNet decode step per row, and the gated RMS norm on its output, in
one PDL launch.

Tier 5 (CuTe DSL written for this target; the recurrence is the gated delta rule of
FlashInfer's `gdn_decode.gated_delta_rule_decode_pretranspose`, reimplemented here). At one
decode row the step reads and writes each v-head's 64 KB fp32 state once and
otherwise does a few hundred flops per state row; the rest of its time is latency.
The state is not produced by anything in the step ahead of this kernel -- the
pool row is this layer's own and was last written by this kernel one step ago, and
the slot index is filled before the step's graph is launched -- so it is read
before `griddepcontrol.wait`, while the input projection ahead of this kernel
streams its weight.

A v-head is split over a cluster of `CLUSTER` CTAs of `WARPS` warps; each warp owns
`V / (CLUSTER * WARPS)` value rows of the head's `[V, K]` state and lane `l` the
`K / 32` key columns from `l * K / 32` of each. Small CTAs fit beside the input
projection's last wave, so their state reads start before it ends. The gated RMS
norm (per v-head, 128 wide) needs the head's whole output: every warp writes the
sum of squares of its rows into each CTA of the cluster through distributed shared
memory, and every CTA then normalizes and gates its own rows.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ..norm.rms_norm_gated_token_major__r30 import rms_norm_gated_token_major as _gated_norm

#: Decode rows per launch this entry is certified for.
MAX_ROWS = 4
#: CTAs per v-head (one cluster) and warps per CTA; their product must be <= 32.
CLUSTER = 8
WARPS = 4


@dsl_user_op
def _st_shared_cluster_f32(ptr, val, *, loc=None, ip=None):
    """`st.shared::cluster.f32` to a pointer from `cute.arch.map_dsmem_ptr`."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(ptr.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
            cutlass.Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared::cluster.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _GdnKernel:
    def __init__(self, *, k_heads, v_heads, kdim, vdim, pdl):
        assert vdim % (CLUSTER * WARPS) == 0 and CLUSTER * WARPS <= 32
        assert kdim % 32 == 0 and v_heads % k_heads == 0
        self.k_heads, self.v_heads, self.kdim, self.vdim, self.pdl = (
            k_heads,
            v_heads,
            kdim,
            vdim,
            pdl,
        )
        self.rows_per_cta = vdim // CLUSTER
        self.rows_per_warp = vdim // (CLUSTER * WARPS)
        self.per_lane = kdim // 32
        self.k_off = k_heads * kdim
        self.v_off = 2 * k_heads * kdim
        self.z_off = self.v_off + v_heads * vdim
        self.a_off = self.z_off + v_heads * vdim
        self.b_off = self.a_off + v_heads

    @cute.jit
    def __call__(
        self,
        mProj,
        mPool,
        mSlot,
        mALog,
        mDt,
        mW,
        mOut,
        eps: cutlass.Float32,
        stream: cuda_drv.CUstream,
    ):
        rows = cute.size(mProj, mode=[0])
        ld_state = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.Float32,
            num_bits_per_copy=32 * self.per_lane,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        st_state = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32 * self.per_lane
        )
        ld_act = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.BFloat16,
            num_bits_per_copy=16 * self.per_lane,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        self.kernel(
            mProj, mPool, mSlot, mALog, mDt, mW, mOut, eps, ld_state, st_state, ld_act
        ).launch(
            grid=[self.v_heads * CLUSTER, rows, 1],
            block=[WARPS * 32, 1, 1],
            cluster=[CLUSTER, 1, 1],
            smem=32 * 4 + 16,
            stream=stream,
            use_pdl=self.pdl,
            # <= 64 registers a thread: the CTAs fit beside the producer's last wave.
            min_blocks_per_mp=max(1, 8 * 4 // WARPS),
        )

    @cute.kernel
    def kernel(
        self,
        mProj,
        mPool,
        mSlot,
        mALog,
        mDt,
        mW,
        mOut,
        eps: cutlass.Float32,
        ld_state: cute.CopyAtom,
        st_state: cute.CopyAtom,
        ld_act: cute.CopyAtom,
    ):
        bidx, n, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        rpw: cutlass.Constexpr = self.rows_per_warp
        pl: cutlass.Constexpr = self.per_lane
        kdim: cutlass.Constexpr = self.kdim
        vdim: cutlass.Constexpr = self.vdim
        parts: cutlass.Constexpr = CLUSTER * WARPS
        hv = bidx // CLUSTER
        part = bidx % CLUSTER  # the CTA's rank in its cluster: clusters tile the grid's x
        kh = hv // (self.v_heads // self.k_heads)
        v0 = part * self.rows_per_cta + warp * rpw
        k0 = lane * pl
        ss_parts = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Float32, cute.make_layout((32,), stride=(1,)), byte_alignment=16
        )
        if const_expr(CLUSTER > 1):
            # Every CTA of the cluster must have started before another writes into its
            # shared memory; this arrival is waited on just before those writes.
            cute.arch.cluster_arrive_relaxed()

        # 1. Static and step-invariant operands, before the wait: the gates' constants,
        # the norm weight of the warp's rows, the slot and the warp's state rows.
        a_log_h = mALog[hv]
        dtb = mDt[hv]
        wn = cute.make_rmem_tensor(cute.make_layout((rpw,), stride=(1,)), cutlass.Float32)
        for r in cutlass.range_constexpr(rpw):
            wn[r] = mW[v0 + r].to(cutlass.Float32)
        slot = mSlot[n]
        # 64-bit: the cache manager coalesces every GDN layer's state of a slot into one
        # page, so a layer's slot stride is ~28M elements and slot * stride overflows int32.
        slot64 = cutlass.Int64(slot)
        # (slot, hv, v, (pl, lane)): lane l's 16-byte vector of a state row.
        pool_vec = cute.logical_divide(mPool, (None, None, None, pl))
        h = cute.make_rmem_tensor(cute.make_layout((rpw, pl), stride=(pl, 1)), cutlass.Float32)
        h.fill(0.0)
        if slot >= 0:
            for r in cutlass.range_constexpr(rpw):
                cute.copy(ld_state, pool_vec[slot64, hv, v0 + r, (None, lane)], h[r, None])
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        # 2. The projection's q, k, v, z, a, b for this row and head.
        # (n, (pl, cols / pl)): 8-byte vectors of the row; every offset below is a multiple of pl.
        proj_vec = cute.logical_divide(mProj, (None, pl))
        raw = cute.make_rmem_tensor(cute.make_layout((2, pl), stride=(pl, 1)), cutlass.BFloat16)
        cute.copy(ld_act, proj_vec[n, (None, (kh * kdim + k0) // pl)], raw[0, None])
        cute.copy(ld_act, proj_vec[n, (None, (self.k_off + kh * kdim + k0) // pl)], raw[1, None])
        qn = cute.make_rmem_tensor(cute.make_layout((pl,), stride=(1,)), cutlass.Float32)
        kn = cute.make_rmem_tensor(cute.make_layout((pl,), stride=(1,)), cutlass.Float32)
        zv = cute.make_rmem_tensor(cute.make_layout((rpw,), stride=(1,)), cutlass.Float32)
        vv = cute.make_rmem_tensor(cute.make_layout((rpw,), stride=(1,)), cutlass.Float32)
        for e in cutlass.range_constexpr(pl):
            qn[e] = raw[0, e].to(cutlass.Float32)
            kn[e] = raw[1, e].to(cutlass.Float32)
        for r in cutlass.range_constexpr(rpw):
            vv[r] = mProj[n, self.v_off + hv * vdim + v0 + r].to(cutlass.Float32)
            zv[r] = mProj[n, self.z_off + hv * vdim + v0 + r].to(cutlass.Float32)
        a = mProj[n, self.a_off + hv].to(cutlass.Float32)
        b = mProj[n, self.b_off + hv].to(cutlass.Float32)

        # q, k L2 norm (eps 1e-6 inside the root), the 1/sqrt(K) scale on q, the gates.
        sq = cutlass.Float32(0.0)
        sk = cutlass.Float32(0.0)
        for e in cutlass.range_constexpr(pl):
            sq = qn[e] * qn[e] + sq
            sk = kn[e] * kn[e] + sk
        sq = cute.arch.warp_reduction_sum(sq)
        sk = cute.arch.warp_reduction_sum(sk)
        rq = cute.math.rsqrt(sq + 1e-6) * (1.0 / cute.math.sqrt(cutlass.Float32(kdim)))
        rk = cute.math.rsqrt(sk + 1e-6)
        for e in cutlass.range_constexpr(pl):
            qn[e] = qn[e] * rq
            kn[e] = kn[e] * rk
        x = a + dtb
        sp = x
        if x <= 20.0:
            sp = cute.math.log1p(cute.math.exp(x))
        decay = cute.math.exp(-cute.math.exp(a_log_h) * sp)
        beta = 1.0 / (1.0 + cute.math.exp(-b))

        # 3. The delta rule, one value row at a time; the new rows go straight back.
        ov = cute.make_rmem_tensor(cute.make_layout((rpw,), stride=(1,)), cutlass.Float32)
        ssw = cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(rpw):
            hk = cutlass.Float32(0.0)
            for e in cutlass.range_constexpr(pl):
                hk = h[r, e] * kn[e] + hk
            hk = cute.arch.warp_reduction_sum(hk)
            u = beta * (vv[r] - decay * hk)
            o = cutlass.Float32(0.0)
            for e in cutlass.range_constexpr(pl):
                h[r, e] = decay * h[r, e] + u * kn[e]
                o = h[r, e] * qn[e] + o
            o = cute.arch.warp_reduction_sum(o)
            if slot >= 0:
                cute.copy(st_state, h[r, None], pool_vec[slot64, hv, v0 + r, (None, lane)])
            else:
                o = cutlass.Float32(0.0)
            # The recurrence's output is a bf16 tensor in the unfused composition.
            ov[r] = o.to(cutlass.BFloat16).to(cutlass.Float32)
            ssw = ov[r] * ov[r] + ssw

        # 4. The head's sum of squares: warp `w` of CTA `c` writes its rows' share into
        # slot `c * WARPS + w` of every CTA of the cluster (lane `i` into CTA `i`).
        if const_expr(CLUSTER > 1):
            cute.arch.cluster_wait()
            if lane < CLUSTER:
                _st_shared_cluster_f32(
                    cute.arch.map_dsmem_ptr(ss_parts.iterator + (part * WARPS + warp), lane), ssw
                )
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            if lane == 0:
                ss_parts[warp] = ssw
            cute.arch.sync_threads()
        ss = cutlass.Float32(0.0)
        if lane < parts:
            ss = ss_parts[lane]
        ss = cute.arch.warp_reduction_sum(ss)
        rstd = cute.math.rsqrt(ss / vdim + eps)

        # 5. Gated RMS norm of the warp's rows: lane `r` writes row `r`.
        o_r = ov[0]
        z_r = zv[0]
        w_r = wn[0]
        for r in cutlass.range_constexpr(1, rpw):
            o_r = cutlass.Float32(cutlass.select_(lane == r, ov[r], o_r))
            z_r = cutlass.Float32(cutlass.select_(lane == r, zv[r], z_r))
            w_r = cutlass.Float32(cutlass.select_(lane == r, wn[r], w_r))
        y = o_r * rstd * w_r * (1.0 / (1.0 + cute.math.exp(-z_r)))
        if lane < rpw:
            mOut[n, hv * vdim + v0 + lane] = y.to(cutlass.BFloat16)


def _dl(t: torch.Tensor, align: int = 4):
    return from_dlpack(t, assumed_align=align)


_COMPILED: dict = {}


def _operands(proj, pool, slots, A_log, dt_bias, norm_weight, out, eps, stream):
    return (
        _dl(proj, 8),
        _dl(pool, 16),
        _dl(slots),
        _dl(A_log),
        _dl(dt_bias),
        _dl(norm_weight, 2),
        _dl(out, 2),
        cutlass.Float32(eps),
        stream,
    )


def launch(proj, pool, slots, A_log, dt_bias, norm_weight, eps, k_heads, kdim, pdl=True):
    """The launch behind `GdnDecodeGatedNormPdl.__call__`; updates `pool` in place and
    returns the gated-normed output `[rows, HV * V]` bf16."""
    rows = proj.shape[0]
    v_heads, vdim = pool.shape[1], pool.shape[2]
    out = torch.empty((rows, v_heads * vdim), dtype=torch.bfloat16, device=proj.device)
    if rows == 0:
        return out
    pdl = bool(pdl and get_env_enable_pdl())
    stream = cuda_drv.CUstream(torch.cuda.current_stream(proj.device).cuda_stream)
    operands = (proj, pool, slots, A_log, dt_bias, norm_weight, out, eps, stream)
    key = (
        rows,
        k_heads,
        v_heads,
        kdim,
        vdim,
        proj.stride(0),
        tuple(pool.shape),
        pool.stride(),
        pdl,
    )
    fn = _COMPILED.get(key)
    if fn is None:
        # Compiled from the call's own operands (a stand-in for the pool would span every
        # layer of the coalesced state cache).
        kern = _GdnKernel(k_heads=k_heads, v_heads=v_heads, kdim=kdim, vdim=vdim, pdl=pdl)
        fn = _COMPILED[key] = cute_ext.compile(kern, *_operands(*operands))
    fn(*_operands(*operands))
    return out


def _recurrence_reference(q, k, v, A_log, a, dt_bias, b, pool, slots):
    """`(o, pool_after)` of one gated-delta-rule decode step per row in fp64, rounded once:
    `o` to bf16, the state to the pool's dtype. Row `n` reads and writes `pool[slots[n]]`
    (a negative slot skips the row: zero output, no state access); q and k are
    L2-normalized, the decay is `exp(-exp(A_log) * softplus(a + dt_bias))`, `beta =
    sigmoid(b)`, the scale `K^-0.5`. `q` / `k` `[n, 1, H, K]`, `v` `[n, 1, HV, V]`, `a` /
    `b` `[n, 1, HV]`, `pool` `[slots, HV, V, K]`. Mutates nothing."""
    dt = torch.float64
    batch, _, num_k_heads, k_dim = q.shape
    hv = v.shape[2]
    rep = hv // num_k_heads
    scale = k_dim**-0.5
    reads = slots.tolist()
    pool_after = pool.clone()
    o = torch.zeros(batch, 1, hv, v.shape[3], dtype=dt, device=q.device)
    neg_a = -torch.exp(A_log.to(dt))
    for n in range(batch):
        if reads[n] < 0:
            continue
        q_n = q[n, 0].to(dt).repeat_interleave(rep, dim=0)
        k_n = k[n, 0].to(dt).repeat_interleave(rep, dim=0)
        q_n = q_n * torch.rsqrt(q_n.pow(2).sum(-1, keepdim=True) + 1e-6)
        k_n = k_n * torch.rsqrt(k_n.pow(2).sum(-1, keepdim=True) + 1e-6)
        x = a[n, 0].to(dt) + dt_bias.to(dt)
        softplus = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
        beta = torch.sigmoid(b[n, 0].to(dt))
        h = pool[reads[n]].to(dt) * torch.exp(neg_a * softplus)[:, None, None]
        u = (v[n, 0].to(dt) - torch.einsum("hvk,hk->hv", h, k_n)) * beta[:, None]
        h = h + u[:, :, None] * k_n[:, None, :]
        o[n, 0] = torch.einsum("hvk,hk->hv", h, q_n * scale)
        pool_after[reads[n]] = h.to(pool.dtype)
    return o.to(torch.bfloat16), pool_after


class GdnDecodeGatedNormPdl(OpWrapper):
    """A single-token Gated-DeltaNet step per row over the fp32 slot pool, then the
    sigmoid-gated RMS norm of its output per v-head. With `proj` the decode input
    projection's output row `[q (H*K) | k (H*K) | v (HV*V) | z (HV*V) | a (HV) | b (HV)]`
    (after the conv step: q, k, v are the conv's SiLU outputs)::

        o, pool = gated_delta_rule_decode_pretranspose(q, k, v, A_log, a, dt_bias, b,
                                                       use_qk_l2norm=True, pool, slots)
        out     = rms_norm_gated_token_major(o (bf16), z, norm_weight, eps, "sigmoid")

    Returns `out` `[rows, HV * V]` bf16 and updates `pool[slots[n]]` in place; a
    negative slot skips the row's state (no read or write) and its output is zero.
    `o` is rounded to bf16 between the two, as the composition stores it.

    Fusion boundary: both gates, the q/k L2 norm, the recurrence, the indexed state
    read and write, the gated RMS norm. The projection and the conv are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows; pool (default a contiguous 461-slot pool | "coalesced": the
    # cache manager's pool, slot stride 36 layers' states, slots 100 and 101);
    # pad (row 1 given slot -1).
    CELLS: tuple[Cell, ...] = (
        Cell(why="qwen3.8 decode, 1 row (con 1), 461-slot pool", spec=dict(rows=1)),
        Cell(why="qwen3.8 decode, 4 rows (MAX_ROWS), scattered slots", spec=dict(rows=4)),
        Cell(
            why="2 rows, coalesced pool (slot stride 36 layers, slot 100), row 1 a pad slot (-1)",
            spec=dict(rows=2, pool="coalesced", pad=True),
        ),
    )

    note = """
    Metadata consumed: `slots` (the cache manager's state indices for the step).

    Preconditions: bf16 `proj` `[rows, 2*H*K + 2*HV*V + 2*HV]` with unit column stride;
    the pool fp32 `[slots, HV, V, K]` with K innermost (any outer strides); `A_log`,
    `dt_bias` fp32 `[HV]`; `norm_weight` bf16 `[V]`; `V % 32 == 0` (rows a warp:
    `V / 32`), `K % 32 == 0`; int32 `slots` `[rows]`. **The pool rows and `slots` must not be
    written by any kernel of the same step queued ahead of this one**: both are read
    before `griddepcontrol.wait`. That holds for this target's decode steps (each
    layer's pool is written only by its own recurrence, `slots` is filled before the
    step's graph launches) and is why the entry is not for verify or prefill steps.

    Launch: `HV * 8` CTAs a row, clusters of 8 (one v-head), 128 threads, <= 64
    registers. Numerics: fp32 inside; `o` rounded to bf16 before the norm, the output
    rounded once. Inside torch's default band for bf16 / fp32 of the fp64 composition.
    Deterministic. Graph-capturable after the first call per geometry (compiled then).

    Certified: 1 and 4 rows over a contiguous 461-slot pool, and 2 rows over the cache
    manager's coalesced pool (slot stride 36 layers' states, slot 100 ~2.8G elements past
    the base) with the second row a pad slot (-1). Under graph replay the op reads the
    projection after `griddepcontrol.wait` (1 row). Not certified: 3 rows, `pdl=False`.
    """

    def __call__(
        self,
        proj: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        k_heads: int,
        kdim: int,
        pdl: bool = True,
    ) -> torch.Tensor:
        return launch(proj, pool, slots, A_log, dt_bias, norm_weight, eps, k_heads, kdim, pdl)

    def reference(
        self,
        proj: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        k_heads: int,
        kdim: int,
        pdl: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, pool_after)`: the fp64 recurrence (`_recurrence_reference`; o rounded to
        bf16, the pool to fp32), then the gated norm entry's fp32 reference."""
        rows = proj.shape[0]
        v_heads, vdim = pool.shape[1], pool.shape[2]
        q, k, v, z, a, b = torch.split(
            proj,
            [k_heads * kdim, k_heads * kdim, v_heads * vdim, v_heads * vdim, v_heads, v_heads],
            1,
        )
        o, pool_after = _recurrence_reference(
            q.reshape(rows, 1, k_heads, kdim),
            k.reshape(rows, 1, k_heads, kdim),
            v.reshape(rows, 1, v_heads, vdim),
            A_log,
            a.reshape(rows, 1, v_heads),
            dt_bias,
            b.reshape(rows, 1, v_heads),
            pool,
            slots,
        )
        out = _gated_norm.reference(
            o.reshape(-1, vdim), z.reshape(rows, v_heads, vdim), norm_weight, eps, None, "sigmoid"
        )
        return out.reshape(rows, v_heads * vdim), pool_after

    def is_valid(
        self,
        proj: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        k_heads: int,
        kdim: int,
        pdl: bool = True,
    ) -> None:
        rows = proj.shape[0]
        v_heads, vdim, kd = pool.shape[1], pool.shape[2], pool.shape[3]
        assert rows <= MAX_ROWS, f"at most {MAX_ROWS} rows"
        assert proj.dtype == torch.bfloat16 and proj.stride(1) == 1, "bf16 proj, unit column stride"
        assert proj.shape[1] == 2 * k_heads * kdim + 2 * v_heads * vdim + 2 * v_heads, "proj width"
        assert pool.dtype == torch.float32 and pool.stride(3) == 1 and kd == kdim, (
            "fp32 pool [S, HV, V, K]"
        )
        assert vdim % (CLUSTER * WARPS) == 0 and kdim % 32 == 0, "V % (CLUSTER * WARPS), K % 32"
        assert v_heads % k_heads == 0, "HV a multiple of H"
        assert slots.dtype == torch.int32 and slots.shape == (rows,), "int32 slots [rows]"
        assert A_log.dtype == torch.float32 and dt_bias.dtype == torch.float32, (
            "fp32 gate constants"
        )
        assert norm_weight.dtype == torch.bfloat16 and norm_weight.shape == (vdim,), (
            "bf16 norm weight [V]"
        )


gdn_decode_gated_norm_pdl = GdnDecodeGatedNormPdl()
