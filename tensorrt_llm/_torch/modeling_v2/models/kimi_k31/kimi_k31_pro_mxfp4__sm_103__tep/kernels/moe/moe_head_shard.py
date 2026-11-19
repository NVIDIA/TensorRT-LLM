# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MoE head sharded over the attention-TP group: router logits and routed latent of every row.

The replicated head reads the whole router ``[896, 7168]`` and latent down ``[3584, 7168]`` weights on every rank.
Here rank ``r`` of ``W`` multiplies the rows by its own weight rows only -- router rows ``[r * 896 / W, ...)`` and
latent rows ``[r * 3584 / W, ...)`` -- and the ranks all-gather the results over the MNNVL Lamport workspace.

Bits. An output element depends only on its row of ``x``, its weight row and the order its K products are summed in;
splitting the weight rows over ranks changes none of these. Two launches:

* :func:`run_head_shard_gemm`: the chunked TGV GEMM (``kernels/gemm/chunked_tgv_gemm``) over the latent rows (BF16
  out) and the router rows (FP32 out), each summing K in the replicated head's chunks;
* :class:`MoeHeadGatherKernel`: each rank multicasts its padded GEMM outputs' valid columns into the workspace's
  current Lamport buffer and polls the other ranks' (the family Lamport all-gather's protocol and rotation state),
  writing the gathered ``logits [M, 896]`` FP32 and ``latent [M, 3584]`` BF16 in place; or
  :class:`MoeHeadRouteQuantKernel`: the same all-gather with the MoE prologue's routing and MXFP8 quantization run
  on the polled slices in the same launch (the routed experts' inputs instead of the logits and the latent).

``-0.0`` is the Lamport sentinel's bit pattern: a logit of ``-0.0`` arrives as ``+0.0``, and a latent pair word of
``0x80000000`` (high BF16 ``-0.0`` with low BF16 ``+0.0``) as two ``+0.0``; the values compare equal.
"""

from __future__ import annotations

import functools
import struct
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Boolean, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm import lamport_ar as _lam
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm.lamport_allgather import (
    sanitize_f32,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.gemm.chunked_tgv_gemm import (
    kind_tiles,
    run_chunked_tgv,
    run_chunked_tgv_split,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.moe_prologue import (
    _QUANT_ROWS,
    MoePrologueKernel,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.route_ptx import _lds_f32
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.route_tail import (
    _WIDE_BIAS_WORDS,
    NUM_EXPERTS,
    TOP_K,
    WEIGHTS_RUNNER,
    _ld_nc_v4_f32,
    _lds_v4_f32,
    _mxfp8_quant_regs,
    _rcp_approx_ftz,
    _st_global_b16,
    _st_global_b32,
    _sts_v4_f32,
)

ROUTER_ROWS = 896
LATENT_ROWS = 3584
#: Threads of a gather CTA; one thread owns one 16-byte vector of a gathered row.
GATHER_THREADS = 128
#: ``routed_scaling_factor`` 1.0 as the prologue's FP64 bits (the runner's routing weights take unit scale only).
_UNIT_SCALE_BITS = struct.unpack("<q", struct.pack("<d", 1.0))[0]


@dsl_user_op
def _add_f32_words(a, b, *, loc=None, ip=None):
    """The FP32 sum (round to nearest even) of two FP32 bit patterns, as a bit pattern."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "{\n.reg .f32 fa, fb;\nmov.b32 fa, $1;\nmov.b32 fb, $2;\nadd.rn.f32 fa, fa, fb;\nmov.b32 $0, fa;\n}",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_of_f32_words(lo, hi, *, loc=None, ip=None):
    """Two FP32 bit patterns rounded to BF16 (nearest even) and packed, ``lo`` in the low half."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(lo).ir_value(loc=loc, ip=ip), Int32(hi).ir_value(loc=loc, ip=ip)],
            "{\n.reg .f32 fl, fh;\nmov.b32 fl, $1;\nmov.b32 fh, $2;\ncvt.rn.bf16x2.f32 $0, fh, fl;\n}",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_v4_b32(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    """16-byte shared-memory store of four words."""
    llvm.inline_asm(
        None,
        [Int32(v).ir_value(loc=loc, ip=ip) for v in (addr, w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _any_pending(words, valid):
    """Whether any vector ``k`` with ``valid[k]`` (its words ``words[4 k : 4 k + 4]``) has not arrived."""
    pending = Boolean(False)
    for k in range(len(valid)):
        pending = pending | (valid[k] & (_lam.clean4(*words[4 * k : 4 * k + 4]) == 0))
    return pending


def head_planes(
    rows: int, w_latent: torch.Tensor, w_router: torch.Tensor, latent_bounds, router_bounds, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Empty GEMM outputs ``(latent, router)`` as planes ``[planes, rows, kind_tiles * 64]``. Where the latent splits K
    (its chunks gate the launch), one plane per K chunk of FP32 partials (one CTA per chunk); else one plane per kind
    with the final dtype (BF16 latent, FP32 router; one CTA per weight tile accumulating every chunk)."""
    split = bool(latent_bounds)
    latent = torch.empty(
        1 + len(latent_bounds) if split else 1,
        rows,
        kind_tiles(w_latent.shape[0]) * 64,
        dtype=torch.float32 if split else torch.bfloat16,
        device=device,
    )
    router = torch.empty(
        1 + len(router_bounds) if split else 1,
        rows,
        kind_tiles(w_router.shape[0]) * 64,
        dtype=torch.float32,
        device=device,
    )
    return latent, router


def run_head_shard_gemm(
    x: torch.Tensor,
    w_latent: torch.Tensor,
    w_router: torch.Tensor,
    out_latent: torch.Tensor,
    out_router: torch.Tensor,
    config: Tuple[int, int, Tuple[int, ...], Tuple[int, ...]],
    pdl: bool,
) -> None:
    """The latent and router rows' products in one launch into the planes of :func:`head_planes`; ``config`` =
    (cta_n, A/B stages, latent K-tile bounds, router K-tile bounds)."""
    cta_n, stages, latent_bounds, router_bounds = config
    if out_latent.shape[0] > 1:
        run_chunked_tgv_split(
            x,
            ((w_latent, out_latent, latent_bounds), (w_router, out_router, router_bounds)),
            cta_n,
            stages,
            pdl,
        )
    else:
        run_chunked_tgv(
            x,
            ((w_latent, out_latent[0], latent_bounds), (w_router, out_router[0], router_bounds)),
            cta_n,
            stages,
            pdl,
        )


class MoeHeadGatherKernel:
    """All-gather of every rank's head slice: ``logits [m, 896]`` FP32 and ``latent [m, 3584]`` BF16.

    A rank's slice is ``896 / W`` FP32 router words and ``3584 / W`` BF16 latent values (``3584 / (2 W)`` words) per
    row, read from the rank's padded GEMM outputs (row strides ``router_pad`` / ``latent_pad`` elements). Gathered row
    ``t`` of the Lamport buffer holds the ranks' slices in rank order. Grid ``m * cols + 1`` CTAs of 128 threads:
    worker CTA ``b`` owns row ``b // cols`` and its vectors ``(b % cols) * 128 + tid``; the last CTA advances the
    rotation state once every worker has read it (the family Lamport all-gather's protocol).
    """

    def __init__(
        self,
        world: int,
        router_pad: int,
        latent_pad: int,
        use_pdl: bool,
        router_planes: int = 1,
        latent_planes: int = 1,
    ):
        self.W = world
        # GEMM outputs in planes (one per K chunk, FP32 partials) are added in order here; one plane is the final
        # value (FP32 router, BF16 latent)
        self.router_planes = router_planes
        self.latent_planes = latent_planes
        self.rw = ROUTER_ROWS // world  # router words per rank
        self.lw = LATENT_ROWS // world  # latent values per rank
        if self.rw % 4 or self.lw % 8:
            raise ValueError(f"world {world}: slices are not whole 16-byte vectors")
        self.rv = self.rw // 4  # router vectors per rank
        self.own_vecs = self.rv + self.lw // 8
        self.vecs = world * self.own_vecs
        self.cols = (self.vecs + GATHER_THREADS - 1) // GATHER_THREADS
        self.router_pad = router_pad
        self.latent_pad = latent_pad
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        router_in: Int64,
        latent_in: Int64,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        logits: Int64,
        latent: Int64,
        m: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(router_in, latent_in, uc_local, mc_base, flags, logits, latent, m, rank).launch(
            grid=(m * self.cols + 1, 1, 1),
            block=(GATHER_THREADS, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def own_vector(self, router_in: Int64, latent_in: Int64, tok: Int32, j: Int32, m: Int32):
        """This rank's vector ``j`` of row ``tok`` (4 FP32 router words, or 8 BF16 latent values as 4 words): the
        planes added in order in FP32, the latent rounded once to BF16."""
        out0 = Int32(0)
        out1 = Int32(0)
        out2 = Int32(0)
        out3 = Int32(0)
        if j < self.rv:
            plane = Int64(m) * self.router_pad * 4
            src = router_in + (Int64(tok) * self.router_pad + Int64(j * 4)) * 4
            w = _lam.ld_v4(src)
            acc = [w[0], w[1], w[2], w[3]]
            for z in cutlass.range_constexpr(1, self.router_planes):
                p = _lam.ld_v4(src + plane * z)
                acc = [_add_f32_words(acc[e], p[e]) for e in range(4)]
            out0, out1, out2, out3 = acc[0], acc[1], acc[2], acc[3]
        else:
            v = j - self.rv
            if cutlass.const_expr(self.latent_planes == 1):
                w = _lam.ld_v4(latent_in + (Int64(tok) * self.latent_pad + Int64(v * 8)) * 2)
                out0, out1, out2, out3 = w[0], w[1], w[2], w[3]
            else:
                plane = Int64(m) * self.latent_pad * 4
                src = latent_in + (Int64(tok) * self.latent_pad + Int64(v * 8)) * 4
                a = _lam.ld_v4(src)
                b = _lam.ld_v4(src + Int64(16))
                acc = [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
                for z in cutlass.range_constexpr(1, self.latent_planes):
                    pa = _lam.ld_v4(src + plane * z)
                    pb = _lam.ld_v4(src + plane * z + Int64(16))
                    nxt = [pa[0], pa[1], pa[2], pa[3], pb[0], pb[1], pb[2], pb[3]]
                    acc = [_add_f32_words(acc[e], nxt[e]) for e in range(8)]
                out0 = _bf16x2_of_f32_words(acc[0], acc[1])
                out1 = _bf16x2_of_f32_words(acc[2], acc[3])
                out2 = _bf16x2_of_f32_words(acc[4], acc[5])
                out3 = _bf16x2_of_f32_words(acc[6], acc[7])
        return [out0, out1, out2, out3]

    @cute.kernel
    def kernel(
        self,
        router_in: Int64,
        latent_in: Int64,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        logits: Int64,
        latent: Int64,
        m: Int32,
        rank: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        workers = m * self.cols
        arrivals = _lam.arrivals_addr(flags)
        row_bytes = self.vecs * 16
        # The GEMM outputs and the rotation state are the previous grids' (the previous call's included).
        if cutlass.const_expr(self.use_pdl):
            _lam.griddepcontrol_wait()
        if bidx == workers:
            if cutlass.const_expr(self.use_pdl):
                _lam.griddepcontrol_launch_dependents()
            if tidx == 0:
                cur, _, nbytes, _, _, _, _, _ = _lam.lamport_state(flags)
                arrived = _lam.ld_acquire_u32(arrivals)
                while arrived < workers:
                    arrived = _lam.ld_acquire_u32(arrivals)
                _lam.st_v4(flags, (cur + 1) % _lam.NUM_BUFFERS, cur, nbytes, Int32(1))
                _lam.st_v4(flags + Int64(16), m * row_bytes, Int32(0), Int32(0), Int32(0))
                _lam.st_u32(arrivals, Int32(0))
        else:
            tok = bidx // self.cols
            by = bidx - tok * self.cols
            v = by * GATHER_THREADS + tidx
            cur, dirty, nbytes, dstages, c0, c1, c2, c3 = _lam.lamport_state(flags)
            if cutlass.const_expr(self.use_pdl):
                _lam.griddepcontrol_launch_dependents()
            cute.arch.barrier()
            if tidx == 0:
                _lam.red_release_add(arrivals, Int32(1))
            if v < self.vecs:
                off = Int64(tok) * row_bytes + Int64(v) * 16
                q = v // self.own_vecs
                j = v - q * self.own_vecs
                # this vector's place in the outputs: router words or latent values of rank q
                dst = logits + (Int64(tok) * ROUTER_ROWS + Int64(q * self.rw + j * 4)) * 4
                if j >= self.rv:
                    dst = (
                        latent
                        + (Int64(tok) * LATENT_ROWS + Int64(q * self.lw + (j - self.rv) * 8)) * 2
                    )
                if q == rank:
                    w = self.own_vector(router_in, latent_in, tok, j, m)
                    s = [sanitize_f32(w[e]) for e in range(4)]
                    _lam.multimem_st_v4(
                        _lam.stage_addr(mc_base, cur, Int32(0), 1, nbytes) + off,
                        s[0],
                        s[1],
                        s[2],
                        s[3],
                    )
                    _lam.st_v4(dst, s[0], s[1], s[2], s[3])
                else:
                    w = _lam.poll_v4(_lam.stage_addr(uc_local, cur, Int32(0), 1, nbytes) + off)
                    _lam.st_v4(dst, w[0], w[1], w[2], w[3])
            # Re-arm the previous call's buffer (every stage it used) while the rows travel.
            _lam.clear_dirty_stages(
                uc_local,
                dirty,
                nbytes,
                dstages,
                c0,
                c1,
                c2,
                c3,
                bidx * GATHER_THREADS + tidx,
                workers * GATHER_THREADS,
            )


class MoeHeadRouteQuantKernel(MoePrologueKernel):
    """The head all-gather and the MoE prologue in one launch: ``topk_ids [m, 16]``, the runner's ``topk_weights
    [m, 16]`` BF16, ``x_fp8 [m, 3584]`` and ``x_sf [m, 112]`` of every row, straight from the Lamport buffer (no
    gathered ``logits`` / ``latent``).

    The rows, slices and protocol of :class:`MoeHeadGatherKernel` (each rank multicasts its slices, the last CTA
    advances the rotation state once every worker has read it) and the routing and quantization code of the MoE
    prologue (:class:`MoePrologueKernel`, the runner's weights), so the outputs are the bits of the gather followed
    by ``k31_moe_prologue(..., runner_weights=True)``. Workers: route CTA ``c < ceil(m / 8)`` multicasts this rank's
    router vectors of tokens ``[8 c, 8 c + 8)``, polls the tokens' 896 logits into shared memory and routes them, one
    warp per token (``select_token``, reading the logits there through :meth:`load_logits` and, on its exact top-k
    fallback, :meth:`load_logits4`); quantization CTA ``c`` multicasts this rank's latent vectors of its 2 rows, then
    one thread per 32-element MXFP8 block polls the block's 4 vectors (one rank's slice) and quantizes them in
    registers. A rank's own vectors come back through the multicast like the others'.
    """

    def __init__(
        self,
        world: int,
        router_pad: int,
        latent_pad: int,
        use_pdl: bool,
        router_planes: int = 1,
        latent_planes: int = 1,
    ):
        super().__init__(LATENT_ROWS, WEIGHTS_RUNNER, use_pdl, _QUANT_ROWS, "all")
        self.g = MoeHeadGatherKernel(
            world, router_pad, latent_pad, use_pdl, router_planes, latent_planes
        )
        if self.g.lw % 32:
            raise ValueError(f"world {world}: a rank's latent slice is not whole MXFP8 blocks")
        self.route_vecs = ROUTER_ROWS // 4  # 16-byte vectors of a token's logits
        self.latent_vecs = self.g.own_vecs - self.g.rv  # 16-byte vectors of a rank's latent slice
        # rounds of this rank's vectors a thread multicasts: a route CTA's tokens' router vectors, a quantization
        # CTA's rows' latent vectors
        self.push_rounds_router = -(-self.tokens * self.g.rv // self.threads)
        self.push_rounds_latent = -(-self.quant_rows * self.latent_vecs // self.threads)
        self.route_polls = -(-self.tokens * self.route_vecs // self.threads)
        self.row_blocks = LATENT_ROWS // 32  # MXFP8 blocks of a latent row
        if self.quant_rows * self.row_blocks > self.threads:
            raise ValueError("a quantization CTA holds one MXFP8 block per thread")

    def load_logits(self, row: Int64, g: Int32, stride: int, n: int):
        """Logit ``s * stride + g`` (``s < n``) of a token from the CTA's shared-memory stage: ``row`` is the stage
        address of the token's logits (``select_token``'s ``logits_addr`` is the stage, offset by the CTA's first
        token)."""
        base = Int32(row)
        return [_lds_f32(base + (s * stride + g) * 4) for s in range(n)]

    def load_logits4(self, row: Int64, lane: Int32, n: int):
        """Logits ``128 j + 4 lane .. + 3`` (``j < n``) of a token from the CTA's shared-memory stage, for the exact
        top-k fallback a token takes when its candidates overflow (``row`` as in :meth:`load_logits`)."""
        base = Int32(row)
        logit = []
        for blk in range(n):
            logit.extend(_lds_v4_f32(base + (blk * 128 + lane * 4) * 4))
        return logit

    @cute.jit
    def __call__(
        self,
        router_in: Int64,
        latent_in: Int64,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        xq_addr: Int64,
        xsf_addr: Int64,
        m: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        scratch_words = self.topk_words
        stage_words = self.tokens * ROUTER_ROWS

        @cute.struct
        class SharedStorage:
            scratch: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, scratch_words], 16]
            stage: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, stage_words], 16]

        self.shared_storage = SharedStorage
        route_ctas = (m + (self.tokens - 1)) // self.tokens
        quant_ctas = (m + (self.quant_rows - 1)) // self.quant_rows
        self.kernel(
            router_in,
            latent_in,
            uc_local,
            mc_base,
            flags,
            bias_addr,
            ids_addr,
            vals_addr,
            xq_addr,
            xsf_addr,
            m,
            rank,
            route_ctas,
        ).launch(
            grid=(route_ctas + quant_ctas + 1, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def push_own(
        self,
        router_in: Int64,
        latent_in: Int64,
        mc: Int64,
        m: Int32,
        rank: Int32,
        first: Int32,
        rows: Int32,
        router: cutlass.Constexpr[bool],
        tidx: Int32,
    ):
        """Multicast this rank's router (``router``) or latent vectors of rows ``[first, first + rows)`` (the GEMM
        planes summed, ``-0.0`` words as ``+0.0``): every vector of the thread read first, then stored."""
        per_row = self.g.rv if cutlass.const_expr(router) else self.latent_vecs
        first_j = 0 if cutlass.const_expr(router) else self.g.rv
        rounds = self.push_rounds_router if cutlass.const_expr(router) else self.push_rounds_latent
        n = rows * per_row
        vals = []
        for k in cutlass.range_constexpr(rounds):
            i = tidx + k * self.threads
            ii = Int32(cutlass.select_(i < n, i, Int32(0)))
            r = ii // per_row
            j = first_j + ii - r * per_row
            w = self.g.own_vector(router_in, latent_in, first + r, j, m)
            off = Int64(first + r) * (self.g.vecs * 16) + Int64(rank * self.g.own_vecs + j) * 16
            vals.append((i < n, off, w))
        for k in cutlass.range_constexpr(rounds):
            ok, off, w = vals[k]
            if ok:
                _lam.multimem_st_v4(
                    mc + off,
                    sanitize_f32(w[0]),
                    sanitize_f32(w[1]),
                    sanitize_f32(w[2]),
                    sanitize_f32(w[3]),
                )

    @cute.jit
    def poll_logits(
        self, uc: Int64, t_begin: Int32, n: Int32, k: cutlass.Constexpr[int], tidx: Int32
    ):
        """``(first load, address, valid, stage slot)`` of the CTA's ``k``-th logit vector of thread ``tidx``."""
        i = tidx + k * self.threads
        valid = i < n
        ii = Int32(cutlass.select_(valid, i, Int32(0)))
        tl = ii // self.route_vecs
        jv = ii - tl * self.route_vecs
        q = (jv * 4) // self.g.rw
        addr = (
            uc
            + Int64(t_begin + tl) * (self.g.vecs * 16)
            + Int64(q * self.g.own_vecs + jv - q * self.g.rv) * 16
        )
        return _lam.ld_volatile_v4(addr), addr, valid, tl * ROUTER_ROWS + jv * 4

    @cute.jit
    def gather_logits(self, uc: Int64, t_begin: Int32, t_end: Int32, stage: Int32, tidx: Int32):
        """Poll the logits of tokens ``[t_begin, t_end)`` into the shared-memory stage ``[token - t_begin][896]``:
        every vector of the thread loaded at once, all of them again until each has arrived (one round trip per
        sweep, not one per vector)."""
        n = (t_end - t_begin) * self.route_vecs
        addrs = []
        valid = []
        slots = []
        words = []
        for k in cutlass.range_constexpr(self.route_polls):
            w, addr, ok, slot = self.poll_logits(uc, t_begin, n, k, tidx)
            addrs.append(addr)
            valid.append(ok)
            slots.append(slot)
            words.extend([w[0], w[1], w[2], w[3]])
        pending = _any_pending(words, valid)
        while pending:
            words = []
            for k in cutlass.range_constexpr(self.route_polls):
                v = _lam.ld_volatile_v4(addrs[k])
                words.extend([v[0], v[1], v[2], v[3]])
            pending = _any_pending(words, valid)
        for k in cutlass.range_constexpr(self.route_polls):
            if valid[k]:
                _sts_v4_b32(
                    stage + slots[k] * 4,
                    words[4 * k],
                    words[4 * k + 1],
                    words[4 * k + 2],
                    words[4 * k + 3],
                )

    @cute.jit
    def quant_block(
        self,
        uc: Int64,
        r_begin: Int32,
        r_end: Int32,
        xq_addr: Int64,
        xsf_addr: Int64,
        rcp448: cutlass.Float32,
        tidx: Int32,
    ):
        """MXFP8 block ``tidx % 112`` of row ``r_begin + tidx // 112``: its 4 vectors polled together (all loaded
        again until each has arrived), quantized in registers."""
        if tidx < (r_end - r_begin) * self.row_blocks:
            rl = tidx // self.row_blocks
            blk = tidx - rl * self.row_blocks
            row = r_begin + rl
            v0 = blk * 32
            q = v0 // self.g.lw
            addr = (
                uc
                + Int64(row) * (self.g.vecs * 16)
                + Int64(q * self.g.own_vecs + self.g.rv + (v0 - q * self.g.lw) // 8) * 16
            )
            words = []
            for k in cutlass.range_constexpr(4):
                v = _lam.ld_volatile_v4(addr + Int64(16 * k))
                words.extend([v[0], v[1], v[2], v[3]])
            pending = _lam.clean4(*words[0:4]) & _lam.clean4(*words[4:8])
            pending = (pending & _lam.clean4(*words[8:12]) & _lam.clean4(*words[12:16])) == 0
            while pending:
                words = []
                for k in cutlass.range_constexpr(4):
                    v = _lam.ld_volatile_v4(addr + Int64(16 * k))
                    words.extend([v[0], v[1], v[2], v[3]])
                pending = _lam.clean4(*words[0:4]) & _lam.clean4(*words[4:8])
                pending = (pending & _lam.clean4(*words[8:12]) & _lam.clean4(*words[12:16])) == 0
            _mxfp8_quant_regs(
                words,
                xq_addr + Int64(row) * Int64(LATENT_ROWS) + Int64(v0),
                xsf_addr + Int64(row) * Int64(self.row_blocks) + Int64(blk),
                rcp448,
            )

    @cute.kernel
    def kernel(
        self,
        router_in: Int64,
        latent_in: Int64,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        xq_addr: Int64,
        xsf_addr: Int64,
        m: Int32,
        rank: Int32,
        route_ctas: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        workers = route_ctas + (m + (self.quant_rows - 1)) // self.quant_rows
        arrivals = _lam.arrivals_addr(flags)
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bias_smem = storage.scratch.data_ptr().toint()
        stage = storage.stage.data_ptr().toint()
        routes = cta < route_ctas
        if routes:
            # The bias is a weight: stage it before waiting on the previous grid.
            for e4 in cutlass.range(tidx, NUM_EXPERTS // 4, self.threads, unroll=1):
                v = _ld_nc_v4_f32(bias_addr + Int64(e4 * 16))
                _sts_v4_f32(bias_smem + e4 * 16, v[0], v[1], v[2], v[3])
        # The GEMM outputs and the rotation state are the previous grids' (the previous call's included).
        if cutlass.const_expr(self.use_pdl):
            _lam.griddepcontrol_wait()
        if cta == workers:
            if cutlass.const_expr(self.use_pdl):
                _lam.griddepcontrol_launch_dependents()
            if tidx == 0:
                cur, _, nbytes, _, _, _, _, _ = _lam.lamport_state(flags)
                arrived = _lam.ld_acquire_u32(arrivals)
                while arrived < workers:
                    arrived = _lam.ld_acquire_u32(arrivals)
                _lam.st_v4(flags, (cur + 1) % _lam.NUM_BUFFERS, cur, nbytes, Int32(1))
                _lam.st_v4(flags + Int64(16), m * (self.g.vecs * 16), Int32(0), Int32(0), Int32(0))
                _lam.st_u32(arrivals, Int32(0))
        else:
            cur, dirty, nbytes, dstages, c0, c1, c2, c3 = _lam.lamport_state(flags)
            if cutlass.const_expr(self.use_pdl):
                _lam.griddepcontrol_launch_dependents()
            cute.arch.barrier()
            if tidx == 0:
                _lam.red_release_add(arrivals, Int32(1))
            uc = _lam.stage_addr(uc_local, cur, Int32(0), 1, nbytes)
            mc = _lam.stage_addr(mc_base, cur, Int32(0), 1, nbytes)
            if routes:
                t_begin = cta * self.tokens
                t_end = t_begin + self.tokens
                if t_end > m:
                    t_end = m
                self.push_own(
                    router_in, latent_in, mc, m, rank, t_begin, t_end - t_begin, True, tidx
                )
                self.gather_logits(uc, t_begin, t_end, stage, tidx)
                cute.arch.barrier()
                cand_all = bias_smem + _WIDE_BIAS_WORDS * 4
                top_all = cand_all + self.cand_words * 4
                team = warp
                cand_smem = cand_all + team * (self.region_slots * 16)
                top_smem = top_all + team * (TOP_K * 8)
                # select_token's logits_addr: the stage, offset by the CTA's first token (see load_logits)
                logits = Int64(stage) - Int64(t_begin) * Int64(NUM_EXPERTS * 4)
                for token in cutlass.range(t_begin + team, t_end, self.teams, unroll=1):
                    expert, weight = self.select_token(
                        token,
                        lane,
                        logits,
                        bias_smem,
                        Int64(_UNIT_SCALE_BITS),
                        cand_smem,
                        top_smem,
                    )
                    if lane < TOP_K:
                        out = token * TOP_K + lane
                        _st_global_b32(ids_addr + Int64(out * 4), expert)
                        _st_global_b16(vals_addr + Int64(out * 2), weight)
            else:
                r_begin = (cta - route_ctas) * self.quant_rows
                r_end = r_begin + self.quant_rows
                if r_end > m:
                    r_end = m
                self.push_own(
                    router_in, latent_in, mc, m, rank, r_begin, r_end - r_begin, False, tidx
                )
                rcp448 = _rcp_approx_ftz(cutlass.Float32(448.0))
                self.quant_block(uc, r_begin, r_end, xq_addr, xsf_addr, rcp448, tidx)
            # Re-arm the previous call's buffer (every stage it used).
            _lam.clear_dirty_stages(
                uc_local,
                dirty,
                nbytes,
                dstages,
                c0,
                c1,
                c2,
                c3,
                cta * self.threads + tidx,
                workers * self.threads,
            )


def gather_bytes(rows: int, world: int) -> int:
    """Bytes of one Lamport buffer a gather of ``rows`` rows needs."""
    return rows * world * (ROUTER_ROWS // world + LATENT_ROWS // (2 * world)) * 4


@functools.cache
def _compiled_gather(
    world: int,
    router_pad: int,
    latent_pad: int,
    use_pdl: bool,
    router_planes: int,
    latent_planes: int,
):
    kernel = MoeHeadGatherKernel(
        world, router_pad, latent_pad, use_pdl, router_planes, latent_planes
    )
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        a,
        a,
        a,
        a,
        a,
        a,
        a,
        Int32(1),
        Int32(0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def _gather_args(out_router: torch.Tensor, out_latent: torch.Tensor) -> Tuple[int, int, int, int]:
    return out_router.shape[2], out_latent.shape[2], out_router.shape[0], out_latent.shape[0]


def compile_head_gather(
    world: int, out_router: torch.Tensor, out_latent: torch.Tensor, use_pdl: bool
) -> None:
    """Compile the gather of a group of ``world`` ranks for these GEMM output planes (before CUDA-graph capture)."""
    rp, lp, rn, ln = _gather_args(out_router, out_latent)
    _compiled_gather(world, rp, lp, bool(use_pdl), rn, ln)


def run_head_gather(
    ws: "_lam.LamportWorkspace",
    out_router: torch.Tensor,
    out_latent: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    use_pdl: bool = True,
) -> None:
    """Gather every rank's head slice into ``logits [M, 896]`` FP32 and ``latent [M, 3584]`` BF16: the GEMM output
    planes ``out_router`` / ``out_latent`` (:func:`head_planes`) added in order, the latent rounded once."""
    m = logits.shape[0]
    need = gather_bytes(m, ws.world)
    if need > ws.bytes_per_buffer:
        raise ValueError(f"Lamport buffer of {ws.bytes_per_buffer} B too small for {m} head rows")
    rp, lp, rn, ln = _gather_args(out_router, out_latent)
    _compiled_gather(ws.world, rp, lp, bool(use_pdl), rn, ln)(
        Int64(out_router.data_ptr()),
        Int64(out_latent.data_ptr()),
        Int64(ws.uc_local),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int64(logits.data_ptr()),
        Int64(latent.data_ptr()),
        Int32(m),
        Int32(ws.rank),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.cache
def _compiled_route_quant(
    world: int,
    router_pad: int,
    latent_pad: int,
    use_pdl: bool,
    router_planes: int,
    latent_planes: int,
):
    kernel = MoeHeadRouteQuantKernel(
        world, router_pad, latent_pad, use_pdl, router_planes, latent_planes
    )
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        *([a] * 10),
        Int32(1),
        Int32(0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def compile_head_route_quant(
    world: int, out_router: torch.Tensor, out_latent: torch.Tensor, use_pdl: bool
) -> None:
    """Compile the fused gather + MoE prologue of a group of ``world`` ranks for these GEMM output planes (before
    CUDA-graph capture)."""
    rp, lp, rn, ln = _gather_args(out_router, out_latent)
    _compiled_route_quant(world, rp, lp, bool(use_pdl), rn, ln)


def run_head_route_quant(
    ws: "_lam.LamportWorkspace",
    out_router: torch.Tensor,
    out_latent: torch.Tensor,
    bias: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    use_pdl: bool = True,
) -> None:
    """Gather every rank's head slice and run the MoE prologue on it, in one launch
    (:class:`MoeHeadRouteQuantKernel`): ``topk_ids [M, 16]`` int32, ``topk_weights [M, 16]`` BF16 (the runner's
    routing weights, unit scale), ``x_fp8 [M, 3584]`` E4M3 and ``x_sf [M, 112]`` UE8M0 from the GEMM output planes
    ``out_router`` / ``out_latent`` (:func:`head_planes`) and the FP32 correction ``bias [896]``."""
    m = topk_ids.shape[0]
    need = gather_bytes(m, ws.world)
    if need > ws.bytes_per_buffer:
        raise ValueError(f"Lamport buffer of {ws.bytes_per_buffer} B too small for {m} head rows")
    rp, lp, rn, ln = _gather_args(out_router, out_latent)
    _compiled_route_quant(ws.world, rp, lp, bool(use_pdl), rn, ln)(
        Int64(out_router.data_ptr()),
        Int64(out_latent.data_ptr()),
        Int64(ws.uc_local),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int64(bias.data_ptr()),
        Int64(topk_ids.data_ptr()),
        Int64(topk_weights.data_ptr()),
        Int64(x_fp8.data_ptr()),
        Int64(x_sf.data_ptr()),
        Int32(m),
        Int32(ws.rank),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


__all__ = [
    "MoeHeadGatherKernel",
    "MoeHeadRouteQuantKernel",
    "compile_head_gather",
    "compile_head_route_quant",
    "gather_bytes",
    "head_planes",
    "kind_tiles",
    "run_head_gather",
    "run_head_route_quant",
    "run_head_shard_gemm",
]
