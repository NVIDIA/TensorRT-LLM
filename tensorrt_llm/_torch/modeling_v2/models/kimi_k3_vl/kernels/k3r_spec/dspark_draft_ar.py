# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark drafter all-reduce sites on a pushed MNNVL all-reduce (SM100 / SM103 / SM107).

The drafter's o_proj and down_proj outputs are TP partial sums that the MNNVL all-reduce reduces
together with the residual add and the following RMSNorm (``AllReduceFusionOp.RESIDUAL_RMS_NORM``).
Here the producer GEMM stores its BF16 output straight into the Lamport slots of the all-reduce
(:mod:`dspark_draft_gemm_push`, or ``k3_lamport_ar.run_lamport_push``), and
:class:`DraftArNormKernel` runs the rest of the call:

* every thread owns eight consecutive values of a row; the rows are split over a cluster of CTAs
  exactly as the C++ launcher splits them (:func:`grid_config`, ``adjustGridConfig``);
* it polls the slots of its eight values (one-shot transport: every rank's slot of the row;
  two-shot: the row's owner sums its scatter slots and multicasts the row, the others poll the
  broadcast row), sums the ranks in FP32 in rank order and rounds to BF16;
* adds the residual in BF16 (``add.rn.bf16x2``) and stores it as the new residual;
* RMSNorm: BF16 squares summed in element order, warp butterflies (a partial last warp adds zero
  for its missing lanes), the warp-0 butterfly over the warp sums, the CTA sums of the row added in
  CTA order, ``rsqrt.approx(div.rn(sum, hidden) + eps)``, then the output as the one-shot kernel
  (``bf16((x * r) * gamma)``) or as ``rmsNormLamport`` after the two-shot all-reduce
  (``bf16((gamma * x) * r)``), whichever C++ path the call stands in for.

The result is bitwise ``mnnvl_fusion_allreduce(RESIDUAL_RMS_NORM)`` of the pushed input.

Launch and ordering (the rules of ``k3_lamport_ar.K3LamportFinishKernel``): the kernel releases its
dependents first (the next GEMM streams its weight while the all-reduce is in flight), reads the
workspace's rotation state and polls without waiting for its producer; that is safe because the
producer releases it only after its own ``griddepcontrol.wait``. The residual and gamma it reads are
older than the producer. CTA (0, 0) alone waits for the producer before it advances the rotation
state, after every CTA has counted itself; the CTAs re-arm the previous call's buffer.

:class:`DraftEmbedKernel` is the drafter's embedding site: the vocab-sharded table lookup followed by
the all-reduce of a tensor whose rows are zero except on the rank that owns the token. The owner
stores its row (``-0`` sent as ``+0``) into slot ``(token, owner)`` of every rank's one-shot buffer
and every rank polls that one slot, so the result is the all-reduce's, bit for bit, in one launch
and one hop.
"""

from __future__ import annotations

import functools
import struct

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op

from ..k3r_comm.k3_lamport_ar import (
    NUM_BUFFERS,
    clean4,
    clear_dirty_stages,
    finish_two_shot_v4,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    lamport_state,
    ld_acquire_u32,
    ld_v4,
    ld_volatile_v4,
    multimem_st_v4,
    pack_bf16x2,
    poll_reduce_v4,
    red_release_add,
    reduce_chunk,
    sanitize_bf16x2,
    st_u32,
    st_v4,
    stage_addr,
)
from ..k3r_comm.k3_latent_ar_norm import (
    _asm,
    _bf16x2_f32,
    _cluster_sync_release_acquire,
    _fadd,
    _fmul,
    _fv,
    _iv,
    _lds_f32,
    _sq_bf16x2,
    _st_cluster_f32,
    _sts_f32,
)

#: Values per thread (one 16-byte access).
ELTS = 8
#: Word of the CTA arrival counter in the workspace's state flags.
_FLAG_ARRIVALS = 8
#: Threads per CTA of the embedding site.
EMBED_THREADS = 128


def grid_config(num_tokens: int, hidden: int, sm_count: int) -> tuple[int, int]:
    """``(threads per CTA, CTAs per row)`` of the C++ one-shot kernel and of ``rmsNormLamport`` for
    BF16 rows (``adjustGridConfig<true>`` with eight values per thread), or ``(0, 0)`` where it
    takes more than one load per thread or no cluster."""
    threads = -(-hidden // ELTS)
    cluster = 8
    while threads % cluster and cluster > 1:
        cluster //= 2
    max_cluster = cluster
    block = -(-threads // cluster)
    while block < 128 and cluster >= 2:
        block *= 2
        cluster //= 2
    while num_tokens * cluster > sm_count and cluster > 1 and block <= 512:
        block *= 2
        cluster //= 2
    while cluster < max_cluster:
        cand = cluster * 2
        cand_block = -(-threads // cand)
        if cand_block < 64 or num_tokens * cand > sm_count:
            break
        cluster, block = cand, cand_block
    if block > 1024 or block * cluster * ELTS != hidden:
        return 0, 0
    return block, cluster


@dsl_user_op
def _add_bf16x2(a, b, *, loc=None, ip=None):
    """Elementwise BF16 sum of two pairs, rounded to nearest (``__hadd2``)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "add.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_add(value, offset: int, lanes: int, lane, *, loc=None, ip=None):
    """One butterfly step of the C++ warp sums over the first ``lanes`` lanes of a warp:
    ``value + (partner < lanes ? partner's value : 0)``."""
    mask = (1 << lanes) - 1 if lanes < 32 else 0xFFFFFFFF
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip), _iv(lane, loc=loc, ip=ip)],
            "{\n.reg .pred p;\n.reg .b32 t;\n.reg .f32 o;\n"
            f"shfl.sync.bfly.b32 o, $1, {offset}, 0x1f, {mask};\n"
            f"xor.b32 t, $2, {offset};\nsetp.lt.u32 p, t, {lanes};\n"
            "selp.f32 o, o, 0f00000000, p;\nadd.rn.f32 $0, $1, o;\n}",
            "=f,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rms_scale(total, eps, inv_hidden_hex: str, *, loc=None, ip=None):
    """``rsqrtf(total / hidden + eps)`` as the C++ kernels compute it (IEEE division, then add)."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(total, loc=loc, ip=ip), _fv(eps, loc=loc, ip=ip)],
            "{\n.reg .f32 q;\n"
            f"div.rn.f32 q, $1, 0f{inv_hidden_hex};\n"
            "add.rn.f32 q, q, $2;\nrsqrt.approx.f32 $0, q;\n}",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


class DraftArNormKernel:
    """``norm(allreduce(pushed) + residual)`` and the new residual for ``M`` rows of ``hidden`` BF16
    values on ``world`` ranks (see the module docstring)."""

    def __init__(
        self,
        world: int,
        hidden: int,
        threads: int,
        cluster: int,
        one_shot: bool,
        cpp_one_shot: bool,
        use_pdl: bool,
    ):
        if threads * cluster * ELTS != hidden:
            raise ValueError(f"{cluster} x {threads} threads do not cover {hidden} values")
        self.world = world
        self.hidden = hidden
        self.threads = threads
        self.cluster = cluster
        self.one_shot = one_shot
        self.cpp_one_shot = cpp_one_shot
        self.use_pdl = use_pdl
        self.warps = (threads + 31) // 32
        self.last_lanes = threads - 32 * (self.warps - 1)
        self.chunk = reduce_chunk(world, cpp_one_shot)
        self.hidden_hex = struct.pack(">f", float(hidden)).hex()

    @cute.jit
    def __call__(
        self,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        res_in: Int64,
        gamma: Int64,
        out: Int64,
        res_out: Int64,
        m: Int32,
        rank: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_sum: cute.struct.MemRange[cutlass.Float32, 32]
            block_sum: cute.struct.MemRange[cutlass.Float32, 1]
            cta_sum: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 8], 16]

        self.shared_storage = SharedStorage
        self.kernel(uc_local, mc_base, flags, res_in, gamma, out, res_out, m, rank, eps).launch(
            grid=(m, self.cluster, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        res_in: Int64,
        gamma: Int64,
        out: Int64,
        res_out: Int64,
        m: Int32,
        rank: Int32,
        eps: Float32,
    ):
        W, H, threads = self.world, self.hidden, self.threads
        tidx, _, _ = cute.arch.thread_idx()
        tok, cy, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        warp_sum = storage.warp_sum.data_ptr().toint()
        block_sum = storage.block_sum.data_ptr().toint()
        cta_sum = storage.cta_sum.data_ptr().toint()

        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()
        cur, dirty, nbytes, dstages, c0, c1, c2, c3 = lamport_state(flags)
        # Count this CTA once every thread has read the state.
        cute.arch.barrier()
        if tidx == 0:
            red_release_add(flags + Int64(_FLAG_ARRIVALS * 4), Int32(1))

        col = (cy * threads + tidx) * ELTS
        row_off = Int64(tok * H + col) * 2
        rw = ld_v4(res_in + row_off)
        gw = ld_v4(gamma + Int64(col) * 2)

        # Re-arm the previous call's buffer while the rows travel.
        clear_dirty_stages(
            uc_local,
            dirty,
            nbytes,
            dstages,
            c0,
            c1,
            c2,
            c3,
            (tok * self.cluster + cy) * threads + tidx,
            m * self.cluster * threads,
        )

        red = [Int32(0)] * 4
        if cutlass.const_expr(self.one_shot):
            buf = stage_addr(uc_local, cur, Int32(0), 1, nbytes)
            acc = poll_reduce_v4(buf + Int64(tok * W * H + col) * 2, H * 2, W, self.chunk)
            red = [pack_bf16x2(acc[2 * j], acc[2 * j + 1]) for j in range(4)]
        else:
            w0, w1, w2, w3 = finish_two_shot_v4(
                uc_local, mc_base, cur, nbytes, tok, col * 2, rank, Int32(1), W, self.chunk, H
            )
            red = [w0, w1, w2, w3]

        y = [_add_bf16x2(red[j], rw[j]) for j in range(4)]
        st_v4(res_out + row_off, y[0], y[1], y[2], y[3])

        # RMSNorm: this thread's BF16 squares in element order.
        thread_sum = Float32(0.0)
        for j in cutlass.range_constexpr(4):
            lo, hi = _bf16x2_f32(_sq_bf16x2(y[j]))
            thread_sum = _fadd(thread_sum, lo)
            thread_sum = _fadd(thread_sum, hi)
        v = thread_sum
        if cutlass.const_expr(self.last_lanes < 32):
            if warp == self.warps - 1:
                for off in (16, 8, 4, 2, 1):
                    v = _shfl_add(v, off, self.last_lanes, lane)
            else:
                for off in (16, 8, 4, 2, 1):
                    v = _shfl_add(v, off, 32, lane)
        else:
            for off in (16, 8, 4, 2, 1):
                v = _shfl_add(v, off, 32, lane)
        if lane == 0:
            _sts_f32(warp_sum + warp * 4, v)
        cute.arch.barrier()
        if warp == 0:
            b = Float32(0.0)
            if lane < self.warps:
                b = _lds_f32(warp_sum + lane * 4)
            for off in (16, 8, 4, 2, 1):
                b = _shfl_add(b, off, 32, lane)
            if lane == 0:
                _sts_f32(block_sum, b)
        cute.arch.barrier()
        total = _lds_f32(block_sum)
        if cutlass.const_expr(self.cluster > 1):
            # The row's CTA sums: CTA c's into every peer's slot c, then added in CTA order.
            if tidx < self.cluster:
                _st_cluster_f32(cta_sum + cy * 4, tidx, total)
            _cluster_sync_release_acquire()
            total = Float32(0.0)
            for c in cutlass.range_constexpr(self.cluster):
                total = _fadd(total, _lds_f32(cta_sum + c * 4))
        scale = _rms_scale(total, eps, self.hidden_hex)

        ow = []
        for j in cutlass.range_constexpr(4):
            x_lo, x_hi = _bf16x2_f32(y[j])
            g_lo, g_hi = _bf16x2_f32(gw[j])
            if cutlass.const_expr(self.cpp_one_shot):
                ow.append(
                    pack_bf16x2(_fmul(_fmul(x_lo, scale), g_lo), _fmul(_fmul(x_hi, scale), g_hi))
                )
            else:
                ow.append(
                    pack_bf16x2(_fmul(_fmul(g_lo, x_lo), scale), _fmul(_fmul(g_hi, x_hi), scale))
                )
        st_v4(out + row_off, ow[0], ow[1], ow[2], ow[3])

        # CTA (0, 0): the next call's state, once the producer (which reads the state after its own
        # wait) has completed and every CTA has read the state.
        if tok == 0:
            if cy == 0:
                if tidx == 0:
                    if cutlass.const_expr(self.use_pdl):
                        griddepcontrol_wait()
                    arrivals = flags + Int64(_FLAG_ARRIVALS * 4)
                    total_ctas = m * self.cluster
                    arrived = ld_acquire_u32(arrivals)
                    while arrived < total_ctas:
                        arrived = ld_acquire_u32(arrivals)
                    nxt = cur + 1
                    if nxt == NUM_BUFFERS:
                        nxt = Int32(0)
                    row_bytes = H * 2
                    if cutlass.const_expr(self.one_shot):
                        st_v4(flags, nxt, cur, nbytes, Int32(1))
                        st_v4(flags + Int64(16), m * (W * row_bytes), Int32(0), Int32(0), Int32(0))
                    else:
                        st_v4(flags, nxt, cur, nbytes, Int32(2))
                        st_v4(
                            flags + Int64(16),
                            ((m + (W - 1)) // W) * (W * row_bytes),
                            m * row_bytes,
                            Int32(0),
                            Int32(0),
                        )
                    st_u32(arrivals, Int32(0))


class DraftEmbedKernel:
    """``all_reduce(vocab-sharded lookup)`` of ``M`` token ids: the owner rank's table row of each
    token, ``-0`` as ``+0``, zero for ids outside ``[0, vocab)`` (see the module docstring)."""

    def __init__(self, world: int, hidden: int, anchor64: bool, use_pdl: bool):
        if hidden % (ELTS * EMBED_THREADS):
            raise ValueError(f"hidden={hidden} must be a multiple of {ELTS * EMBED_THREADS}")
        self.world = world
        self.hidden = hidden
        self.ids64 = anchor64
        self.use_pdl = use_pdl
        self.cols = hidden // (ELTS * EMBED_THREADS)

    @cute.jit
    def __call__(
        self,
        ids: Int64,
        table: Int64,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        out: Int64,
        m: Int32,
        rank: Int32,
        slice_width: Int32,
        vocab: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(ids, table, uc_local, mc_base, flags, out, m, rank, slice_width, vocab).launch(
            grid=(m, self.cols, 1),
            block=(EMBED_THREADS, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        ids: Int64,
        table: Int64,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        out: Int64,
        m: Int32,
        rank: Int32,
        slice_width: Int32,
        vocab: Int32,
    ):
        W, H = self.world, self.hidden
        tidx, _, _ = cute.arch.thread_idx()
        tok, cy, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()
            griddepcontrol_launch_dependents()
        cur, dirty, nbytes, dstages, c0, c1, c2, c3 = lamport_state(flags)
        cute.arch.barrier()
        if tidx == 0:
            red_release_add(flags + Int64(_FLAG_ARRIVALS * 4), Int32(1))

        col = (cy * EMBED_THREADS + tidx) * ELTS
        token = _ld_id(ids + Int64(tok) * (8 if self.ids64 else 4))
        owner = Int32(-1)
        if token >= 0:
            if token < vocab:
                owner = token // slice_width
        buf_off = Int64(cur) * Int64(nbytes)
        slot = Int64(((tok * W + owner) * H + col) * 2)
        if owner == rank:
            w = ld_v4(table + Int64((token - rank * slice_width) * H + col) * 2)
            multimem_st_v4(
                mc_base + buf_off + slot,
                sanitize_bf16x2(w[0]),
                sanitize_bf16x2(w[1]),
                sanitize_bf16x2(w[2]),
                sanitize_bf16x2(w[3]),
            )
        clear_dirty_stages(
            uc_local,
            dirty,
            nbytes,
            dstages,
            c0,
            c1,
            c2,
            c3,
            (tok * self.cols + cy) * EMBED_THREADS + tidx,
            m * self.cols * EMBED_THREADS,
        )
        r0 = Int32(0)
        r1 = Int32(0)
        r2 = Int32(0)
        r3 = Int32(0)
        if owner >= 0:
            w = ld_volatile_v4(uc_local + buf_off + slot)
            while clean4(w[0], w[1], w[2], w[3]) == 0:
                w = ld_volatile_v4(uc_local + buf_off + slot)
            r0 = w[0]
            r1 = w[1]
            r2 = w[2]
            r3 = w[3]
        st_v4(out + Int64(tok * H + col) * 2, r0, r1, r2, r3)

        if tok == 0:
            if cy == 0:
                if tidx == 0:
                    arrivals = flags + Int64(_FLAG_ARRIVALS * 4)
                    arrived = ld_acquire_u32(arrivals)
                    while arrived < m * self.cols:
                        arrived = ld_acquire_u32(arrivals)
                    nxt = cur + 1
                    if nxt == NUM_BUFFERS:
                        nxt = Int32(0)
                    st_v4(flags, nxt, cur, nbytes, Int32(1))
                    st_v4(flags + Int64(16), m * (W * H * 2), Int32(0), Int32(0), Int32(0))
                    st_u32(arrivals, Int32(0))


@dsl_user_op
def _ld_id(addr, *, loc=None, ip=None):
    """A token id (an int32, or the low word of an int64) written by an earlier grid."""
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@functools.cache
def _compiled_ar_norm(world, hidden, threads, cluster, one_shot, cpp_one_shot, use_pdl):
    kernel = DraftArNormKernel(world, hidden, threads, cluster, one_shot, cpp_one_shot, use_pdl)
    a = Int64(1 << 20)
    return cute.compile(kernel, a, a, a, a, a, a, a, Int32(1), Int32(0), Float32(1e-6), _stream())


@functools.cache
def _compiled_embed(world, hidden, ids64, use_pdl):
    kernel = DraftEmbedKernel(world, hidden, ids64, use_pdl)
    a = Int64(1 << 20)
    i = Int32(1)
    return cute.compile(kernel, a, a, a, a, a, a, i, i, i, i, _stream())


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def ar_norm_config(num_tokens: int, hidden: int, device: torch.device) -> tuple[int, int]:
    """``(threads, cluster)`` of :class:`DraftArNormKernel` for this call, ``(0, 0)`` if none."""
    return grid_config(num_tokens, hidden, _sm_count(device.index))


def run_draft_ar_norm(
    workspace,
    one_shot: bool,
    cpp_one_shot: bool,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    residual_out: torch.Tensor,
    use_pdl: bool = True,
) -> None:
    """Finish the all-reduce every rank pushed into ``workspace`` (a ``k3_lamport_ar.LamportWorkspace``)
    with the ``one_shot`` or two-shot transport, add ``residual`` and RMSNorm with ``gamma``:
    ``out`` / ``residual_out`` (contiguous BF16 ``[M, hidden]``) as the C++ fused all-reduce
    (``cpp_one_shot``: its one-shot kernel, else two-shot + ``rmsNormLamport``) returns them."""
    m, hidden = out.shape
    threads, cluster = ar_norm_config(m, hidden, out.device)
    if not threads:
        raise ValueError(f"no single-load launch shape for {m} x {hidden}")
    _compiled_ar_norm(workspace.world, hidden, threads, cluster, one_shot, cpp_one_shot, use_pdl)(
        Int64(workspace.uc_local),
        Int64(workspace.mc_base),
        Int64(workspace.flags.data_ptr()),
        Int64(residual.data_ptr()),
        Int64(gamma.data_ptr()),
        Int64(out.data_ptr()),
        Int64(residual_out.data_ptr()),
        Int32(m),
        Int32(workspace.rank),
        Float32(eps),
        _stream(),
    )


def run_draft_embed(
    workspace,
    ids: torch.Tensor,
    table: torch.Tensor,
    slice_width: int,
    vocab: int,
    out: torch.Tensor,
    use_pdl: bool = True,
) -> None:
    """``out`` (contiguous BF16 ``[M, hidden]``) = the all-reduce of the vocab-sharded lookup of
    ``ids`` (contiguous int32 / int64 ``[M]``) in this rank's ``table`` rows
    ``[rank * slice_width, ...)``, through the one-shot buffer of ``workspace``."""
    m, hidden = out.shape
    _compiled_embed(workspace.world, hidden, ids.dtype == torch.int64, use_pdl)(
        Int64(ids.data_ptr()),
        Int64(table.data_ptr()),
        Int64(workspace.uc_local),
        Int64(workspace.mc_base),
        Int64(workspace.flags.data_ptr()),
        Int64(out.data_ptr()),
        Int32(m),
        Int32(workspace.rank),
        Int32(slice_width),
        Int32(vocab),
        _stream(),
    )


__all__ = [
    "DraftArNormKernel",
    "DraftEmbedKernel",
    "ar_norm_config",
    "grid_config",
    "run_draft_ar_norm",
    "run_draft_embed",
]
