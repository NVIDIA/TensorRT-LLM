# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA input projection and spec verify recurrence in one kernel for ``requests`` x
``tokens`` verify rows (Rubin TP16; M16 = 2 requests x 8, M8 = 2 requests x 4):
``KdaHeadVerify`` with a request axis.

The projection is ``KdaHeadVerify``'s (``KdaHeadWide``'s 128-row Q / K / V / gate tiles
split over K by four CTAs, weights streamed and staged in tensor memory before the PDL
wait) with MMA N = tokens x requests, so the weight stream, its placement and its
pre-wait share are the same as at one request. Token rows are request-major (row
``tokens`` r + t).

Hop 1: CTA c < N owns token row c. Every CTA pushes its FP32 partial of each token's
section rows to that token's owner, which sums the four split ranks in rank order and
rounds to BF16 (bitwise the split-K projection's (128, N, 4) tactic, as
``KdaHeadVerify``). Hop 2: the owner sends its Q / K channels to the eight KDA CTAs of
its request and each V row block to the KDA CTA owning those rows; it keeps the output
gate.
Recurrence: CTA c is KDA CTA c % 8 of request c // 8 (state rows [16 (c % 8), + 16)) and
runs ``KdaHeadVerify``'s recurrence (``KdaSplit``'s B1 verify schedule) for its request,
so given the same projected inputs the outputs, checkpoints and conv windows are bit for
bit ``KdaSplit``'s. KDA CTA j < tokens of a request normalizes token j's q / k. Owner c
then applies the gated RMS norm of token row c.

At q = 4 every cluster CTA still runs one KDA role (two requests x eight 16-row
blocks, each a 4-token loop), where the B1 x q 8 head leaves eight of its sixteen
CTAs without one; only the token rows' ownership (eight owners) differs from M16."""

from functools import cache

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass.cute import experimental as cute_ext
from cutlass.cute.arch import store_async_dsmem
from cutlass.cute.nvgpu import tcgen05

from ..attn_res.kernel import BESIDE_SMEM
from ..common.primitives import (
    asm,
    at,
    float_bits,
    fma,
    load_f32x4,
    load_words,
    mma_into,
    named_barrier,
    push_float,
    sigmoid,
    wait,
    warp_sum,
    zeros,
)
from ..gemm.splitk import SplitKGemm, SplitKTactic, _fence_tcgen05, _smem
from .head import (
    A_DESC_HIGH,
    B_DESC_HIGH,
    CLUSTER,
    SECTIONS,
    _copy_to_tmem,
    load_bf16x4,
    store_bf16_canonical,
)
from .head_verify import KDA_CTAS, ROWS, KdaHeadVerify
from .kernel import sigmoids

TOKEN_INPUTS = 4 * 128  # an owned token's Q | K | V | gate (128 channels / rows each)


def _idesc(n):
    """tcgen05 kind::f16 instruction descriptor: BF16 A / B (K-major), FP32 D, M128."""
    return 0x08000490 | (n >> 3) << 17


# SW128 K-major B (x) descriptor's upper word with SBO = 1024 (the next 8 token rows):
# ``B_DESC_HIGH`` leaves SBO 0, which only an N = 8 tile (one core-matrix row group)
# can use.
B_DESC_HIGH_SBO = 0x40004040


@cache
def _mma_ops(n):
    """(smem-A, tmem-A) MMAs with an N = ``n`` instruction descriptor."""
    idesc = _idesc(n)
    b_high = B_DESC_HIGH if n == 8 else B_DESC_HIGH_SBO

    @asm()
    def mma_smem(d, a_desc, b_desc, accumulate):
        code = (
            "{ .reg .pred p; .reg .b64 ad, bd; .reg .b32 ah, bh; setp.ne.b32 p, $3, 0; "
        )
        code += f"mov.b32 ah, {A_DESC_HIGH}; mov.b32 bh, {b_high}; "
        code += "mov.b64 ad, {$1, ah}; mov.b64 bd, {$2, bh}; "
        code += f"tcgen05.mma.cta_group::1.kind::f16 [$0], ad, bd, {idesc}, p; }}"
        operands = (Int32(d), Int32(a_desc), Int32(b_desc), Int32(accumulate))
        return code, "r,r,r,r,~{memory}", operands

    @asm()
    def mma_tmem(d, a, b_desc, accumulate):
        code = "{ .reg .pred p; .reg .b64 bd; .reg .b32 bh; setp.ne.b32 p, $3, 0; "
        code += f"mov.b32 bh, {b_high}; mov.b64 bd, {{$2, bh}}; "
        code += f"tcgen05.mma.cta_group::1.kind::f16 [$0], [$1], bd, {idesc}, p; }}"
        operands = (Int32(d), Int32(a), Int32(b_desc), Int32(accumulate))
        return code, "r,r,r,r,~{memory}", operands

    return mma_smem, mma_tmem


def _align(offset, alignment):
    return -(-offset // alignment) * alignment


class KdaHeadMulti(KdaHeadVerify):
    """``KdaHeadVerify`` for ``requests`` x ``tokens`` verify rows (request-major);
    ``accepted`` [requests], ``slots`` [requests, >= tokens]. Ring ``stages`` shrink so
    the resident x tiles of every token fit beside an AttnRes CTA."""

    def __init__(
        self, heads, *, requests=2, tokens=8, stages=7, cta_k=64, tmem_tiles=15,
        early_loads=21, bf16_state=True, has_bias=False, hidden=7168,
    ):  # fmt: skip
        # KdaHeadVerify's set-up (its B1 x q 8 geometry check), then this kernel's q
        super().__init__(
            heads, tokens=8, cta_k=cta_k, tmem_tiles=tmem_tiles,
            early_loads=early_loads, bf16_state=bf16_state, has_bias=has_bias,
            hidden=hidden,
        )  # fmt: skip
        assert requests * KDA_CTAS == CLUSTER, "one KDA role per cluster CTA"
        assert tokens in (4, 8), "an even token loop, at most one token per KDA CTA"
        self.tokens = tokens
        n = requests * tokens
        assert n <= CLUSTER, "at most one owned token row per cluster CTA"
        self.requests, self.rows_total = requests, n
        SplitKGemm.__init__(self, tactic=SplitKTactic(128, n, SECTIONS, stages))
        self.cluster_shape = (CLUSTER, 1, 1)
        self.cta_k, self.k_tiles = cta_k, hidden // cta_k // SECTIONS
        assert n + tmem_tiles * cta_k // 2 <= 512 and tmem_tiles > 0
        assert stages > early_loads - tmem_tiles, "pre-wait A tiles beyond TMEM"
        T = tokens
        # The kernel's shared memory in allocation order (with each buffer's alignment).
        parts = (
            (stages * 128 * cta_k * 2, 1024),  # A ring
            (self.k_tiles * n * cta_k * 2, 1024),  # resident x
            (stages * 8, 8),
            (stages * 8, 8),
            (self.k_tiles * 8, 8),  # ring, x barriers
            (8, 8),
            (8, 8),
            (4, 4),  # epilogue / TMEM barriers, TMEM base
            (128 * n * 4, 128),  # FP32 partial [128 rows, n tokens]
            (SECTIONS * TOKEN_INPUTS * 4, 128),  # hop 1: four splits of the owned token
            (2 * T * 128 * 2, 128),
            (T * ROWS * 2, 16),  # hop 2: raw Q / K, V rows
            (TOKEN_INPUTS * 2, 16),  # the owned token's rounded inputs
            (3 * T * 128 * 4, 16),  # q, k, decays (three buffers of T x 128)
            (T * ROWS * 4, 16),
            (128 * 4, 16),
            (T * ROWS * 4, 16),
            (T * 4, 16),
            (8 * 4, 16),
            (8 * 4, 16),  # v, o, local outputs, slots, norms, betas
            (5 * 8, 8),  # hop / gate / output / q-k / input barriers
        )
        used = 0
        for size, alignment in parts:
            used = _align(used, alignment) + size
        self.smem_used = used
        self.smem_bytes = (used + 2560 + 1023) // 1024 * 1024
        assert self.smem_bytes <= BESIDE_SMEM, (self.smem_bytes, BESIDE_SMEM)

    @cute.experimental.jit
    def __call__(
        self, mA: cute.Tensor, mB: cute.Tensor, a_map: cutlass.Int64,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        fa_beta: cute.Tensor, a_log: cute.Tensor, dt_bias: cute.Tensor,
        slots: cute.Tensor, state: cute.Tensor, output: cute.Tensor,
        norm_w: cute.Tensor, f_b: cute.Tensor, accepted: cute.Tensor,
        lower_bound: Float32, scale: Float32, eps: Float32, counter: cutlass.Int64,
        target: Int32, out_rearm: cutlass.Int64, stream,
    ):  # fmt: skip
        self.kernel(
            mA, mB, a_map, conv_w, bias, conv_state, fa_beta, a_log, dt_bias, slots,
            state, output, norm_w, f_b, accepted, lower_bound, scale, eps, counter,
            target, out_rearm,
        ).launch(
            grid=(CLUSTER, self.heads, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )  # fmt: skip

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, a_map: cutlass.Int64,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        fa_beta: cute.Tensor, a_log: cute.Tensor, dt_bias: cute.Tensor,
        slots: cute.Tensor, state: cute.Tensor, output: cute.Tensor,
        norm_w: cute.Tensor, f_b: cute.Tensor, accepted: cute.Tensor,
        lower_bound: Float32, scale: Float32, eps: Float32, counter: cutlass.Int64,
        target: Int32, out_rearm: cutlass.Int64,
    ):  # fmt: skip
        T: cutlass.Constexpr = self.tokens
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        rank, head, _ = cute.arch.block_idx()
        section, split_rank = rank % SECTIONS, rank // SECTIONS
        row0 = section * self.dim + head * 128
        request, block = rank // KDA_CTAS, rank % KDA_CTAS  # this CTA's KDA role
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tid, _, _ = cute.arch.thread_idx()

        a_shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        b_shape = (tiled_mma, mnk_tiler, ab_dtype, self.k_tiles)
        sA = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*a_shape))
        sB = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*b_shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        b_full = _smem(cutlass.Int64, self.k_tiles, 8).iterator
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        tile_layout = cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m))
        partial = _smem(Float32, None, 128, tile_layout)
        x_in = _smem(Float32, SECTIONS * TOKEN_INPUTS, 128)
        raw_qk = _smem(BFloat16, 2 * T * 128, 128)
        raw_v = _smem(BFloat16, T * ROWS, 16)
        mine = _smem(BFloat16, TOKEN_INPUTS, 16)
        shared_q = _smem(Float32, T * 128, 16)
        shared_k = _smem(Float32, T * 128, 16)
        shared_decay = _smem(Float32, T * 128, 16)
        shared_v = _smem(Float32, T * ROWS, 16)
        shared_o = _smem(Float32, 128, 16)
        local_o = _smem(Float32, T * ROWS, 16)
        slot_rows = _smem(Int32, T, 16)
        reductions = _smem(Float32, 8, 16)
        shared_beta = _smem(Float32, 8, 16)
        bar_in = _smem(cutlass.Int64, 1, 8).iterator
        bar_hop = _smem(cutlass.Int64, 1, 8).iterator
        bar_gate = _smem(cutlass.Int64, 1, 8).iterator
        bar_out = _smem(cutlass.Int64, 1, 8).iterator
        bar_qk = _smem(cutlass.Int64, 1, 8).iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 1)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                for i in range(self.k_tiles):
                    cute.arch.mbarrier_init(b_full + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
        if tid == 128:
            cute.arch.mbarrier_init(bar_in, 1)  # 4 splits x 4 sections x 128 rows
            cute.arch.mbarrier_arrive_and_expect_tx(bar_in, SECTIONS * TOKEN_INPUTS * 4)
            # Q / K of the request's T tokens x 128 channels; V: T x 16 rows
            cute.arch.mbarrier_init(bar_hop, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(
                bar_hop, 2 * T * 128 * 2 + T * ROWS * 2
            )
            cute.arch.mbarrier_init(bar_gate, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(
                bar_gate, KDA_CTAS * ROWS * T * 4 + T * 4
            )
            cute.arch.mbarrier_init(bar_out, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(bar_out, 512)
            cute.arch.mbarrier_init(bar_qk, 1)
            # the normalized q / k of the request's tokens (KDA CTA j: token j)
            qk_bytes = min(T, KDA_CTAS) * 2 * 128 * 4
            cute.arch.mbarrier_arrive_and_expect_tx(bar_qk, qk_bytes)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()  # every barrier before any DSMEM traffic

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        tiles = (split_rank * k_tile_count, k_tile_count)
        offset = accepted[request] - 1
        slot = slots[request, offset]
        if warp_idx < 3:
            cute.arch.cluster_wait()
            if warp_idx == 0:
                self.weight_warp(bar_full, bar_empty, sA, a_map, row0, *tiles)
            elif warp_idx == 1:
                b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (0, None, 0))
                b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
                self.resident_b(b_full, b_tile, sB, b_map, tiles[0], 0)
                if cutlass.const_expr(self.polled_out):  # after the PDL wait above
                    self.rearm_out(out_rearm, output.shape[0], head * CLUSTER + rank)
            else:
                self.staged_mma_multi(
                    bar_full, bar_empty, b_full, bar_mma_epilog, bar_tmem_alloc, sA, sB,
                    tmem_base_ptr,
                )  # fmt: skip
            cute.arch.cluster_arrive_relaxed()
        elif warp_idx == 3:
            gate_args = (fa_beta, a_log, dt_bias, f_b, lower_bound, counter, target)
            coords = (head, rank, slot, request, block)
            self.multi_gate(gate_args, coords, shared_decay, shared_beta, bar_gate)
            cute.arch.cluster_arrive_relaxed()
        else:
            kt = tid - 128
            tensors = (conv_w, bias, conv_state, state, output, norm_w)
            coords = (head, rank, slot, offset, request, block)
            early = self.multi_early(kt, tensors, coords, slots)
            cute.arch.cluster_wait()
            # the other B_ISSUERS - 1 warps of x's tiles
            b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (0, None, 0))
            b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
            self.resident_b(b_full, b_tile, sB, b_map, tiles[0], warp_idx - 3)
            self.push_tokens(
                kt, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, partial,
                section, split_rank, x_in, bar_in,
            )  # fmt: skip
            if cutlass.const_expr(self.rows_total == CLUSTER):  # one row per CTA
                wait(bar_in)
                self.round_owned(kt, x_in, mine)
                named_barrier(1, 128)
                self.multi_hop(
                    kt, request, rank % KDA_CTAS, mine, raw_qk, raw_v, bar_hop
                )
            elif rank < self.rows_total:  # CTAs past the token rows own none
                wait(bar_in)
                self.round_owned(kt, x_in, mine)
                named_barrier(1, 128)
                self.multi_hop(kt, rank // T, rank % T, mine, raw_qk, raw_v, bar_hop)
            cute.arch.cluster_arrive_relaxed()  # hop 1 done; hop 2 issued
            rows = (
                shared_q, shared_k, shared_decay, shared_v, shared_o, reductions,
                local_o, slot_rows,
            )  # fmt: skip
            raws = (raw_qk, raw_v, mine, bar_hop)
            bars = (bar_gate, bar_out, shared_beta, bar_qk)
            self.multi_main(
                kt, early, raws, rows, bars, tensors, coords, slots, scale, eps
            )
            if cutlass.const_expr(T < KDA_CTAS):  # noqa: SIM102 (owners: the norm)
                if rank < self.rows_total:
                    owned = (mine, shared_o, bar_out, output, norm_w, accepted)
                    self.owned_norm(kt, owned, slots, head, rank, eps)
        cute.arch.cluster_wait()  # no CTA leaves while a peer may write it

    @cute.experimental.jit
    def staged_mma_multi(
        self, bar_full, bar_empty, b_full, bar_mma_epilog, bar_tmem_alloc,
        sA: cute.Tensor, sB: cute.Tensor, tmem_base_ptr,
    ):  # fmt: skip
        """``KdaHeadWide.staged_mma_warp`` with an N = 8 x requests accumulator in the
        first N tensor-memory columns and the staged A tiles after it."""
        N: cutlass.Constexpr = self.rows_total
        mma_smem, mma_tmem = _mma_ops(self.rows_total)
        stages, staged, k_blocks = self.num_ab_stage, self.tmem_tiles, self.cta_k // 16
        a_layout = cute.slice_(sA.layout, (None, None, None, 0))
        b_layout = cute.slice_(sB.layout, (None, None, None, 0))
        a_step = cute.size_in_bytes(sA.element_type, a_layout) // 16
        b_step = cute.size_in_bytes(sB.element_type, b_layout) // 16
        columns = self.cta_k // 2  # tensor-memory columns of one A tile
        cute.arch.alloc_tmem(512, tmem_base_ptr, is_two_cta=False)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        a_desc = ((sA.iterator.toint() >> 4) & 0x3FFF) | 0x10000
        b_desc = ((sB.iterator.toint() >> 4) & 0x3FFF) | 0x10000
        for k in cutlass.range_constexpr(staged):
            stage, phase = k % stages, (k // stages) % 2
            cute.arch.mbarrier_wait(bar_full + stage, phase)
            _fence_tcgen05()
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(k_blocks):
                    source = a_desc + stage * a_step + 2 * kb
                    _copy_to_tmem(base + N + columns * k + 8 * kb, source)
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
        for k in cutlass.range_constexpr(staged):  # A from tensor memory
            cute.arch.mbarrier_wait(b_full + k, 0)
            _fence_tcgen05()
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(k_blocks):
                    b = b_desc + k * b_step + 2 * kb
                    accumulate = Int32(0 if k == 0 and kb == 0 else 1)
                    mma_tmem(base, base + N + columns * k + 8 * kb, b, accumulate)
        for k in cutlass.range(staged, self.k_tiles, unroll=1):  # A from the ring
            ring_stage, ring_phase = k % stages, (k // stages) % 2
            cute.arch.mbarrier_wait(bar_full + ring_stage, ring_phase)
            cute.arch.mbarrier_wait(b_full + k, 0)
            _fence_tcgen05()
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(k_blocks):
                    a_ring = a_desc + ring_stage * a_step + 2 * kb
                    mma_smem(base, a_ring, b_desc + k * b_step + 2 * kb, Int32(1))
                tcgen05.commit(bar_empty + ring_stage, None, self.cta_group)
        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)  # the release lands in phase 1
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        _fence_tcgen05()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        cute.arch.dealloc_tmem(tmem_ptr, 512, is_two_cta=False)

    @cute.experimental.jit
    def push_tokens(
        self, epi_tid: cutlass.Int32, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        acc_layout: cutlass.Constexpr, partial: cute.Tensor, section: cutlass.Int32,
        split_rank: cutlass.Int32, x_in: cute.Tensor, bar_in,
    ):  # fmt: skip
        """The accumulator -> ``partial`` (FP32, shared memory); then each token's 128
        section rows to the token's owner (CTA = token row), slot ``split_rank``."""
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()
        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr), acc_layout
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        d_layout = utils.LayoutEnum.from_tensor(partial)
        load_op = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k), d_layout, BFloat16, self.acc_dtype,
            epi_tile, False,
        )  # fmt: skip
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(load_op, acc_view)
        part_epi = cute.flat_divide(partial, epi_tile)
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, part_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _fence_tcgen05()
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute_ext.partition_and_copy(thr_t2r, rAcc, part_epi[None, None, 0, 0])
        named_barrier(3, 128)
        rows = self.cta_m
        for m in cutlass.range_constexpr(self.rows_total * 32 // 128):
            i = epi_tid + 128 * m
            token, chunk = i // 32, i % 32
            src = partial.iterator + cute.assume(token * rows + 4 * chunk, divby=4)
            words = load_words(src, count=4, shared=True)
            column = split_rank * TOKEN_INPUTS + section * 128 + 4 * chunk
            dst = x_in.iterator + cute.assume(column, divby=4)
            smem = cute.AddressSpace.smem
            dst = cute.make_ptr(Uint8, dst.llvm_ptr, smem, assumed_align=16)
            store_async_dsmem(dst, words, bar_in, token)

    @cute.jit
    def round_owned(self, kt, x_in, mine):
        """The owned token's Q, K, V and gate (channel / row ``kt``): the four split
        ranks' partials summed in rank order, rounded to BF16, into ``mine``."""
        for c in cutlass.range_constexpr(4):
            value = x_in[c * 128 + kt]
            for q in cutlass.range_constexpr(1, SECTIONS):
                value = value + x_in[q * TOKEN_INPUTS + c * 128 + kt]
            mine[c * 128 + kt] = BFloat16(value)

    @cute.jit
    def multi_hop(self, kt, request, position, mine, raw_qk, raw_v, bar_hop):
        """The owned token (``position`` of ``request``) to its request's KDA CTAs, 16
        bytes per store: Q / K channels to all eight, V rows [16 j, + 16) to KDA CTA
        j."""
        T: cutlass.Constexpr = self.tokens
        smem = cute.AddressSpace.smem
        first = KDA_CTAS * request
        for m in cutlass.range_constexpr(2):
            i = kt + 128 * m
            dest, c = i // 32, i % 32
            sec, sub = c // 16, c % 16
            src = mine.iterator + 128 * sec + 8 * sub
            dst = raw_qk.iterator + (sec * T + position) * 128 + 8 * sub
            words = load_words(src, count=4, shared=True)
            dst = cute.make_ptr(Uint8, dst.llvm_ptr, smem, assumed_align=16)
            store_async_dsmem(dst, words, bar_hop, first + dest)
        if kt < 16:  # V: rows 8 kt .. 8 kt + 7 -> KDA CTA kt // 2
            j, h = kt // 2, kt % 2
            src = mine.iterator + 256 + 8 * kt
            dst = raw_v.iterator + position * ROWS + 8 * h
            words = load_words(src, count=4, shared=True)
            dst = cute.make_ptr(Uint8, dst.llvm_ptr, smem, assumed_align=16)
            store_async_dsmem(dst, words, bar_hop, first + j)

    @cute.jit
    def multi_early(self, kt, tensors, coords, slots):
        """``KdaHeadVerify.verify_early`` for this CTA's request and row block."""
        conv_w, bias, conv_state, state, output, norm_w = tensors
        head, rank, slot, offset, request, block = coords
        group, group_lane = kt // self.vkda.lanes_per_row, kt % self.vkda.lanes_per_row
        value_base = block * ROWS
        safe = slot
        if safe < 0:
            safe = Int32(0)
        state_page = state[Int64(safe), None, None, None]
        history = zeros((self.vkda.passes, self.vkda.chunks, 4), Float32)
        for j in cutlass.range_constexpr(self.vkda.chunks):
            column = 4 * (group_lane + self.vkda.lanes_per_row * j)
            pointer = at(state_page, head, value_base + group, column)
            values = self.vkda.load_state(pointer)
            for e in cutlass.range_constexpr(4):
                history[0, j, e] = values[e]
        taps = zeros((3, 3), BFloat16)
        coeff = zeros((3, 4), conv_w.element_type)
        accum = zeros(3 * self.tokens, Float32)
        if slot > 0:
            dim = self.dim
            qk_channel = head * 128 + kt
            value_channel = head * 128 + value_base + kt
            conv_page = conv_state[Int64(slots[request, 0]), None, None]
            for s in cutlass.range_constexpr(3):
                channel = value_channel if s == 2 else qk_channel
                if (s != 2) | (kt < ROWS):
                    for t in cutlass.range_constexpr(3):
                        taps[s, t] = conv_page[s * dim + channel, offset + t]
                    for t in cutlass.range_constexpr(4):
                        coeff[s, t] = conv_w[s, t, channel]
                    if cutlass.const_expr(self.has_bias):
                        accum[s] = bias[s * dim + channel]
                        for t in cutlass.range_constexpr(1, self.tokens):
                            accum[3 * t + s] = accum[s]
        return history, taps, coeff, accum, norm_w[kt]

    @cute.jit
    def multi_gate(self, args, coords, shared_decay, shared_beta, barrier):
        """``KdaHeadVerify.verify_gate`` for this CTA's request and row block: F_A of
        the request's token rows, decays to the request's eight KDA CTAs."""
        fa_beta, a_log, dt_bias, weight, lower_bound, counter, target = args
        head, rank, slot, request, block = coords
        T: cutlass.Constexpr = self.tokens
        lane = cute.arch.lane_idx()
        group, quarter = lane >> 2, lane & 3
        value_base = block * ROWS
        features = [group, group + 8]
        dt = [dt_bias[head * 128 + value_base + f] for f in features]
        wr = cute.make_rmem_tensor((8, 4), cutlass.Uint32)
        row = head * 128 + value_base + group
        for chunk in cutlass.range_constexpr(8):
            col = 16 * chunk + 4 * quarter
            low = load_bf16x4(at(weight, row, col))
            high = load_bf16x4(at(weight, row + 8, col))
            wr[chunk, 0], wr[chunk, 2] = low
            wr[chunk, 1], wr[chunk, 3] = high
        a = a_log[head]
        cute.arch.cluster_wait()
        token = T * request + group % T
        ar, beta = self.poll_fa(fa_beta, counter, target, token, quarter, head)
        cute.arch.griddepcontrol_launch_dependents()
        if slot > 0:
            acc = zeros(4, Float32)
            for chunk in cutlass.range_constexpr(8):
                wa = tuple(wr[chunk, e] for e in range(4))
                mma_into(acc, wa, ar[chunk], bf16=True)
            for i in cutlass.range_constexpr(2):
                for column in cutlass.range_constexpr(2):
                    raw_g = Float32(BFloat16(acc[2 * i + column]))
                    gate = cute.math.exp(a, fastmath=True) * (raw_g + dt[i])
                    gate = sigmoid(gate, False)
                    decay = float_bits(cute.math.exp(lower_bound * gate, fastmath=True))
                    t = 2 * quarter + column
                    pointer = shared_decay.iterator + t * 128 + value_base + features[i]
                    if t < T:
                        for dest in cutlass.range_constexpr(KDA_CTAS):
                            first = KDA_CTAS * request
                            store_async_dsmem(pointer, decay, barrier, first + dest)
            if (quarter == 0) & (group < T):  # this CTA's betas (bytes on its barrier)
                beta_gate = sigmoid(Float32(beta), False)
                push_float(shared_beta.iterator + group, beta_gate, barrier, rank)
        self.rearm(counter, target, lane, fa_beta)

    @cute.jit
    def multi_main(
        self, kt, early, raws, rows, bars, tensors, coords, slots, scale, eps
    ):
        """``KdaHeadVerify.verify_main`` for this CTA's request and row block; at q = 8
        also the gated RMS norm of token row ``rank`` (the same request's position
        ``block``; at q < 8 ``owned_norm`` runs it)."""
        T: cutlass.Constexpr = self.tokens
        dim = self.dim
        history, taps, coeff, accum, nw = early
        raw_qk, raw_v, mine, bar_hop = raws
        shared_q, shared_k, shared_decay, shared_v, shared_o, reductions = rows[:6]
        local_o, slot_rows = rows[6:]
        bar_gate, bar_out, shared_beta, bar_qk = bars
        conv_w, bias, conv_state, state, output, norm_w = tensors
        head, rank, slot, offset, request, block = coords
        lane, warp = kt % 32, kt // 32
        group, group_lane = kt // self.vkda.lanes_per_row, kt % self.vkda.lanes_per_row
        value_base, qk_channel = block * ROWS, head * 128 + kt
        value_channel = head * 128 + value_base + kt
        first = KDA_CTAS * request
        conv_slot = slots[request, 0]
        if slot <= 0:
            if cutlass.const_expr(T == KDA_CTAS):
                output[rank, qk_channel] = BFloat16(0)
        else:
            conv_page = conv_state[Int64(conv_slot), None, None]
            wait(bar_hop)
            inputs = cute.make_rmem_tensor(3 * T, BFloat16)
            for t in cutlass.range_constexpr(T):
                inputs[3 * t] = raw_qk[t * 128 + kt]
                inputs[3 * t + 1] = raw_qk[(T + t) * 128 + kt]
                inputs[3 * t + 2] = BFloat16(0)
            if kt < ROWS:
                for t in cutlass.range_constexpr(T):
                    inputs[3 * t + 2] = raw_v[t * ROWS + kt]
            if cutlass.const_expr(T == KDA_CTAS):  # the owned row's output gate
                gate_owned = mine[384 + kt]
            for t in cutlass.range_constexpr(T):  # window: 3 taps, then the inputs
                for s in cutlass.range_constexpr(3):
                    for i in cutlass.range_constexpr(4):
                        w = t + i
                        term = taps[s, w] if w < 3 else inputs[3 * (w - 3) + s]
                        cf = Float32(coeff[s, i])
                        acc = accum[3 * t + s]
                        accum[3 * t + s] = fma(Float32(term), cf, acc)
            # KDA CTA j of the request activates and normalizes only token j's Q / K
            # channels and all-gathers them to the request's eight KDA CTAs.
            pre_q, pre_k = accum[0], accum[1]
            for t in cutlass.range_constexpr(1, T):
                pre_q = Float32(cutlass.select_(block == t, accum[3 * t], pre_q))
                pre_k = Float32(cutlass.select_(block == t, accum[3 * t + 1], pre_k))
            if kt < ROWS:
                for t in cutlass.range_constexpr(T):
                    shared_v[t * ROWS + kt] = accum[3 * t + 2]
            if cutlass.const_expr(T == KDA_CTAS):
                act = sigmoids([pre_q, pre_k, Float32(gate_owned)])
                q, k, out_gate = pre_q * act[0], pre_k * act[1], act[2]
            else:  # (owners gate their rows in ``owned_norm``)
                act = sigmoids([pre_q, pre_k])
                q, k = pre_q * act[0], pre_k * act[1]
            square_q, square_k = warp_sum(q * q), warp_sum(k * k)
            if lane == 0:
                reductions[warp] = square_q
                reductions[warp + 4] = square_k
            named_barrier(1, 128)
            if cutlass.const_expr(T * ROWS == 128):  # noqa: SIM114
                pre_v = shared_v[kt]
                shared_v[kt] = pre_v * sigmoids([pre_v])[0]
            elif kt < T * ROWS:  # (row kt % 16, token kt // 16)
                pre_v = shared_v[kt]
                shared_v[kt] = pre_v * sigmoids([pre_v])[0]
            qn = (reductions[0] + reductions[2]) + (reductions[1] + reductions[3])
            kn = (reductions[4] + reductions[6]) + (reductions[5] + reductions[7])
            fq = cute.math.rsqrt(qn + 1.0e-6, fastmath=True) * scale
            fk = cute.math.rsqrt(kn + 1.0e-6, fastmath=True)
            q_ptr = shared_q.iterator + block * 128 + kt
            k_ptr = shared_k.iterator + block * 128 + kt
            if cutlass.const_expr(T == KDA_CTAS):  # noqa: SIM114
                for dest in cutlass.range_constexpr(KDA_CTAS):
                    push_float(q_ptr, q * fq, bar_qk, first + dest)
                    push_float(k_ptr, k * fk, bar_qk, first + dest)
            elif block < T:  # KDA CTAs past the request's tokens normalize none
                for dest in cutlass.range_constexpr(KDA_CTAS):
                    push_float(q_ptr, q * fq, bar_qk, first + dest)
                    push_float(k_ptr, k * fk, bar_qk, first + dest)
            if kt < T:
                slot_rows[kt] = slots[request, kt]
            named_barrier(1, 128)  # shared_v, slot_rows
            wait(bar_qk)
            wait(bar_gate)
            lanes = (group, group_lane)
            chunks: cutlass.Constexpr = self.vkda.chunks
            buffers = (
                cute.make_rmem_tensor((3, chunks, 4), Float32),
                cute.make_rmem_tensor((3, chunks, 4), Float32),
            )
            for j in cutlass.range_constexpr(chunks):
                for e in cutlass.range_constexpr(4):
                    buffers[1][0, j, e] = Float32(0)
            sources = (shared_q, shared_k, shared_decay)
            self.load_operands(buffers[0], sources, 0, group_lane)
            for pair in cutlass.range(T // 2, unroll=1):
                for half in cutlass.range_constexpr(2):
                    t = 2 * pair + half
                    cur, other = buffers[half], buffers[1 - half]
                    v_row = shared_v[t * ROWS + group % ROWS]
                    beta = shared_beta[t]
                    out_prev = self.rolled_update(
                        history, cur, other, sources, t, group_lane, v_row, beta
                    )
                    if (group_lane == 0) & (t > 0):
                        local_o[(t - 1) * ROWS + group] = out_prev
                    slot_t = slot_rows[t]
                    self.store_checkpoint(
                        history, state, slot_t, head, value_base, lanes
                    )
            out_last = self.query_dot(history, buffers[1][0, None, None])
            if group_lane == 0:
                local_o[(T - 1) * ROWS + group] = out_last
            # thread kt pushes row kt % 16 of token kt // 16 to that token row's owner
            named_barrier(1, 128)
            position = self.vkda.output_position(value_base + kt % ROWS)
            if cutlass.const_expr(T == KDA_CTAS):
                push_float(
                    shared_o.iterator + position,
                    local_o[kt],
                    bar_out,
                    first + kt // ROWS,
                )
            elif kt < T * ROWS:  # token kt // 16's owner: token row T request + t
                owner = T * request + kt // ROWS
                push_float(shared_o.iterator + position, local_o[kt], bar_out, owner)
            if kt < ROWS:
                channel = 2 * dim + value_channel
                window = (
                    taps[2, 1],
                    taps[2, 2],
                    *(inputs[3 * t + 2] for t in range(T)),
                )
                for i, value in enumerate(window):
                    conv_page[channel, i] = value
            if cutlass.const_expr(T == KDA_CTAS):
                # token row ``rank`` (position ``block`` of this CTA's request)
                wait(bar_out)
                self.norm_row(kt, shared_o, output, rank, qk_channel, nw, out_gate, eps)
            if block == 0:
                for s in cutlass.range_constexpr(2):
                    channel = s * dim + qk_channel
                    window = (taps[s, 1], taps[s, 2])
                    window += tuple(inputs[3 * u + s] for u in range(T))
                    for i, value in enumerate(window):
                        conv_page[channel, i] = value

    @cute.jit
    def norm_row(self, kt, shared_o, output, row, qk_channel, nw, out_gate, eps):
        """The gated RMS norm of token ``row``'s outputs (landed in ``shared_o``),
        channel ``kt``."""
        lane = kt % 32
        raw_o = shared_o[self.vkda.output_position(kt)]
        sums = cute.make_rmem_tensor(4, Float32)
        for g in cutlass.range_constexpr(4):
            base = lane % 8 * 16 + g * 4
            o4 = load_f32x4(shared_o.iterator + base, shared=True)
            s0, s1, s2, s3 = (o * o for o in o4)
            sums[g] = warp_sum((s0 + s2) + (s1 + s3), 8)
        sumsq = (sums[0] + sums[2]) + (sums[1] + sums[3])
        rstd = cute.math.rsqrt(sumsq * (1.0 / 128) + eps, fastmath=True)
        result = raw_o * rstd * nw * out_gate
        if cutlass.const_expr(self.polled_out):
            store_bf16_canonical(at(output, row, qk_channel), result)
        else:
            output[row, qk_channel] = BFloat16(result)

    @cute.jit
    def owned_norm(self, kt, owned, slots, head, rank, eps):
        """At q < 8, token row ``rank`` (position rank % q of request rank // q, whose
        KDA CTAs may be other CTAs than this one's role): zeros for a padding request,
        else the gated RMS norm of the outputs its request's KDA CTAs pushed here."""
        mine, shared_o, bar_out, output, norm_w, accepted = owned
        T: cutlass.Constexpr = self.tokens
        request = rank // T
        qk_channel = head * 128 + kt
        if slots[request, accepted[request] - 1] <= 0:
            output[rank, qk_channel] = BFloat16(0)
        else:
            out_gate = sigmoids([Float32(mine[384 + kt])])[0]
            nw = norm_w[kt]
            wait(bar_out)
            self.norm_row(kt, shared_o, output, rank, qk_channel, nw, out_gate, eps)


@cache
def compile_head_multi(
    heads, bf16_state, has_bias, specs, arch, requests, hidden, polled=False, tokens=8
):  # fmt: skip
    """``KdaHeadMulti`` for ``specs`` (``PreparedKdaHead._tensors`` plus accepted) at
    ``requests`` x ``tokens``; ``polled``: the output is a polled buffer
    (``POLLED_OUT``'s protocol)."""
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    from ..common.host import fake

    def matrix(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    tensors = [fake(shape, stride, dtype) for shape, stride, dtype in specs]
    scalars = (Float32(-5), Float32(128**-0.5), Float32(1e-6))
    stages, tmem_tiles, early = MULTI_TILE[requests, tokens]
    kernel = KdaHeadMulti(
        heads, requests=requests, tokens=tokens, stages=stages, tmem_tiles=tmem_tiles,
        early_loads=early, hidden=hidden, bf16_state=bf16_state, has_bias=has_bias,
    )  # fmt: skip
    kernel.polled_out = polled
    args = (matrix(1), matrix(1), cutlass.Int64(0), *tensors, *scalars)
    args += (cutlass.Int64(0), Int32(0), cutlass.Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


# ``KdaHeadMulti``'s (ring stages, tensor-memory K tiles, A tiles before the PDL wait)
# per (requests, tokens): KdaHeadVerify's 15 / 21 with the ring cut to what fits beside
# an AttnRes CTA once x is resident for every token (2 x 8: 7 stages; the 21 early
# tiles still fit: 15 to tensor memory, 6 in the ring). 2 x 4 has the M8 head's x and
# keeps its tile (VERIFY_STAGES 10).
MULTI_TILE = {(2, 8): (7, 15, 21), (2, 4): (10, 15, 21)}
