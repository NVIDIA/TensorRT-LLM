# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# TensorRT-LLM local change: update imports for the Kimi K3 family layout.
"""KDA input projection and recurrence in one kernel (q = 1, B <= 8): one 16-CTA
cluster per head.

The head's Q, K, V and output-gate projection rows are eight 64-row M tiles, each
split over K by two CTAs exactly as ``SplitKGemm`` with the (64, 8, 2, stages) tactic
(so the rounded BF16 tiles are bit for bit the split-K projection's). The tile owners
hand the BF16 tiles over DSMEM (bulk copies) to the cluster's KDA CTAs instead of
storing them: CTA ``2 b + p`` runs ``KdaSplit``'s q = 1 recurrence (lanes 16, two CTAs
per head, register-resident state) for token b and value partition p. F_A and beta
come from a separate projection of the remaining rows (``fa_beta``); the gate warp
waits for it on a counter pair (``wait_counter``), re-armed by the last reader.

This removes the projection -> KdaSplit kernel boundary: the projection's output store,
the PDL completion wait and KdaSplit's reload of its inputs from L2.

The MMA warp tests the next stage's barrier between a stage's MMAs instead of a blocking
wait before them (a blocking wait drains the tensor pipe even when the stage is full),
and the tile owners push their rows as 16-byte ``st.async`` stores (bulk copies from one
warp serialize in the TMA unit). Neither changes an operation or its order.

Roles: warp 0 weight TMA (before the PDL wait), warp 1 activation TMA (after it),
warp 2 MMA, warp 3 the gate warp (F_B projection of its partition's decays, beta),
warps 4-7 the GEMM epilogue, then the KDA main threads.

``KdaHeadWide`` (Rubin TP16 B8) keeps the clusters and roles with 128-row tiles split
over K by four, A staged in tensor memory before the PDL wait and a one-hop hand-off
of FP32 partials (see its docstring)."""

from functools import cache
from typing import Any

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.arch import store_async_dsmem
from cutlass.cute.nvgpu import tcgen05

from ...k3r_attn_res.ull.kernel import BESIDE_SMEM
from ....ops.common.k3_ull_cute_host import architecture, layout, stream, to_cute
from ...k3r_common.ull.policy import batch_table
from ...k3r_common.ull.primitives import (
    asm,
    at,
    bits_float,
    float_bits,
    fma,
    load_f32x4,
    load_words,
    mma_into,
    named_barrier,
    push_float,
    sigmoid,
    store_words,
    wait,
    warp_sum,
    zeros,
)
from ...k3r_gemm.ull.projection import matrix_view, weight_map
from ...k3r_gemm.ull.splitk import (
    ATOM_K,
    SplitKGemm,
    SplitKTactic,
    _acquire_tensor_map,
    _fence_tcgen05,
    _smem,
    _store_shared_remote_v4,
    _tma_load_evict_first,
    _wait_cluster,
)
from .kernel import KdaSplit, load_activation_bf16x4, load_bf16x4

CLUSTER = 16  # 8 M tiles x split-K 2 = 8 tokens x 2 value partitions
# KdaHeadWide: warps issuing x's K tiles after the PDL wait (warp 1 and the four main
# warps; ``resident_b``)
B_ISSUERS = 5
# ``polled_out``: the sentinel of every BF16 element of a KDA layer's polled output (a
# NaN the canonical output store never writes: BF16 NaN results are stored as 0x7FFF).
OUT_SENT = 0xFFFF
INPUT_BYTES = (128 + 128 + 64 + 128) * 2  # Q, K, V (one partition), gate: BF16
GATE_BYTES = 128 * 4 + 4  # 128 decays (both partitions' gate warps) + beta


@asm()
def store_bf16_canonical(pointer, value):
    """BF16 (RN) of ``value`` at ``pointer``; a 0xFFFF NaN becomes 0x7FFF."""
    code = "{ .reg .b16 h; .reg .pred p; cvt.rn.bf16.f32 h,$1; "
    code += "setp.eq.b16 p,h,0xFFFF; @p mov.b16 h,0x7FFF; st.global.b16 [$0],h; }"
    return code, "l,f,~{memory}", (pointer, Float32(value))


@asm(Uint32)
def _acquire_load(address):
    return "ld.acquire.gpu.global.u32 $0,[$1];", "=r,l,~{memory}", (Int64(address),)


@asm(Uint32)
def _fetch_add(address):
    code = "atom.relaxed.gpu.global.add.u32 $0,[$1],1;"
    return code, "=r,l,~{memory}", (Int64(address),)


@asm()
def _store_zero(address):
    return "st.relaxed.gpu.global.u32 [$0],0;", "l,~{memory}", (Int64(address),)


@asm()
def _release_add(address):
    return "red.release.gpu.global.add.u32 [$0],1;", "l,~{memory}", (Int64(address),)


@asm(Int32)
def _test_wait(barrier, phase):
    """Non-blocking ``mbarrier`` phase test (acquire.cta): 1 once ``phase`` is done."""
    code = "{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p,[$1],$2; "
    code += "selp.u32 $0,1,0,p; }"
    return code, "=r,r,r,~{memory}", (barrier, Int32(phase))


# tcgen05 BF16 M128 N8 K16 (FP32 D, K-major A and B) and the upper words of the SW128
# K-major shared-memory descriptors (SBO 1024 / unused, version 1, 128-byte swizzle):
# the values cute emits for KdaHeadWide's tiles.
IDESC_M128_N8 = 0x08020490
A_DESC_HIGH, B_DESC_HIGH = 0x40004040, 0x40004000


@asm()
def _mma_smem(d, a_desc, b_desc, accumulate):
    """D[tmem] (+)= A[smem] B[smem]^T (low descriptor words)."""
    code = "{ .reg .pred p; .reg .b64 ad, bd; .reg .b32 ah, bh; setp.ne.b32 p, $3, 0; "
    code += f"mov.b32 ah, {A_DESC_HIGH}; mov.b32 bh, {B_DESC_HIGH}; "
    code += "mov.b64 ad, {$1, ah}; mov.b64 bd, {$2, bh}; "
    code += f"tcgen05.mma.cta_group::1.kind::f16 [$0], ad, bd, {IDESC_M128_N8}, p; }}"
    operands = (Int32(d), Int32(a_desc), Int32(b_desc), Int32(accumulate))
    return code, "r,r,r,r,~{memory}", operands


@asm()
def _mma_tmem(d, a, b_desc, accumulate):
    """D[tmem] (+)= A[tmem] B[smem]^T."""
    code = "{ .reg .pred p; .reg .b64 bd; .reg .b32 bh; setp.ne.b32 p, $3, 0; "
    code += f"mov.b32 bh, {B_DESC_HIGH}; mov.b64 bd, {{$2, bh}}; "
    code += f"tcgen05.mma.cta_group::1.kind::f16 [$0], [$1], bd, {IDESC_M128_N8}, p; }}"
    operands = (Int32(d), Int32(a), Int32(b_desc), Int32(accumulate))
    return code, "r,r,r,r,~{memory}", operands


@asm()
def _copy_to_tmem(t, a_desc):
    """128 rows x 32 bytes of A (one K16 slice) from shared to tensor memory."""
    code = f"{{ .reg .b64 sd; .reg .b32 sh; mov.b32 sh, {A_DESC_HIGH}; "
    code += "mov.b64 sd, {$1, sh}; tcgen05.cp.cta_group::1.128x256b [$0], sd; }"
    return code, "r,r,~{memory}", (Int32(t), Int32(a_desc))


class SignalGemm(SplitKGemm):
    """``SplitKGemm`` whose tile owners each add one to ``counter`` (gpu-scope release,
    after every epilogue thread's store): ``KdaHead``'s F_A / beta rows."""

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, a_map: cutlass.Int64,
        counter: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(a, b, c, a_map, counter).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, a_map: cutlass.Int64,
        counter: cutlass.Int64,
    ):  # fmt: skip
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        sA = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*shape))
        sB = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        mailbox = _smem(cutlass.Float32, self.mailbox_elements, 128)
        bar_reduce = _smem(cutlass.Int64, 1, 8).iterator
        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                cute.arch.mbarrier_init(bar_reduce, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        tiles = (split_rank * k_tile_count, k_tile_count)
        cute.arch.cluster_wait()
        if warp_idx == 0:
            self.weight_warp(bar_full, bar_empty, sA, a_map, bidx * self.cta_m, *tiles)
        elif warp_idx == 1:
            b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx))
            b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
            self.activation_warp(bar_full, bar_empty, b_tile, sB, b_map, *tiles)
        elif warp_idx == 2:
            k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
            self.mma_warp(
                bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA, sB,
                tmem_base_ptr, acc_layout, k_blocks, k_tile_count,
            )  # fmt: skip
        elif warp_idx >= 4:
            tile = cute.local_tile(mC, (self.cta_m, self.cta_n), (bidx, n_idx, l_idx))
            self.epilog_warp(
                bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, tile,
                cute.arch.thread_idx()[0] - 128, mC.element_type,
                utils.LayoutEnum.from_tensor(mC), mailbox, bar_reduce, split_rank,
            )  # fmt: skip
            if split_rank == 0:
                named_barrier(3, 128)  # every owner thread's output stores
                if cute.arch.thread_idx()[0] == 128:
                    cute.arch.fence_acq_rel_gpu()
                    _release_add(counter)


@cache
def compile_signal(tactic, arch):
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    def fake(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    kernel = SignalGemm(tactic=SplitKTactic(*tactic))
    args = (fake(1), fake(1), fake(0), cutlass.Int64(0), cutlass.Int64(0))
    stream_arg = make_fake_stream()
    return cute.experimental.compile(
        kernel, *args, stream_arg, options=f"--gpu-arch={arch}"
    )


class KdaHead(SplitKGemm):
    """``heads`` KDA heads per rank. F_A / beta come from ``fa_beta`` once ``counter``
    reaches ``target`` (the producer's tile count; 0: complete at launch)."""

    def __init__(self, heads, *, stages=11, bf16_state=True, has_bias=False):
        super().__init__(tactic=SplitKTactic(64, 8, 2, stages))
        self.heads, self.dim, self.has_bias = heads, heads * 128, has_bias
        self.cluster_shape = (CLUSTER, 1, 1)
        kda = KdaSplit(heads, 8, 0, has_bias, (1, 16, 0, 2), 1, bf16_state)
        kda.wave, kda.evict = kda.passes, True  # one wave; evict-first state stores
        self.kda = kda
        self.late_rearm = False  # re-arm F_A's counter after the decays (not before)
        self.fa_poll = False  # poll F_A's rows (sentinel), its counter as the fallback
        # ``polled_out`` (KdaHeadWide, Rubin TP16 B8, see ``POLLED_OUT``): the output
        # is a per-layer Lamport buffer the fused o-projection polls (every BF16
        # element starts as OUT_SENT); its elements are stored canonical (never
        # OUT_SENT) and warp 1 re-arms the previous KDA layer's buffer (``out_rearm``)
        # after its PDL wait.
        self.polled_out = False
        self.smem_bytes = utils.get_smem_capacity_in_bytes("sm_100")

    @cute.experimental.jit
    def __call__(
        self, mA: cute.Tensor, mB: cute.Tensor, a_map: cutlass.Int64,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        fa_beta: cute.Tensor, a_log: cute.Tensor, dt_bias: cute.Tensor,
        slots: cute.Tensor, state: cute.Tensor, output: cute.Tensor,
        norm_w: cute.Tensor, f_b: cute.Tensor, lower_bound: Float32, scale: Float32,
        eps: Float32, counter: cutlass.Int64, target: Int32, out_rearm: cutlass.Int64,
        stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(
            mA,
            mB,
            a_map,
            conv_w,
            bias,
            conv_state,
            fa_beta,
            a_log,
            dt_bias,
            slots,
            state,
            output,
            norm_w,
            f_b,
            lower_bound,
            scale,
            eps,
            counter,
            target,
            out_rearm,
        ).launch(
            grid=(CLUSTER, self.heads, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, a_map: cutlass.Int64,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        fa_beta: cute.Tensor, a_log: cute.Tensor, dt_bias: cute.Tensor,
        slots: cute.Tensor, state: cute.Tensor, output: cute.Tensor,
        norm_w: cute.Tensor, f_b: cute.Tensor, lower_bound: Float32, scale: Float32,
        eps: Float32, counter: cutlass.Int64, target: Int32, out_rearm: cutlass.Int64,
    ):  # fmt: skip
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        rank, head, _ = cute.arch.block_idx()
        tile, split_rank = rank % 8, rank // 8
        section = tile // 2  # Q, K, V, gate
        row0 = section * self.dim + head * 128 + (tile % 2) * 64
        token, partition = rank // 2, rank % 2
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tid, _, _ = cute.arch.thread_idx()

        shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        sA = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*shape))
        sB = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        mailbox = _smem(cutlass.Float32, self.mailbox_elements, 128)
        bar_reduce = _smem(cutlass.Int64, 1, 8).iterator
        # KDA: the owner's BF16 tile [token, 64 rows], the inputs [Q 128 | K 128 |
        # V 64 | gate 128], FP32 work rows and the input / gate / output barriers.
        tile_layout = cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m))
        tile_out = _smem(BFloat16, None, 128, tile_layout)
        x_in = _smem(BFloat16, INPUT_BYTES // 2, 128)
        shared_q = _smem(Float32, 128, 16)
        shared_k = _smem(Float32, 128, 16)
        shared_decay = _smem(Float32, 128, 16)
        shared_v = _smem(Float32, 64, 16)
        shared_o = _smem(Float32, 128, 16)
        reductions = _smem(Float32, 8, 16)
        shared_beta = _smem(Float32, 4, 16)
        bar_in = _smem(cutlass.Int64, 1, 8).iterator
        bar_gate = _smem(cutlass.Int64, 1, 8).iterator
        bar_out = _smem(cutlass.Int64, 1, 8).iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                cute.arch.mbarrier_init(bar_reduce, 1)
        if tid == 128:
            cute.arch.mbarrier_init(bar_in, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(bar_in, INPUT_BYTES)
            cute.arch.mbarrier_init(bar_gate, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(bar_gate, GATE_BYTES)
            if partition == 0:
                cute.arch.mbarrier_init(bar_out, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bar_out, 512)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()  # every barrier before any DSMEM traffic

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        tiles = (split_rank * k_tile_count, k_tile_count)
        slot = slots[token]
        if warp_idx < 3:
            cute.arch.cluster_wait()
            if warp_idx == 0:
                self.weight_warp(bar_full, bar_empty, sA, a_map, row0, *tiles)
            elif warp_idx == 1:
                b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (0, None, 0))
                b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
                self.activation_warp(bar_full, bar_empty, b_tile, sB, b_map, *tiles)
            else:
                k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
                self.mma_warp(
                    bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA,
                    sB, tmem_base_ptr, acc_layout, k_blocks, tiles[1],
                )  # fmt: skip
            cute.arch.cluster_arrive_relaxed()
        elif warp_idx == 3:
            gate_coords = (head, token, partition, rank, slot)
            gate_args = (fa_beta, a_log, dt_bias, f_b, lower_bound, counter, target)
            self.gate(gate_args, gate_coords, shared_decay, shared_beta, bar_gate)
            cute.arch.cluster_arrive_relaxed()
        else:
            kt = tid - 128
            tensors = (conv_w, bias, conv_state, state, output, norm_w)
            coords = (head, token, partition, slot)
            early = self.main_early(kt, tensors, coords)
            cute.arch.cluster_wait()
            self.epilogue(
                kt, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, mailbox,
                bar_reduce, split_rank, tile, tile_out, x_in, bar_in,
            )  # fmt: skip
            wait(bar_in)
            cute.arch.cluster_arrive_relaxed()  # this CTA's input copies are complete
            rows = (shared_q, shared_k, shared_decay, shared_v, shared_o, reductions)
            bars = (bar_gate, bar_out, shared_beta)
            self.main(kt, early, x_in, rows, bars, tensors, coords, scale, eps)
        cute.arch.cluster_wait()  # no CTA leaves while a peer may read its tile

    @cute.experimental.jit
    def weight_warp(
        self, bar_full, bar_empty, s_tile: cute.Tensor, a_map: cutlass.Int64,
        row0: cutlass.Int32, k_tile_start: cutlass.Int32, k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """``SplitKGemm.weight_warp`` without the dependents' trigger: the gate warp
        releases them once F_A is ready, so they never hold the SMs its projection
        needs."""
        stages = self.num_ab_stage
        stage_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        _acquire_tensor_map(a_map)
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, stage_bytes)
            with cute.arch.elect_one():
                k0 = (k_tile_start + k_tile) * self.cta_k
                tile = s_tile[None, None, None, stage].iterator
                for atom in cutlass.range_constexpr(self.cta_k // ATOM_K):
                    box, mbar = tile + atom * self.cta_m * ATOM_K, bar_full + stage
                    k = k0 + atom * ATOM_K
                    _tma_load_evict_first(box, mbar, a_map, k, row0)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1
        self.drain(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def mma_warp(
        self, bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc,
        tiled_mma: cute.TiledMma, sA: cute.Tensor, sB: cute.Tensor, tmem_base_ptr,
        acc_layout: cutlass.Constexpr, mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """``SplitKGemm.mma_warp`` (the same MMAs in the same order) that tests the
        next stage's full barrier after half of this stage's MMAs and waits only if
        it is not complete yet."""
        num_tmem_cols, stages = 256, self.num_ab_stage
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=False)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        full_phase, ready = cutlass.Int32(0), Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            if ready == 0:
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)
            _fence_tcgen05()
            for k_block in cutlass.range_constexpr(self.cta_k // 16):
                accumulate = k_tile != 0 if k_block == 0 else True
                mma_atom.set(tcgen05.Field.ACCUMULATE, accumulate)
                cute_ext.dot(
                    mma_atom,
                    cute.append_ones(sA[None, None, k_block, stage], up_to_rank=3),
                    cute.append_ones(sB[None, None, k_block, stage], up_to_rank=3),
                    accumulator,
                )
                if cutlass.const_expr(k_block == self.cta_k // 32 - 1):
                    phase = full_phase ^ Int32(stage == stages - 1)
                    ready = _test_wait(bar_full + (k_tile + 1) % stages, phase)
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
            if stage == stages - 1:
                full_phase = full_phase ^ 1

        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)  # the release lands in phase 1
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        _fence_tcgen05()
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=False)

    @cute.jit
    def gate(self, args, coords, shared_decay, shared_beta, barrier):
        """``KdaSplit.gate_warp`` (mode 0, two decays per lane, 16-byte F_B rows) with
        the F_B rows in registers before the cluster sync, F_A from ``fa_beta`` after
        the counter, and beta's sigmoid sent to this CTA with the decays."""
        fa_beta, a_log, dt_bias, weight, lower_bound, counter, target = args
        head, token, partition, rank, slot = coords
        lane = cute.arch.lane_idx()
        group, quarter = lane >> 2, lane & 3
        feature = 16 * (quarter >> 1) + group + 8 * (quarter & 1)
        value_base = partition * 64
        a = a_log[head]
        dt = [dt_bias[head * 128 + value_base + 32 * r + feature] for r in range(2)]
        # 8-byte F_B rows per chunk (chunk c covers columns 16 c + 4 quarter + [0, 4)):
        # the K order of the four-CTA KdaSplit schedule the BF16 state uses at B5-8.
        wr = cute.make_rmem_tensor((4, 8, 4), Uint32)
        for tile in cutlass.range_constexpr(4):
            row = head * 128 + value_base + 16 * tile + group
            for chunk in cutlass.range_constexpr(8):
                col = 16 * chunk + 4 * quarter
                w_low = load_bf16x4(at(weight, row, col))
                w_high = load_bf16x4(at(weight, row + 8, col))
                wr[tile, chunk, 0], wr[tile, chunk, 2] = w_low
                wr[tile, chunk, 1], wr[tile, chunk, 3] = w_high
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.fa_poll):
            ar, beta = self.poll_fa(fa_beta, counter, target, token, quarter, head)
            cute.arch.griddepcontrol_launch_dependents()
        else:
            if target > 0:  # every CTA, so the last reader re-arms after the producer
                if lane == 0:
                    seen = Int32(_acquire_load(counter))
                    while seen < target:
                        seen = Int32(_acquire_load(counter))
                cute.arch.sync_warp()
            cute.arch.griddepcontrol_launch_dependents()
            ar = [
                load_activation_bf16x4(at(fa_beta, token, 16 * c + 4 * quarter))
                for c in range(8)
            ]
            beta = fa_beta[token, 128 + head]
        if cutlass.const_expr(not self.late_rearm):
            self.rearm(counter, target, lane)
        if slot > 0:
            acc = zeros((4, 4), Float32)
            for chunk in cutlass.range_constexpr(8):
                for tile in cutlass.range_constexpr(4):
                    wa = tuple(wr[tile, chunk, e] for e in range(4))
                    mma_into(acc[tile, None], wa, ar[chunk], bf16=True)
            decay = []
            for r in cutlass.range_constexpr(2):
                selected = acc[2 * r, 0]
                if quarter & 1:
                    selected = acc[2 * r, 2]
                if quarter >> 1:
                    selected = acc[2 * r + 1, 0]
                    if quarter & 1:
                        selected = acc[2 * r + 1, 2]
                raw_g = Float32(BFloat16(selected))
                gate = cute.math.exp(a, fastmath=True) * (raw_g + dt[r])
                gate = sigmoid(gate, False)
                decay.append(cute.math.exp(lower_bound * gate, fastmath=True))
            packet, destination = lane % 16, lane // 16
            values = cute.make_rmem_tensor(4, Float32)
            for e in cutlass.range_constexpr(4):
                source = (4 * packet + e) & 31
                source = (source & 7) << 2 | (source >> 4) << 1 | source >> 3 & 1
                low = cute.arch.shuffle_sync(decay[0], source)
                high = cute.arch.shuffle_sync(decay[1], source)
                values[e] = low
                if packet >= 8:
                    values[e] = high
            offset = shared_decay.iterator + value_base + 4 * packet
            pointer = cute.make_ptr(
                Float32, offset.llvm_ptr, cute.AddressSpace.smem, assumed_align=16
            )
            words = tuple(float_bits(values[e]) for e in range(4))
            store_async_dsmem(pointer, words, barrier, 2 * token + destination)
            if lane == 0:
                beta_gate = sigmoid(Float32(beta), False)
                push_float(shared_beta.iterator, beta_gate, barrier, rank)
        if cutlass.const_expr(self.late_rearm):
            self.rearm(counter, target, lane, fa_beta)

    @cute.jit
    def poll_fa(self, fa_beta, counter, target, token, quarter, head):
        """This lane's F_A words (8 x 2) and beta: F_A's rows are polled directly
        until no BF16 half is the 0xFFFF sentinel the last reader re-fills (one round
        trip less than the counter, then the row). A genuine 0xFFFF value cannot
        hang: once F_A's counter completes the row is reloaded after the acquire."""
        lane = cute.arch.lane_idx()
        words = cute.make_rmem_tensor(17, Uint32)
        beta_word = at(fa_beta, token, 128 + (head - head % 2))
        state = Int32(2)
        if target > 0:
            state = Int32(0)
        while state == 0:
            bad = False
            for c in cutlass.range_constexpr(8):
                source = at(fa_beta, token, 16 * c + 4 * quarter)
                pair = load_words(source, count=2, word_type="u32", scope="gpu")
                words[2 * c], words[2 * c + 1] = pair
            words[16] = load_words(beta_word, count=1, word_type="u32", scope="gpu")[0]
            for i in cutlass.range_constexpr(17):
                low, high = words[i] & Uint32(0xFFFF), words[i] >> Uint32(16)
                bad |= (low == Uint32(0xFFFF)) | (high == Uint32(0xFFFF))
            if cute.arch.vote_all_sync(not bad):
                state = Int32(1)
            else:
                done = Int32(0)
                if lane == 0:
                    done = Int32(Int32(_acquire_load(counter)) >= target)
                if cute.arch.shuffle_sync(done, 0) != 0:
                    state = Int32(2)
        if state == 2:
            for c in cutlass.range_constexpr(8):
                source = at(fa_beta, token, 16 * c + 4 * quarter)
                words[2 * c], words[2 * c + 1] = load_activation_bf16x4(source)
            words[16] = load_words(beta_word, count=1, word_type="u32", mutable=True)[0]
        ar = [(words[2 * c], words[2 * c + 1]) for c in range(8)]
        half = (words[16] >> Uint32(16 * (head % 2))) & Uint32(0xFFFF)
        return ar, BFloat16(bits_float(half << Uint32(16)))

    @cute.jit
    def rearm(self, counter, target, lane, fa_beta=None):
        """The last of the head CTAs past the F_A counter re-arms the pair (and, with
        ``fa_poll``, once F_A's counter is complete, re-fills its rows with the
        sentinel)."""
        last = Int32(0)
        if (target > 0) & (lane == 0):
            readers = Int32(CLUSTER * self.heads)
            if Int32(_fetch_add(counter + 4)) == readers - 1:
                last = Int32(1)
                if cutlass.const_expr(self.fa_poll):
                    # readers may have seen F_A's rows before its counter adds landed
                    seen = Int32(_acquire_load(counter))
                    while seen < target:
                        seen = Int32(_acquire_load(counter))
                _store_zero(counter)
                _store_zero(counter + 4)
        if cutlass.const_expr(self.fa_poll):
            self.refill_fa(fa_beta, cute.arch.shuffle_sync(last, 0), lane)

    @cute.jit
    def refill_fa(self, fa_beta, last, lane):
        """The last reader writes the 0xFFFF sentinel over F_A's rows again."""
        if last != 0:
            rows, pitch = fa_beta.layout.shape[0], fa_beta.layout.stride[0] // 2
            block = cute.make_tensor(
                cute.recast_ptr(fa_beta.iterator, dtype=Uint32),
                cute.make_layout(rows * pitch),
            )
            for i in cutlass.range(lane, rows * pitch, 32, unroll=1):
                block[i] = Uint32(0xFFFFFFFF)

    @cute.jit
    def main_early(self, kt, tensors, coords):
        """Loads that do not depend on this step's projection, before the cluster
        sync: state rows (registers), conv taps and weights, the norm weight."""
        dim = self.dim
        conv_w, bias, conv_state, state, output, norm_w = tensors
        head, token, partition, slot = coords
        group, group_lane = kt // self.kda.lanes_per_row, kt % self.kda.lanes_per_row
        value_base, qk_channel = partition * 64, head * 128 + kt
        value_channel = head * 128 + value_base + kt
        safe = slot
        if safe < 0:
            safe = Int32(0)
        state_page = state[Int64(safe), None, None, None]
        conv_page = conv_state[Int64(safe), None, None]
        state_row = lambda r, c: at(state_page, head, value_base + r, c)
        history = zeros((self.kda.passes, self.kda.chunks, 4), Float32)
        for p in cutlass.range_constexpr(self.kda.passes):
            row = p * self.kda.groups + group
            for j in cutlass.range_constexpr(self.kda.chunks):
                column = 4 * (group_lane + self.kda.lanes_per_row * j)
                values = self.kda.load_state(state_row(row, column))
                for e in cutlass.range_constexpr(4):
                    history[p, j, e] = values[e]
        taps = zeros((3, 3), BFloat16)
        coeff = zeros((3, 4), conv_w.element_type)
        accum = zeros(3, Float32)
        for s in cutlass.range_constexpr(3):
            channel = value_channel if s == 2 else qk_channel
            if (s != 2) | (kt < 64):
                for t in cutlass.range_constexpr(3):
                    taps[s, t] = conv_page[s * dim + channel, t]
                for t in cutlass.range_constexpr(4):
                    coeff[s, t] = conv_w[s, t, channel]
                if cutlass.const_expr(self.has_bias):
                    accum[s] = bias[s * dim + channel]
        return history, taps, coeff, accum, norm_w[kt]

    @cute.jit
    def read_inputs(self, kt, x_in):
        """This thread's projected Q, K, V (its partition's rows) and gate values."""
        inputs = cute.make_rmem_tensor(3, BFloat16)
        inputs[0], inputs[1], inputs[2] = x_in[kt], x_in[128 + kt], BFloat16(0)
        if kt < 64:
            inputs[2] = x_in[256 + kt]
        return inputs, x_in[320 + kt]

    @cute.jit
    def main(self, kt, early, x_in, rows, bars, tensors, coords, scale, eps):
        """``KdaSplit``'s q = 1 main threads on the DSMEM inputs."""
        dim = self.dim
        history, taps, coeff, accum, nw = early
        shared_q, shared_k, shared_decay, shared_v, shared_o, reductions = rows
        bar_gate, bar_out, shared_beta = bars
        conv_w, bias, conv_state, state, output, norm_w = tensors
        head, token, partition, slot = coords
        lane, warp = kt % 32, kt // 32
        group, group_lane = kt // self.kda.lanes_per_row, kt % self.kda.lanes_per_row
        value_base, qk_channel = partition * 64, head * 128 + kt
        value_channel = head * 128 + value_base + kt
        safe = slot
        if safe < 0:
            safe = Int32(0)
        state_page = state[Int64(safe), None, None, None]
        conv_page = conv_state[Int64(safe), None, None]
        state_row = lambda r, c, s=state_page: at(s, head, value_base + r, c)
        if slot > 0:
            inputs, gate_in = self.read_inputs(kt, x_in)
            for s in cutlass.range_constexpr(3):
                for i in cutlass.range_constexpr(4):
                    term = Float32(taps[s, i] if i < 3 else inputs[s])
                    accum[s] = fma(term, Float32(coeff[s, i]), accum[s])
            q = accum[0] * sigmoid(accum[0], False)
            k = accum[1] * sigmoid(accum[1], False)
            v = accum[2] * sigmoid(accum[2], False)
            out_gate = sigmoid(Float32(gate_in), False)
            shared_q[kt] = q
            shared_k[kt] = k
            if kt < 64:
                shared_v[kt] = v
            squares = (warp_sum(q * q), warp_sum(k * k))
            if lane == 0:
                reductions[warp] = squares[0]
                reductions[warp + 4] = squares[1]
            named_barrier(1, 128)
            r = lambda i: reductions[i]
            qn = (r(0) + r(2)) + (r(1) + r(3))
            kn = (r(4) + r(6)) + (r(5) + r(7))
            fq = cute.math.rsqrt(qn + 1.0e-6, fastmath=True) * scale
            fk = cute.math.rsqrt(kn + 1.0e-6, fastmath=True)
            qr = cute.make_rmem_tensor((self.kda.chunks, 4), Float32)
            kr = cute.make_rmem_tensor((self.kda.chunks, 4), Float32)
            for j in cutlass.range_constexpr(self.kda.chunks):
                column = 4 * (group_lane + self.kda.lanes_per_row * j)
                q4 = load_f32x4(shared_q.iterator + column, shared=True)
                k4 = load_f32x4(shared_k.iterator + column, shared=True)
                for e in cutlass.range_constexpr(4):
                    qr[j, e] = q4[e] * fq
                    kr[j, e] = k4[e] * fk
            wait(bar_gate)
            decays = []
            for j in cutlass.range_constexpr(self.kda.chunks):
                column = 4 * (group_lane + self.kda.lanes_per_row * j)
                decays.append(load_f32x4(shared_decay.iterator + column, shared=True))
            beta = shared_beta[0]
            dots = cute.make_rmem_tensor(self.kda.passes, Float32)
            current = (kr, qr, decays, shared_v, beta)
            lanes = (group, group_lane)
            self.kda.update(history, dots, current, lanes, (0, self.kda.passes), None)
            owner = 2 * token
            for p in cutlass.range_constexpr(self.kda.passes):
                row = p * self.kda.groups + group
                if group_lane == 0:
                    position = self.kda.output_position(value_base + row)
                    dst = shared_o.iterator + position
                    store_async_dsmem(dst, float_bits(dots[p]), bar_out, owner)
            self.kda.store_rows(history, state_row, (0, self.kda.passes), lanes)
            if kt < 64:
                window = (taps[2, 1], taps[2, 2], inputs[2])
                for i, value in enumerate(window):
                    conv_page[2 * dim + value_channel, i] = value
            if partition == 0:
                wait(bar_out)
                raw_o = shared_o[self.kda.output_position(kt)]
                sums = cute.make_rmem_tensor(4, Float32)
                for g in cutlass.range_constexpr(4):
                    base = lane % 8 * 16 + g * 4
                    o4 = load_f32x4(shared_o.iterator + base, shared=True)
                    s0, s1, s2, s3 = (o * o for o in o4)
                    sums[g] = warp_sum((s0 + s2) + (s1 + s3), 8)
                sumsq = (sums[0] + sums[2]) + (sums[1] + sums[3])
                rstd = cute.math.rsqrt(sumsq * (1.0 / 128) + eps, fastmath=True)
                value = raw_o * rstd * nw * out_gate
                if cutlass.const_expr(self.polled_out):
                    store_bf16_canonical(at(output, token, qk_channel), value)
                else:
                    output[token, qk_channel] = BFloat16(value)
                for s in cutlass.range_constexpr(2):
                    window = (taps[s, 1], taps[s, 2], inputs[s])
                    for i, value in enumerate(window):
                        conv_page[s * dim + qk_channel, i] = value
        else:
            if partition == 0:
                output[token, qk_channel] = BFloat16(0)

    @cute.experimental.jit
    def epilogue(
        self, epi_tid: cutlass.Int32, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        acc_layout: cutlass.Constexpr, mailbox, bar_reduce, split_rank: cutlass.Int32,
        tile: cutlass.Int32, tile_out: cute.Tensor, x_in: cute.Tensor, bar_in,
    ):  # fmt: skip
        """``SplitKGemm.epilog_warp`` with the owner's BF16 tile staged in shared
        memory and pushed to the KDA CTAs instead of stored."""
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()
        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr), acc_layout
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        d_layout = utils.LayoutEnum.from_tensor(tile_out)
        load_op = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k), d_layout, BFloat16, self.acc_dtype,
            epi_tile, False,
        )  # fmt: skip
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(load_op, acc_view)
        gD_epi = cute.flat_divide(tile_out, epi_tile)
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rmem = cute.AddressSpace.rmem
        rAcc = cute_ext.allocate(self.acc_dtype, rmem, rmem_layout, alignment=32)
        rD = cute_ext.allocate(BFloat16, rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _fence_tcgen05()
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        vpt = cutlass.const_expr(cute.size(rmem_layout))
        if split_rank != 0:
            for i in cutlass.range_constexpr(0, vpt, 4):
                words = (rAcc[i], rAcc[i + 1], rAcc[i + 2], rAcc[i + 3])
                box = mailbox.iterator + epi_tid * vpt + i
                _store_shared_remote_v4(*words, box, bar_reduce, tile)
        else:
            if epi_tid == 0:
                expected = self.expected_transaction_bytes
                cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, expected)
            _wait_cluster(bar_reduce, Int32(0))
            for i in cutlass.range_constexpr(vpt):
                rAcc[i] = rAcc[i] + mailbox[epi_tid * vpt + i]
            rD.store(rAcc.load().to(BFloat16))
            cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])
            named_barrier(3, 128)
            # Token b's 64 rows (128 bytes) to the KDA CTAs that read them, 16 bytes
            # per thread: Q / K / gate rows to both CTAs of the token, V rows to one.
            section, half = tile // 2, tile % 2
            token, chunk = epi_tid // 16, epi_tid % 8
            peer, column = epi_tid // 8, 128 * section - 64 * (section // 3) + 64 * half
            if section == 2:
                token, peer, column = epi_tid // 8, 2 * (epi_tid // 8) + half, 256
            if (section != 2) | (epi_tid < 64):
                src = tile_out.iterator + cute.assume(64 * token + 8 * chunk, divby=8)
                words = load_words(src, count=4, shared=True)
                dst = x_in.iterator + cute.assume(column + 8 * chunk, divby=8)
                smem = cute.AddressSpace.smem
                dst = cute.make_ptr(Uint8, dst.llvm_ptr, smem, assumed_align=16)
                store_async_dsmem(dst, words, bar_in, peer)


SECTIONS = 4  # Q, K, V, gate: one 128-row M tile each in ``KdaHeadWide``


class KdaHeadWide(KdaHead):
    """``KdaHead`` with 128-row M tiles split over K by four CTAs (Rubin): CTA
    ``t + 4 s`` computes section t (Q, K, V, gate) of the head over K quarter s with
    M128 MMAs (~18% more of A per tensor-core cycle than M64) and pushes its FP32
    partial of each KDA CTA's rows to that CTA (16 bytes = 4 rows per ``st.async``, one
    DSMEM hop); the KDA CTA sums the four in rank order, ((p0 + p1) + p2) + p3 as
    ``SplitKGemm``'s split-K 4 owner does, and rounds to BF16. The inputs are bitwise
    the split-K projection's with the (128, 8, 4) tactic: an accumulation reorder of
    the (64, 8, 2) tactic of ``KdaHead`` (FP32 partials over K/4 instead of K/2).

    It asks only for the shared memory it uses, so a CTA can be placed beside a small
    AttnRes CTA (see ``HEAD_BEFORE_CARVEOUT`` in ``runtime/providers.py``).

    ``tmem_tiles`` > 0: before the PDL wait the MMA warp copies K tiles 0 ..
    tmem_tiles - 1 from the A ring to tensor memory (``tcgen05.cp``), which frees
    their ring stages for the next tiles, so more of A is on chip when x lands; their
    MMAs then read A from tensor memory (~4x the M128 rate from shared memory). B (x)
    is then resident for every K tile. Same MMAs in the same order. ``early_loads``
    caps the A tiles requested before the PDL wait.

    The gate warp re-arms F_A's counter after sending its decays (off the decays'
    path); a counter pair must then not be reused before a graph boundary (the model
    has one pair per layer)."""

    def __init__(
        self, heads, *, stages=12, cta_k=64, tmem_tiles=0, early_loads=None,
        bf16_state=True, has_bias=False, hidden=7168,
    ):  # fmt: skip
        super().__init__(heads, stages=stages, bf16_state=bf16_state, has_bias=has_bias)
        SplitKGemm.__init__(self, tactic=SplitKTactic(128, 8, SECTIONS, stages))
        self.cluster_shape = (CLUSTER, 1, 1)
        self.cta_k, self.late_rearm, self.fa_poll = cta_k, True, True
        self.k_tiles, self.tmem_tiles = hidden // cta_k // SECTIONS, tmem_tiles
        # A tiles the weight warp issues before the PDL wait (None: as the ring allows)
        self.early_loads = early_loads
        assert 8 + tmem_tiles * cta_k // 2 <= 512 and tmem_tiles <= self.k_tiles
        # Only the shared memory the kernel uses (+ alignment slack): a CTA then fits
        # beside a small-smem CTA (the AttnRes at carve-out 100) on one SM.
        b_tiles = self.k_tiles if tmem_tiles else stages
        used = stages * 128 * cta_k * 2 + b_tiles * 8 * cta_k * 2
        used += (2 * stages + (b_tiles if tmem_tiles else 1) + 3) * 8 + 4
        used += 128 * 8 * 4 + SECTIONS * INPUT_BYTES * 2 + (3 * 128 + 64 + 128) * 4
        used += 8 * 4 + 4 * 4 + 3 * 8
        self.smem_used = used
        self.smem_bytes = (used + 2560 + 1023) // 1024 * 1024
        assert self.smem_bytes <= BESIDE_SMEM  # beside an AttnRes CTA

    @cute.experimental.jit
    def weight_warp(
        self, bar_full, bar_empty, s_tile: cute.Tensor, a_map: cutlass.Int64,
        row0: cutlass.Int32, k_tile_start: cutlass.Int32, k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """``KdaHead.weight_warp``; with ``early_loads``, tiles from that index on
        are issued only after the PDL wait (the predecessor has completed)."""
        stages = self.num_ab_stage
        stage_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        early = self.k_tiles if self.early_loads is None else self.early_loads
        _acquire_tensor_map(a_map)
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            if k_tile == early:
                cute.arch.griddepcontrol_wait()
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, stage_bytes)
            with cute.arch.elect_one():
                k0 = (k_tile_start + k_tile) * self.cta_k
                tile = s_tile[None, None, None, stage].iterator
                for atom in cutlass.range_constexpr(self.cta_k // ATOM_K):
                    box, mbar = tile + atom * self.cta_m * ATOM_K, bar_full + stage
                    k = k0 + atom * ATOM_K
                    _tma_load_evict_first(box, mbar, a_map, k, row0)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1
        self.drain(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, a_map: cutlass.Int64,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        fa_beta: cute.Tensor, a_log: cute.Tensor, dt_bias: cute.Tensor,
        slots: cute.Tensor, state: cute.Tensor, output: cute.Tensor,
        norm_w: cute.Tensor, f_b: cute.Tensor, lower_bound: Float32, scale: Float32,
        eps: Float32, counter: cutlass.Int64, target: Int32, out_rearm: cutlass.Int64,
    ):  # fmt: skip
        stages, staged = self.num_ab_stage, self.tmem_tiles > 0
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
        token, partition = rank // 2, rank % 2
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tid, _, _ = cute.arch.thread_idx()

        a_shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        b_shape = (tiled_mma, mnk_tiler, ab_dtype, self.k_tiles if staged else stages)
        sA = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*a_shape))
        sB = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*b_shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        b_full = _smem(cutlass.Int64, self.k_tiles if staged else 1, 8).iterator
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        # This CTA's FP32 partial [128 rows, 8 tokens]; every split rank's FP32 inputs
        # of this CTA's token [4][Q 128 | K 128 | V 64 | gate 128].
        tile_layout = cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m))
        partial = _smem(Float32, None, 128, tile_layout)
        x_in = _smem(Float32, SECTIONS * INPUT_BYTES // 2, 128)
        shared_q = _smem(Float32, 128, 16)
        shared_k = _smem(Float32, 128, 16)
        shared_decay = _smem(Float32, 128, 16)
        shared_v = _smem(Float32, 64, 16)
        shared_o = _smem(Float32, 128, 16)
        reductions = _smem(Float32, 8, 16)
        shared_beta = _smem(Float32, 4, 16)
        bar_in = _smem(cutlass.Int64, 1, 8).iterator
        bar_gate = _smem(cutlass.Int64, 1, 8).iterator
        bar_out = _smem(cutlass.Int64, 1, 8).iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 1 if staged else 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                if cutlass.const_expr(staged):
                    for i in range(self.k_tiles):
                        cute.arch.mbarrier_init(b_full + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
        if tid == 128:
            cute.arch.mbarrier_init(bar_in, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(bar_in, SECTIONS * INPUT_BYTES * 2)
            cute.arch.mbarrier_init(bar_gate, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(bar_gate, GATE_BYTES)
            if partition == 0:
                cute.arch.mbarrier_init(bar_out, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bar_out, 512)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()  # every barrier before any DSMEM traffic

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        tiles = (split_rank * k_tile_count, k_tile_count)
        slot = slots[token]
        if warp_idx < 3:
            cute.arch.cluster_wait()
            if warp_idx == 0:
                self.weight_warp(bar_full, bar_empty, sA, a_map, row0, *tiles)
            elif warp_idx == 1:
                b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (0, None, 0))
                b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
                if cutlass.const_expr(staged):
                    self.resident_b(b_full, b_tile, sB, b_map, tiles[0], 0)
                else:
                    self.activation_warp(bar_full, bar_empty, b_tile, sB, b_map, *tiles)
                if cutlass.const_expr(self.polled_out):  # after the PDL wait above
                    self.rearm_out(out_rearm, output.shape[0], head * CLUSTER + rank)
            elif cutlass.const_expr(staged):
                self.staged_mma_warp(
                    bar_full, bar_empty, b_full, bar_mma_epilog, bar_tmem_alloc, sA, sB,
                    tmem_base_ptr,
                )  # fmt: skip
            else:
                k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
                self.mma_warp(
                    bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA,
                    sB, tmem_base_ptr, acc_layout, k_blocks, tiles[1],
                )  # fmt: skip
            cute.arch.cluster_arrive_relaxed()
        elif warp_idx == 3:
            gate_coords = (head, token, partition, rank, slot)
            gate_args = (fa_beta, a_log, dt_bias, f_b, lower_bound, counter, target)
            self.gate(gate_args, gate_coords, shared_decay, shared_beta, bar_gate)
            cute.arch.cluster_arrive_relaxed()
        else:
            kt = tid - 128
            tensors = (conv_w, bias, conv_state, state, output, norm_w)
            coords = (head, token, partition, slot)
            early = self.main_early(kt, tensors, coords)
            cute.arch.cluster_wait()
            if cutlass.const_expr(staged):  # the other B_ISSUERS - 1 warps of x's tiles
                b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (0, None, 0))
                b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
                self.resident_b(b_full, b_tile, sB, b_map, tiles[0], warp_idx - 3)
            self.push_partials(
                kt, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, partial,
                section, split_rank, x_in, bar_in,
            )  # fmt: skip
            wait(bar_in)
            cute.arch.cluster_arrive_relaxed()  # this CTA's input copies are complete
            rows = (shared_q, shared_k, shared_decay, shared_v, shared_o, reductions)
            bars = (bar_gate, bar_out, shared_beta)
            self.main(kt, early, x_in, rows, bars, tensors, coords, scale, eps)
        cute.arch.cluster_wait()  # no CTA leaves while a peer may write its inputs

    @cute.experimental.jit
    def rearm_out(self, address: cutlass.Int64, rows, cta):
        """``polled_out``: OUT_SENT over the previous KDA layer's [rows, dim] output
        buffer, 16 bytes per lane, the chunks spread over the grid's CTAs. Called by
        warp 1 after its ``griddepcontrol.wait`` (see ``POLLED_OUT`` for the order)."""
        lane = cute.arch.lane_idx()
        ctas: cutlass.Constexpr = CLUSTER * self.heads
        chunks = rows * (self.dim // 8)
        chunk = cta + ctas * lane
        if chunk < chunks:
            sentinel = Uint32(OUT_SENT | OUT_SENT << 16)
            store_words(address + Int64(chunk) * 16, (sentinel,) * 4)

    @cute.experimental.jit
    def resident_b(
        self, b_full, g_tile: cute.Tensor, s_tile: cute.Tensor, cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32, first: cutlass.Int32,
    ):  # fmt: skip
        """K tiles ``first``, ``first`` + B_ISSUERS, ... of B (the activations) after
        the PDL wait, each on its own barrier (``tmem_tiles`` > 0). B_ISSUERS warps
        split x's tiles: one warp issues a TMA every ~55 cycles, so x's last
        tensor-memory tiles would otherwise land after the MMAs reach them."""
        tile_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        cute.arch.griddepcontrol_wait()
        for k_tile in cutlass.range(first, self.k_tiles, B_ISSUERS, unroll=1):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(b_full + k_tile, tile_bytes)
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, k_tile],
                (b_full + k_tile).value,
                cta_v_map=cta_v_map,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )

    @cute.experimental.jit
    def staged_mma_warp(
        self, bar_full, bar_empty, b_full, bar_mma_epilog, bar_tmem_alloc,
        sA: cute.Tensor, sB: cute.Tensor, tmem_base_ptr,
    ):  # fmt: skip
        """K tiles 0 .. tmem_tiles - 1 to tensor memory as they land (their ring
        stages refill), then every tile's MMAs in K order: A from tensor memory for
        those, from the ring for the rest. Raw tcgen05 with the descriptors cute emits
        for these SW128 K-major tiles; the accumulator is the first 8 columns."""
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

        # The ring's tiles run as a rolled loop: fully unrolled over the K tiles, the
        # issue code (~1.7k instructions with the staging and tensor-memory tiles)
        # missed in the instruction cache inside the decode chain, and the
        # shared-memory MMAs ran ~30% under their rate. The short tensor-memory loops
        # stay unrolled.
        for k in cutlass.range_constexpr(staged):
            stage, phase = k % stages, (k // stages) % 2
            cute.arch.mbarrier_wait(bar_full + stage, phase)
            _fence_tcgen05()
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(k_blocks):
                    source = a_desc + stage * a_step + 2 * kb
                    _copy_to_tmem(base + 8 + columns * k + 8 * kb, source)
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
        for k in cutlass.range_constexpr(staged):  # A from tensor memory
            cute.arch.mbarrier_wait(b_full + k, 0)
            _fence_tcgen05()
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(k_blocks):
                    b = b_desc + k * b_step + 2 * kb
                    accumulate = Int32(0 if k == 0 and kb == 0 else 1)
                    _mma_tmem(base, base + 8 + columns * k + 8 * kb, b, accumulate)
        for k in cutlass.range(staged, self.k_tiles, unroll=1):  # A from the ring
            ring_stage, ring_phase = k % stages, (k // stages) % 2
            cute.arch.mbarrier_wait(bar_full + ring_stage, ring_phase)
            cute.arch.mbarrier_wait(b_full + k, 0)
            _fence_tcgen05()
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(k_blocks):
                    a_ring = a_desc + ring_stage * a_step + 2 * kb
                    _mma_smem(base, a_ring, b_desc + k * b_step + 2 * kb, Int32(1))
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
    def push_partials(
        self, epi_tid: cutlass.Int32, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        acc_layout: cutlass.Constexpr, partial: cute.Tensor, section: cutlass.Int32,
        split_rank: cutlass.Int32, x_in: cute.Tensor, bar_in,
    ):  # fmt: skip
        """The accumulator -> ``partial`` (FP32, shared memory); then every KDA CTA
        gets this CTA's partial of its token's section rows in slot ``split_rank``:
        Q / K / gate rows to both CTAs of a token, V rows 64 p .. 64 p + 63 to
        partition p."""
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
        rows, values = self.cta_m, INPUT_BYTES // 2
        base = 128 * section - 64 * (section // 3)
        for m in cutlass.range_constexpr(4):
            i = epi_tid + 128 * m
            token, p, chunk = i // 64, i // 32 % 2, i % 32
            src_row, column = 4 * chunk, base + 4 * chunk
            if section == 2:
                token, p, chunk = i // 32, i // 16 % 2, i % 16
                src_row, column = 64 * p + 4 * chunk, 256 + 4 * chunk
            if (section != 2) | (m < 2):
                src = partial.iterator + cute.assume(token * rows + src_row, divby=4)
                words = load_words(src, count=4, shared=True)
                offset = split_rank * values + column
                dst = x_in.iterator + cute.assume(offset, divby=4)
                smem = cute.AddressSpace.smem
                dst = cute.make_ptr(Uint8, dst.llvm_ptr, smem, assumed_align=16)
                store_async_dsmem(dst, words, bar_in, 2 * token + p)

    @cute.jit
    def read_inputs(self, kt, x_in):
        """The four split ranks' partials summed in rank order, rounded to BF16."""
        values, sums = INPUT_BYTES // 2, cute.make_rmem_tensor(4, Float32)
        for c, base in enumerate((0, 128, 256, 320)):
            if (c != 2) | (kt < 64):
                sums[c] = x_in[base + kt]
                for q in cutlass.range_constexpr(1, SECTIONS):
                    sums[c] = sums[c] + x_in[q * values + base + kt]
        inputs = cute.make_rmem_tensor(3, BFloat16)
        inputs[0], inputs[1] = BFloat16(sums[0]), BFloat16(sums[1])
        inputs[2] = BFloat16(0)
        if kt < 64:
            inputs[2] = BFloat16(sums[2])
        return inputs, BFloat16(sums[3])


@cache
def compile_head(
    heads, stages, bf16_state, has_bias, specs, arch, tile=None, hidden=7168,
    polled=False,
):  # fmt: skip
    """``specs``: the (shape, stride, dtype) of every tensor argument in order;
    ``tile``: ``KdaHeadWide``'s (stages, K per stage, tensor-memory tiles[, early A
    tiles]), None: ``KdaHead`` with ``stages``; ``polled``: ``KdaHeadWide`` with
    ``polled_out`` (``POLLED_OUT``)."""
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    from ....ops.common.k3_ull_cute_host import fake

    def matrix(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    tensors = [fake(shape, stride, dtype) for shape, stride, dtype in specs]
    scalars = (Float32(-5), Float32(128**-0.5), Float32(1e-6))
    options = dict(bf16_state=bf16_state, has_bias=has_bias)
    if tile is None:
        kernel = KdaHead(heads, stages=stages, **options)
    else:
        wide_stages, cta_k, tmem_tiles, early_loads = (*tile, None)[:4]
        kernel = KdaHeadWide(
            heads, stages=wide_stages, cta_k=cta_k, tmem_tiles=tmem_tiles,
            early_loads=early_loads, hidden=hidden, **options,
        )  # fmt: skip
    kernel.polled_out = polled and tile is not None
    args = (matrix(1), matrix(1), cutlass.Int64(0), *tensors, *scalars)
    args += (cutlass.Int64(0), Int32(0), cutlass.Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


# The F_A / beta projection beside KdaHead: split-K 8 (24 CTAs, 7 K tiles each, all
# preloaded) on the side stream, from Rubin sweeps (TP16 B8 behind an AttnRes-like
# predecessor, us per layer vs the split-K projection -> KdaSplit pair: split 4 -0.11,
# 7 -0.36, 8 -0.36, 14 slower; KdaHead's own extra cluster -> +0.8..+1.9: it is only
# placed once the predecessor leaves). Split 8 sums the K slices in another order
# than the split-K 2 projection (an accumulation reorder of F_A and beta).
FA_TACTIC = (64, 8, 8, 7)
# ``KdaHeadMulti``'s F_A / beta projection per token rows: the same split (the same
# sums) with one N tile for every row, so the side projection keeps FA_TACTIC's 24
# full-SM CTAs instead of 24 per 8 rows (1-GPU KDA chain at M16 beside tail / AttnRes /
# o-proj footprints, see notes/smallm.md).
FA_TACTIC_ROWS = {16: (64, 16, 8, 7)}
HEAD_STAGES = 11
# Rubin: KdaHeadWide's (ring stages, K per stage, K tiles staged in tensor memory, A
# tiles loaded before the PDL wait), from the 1-GPU KDA chain (TP16 B8, 69 layers, one
# arm per process; notes/kdahead3.md). Loading more before the wait slows the AttnRes
# (its post-arrival reads share the memory system with the weight stream).
HEAD_TILE = (11, 64, 12, 17)
# Rubin (compute capability 10.7) KdaHead batches per TP whose output is polled by the
# fused o-projection (``OprojFused(polled_x="half")``) instead of read after its PDL
# wait. Each of the model's n >= 2 KdaHead layers owns a [8, dim] buffer (per layer:
# with 69 KDA layers a two-parity scheme would put layers n - 1 and 0 of consecutive
# steps on one parity). Arm / write / consume / re-arm of buffer i:
# - armed (every BF16 element OUT_SENT) once at allocation;
# - written once per step by KdaHead(i), one canonical BF16 store per element (a NaN
#   result is stored as 0x7FFF), so every element is OUT_SENT or this step's value;
# - consumed only by o-proj(i): it polls 16-byte chunks (volatile loads) until no
#   element is OUT_SENT, reads nothing else of KdaHead's before its own
#   griddepcontrol.wait, and executes that wait before it exits;
# - re-armed by KdaHead(i + 1 mod n), warp 1, after that kernel's griddepcontrol.wait.
# Why that is safe (transitive completion): the wait returns only after the
# predecessor grid completed, and a grid completes only after all its CTAs exit.
# **Every kernel of the decode graph's chain between o-proj(i) and KdaHead(i + 1)
# executes griddepcontrol.wait in at least one CTA before that CTA exits** (WsGemm,
# routing, the experts, WaveMoeTail, AttnRes, the MLA kernels, the dense MLP, and
# o-proj(i) itself at its end), so the completion of each implies the completion of
# the one before it: when KdaHead(i + 1)'s wait returns, o-proj(i), the only reader,
# has completed. A kernel added to that chain without such a wait breaks this
# argument. RouteSitu's lean polled route role and its copy role (9-64 rows) do not
# wait themselves, and the front before them may trigger before its wait (Rubin's
# MoeFrontTmem64 weight warp), so one RouteSitu thread waits before its CTA exits
# (``RouteSitu.exits_unwaited``, TensorRT-LLM local change). The next write of buffer
# i follows the re-arm:
# for i < n - 1 it is
# KdaHead(i) of the next step (a later graph launch); for i = n - 1 (re-armed by
# KdaHead(0) of the same step) KdaHead(n - 1) writes after its own wait, i.e. after
# KdaHead(0) completed. The next read cannot see the old value either: o-proj(n - 1)
# is launched behind the front GEMM of an earlier MoE layer, which triggers its
# dependents only after its wait, i.e. after KdaHead(0) completed (other buffers:
# a later graph launch). Steps that do not run KdaHead at B8 never touch the buffers.
POLLED_OUT = {16: "8:1"}


def polled_out_batches(tp, device):
    """``POLLED_OUT`` batches on this device (compute capability 10.7 only)."""
    if torch.cuda.get_device_capability(device) != (10, 7):
        return ()
    table = batch_table(POLLED_OUT.get(tp, ""))
    return tuple(b for b, on in table.items() if on)


class KdaSide:
    """The step's side stream for F_A / beta: forked per layer after the attention
    input is launched, joined once at the end of the step (as the L2 prefetch)."""

    def __init__(self, device):
        self.device, self.pending = device, False
        self.stream = torch.cuda.Stream(device=device)

    def fork(self):
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.pending = True
        return self.stream

    def join(self):
        if self.pending:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            self.pending = False


class PreparedKdaHead:
    """``KdaHead`` for a layer's KDA input projection + recurrence at the prepared
    batches, with F_A / beta from ``SignalGemm`` on ``side``; ``counter`` is this
    layer's [ready, done] pair (zeros; the last reader re-arms it), ``fa_out`` the
    layer's own [>= 8, 144] BF16 F_A buffer: ``KdaHeadWide`` polls its rows against a
    sentinel that its last reader writes back, so a buffer is reused only across graph
    launches."""

    def __init__(
        self, weight, core, batches, side, counter, fa_out, tile=HEAD_TILE, polled=(),
        verify=(),
    ):  # fmt: skip
        self.weight, self.core, self.side = weight, core, side
        self.counter, self.fa_out, self.batches = counter, fa_out, tuple(batches)
        # batches whose output is a polled buffer (``POLLED_OUT``; KdaHeadWide only)
        self.polled = tuple(b for b in polled if b in self.batches and tile is not None)
        self.polled_kernels: dict = {}
        self.device, self.tile = weight.device, tile
        self.map = weight_map(weight, 64 if tile is None else 128)
        if tile is not None:  # KdaHeadWide polls F_A's rows: start from the sentinel
            fa_out.view(torch.int16).fill_(-1)
        self.fa_weight = weight[4 * core.dim :]
        self.fa_map = weight_map(self.fa_weight, 64)
        self.fa_tiles = -(-self.fa_weight.shape[0] // 64)
        self.kernels: dict = {}
        self.fa_code: Any = None
        # verify shapes (rows, q) run by ``KdaHeadVerify`` (KdaHeadWide only); keyed by
        # rows: the polled variant follows ``polled``
        self.verify = tuple(v for v in verify if tile is not None)
        self.verify_kernels: dict = {}

    def runs(self, batch, q):
        """Whether steps of ``batch`` rows at verify length ``q`` run this head (the
        q = 1 KdaHead batches, the KdaHeadVerify shapes): the step policies that assume
        a KdaHead next layer key on it (runtime.decode.head_batches)."""
        return (q == 1 and batch in self.batches) or (batch, q) in self.verify

    def _tensors(self, caches, slots, out):
        conv, state = caches[:2]
        conv = conv if self.core.conv_dim_first else conv.transpose(-1, -2)
        w, bias, fb, a_log, dt, norm = self.core.weights
        fa = self.fa_out[: out.shape[0]]
        return (w, bias, conv, fa, a_log, dt, slots, state, out, norm, fb)

    def prepare_caches(self, caches):
        arch = architecture(self.device)
        bf16_state = caches[1].dtype == torch.bfloat16
        meta = dict(device="meta")
        with torch.accelerator.device_index(self.device.index):
            self.fa_code = compile_signal(FA_TACTIC, arch)
            for b in self.batches:
                slots = torch.empty((b,), dtype=torch.int32, **meta)
                out = torch.empty(b, self.core.dim, dtype=torch.bfloat16, **meta)
                specs = tuple(map(layout, self._tensors(caches, slots, out)))
                self.kernels[b] = compile_head(
                    self.core.heads, HEAD_STAGES, bf16_state, self.core.has_bias,
                    specs, arch, self.tile, self.weight.shape[1],
                )  # fmt: skip
                if b in self.polled:
                    self.polled_kernels[b] = compile_head(
                        self.core.heads, HEAD_STAGES, bf16_state, self.core.has_bias,
                        specs, arch, self.tile, self.weight.shape[1], True,
                    )  # fmt: skip
            for b, q in self.verify:
                from .head_multi import compile_head_multi

                if b in FA_TACTIC_ROWS:
                    compile_signal(FA_TACTIC_ROWS[b], arch)
                from .head_verify import compile_head_verify

                slots = torch.empty((b // q, q), dtype=torch.int32, **meta)
                out = torch.empty(b, self.core.dim, dtype=torch.bfloat16, **meta)
                accepted = torch.empty((b // q,), dtype=torch.int32, **meta)
                tensors = (*self._tensors(caches, slots, out), accepted)
                specs = tuple(map(layout, tensors))
                if b == q:
                    self.verify_kernels[b, q] = compile_head_verify(
                        self.core.heads, bf16_state, self.core.has_bias, specs, arch,
                        self.tile, self.weight.shape[1], b in self.polled,
                    )  # fmt: skip
                else:  # several requests: KdaHeadMulti
                    self.verify_kernels[b, q] = compile_head_multi(
                        self.core.heads, bf16_state, self.core.has_bias, specs, arch,
                        b // q, self.weight.shape[1], b in self.polled, tokens=q,
                    )  # fmt: skip
        return self

    def run_verify(self, x, out, *, caches, slots, accepted, rearm=None):
        """``KdaHeadVerify`` of one request x q verify rows (``slots`` [1, q], vLLM's
        spec slots; ``accepted`` [1]), or ``KdaHeadMulti`` of b // q requests;
        ``rearm`` as ``run`` (the kernel is the polled variant when the batch is in
        ``polled``)."""
        b, q = x.shape[0], slots.shape[1]
        requests = b // q
        fa = self.fa_out[:b]
        counter = cutlass.Int64(self.counter.data_ptr())
        fa_tactic = FA_TACTIC_ROWS.get(b, FA_TACTIC)
        fa_code = self.fa_code
        if fa_tactic != FA_TACTIC:
            fa_code = compile_signal(fa_tactic, architecture(self.device))
        with torch.cuda.stream(self.side.fork()):
            fa_code(
                matrix_view(self.fa_weight), matrix_view(x),
                matrix_view(fa, output=True), cutlass.Int64(self.fa_map.data_ptr()),
                counter, stream(self.device),
            )  # fmt: skip
        if slots.stride() != (slots.shape[1], 1):
            slots = slots.contiguous()
        tensors = (*self._tensors(caches, slots[:requests], out), accepted[:requests])
        views = [to_cute(t, 16) for t in tensors]
        if (rearm is None) != (b not in self.polled):
            raise ValueError("KdaHeadVerify: a polled batch needs the buffer to re-arm")
        address = rearm.data_ptr() if rearm is not None else 0
        target = self.fa_tiles * -(-b // fa_tactic[1])  # SignalGemm's tile owners
        self.verify_kernels[b, q](
            matrix_view(self.weight), matrix_view(x),
            cutlass.Int64(self.map.data_ptr()),
            *views, *self.core.scalars, counter, Int32(target),
            cutlass.Int64(address), stream(self.device),
        )  # fmt: skip
        return out

    def run(self, x, out, *, caches, slots, rearm=None):
        """``rearm``: with a polled batch, ``out`` is this layer's polled buffer and
        ``rearm`` the previous KDA layer's (``POLLED_OUT``)."""
        b = x.shape[0]
        fa = self.fa_out[:b]
        counter = cutlass.Int64(self.counter.data_ptr())
        with torch.cuda.stream(self.side.fork()):
            self.fa_code(
                matrix_view(self.fa_weight), matrix_view(x),
                matrix_view(fa, output=True), cutlass.Int64(self.fa_map.data_ptr()),
                counter, stream(self.device),
            )  # fmt: skip
        views = [to_cute(t, 16) for t in self._tensors(caches, slots[:b], out)]
        weight_map_address = cutlass.Int64(self.map.data_ptr())
        code, address = self.kernels[b], 0
        if rearm is not None:
            code, address = self.polled_kernels[b], rearm.data_ptr()
        code(
            matrix_view(self.weight), matrix_view(x), weight_map_address, *views,
            *self.core.scalars, counter, Int32(self.fa_tiles), cutlass.Int64(address),
            stream(self.device),
        )  # fmt: skip
        return out
