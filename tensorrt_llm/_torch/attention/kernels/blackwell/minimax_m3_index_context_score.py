# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL MiniMax-M3 index context block-scoring kernel (Blackwell SM100 family).

For one context request with prefix_len cached tokens and q_len new ones, computes for every index
head h, new token t and 128-key block g the maximum causally visible Q . K product

    out[h, g, token_offset + t] = max_{key j in block g, j <= prefix_len + t} q[t, h] . k[j]

(-inf when the block holds no visible key), the per-block maxima the MSA indexer ranks to pick the
top-k blocks each token attends.

The head dim (128) is the whole reduction, so a 128-query x 128-key score tile is a single tcgen05
MMA chain into TMEM, and the epilogue reduces each accumulator row of a tile to one float: the
score matrix never leaves TMEM. On SM103 the row max comes with the TMEM load
(tcgen05.ld.red.max); other GPUs reduce the loaded scores with 3-input max.f32. A CTA pair
(cta_group::2) runs 256 x 256 MMAs, each CTA loading one page of the 256-key tile straight from
the paged index-K cache. The (query block, key tile, MMA) units holding a visible key are split into
equal contiguous ranges over persistent CTA pairs; units never share an output, so no cross-CTA
reduction is needed.
"""

import functools

import cutlass
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float8E4M3FN, Float32, Int32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = ["minimax_m3_index_context_score"]

_PAGE_SIZE = 128  # keys per index-K page, which is also the scored block
_HEAD_DIM = 128
_BLOCK_Q = 128  # queries per A tile: the MMA rows one CTA holds
_TILE_N = 2 * _PAGE_SIZE  # keys per MMA: one page from each CTA of the pair
_SUPPORTED_NUM_HEADS = (1, 2, 4)
# With two heads the query split feeds every key tile to both heads, but its 256-query blocks
# widen the causal diagonal and halve the number of work units. It only wins from this many
# head-split units per CTA pair (tuned on B300).
_QSPLIT_MIN_UNITS_PER_PAIR = 20
_NEG_INF = float("-inf")

_compiled_kernels = {}


@dsl_user_op
def _tmem_load(taddrs, rep: int, with_max: bool, *, loc=None, ip=None):
    """Warp-collective 32x32b TMEM loads of ``rep`` columns at each address, then one wait.

    Returns the len(taddrs) * rep loaded values in column order and, if ``with_max``, the per-lane
    max of each load (tcgen05.ld.red, SM103), else an empty list.
    """
    num_loads = len(taddrs)
    num_values = num_loads * rep
    num_outs = num_values + (num_loads if with_max else 0)
    lines = []
    for i in range(num_loads):
        regs = ", ".join(f"${i * rep + j}" for j in range(rep))
        addr = f"[${num_outs + i}]"
        if with_max:
            lines.append(
                f"tcgen05.ld.red.sync.aligned.32x32b.x{rep}.f32.max {{{regs}}}, ${num_values + i}, {addr};"
            )
        else:
            lines.append(f"tcgen05.ld.sync.aligned.32x32b.x{rep}.b32 {{{regs}}}, {addr};")
    lines.append("tcgen05.wait::ld.sync.aligned;")
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * num_outs),
        [Int32(a).ir_value(loc=loc, ip=ip) for a in taddrs],
        "\n".join(lines),
        ",".join(["=f"] * num_outs + ["r"] * num_loads),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    outs = [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(num_outs)]
    return outs[:num_values], outs[num_values:]


@dsl_user_op
def _fmax3(a, b, c, *, loc=None, ip=None):
    """3-input max.f32 (a single FMNMX3 on SM100+)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c)],
            "max.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _max_tree(values):
    """Max of a list of Float32, three at a time."""
    values = list(values)
    while len(values) > 1:
        reduced = []
        for i in range(0, len(values), 3):
            group = values[i : i + 3]
            if len(group) == 3:
                reduced.append(_fmax3(*group))
            elif len(group) == 2:
                reduced.append(cute.arch.fmax(*group))
            else:
                reduced.append(group[0])
        values = reduced
    return values[0]


class _IndexContextScoreKernel:
    """Block-max scorer of one context request on CTA pairs (cta_group::2).

    The pair runs M=256 x N=256 MMAs: CTA r loads page 2 * kt + r of key tile kt and holds 128 of
    the 256 A rows, so its TMEM receives its 128 rows x both pages. Only the leader issues MMAs.
    Warps 0-7 drain TMEM (warp w reads lane quarter w % 4 of the tile's page w // 4), warp 8
    issues MMAs and warp 9 TMA loads.

    Head split (``qsplit=False``, even H): a schedule block is one 128-query block; CTA r holds
    heads [r * H / 2, (r + 1) * H / 2) and MMA j multiplies [head j; head H / 2 + j].
    Query split (``qsplit=True``): a schedule block is two 128-query blocks; CTA r holds block
    2b + r of every head and MMA j multiplies [head j of block 2b; head j of block 2b + 1].
    """

    num_a_stages = 2
    num_b_stages = 6
    num_acc_stages = 2
    ld_rep = 64  # TMEM columns per tcgen05.ld
    num_epi_warps = 8
    mma_warp = 8
    tma_warp = 9
    threads = 32 * 10
    tmem_alloc_bar_id = 1
    epi_bar_id = 2
    cluster_shape_mn = (2, 1)
    ab_dtype = Float8E4M3FN

    def __init__(self, num_heads: int, qsplit: bool, use_ld_red: bool) -> None:
        self.num_heads = num_heads
        self.qsplit = qsplit
        self.use_ld_red = use_ld_red
        # MMAs per (schedule block, key tile), which is also the A tiles per CTA and stage.
        self.mmas_per_tile = num_heads if qsplit else num_heads // 2
        self.block_q = 2 * _BLOCK_Q if qsplit else _BLOCK_Q  # queries per schedule block
        self.mma_tiler = (2 * _BLOCK_Q, _TILE_N, _HEAD_DIM)
        self.num_tmem_cols = self.num_acc_stages * _TILE_N
        # The leader's full barriers count both CTAs' TMA bytes; an FP8 128 x 128 tile is 16 KiB.
        self.a_tx_bytes = 2 * self.mmas_per_tile * _BLOCK_Q * _HEAD_DIM
        self.b_tx_bytes = 2 * _PAGE_SIZE * _HEAD_DIM

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # [q_len, num_heads, head_dim]
        k_cache: cute.Tensor,  # [num_pages, 1, page_size, head_dim]
        page_table: cute.Tensor,  # [>= num_blocks]
        out: cute.Tensor,  # [num_heads, >= num_blocks, >= token_offset + q_len]
        prefix_len: Int32,
        token_offset: Int32,
        num_ctas: Int32,
        stream: CUstream,
    ):
        q_len = q.shape[0]
        ab_dtype = self.ab_dtype
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.TWO,
            self.mma_tiler[:2],
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        num_a_slots = self.num_a_stages * self.mmas_per_tile
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, ab_dtype, num_a_slots
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, ab_dtype, self.num_b_stages
        )
        # TMA views: one 128 x 128 K-major tile per A slot / B stage, the MMA's per-CTA halves.
        a_tma_view = sm100_utils.make_smem_layout(
            tcgen05.OperandMajorMode.K, (_BLOCK_Q, _HEAD_DIM), ab_dtype, num_a_slots
        )
        a_tma_view = cute.select(cute.tiled_divide(a_tma_view, (_BLOCK_Q, _HEAD_DIM)), mode=[0, 3])
        b_tma_view = sm100_utils.make_smem_layout(
            tcgen05.OperandMajorMode.K, (_PAGE_SIZE, _HEAD_DIM), ab_dtype, self.num_b_stages
        )
        b_tma_view = cute.select(
            cute.tiled_divide(b_tma_view, (_PAGE_SIZE, _HEAD_DIM)), mode=[0, 3]
        )
        # A size-1 head mode never advances; the token stride stands in for its stride, which
        # can be anything and would otherwise reach the TMA descriptor.
        head_stride = q.stride[1] if self.num_heads > 1 else q.stride[0]
        a = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (q_len, _HEAD_DIM, self.num_heads), stride=(q.stride[0], 1, head_stride)
            ),
        )
        # (row in page, dim, physical page)
        b = cute.make_tensor(
            k_cache.iterator,
            cute.make_layout(
                (_PAGE_SIZE, _HEAD_DIM, k_cache.shape[0]),
                stride=(k_cache.stride[2], 1, k_cache.stride[0]),
            ),
        )
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_op, a, a_tma_view[0], (_BLOCK_Q, _HEAD_DIM)
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            tma_op, b, b_tma_view[0], (_PAGE_SIZE, _HEAD_DIM)
        )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            page_table,
            out,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            a_tma_view,
            b_tma_view,
            q_len,
            prefix_len + q_len,
            prefix_len,
            token_offset,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        mPageTable: cute.Tensor,
        mOut: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        a_tma_view: cute.ComposedLayout,
        b_tma_view: cute.ComposedLayout,
        n: Int32,
        L: Int32,
        prefix_len: Int32,
        token_offset: Int32,
    ):
        NG = self.mmas_per_tile
        BQ = self.block_q
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nctas, _, _ = cute.arch.grid_dim()
        rank = bidx % 2  # CTA rank in the pair (cluster (2, 1, 1) along x)
        is_leader = rank == 0

        if warp_idx == self.tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, self.num_a_stages * 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, self.num_b_stages * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, self.num_acc_stages * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        a_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(),
            num_stages=self.num_a_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.a_tx_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar.data_ptr(),
            num_stages=self.num_b_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.b_tx_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # Consumers: the epilogue warps of both CTAs.
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=self.num_acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epi_warps * 2),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_bar_id, num_threads=32 * (self.num_epi_warps + 1)
        )
        epi_barrier = pipeline.NamedBarrier(
            barrier_id=self.epi_bar_id, num_threads=32 * self.num_epi_warps
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline.pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_layout_staged.inner,
        )
        sA_tma = cute.make_tensor(sA.iterator, a_tma_view.outer)  # ((128, 128), slot)
        sB_tma = cute.make_tensor(sB.iterator, b_tma_view.outer)  # ((128, 128), stage)
        gA = cute.local_tile(mA, (_BLOCK_Q, _HEAD_DIM), (None, None, None))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), sA_tma, cute.group_modes(gA, 0, 2)
        )
        gB = cute.local_tile(mB, (_PAGE_SIZE, _HEAD_DIM), (None, None, None))
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1), sB_tma, cute.group_modes(gB, 0, 2)
        )
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stages))
        num_kblocks = cute.size(tCrA, mode=[2])

        pipeline.pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # Static schedule: the units are (schedule block, key tile, MMA) triples holding a visible
        # key, in that order; CTA pair c takes the c-th of nctas / 2 equal contiguous ranges.
        C = prefix_len
        nqb = (n + _BLOCK_Q - 1) // _BLOCK_Q
        nsb = (n + BQ - 1) // BQ
        total = Int32(0)
        for sbi in cutlass.range(nsb, unroll=1):
            total += (C + min(sbi * BQ + BQ - 1, n - 1)) // _TILE_N + 1
        cid = bidx // 2
        ncl = nctas // 2
        total_u = total * NG
        # floor(total_u * cid / ncl) without int32 overflow
        u_begin = (total_u // ncl) * cid + ((total_u % ncl) * cid) // ncl
        u_end = (total_u // ncl) * (cid + 1) + ((total_u % ncl) * (cid + 1)) // ncl
        t_first = u_begin // NG
        t_last = (u_end - 1) // NG
        sb0 = Int32(0)
        s0 = Int32(0)
        cnt0 = (C + min(BQ - 1, n - 1)) // _TILE_N + 1
        while s0 + cnt0 <= t_first:
            s0 += cnt0
            sb0 += 1
            cnt0 = (C + min(sb0 * BQ + BQ - 1, n - 1)) // _TILE_N + 1
        kt0 = t_first - s0

        # TMA (both CTAs)
        if warp_idx == self.tma_warp:
            a_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_a_stages
            )
            b_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_b_stages
            )
            num_pages = (L + _PAGE_SIZE - 1) // _PAGE_SIZE
            sb = sb0
            kt = kt0
            cnt = cnt0
            cur_sb = Int32(-1)
            t = t_first
            while t <= t_last:
                if sb != cur_sb:
                    a_pipeline.producer_acquire(a_state)
                    for j in cutlass.range_constexpr(NG):
                        if cutlass.const_expr(self.qsplit):
                            # Past the last query block, reload it: its rows are never stored.
                            a_src = tAgA[(None, min(2 * sb + rank, nqb - 1), 0, j)]
                        else:
                            a_src = tAgA[(None, sb, 0, rank * NG + j)]
                        cute.copy(
                            tma_atom_a,
                            a_src,
                            tAsA[(None, a_state.index * NG + j)],
                            tma_bar_ptr=a_pipeline.producer_get_barrier(a_state),
                        )
                    a_state.advance()
                    cur_sb = sb
                b_pipeline.producer_acquire(b_state)
                # A tile's second page can lie past the request; its block is never stored.
                page = min(kt * 2 + rank, num_pages - 1)
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, 0, mPageTable[page])],
                    tBsB[(None, b_state.index)],
                    tma_bar_ptr=b_pipeline.producer_get_barrier(b_state),
                )
                b_state.advance()
                t += 1
                kt += 1
                if kt == cnt:
                    kt = Int32(0)
                    sb += 1
                    cnt = (C + min(sb * BQ + BQ - 1, n - 1)) // _TILE_N + 1
            a_pipeline.producer_tail(a_state)
            b_pipeline.producer_tail(b_state)

        # MMA (the leader issues for the pair)
        if warp_idx == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            a_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_a_stages
            )
            b_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_b_stages
            )
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stages
            )
            if is_leader:
                sb = sb0
                kt = kt0
                cnt = cnt0
                cur_sb = Int32(-1)
                t = t_first
                while t <= t_last:
                    if sb != cur_sb:
                        if cur_sb >= 0:
                            a_pipeline.consumer_release(a_state)
                            a_state.advance()
                        a_pipeline.consumer_wait(a_state)
                        cur_sb = sb
                    b_pipeline.consumer_wait(b_state)
                    u_lo = u_begin - t * NG
                    u_hi = u_end - 1 - t * NG
                    for j in cutlass.range_constexpr(NG):
                        if (u_lo <= j) & (j <= u_hi):
                            acc_pipeline.producer_acquire(acc_state)
                            tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for kb in cutlass.range_constexpr(num_kblocks):
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[(None, None, kb, a_state.index * NG + j)],
                                    tCrB[(None, None, kb, b_state.index)],
                                    tCtAcc,
                                )
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            acc_pipeline.producer_commit(acc_state)
                            acc_state.advance()
                    b_pipeline.consumer_release(b_state)
                    b_state.advance()
                    t += 1
                    kt += 1
                    if kt == cnt:
                        kt = Int32(0)
                        sb += 1
                        cnt = (C + min(sb * BQ + BQ - 1, n - 1)) // _TILE_N + 1
                if cur_sb >= 0:
                    a_pipeline.consumer_release(a_state)
                acc_pipeline.producer_tail(acc_state)

        # Epilogue (both CTAs)
        if warp_idx < self.num_epi_warps:
            tmem.allocate(self.num_tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tmem_base = tmem_ptr.toint()
            lane = tidx % 32
            quarter = warp_idx % 4
            ch = warp_idx // 4  # page of the tile: 0 is the leader's, 1 the peer's
            row = quarter * 32 + lane
            lane_addr = tmem_base + ((quarter * 32) << 16) + ch * _PAGE_SIZE
            G = (L + _PAGE_SIZE - 1) // _PAGE_SIZE
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stages
            )
            num_loads = _PAGE_SIZE // self.ld_rep
            sb = sb0
            kt = kt0
            cnt = cnt0
            t = t_first
            while t <= t_last:
                if cutlass.const_expr(self.qsplit):
                    qb = 2 * sb + rank  # this CTA's 128-query block
                else:
                    qb = sb
                q = qb * _BLOCK_Q + row
                g = kt * 2 + ch
                # Warp-uniform: does any row of this query block see only part of block g?
                need_mask = g * _PAGE_SIZE + _PAGE_SIZE - 1 > C + qb * _BLOCK_Q
                limit = C + q - g * _PAGE_SIZE  # this row sees columns [0, limit] of block g
                do_store = (q < n) & (g < G)
                do_fill = (kt == cnt - 1) & (q < n)
                u_lo = u_begin - t * NG
                u_hi = u_end - 1 - t * NG
                for j in cutlass.range_constexpr(NG):
                    if (u_lo <= j) & (j <= u_hi):
                        if cutlass.const_expr(self.qsplit):
                            h = j
                        else:
                            h = rank * NG + j
                        acc_pipeline.consumer_wait(acc_state)
                        base = lane_addr + acc_state.index * _TILE_N
                        addrs = [base + i * self.ld_rep for i in range(num_loads)]
                        scores, load_maxima = _tmem_load(addrs, self.ld_rep, self.use_ld_red)
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_state)
                        acc_state.advance()
                        m = Float32(_NEG_INF)
                        if need_mask:
                            ninf = Float32(_NEG_INF)
                            m = _max_tree(
                                [
                                    Float32(cutlass.select_(limit >= i, scores[i], ninf))
                                    for i in range(_PAGE_SIZE)
                                ]
                            )
                        else:
                            if cutlass.const_expr(self.use_ld_red):
                                m = _max_tree(load_maxima)
                            else:
                                m = _max_tree(scores)
                        if do_store:
                            mOut[(h, g, token_offset + q)] = m
                        if do_fill:
                            # No row of this schedule block sees the blocks past its last tile.
                            gg = cnt * 2 + ch
                            while gg < G:
                                mOut[(h, gg, token_offset + q)] = Float32(_NEG_INF)
                                gg += 2
                t += 1
                kt += 1
                if kt == cnt:
                    kt = Int32(0)
                    sb += 1
                    cnt = (C + min(sb * BQ + BQ - 1, n - 1)) // _TILE_N + 1
            epi_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


@functools.lru_cache(maxsize=None)
def _device_traits(device_index: int) -> tuple[int, bool]:
    """SM count of an SM100-family GPU and whether it has tcgen05.ld.red (SM103 and later)."""
    props = torch.cuda.get_device_properties(device_index)
    if props.major != 10:
        raise ValueError(
            "The MiniMax-M3 index context scorer requires an SM100-family GPU; "
            f"got SM{props.major}{props.minor}."
        )
    return props.multi_processor_count, props.minor >= 3


def _num_work_tiles(q_len: int, prefix_len: int, block_q: int) -> int:
    """(block_q-query block, 256-key tile) pairs of a request whose tile holds a visible key."""
    return sum(
        (prefix_len + min(start + block_q, q_len) - 1) // _TILE_N + 1
        for start in range(0, q_len, block_q)
    )


@functools.lru_cache(maxsize=1024)
def _schedule(num_heads: int, q_len: int, prefix_len: int, num_sms: int) -> tuple[bool, int]:
    """Whether to split the CTA pair over queries rather than heads, and the CTAs to launch."""
    num_pairs = num_sms // 2
    if num_heads == 2:
        qsplit = (
            _num_work_tiles(q_len, prefix_len, _BLOCK_Q) >= _QSPLIT_MIN_UNITS_PER_PAIR * num_pairs
        )
    else:
        qsplit = num_heads == 1
    mmas_per_tile = num_heads if qsplit else num_heads // 2
    num_units = mmas_per_tile * _num_work_tiles(
        q_len, prefix_len, 2 * _BLOCK_Q if qsplit else _BLOCK_Q
    )
    return qsplit, 2 * min(num_pairs, num_units)


def _compiled_kernel(num_heads: int, qsplit: bool, use_ld_red: bool):
    key = (num_heads, qsplit, use_ld_red)
    if key not in _compiled_kernels:

        def stride16():
            # TMA strides must be multiples of 16 B, i.e. 16 FP8 elements.
            return cute.sym_int64(divisibility=16)

        # A single head's stride is never used, so it carries no constraint.
        q = cute.runtime.make_fake_tensor(
            Float8E4M3FN,
            (cute.sym_int(), num_heads, _HEAD_DIM),
            stride=(stride16(), stride16() if num_heads > 1 else cute.sym_int64(), 1),
            assumed_align=16,
        )
        k_cache = cute.runtime.make_fake_tensor(
            Float8E4M3FN,
            (cute.sym_int(), 1, _PAGE_SIZE, _HEAD_DIM),
            stride=(stride16(), cute.sym_int64(), stride16(), 1),
            assumed_align=16,
        )
        page_table = cute.runtime.make_fake_tensor(Int32, (cute.sym_int(),), stride=(1,))
        out = cute.runtime.make_fake_tensor(
            Float32,
            (num_heads, cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int64(), cute.sym_int64(), 1),
        )
        # Two heads pick the schedule per request, so the first call (in warmup) compiles both.
        for split in (False, True) if num_heads == 2 else (qsplit,):
            _compiled_kernels[(num_heads, split, use_ld_red)] = cute.compile(
                _IndexContextScoreKernel(num_heads, split, use_ld_red),
                q,
                k_cache,
                page_table,
                out,
                Int32(0),
                Int32(0),
                Int32(2),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
    return _compiled_kernels[key]


def minimax_m3_index_context_score(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    prefix_len: int,
    out: torch.Tensor,
    token_offset: int,
) -> None:
    """Write the per-block max index scores of one context request into ``out``.

    New token t sits at position prefix_len + t and sees keys [0, prefix_len + t]. For every index
    head h and 128-key block g of the request, out[h, g, token_offset + t] becomes the max of
    q[t, h] . k over the keys of block g that token t sees, or -inf if it sees none. The scores are
    raw Q.K, without a softmax scale. Nothing else in ``out`` is written.

    Args:
        q: [q_len, num_heads, 128] float8_e4m3fn index queries of the new tokens, num_heads in
            {1, 2, 4}. 16 B aligned and head dim contiguous, with token and head strides that are
            multiples of 16 elements.
        k_cache: [num_pages, 1, 128, 128] float8_e4m3fn paged index-K cache, 16 B aligned. The page
            stride may be any multiple of 16 elements, e.g. a view into a wider pool.
        page_table: [>= ceil((prefix_len + q_len) / 128)] int32 contiguous cache pages of the
            request's keys, in order.
        prefix_len: Cached tokens ahead of the new ones.
        out: [num_heads, >= ceil((prefix_len + q_len) / 128), >= token_offset + q_len] float32
            with a unit stride along the last dim.
        token_offset: Index along the last dim of ``out`` that receives token 0.

    Raises:
        ValueError: For an unsupported head count or GPU, or a page table or ``out`` too small
            for the request.
    """
    q_len, num_heads, _ = q.shape
    if q_len == 0:
        return
    if num_heads not in _SUPPORTED_NUM_HEADS:
        raise ValueError(f"num_heads must be one of {_SUPPORTED_NUM_HEADS}; got {num_heads}.")
    num_blocks = (prefix_len + q_len + _PAGE_SIZE - 1) // _PAGE_SIZE
    if (
        page_table.shape[0] < num_blocks
        or out.shape[1] < num_blocks
        or out.shape[2] < token_offset + q_len
    ):
        raise ValueError(
            f"A request of {num_blocks} blocks and {q_len} new tokens at token_offset "
            f"{token_offset} needs page_table [>= {num_blocks}] and out [{num_heads}, "
            f">= {num_blocks}, >= {token_offset + q_len}]; got {tuple(page_table.shape)} and "
            f"{tuple(out.shape)}."
        )
    num_sms, use_ld_red = _device_traits(q.device.index)
    qsplit, num_ctas = _schedule(num_heads, q_len, prefix_len, num_sms)
    _compiled_kernel(num_heads, qsplit, use_ld_red)(
        q, k_cache, page_table, out, prefix_len, token_offset, num_ctas
    )
