# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 Reindex-layer decode: DeepGEMM's sparse candidate logits and the mapped Top-K in one launch,
one 16- or 8-CTA cluster per query row, bit for bit the two launches' selection."""

import torch

_COMPILED: dict = {}
THREADS = 512
WIDTH = 16384
SPARSE_BLOCK = 8
PAGE_ROWS = 64
HEADS = 32
TOP_K = 512
_FUSED_SELECT_MAX_ROWS = 19  # one wave of 8-CTA clusters on 152 SMs


def cluster_for_rows(rows: int, num_sms: int) -> int:
    """CTAs per row: 16 while the rows' clusters fit one wave, else 8 (0: not supported)."""
    for cluster in (16, 8):
        if 0 < rows * cluster <= num_sms:
            return cluster
    return 0


def _kernel_class():
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.blackwell_helpers as sm100_utils
    import cutlass.utils.blockscaled_layout as blockscaled_utils
    from cutlass import Float4E2M1FN, Float8E8M0FNU, Float32, Int32, Int64, Uint16, Uint32
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.utils.smem_allocator import SmemAllocator

    from tensorrt_llm._torch.kernels.blackwell.top_k.single_pass_multi_cta_radix_topk import (
        _TIE_LIST_COUNT,
    )
    from tensorrt_llm._torch.kernels.blackwell.top_k.single_pass_multi_cta_radix_topk_cluster import (
        SinglePassMultiCTARadixTopKClusterKernel,
        mapa_shared_cluster,
    )
    from tensorrt_llm._torch.kernels.blackwell.utils import (
        TRTLLM_ENABLE_PDL,
        griddepcontrol_launch_dependents,
        griddepcontrol_wait,
    )

    def _asm(operands, text, constraints, result=None, loc=None, ip=None):
        return llvm.inline_asm(
            result,
            operands,
            text,
            constraints,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def cp_async_16(dst, src, *, loc=None, ip=None):
        _asm(
            [Int32(dst).ir_value(loc=loc, ip=ip), Int64(src).ir_value(loc=loc, ip=ip)],
            "cp.async.cg.shared.global [$0], [$1], 16;",
            "r, l",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def cp_async_4(dst, src, *, loc=None, ip=None):
        _asm(
            [Int32(dst).ir_value(loc=loc, ip=ip), Int64(src).ir_value(loc=loc, ip=ip)],
            "cp.async.ca.shared.global [$0], [$1], 4;",
            "r, l",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def cp_async_arrive_noinc(mbar, *, loc=None, ip=None):
        _asm(
            [Int32(mbar).ir_value(loc=loc, ip=ip)],
            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];",
            "r",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def tcgen05_before_thread_sync(*, loc=None, ip=None):
        _asm([], "tcgen05.fence::before_thread_sync;", "", loc=loc, ip=ip)

    @dsl_user_op
    def tcgen05_after_thread_sync(*, loc=None, ip=None):
        _asm([], "tcgen05.fence::after_thread_sync;", "", loc=loc, ip=ip)

    @dsl_user_op
    def st_shared_u16(addr, value, *, loc=None, ip=None):
        _asm(
            [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\nst.shared.b16 [$0], h;\n}",
            "r, r",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def red_add_cta(addr, value, *, loc=None, ip=None):
        """``red`` (no return value) at CTA scope lets ptxas aggregate the adds per warp."""
        _asm(
            [Int32(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
            "red.relaxed.cta.shared::cta.add.u32 [$0], $1;",
            "r, r",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def st_cluster_v4(cluster_addr, a, b, c, d, *, loc=None, ip=None):
        _asm(
            [Int32(cluster_addr).ir_value(loc=loc, ip=ip)]
            + [Int32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
            "st.shared::cluster.v4.b32 [$0], {$1, $2, $3, $4};",
            "r, r, r, r, r",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def tmem_ld_32x32b_x32(taddr, *, loc=None, ip=None):
        """Thread t gets TMEM lane taddr.lane + t, columns [col, col + 32), as raw 32-bit words."""
        i32 = T.i32()
        regs = ", ".join(f"${k}" for k in range(32))
        text = (
            f"tcgen05.ld.sync.aligned.32x32b.x32.b32 {{{regs}}}, [$32];\n"
            "tcgen05.wait::ld.sync.aligned;"
        )
        res = _asm(
            [Int32(taddr).ir_value(loc=loc, ip=ip)],
            text,
            ",".join(["=r"] * 32 + ["r"]),
            result=llvm.StructType.get_literal([i32] * 32),
            loc=loc,
            ip=ip,
        )
        return [Uint32(llvm.extractvalue(i32, res, [k], loc=loc, ip=ip)) for k in range(32)]

    @dsl_user_op
    def head_sum_bf16(acc, w2, *, loc=None, ip=None) -> Uint32:
        """DeepGEMM's sparse BF16 head sum of one key's 32 FP32 head accumulators ``acc`` and 16
        BF16 weight pairs ``w2`` (head 2i low); the BF16 logit in the low 16 bits."""
        lines = ["{", ".reg .b32 s0, s1, p, s, t;", "mov.b32 s0, 0;", "mov.b32 s1, 0;"]
        # sum_0 over pairs (h, h + 1), h = 0, 4, .., 28; sum_1 over h = 2, 6, .., 30: DeepGEMM's
        # head_base (0, 16) x head_offset (0, 4, 8, 12) loop, one cvt + fma per chain and step.
        for hb in (0, 16):
            for ho in (0, 4, 8, 12):
                h = hb + ho
                # cvt.rn.relu.bf16x2.f32 d, a, b: d.hi = a, d.lo = b, so lo = head h, hi = h + 1.
                lines.append(f"cvt.rn.relu.bf16x2.f32 p, ${1 + h + 1}, ${1 + h};")
                lines.append(f"fma.rn.bf16x2 s0, p, ${33 + h // 2}, s0;")
                lines.append(f"cvt.rn.relu.bf16x2.f32 p, ${1 + h + 3}, ${1 + h + 2};")
                lines.append(f"fma.rn.bf16x2 s1, p, ${33 + (h + 2) // 2}, s1;")
        lines += [
            "add.rn.bf16x2 s, s0, s1;",
            # The halves added in BF16: (lo + hi) in the low half of add.rn.bf16x2(s, swap(s)).
            "prmt.b32 t, s, s, 0x1032;",
            "add.rn.bf16x2 $0, s, t;",
            "}",
        ]
        res = _asm(
            [Uint32(v).ir_value(loc=loc, ip=ip) for v in acc]
            + [Uint32(v).ir_value(loc=loc, ip=ip) for v in w2],
            "\n".join(lines),
            ",".join(["=r"] + ["r"] * 48),
            result=T.i32(),
            loc=loc,
            ip=ip,
        )
        return Uint32(res)

    class ReindexSelectKernel(SinglePassMultiCTARadixTopKClusterKernel):
        def __init__(self, cluster: int, num_sms: int):
            if cluster not in (8, 16):
                raise ValueError("CSA2 reindex select: 8 or 16 CTAs per row")
            super().__init__(
                cutlass.BFloat16,
                WIDTH // cluster,
                TOP_K,
                next_n=1,
                num_copy_bits=256,
                ctas_per_group=cluster,
                num_sms=num_sms,
            )
            if self.ctas_per_group != cluster:
                raise ValueError(f"CSA2 reindex select: no {cluster}-CTA clusters")
            self.num_threads = THREADS
            self.num_warps = THREADS // 32
            self.words = self.chunk_size // 32
            self.tiles = self.chunk_size // 128
            self.warps_per_tile = self.num_warps // self.tiles
            self.cluster_steps = cluster.bit_length() - 1
            self.mma_tiler = (128, HEADS, 128)
            self.acc_stages = 4
            self.sf_cols = (128 // 32) * 2
            self.sfa_base = self.acc_stages * HEADS
            self.sfb_base = self.sfa_base + self.tiles * self.sf_cols
            self.tmem_cols = 256 if self.sfb_base + self.sf_cols <= 256 else 512

        def _setup_mma(self):
            tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                Float4E2M1FN,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                Float8E8M0FNU,
                32,
                tcgen05.CtaGroup.ONE,
                self.mma_tiler[:2],
            )
            a_smem = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, Float4E2M1FN, self.tiles
            )
            b_smem = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, Float4E2M1FN, 1)
            sfa_chunk = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, 32, 1)
            sfb_chunk = blockscaled_utils.make_smem_layout_sfb(tiled_mma, self.mma_tiler, 32, 1)
            sfa_rank3 = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, 32, cute.slice_(sfa_chunk, (None, None, None, 0))
            )
            sfb_rank3 = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, self.mma_tiler, 32, cute.slice_(sfb_chunk, (None, None, None, 0))
            )
            tmem_sfa = cute.append(sfa_rank3, cute.make_layout(1, stride=0))
            tmem_sfb = cute.append(sfb_rank3, cute.make_layout(1, stride=0))
            # SF TMEM K-mode count differs by DSL version (as in fp4_paged_mqa_logits).
            self.sf_k_step = cute.size(sfa_rank3, mode=[2]) // 2
            copy_atom_s2t = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), Float8E8M0FNU
            )
            return (
                tiled_mma,
                copy_atom_s2t,
                (a_smem, b_smem, tmem_sfa, tmem_sfb, sfa_chunk, sfb_chunk),
            )

        @cute.jit
        def __call__(
            self,
            q_ptr: cute.Pointer,
            sfq_ptr: cute.Pointer,
            w_ptr: cute.Pointer,
            pages_ptr: cute.Pointer,
            table_ptr: cute.Pointer,
            req_ptr: cute.Pointer,
            blocks_ptr: cute.Pointer,
            counts_ptr: cute.Pointer,
            pos_ptr: cute.Pointer,
            out_ptr: cute.Pointer,
            rows: Int32,
            table_stride: Int32,
            blocks_stride: Int32,
            pos_stride: Int32,
            page_stride: Int64,
            stream,
        ):
            tiled_mma, copy_atom_s2t, layouts = self._setup_mma()
            self.fused_kernel(
                tiled_mma,
                copy_atom_s2t,
                *layouts,
                q_ptr,
                sfq_ptr,
                w_ptr,
                pages_ptr,
                table_ptr,
                req_ptr,
                blocks_ptr,
                counts_ptr,
                pos_ptr,
                out_ptr,
                rows,
                table_stride,
                blocks_stride,
                pos_stride,
                page_stride,
            ).launch(
                grid=(rows * self.ctas_per_group, 1, 1),
                block=(self.num_threads, 1, 1),
                cluster=(self.ctas_per_group, 1, 1),
                stream=stream,
                use_pdl=TRTLLM_ENABLE_PDL,
            )

        @cute.kernel
        def fused_kernel(
            self,
            tiled_mma: cute.TiledMma,
            copy_atom_s2t: cute.CopyAtom,
            a_smem_layout: cute.ComposedLayout,
            b_smem_layout: cute.ComposedLayout,
            tmem_sfa_layout: cute.Layout,
            tmem_sfb_layout: cute.Layout,
            sfa_chunk_layout: cute.Layout,
            sfb_chunk_layout: cute.Layout,
            q_ptr: cute.Pointer,
            sfq_ptr: cute.Pointer,
            w_ptr: cute.Pointer,
            pages_ptr: cute.Pointer,
            table_ptr: cute.Pointer,
            req_ptr: cute.Pointer,
            blocks_ptr: cute.Pointer,
            counts_ptr: cute.Pointer,
            pos_ptr: cute.Pointer,
            out_ptr: cute.Pointer,
            rows: Int32,
            table_stride: Int32,
            blocks_stride: Int32,
            pos_stride: Int32,
            page_stride: Int64,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            cluster = cutlass.const_expr(self.ctas_per_group)
            top_k = cutlass.const_expr(self.top_k)
            chunk_size = cutlass.const_expr(self.chunk_size)
            num_threads = cutlass.const_expr(self.num_threads)
            num_warps = cutlass.const_expr(self.num_warps)
            words = cutlass.const_expr(self.words)
            width = cutlass.const_expr(WIDTH)
            tiles = cutlass.const_expr(self.tiles)
            acc_stages = cutlass.const_expr(self.acc_stages)
            cta = cute.arch.block_idx_in_cluster()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            lane = tidx % 32
            mma_warp = 8

            # ---- Shared memory: the logits half, then the selection's buffers.
            smem = SmemAllocator()
            nbars = cutlass.const_expr(tiles + 2 + 2 * acc_stages)
            bars = smem.allocate_array(Int64, nbars, byte_alignment=8)
            tmem_holder = smem.allocate_array(Int32, 1, byte_alignment=4)
            sKV = smem.allocate_tensor(Float4E2M1FN, a_smem_layout.outer, 1024, a_smem_layout.inner)
            sQ = smem.allocate_tensor(Float4E2M1FN, b_smem_layout.outer, 1024, b_smem_layout.inner)
            # UE8M0 scales as int32 words in UTCCP order (DeepGEMM's transpose, done by the loads).
            sSFA = smem.allocate_array(Int32, tiles * 128, byte_alignment=128)
            sSFB = smem.allocate_array(Int32, 128, byte_alignment=128)
            sPos = smem.allocate_array(Int32, chunk_size, byte_alignment=128)
            local_histogram = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_ordered_layout((self.radix,), order=(0,)),
                byte_alignment=128,
            )
            # Every CTA's per-round histograms ([round][cta][bin], written by their owners' pushes),
            # the row's histogram of the current round, and every CTA's (above, ties) counts.
            hist_all = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_ordered_layout(
                    (self.num_rounds * self.ctas_per_group * self.radix,), order=(0,)
                ),
                byte_alignment=128,
            )
            merged = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_ordered_layout((self.radix,), order=(0,)),
                byte_alignment=128,
            )
            cta_counts = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_ordered_layout((2 * self.ctas_per_group,), order=(0,)),
                byte_alignment=16,
            )
            prefix_buf = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_ordered_layout((self.radix,), order=(0,)),
                byte_alignment=128,
            )
            s_scalars = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_ordered_layout((4,), order=(0,)),
                byte_alignment=16,
            )
            num_warps_scan = cutlass.const_expr(max(min(self.radix, 256) // 32, self.num_warps))
            s_warp_sums = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_ordered_layout((num_warps_scan,), order=(0,)),
                byte_alignment=128,
            )
            shared_ordered = smem.allocate_tensor(
                element_type=self.ordered_type,
                layout=cute.make_ordered_layout((self.chunk_size,), order=(0,)),
                byte_alignment=128,
            )
            flags = smem.allocate_tensor(cutlass.Uint32, cute.make_layout(words), 16)
            warp_totals = smem.allocate_tensor(Int32, cute.make_layout(num_warps), 16)

            q_bar = tiles
            pos_bar = tiles + 1
            full_bar = tiles + 2
            empty_bar = tiles + 2 + acc_stages
            if tidx == 0:
                for t in cutlass.range_constexpr(tiles):
                    cute.arch.mbarrier_init(bars + t, 32 * self.warps_per_tile)
                cute.arch.mbarrier_init(bars + q_bar, 32)
                cute.arch.mbarrier_init(bars + pos_bar, num_threads)
                for s in cutlass.range_constexpr(acc_stages):
                    cute.arch.mbarrier_init(bars + full_bar + s, 1)
                    cute.arch.mbarrier_init(bars + empty_bar + s, 128)
                cute.arch.mbarrier_init_fence()
            # SFB words of head h at 4 h; the rest stay zero (DeepGEMM zeroes its padded Q scales).
            sfb_t = cute.make_tensor(sSFB, cute.make_layout(128))
            if tidx < 128:
                sfb_t[tidx] = Int32(0)
            for i in range(tidx, self.num_rounds * self.ctas_per_group * self.radix, num_threads):
                hist_all[i] = Int32(0)
            if warp == mma_warp:
                cute.arch.alloc_tmem(self.tmem_cols, tmem_holder)
                cute.arch.relinquish_tmem_alloc_permit()
                tcgen05_before_thread_sync()
            cute.arch.sync_threads()
            # Every CTA's histogram buffers are zero before any push (waited in the first round).
            cute.arch.cluster_arrive()
            tcgen05_after_thread_sync()
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_holder)
            tmem_base = cute.make_tensor(tmem_holder, cute.make_layout(1))[0]

            griddepcontrol_wait()

            row_idx = bidx // cluster
            # Independent loads first (one round trip): the count, this warp's block ids (the
            # publication pads past the count with valid ids) and the requests of the last 32 rows.
            spw = cutlass.const_expr(16 // self.warps_per_tile)
            pre_block = Int32(0)
            if lane < spw:
                blocks_row = cute.make_tensor(
                    blocks_ptr
                    + Int64(row_idx) * Int64(blocks_stride)
                    + cta * (chunk_size // SPARSE_BLOCK),
                    cute.make_layout(chunk_size // SPARSE_BLOCK),
                )
                pre_block = blocks_row[warp * spw + lane]
            req_t = cute.make_tensor(req_ptr, cute.make_layout(rows))
            back = row_idx - lane
            if back < 0:
                back = Int32(0)
            pre_req = req_t[back]
            counts_t = cute.make_tensor(counts_ptr, cute.make_layout(rows))
            length = counts_t[row_idx]
            if length > width:
                length = Int32(width)
            chunk_start = cta * chunk_size
            actual = Int32(0)
            if chunk_start < length:
                actual = length - chunk_start
                if actual > chunk_size:
                    actual = Int32(chunk_size)
            out_row = cute.make_tensor(out_ptr + Int64(row_idx) * top_k, cute.make_layout(top_k))
            pos_row = cute.make_tensor(
                pos_ptr + Int64(row_idx) * Int64(pos_stride), cute.make_layout(width)
            )
            if top_k >= length:
                cute.arch.cluster_wait()
                for i in range(tidx, actual, num_threads):
                    out_row[chunk_start + i] = Int32(pos_row[chunk_start + i])
                if cta == 0:
                    for i in range(length + tidx, top_k, num_threads):
                        out_row[i] = Int32(-1)
            else:
                if tidx < words:
                    flags[tidx] = cutlass.Uint32(0)
                self.logits_to_smem(
                    tiled_mma,
                    copy_atom_s2t,
                    tmem_sfa_layout,
                    tmem_sfb_layout,
                    sfa_chunk_layout,
                    sfb_chunk_layout,
                    sKV,
                    sQ,
                    sSFA,
                    sSFB,
                    sPos,
                    bars,
                    tmem_ptr,
                    tmem_base,
                    shared_ordered,
                    q_ptr,
                    sfq_ptr,
                    w_ptr,
                    pages_ptr,
                    table_ptr,
                    pos_row,
                    row_idx,
                    pre_block,
                    pre_req,
                    actual,
                    table_stride,
                    page_stride,
                    warp,
                    lane,
                )
                cute.arch.barrier()
                # ---- The selection (select_mapped_consumer_topk_'s) over the chunk as the radix
                # select's load_chunk_to_smem leaves it; ties lowest column first, -1 padding last.
                prologue_elems = Int32(0)
                aligned_size = (actual // self.vec_size) * self.vec_size
                left_size = actual - aligned_size
                prefix = cutlass.Uint16(0)
                remaining_k = Int32(top_k)
                for r in cutlass.range_constexpr(self.num_rounds):
                    shift = cutlass.const_expr(self.ordered_bits - (r + 1) * self.radix_bits)
                    prefix, remaining_k = self._radix_round_push(
                        r,
                        shift,
                        shared_ordered,
                        actual,
                        prefix,
                        remaining_k,
                        local_histogram,
                        hist_all,
                        merged,
                        prefix_buf,
                        s_scalars,
                        s_warp_sums,
                        tidx,
                    )
                # Every CTA's columns above the pivot and ties: warp c sums CTA c's round-0 bins
                # below the pivot's high byte and round-1 bins below its low byte.
                b0 = Int32(prefix >> Uint16(8))
                b1 = Int32(prefix & Uint16(0xFF))
                if warp < cluster:
                    h0 = cute.make_tensor(
                        hist_all.iterator + warp * self.radix, cute.make_layout(self.radix)
                    )
                    h1 = cute.make_tensor(
                        hist_all.iterator + (cluster + warp) * self.radix,
                        cute.make_layout(self.radix),
                    )
                    part = Int32(0)
                    for i in cutlass.range_constexpr(self.radix // 32):
                        bin_ = i * 32 + lane
                        if bin_ < b0:
                            part = part + h0[bin_]
                        if bin_ < b1:
                            part = part + h1[bin_]
                    for off in cutlass.range_constexpr(5):
                        part = part + cute.arch.shuffle_sync_bfly(part, offset=1 << (4 - off))
                    if lane == 0:
                        cta_counts[warp] = part
                        cta_counts[cluster + warp] = h1[b1]
                if tidx == 0:
                    local_histogram[2] = Int32(0)
                    local_histogram[_TIE_LIST_COUNT] = Int32(0)
                cute.arch.barrier()
                self._collect_gt_and_list_ties(
                    shared_ordered,
                    Int32(0),
                    prologue_elems,
                    aligned_size,
                    left_size,
                    prefix,
                    local_histogram.iterator + Int32(2),
                    Int32(0),
                    local_histogram,
                    prefix_buf,
                    out_row,
                    None,
                    tidx,
                    flags,
                )
                cute.arch.barrier()
                # The CTAs before this one decide its admitted ties and its first output slot.
                above = Int32(0)
                ties = Int32(0)
                if lane < cluster:
                    above = cta_counts[lane]
                    ties = cta_counts[cluster + lane]
                ties_incl = ties
                for step in cutlass.range_constexpr(self.cluster_steps):
                    up = cute.arch.shuffle_sync_up(
                        ties_incl, 1 << step, mask=0xFFFFFFFF, mask_and_clamp=0
                    )
                    if lane >= (1 << step):
                        ties_incl = ties_incl + up
                admit = remaining_k - (ties_incl - ties)
                if admit > ties:
                    admit = ties
                if admit < 0:
                    admit = Int32(0)
                taken = above + admit
                taken_incl = taken
                for step in cutlass.range_constexpr(self.cluster_steps):
                    up = cute.arch.shuffle_sync_up(
                        taken_incl, 1 << step, mask=0xFFFFFFFF, mask_and_clamp=0
                    )
                    if lane >= (1 << step):
                        taken_incl = taken_incl + up
                need = cute.arch.shuffle_sync(admit, cta)
                base = cute.arch.shuffle_sync(taken_incl - taken, cta)
                total = cute.arch.shuffle_sync(taken_incl, cluster - 1)
                if need > 0:
                    self._write_listed_ties(
                        shared_ordered,
                        Int32(0),
                        prologue_elems,
                        aligned_size,
                        actual,
                        prefix,
                        need,
                        Int32(0),
                        local_histogram,
                        prefix_buf,
                        out_row,
                        None,
                        tidx,
                        flags,
                    )
                cute.arch.mbarrier_wait(bars + pos_bar, 0)
                cute.arch.barrier()
                # Column order: a word's columns start at ``base`` + the earlier words' counts.
                pos_s = cute.make_tensor(sPos, cute.make_layout(chunk_size))
                bits = cutlass.Uint32(0)
                if tidx < words:
                    bits = flags[tidx]
                count = Int32(cute.arch.popc(bits))
                inclusive = count
                for step in cutlass.range_constexpr(5):
                    up = cute.arch.shuffle_sync_up(
                        inclusive, 1 << step, mask=0xFFFFFFFF, mask_and_clamp=0
                    )
                    if lane >= (1 << step):
                        inclusive = inclusive + up
                if lane == 31:
                    warp_totals[warp] = inclusive
                cute.arch.barrier()
                slot = base + inclusive - count
                for w in cutlass.range_constexpr(num_warps):
                    if w < warp:
                        slot = slot + warp_totals[w]
                while bits != cutlass.Uint32(0):
                    lowest = bits & (cutlass.Uint32(0) - bits)
                    column = tidx * 32 + Int32(cute.arch.popc(lowest - cutlass.Uint32(1)))
                    out_row[slot] = pos_s[column]
                    slot = slot + 1
                    bits = bits & (bits - cutlass.Uint32(1))
                if cta == cluster - 1:
                    for k in range(total + tidx, top_k, num_threads):
                        out_row[k] = Int32(-1)
                # No peer touches this CTA's shared memory after the last round's cluster barrier.

            cute.arch.sync_threads()
            if warp == mma_warp:
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols)
            griddepcontrol_launch_dependents()

        @cute.jit
        def _radix_round_push(
            self,
            round_idx: cutlass.Constexpr,
            shift: cutlass.Constexpr,
            shared_ordered,
            actual,
            prefix,
            remaining_k,
            local_histogram,
            hist_all,
            merged,
            prefix_buf,
            s_scalars,
            s_warp_sums,
            tidx,
        ):
            """One radix round: the chunk's histogram under ``prefix`` goes to every CTA; after one
            cluster barrier each CTA sums the row's histogram and picks the threshold."""
            cluster = cutlass.const_expr(self.ctas_per_group)
            prefix_mask = self._compute_prefix_mask(round_idx)
            cta = cute.arch.block_idx_in_cluster()
            for i in range(tidx, self.radix, self.num_threads):
                local_histogram[i] = Int32(0)
            cute.arch.barrier()
            hist_base = local_histogram.iterator.toint()
            for i in range(tidx, actual, self.num_threads):
                ordered = shared_ordered[i]
                if (ordered & prefix_mask) == prefix:
                    bucket = Int32((ordered >> Uint16(shift)) & Uint16(0xFF))
                    red_add_cta(hist_base + bucket * 4, Int32(1))
            cute.arch.barrier()
            if cutlass.const_expr(round_idx == 0):
                # Every CTA's histogram buffers are zero (the arrive after their initialisation).
                cute.arch.cluster_wait()
            # Nonzero 4-bin groups go to every peer as one 16-byte store (its buffers start zeroed).
            mine = hist_all.iterator + (round_idx * cluster + cta) * self.radix
            groups = cutlass.const_expr(self.radix // 4)
            peers_per_part = cutlass.const_expr(
                max(1, self.ctas_per_group * groups // self.num_threads)
            )
            g = tidx % groups
            part = tidx // groups
            if part * peers_per_part < self.ctas_per_group:
                c0 = local_histogram[4 * g]
                c1 = local_histogram[4 * g + 1]
                c2 = local_histogram[4 * g + 2]
                c3 = local_histogram[4 * g + 3]
                if (c0 | c1 | c2 | c3) != Int32(0):
                    for j in cutlass.range_constexpr(peers_per_part):
                        peer = part * peers_per_part + j
                        st_cluster_v4(
                            mapa_shared_cluster(mine + 4 * g, Int32(peer)), c0, c1, c2, c3
                        )
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            found_bucket, found_remaining_k = self._row_threshold(
                hist_all, round_idx, remaining_k, s_warp_sums, s_scalars, tidx
            )
            prefix = prefix | Uint16(Uint16(found_bucket) << Uint16(shift))
            return prefix, found_remaining_k

        @cute.jit
        def _row_threshold(
            self, hist_all, round_idx: cutlass.Constexpr, remaining_k, warp_totals, s_scalars, tidx
        ):
            """``prefix_sum_and_find_threshold``'s (bucket, remaining_k), fallback included, over
            the sum of every CTA's pushed histogram; two threads (a lane pair) per bucket."""
            cluster = cutlass.const_expr(self.ctas_per_group)
            half = cutlass.const_expr(self.ctas_per_group // 2)
            lane = tidx % 32
            warp = tidx // 32
            bucket = tidx // 2
            base = cute.make_tensor(
                hist_all.iterator + round_idx * cluster * self.radix,
                cute.make_layout(cluster * self.radix),
            )
            part = Int32(0)
            for j in cutlass.range_constexpr(half):
                part = part + base[((tidx % 2) * half + j) * self.radix + bucket]
            count = part + cute.arch.shuffle_sync_bfly(part, offset=1)
            # Inclusive scan over the warp's 16 buckets (even lanes carry them).
            inclusive = count
            for step in cutlass.range_constexpr(4):
                up = cute.arch.shuffle_sync_up(
                    inclusive, 2 << step, mask=0xFFFFFFFF, mask_and_clamp=0
                )
                if lane >= (2 << step):
                    inclusive = inclusive + up
            if lane == 31:
                warp_totals[warp] = inclusive
            if tidx == 0:
                s_scalars[0] = Int32(0)
                s_scalars[1] = remaining_k
            cute.arch.barrier()
            before = Int32(0)
            for w in cutlass.range_constexpr(self.num_warps):
                if w < warp:
                    before = before + warp_totals[w]
            inclusive = inclusive + before
            previous = inclusive - count
            if (tidx % 2) == 0:
                if (inclusive >= remaining_k) & (previous < remaining_k):
                    s_scalars[0] = bucket
                    s_scalars[1] = remaining_k - previous
            cute.arch.barrier()
            return s_scalars[0], s_scalars[1]

        @cute.jit
        def logits_to_smem(
            self,
            tiled_mma,
            copy_atom_s2t,
            tmem_sfa_layout,
            tmem_sfb_layout,
            sfa_chunk_layout,
            sfb_chunk_layout,
            sKV,
            sQ,
            sSFA,
            sSFB,
            sPos,
            bars,
            tmem_ptr,
            tmem_base,
            shared_ordered,
            q_ptr,
            sfq_ptr,
            w_ptr,
            pages_ptr,
            table_ptr,
            pos_row,
            row_idx,
            pre_block,
            pre_req,
            actual,
            table_stride,
            page_stride,
            warp,
            lane,
        ):
            """BF16 logits of this CTA's ``actual`` leading columns as ordered keys in
            ``shared_ordered``; also starts the copy of the chunk's positions into ``sPos``."""
            tiles = cutlass.const_expr(self.tiles)
            acc_stages = cutlass.const_expr(self.acc_stages)
            chunk_size = cutlass.const_expr(self.chunk_size)
            wpt = cutlass.const_expr(self.warps_per_tile)
            spw = cutlass.const_expr(16 // wpt)
            mma_warp = 8
            cta = cute.arch.block_idx_in_cluster()
            tidx, _, _ = cute.arch.thread_idx()
            nslots = (actual + SPARSE_BLOCK - 1) // SPARSE_BLOCK
            ntiles = (nslots + 15) // 16
            q_bar = tiles
            pos_bar = tiles + 1
            full_bar = tiles + 2
            empty_bar = tiles + 2 + acc_stages

            # Lane l holds the request of row (row_idx - l): the parity of this row's offset in its
            # request gives its Q pair's first row (DeepGEMM's block-table row).
            mine_req = cute.arch.shuffle_sync(pre_req, 0)
            same = cute.arch.vote_ballot_sync((pre_req == mine_req) & (lane <= row_idx))
            run = ~Uint32(same)
            run_len = Int32(cute.arch.popc((run & (Uint32(0) - run)) - Uint32(1)))
            table_row = row_idx - ((run_len - 1) % 2)

            # ---- Loads: warp w gathers slots [spw w, spw (w + 1)) (8 keys each) of tile w // wpt
            # and their scales; the warps of a tile arrive on its barrier.
            pages = pages_ptr.toint()
            kv_base = sKV.iterator.toint()
            tile = warp // wpt
            if tile < ntiles:
                slot0 = warp * spw
                mine_page = Int32(0)
                if lane < spw:
                    if slot0 + lane < nslots:
                        table = cute.make_tensor(
                            table_ptr + Int64(table_row) * Int64(table_stride),
                            cute.make_layout(1 << 30),
                        )
                        mine_page = table[pre_block // (PAGE_ROWS // SPARSE_BLOCK)]
                for p in cutlass.range_constexpr(spw // 2):
                    j = 2 * p + (lane // 16)
                    blk = cute.arch.shuffle_sync(pre_block, j)
                    pg = cute.arch.shuffle_sync(mine_page, j)
                    if slot0 + j < nslots:
                        src = pages + Int64(pg) * page_stride + Int64((blk % 8) * 512)
                        for half in cutlass.range_constexpr(2):
                            ch = (lane % 16) + 16 * half
                            m = (slot0 + j) % 16 * 8 + ch // 4
                            dst = (
                                kv_base + tile * 8192 + m * 64 + (((ch % 4) ^ ((m // 2) % 4)) * 16)
                            )
                            cp_async_16(dst, src + Int64(ch * 16))
                sfa_tile = sSFA.toint() + tile * 512
                for i in cutlass.range_constexpr(spw * 8 // 32):
                    mw = i * 32 + lane
                    j = mw // 8
                    blk = cute.arch.shuffle_sync(pre_block, j)
                    pg = cute.arch.shuffle_sync(mine_page, j)
                    if slot0 + j < nslots:
                        m = (slot0 % 16) * 8 + mw
                        src = (
                            pages
                            + Int64(pg) * page_stride
                            + Int64(PAGE_ROWS * 64 + (blk % 8) * 32 + (mw % 8) * 4)
                        )
                        cp_async_4(sfa_tile + ((m % 32) * 4 + m // 32) * 4, src)
                cp_async_arrive_noinc((bars + tile).toint())
            if warp == mma_warp:
                # The query (32 heads x 64 bytes, 64-byte swizzle like the keys) and its scales.
                q_row = q_ptr.toint() + Int64(row_idx) * 2048
                q_base = sQ.iterator.toint()
                for i in cutlass.range_constexpr(4):
                    g = lane + 32 * i
                    head = g // 4
                    dst = q_base + head * 64 + (((g % 4) ^ ((head // 2) % 4)) * 16)
                    cp_async_16(dst, q_row + Int64(g * 16))
                cp_async_4(
                    sSFB.toint() + lane * 16, sfq_ptr.toint() + Int64(row_idx * HEADS + lane) * 4
                )
                cp_async_arrive_noinc((bars + q_bar).toint())
            pos_src = pos_row.iterator.toint() + Int64(cta * chunk_size * 4)
            for i in range(tidx, chunk_size // 4, self.num_threads):
                cp_async_16(sPos.toint() + i * 16, pos_src + Int64(i * 16))
            cp_async_arrive_noinc((bars + pos_bar).toint())

            # ---- MMA warp: two K-64 MXF4 MMAs per tile as DeepGEMM splits K (same accumulators).
            if warp == mma_warp:
                tCrA = tiled_mma.make_fragment_A(sKV)
                tCrB = tiled_mma.make_fragment_B(sQ)
                acc_layout = tiled_mma.make_fragment_C(
                    tiled_mma.partition_shape_C(self.mma_tiler[:2])
                ).layout
                sfb_tmem = cute.recast_ptr(tmem_ptr + self.sfb_base, dtype=Float8E8M0FNU)
                tCtSFB = cute.make_tensor(sfb_tmem, tmem_sfb_layout)
                cute.arch.mbarrier_wait(bars + q_bar, 0)
                cute.arch.fence_proxy("async.shared", space="cta")
                sfb_chunk = cute.make_tensor(
                    cute.recast_ptr(sSFB, dtype=Float8E8M0FNU), sfb_chunk_layout
                )
                self._s2t(copy_atom_s2t, sfb_chunk, tCtSFB)
                for t in cutlass.range(ntiles, unroll=1):
                    cute.arch.mbarrier_wait(bars + t, 0)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    sfa_tmem = cute.recast_ptr(
                        tmem_ptr + self.sfa_base + t * self.sf_cols, dtype=Float8E8M0FNU
                    )
                    tCtSFA = cute.make_tensor(sfa_tmem, tmem_sfa_layout)
                    sfa_chunk = cute.make_tensor(
                        cute.recast_ptr(sSFA + t * 128, dtype=Float8E8M0FNU), sfa_chunk_layout
                    )
                    self._s2t(copy_atom_s2t, sfa_chunk, tCtSFA)
                    st = t % acc_stages
                    if t >= acc_stages:
                        cute.arch.mbarrier_wait(bars + empty_bar + st, ((t // acc_stages) - 1) % 2)
                    tcgen05_after_thread_sync()
                    tAcc = cute.make_tensor(tmem_ptr + st * HEADS, acc_layout)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k in cutlass.range_constexpr(2):
                        tCtSFA_k = cute.slice_(tCtSFA, (None, None, k * self.sf_k_step, None))
                        tCtSFB_k = cute.slice_(tCtSFB, (None, None, k * self.sf_k_step, None))
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA_k.iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB_k.iterator)
                        cute.gemm(
                            tiled_mma, tAcc, tCrA[None, None, k, t], tCrB[None, None, k, 0], tAcc
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(bars + full_bar + st)

            # ---- Epilogue warps 0-7: warpgroup g takes tiles g, g + 2, ..; a thread per key row.
            if warp < 8:
                wg = warp // 4
                sub = warp % 4
                w32 = cute.make_tensor(
                    cute.recast_ptr(w_ptr, dtype=Uint32) + Int64(row_idx) * 16, cute.make_layout(16)
                )
                w2 = []
                for i in cutlass.range_constexpr(16):
                    w2.append(Uint32(w32[i]))
                ordered_base = shared_ordered.iterator.toint()
                for t in cutlass.range(wg, ntiles, 2, unroll=1):
                    st = t % acc_stages
                    cute.arch.mbarrier_wait(bars + full_bar + st, (t // acc_stages) % 2)
                    tcgen05_after_thread_sync()
                    acc = tmem_ld_32x32b_x32(tmem_base + st * HEADS + ((sub * 32) << 16))
                    tcgen05_before_thread_sync()
                    cute.arch.mbarrier_arrive(bars + empty_bar + st)
                    bits = head_sum_bf16(acc, w2) & Uint32(0xFFFF)
                    key = bits
                    if (bits & Uint32(0x8000)) == Uint32(0):
                        key = (bits ^ Uint32(0xFFFF)) & Uint32(0x7FFF)
                    col = t * 128 + sub * 32 + lane
                    if col < actual:
                        st_shared_u16(ordered_base + col * 2, key)

        @cute.jit
        def _s2t(self, copy_atom_s2t, smem_chunk, tmem_tensor):
            """UTCCP of one scale-factor chunk (fp4_paged_mqa_logits' S2T copy)."""
            s_compact = cute.filter_zeros(smem_chunk)
            t_compact = cute.filter_zeros(tmem_tensor)
            tiled = tcgen05.make_s2t_copy(copy_atom_s2t, t_compact)
            thr = tiled.get_slice(0)
            src = tcgen05.get_s2t_smem_desc_tensor(tiled, thr.partition_S(s_compact))
            dst = thr.partition_D(t_compact)
            cute.copy(tiled, src, dst)

    return ReindexSelectKernel


def reindex_select_supported(rows: int, width: int, top_k: int, num_sms: int) -> bool:
    return width == WIDTH and top_k == TOP_K and cluster_for_rows(rows, num_sms) > 0


_APPLIES: dict = {}


def reindex_select_applies(rows: int, width: int, top_k: int) -> bool:
    """Whether a candidate consumer's decode rows take the fused launch: the shape fits and the
    two-launch path would select them with ``select_mapped_consumer_topk_``."""
    if rows > _FUSED_SELECT_MAX_ROWS:
        return False
    key = (rows, width, top_k)
    cached = _APPLIES.get(key)
    if cached is not None:
        return cached
    ok = False
    try:
        import cutlass  # noqa: F401

        from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops
        from tensorrt_llm._utils import is_blackwell_datacenter

        from . import decode_select

        num_sms = ops._get_num_sms()
        ok = (
            is_blackwell_datacenter()
            and reindex_select_supported(rows, width, top_k, num_sms)
            and width % 32 == 0
            and width <= decode_select._MAX_COLUMNS
            and rows <= num_sms
            and ops._radix_select_preferred(torch.bfloat16, width, rows)
        )
        if ok:
            runner = ops.CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner
            chunk, ctas, _ = runner._get_chunk_config(
                ops._TORCH_TO_CUTLASS_DTYPE[torch.bfloat16], width, None, 256, rows
            )
            ok = ctas == 1 and chunk is not None and chunk >= width
    except ImportError:
        ok = False
    _APPLIES[key] = ok
    if ok and key not in _LOGGED:
        _LOGGED.add(key)
        from tensorrt_llm.logger import logger

        logger.info(
            f"CSA2 reindex select: candidate logits + Top-{top_k} in one launch "
            f"({rows} rows x {width} columns, {cluster_for_rows(rows, num_sms)} CTAs per row)"
        )
    return ok


_LOGGED: set = set()


def reindex_select_(
    q: torch.Tensor,
    sf_q: torch.Tensor,
    weights: torch.Tensor,
    pages: torch.Tensor,
    table: torch.Tensor,
    row_requests: torch.Tensor,
    blocks: torch.Tensor,
    counts: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    cluster: int | None = None,
) -> bool:
    """Top-512 positions per row into ``output`` [R, 512] int32 from ``q`` [R, 32, 64] MXFP4 and
    index ``pages`` [P, 64, 1, 68]; False, launching nothing, if a capture would compile it."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops

    rows = q.shape[0]
    num_sms = ops._get_num_sms()
    if not reindex_select_supported(rows, WIDTH, output.shape[1], num_sms):
        raise ValueError("CSA2 reindex select: unsupported shape")
    cluster = cluster_for_rows(rows, num_sms) if cluster is None else cluster
    for t, name in ((q, "q"), (sf_q, "sf_q"), (weights, "weights"), (output, "output")):
        if not t.is_contiguous():
            raise ValueError(f"CSA2 reindex select: {name} must be contiguous")
    if pages.stride(1) != 68 or pages.stride(3) != 1 or pages.shape[1] != PAGE_ROWS:
        raise ValueError("CSA2 reindex select: native [P, 64, 1, 68] index pages")
    if table.stride(1) != 1 or blocks.stride(1) != 1 or positions.stride(1) != 1:
        raise ValueError("CSA2 reindex select: row-major table, blocks and positions")
    if positions.stride(0) % 4 or positions.data_ptr() % 16:
        raise ValueError("CSA2 reindex select: 16-byte aligned position rows")
    counts = counts.reshape(-1)
    key = (num_sms, cluster, q.device.index)
    gmem = cute.AddressSpace.gmem
    args = (
        make_ptr(cutlass.Uint8, q.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Int32, sf_q.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.BFloat16, weights.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.Uint8, pages.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Int32, table.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.Int32, row_requests.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.Int32, blocks.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.Int32, counts.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.Int32, positions.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Int32, output.data_ptr(), gmem, assumed_align=4),
        cutlass.Int32(rows),
        cutlass.Int32(table.stride(0)),
        cutlass.Int32(blocks.stride(0)),
        cutlass.Int32(positions.stride(0)),
        cutlass.Int64(pages.stride(0)),
        cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream),
    )
    fn = _COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        fn = _COMPILED[key] = cute.compile(_kernel_class()(cluster, num_sms), *args)
    fn(*args)
    return True
