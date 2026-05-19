# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CuTe DSL rewrite of `fused_inv_rope_fp8_quant_per_head` for DSv4-Pro on SM100.

## V0 — minimal-port, single warp per (token, head)
- Grid: (M_pad, n_groups × heads_per_group, 1); Block: 32 threads (1 warp).
- Each thread reads HEAD_DIM/32 = 16 bf16 elements via direct gmem loads
  (no TMA / no smem in V0).
- Inverse RoPE on the rope portion (NEOX layout, partner reload via gmem).
- Per-128 block-wise absmax via warp shuffle-bfly with strides {1, 2, 4}
  inside each 8-lane sub-warp (one quant block = 8 lanes × 16 elems = 128).
- FP32 → FP8 e4m3fn via `f32_val.to(cutlass.Float8E4M3FN)`.

Bit-equivalent to the Triton BTM=1 kernel
(`_fused_inv_rope_fp8_quant_per_head` in
`tensorrt_llm/_torch/custom_ops/triton_fused_inv_rope_fp8_quant.py`).

## V1.5 (this file) — adds BTM>1 multi-token-per-block path
- BTM dispatch mirrors Triton's `_choose_block_tokens_m`:
  M<1024 → BTM=1, [1024,2048) → 8, [2048,4096) → 16, ≥4096 → 32.
- Grid X = ceil(M_pad / BTM). Each block now processes BTM tokens × 1 head
  with a runtime inner loop. Block is still 1 warp (32 threads).
- Pipelining: each iter pre-loads token (i+1)'s bf16 input into a separate
  register fragment **before** consuming token i, so the compiler can
  overlap the LDG.E.128 issue for next-token data with the RoPE+quant math
  for the current token. Functionally equivalent to Triton's
  `num_stages=2` on the BTM>1 path.
- Why register-prefetch rather than full TMA bulk + smem staging:
  per-CTA tile is BTM×HEAD_DIM×2B (e.g. 16×512×2 = 16 KB) so smem
  staging is overkill, and the framework's TMA-bulk pipeline machinery
  (`pipeline.PipelineTmaAsync`, warp-specialised producer/consumer,
  cluster_layout_vmnk, tx_count mbarriers — see `Sm100BlockwiseGemmKernel`)
  is designed for GEMM. The register-prefetch pattern below mirrors what
  Triton's `num_stages=2` actually generates for the same kernel —
  back-to-back LDG.E.128 issues with reordered consumption — without the
  warp-specialisation overhead.

## Future V2
- True TMA bulk + smem 2-stage circular buffer + mbarrier per stage.
- Multi-warp consumer (4 warps × BTM tokens per warp). Only useful if
  register-prefetch V1.5 still bottlenecks on gmem latency.

## Activation
- Default backend on SM100. Wired up in
  `triton_fused_inv_rope_fp8_quant.py:_fused_inv_rope_fp8_quant_impl`, which
  falls back to the Triton kernel if the cutlass DSL stack is unavailable
  or `TLLM_DISABLE_CUTE_DSL_FUSED_INV_ROPE=1` is set.
"""

from __future__ import annotations

import threading

import torch

from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    def _log2(x: int) -> int:
        assert x > 0 and (x & (x - 1)) == 0, f"{x} is not a power of two"
        out = 0
        while (1 << out) < x:
            out += 1
        return out

    @cute.jit
    def _fabs_f32(x: cutlass.Float32) -> cutlass.Float32:
        """fp32 |x|. cute.math has no abs, so use fmax(x, -x). The compiler
        folds this to a single abs.f32 instruction."""
        return cute.arch.fmax(x, cutlass.Float32(0.0) - x)

    class Sm100FusedInvRopeFp8QuantKernel:
        """CuTe DSL kernel for fused inv-RoPE + 1×128 block-scaled FP8 quant.

        Matches the public op shape contract of
        ``tensorrt_llm/_torch/custom_ops/triton_fused_inv_rope_fp8_quant.py``.
        V0 is a minimal port: 1 warp per (token × head) cell, plain gmem loads.
        """

        def __init__(
            self,
            n_groups: int,
            heads_per_group: int,
            nope_dim: int,
            rope_dim: int,
            quant_group_size: int = 128,
            is_neox: bool = True,
            block_tokens_m: int = 1,
        ):
            self.n_groups = n_groups
            self.heads_per_group = heads_per_group
            self.num_heads = n_groups * heads_per_group
            self.head_dim = nope_dim + rope_dim
            self.nope_dim = nope_dim
            self.rope_dim = rope_dim
            self.quant_group_size = quant_group_size
            self.is_neox = is_neox
            # Multi-token-per-block factor. V0 path is BTM=1; BTM>1 enables
            # the prefetched inner loop (mirror of Triton's BTM>1 kernel).
            self.block_tokens_m = block_tokens_m

            self.chunks_per_head = self.head_dim // self.quant_group_size
            # Same as Triton's `rope_abs_start = (CHUNKS-1)*QGS + ROPE_START`
            # where ROPE_START = nope_dim % QGS. For DSv4-Pro (nope=448,
            # rope=64, qgs=128): rope_abs_start = 3*128 + 64 = 448 = nope_dim.
            self.rope_abs_start = (self.chunks_per_head - 1) * self.quant_group_size + (
                self.nope_dim % self.quant_group_size
            )
            self.half_rope = self.rope_dim // 2
            self.fp8_max = 448.0  # finfo(e4m3fn).max
            self.eps = 1e-10

            # 1 warp per CTA — single-warp BTM>1 mirrors the Triton config
            # (num_warps=2 at BTM>=8 in Triton, but the in-warp shuffle-bfly
            # absmax reduction requires staying within a warp, so we
            # quadruple the per-warp work via BTM instead).
            self.threads_per_block = 32
            # Each thread handles HEAD_DIM / 32 elements (= 16 for HEAD_DIM=512).
            self.elems_per_thread = self.head_dim // self.threads_per_block
            # Threads-per-quant-block: 8 = 128 elems / 16 elems-per-thread.
            self.lanes_per_block = self.quant_group_size // self.elems_per_thread
            # Pre-compute log2 for the butterfly-reduction unroll.
            self.log2_lanes_per_block = _log2(self.lanes_per_block)

        @staticmethod
        def can_implement(num_tokens: int, num_heads: int, head_dim: int) -> bool:
            # Same shape constraints as V0; the BTM>1 path adds no new ones
            # because each block's inner loop iterates the same single-token
            # algorithm. HEAD_DIM must be a multiple of WARP_SIZE (=32) so
            # each thread owns an integer number of elements; HEAD_DIM must
            # also divide cleanly into per-128 quant blocks.
            if head_dim % 128 != 0:
                return False
            if head_dim % 32 != 0:
                return False
            elems_per_thread = head_dim // 32
            # Each thread must own exactly elements from one quant block —
            # i.e. elems_per_thread must divide 128 (the quant block size).
            if 128 % elems_per_thread != 0:
                return False
            return True

        @cute.jit
        def __call__(
            self,
            mA: cute.Tensor,  # bf16, (M, num_heads, head_dim)
            mPos: cute.Tensor,  # int32 or int64, (M,)
            mCosSin: cute.Tensor,  # fp32, (max_pos, rope_dim)
            mFp8: cute.Tensor,  # fp8 e4m3fn, (n_groups, M, d)
            mScales: cute.Tensor,  # fp32, (n_groups, num_scale_blocks, M_pad)
            num_tokens: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            # Grid: (ceil(M_pad / BTM), n_groups * heads_per_group, 1).
            # M_pad is the third dim of the scales buffer, padded by the
            # host wrapper to lcm(BTM, 4) so M_pad % BTM == 0.
            m_pad = mScales.shape[2]
            grid_x = m_pad // self.block_tokens_m
            self.kernel(mA, mPos, mCosSin, mFp8, mScales, num_tokens).launch(
                grid=[grid_x, self.n_groups * self.heads_per_group, 1],
                block=[self.threads_per_block, 1, 1],
                cluster=None,
                smem=0,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mA: cute.Tensor,
            mPos: cute.Tensor,
            mCosSin: cute.Tensor,
            mFp8: cute.Tensor,
            mScales: cute.Tensor,
            num_tokens: cutlass.Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()

            # Pack constants for readability — these are all known at compile
            # time so the loop bodies fully unroll.
            HEAD_DIM = cutlass.const_expr(self.head_dim)
            CHUNKS = cutlass.const_expr(self.chunks_per_head)
            ROPE_ABS_START = cutlass.const_expr(self.rope_abs_start)
            HALF_ROPE = cutlass.const_expr(self.half_rope)
            ROPE_END = cutlass.const_expr(self.rope_abs_start + 2 * self.half_rope)
            ELEMS = cutlass.const_expr(self.elems_per_thread)
            LANES_PER_BLOCK = cutlass.const_expr(self.lanes_per_block)
            HEADS_PER_GROUP = cutlass.const_expr(self.heads_per_group)
            FP8_MAX = cutlass.const_expr(self.fp8_max)
            INV_FP8_MAX = cutlass.const_expr(1.0 / self.fp8_max)
            EPS = cutlass.const_expr(self.eps)
            IS_NEOX = cutlass.const_expr(self.is_neox)
            BTM = cutlass.const_expr(self.block_tokens_m)

            # Each thread handles ELEMS contiguous elements in the head:
            # offsets [tidx * ELEMS, tidx * ELEMS + ELEMS).
            # All ELEMS elements live in the same per-128 quant block (since
            # LANES_PER_BLOCK = 128/ELEMS lanes share the block).
            my_block = tidx // LANES_PER_BLOCK  # 0..CHUNKS-1
            my_lane_in_block = tidx % LANES_PER_BLOCK  # 0..LANES_PER_BLOCK-1
            my_first_offset = tidx * ELEMS

            g = bidy // HEADS_PER_GROUP
            head_in_group = bidy % HEADS_PER_GROUP
            qb_start = head_in_group * CHUNKS

            neg_fp8_max = cutlass.Float32(-FP8_MAX)

            if cutlass.const_expr(BTM == 1):
                # ============ BTM=1 path (V0, byte-identical) ============
                # Grid X already equals m_pad (BTM==1), so pid_token = bidx.
                pid_token = bidx

                # --- Padding-row handling: just zero the scale row this
                # block owns (CHUNKS entries). Use a runtime if/else (no
                # early-return) so the CuTe DSL JIT sees a single linear
                # function body. ---
                if pid_token >= num_tokens:
                    if tidx < CHUNKS:
                        mScales[g, qb_start + tidx, pid_token] = cutlass.Float32(0.0)
                else:
                    # Load position index (may be int32 or int64; the
                    # indexing math is performed in int32 since
                    # max_position fits).
                    pos = cutlass.Int32(mPos[pid_token])

                    # --- 1) Load this thread's ELEMS bf16 elements into a
                    # register fragment (kept as fp32 for the math). ---
                    x_reg = cute.make_rmem_tensor((ELEMS,), cutlass.Float32)
                    for i in cutlass.range_constexpr(ELEMS):
                        offset = my_first_offset + i
                        x_reg[i] = cutlass.Float32(mA[pid_token, bidy, offset])

                    # --- 2) Inverse RoPE on the rope portion. ---
                    if cutlass.const_expr(IS_NEOX):
                        for i in cutlass.range_constexpr(ELEMS):
                            offset = my_first_offset + i
                            # `offset` is a per-thread runtime value
                            # (depends on tidx). The static unroll generates
                            # the `if` body unconditionally; for DSv4-Pro
                            # (rope at offsets [448, 512)) only tids 28..31
                            # ever take the branch.
                            if (offset >= ROPE_ABS_START) and (offset < ROPE_END):
                                rope_local = offset - ROPE_ABS_START
                                is_first_half = rope_local < HALF_ROPE
                                # Arithmetic selection avoids flow-sensitive
                                # variable redefinition inside the if-body.
                                sign_v = (
                                    cutlass.Float32(1.0) if is_first_half else cutlass.Float32(-1.0)
                                )
                                partner_local = (
                                    (rope_local + HALF_ROPE)
                                    if is_first_half
                                    else (rope_local - HALF_ROPE)
                                )
                                cs_idx = rope_local if is_first_half else (rope_local - HALF_ROPE)
                                partner_offset = ROPE_ABS_START + partner_local
                                x_partner = cutlass.Float32(mA[pid_token, bidy, partner_offset])
                                cos_v = mCosSin[pos, cs_idx]
                                sin_v = mCosSin[pos, HALF_ROPE + cs_idx]
                                x_reg[i] = x_reg[i] * cos_v + sign_v * sin_v * x_partner
                    else:
                        # GPT-J / interleaved (pairs (x[2i], x[2i+1])).
                        for i in cutlass.range_constexpr(ELEMS):
                            offset = my_first_offset + i
                            if (offset >= ROPE_ABS_START) and (offset < ROPE_END):
                                rope_local = offset - ROPE_ABS_START
                                partner_offset = offset ^ 1
                                x_partner = cutlass.Float32(mA[pid_token, bidy, partner_offset])
                                cs_idx = rope_local >> 1
                                cos_v = mCosSin[pos, cs_idx]
                                sin_v = mCosSin[pos, HALF_ROPE + cs_idx]
                                is_even = (rope_local & 1) == 0
                                sign_v = cutlass.Float32(1.0) if is_even else cutlass.Float32(-1.0)
                                x_reg[i] = x_reg[i] * cos_v + sign_v * sin_v * x_partner

                    # --- 3) Per-128 absmax via warp shuffle-bfly. Each
                    # per-128 quant block is owned by LANES_PER_BLOCK
                    # adjacent lanes; reduce across them with strides
                    # log2(LANES_PER_BLOCK) — for LANES_PER_BLOCK=8 this is
                    # offsets {1, 2, 4}. ---
                    local_max = cutlass.Float32(0.0)
                    for i in cutlass.range_constexpr(ELEMS):
                        local_max = cute.arch.fmax(local_max, _fabs_f32(x_reg[i]))

                    # Butterfly within the LANES_PER_BLOCK-lane sub-warp.
                    # Static unroll over log2(LANES_PER_BLOCK) stages.
                    for _stage in cutlass.range_constexpr(self.log2_lanes_per_block):
                        other = cute.arch.shuffle_sync_bfly(local_max, offset=1 << _stage)
                        local_max = cute.arch.fmax(local_max, other)

                    absmax = cute.arch.fmax(local_max, cutlass.Float32(EPS))
                    scale_k = absmax * cutlass.Float32(INV_FP8_MAX)
                    inv_scale = cutlass.Float32(1.0) / scale_k

                    # --- 4) Lane 0 of the block writes the scale. ---
                    if my_lane_in_block == 0:
                        mScales[g, qb_start + my_block, pid_token] = scale_k

                    # --- 5) Quantize and store FP8. ---
                    for i in cutlass.range_constexpr(ELEMS):
                        q = x_reg[i] * inv_scale
                        # Clamp to FP8 e4m3fn finite range. Triton uses
                        # `tl.clamp`, which is equivalent to
                        # fmin(fmax(x, -fp8_max), fp8_max).
                        q = cute.arch.fmax(q, neg_fp8_max)
                        # Manual fmin via -fmax(-a, -b) keeps us on the
                        # documented cute.arch surface (no fmin export).
                        q = cutlass.Float32(0.0) - cute.arch.fmax(
                            neg_fp8_max, cutlass.Float32(0.0) - q
                        )
                        offset = my_first_offset + i
                        fp8_offset_in_d = head_in_group * HEAD_DIM + offset
                        mFp8[g, pid_token, fp8_offset_in_d] = q.to(cutlass.Float8E4M3FN)
            else:
                # ============ BTM>1 path (V1.5, prefetch loop) ============
                # Each block owns BTM consecutive tokens of one head.
                # The outer loop iterates at runtime (`cutlass.range`,
                # `unroll=1`) to keep code size bounded for BTM≤32; the
                # per-element inner unrolls stay constexpr for ILP.
                # Cross-iter pipelining: at the top of each iter we issue
                # the NEXT token's bf16 LDG.E.128s into a separate register
                # fragment (`x_next`) BEFORE consuming the CURRENT token's
                # already-resident fragment (`x_curr`). The compiler then
                # has enough independent ops to interleave the load
                # latency with the curr-token RoPE+quant math — matches
                # Triton's `num_stages=2` schedule.
                #
                # `pid_x = bidx`, `pid_token = pid_x * BTM + m_in_block`.
                # m_pad guarantees M_pad % BTM == 0, so the loop has
                # exactly BTM iters per block; per-iter padding-row guard
                # uses the same runtime if/else as BTM=1.
                base_token = cutlass.Int32(bidx) * cutlass.Int32(BTM)

                # Register fragments for the running ("curr") and prefetched
                # ("next") bf16 input slices for this thread (ELEMS f32 each).
                # The bf16 partner (for RoPE) and cos/sin are reloaded per
                # iter from gmem, matching V0 and Triton (the partner reload
                # only touches ≤4 lanes' worth of rope-portion offsets).
                x_curr = cute.make_rmem_tensor((ELEMS,), cutlass.Float32)
                x_next = cute.make_rmem_tensor((ELEMS,), cutlass.Float32)

                # --- Prime the pipeline: load token 0 into x_curr. ---
                tok0 = base_token
                curr_valid = cutlass.Int32(0)
                if tok0 < num_tokens:
                    curr_valid = cutlass.Int32(1)
                    for i in cutlass.range_constexpr(ELEMS):
                        offset = my_first_offset + i
                        x_curr[i] = cutlass.Float32(mA[tok0, bidy, offset])

                # --- Inner loop over the BTM tokens this block owns. ---
                # `unroll=1` → runtime loop (no full unroll). Body is large
                # (ELEMS=16 × per-element unrolls); fully unrolling at
                # BTM=32 would explode code size and spill registers.
                for m_in_block in cutlass.range(0, BTM, 1, unroll=1):
                    pid_token = base_token + cutlass.Int32(m_in_block)

                    # 1) Issue the prefetch for the NEXT token (if any).
                    #    The LDG.E.128s are dispatched ASAP — the compiler
                    #    then interleaves their latency with the
                    #    curr-token math/store below.
                    next_valid = cutlass.Int32(0)
                    if (m_in_block + 1) < BTM:
                        tok_next = base_token + cutlass.Int32(m_in_block + 1)
                        if tok_next < num_tokens:
                            next_valid = cutlass.Int32(1)
                            for i in cutlass.range_constexpr(ELEMS):
                                offset = my_first_offset + i
                                x_next[i] = cutlass.Float32(mA[tok_next, bidy, offset])

                    # 2) Process the CURRENT token using x_curr.
                    if curr_valid == 0:
                        # Padding row: zero the scale entry and skip quant.
                        if tidx < CHUNKS:
                            mScales[g, qb_start + tidx, pid_token] = cutlass.Float32(0.0)
                    else:
                        pos = cutlass.Int32(mPos[pid_token])

                        # Inverse RoPE — same NEOX / interleaved branches as
                        # the BTM=1 path. `x_curr` is mutated in place.
                        if cutlass.const_expr(IS_NEOX):
                            for i in cutlass.range_constexpr(ELEMS):
                                offset = my_first_offset + i
                                if (offset >= ROPE_ABS_START) and (offset < ROPE_END):
                                    rope_local = offset - ROPE_ABS_START
                                    is_first_half = rope_local < HALF_ROPE
                                    sign_v = (
                                        cutlass.Float32(1.0)
                                        if is_first_half
                                        else cutlass.Float32(-1.0)
                                    )
                                    partner_local = (
                                        (rope_local + HALF_ROPE)
                                        if is_first_half
                                        else (rope_local - HALF_ROPE)
                                    )
                                    cs_idx = (
                                        rope_local if is_first_half else (rope_local - HALF_ROPE)
                                    )
                                    partner_offset = ROPE_ABS_START + partner_local
                                    x_partner = cutlass.Float32(mA[pid_token, bidy, partner_offset])
                                    cos_v = mCosSin[pos, cs_idx]
                                    sin_v = mCosSin[pos, HALF_ROPE + cs_idx]
                                    x_curr[i] = x_curr[i] * cos_v + sign_v * sin_v * x_partner
                        else:
                            for i in cutlass.range_constexpr(ELEMS):
                                offset = my_first_offset + i
                                if (offset >= ROPE_ABS_START) and (offset < ROPE_END):
                                    rope_local = offset - ROPE_ABS_START
                                    partner_offset = offset ^ 1
                                    x_partner = cutlass.Float32(mA[pid_token, bidy, partner_offset])
                                    cs_idx = rope_local >> 1
                                    cos_v = mCosSin[pos, cs_idx]
                                    sin_v = mCosSin[pos, HALF_ROPE + cs_idx]
                                    is_even = (rope_local & 1) == 0
                                    sign_v = (
                                        cutlass.Float32(1.0) if is_even else cutlass.Float32(-1.0)
                                    )
                                    x_curr[i] = x_curr[i] * cos_v + sign_v * sin_v * x_partner

                        # Per-128 absmax via warp shuffle-bfly.
                        local_max = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(ELEMS):
                            local_max = cute.arch.fmax(local_max, _fabs_f32(x_curr[i]))
                        for _stage in cutlass.range_constexpr(self.log2_lanes_per_block):
                            other = cute.arch.shuffle_sync_bfly(local_max, offset=1 << _stage)
                            local_max = cute.arch.fmax(local_max, other)

                        absmax = cute.arch.fmax(local_max, cutlass.Float32(EPS))
                        scale_k = absmax * cutlass.Float32(INV_FP8_MAX)
                        inv_scale = cutlass.Float32(1.0) / scale_k

                        # Lane 0 of the per-128 block writes the scale.
                        if my_lane_in_block == 0:
                            mScales[g, qb_start + my_block, pid_token] = scale_k

                        # Quantize and store FP8.
                        for i in cutlass.range_constexpr(ELEMS):
                            q = x_curr[i] * inv_scale
                            q = cute.arch.fmax(q, neg_fp8_max)
                            q = cutlass.Float32(0.0) - cute.arch.fmax(
                                neg_fp8_max, cutlass.Float32(0.0) - q
                            )
                            offset = my_first_offset + i
                            fp8_offset_in_d = head_in_group * HEAD_DIM + offset
                            mFp8[g, pid_token, fp8_offset_in_d] = q.to(cutlass.Float8E4M3FN)

                    # 3) Promote x_next → x_curr for the next iter.
                    #    The element-wise copy is a register-rename; the
                    #    compiler folds it away when x_next has a single use.
                    curr_valid = next_valid
                    for i in cutlass.range_constexpr(ELEMS):
                        x_curr[i] = x_next[i]

    # ------------------------------------------------------------------------
    # Python wrapper. Mirrors the Triton wrapper's signature so it's a
    # drop-in replacement under the env gate.
    # ------------------------------------------------------------------------
    # Compile cache keyed on the static kernel-shape constants. The kernel
    # body depends only on the (shape, dtype) tuple — M is dynamic — so we
    # don't need to re-JIT per batch size.
    _compile_cache = {}
    _compile_cache_lock = threading.Lock()

    def _tma_aligned_size(x: int, align: int = 4) -> int:
        return (x + align - 1) // align * align

    def _choose_block_tokens_m(num_tokens: int) -> int:
        """Mirror of Triton's `_choose_block_tokens_m` in
        ``triton_fused_inv_rope_fp8_quant.py`` (lines ~300-308). At small M
        the BTM=1 path wins (no inner-loop overhead); at M >= 1024 multi-
        token blocks reduce grid size which lifts SM occupancy. Same
        thresholds as the Triton dispatcher so behaviour is comparable
        in A/B microbenches.
        """
        if num_tokens >= 4096:
            return 32
        if num_tokens >= 2048:
            return 16
        if num_tokens >= 1024:
            return 8
        return 1

    def _convert_input_tensor(t: torch.Tensor, assumed_align: int = 16) -> "cute.Tensor":
        """Wrap a torch tensor for CuTe DSL kernel use.

        Uses `mark_layout_dynamic` so the kernel can be reused across
        different M values without re-JIT. (More precise per-mode dynamism
        via `mark_compact_shape_dynamic` is possible but adds complexity for
        V0; revisit in V2 along with full TMA.)
        """
        return from_dlpack(t.detach(), assumed_align=assumed_align).mark_layout_dynamic()

    def _fused_inv_rope_fp8_quant_impl_cute_dsl(
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        n_groups: int,
        heads_per_group: int,
        nope_dim: int,
        rope_dim: int,
        quant_group_size: int,
        is_neox: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CuTe DSL backend. Same I/O contract as
        ``_fused_inv_rope_fp8_quant_impl`` in the Triton file."""
        num_tokens, num_heads, head_dim = o.shape
        assert num_heads == n_groups * heads_per_group, (
            f"num_heads={num_heads} != n_groups({n_groups}) * heads_per_group({heads_per_group})"
        )
        assert head_dim == nope_dim + rope_dim
        assert head_dim % quant_group_size == 0
        assert Sm100FusedInvRopeFp8QuantKernel.can_implement(num_tokens, num_heads, head_dim), (
            f"Sm100FusedInvRopeFp8QuantKernel cannot implement shape "
            f"(num_tokens={num_tokens}, num_heads={num_heads}, head_dim={head_dim})"
        )

        d = heads_per_group * head_dim
        num_scale_blocks = d // quant_group_size

        # BTM dispatch (mirror of Triton). Same thresholds → same A/B knob.
        block_tokens_m = _choose_block_tokens_m(num_tokens)
        # Pad M to max(BTM, 4): consumer needs 4-alignment, and each grid-X
        # block writes BTM scale rows → M must align to BTM too. lcm(BTM,
        # 4) = max(BTM, 4) for BTM ∈ {1, 2, 4, 8, 16, 32} (powers of 2).
        tma_aligned_T = _tma_aligned_size(num_tokens, max(block_tokens_m, 4))

        fp8_buf = torch.empty(
            (n_groups, num_tokens, d),
            dtype=torch.float8_e4m3fn,
            device=o.device,
        )
        scale_buf = torch.empty(
            (n_groups, num_scale_blocks, tma_aligned_T),
            dtype=torch.float32,
            device=o.device,
        )

        # Flatten cos_sin_cache last two dims (TRT-LLM stores it as
        # [max_pos, 2, half_rope]; the kernel wants [max_pos, rope_dim]).
        cos_sin_view = cos_sin_cache.view(cos_sin_cache.shape[0], -1)
        assert cos_sin_view.shape[-1] == rope_dim

        # Normalise positions dtype (CuTe DSL prefers int32 / int64).
        if positions.dtype != torch.int32 and positions.dtype != torch.int64:
            positions = positions.to(torch.int64)
        positions = positions.contiguous()

        # `o`, `fp8_buf`, `scale_buf`, `cos_sin_view` are all torch.empty/
        # torch.randn allocations, so the underlying caching allocator hands
        # back 256-byte-aligned pointers — 16-byte alignment is trivially
        # satisfied. `positions` may be a smaller allocation; cap its
        # assumed_align at 4 bytes (the int32/int64 natural alignment).
        a_tensor = _convert_input_tensor(o)
        pos_tensor = _convert_input_tensor(positions, assumed_align=4)
        cos_sin_tensor = _convert_input_tensor(cos_sin_view)
        fp8_tensor = _convert_input_tensor(fp8_buf)
        scale_tensor = _convert_input_tensor(scale_buf)
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # Compile-cache key: static shape constants + BTM. M is dynamic
        # via mark_layout_dynamic so we only need to re-JIT when BTM
        # changes (i.e. crossing one of the 1024/2048/4096 thresholds).
        cache_key = (
            n_groups,
            heads_per_group,
            nope_dim,
            rope_dim,
            quant_group_size,
            is_neox,
            block_tokens_m,
            o.dtype,
            positions.dtype,
        )
        with _compile_cache_lock:
            compiled = _compile_cache.get(cache_key)
        if compiled is None:
            kernel = Sm100FusedInvRopeFp8QuantKernel(
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
                quant_group_size=quant_group_size,
                is_neox=is_neox,
                block_tokens_m=block_tokens_m,
            )
            new_compiled = cute.compile(
                kernel,
                a_tensor,
                pos_tensor,
                cos_sin_tensor,
                fp8_tensor,
                scale_tensor,
                cutlass.Int32(num_tokens),
                current_stream,
            )
            with _compile_cache_lock:
                compiled = _compile_cache.setdefault(cache_key, new_compiled)

        compiled(
            a_tensor,
            pos_tensor,
            cos_sin_tensor,
            fp8_tensor,
            scale_tensor,
            cutlass.Int32(num_tokens),
            current_stream,
        )
        return fp8_buf, scale_buf

else:
    Sm100FusedInvRopeFp8QuantKernel = None  # type: ignore[assignment]
    _fused_inv_rope_fp8_quant_impl_cute_dsl = None  # type: ignore[assignment]
