# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Vendored from vLLM (Apache-2.0):
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/vllm/models/minimax_m3/nvidia/ops/index_decode_score.py
"""CuTe DSL MiniMax-M3 index decode block-scoring kernel (Blackwell SM100).

Computes, for every (index head, decode query token, KV block), the maximum
causally-valid Q . K dot product over the 128 index-K positions of that block.
Those per-block maxima are what minimax_m3_select_blocks ranks to pick the
top-k blocks the sparse attention then attends.

Uses TMA plus warp-level mma.sync rather than tcgen05: the score GEMM's N
dimension is one decode token times a handful of index heads, so CTA occupancy
matters far more than a deep single-CTA pipeline.

Vendored from the vLLM source linked in the file header (v0.26.1rc0-77-g6f91edf96).
Differences from upstream:

* cpasync.make_tiled_tma_atom returns (atom, tensor) in the CuTe DSL version
  pinned here rather than a TmaInfo, so the shared-memory layouts are rebuilt
  from the same compile-time constants instead of being read back off the
  descriptor.
* PDL follows TRTLLM_ENABLE_PDL and grid dependency control comes from
  blackwell.utils rather than cute.arch.
* Compilation and caching live in the trtllm::cute_dsl_minimax_m3_index_decode_score
  runner, matching the other CuTe DSL ops in this tree.
"""

import cutlass
from cuda.bindings.driver import CUstream
from cutlass import Float8E4M3FN, Float16, Float32, Int64, Uint32, cute
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.utils.smem_allocator import SmemAllocator

from .cute_ptx_utils import EVICT_FIRST, fp8x4_to_fp16x4, mma_sync, simple_tma_copy
from .utils import TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait

__all__ = ["IndexDecodeScoreKernel"]


@cute.jit
def _fp8_to_f16_mma_fragments(src: cute.Tensor):
    """Split an FP8 ldmatrix fragment into the two FP16 k-fragments MMA wants."""
    src_elems = cute.size(src)
    src_u32 = cute.recast_tensor(src, Uint32)
    src_f16 = cute.make_rmem_tensor(src_elems, Float16)
    src_f16_u32 = cute.recast_tensor(src_f16, Uint32)
    # Packed conversion; faster and fewer SASS instructions than
    # src.load().to(Float16).
    for i in cutlass.range_constexpr(src_elems // 4):
        converted = fp8x4_to_fp16x4(src_u32[i])
        src_f16_u32[i * 2] = converted[0]
        src_f16_u32[i * 2 + 1] = converted[1]
    lower = cute.make_rmem_tensor(src_elems // 2, Float16)
    upper = cute.make_rmem_tensor(src_elems // 2, Float16)

    # FP8 ldmatrix yields four consecutive values along K. Split each group
    # into the lower two and upper two for the two FP16 MMA k-fragments.
    for i in cutlass.range_constexpr(src_elems // 2):
        lower[i] = src_f16[(i // 2) * 4 + i % 2]
        upper[i] = src_f16[(i // 2) * 4 + 2 + i % 2]
    return lower, upper


class IndexDecodeScoreKernel:
    """Per-block max index score for one decode step.

    Grid is (batch, split_k): CTA (b, s) walks KV blocks
    s, s + split_k, s + 2 * split_k, ... of request b and writes each block's
    score directly, so no cross-CTA reduction is needed. CTAs whose split_id
    exceeds the request's block count exit immediately.
    """

    BLOCK_K = 128
    BAR_MMA = 1
    num_stages = 2

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        num_heads: int,
        max_decode_query_len: int,
        split_k: int,
        head_dim: int = 128,
    ):
        self.dtype = dtype
        self.num_heads = num_heads
        self.max_decode_query_len = max_decode_query_len
        self.split_k = split_k
        self.head_dim = head_dim

    def _swizzle_elems(self) -> int:
        """Elements spanned by one 128-byte swizzle atom."""
        return 128 * 8 // self.dtype.width

    def _sq_layout(self):
        """Composed SMEM layout for the Q tile, shared by the descriptor and the kernel."""
        elems = self._swizzle_elems()
        head_dim = self.head_dim
        block_q = self.num_heads * self.max_decode_query_len
        layout = cute.make_layout(
            (self.max_decode_query_len, self.num_heads, (elems, head_dim // elems)),
            stride=(elems, self.max_decode_query_len * elems, (1, block_q * elems)),
        )
        return cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, layout)

    def _sk_layout(self):
        """Composed SMEM layout for the pipelined K tiles."""
        elems = self._swizzle_elems()
        head_dim = self.head_dim
        block_k = self.BLOCK_K
        layout = cute.make_layout(
            (1, block_k, (elems, head_dim // elems), self.num_stages),
            stride=(0, elems, (1, block_k * elems), block_k * head_dim),
        )
        return cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, layout)

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,  # [bs * runtime_decode_query_len, num_heads, head_dim]
        gK_cache: cute.Tensor,  # [num_pages, page_size, head_dim]
        block_table: cute.Tensor,  # [bs, max_pages]
        score: cute.Tensor,  # [num_heads, bs * runtime_decode_query_len, max_pages]
        seq_lens: cute.Tensor,  # [bs]
        stream: CUstream,
    ):
        num_heads = self.num_heads
        head_dim = self.head_dim
        MAX_DQL = self.max_decode_query_len
        assert num_heads * MAX_DQL <= 32

        batch = seq_lens.shape[0]
        decode_query_len = gQ.shape[0] // batch
        grid = (batch, self.split_k, 1)
        block = (32 * 5, 1, 1)

        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        elems = self._swizzle_elems()

        q_tma_atom, q_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gQ, (None, None, elems)),
            self._sq_layout(),
            (MAX_DQL, num_heads, head_dim),
        )
        k_tma_atom, k_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gK_cache, (None, None, elems)),
            self._sk_layout(),
            (1, self.BLOCK_K, head_dim),
        )

        self.kernel(
            q_tma_atom,
            q_tma_tensor,
            k_tma_atom,
            k_tma_tensor,
            block_table,
            score,
            seq_lens,
            decode_query_len,
        ).launch(grid=grid, block=block, stream=stream, use_pdl=TRTLLM_ENABLE_PDL)

    @cute.kernel
    def kernel(
        self,
        q_tma_atom: cute.CopyAtom,
        q_tma_tensor: cute.Tensor,
        k_tma_atom: cute.CopyAtom,
        k_tma_tensor: cute.Tensor,
        block_table: cute.Tensor,
        score: cute.Tensor,
        seq_lens: cute.Tensor,
        decode_query_len,
    ):
        batch_id, split_id, _ = cute.arch.block_idx()
        _, split_k, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = cute.arch.lane_idx()

        NUM_HEADS = self.num_heads
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = NUM_HEADS * MAX_DQL
        BLOCK_K = self.BLOCK_K
        head_dim = self.head_dim
        dtype = self.dtype
        MMA_N = 8
        num_stages = self.num_stages
        Q_TILES = cute.ceil_div(BLOCK_Q, MMA_N)
        EPI_Q = Q_TILES * MMA_N

        sq_layout = self._sq_layout()
        sk_layout = self._sk_layout()

        smem = SmemAllocator()
        sK = smem.allocate_tensor(
            dtype,
            sk_layout.outer,
            byte_alignment=128,
            swizzle=sk_layout.inner,
        )[0, None, None, None]
        # sQ aliases the first K stage: Q is consumed into registers before the
        # first K tile is needed, so the two never overlap in time.
        sQ_tma = cute.make_tensor(sK[None, None, 0].iterator, layout=sq_layout.outer)
        # TMA sees Q as (query, head, dim) while ldmatrix consumes a flattened Q
        # column mode. The target profile keeps the rank-2 view even for
        # degenerate shapes such as MAX_DQL == 1.
        q_tma_elems = self._swizzle_elems()
        sQ = cute.coalesce(
            cute.group_modes(sQ_tma, 0, 2),
            target_profile=(BLOCK_Q, (q_tma_elems, head_dim // q_tma_elems)),
        )
        epi_buffer = smem.allocate_tensor(Float32, cute.make_layout((EPI_Q, 4)))

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        # TODO: this load precedes the griddepcontrol_wait() below, so under PDL
        # it can observe seq_lens as the predecessor grid left it. num_blocks
        # bounds both the block_table load and the score store, neither of which
        # is otherwise masked, so a stale length here is an out-of-bounds write.
        # Safe only while no PDL predecessor writes seq_lens (nothing between
        # on_update_kv_lens and this kernel does today). Either move the load
        # past the wait, as triton_sparse_decode.py orders its gdc_wait against
        # the same tensor, or record the constraint here.
        seqlen = seq_lens[batch_id]
        num_blocks = cute.ceil_div(seqlen, BLOCK_K)

        if split_id < num_blocks:
            if warp_id == 0:
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(num_stages):
                        cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                        cute.arch.mbarrier_init(tma_empty_mbar + i, 128)
                    cute.arch.mbarrier_init_fence()
            elif warp_id == 1:
                cpasync.prefetch_descriptor(q_tma_atom)
                cpasync.prefetch_descriptor(k_tma_atom)
            cute.arch.sync_threads()

            griddepcontrol_wait()
            # TODO: releasing dependents here, rather than after the epilogue
            # stores, is safe only while no PDL-launched successor reads score.
            # Confirm that and either move the release past the stores or record
            # the constraint here.
            griddepcontrol_launch_dependents()

            if warp_id == 4:
                # TMA warp
                tma_stage = 0
                tma_parity = 1

                gQ_tile = cute.local_tile(
                    cute.domain_offset((batch_id * decode_query_len, 0, 0), q_tma_tensor),
                    tiler=(MAX_DQL, NUM_HEADS, head_dim),
                    coord=(0, 0, 0),
                )
                cute.arch.mbarrier_wait(tma_empty_mbar, tma_parity)
                with cute.arch.elect_one():
                    Q_size = BLOCK_Q * head_dim * (dtype.width // 8)
                    cute.arch.mbarrier_arrive_and_expect_tx(tma_full_mbar, Q_size)
                # TMA bounds-checks rows when the runtime decode_query_len is
                # below MAX_DQL; padded Q columns are masked before the stores.
                simple_tma_copy(q_tma_atom, gQ_tile, sQ_tma, tma_full_mbar)

                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    tma_parity ^= 1

                for block_id in range(split_id, num_blocks, split_k):
                    page_id = block_table[batch_id, block_id]
                    gK_tile = k_tma_tensor[page_id, None, None]
                    k_mbar = tma_full_mbar + tma_stage

                    cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, tma_parity)
                    with cute.arch.elect_one():
                        K_size = BLOCK_K * head_dim * (dtype.width // 8)
                        cute.arch.mbarrier_arrive_and_expect_tx(k_mbar, K_size)
                    simple_tma_copy(
                        k_tma_atom,
                        gK_tile,
                        sK[None, None, tma_stage],
                        k_mbar,
                        cache_policy=EVICT_FIRST,
                    )

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

            else:
                # MMA warps; each handles K[32, head_dim] @ Q[BLOCK_Q, head_dim].T
                sK_warp = cute.local_tile(sK, (32, head_dim, num_stages), (warp_id, 0, 0))
                q_start = seqlen - decode_query_len

                elems = 128 // dtype.width  # 16B
                MMA_K = 32 * 8 // dtype.width  # 32B

                # Pre-compute ldmatrix addresses.
                # sK loads a [16 x 16B] tile:
                #   ((16, (16B, 2), 1), (32 / 16, head_dim / 32B, num_stages))
                # sQ loads an [8 x 32B] tile:
                #   ((8, (16B, 4)), (BLOCK_Q / MMA_N, head_dim / 64B))
                sK_ldsm = cute.zipped_divide(sK_warp, (16, cute.make_layout((elems, 2)), 1))
                sQ_ldsm = cute.zipped_divide(sQ, (MMA_N, cute.make_layout((elems, 4))))

                # sK: (16B, (32 / 16, head_dim / 32B, num_stages))
                # sQ: (16B, (BLOCK_Q / MMA_N, head_dim / 64B))
                sK_ldsm = sK_ldsm[(lane_id % 16, (None, lane_id // 16), 0), None]
                sQ_ldsm = sQ_ldsm[(lane_id % MMA_N, (None, lane_id // 8)), None]

                ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
                ldsm_atom = cute.make_copy_atom(ldsm_op, dtype)

                rQ = cute.make_rmem_tensor(
                    ((elems // 2, 2), head_dim // (MMA_K * 2), Q_TILES), dtype
                )
                rK = cute.make_rmem_tensor((elems, 2, head_dim // MMA_K), dtype)
                rC = cute.make_rmem_tensor((4, 2, Q_TILES), Float32)

                if warp_id == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar, 0)
                cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)
                for q in cutlass.range_constexpr(Q_TILES):
                    cute.copy(ldsm_atom, sQ_ldsm[None, (q, None)], rQ[None, None, q])
                cute.arch.mbarrier_arrive(tma_empty_mbar)

                tma_stage = 1 % self.num_stages
                tma_parity = 0
                if tma_stage == 0:
                    tma_parity ^= 1

                if cutlass.const_expr(dtype is Float8E4M3FN):
                    rQ_f16 = cute.make_rmem_tensor((4, head_dim // MMA_K, Q_TILES, 2), Float16)
                    q_lower, q_upper = _fp8_to_f16_mma_fragments(rQ)
                    rQ_f16[None, None, None, 0].store(q_lower.load())
                    rQ_f16[None, None, None, 1].store(q_upper.load())

                for block_id in range(split_id, num_blocks, split_k):
                    rC.fill(0.0)

                    if warp_id == 0:
                        cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, tma_parity)
                    cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)

                    for k in cutlass.range_constexpr(head_dim // MMA_K):
                        cute.copy(
                            ldsm_atom,
                            sK_ldsm[None, (None, k, tma_stage)],
                            rK[None, None, k],
                        )
                        for m in cutlass.range_constexpr(2):
                            if cutlass.const_expr(dtype is Float8E4M3FN):
                                rK_lower, rK_upper = _fp8_to_f16_mma_fragments(rK[None, m, k])
                                for n in cutlass.range_constexpr(Q_TILES):
                                    rC[None, m, n] = mma_sync(
                                        rK_lower,
                                        rQ_f16[None, k, n, 0],
                                        rC[None, m, n],
                                    )
                                    rC[None, m, n] = mma_sync(
                                        rK_upper,
                                        rQ_f16[None, k, n, 1],
                                        rC[None, m, n],
                                    )
                            else:
                                for n in cutlass.range_constexpr(Q_TILES):
                                    rC[None, m, n] = mma_sync(
                                        rK[None, m, k],
                                        rQ[(None, k % 2), k // 2, n],
                                        rC[None, m, n],
                                    )

                    cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)

                    k_start = block_id * BLOCK_K + warp_id * 32

                    for q in cutlass.range_constexpr(Q_TILES):
                        for i in cutlass.range_constexpr(4):
                            for j in cutlass.range_constexpr(2):
                                col = q * 8 + (lane_id % 4) * 2 + j
                                q_local_pos = col % MAX_DQL
                                q_pos = q_start + q_local_pos
                                k_pos = k_start + i * 8 + lane_id // 4
                                rC[q * 8 + i * 2 + j] = (
                                    rC[q * 8 + i * 2 + j] if q_pos >= k_pos else float("-inf")
                                )

                    for q in cutlass.range_constexpr(Q_TILES):
                        # Thread-local reduction along the BLOCK_K dim.
                        rScore = cute.make_rmem_tensor(2, Float32)
                        rScore.fill(float("-inf"))
                        for i in cutlass.range_constexpr(4):
                            rScore[0] = cute.arch.fmax(rScore[0], rC[i * 2 + 0 + q * 8])
                            rScore[1] = cute.arch.fmax(rScore[1], rC[i * 2 + 1 + q * 8])

                        # Warp reduction among lanes 0, 4, 8, 12, ...
                        for i in cutlass.range_constexpr(3):
                            offset = 4 << i
                            other0 = cute.arch.shuffle_sync_bfly(
                                rScore[0], offset=offset, mask=-1, mask_and_clamp=31
                            )
                            other1 = cute.arch.shuffle_sync_bfly(
                                rScore[1], offset=offset, mask=-1, mask_and_clamp=31
                            )
                            rScore[0] = cute.arch.fmax(rScore[0], other0)
                            rScore[1] = cute.arch.fmax(rScore[1], other1)

                        if lane_id * 2 < MMA_N:
                            epi_buffer[q * MMA_N + lane_id * 2 + 0, warp_id] = rScore[0]
                            epi_buffer[q * MMA_N + lane_id * 2 + 1, warp_id] = rScore[1]
                    cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)

                    head_id = lane_id // MAX_DQL
                    q_local_pos = lane_id - head_id * MAX_DQL
                    valid_q = head_id < NUM_HEADS and q_local_pos < decode_query_len
                    if lane_id < BLOCK_Q and valid_q:
                        final_score = epi_buffer[lane_id, 0]
                        for i in cutlass.range_constexpr(1, 4):
                            final_score = cute.arch.fmax(final_score, epi_buffer[lane_id, i])

                        t = batch_id * decode_query_len + q_local_pos
                        score[head_id, t, block_id] = final_score

                    tma_stage = (tma_stage + 1) % self.num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1
