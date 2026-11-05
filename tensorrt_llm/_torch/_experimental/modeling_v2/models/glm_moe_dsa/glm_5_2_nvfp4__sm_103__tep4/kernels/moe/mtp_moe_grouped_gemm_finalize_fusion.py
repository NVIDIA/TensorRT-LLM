# Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
MTP-layer MoE FC2: grouped BF16 GEMM + finalize scatter, fused.

Replaces production's GEMM2 and its scatter with one kernel. The epilogue applies the
router weight per row and accumulates atomically into the token's output row, so no
separate finalize pass runs.

  For each expert i:
    gemm_out = A_rows[i] x B[i]                       // BF16 x BF16, F32 accumulate
  For each row r in gemm_out:
    token_idx = permuted_idx_to_expanded_idx[r] // topK
    topk_idx  = permuted_idx_to_expanded_idx[r] %  topK
    scale     = token_final_scales[token_idx * topK + topk_idx]
    out[token_idx, :] += bf16( scale * gemm_out[r, :] )              // atomic add

which is production's formula exactly (`sm90_visitor_scatter.hpp`, the
`ScaledAccPerRowBiasPerColScaleScatter` fusion): its `alpha` is literally 1.0 for BF16,
since `moe_kernels.cu:4416-4430` nulls `alpha_scale_ptr_array` on every unquantized path,
and GLM supplies no fc2 bias.

Data types: A, B and C are all BFloat16. A is FC1's output and arrives with no repack --
FC1 writes (N_out, valid_m, 1) with dim 0 contiguous, which is what this kernel declares
for A.

The output must be zero-initialised: the scatter accumulates with atomics because the
topK experts of one token collide on the same row.
"""

from __future__ import annotations

import os

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.experimental import primitives as prims

from ..common import compat48 as _compat48
from .device_bound_scheduler import DeviceBoundPersistentTileScheduler
from .glm_mtp_bf16_utils import a_dtype, b_dtype, c_dtype, compute_ktile_config, mma_inst_k
from .moe_kernel_ops import vectorized_atomic_add_bf16x8_relaxed
from .static_persistent_tile_scheduler import PersistentTileSchedulerParams, WorkTileInfo

# Kernel configuration
smem_capacity = cutlass.memory.get_smem_capacity_in_bytes("sm_100")
num_tmem_alloc_cols = 512

# L2 eviction policies, one per operand stream. The pairing is derived from the reuse
# structure, then measured.
#
#   B (weights)  read exactly once. The 126 permuted rows spread over 57 activated experts
#                give every expert a single M-tile, so the N-tile loop reads each expert's
#                weights once and never returns to them.                  -> evict_first
#   A (tokens)   FC1's output, re-read by every one of the 48 N-tiles per expert. At
#                29.9 MB it fits in 132.6 MB of L2 but is genuinely at risk from the
#                1.43 GB weight stream running past it.                   -> evict_last
#
# Measured on B200 with cold weights and warm tokens, which is what the layer sees in
# serving: evict_last on A alone is about 1.6% faster, evict_first on B alone about 4.3%,
# and both together give no gain over B alone. Both hints protect A's residency, and B's
# is the stronger. A's hint does nothing on FC1 because FC1's gather source is small and
# stays resident anyway.
#
# The mechanism behind B's hint is not established. DRAM traffic and L2 hit rate do not
# change with it, and the effect shows only when launches run back to back. Documented as
# measured for this shape, not derived.
L2_POLICY_A = "evict_last"
L2_POLICY_B = "evict_first"

# 0 = use the measured default (6). Set to sweep pipeline depth; see the use site.
MAX_AB_STAGE = 0

# Programmatic dependent launch, matching TRT-LLM's setting for the production kernels.
# An e2e comparison with this off measures the missing feature rather than the kernel.
USE_PDL = os.environ.get("TRTLLM_ENABLE_PDL", "1").lower() in ("1", "true", "yes", "on")
for _n, _v in (
    ("L2_POLICY_B", L2_POLICY_B),
    ("L2_POLICY_A", L2_POLICY_A),
):
    assert _v in (
        "",
        "evict_first",
        "evict_last",
        "evict_normal",
    ), f"{_n}={_v!r} is not a valid createpolicy.fractional priority"

epilog_warp_id = (0, 1, 2, 3)
mma_warp_id = 4
tma_warp_id = 5
tile_sched_warp_id = 6  # scheduler producer warp
idle_warp_id = 7
threads_per_cta = 256  # 8 warps

# Register reconfig: decrease low-pressure warps, increase epilogue warps.
# SP0: w0(epilog→232) + w4(mma→48) = 8960 regs ≤ 16384
# SP1: w1(epilog→232) + w5(tma→40) = 8704 regs ≤ 16384
# SP2: w2(epilog→232) + w6(sched→80) = 9984 regs ≤ 16384
# SP3: w3(epilog→232) + w7(idle→24) = 8192 regs ≤ 16384
num_regs_epilog = 232
num_regs_mma = 48
num_regs_tma = 40
num_regs_sched = 80
num_regs_idle = 24


@cute.jit
def sched_load_work_tile(
    s_work_tile,
    sched_full_mbar_ptr,
    sched_empty_mbar_ptr,
    sched_full_phase,
):
    """Wait for scheduler warp, load work_tile from SMEM, signal done.

    Inserts fence.acq_rel.cta between the SMEM load and mbarrier_arrive so the
    arrive is ordered after the load.
    """
    while not prims.mbarrier_try_wait_parity(sched_full_mbar_ptr, sched_full_phase):
        pass
    wt_vec = s_work_tile.load(vector_size=4)  # one 128-bit ld.shared
    prims.fence_acq_rel(prims.MemScope.CTA)
    prims.mbarrier_arrive(sched_empty_mbar_ptr)
    work_tile = WorkTileInfo(
        (wt_vec[1], wt_vec[2], wt_vec[3]),
        wt_vec[0] != cutlass.Int32(0),
    )
    return work_tile, sched_full_phase ^ 1


@cute.jit
def host_function(
    a_tensor: cute.Tensor,
    b_tensor: cute.Tensor,
    out_ptr: cutlass.Int64,
    tile_idx_ptr: cutlass.Int64,
    bias_ptr: cutlass.Int64,
    perm_idx_ptr: cutlass.Int64,
    token_scales_ptr: cutlass.Int64,
    mn_limit_ptr: cutlass.Int64,
    problem_k: cutlass.Int32,
    problem_n: cutlass.Int32,
    topK: cutlass.Int32,
    num_tiles_m_ptr: cutlass.Int64,
    max_tiles_m: cutlass.Int32,
    mma_tiler_n: cutlass.Constexpr[int],
    mma_tiler_k: cutlass.Constexpr[int],
    use_blkred: cutlass.Constexpr[bool],
    blkred_bufs: cutlass.Constexpr[int],
    # Whether a bias tensor is actually supplied. GLM's fc2 has none
    # (`moe_kernels.cu:4407-4411` passes a null bias for the BF16 path), so the production
    # configuration sets this False and the epilogue skips the load entirely. Left as a
    # parameter, and tested both ways, because an unconditional prefetch behind a caller
    # that supplies no bias is what produced ~1e32 outputs in the NVFP4 sibling: it read
    # `expert_idx * problem_n` floats past a small allocation, inside torch's caching
    # allocator slab, where memcheck reports nothing.
    has_bias: cutlass.Constexpr[bool],
    use_bias_smem: cutlass.Constexpr[bool],
    epi_subtile_n: cutlass.Constexpr[int],
    raster_along_m: cutlass.Constexpr[bool],
    # SM count, supplied by the caller rather than queried here: a
    # `torch.cuda.get_device_properties` call inside this function runs during
    # `cute.compile`, which would make compiling without a GPU impossible. The caller
    # passes the device SM count.
    num_sms: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
) -> None:
    """Host-side JIT for the grouped BF16 GEMM + finalize scatter.

    Args:
        a_tensor: Input matrix A, shape (K, valid_m, 1), BFloat16.
        b_tensor: Input matrix B, shape (K, N, L), BFloat16.
        out_tensor: Output tensor, shape (seq_len, problem_n), BFloat16.
            Must be zero-initialized.
            Scatter target — rows indexed by original token position.
        tile_idx_to_expert_idx: Int32 (num_tiles_m,) → expert index per M-tile.
        bias: Float32 (L, problem_n) → per-expert output bias.
        permuted_idx_to_expanded_idx: Int32 (valid_m,) → expanded_idx per row.
        token_final_scales: Float32 (seq_len * topK,) → router softmax weights.
        tile_idx_to_mn_limit: Int32 (num_tiles_m,) → row validity bound per tile.
        problem_k: K dimension.
        problem_n: Output width / bias stride in N dimension. May be smaller
            than B's padded N dimension.
        topK: Number of experts per token.
        num_tiles_m: Total M-tiles.
        mma_tiler_n: N dimension of the MMA tile.
        mma_tiler_k: K dimension of the MMA tile (128 or 256).
        use_bias_smem: If True, cooperatively load bias into SMEM for reuse.
        epi_subtile_n: Override for epilogue subtile N (0=default 32).
        raster_along_m: If True, rasterize tiles M-first; if False, N-first.
            N-first (default) co-locates N-tiles sharing the same A data for
            L2 cache reuse.
        stream: CUDA stream.
    """

    # MMA tile configuration for 1CTA
    mma_tiler_mnk = [128, mma_tiler_n, mma_tiler_k]  # Tile dimensions (M, N, K)
    # Per-instruction K granule: 16 for the F16/BF16 atom, against 64 for the NVFP4
    # block-scaled one the sibling kernels use. `Tcgen05InstrDesc.build` +
    # `tcgen05_mma(Tcgen05MMAKind.F16, ...)` consumes 16 K-elements per instruction, so a
    # 64-element K tile is four instructions -- the same instruction count per tile as the
    # NVFP4 kernels reach with a 256-element tile. Carrying 64 over from those would
    # misprogram the whole K loop.
    mma_inst_mnk = [128, mma_tiler_n, mma_inst_k]

    # K-tile config
    _kc = compute_ktile_config(mma_tiler_k, mma_tiler_n)

    # A tile. K-major, L=1 (the permuted rows are one contiguous stack).
    #
    # dtype and tma_format are left to inference here, unlike the NVFP4 kernels. Those
    # have to name `Float4E2M1FN` / `B4X16` explicitly because the default for a scalar
    # FP4 view resolves to `B4X16_P64`, a pairing that is not legal and faults at
    # execution. BF16 has no such ambiguity -- there is one 16-bit format and inference
    # picks it. Naming a format here would be copying a workaround for a problem this
    # kernel does not have.
    tma_a_desc = cuda.create_tensor_map_tiled_from_view(
        a_tensor,
        box_dims=(_kc["tma_b_box_k"], mma_tiler_mnk[0], 1),  # (K_tile, M_tile, L_tile)
        stride_order=(0, 1, 2),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )

    # B tile. K-major, L=num_experts: the TMA warp selects the expert's weights by
    # indexing this dimension with the tile's expert_idx.
    tma_b_desc = cuda.create_tensor_map_tiled_from_view(
        b_tensor,
        box_dims=(_kc["tma_b_box_k"], mma_tiler_mnk[1], 1),  # (K_tile, N_tile, L_tile)
        stride_order=(0, 1, 2),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )

    # No C TMA descriptor — finalize fusion uses atomic scatter instead of TMA store.
    # Pointers (out_ptr, tile_idx_ptr, etc.) are passed as runtime Int64 arguments
    # to avoid JIT caching issues with baked-in addresses.

    # A and B are both BF16 here, so SMEM and GMEM widths coincide; the NVFP4 kernels
    # keep a separate `num_bytes_b_gmem` only because TMA counts transactions against the
    # packed width. At 16 bits the K tile has to be 64: 256 would need 131072 B per stage
    # and leave room for a single stage, which is not a pipeline.
    num_bytes_a = mma_tiler_mnk[0] * mma_tiler_mnk[2] * a_dtype.width // 8
    num_bytes_b_smem = mma_tiler_mnk[1] * mma_tiler_mnk[2] * b_dtype.width // 8
    num_bytes_b_gmem = num_bytes_b_smem
    # SMEM for blkred staging: each epilogue thread owns `blkred_bufs` scratch rows of
    # epi_subtile_n each. 4 warps × 32 = 128 threads.
    #
    # With one buffer the scratch has to be rewritten every subtile, so the next store
    # must wait for the previous `cp.reduce.async.bulk` to finish reading it -- which
    # turns an asynchronous instruction into a synchronous one and makes it *slower*
    # than the per-element atomic path it replaces (measured: +1.28 µs, 4/4 rows).
    # More buffers let consecutive reductions overlap, at the cost of SMEM that comes
    # out of the A/B pipeline depth. Which way that trade lands is a measurement.
    num_epi_threads = 32 * len(epilog_warp_id)  # 128
    num_bytes_c_stage = num_epi_threads * epi_subtile_n * blkred_bufs * c_dtype.width // 8
    # Reserve SMEM for mbar, blkred buffer, bias buffer, and alignment overhead.
    # The kernel allocates sBias_dbuf (aligned to 16) and sC_blkred (aligned to
    # 1024) AFTER the AB pipeline stages. Without accounting for them here, the
    # stage count can overflow SMEM (e.g. N=64 + blkred + bias_smem overflows
    # by 512 bytes due to 1024-byte alignment padding before sC_blkred).
    # Resolve epi_subtile_n = 0 ("use the default") here rather than relying on the
    # caller. A caller that passes 0 would otherwise hit a ZeroDivisionError in the
    # ceil_div below.
    epi_subtile_n = min(epi_subtile_n or 32, mma_tiler_n)
    # `use_bias_smem` stages bias through SMEM and reads `bias_expert_base`
    # unconditionally in the epilogue -- unlike the register path, which is guarded by
    # `has_bias`. With has_bias=False the caller may pass a placeholder pointer, and the
    # staging path would read `expert_idx * problem_n` floats past it, inside torch's
    # caching allocator slab where memcheck reports nothing. That is where the ~1e32
    # outputs in the NVFP4 sibling came from; it carries the same assertion.
    assert not use_bias_smem or has_bias, (
        "use_bias_smem=True requires has_bias=True; the SMEM staging path reads the bias "
        "tensor unconditionally and there is nothing to read when has_bias is False."
    )
    # Double-buffer bias only when there is a next subtile to prefetch.
    _subtile_cnt = cute.ceil_div(mma_tiler_n, epi_subtile_n)
    _bias_bufs = 2 if _subtile_cnt > 1 else 1
    bias_smem_bytes = (_bias_bufs * epi_subtile_n * 4) if use_bias_smem else 0
    smem_reserved = (num_bytes_c_stage if use_blkred else 0) + bias_smem_bytes
    # Tensor buffers need 1024B alignment
    if cutlass.const_expr(bias_smem_bytes % 1024 == 0):
        smem_reserved += 1024
    num_ab_stage = (smem_capacity - smem_reserved) // (num_bytes_a + num_bytes_b_smem)
    # Pipeline depth is capped at 6, below the 7 that SMEM allows, because 7 stages
    # measured about 3% slower. Fewer stages are much slower (2 stages: +63%). The cause
    # of the 7-stage slowdown is not established. The FC1 kernel behaves differently: it
    # still gets faster up to 6 stages, which is its SMEM limit.
    if cutlass.const_expr(MAX_AB_STAGE > 0):
        num_ab_stage = min(num_ab_stage, MAX_AB_STAGE)
    else:
        num_ab_stage = min(num_ab_stage, 6)

    # The caller converts pointers before host_function() to avoid JIT caching.
    # tile_idx_ptr is already an Int64 runtime parameter.

    # Persistent tile scheduling: acc_stages gives MMA/epilogue overlap. One accumulator
    # stage is `mma_tiler_n` TMEM columns of F32, so two stages fit whenever
    # 2*N <= 512.
    #
    # The NVFP4 kernels need a third mechanism here -- `overlapping_accum`, which at
    # N=256 folds the two 256-column regions into 512 columns by letting them overlap on
    # the columns the scale factors occupy, and pays for it with a phase ping-pong and a
    # single logical barrier. That exists only because SFA/SFB consume TMEM. Here they do
    # not, 2*256 is exactly 512, and the two accumulators are simply disjoint. Porting
    # the overlap would be carrying a workaround with nothing left to work around.
    acc_stages = 2 if 2 * mma_tiler_mnk[1] <= num_tmem_alloc_cols else 1
    # Launch only the tiles that produce real output columns.
    n_tiles = cute.ceil_div(problem_n, mma_tiler_mnk[1])

    # Cooperative cluster scheduling (unified pattern for any cluster_m).
    cluster_m = 1
    tile_grid = (max_tiles_m, n_tiles, 1)
    scheduler_cluster = (1, 1, 1)
    tile_sched_params = PersistentTileSchedulerParams(
        tile_grid, scheduler_cluster, raster_along_m=raster_along_m
    )
    max_active_clusters = num_sms // cluster_m
    grid = DeviceBoundPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

    # Launch the kernel
    mtp_moe_grouped_gemm_finalize_fusion_kernel(
        tma_a_desc,
        tma_b_desc,
        tile_idx_ptr,
        bias_ptr,
        out_ptr,
        perm_idx_ptr,
        token_scales_ptr,
        mn_limit_ptr,
        num_tiles_m_ptr,
        problem_k,
        problem_n,
        topK,
        mma_tiler_mnk,
        mma_inst_mnk,
        epi_subtile_n,
        num_bytes_a,
        num_bytes_b_gmem,
        num_bytes_b_smem,
        num_ab_stage,
        use_blkred,
        blkred_bufs,
        has_bias,
        use_bias_smem,
        num_bytes_c_stage,
        tile_sched_params,
        DeviceBoundPersistentTileScheduler,
        acc_stages,
        cluster_m,
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(cluster_m, 1, 1),
        min_blocks_per_mp=1,
        stream=stream,
        use_pdl=USE_PDL,
    )
    return


# GPU device kernel.
#
# See the note on the naming in moe_gather_grouped_gemm_act_fusion.py: the DSL builds the cubin symbol from
# this function's name plus the argument reprs and truncates at 180 characters, so naming
# both MoE kernels `kernel` makes them share one symbol and one profiler row. Keep the
# names distinct.
@cute.kernel
# Named distinctly from the NVFP4 sibling on purpose. CuTe DSL builds the cubin
# symbol from this function's name plus the argument reprs and truncates at 180
# characters, so two kernels called `kernel` with similar leading arguments share
# one symbol and nsys merges them into a single row -- which looks exactly like a
# correct single-kernel trace. The NVFP4 pair already occupies
# `moe_gather_grouped_gemm_act_fusion_kernel` and
# `moe_grouped_gemm_finalize_fusion_kernel`; these carry the `mtp_` prefix
# so a trace with both layers' MoE in it separates all four.
def mtp_moe_grouped_gemm_finalize_fusion_kernel(
    tma_a_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_b_desc: cutlass.GridConstant[cuda.TensorMap],
    tile_idx_ptr: cutlass.Int64,
    bias_ptr: cutlass.Int64,
    out_ptr: cutlass.Int64,
    perm_idx_ptr: cutlass.Int64,
    token_scales_ptr: cutlass.Int64,
    mn_limit_ptr: cutlass.Int64,
    num_tiles_m_ptr: cutlass.Int64,
    problem_k: cutlass.Int32,
    problem_n: cutlass.Int32,
    topK: cutlass.Int32,
    mma_tiler_mnk: cutlass.Constexpr[tuple[int, int, int]],
    mma_inst_mnk: cutlass.Constexpr[tuple[int, int, int]],
    epi_subtile_n: cutlass.Constexpr[int],
    num_bytes_a: cutlass.Constexpr[int],
    num_bytes_b_gmem: cutlass.Constexpr[int],
    num_bytes_b_smem: cutlass.Constexpr[int],
    num_ab_stage: cutlass.Constexpr[int],
    use_blkred: cutlass.Constexpr[bool],
    blkred_bufs: cutlass.Constexpr[int],
    has_bias: cutlass.Constexpr[bool],
    use_bias_smem: cutlass.Constexpr[bool],
    num_bytes_c_stage: cutlass.Constexpr[int],
    tile_sched_params: PersistentTileSchedulerParams,
    scheduler_type: cutlass.Constexpr,
    acc_stages: cutlass.Constexpr[int],
    cluster_m: cutlass.Constexpr[int],
) -> None:
    """GPU device kernel for the MTP layer's contiguous grouped BF16 GEMM.

    Extends a dense GEMM with contiguous grouped GEMM support:
    - tile_idx_to_expert_idx maps each M-tile to an expert index
    - TMA warp uses expert_idx to index B in the L dimension
    - Epilogue applies per-expert bias[expert_idx, :], scales by the router weight,
      and atomically scatters into the token's output row

    Args:
        tma_a_desc: TMA descriptor for matrix A (BF16, K-major, L=1).
        tma_b_desc: TMA descriptor for matrix B (BF16, K-major, L=num_experts).
        tile_idx_ptr: GPU pointer to Int32 array mapping M-tile → expert index.
        bias_ptr: GPU pointer to Float32 per-expert bias (L, problem_n).
        out_ptr: GPU pointer to BFloat16 output (seq_len × problem_n),
            scatter target.
        perm_idx_ptr: GPU pointer to Int32 permuted_idx_to_expanded_idx.
        token_scales_ptr: GPU pointer to Float32 token_final_scales (seq_len*topK).
        mn_limit_ptr: GPU pointer to Int32 tile_idx_to_mn_limit.
        problem_k: K dimension.
        problem_n: Output width / bias stride in N dimension.
        topK: Number of experts per token. Runtime rather than compile-time: the
            production runner passes it as a dynamic value and its cache key omits it, so
            baking it in would silently reuse a kernel built for a different top_k. On the
            device it costs one integer division per row in the epilogue.
        mma_tiler_mnk: Tile dimensions [M, N, K].
        mma_inst_mnk: MMA instruction dimensions [M, N, K].
        epi_subtile_n: Epilog subtile N dimension.
        num_bytes_*: Byte sizes for SMEM allocation.
        num_ab_stage: A/B pipeline depth.
        use_blkred, blkred_bufs: use `cp.reduce.async.bulk` for the scatter instead of
            per-element atomics, and how many staging buffers it gets.
        has_bias: whether a bias tensor is actually supplied. False for GLM.
        use_bias_smem: stage bias through SMEM; requires has_bias.
        tile_sched_params, scheduler_type: persistent tile scheduler configuration.
        acc_stages, cluster_m: accumulator double-buffering and cluster width.

    Kernel Architecture (device-bound persistent scheduler):
        - Block: 256 threads (8 warps)
        - Warp roles: warp 0-3 (epilogue), warp 4 (MMA), warp 5 (TMA),
          warp 6 (tile scheduler producer), warp 7 (idle, donates regs)
        - TMA warp loads B[expert_idx] per tile
        - Epilogue adds bias[expert_idx, :] before type conversion
        - Warp 6 runs DeviceBoundPersistentTileScheduler, writes work_tiles
          to SMEM; consumer warps (0-5) read via mbarrier handshake.
    """
    #
    # K-tile config — computed inline from mma_tiler_mnk at compile time
    #
    _kc = compute_ktile_config(mma_tiler_mnk[2], mma_tiler_mnk[1])
    smem_row_bytes = _kc["smem_row_bytes"]
    slice_row_bytes = _kc["slice_row_bytes"]
    num_k_slices = _kc["num_k_slices"]
    tma_b_box_k = _kc["tma_b_box_k"]
    mma_layout = _kc["mma_layout"]
    mma_stride = _kc["mma_stride"]
    mma_leading_a = _kc["mma_leading_a"]
    mma_leading_b = _kc["mma_leading_b"]
    # k_chunks_per_slice / swizzle_mask are part of the k-tile config but are only
    # needed by FC1's cp.async gather addressing; this kernel loads A by TMA.

    #
    # Thread / block identification
    #
    warp_idx = cute.arch.warp_idx()
    tidx, _, _ = cute.arch.thread_idx()
    k = problem_k

    #
    # Shared memory allocation (mbarriers + data tiles)
    #
    if cutlass.const_expr(use_blkred):
        sC_blkred = cutlass.Array(
            c_dtype,
            num_bytes_c_stage // (c_dtype.width // 8),
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
    # Bias SMEM for pipelined per-subtile cp.async.
    # Double-buffer when subtile_cnt > 1 (pipeline prefetch), else single-buffer.
    _subtile_cnt = cute.ceil_div(mma_tiler_mnk[1], epi_subtile_n)
    _bias_bufs = 2 if _subtile_cnt > 1 else 1
    if cutlass.const_expr(use_bias_smem):
        sBias_dbuf = cutlass.Array(
            cutlass.Float32,
            _bias_bufs * epi_subtile_n,
            space=cutlass.AddressSpace.smem,
            alignment=16,
        )

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    # Warp-specialized scheduler: producer→consumer work_tile via SMEM
    sched_full_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    sched_empty_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    num_tiles_m_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    # TMEM address buffer - will hold the allocated TMEM address
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    s_num_tiles_m = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # Work tile SMEM: [is_valid_i32, tile_coord_m, tile_coord_n, tile_coord_l] — 4×Int32 for one 128-bit ld.shared
    s_work_tile = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    # Expert index for this tile — written by TMA warp, read by epilogue warps

    # Allocate tensors using cutlass (as flat arrays with alignment)
    # Declared as byte arrays, not BF16 arrays: everything downstream -- the stage
    # subviews, the Tcgen05SmemDesc increments, the TMA transaction counts -- is in
    # bytes, so a typed array would only add a conversion at every use.
    sA = cutlass.Array(
        cutlass.Int8,
        num_bytes_a * num_ab_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    sB = cutlass.Array(
        cutlass.Int8,
        num_bytes_b_smem * num_ab_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )

    # Bytes one TMA issue set brings per stage. BF16 is stored as it is read, so the
    # SMEM and GMEM figures coincide; the NVFP4 kernels have to count the packed GMEM
    # width here because their transaction count and their SMEM footprint differ.
    num_tma_load_bytes = num_bytes_a + num_bytes_b_gmem

    k_tile_cnt = cute.ceil_div(k, mma_tiler_mnk[2])

    # TMEM allocation synchronization barrier
    # MMA warp (4) and epilogue warps (0,1,2,3) = 5 warps * 32 threads = 160
    tmem_bar_id = 1
    tmem_bar_threads = 32 * len((mma_warp_id, *epilog_warp_id))

    # Epilog synchronization barrier
    epilog_sync_bar_id = 2
    threads_in_epilogue = 32 * len(epilog_warp_id)

    # Number of consumer warps for sched_empty (warps 0-5 = 6 warps)
    num_consumer_warps = 6

    #
    # Barrier initialization
    #
    # num_tiles_m is written by the PRECEDING kernel (the routing kernel), which under PDL
    # calls cudaTriggerProgrammaticLaunchCompletion() only after all of its global writes --
    # RoutingCustom.cu: "Trigger the secondary kernel AFTER all global memory writes ... The
    # downstream kernels depend on all routing outputs being visible." Its buffer is
    # torch::empty and is never zeroed (cuteDslMoeUtilsOp.cpp). Issuing the cp.async of that
    # word before griddepcontrol("wait") therefore bypasses the only visibility guarantee we
    # have and can return whatever the caching allocator left behind.
    #
    # Only the mbarrier init stays here: it must precede fence_mbarrier_init() /
    # barrier_cta_sync(0) below. The load itself moves to just after the wait.
    if warp_idx == 1:
        if prims.elect_sync():
            prims.mbarrier_init(num_tiles_m_mbar_ptr, 1)

    if warp_idx == 0:
        if prims.elect_sync():
            for i in range(num_ab_stage):
                prims.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
                prims.mbarrier_init(ab_empty_mbar_ptr.subview(i), 1)
            for i in range(acc_stages):
                prims.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
                # All epilogue threads signal acc_empty once T2R is done.
                prims.mbarrier_init(acc_empty_mbar_ptr.subview(i), len(epilog_warp_id) * 32)
            # Warp-specialized scheduler: producer (warp 6) → consumers (warps 0-5)
            prims.mbarrier_init(sched_full_mbar_ptr, 1)
            prims.mbarrier_init(sched_empty_mbar_ptr, 32 * num_consumer_warps)

    # This fence orders the mbar init operations with respect to the barrier below
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    # Placed after the init barrier, matching production: every warp's mbarriers must be
    # constructed before this CTA honours a dependence on the predecessor kernel.
    if cutlass.const_expr(USE_PDL):
        prims.griddepcontrol("wait")

    # Moved out of the barrier-init block: the predecessor's output may only be read after
    # the wait above.
    if warp_idx == 1:
        if prims.elect_sync():
            num_tiles_m_gmem_ptr = cutlass.inttoptr(
                num_tiles_m_ptr, mem_space=1, dtype=cutlass.Int32
            )
            prims.cp_async_shared_global(
                s_num_tiles_m,
                num_tiles_m_gmem_ptr,
                size=4,
                modifier="ca",
                cp_size=4,
            )
            prims.cp_async_mbarrier_arrive(num_tiles_m_mbar_ptr, noinc=True)

    # Initial phase bit states
    ab_mbar_empty_phase_bit = 1
    ab_mbar_full_phase_bit = 0
    acc_full_phase_bit = 0
    acc_empty_phase_bit = 1

    # ---- Construct typed GMEM arrays from Int64 pointers ----
    tile_idx_arr = cutlass.Array(
        cutlass.inttoptr(tile_idx_ptr, mem_space=1, dtype=cutlass.Int32),
        dtype=cutlass.Int32,
    )
    bias_arr = cutlass.Array(
        cutlass.inttoptr(bias_ptr, mem_space=1, dtype=cutlass.Float32),
        dtype=cutlass.Float32,
    )
    out_arr = cutlass.Array(
        cutlass.inttoptr(out_ptr, mem_space=1, dtype=c_dtype),
        dtype=c_dtype,
    )
    perm_idx_arr = cutlass.Array(
        cutlass.inttoptr(perm_idx_ptr, mem_space=1, dtype=cutlass.Int32),
        dtype=cutlass.Int32,
    )
    token_scales_arr = cutlass.Array(
        cutlass.inttoptr(token_scales_ptr, mem_space=1, dtype=cutlass.Float32),
        dtype=cutlass.Float32,
    )
    mn_limit_arr = cutlass.Array(
        cutlass.inttoptr(mn_limit_ptr, mem_space=1, dtype=cutlass.Int32),
        dtype=cutlass.Int32,
    )

    # =========================================================================
    # Warp 7: idle, releases registers.
    # =========================================================================
    if warp_idx == idle_warp_id:
        prims.setmaxregister(num_regs_idle, prims.SetMaxRegisterAction.DECREASE)

    # =========================================================================
    # Warp 6: scheduler producer (tile_sched lives here only).
    # Writes work_tile to SMEM; consumer warps (0-5) read via mbarrier.
    # =========================================================================
    if warp_idx == tile_sched_warp_id:
        prims.setmaxregister(num_regs_sched, prims.SetMaxRegisterAction.DECREASE)
        # Build scheduler entirely inside this warp's scope
        while not prims.mbarrier_try_wait_parity(num_tiles_m_mbar_ptr, 0):
            pass
        actual_tiles_m = s_num_tiles_m.load()
        tile_sched = DeviceBoundPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            actual_tiles_m=actual_tiles_m,
        )
        work_tile = tile_sched.initial_work_tile_info()
        # Write initial work_tile to SMEM and signal consumers
        if prims.elect_sync():
            s_work_tile[0:4] = (
                work_tile.is_valid_tile * cutlass.Int32(1),
                work_tile.tile_idx[0],
                work_tile.tile_idx[1],
                work_tile.tile_idx[2],
            )
            prims.fence_acq_rel(prims.MemScope.CTA)
            prims.mbarrier_arrive(sched_full_mbar_ptr)
        # Persistent loop: advance and write subsequent work_tiles
        sched_empty_phase = 0
        while work_tile.is_valid_tile:
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
            while not prims.mbarrier_try_wait_parity(sched_empty_mbar_ptr, sched_empty_phase):
                pass
            if prims.elect_sync():
                s_work_tile[0:4] = (
                    work_tile.is_valid_tile * cutlass.Int32(1),
                    work_tile.tile_idx[0],
                    work_tile.tile_idx[1],
                    work_tile.tile_idx[2],
                )
                prims.fence_acq_rel(prims.MemScope.CTA)
                prims.mbarrier_arrive(sched_full_mbar_ptr)
            sched_empty_phase ^= 1

    # =========================================================================
    # TMA warp (warp 5): GMEM -> SMEM async loads for A and B
    # =========================================================================
    if warp_idx == tma_warp_id:
        prims.setmaxregister(num_regs_tma, prims.SetMaxRegisterAction.DECREASE)

        # L2 eviction policies. The reuse argument and the measurements behind the
        # defaults are at the top of the file; the only thing local to here is that the
        # primitives package has no `createpolicy` wrapper, so each 64-bit descriptor is
        # built with one PTX instruction.
        if cutlass.const_expr(L2_POLICY_B):
            b_policy = prims.inline_ptx_hl(
                f"createpolicy.fractional.L2::{L2_POLICY_B}.b64 {{$w0}}, 1.0;",
                write_only_types=[cutlass.Int64],
            )
        else:
            b_policy = None
        if cutlass.const_expr(L2_POLICY_A):
            a_policy = prims.inline_ptx_hl(
                f"createpolicy.fractional.L2::{L2_POLICY_A}.b64 {{$w0}}, 1.0;",
                write_only_types=[cutlass.Int64],
            )
        else:
            a_policy = None

        # AB pipeline producer state: stage index carries across tiles.
        ab_stage_idx = 0
        # ---- Persistent loop: TMA warp ----
        # Read initial work_tile from SMEM (written by sched warp)
        work_tile, sched_full_phase = sched_load_work_tile(
            s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, 0
        )
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            # Scheduler returns (m, n, l) — raster order set at host
            tile_coord_m = cur_tile_coord[0]
            tile_coord_n = cur_tile_coord[1]
            # Note: expert_idx could be prefetched here.
            expert_idx = tile_idx_arr.subview(tile_coord_m).load()
            coord_l_a = cutlass.Int32(0)
            coord_l_b = expert_idx
            coord_m = tile_coord_m * mma_tiler_mnk[0]
            coord_n = tile_coord_n * mma_tiler_mnk[1]

            for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                current_ab_stage = ab_stage_idx
                current_ab_empty_phase = ab_mbar_empty_phase_bit
                ab_stage_idx = ab_stage_idx + 1
                if ab_stage_idx == num_ab_stage:
                    ab_stage_idx = 0
                    ab_mbar_empty_phase_bit = ab_mbar_empty_phase_bit ^ 1
                while not prims.mbarrier_try_wait_parity(
                    ab_empty_mbar_ptr.subview(current_ab_stage),
                    current_ab_empty_phase,
                    time_limit=10_000_000,
                ):
                    pass
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(
                        ab_full_mbar_ptr.subview(current_ab_stage), num_tma_load_bytes
                    )

                coord_k = k_tile * mma_tiler_mnk[2]

                sA_staged = sA.subview(num_bytes_a * current_ab_stage)
                sB_staged = sB.subview(num_bytes_b_smem * current_ab_stage)

                if prims.elect_sync():
                    slice_size_a = mma_tiler_mnk[0] * slice_row_bytes
                    for slice_i in cutlass.range_constexpr(num_k_slices):
                        coord_k_slice = coord_k + slice_i * tma_b_box_k
                        sA_slice = sA_staged.subview(slice_i * slice_size_a)
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sA_slice,
                            tma_a_desc.get_ptr(),
                            (coord_k_slice, coord_m, coord_l_a),
                            ab_full_mbar_ptr.subview(current_ab_stage),
                            l2_cache_hint=a_policy,
                        )
                    # TMA load B: per-expert, selected by coord_l_b.
                    slice_size_b = mma_tiler_mnk[1] * slice_row_bytes
                    for slice_i in cutlass.range_constexpr(num_k_slices):
                        coord_k_slice = coord_k + slice_i * tma_b_box_k
                        sB_slice = sB_staged.subview(slice_i * slice_size_b)
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sB_slice,
                            tma_b_desc.get_ptr(),
                            (coord_k_slice, coord_n, coord_l_b),
                            ab_full_mbar_ptr.subview(current_ab_stage),
                            l2_cache_hint=b_policy,
                        )

            # Wait for producer to write next work_tile
            work_tile, sched_full_phase = sched_load_work_tile(
                s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, sched_full_phase
            )

    # =========================================================================
    # MMA warp (warp 4): TMEM alloc, tcgen05.mma compute
    #
    # No S2T stage, unlike the NVFP4 kernels: block-scaled MMA needs the scale factors
    # copied SMEM -> TMEM ahead of every K tile, and that copy plus its TMEM real estate
    # is the bulk of what those kernels' MMA warp does. A dense BF16 MMA reads both
    # operands straight from SMEM descriptors.
    # =========================================================================
    if warp_idx == mma_warp_id:
        prims.setmaxregister(num_regs_mma, prims.SetMaxRegisterAction.DECREASE)
        # Allocate TMEM using cutlass - MMA warp is the allocator (once)
        prims.tcgen05_alloc(tmem_ptr_i32, num_tmem_alloc_cols, group="cta_1")
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")
        # Sync with epilogue warps after allocation
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)
        # Retrieve TMEM pointer from smem buffer (returns cutlass.Array)
        tmem_raw_addr = tmem_ptr_i32.load()
        acc_tmem_ptr = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Float32)

        # MMA instruction descriptor, loop-invariant.
        #
        # `Tcgen05InstrDesc`, not `Tcgen05MxOmmaInstrDesc`: the Mx/Omma descriptors carry
        # scale-format and FP4 k_dim fields and have a different bit layout entirely. They
        # exist for the block-scaled kinds. This is a dense F16/BF16 MMA, so the plain
        # descriptor is the right one, and `c_dtype=Float32` selects the F32 accumulator
        # (the field's default, 0, is the F16-accumulator layout -- which would not fault,
        # it would just accumulate in half precision across a K of 2048).
        #
        # a_major / b_major stay 0 (K-major), matching the s128b SMEM descriptors built
        # below. These two have to agree: a mismatch reads valid SMEM as if transposed and
        # produces wrong numbers with no error anywhere.
        idesc = prims.Tcgen05InstrDesc.build(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=cutlass.Float32,
            n_dim=mma_tiler_mnk[1],
            m_dim=mma_tiler_mnk[0],
        )

        # TMEM base addresses (loop-invariant)
        base_col_id = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        num_kblocks = cute.ceil_div(mma_tiler_mnk[2], mma_inst_mnk[2])

        # Accumulator stage geometry: partition TMEM columns across acc_stages. One F32
        # accumulator column per output column, and nothing else lives in TMEM.
        cols_per_acc_stage = mma_tiler_mnk[1]

        # Build SMEM descriptors once (stage 0).
        desc_a_mma_base = _compat48.tcgen05_smem_desc(
            sA,
            leading_byte_offset=mma_leading_a,
            stride_byte_offset=mma_stride,
            version=0,
            base_offset=0,
            layout=mma_layout,
        )
        desc_b_mma_base = _compat48.tcgen05_smem_desc(
            sB,
            leading_byte_offset=mma_leading_b,
            stride_byte_offset=mma_stride,
            version=0,
            base_offset=0,
            layout=mma_layout,
        )
        stage_inc_a = num_bytes_a
        stage_inc_b = num_bytes_b_smem

        # AB pipeline consumer state: stage index carries across tiles.
        ab_stage_idx = 0
        acc_stage_idx = 0

        # ---- Persistent loop: MMA warp ----
        # Read initial work_tile from SMEM
        work_tile, sched_full_phase = sched_load_work_tile(
            s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, 0
        )
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            # Scheduler returns (m, n, l) — raster order set at host
            tile_coord_n = cur_tile_coord[1]

            # Wait for acc buffer empty (from epilogue) before starting new tile.
            while not prims.mbarrier_try_wait_parity(
                acc_empty_mbar_ptr.subview(acc_stage_idx),
                acc_empty_phase_bit,
                time_limit=10_000_000,
            ):
                pass

            # TMEM column base for the accumulator write region.
            cur_base_col_id = base_col_id + acc_stage_idx * cols_per_acc_stage
            acc_tmem_raw_addr = (base_row_id << 16) | cur_base_col_id
            acc_tmem_ptr_stage = prims.make_tmem_ptr(acc_tmem_raw_addr, cutlass.Float32)

            # Reset accumulator for new tile
            scale_d = False

            for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                current_ab_stage = ab_stage_idx
                current_ab_full_phase = ab_mbar_full_phase_bit
                ab_stage_idx = ab_stage_idx + 1
                if ab_stage_idx == num_ab_stage:
                    ab_stage_idx = 0
                    ab_mbar_full_phase_bit = ab_mbar_full_phase_bit ^ 1

                # Wait for AB buffer full
                while not prims.mbarrier_try_wait_parity(
                    ab_full_mbar_ptr.subview(current_ab_stage),
                    current_ab_full_phase,
                    time_limit=10_000_000,
                ):
                    pass

                # Update SMEM descriptor start addresses for this AB stage.
                desc_a_nvvm_mma_base = desc_a_mma_base.advance_start_address(
                    current_ab_stage * stage_inc_a
                )
                desc_b_nvvm_mma_base = desc_b_mma_base.advance_start_address(
                    current_ab_stage * stage_inc_b
                )

                # Execute MMA over the K tile, one instruction per 16 K-elements.
                #
                # `num_kblocks` counts instructions, so both terms must be in elements:
                # mma_tiler_k / mma_inst_k. The NVFP4 sibling has a note here about a
                # bytes-vs-elements mix that only coincided with the right answer at one
                # tile size; the same trap exists at any width.
                kblocks_per_slice = num_kblocks // num_k_slices
                # Bytes the SMEM descriptor advances per instruction, in 16-byte units.
                # For BF16 K-major this is 16 elements x 2 B = 32 B = 2 units, which is
                # the increment the descriptor documentation states for the F16/BF16
                # K-major path.
                inst_k_bytes_16B = (mma_inst_mnk[2] * a_dtype.width // 8) >> 4
                slice_size_a_16B = mma_tiler_mnk[0] * smem_row_bytes // 16
                slice_size_b_16B = mma_tiler_mnk[1] * smem_row_bytes // 16
                for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                    slice_idx = kblock_idx // kblocks_per_slice
                    kblock_in_slice = kblock_idx % kblocks_per_slice
                    inc_a = slice_idx * slice_size_a_16B + kblock_in_slice * inst_k_bytes_16B
                    inc_b = slice_idx * slice_size_b_16B + kblock_in_slice * inst_k_bytes_16B

                    desc_a_nvvm_mma = desc_a_nvvm_mma_base + inc_a
                    desc_b_nvvm_mma = desc_b_nvvm_mma_base + inc_b

                    if prims.elect_sync():
                        # kind::f16 covers both F16 and BF16 operands -- the operand
                        # format is carried by the instruction descriptor's a_format /
                        # b_format, not by the kind. K is 16 per instruction.
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            acc_tmem_ptr_stage,
                            desc_a_nvvm_mma,
                            desc_b_nvvm_mma,
                            idesc,
                            scale_d,
                        )
                    scale_d = True

                # Async arrive AB buffer empty
                if prims.elect_sync():
                    prims.tcgen05_commit(ab_empty_mbar_ptr.subview(current_ab_stage))

            # Signal acc ready for epilogue
            if prims.elect_sync():
                prims.tcgen05_commit(acc_full_mbar_ptr.subview(acc_stage_idx))

            acc_stage_idx = acc_stage_idx + 1
            if acc_stage_idx == acc_stages:
                acc_stage_idx = 0
                acc_empty_phase_bit = acc_empty_phase_bit ^ 1

            # Wait for producer to write next work_tile
            work_tile, sched_full_phase = sched_load_work_tile(
                s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, sched_full_phase
            )

    # =========================================================================
    # Epilogue warps (warps 0-3): T2R -> scale -> atomic scatter
    # =========================================================================
    if warp_idx < mma_warp_id:
        prims.setmaxregister(num_regs_epilog, prims.SetMaxRegisterAction.INCREASE)
        # Sync with MMA warp after TMEM allocation (once)
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)

        # Retrieve TMEM pointer from smem buffer (same as MMA warp)
        tmem_raw_addr = tmem_ptr_i32.load()
        acc_tmem_ptr = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Float32)

        base_col_id = tmem_raw_addr & 0xFFFF

        # T2R instruction configuration (loop-invariant)
        t2r_inst_shape = "32x32b"
        t2r_inst_repx = epi_subtile_n
        subtile_cnt = cute.ceil_div(mma_tiler_mnk[1], epi_subtile_n)
        epi_thread_idx = tidx

        # Accumulator stage geometry (same as MMA warp)
        cols_per_acc_stage = mma_tiler_mnk[1]

        # Blkred SMEM offset (loop-invariant)
        if cutlass.const_expr(use_blkred):
            sC_row_offset = epi_thread_idx * mma_tiler_mnk[1]

        acc_stage_idx = 0

        # ---- Persistent loop: Epilogue warps ----
        # Read initial work_tile from SMEM
        work_tile, sched_full_phase = sched_load_work_tile(
            s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, 0
        )
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            # Scheduler returns (m, n, l) — raster order set at host
            tile_coord_m = cur_tile_coord[0]
            tile_coord_n = cur_tile_coord[1]
            # Per-tile metadata
            expert_idx = tile_idx_arr.subview(tile_coord_m).load()
            base_m = tile_coord_m * mma_tiler_mnk[0]
            base_n = tile_coord_n * mma_tiler_mnk[1]

            # Per-expert metadata
            bias_expert_base = bias_arr.subview(expert_idx * problem_n)
            mn_limit = mn_limit_arr.subview(tile_coord_m).load()

            abs_row = base_m + epi_thread_idx
            is_valid_row = abs_row < mn_limit
            expanded_idx = cutlass.Int32(0)
            token_idx = cutlass.Int32(0)
            token_scale_val = cutlass.Float32(0.0)
            if is_valid_row:
                expanded_idx = perm_idx_arr.subview(abs_row).load()
                token_idx = expanded_idx // topK
                token_scale_val = token_scales_arr.subview(expanded_idx).load()

            # ---- Bias loading ----
            # Subtiles are always walked forward, low N first. The NVFP4 kernels
            # alternate the direction per accumulator region so that the columns the two
            # regions share get read before the MMA warp overwrites them; with disjoint
            # accumulators there is nothing to read early and the order is free.
            _first_subtile_col = base_n
            # Pipelined bias cp.async: prefetch first subtile into dbuf[0]
            if cutlass.const_expr(use_bias_smem):
                _bias_elts_per_ld = 16 // 4  # 4 F32 per 16B cp.async
                _bias_threads = epi_subtile_n // _bias_elts_per_ld
                if epi_thread_idx < _bias_threads:
                    _pf_src = bias_expert_base.subview(
                        _first_subtile_col + epi_thread_idx * _bias_elts_per_ld
                    )
                    _pf_dst = sBias_dbuf.subview(epi_thread_idx * _bias_elts_per_ld)
                    if _first_subtile_col < problem_n:
                        prims.cp_async_shared_global(
                            _pf_dst,
                            _pf_src,
                            size=16,
                            modifier="cg",
                        )
                prims.cp_async_commit_group()
            else:
                prefetched_bias = cutlass.vector.full([t2r_inst_repx], 0.0, cutlass.Float32)
                if cutlass.const_expr(has_bias):
                    if is_valid_row and base_n < problem_n:
                        prefetched_bias = bias_expert_base.subview(base_n).load(
                            vector_size=t2r_inst_repx,
                            alignment=t2r_inst_repx * 4,
                        )

            # Wait for acc ready from MMA warp
            while not prims.mbarrier_try_wait_parity(
                acc_full_mbar_ptr.subview(acc_stage_idx), acc_full_phase_bit
            ):
                pass

            # TMEM column base for this acc_stage
            cur_base_col_id = base_col_id + acc_stage_idx * cols_per_acc_stage

            # Blkred: compute copy_size_bytes per tile
            if cutlass.const_expr(use_blkred):
                # Base of this thread's `blkred_bufs` scratch rows; the per-subtile
                # buffer is selected inside the loop.
                sC_row_offset = epi_thread_idx * epi_subtile_n * blkred_bufs
                copy_size_bytes = epi_subtile_n * c_dtype.width // 8

            # Epilogue tiling loop, forward over subtiles.
            for subtile_idx in cutlass.range(subtile_cnt):
                real_subtile_idx = subtile_idx
                col_offset = base_n + real_subtile_idx * epi_subtile_n
                # Wait for current subtile's bias data, then prefetch next
                if cutlass.const_expr(use_bias_smem):
                    prims.cp_async_wait_group(0)
                    prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)
                    # Prefetch NEXT subtile into the other half of dbuf.
                    _next_col = col_offset + epi_subtile_n
                    _has_next = _next_col < base_n + mma_tiler_mnk[1]
                    if _has_next:
                        _next_buf_off = ((subtile_idx + 1) % 2) * epi_subtile_n
                        if epi_thread_idx < _bias_threads:
                            _pf_src = bias_expert_base.subview(
                                _next_col + epi_thread_idx * _bias_elts_per_ld
                            )
                            _pf_dst = sBias_dbuf.subview(
                                _next_buf_off + epi_thread_idx * _bias_elts_per_ld
                            )
                            if _next_col + epi_subtile_n <= problem_n:
                                prims.cp_async_shared_global(
                                    _pf_dst,
                                    _pf_src,
                                    size=16,
                                    modifier="cg",
                                )
                        prims.cp_async_commit_group()

                # TMEM -> RMEM (synchronous T2R load) from this acc_stage;
                # helper applies the warp*32 row offset.
                tmem_subtile_ptr = prims.make_tmem_ptr_from_warp_row_col(
                    tmem_raw_addr,
                    warp_idx,
                    cur_base_col_id + real_subtile_idx * t2r_inst_repx,
                    cutlass.Float32,
                )
                t2r_rmem_f32 = prims.tcgen05_ld(t2r_inst_shape, tmem_subtile_ptr, num=t2r_inst_repx)
                if cutlass.const_expr(not use_bias_smem):
                    next_col_offset = col_offset + epi_subtile_n
                    _has_next_ldg = next_col_offset < base_n + mma_tiler_mnk[1]
                    next_bias = cutlass.vector.full([t2r_inst_repx], 0.0, cutlass.Float32)
                    # Same guard as the first prefetch: this is the *next* subtile's
                    # bias, and it feeds `prefetched_bias` for the following iteration.
                    if cutlass.const_expr(has_bias):
                        if _has_next_ldg:
                            if is_valid_row and next_col_offset < problem_n:
                                next_bias = bias_expert_base.subview(next_col_offset).load(
                                    vector_size=t2r_inst_repx,
                                    alignment=t2r_inst_repx * 4,
                                )

                if is_valid_row:
                    if col_offset < problem_n:
                        out_row_base = out_arr.subview(token_idx * problem_n + col_offset)

                        if cutlass.const_expr(use_bias_smem):
                            _cur_buf_off = (subtile_idx % 2) * epi_subtile_n
                            bias_vec = sBias_dbuf.subview(_cur_buf_off).load(
                                vector_size=t2r_inst_repx,
                            )
                        else:
                            bias_vec = prefetched_bias

                        scaled_vec = t2r_rmem_f32 + bias_vec
                        token_scale_vec = cutlass.vector.full_like(scaled_vec, token_scale_val)
                        scaled_vec = scaled_vec * token_scale_vec
                        scaled_bf16 = scaled_vec.to(c_dtype)

                        if cutlass.const_expr(use_blkred):
                            _buf = (subtile_idx % blkred_bufs) * epi_subtile_n
                            sC_elem = sC_blkred.subview(sC_row_offset + _buf)
                            # Before overwriting this buffer, the reduction that last
                            # read it -- group (subtile_idx - blkred_bufs) -- must be
                            # done, which leaves at most blkred_bufs-1 groups pending.
                            # Automatically a no-op for the first blkred_bufs subtiles,
                            # so no guard is needed.
                            prims.cp_async_bulk_wait_group(blkred_bufs - 1, read=True)
                            sC_elem.store(
                                scaled_bf16,
                                vector_size=t2r_inst_repx,
                                alignment=t2r_inst_repx * 2,
                            )
                            cute.arch.fence_view_async_shared()
                            prims.cp_reduce_async_bulk_global_shared_cta(
                                out_row_base,
                                sC_elem,
                                copy_size_bytes,
                                op=prims.CpReduceOp.ADD,
                                type=prims.CpReduceType.BF16,
                                noftz=True,
                            )
                            prims.cp_async_bulk_commit_group()
                        else:
                            _store_n = epi_subtile_n
                            for elem_idx in cutlass.range(0, _store_n, 8, unroll_full=True):
                                atomic_fn = vectorized_atomic_add_bf16x8_relaxed
                                atomic_fn(
                                    out_row_base.subview(elem_idx),
                                    scaled_bf16[elem_idx],
                                    scaled_bf16[elem_idx + 1],
                                    scaled_bf16[elem_idx + 2],
                                    scaled_bf16[elem_idx + 3],
                                    scaled_bf16[elem_idx + 4],
                                    scaled_bf16[elem_idx + 5],
                                    scaled_bf16[elem_idx + 6],
                                    scaled_bf16[elem_idx + 7],
                                )

                if cutlass.const_expr(not use_bias_smem):
                    prefetched_bias = next_bias

            # Signal acc_empty: all T2R reads done, MMA can reuse this acc_stage.
            # Finalize writes only depend on RMEM / scratch, so release the acc
            # stage as soon as T2R is complete.
            prims.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage_idx))

            if cutlass.const_expr(use_blkred):
                prims.cp_async_bulk_wait_group(0, read=True)

            acc_stage_idx = acc_stage_idx + 1
            if acc_stage_idx == acc_stages:
                acc_stage_idx = 0
                acc_full_phase_bit = acc_full_phase_bit ^ 1

            # Wait for producer to write next work_tile
            work_tile, sched_full_phase = sched_load_work_tile(
                s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, sched_full_phase
            )

        # Deallocate TMEM - all epilogue warps sync then one deallocates.
        # Only epilogue warps (warp_idx < mma_warp_id) reach this point, so
        # use the epilogue-scoped barrier rather than a full-CTA sync.
        prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)
        if warp_idx == 0:
            prims.tcgen05_dealloc(acc_tmem_ptr, num_tmem_alloc_cols, group="cta_1")

    # Release the successor kernel as early as this CTA can.
    if cutlass.const_expr(USE_PDL):
        prims.griddepcontrol("launch_dependents")

    return


mtp_moe_grouped_gemm_finalize_fusion_kernel.set_name_prefix(
    "glm52_moe_mtp_bf16_gg2_finalize", remove_cutlass_symbol=True
)
