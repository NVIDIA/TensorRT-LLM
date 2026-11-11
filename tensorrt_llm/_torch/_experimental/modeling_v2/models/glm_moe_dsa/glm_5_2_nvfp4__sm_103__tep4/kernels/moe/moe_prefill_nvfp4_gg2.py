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

"""Compact-M128 GLM-5.2 prefill NVFP4 GG2 implementation.

The paired entry in moe_prefill_nvfp4_pair.py is the supported compilation
boundary. It selects CTA1, the local scheduler, mixed K96+K96+K64 issue, K256,
N128/N256, and weighted BF16 bulk-atomic finalize without bias. GG2 consumes the
current GG1 packed values, NVFP4 scale atoms and compact 128-row routing metadata;
its output must be zeroed, and that dependency joined, before accumulation.

Inherited constexpr alternatives remain implementation details to preserve the
compiled specialization, not supported M256 or alternate-scheduler entry points.

The B-weight policy is fixed to evict_first independently of GG1's
policy. TRTLLM_ENABLE_PDL is read once at import; callers must select one PDL mode
for the pair and retain the original producer/output-zero dependency contract.
"""

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import torch
from cutlass.experimental import primitives as prims

from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL as _TRTLLM_ENABLE_PDL

from ..common import compat48 as _compat48
from .device_bound_scheduler import DeviceBoundPersistentTileScheduler
from .glm_nvfp4_utils import (
    a_dtype,
    b_dtype,
    compute_ktile_config,
    compute_sf_rest,
    fp4_lanes_per_storage,
    fp4x2_dtype,
    num_elts_atom_sf_e8,
    num_elts_atom_sf_fp16,
    num_k_per_sf_atom,
    num_m0_per_sf_atom,
    num_m1_per_sf_atom,
    num_tmem_cols_per_sf_atom,
    sf_dtype,
    sf_vec_size,
)
from .moe_kernel_ops import vectorized_atomic_add_bf16x8_relaxed
from .static_persistent_tile_scheduler import PersistentTileSchedulerParams, WorkTileInfo

# ---------------------------------------------------------------------------
# GLM-5.2 data types -- deliberately NOT imported from moe_utils, which defines the
# GPT-OSS configuration (A = FP8E4M3 against FP4 B, SF = E8M0 MX at vec 32).
#
# GLM-5.2 is NVFP4 end to end. FC2's A operand is FC1's C output, so these must
# stay in lockstep with moe_prefill_nvfp4_gg1.py: same sf_vec_size, same
# sf_dtype. If they
# diverge, gg2 reads gg1's SFC with the wrong atom stride and the two kernels
# silently disagree about the layout of the tensor they share.
#
#   moe_utils (GPT-OSS)                 this file (GLM-5.2)
#   a_dtype     = Float8E4M3FN     ->   Float4E2M1FN   (mixed-precision -> not)
#   b_dtype     = Float4E2M1FN     ->   Float4E2M1FN   (unchanged)
#   sf_dtype    = Float8E8M0FNU    ->   Float8E4M3FN   (MX -> NVFP4)
#   sf_vec_size = 32               ->   16
#
# Dropping mixed precision also means the 4-bit operand no longer has to be padded
# to the 8-bit operand's width for tcgen05.mma -- production rejects a_dtype != b_dtype
# outright, so one byte count describes both B's TMA transfer and SMEM tile.
# ---------------------------------------------------------------------------

# Output C stays BFloat16: FC2 emits the final MoE result, scattered and summed
# across topK. This was already the reference's dtype, not a GLM change.
c_dtype = cutlass.BFloat16


# Kernel configuration
smem_capacity = cutlass.memory.get_smem_capacity_in_bytes("sm_100")
num_tmem_alloc_cols = 512

# Warp specialization: warps 0-3 epilog, 4 MMA, 5 TMA, 6 scheduler, 7 idle

# Fixed GG2 weight policy, independent of the paired GG1 policy.
L2_POLICY_B = "evict_first"

# Programmatic dependent launch, matching TRT-LLM's setting for the production kernels.
# Set the mode before either paired kernel is imported; never mutate it later.
USE_PDL = bool(_TRTLLM_ENABLE_PDL)

epilog_warp_id = (0, 1, 2, 3)
mma_warp_id = 4
tma_warp_id = 5
tile_sched_warp_id = 6  # scheduler producer warp
idle_warp_id = 7
threads_per_cta = 256  # 8 warps

# Register reconfig: epilogue warps increase to 232, other warps decrease.
num_regs_epilog = 232
num_regs_mma = 48
num_regs_tma = 40
num_regs_sched = 80
num_regs_idle = 24

# Scheduler and instruction choices are explicit default-off specializations.
# Import-time environment changes never select these two mechanisms.


@cute.jit
def local_work_tile(
    linear_idx,
    actual_tiles_m,
    n_tiles,
    major_divisor,
    raster_along_m: cutlass.Constexpr[bool],
):
    """Decode the same valid tile order without mutating shared scheduler params.

    The host launches cluster=(1,1,1), with logical worker ID in blockIdx.z.
    L is one, so the second modulo in the generic scheduler is redundant for
    valid work. Invalid sentinel coordinates are never dereferenced. A divisor
    of one is used for an empty M dimension so empty routing remains legal.
    """
    minor, major = divmod(linear_idx, major_divisor)
    if cutlass.const_expr(raster_along_m):
        tile_m, tile_n = major, minor
    else:
        tile_m, tile_n = minor, major
    return WorkTileInfo(
        (
            cute.arch.make_warp_uniform(tile_m),
            cute.arch.make_warp_uniform(tile_n),
            cutlass.Int32(0),
        ),
        cute.arch.make_warp_uniform(linear_idx < actual_tiles_m * n_tiles),
    )


@cute.jit
def sched_load_work_tile(
    s_work_tile,
    sched_full_mbar_ptr,
    sched_empty_mbar_ptr,
    sched_full_phase,
):
    """Wait for scheduler warp, load work_tile from SMEM, signal done.

    Inserts fence.acq_rel.cta between the shared-memory load and
    mbarrier_arrive so the load is ordered before the arrive.
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
    sfa_tensor: cute.Tensor,
    sfb_tensor: cute.Tensor,
    out_ptr: cutlass.Int64,
    tile_idx_ptr: cutlass.Int64,
    bias_ptr: cutlass.Int64,
    perm_idx_ptr: cutlass.Int64,
    token_scales_ptr: cutlass.Int64,
    # Per-expert dequant scale, folded into the same product as the token scale
    # (`sMetaScale = alpha_val * token_scale`, indexed by expert_idx).
    alpha_ptr: cutlass.Int64,
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
    # Whether a bias tensor is actually supplied. Without a bias the bias prefetch
    # must not run: `bias_expert_base` would point past the placeholder buffer and
    # fold unrelated memory into the accumulator.
    has_bias: cutlass.Constexpr[bool],
    use_bias_smem: cutlass.Constexpr[bool],
    epi_subtile_n: cutlass.Constexpr[int],
    raster_along_m: cutlass.Constexpr[bool],
    # SM count, supplied by the caller rather than queried here: a
    # `torch.cuda.get_device_properties` call inside this function runs during
    # `cute.compile`, which would make compilation require a GPU. The caller passes
    # the real value.
    num_sms: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
    blkred_row_padding: cutlass.Constexpr[int] = 0,
    blkred_chunk_n: cutlass.Constexpr[int] = 0,
    defer_bulk_wait: cutlass.Constexpr[bool] = False,
    dual_cta: cutlass.Constexpr[bool] = False,
    use_local_scheduler: cutlass.Constexpr[bool] = False,
    use_mixed_k96: cutlass.Constexpr[bool] = False,
    native_metadata_m: cutlass.Constexpr[int] = 128,
) -> None:
    """Host-side JIT function for contiguous grouped block-scaled GEMM + finalize fusion.

    Args:
        a_tensor: Input matrix A, shape (K/2, valid_m, 1), Float4E2M1FNx2.
        b_tensor: Input matrix B, shape (K/2, N, L), Float4E2M1FNx2.
        sfa_tensor: Scale factors for A, shape (512, K//128, valid_m//128, 1).
        sfb_tensor: Scale factors for B, shape (512, K//128, N//128, L).
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

    if use_mixed_k96:
        assert torch.cuda.get_device_capability() == (10, 3), "mixed K96 requires an SM103 device"
        assert mma_tiler_k == 256 and mma_tiler_n in (128, 256) and not dual_cta, (
            "mixed K96 path supports only CTA1 M128/N128-or-256/K256"
        )
        assert cute.size(a_tensor, mode=[0]) % 128 == 0, (
            "mixed K96 path requires problem K divisible by 256"
        )

    assert native_metadata_m in (128, 256), "native metadata M must be 128 or 256"
    assert not dual_cta, "native-layout path uses only the CTA1 consumer"

    # MMA tile configuration for 1CTA
    mma_tiler_mnk = [128, mma_tiler_n, mma_tiler_k]  # Tile dimensions (M, N, K)
    # Per-instruction K granule: 64 for NVFP4 (kind::mxf4nvf4), not the 32 the
    # GPT-OSS reference hardcoded for kind::mxf8f6f4.
    # make_blockscaled_trivial_tiled_mma for the all-FP4 config reports
    # shape_mnk = (128, 128, 64). num_kblocks, num_sf_ids and
    # inst_k_bytes_16B all derive from this, so getting it wrong misprograms the
    # whole K loop.
    mma_inst_mnk = [128, mma_tiler_n, 64]  # MMA instruction shape for MXF4NVF4

    # A/B are packed FP4x2 tensors. Their descriptors and K boxes use packed-pair
    # coordinates while SMEM and MMA geometry below remain in scalar FP4 lanes.
    tma_a_desc = cuda.create_tensor_map_tiled_from_view(
        a_tensor,
        box_dims=(
            mma_tiler_mnk[2] // fp4_lanes_per_storage,
            mma_tiler_mnk[0],
            1,
        ),  # (Kx2_slice, M_tile, L_tile)
        stride_order=(0, 1, 2),
        swizzle=(cuda.TensorMapSwizzle.s64b if mma_tiler_k == 128 else cuda.TensorMapSwizzle.s128b),
        dtype=fp4x2_dtype,
        tma_format=cuda.TensorMapDataFormat.BYTE,
    )

    tma_b_desc = cuda.create_tensor_map_tiled_from_view(
        b_tensor,
        box_dims=(
            mma_tiler_mnk[2] // fp4_lanes_per_storage,
            mma_tiler_mnk[1],
            1,
        ),  # (Kx2_slice, N_tile, L_tile)
        stride_order=(0, 1, 2),
        swizzle=(cuda.TensorMapSwizzle.s64b if mma_tiler_k == 128 else cuda.TensorMapSwizzle.s128b),
        dtype=fp4x2_dtype,
        tma_format=cuda.TensorMapDataFormat.BYTE,
    )

    # No C TMA descriptor — finalize fusion uses atomic scatter instead of TMA store.
    # Pointers (out_ptr, tile_idx_ptr, etc.) are passed as runtime Int64 arguments
    # to avoid JIT caching issues with baked-in addresses.

    # Compute SF atom counts per tile and TMEM columns per MMA kblock
    tile_sf, num_tmem_cols_per_kblock_sfa, num_tmem_cols_per_kblock_sfb = compute_sf_rest(
        mma_tiler_mnk, mma_inst_mnk
    )
    rest_k_sf_per_ktile = tile_sf["k"]
    rest_m_sf_per_ktile = tile_sf["m"]
    rest_n_sf_per_ktile = tile_sf["n"]

    # SFA/SFB tensors have flat layout (512, rest_k_sf, rest_m/n_sf, L) as 8-bit
    # Recast to 16-bit (256 elements) for TMA descriptor
    sfa_tensor_fp16 = cute.recast_tensor(sfa_tensor, cutlass.Uint16)
    sfb_tensor_fp16 = cute.recast_tensor(sfb_tensor, cutlass.Uint16)

    # SFA TMA descriptor
    tma_sfa_desc = cuda.create_tensor_map_tiled_from_view(
        sfa_tensor_fp16,
        box_dims=(num_elts_atom_sf_fp16, rest_k_sf_per_ktile, rest_m_sf_per_ktile, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
    )

    # SFB TMA descriptor
    tma_sfb_desc = cuda.create_tensor_map_tiled_from_view(
        sfb_tensor_fp16,
        box_dims=(num_elts_atom_sf_fp16, rest_k_sf_per_ktile, rest_n_sf_per_ktile, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
    )

    # A and B are both NVFP4 (0.5 bytes per element), so B has one size for its
    # TMA transfer and SMEM tile.
    num_bytes_a = mma_tiler_mnk[0] * mma_tiler_mnk[2] * a_dtype.width // 8
    num_bytes_b = mma_tiler_mnk[1] * mma_tiler_mnk[2] * b_dtype.width // 8
    num_bytes_sfa = (
        num_elts_atom_sf_e8 * rest_k_sf_per_ktile * rest_m_sf_per_ktile * sf_dtype.width // 8
    )
    num_bytes_sfb = (
        num_elts_atom_sf_e8 * rest_k_sf_per_ktile * rest_n_sf_per_ktile * sf_dtype.width // 8
    )
    # Resolve the 0 = "default" sentinel before using the value in any SMEM
    # extent. TRT-LLM calls host_function directly and therefore legitimately
    # supplies 0 instead of going through the standalone run wrapper.
    epi_subtile_n = min(epi_subtile_n or 64, mma_tiler_n)
    if cutlass.const_expr(epi_subtile_n != 64):
        raise ValueError("bank-spreading experiment requires 64-column epilogue fragments")

    # SMEM for production-style whole-row bulk reduction. Each of the 128
    # epilogue threads owns one complete N tile row. Staging every subtile and
    # issuing one cp.reduce per row is important: issuing one bulk operation per
    # 32-column subtile pays commit/wait overhead four times for an N=128 tile.
    num_epi_threads = 32 * len(epilog_warp_id)  # 128
    num_bytes_c_stage = (
        num_epi_threads
        * ((blkred_chunk_n or mma_tiler_n) + blkred_row_padding)
        * c_dtype.width
        // 8
    )
    # Reserve SMEM for mbar, blkred buffer, bias buffer, and alignment overhead.
    # The kernel allocates sBias_dbuf (aligned to 16) and sC_blkred (aligned to
    # 1024) AFTER the AB pipeline stages. Without accounting for them here, the
    # stage count can overflow SMEM (e.g. N=64 + blkred + bias_smem overflows
    # by 512 bytes due to 1024-byte alignment padding before sC_blkred).
    # `use_bias_smem` stages bias through SMEM and reads `bias_expert_base`
    # unconditionally in the epilogue. That is only meaningful when a bias tensor
    # actually exists: with `has_bias=False` the bias pointer is a placeholder and those
    # reads run off the end of it. The non-SMEM path guards each load with `has_bias`;
    # this path asserts the contract instead, because staging a bias that is never used
    # has no purpose.
    assert not use_bias_smem or has_bias, (
        "use_bias_smem=True requires has_bias=True; the SMEM staging path reads the "
        "bias tensor unconditionally"
    )
    # Double-buffer bias only when there is a next subtile to prefetch.
    _subtile_cnt = cute.ceil_div(mma_tiler_n, epi_subtile_n)
    _bias_bufs = 2 if _subtile_cnt > 1 else 1
    bias_smem_bytes = (_bias_bufs * epi_subtile_n * 4) if use_bias_smem else 0
    smem_reserved = (num_bytes_c_stage if use_blkred else 0) + bias_smem_bytes
    # Tensor buffers need 1024B alignment: round the reservation up to the next
    # boundary, then keep one spare block. The spare block keeps the validated stage
    # count (with no bias and no blkred the reservation is exactly 1024).
    smem_reserved = ((smem_reserved + 1023) // 1024) * 1024 + 1024
    num_ab_stage = (smem_capacity - smem_reserved) // (
        num_bytes_a + num_bytes_b + num_bytes_sfa + num_bytes_sfb
    )
    if cutlass.const_expr(dual_cta):
        # A resident pair needs <= half of the SMEM budget per CTA. Preserve
        # the default schedule unless this explicit experimental mode is used.
        num_ab_stage = (smem_capacity // 2 - smem_reserved) // (
            num_bytes_a + num_bytes_b + num_bytes_sfa + num_bytes_sfb
        )
        assert num_ab_stage >= 1, "dual_cta has no room for one AB pipeline stage"

    # The outer ``moe`` JIT adapter converts runtime tensor iterators to Int64
    # pointers.  host_function therefore receives dynamic addresses instead of
    # capturing Python-side device pointers in the JIT cache.

    # Persistent tile scheduling: acc_stages for MMA/epilogue overlap.
    # N<256: acc_stages=2, two disjoint acc stages in 512 TMEM cols.
    # N>=256: acc_stages=1 with overlapping_accum — one logical barrier,
    #   two physical 256-col TMEM regions overlapping by sf_tmem_cols.
    sf_tmem_cols = (
        rest_k_sf_per_ktile
        * (rest_m_sf_per_ktile + rest_n_sf_per_ktile)
        * num_tmem_cols_per_sf_atom
    )
    overlapping_accum = mma_tiler_mnk[1] >= 256
    acc_stages = 1 if overlapping_accum else 2
    if cutlass.const_expr(dual_cta):
        # N<=128 plus scales fits in 256 TMEM columns with one accumulator.
        # A second512-column allocation would serialize even if SMEM fits.
        acc_stages = 1
    # Launch only the tiles that produce real output columns.
    n_tiles = cute.ceil_div(problem_n, mma_tiler_mnk[1])

    # Cooperative cluster scheduling (unified pattern for any cluster_m).
    cluster_m = 1
    # max_tiles_m is native metadata capacity, never the SF128 atom extent.
    virtual_capacity = max_tiles_m * (native_metadata_m // 128)
    tile_grid = (virtual_capacity, n_tiles, 1)
    scheduler_cluster = (1, 1, 1)
    tile_sched_params = PersistentTileSchedulerParams(
        tile_grid, scheduler_cluster, raster_along_m=raster_along_m
    )
    max_active_clusters = num_sms * (2 if dual_cta else 1) // cluster_m
    grid = DeviceBoundPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

    # Launch the kernel
    moe_grouped_gemm_finalize_fusion_kernel(
        tma_a_desc,
        tma_b_desc,
        tma_sfa_desc,
        tma_sfb_desc,
        tile_idx_ptr,
        bias_ptr,
        out_ptr,
        perm_idx_ptr,
        token_scales_ptr,
        alpha_ptr,
        mn_limit_ptr,
        num_tiles_m_ptr,
        problem_k,
        problem_n,
        topK,
        mma_tiler_mnk,
        mma_inst_mnk,
        epi_subtile_n,
        num_bytes_a,
        num_bytes_b,
        num_bytes_sfa,
        num_bytes_sfb,
        num_ab_stage,
        rest_k_sf_per_ktile,
        rest_m_sf_per_ktile,
        rest_n_sf_per_ktile,
        num_tmem_cols_per_kblock_sfa,
        num_tmem_cols_per_kblock_sfb,
        use_blkred,
        blkred_bufs,
        has_bias,
        use_bias_smem,
        num_bytes_c_stage,
        tile_sched_params,
        acc_stages,
        overlapping_accum,
        sf_tmem_cols,
        blkred_row_padding,
        blkred_chunk_n,
        defer_bulk_wait,
        dual_cta,
        use_local_scheduler,
        use_mixed_k96,
        native_metadata_m,
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(cluster_m, 1, 1),
        min_blocks_per_mp=2 if dual_cta else 1,
        preferred_smem_carveout=100 if dual_cta else None,
        stream=stream,
        use_pdl=USE_PDL,
    )
    return


# GPU device kernel.
#
# See the note on the naming in moe_gather_grouped_gemm_act_fusion.py: the DSL builds the
# cubin symbol from this function's name plus the argument reprs and truncates at 180
# characters, so naming both MoE kernels `kernel` makes them share one symbol and one
# profiler row. Keep the names distinct.
@cute.kernel
def moe_grouped_gemm_finalize_fusion_kernel(
    tma_a_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_b_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfa_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfb_desc: cutlass.GridConstant[cuda.TensorMap],
    tile_idx_ptr: cutlass.Int64,
    bias_ptr: cutlass.Int64,
    out_ptr: cutlass.Int64,
    perm_idx_ptr: cutlass.Int64,
    token_scales_ptr: cutlass.Int64,
    alpha_ptr: cutlass.Int64,
    mn_limit_ptr: cutlass.Int64,
    num_tiles_m_ptr: cutlass.Int64,
    problem_k: cutlass.Int32,
    problem_n: cutlass.Int32,
    topK: cutlass.Int32,
    mma_tiler_mnk: cutlass.Constexpr[tuple[int, int, int]],
    mma_inst_mnk: cutlass.Constexpr[tuple[int, int, int]],
    epi_subtile_n: cutlass.Constexpr[int],
    num_bytes_a: cutlass.Constexpr[int],
    num_bytes_b: cutlass.Constexpr[int],
    num_bytes_sfa: cutlass.Constexpr[int],
    num_bytes_sfb: cutlass.Constexpr[int],
    num_ab_stage: cutlass.Constexpr[int],
    rest_k_sf_per_ktile: cutlass.Constexpr[int],
    rest_m_sf_per_ktile: cutlass.Constexpr[int],
    rest_n_sf_per_ktile: cutlass.Constexpr[int],
    num_tmem_cols_per_kblock_sfa: cutlass.Constexpr[int],
    num_tmem_cols_per_kblock_sfb: cutlass.Constexpr[int],
    use_blkred: cutlass.Constexpr[bool],
    blkred_bufs: cutlass.Constexpr[int],
    has_bias: cutlass.Constexpr[bool],
    use_bias_smem: cutlass.Constexpr[bool],
    num_bytes_c_stage: cutlass.Constexpr[int],
    tile_sched_params: PersistentTileSchedulerParams,
    acc_stages: cutlass.Constexpr[int],
    overlapping_accum: cutlass.Constexpr[bool],
    sf_tmem_cols: cutlass.Constexpr[int],
    blkred_row_padding: cutlass.Constexpr[int] = 0,
    blkred_chunk_n: cutlass.Constexpr[int] = 0,
    defer_bulk_wait: cutlass.Constexpr[bool] = False,
    dual_cta: cutlass.Constexpr[bool] = False,
    use_local_scheduler: cutlass.Constexpr[bool] = False,
    use_mixed_k96: cutlass.Constexpr[bool] = False,
    native_metadata_m: cutlass.Constexpr[int] = 128,
) -> None:
    """GPU device kernel for contiguous grouped mixed precision block-scaled GEMM.

    Extends the dense block-scaled GEMM kernel with contiguous grouped GEMM support:
    - tile_idx_to_expert_idx maps each M-tile to an expert index
    - TMA warp uses expert_idx to index B/SFB in the L dimension
    - Epilogue applies per-expert bias[expert_idx, :]

    Args:
        tma_a_desc: TMA descriptor for matrix A (NVFP4, K-major, L=1).
        tma_b_desc: TMA descriptor for matrix B (FP4, K-major, L=num_experts).
        tma_sfa_desc: TMA descriptor for A's scale factors (L=1).
        tma_sfb_desc: TMA descriptor for B's scale factors (L=num_experts).
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
            production runner passes it as a dynamic value and its gg2 cache key omits
            it, so baking it in would silently reuse a kernel built for a different
            top_k. Device use is one integer division in the epilogue, which is off
            the critical path.
        mma_tiler_mnk: Tile dimensions [M, N, K].
        mma_inst_mnk: MMA instruction dimensions [M, N, K].
        epi_subtile_n: Epilog subtile N dimension.
        num_bytes_*: Byte sizes for SMEM allocation.
        num_ab_stage: Pipeline depth.
        rest_*_sf_per_ktile: SF atom counts per tile dimension.
        num_tmem_cols_per_kblock_*: TMEM columns per kblock for scale factors.

    Kernel Architecture (device-bound persistent scheduler):
        - Block: 256 threads (8 warps)
        - Warp roles: warp 0-3 (epilogue), warp 4 (MMA+S2T), warp 5 (TMA),
          warp 6 (tile scheduler producer), warp 7 (idle, donates regs)
        - TMA warp loads B[expert_idx] and SFB[expert_idx] per tile
        - Epilogue adds bias[expert_idx, :] before type conversion
        - Warp 6 runs DeviceBoundPersistentTileScheduler, writes work_tiles
          to SMEM; consumer warps (0-5) read via mbarrier handshake.
    """
    #
    # K-tile config — computed inline from mma_tiler_mnk at compile time
    #
    _kc = compute_ktile_config(mma_tiler_mnk[2])
    # Retain the frozen constexpr binding for device-source/codegen equivalence.
    smem_row_bytes = _kc["smem_row_bytes"]  # noqa: F841
    mma_layout = _kc["mma_layout"]
    mma_stride = _kc["mma_stride"]
    mma_leading = _kc["mma_leading"]

    #
    # Thread / block identification
    #
    warp_idx = cute.arch.warp_idx()
    tidx, _, _ = cute.arch.thread_idx()
    k = problem_k
    tmem_allocation_cols = 256 if dual_cta else num_tmem_alloc_cols

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
    sA = cutlass.Array(
        cutlass.Int8,
        num_bytes_a * num_ab_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )  # NVFP4 payload in an Int8 byte container
    sB = cutlass.Array(
        cutlass.Int8,
        num_bytes_b * num_ab_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )  # NVFP4 payload in an Int8 byte container
    sSFA = cutlass.Array(
        cutlass.Int8,
        num_bytes_sfa * num_ab_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )  # E4M3 scale factors, 8 bits each
    sSFB = cutlass.Array(
        cutlass.Int8,
        num_bytes_sfb * num_ab_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )  # E4M3 scale factors, 8 bits each

    # Compute copy sizes
    num_tma_load_bytes = num_bytes_a + num_bytes_b + num_bytes_sfa + num_bytes_sfb

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
    # RoutingCustomKernels.cuh: "Trigger the secondary kernel AFTER all global memory writes ... The
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
            if cutlass.const_expr(not use_local_scheduler):
                # Dedicated scheduler producer and six consumer warps.
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

    if cutlass.const_expr(use_local_scheduler):
        # This is after PDL WAIT and the asynchronous metadata transaction.
        # Values dominate all role branches; no Python scheduler params are
        # mutated with branch-local SSA values.
        while not prims.mbarrier_try_wait_parity(num_tiles_m_mbar_ptr, 0):
            pass
        # Preserve PDL/count readiness, then give every role identical virtual work.
        local_actual_tiles_m = cute.arch.make_warp_uniform(
            s_num_tiles_m.load() * (native_metadata_m // 128)
        )
        local_n_tiles = tile_sched_params.problem_shape_ntile_mnl[1]
        if cutlass.const_expr(tile_sched_params.raster_along_m):
            local_major_divisor = cute.fast_divmod_create_divisor(
                max(local_actual_tiles_m, cutlass.Int32(1))
            )
        else:
            local_major_divisor = tile_sched_params.cluster_shape_major_fdd
        local_stride = cutlass.Int32(cute.arch.grid_dim()[2])

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
    alpha_arr = cutlass.Array(
        cutlass.inttoptr(alpha_ptr, mem_space=1, dtype=cutlass.Float32),
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
        if cutlass.const_expr(not use_local_scheduler):
            # Build scheduler entirely inside this warp's scope
            while not prims.mbarrier_try_wait_parity(num_tiles_m_mbar_ptr, 0):
                pass
            actual_tiles_m = s_num_tiles_m.load() * (native_metadata_m // 128)
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
    # TMA warp (warp 5): GMEM -> SMEM async loads for A, B, SFA, SFB
    # =========================================================================
    if warp_idx == tma_warp_id:
        prims.setmaxregister(num_regs_tma, prims.SetMaxRegisterAction.DECREASE)

        # L2 eviction policy for the B stream only. B is read exactly once (reuse 1);
        # A is reused by all 48 n-tiles of an expert and is small enough to stay
        # resident, so telling L2 to drop B first stops the streamed weights from
        # competing with the one tensor that benefits from being cached. No
        # `createpolicy` wrapper exists, so the 64-bit
        # descriptor is built with one PTX instruction.
        if cutlass.const_expr(L2_POLICY_B):
            b_policy = prims.inline_ptx_hl(
                f"createpolicy.fractional.L2::{L2_POLICY_B}.b64 {{$w0}}, 1.0;",
                write_only_types=[cutlass.Int64],
            )
        else:
            b_policy = None

        # AB pipeline producer state: stage index carries across tiles.
        ab_stage_idx = 0
        # ---- Persistent loop: TMA warp ----
        # Read initial work_tile from SMEM (written by sched warp)
        if cutlass.const_expr(use_local_scheduler):
            work_linear_idx = cutlass.Int32(cute.arch.block_idx()[2])
            work_tile = local_work_tile(
                work_linear_idx,
                local_actual_tiles_m,
                local_n_tiles,
                local_major_divisor,
                tile_sched_params.raster_along_m,
            )
        else:
            work_tile, sched_full_phase = sched_load_work_tile(
                s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, 0
            )
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            # Scheduler returns (m, n, l) — raster order set at host
            tile_coord_m = cur_tile_coord[0]
            tile_coord_n = cur_tile_coord[1]
            native_tile_idx = tile_coord_m // (native_metadata_m // 128)
            expert_idx = tile_idx_arr.subview(native_tile_idx).load()
            coord_l_a = cutlass.Int32(0)
            coord_l_b = expert_idx
            coord_m = tile_coord_m * mma_tiler_mnk[0]
            coord_n = tile_coord_n * mma_tiler_mnk[1]
            coord_m_sf = coord_m // (num_m0_per_sf_atom * num_m1_per_sf_atom)
            coord_n_sf = coord_n // (num_m0_per_sf_atom * num_m1_per_sf_atom)

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

                coord_k_packed = k_tile * mma_tiler_mnk[2] // fp4_lanes_per_storage
                coord_k_sf = k_tile * rest_k_sf_per_ktile

                sA_staged = sA.subview(num_bytes_a * current_ab_stage)
                sB_staged = sB.subview(num_bytes_b * current_ab_stage)
                sSFA_staged = sSFA.subview(num_bytes_sfa * current_ab_stage)
                sSFB_staged = sSFB.subview(num_bytes_sfb * current_ab_stage)

                if prims.elect_sync():
                    # Each descriptor covers the complete logical K tile in one
                    # packed-FP4x2 transaction.
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sA_staged,
                        tma_a_desc.get_ptr(),
                        (coord_k_packed, coord_m, coord_l_a),
                        ab_full_mbar_ptr.subview(current_ab_stage),
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sB_staged,
                        tma_b_desc.get_ptr(),
                        (coord_k_packed, coord_n, coord_l_b),
                        ab_full_mbar_ptr.subview(current_ab_stage),
                        l2_cache_hint=b_policy,
                    )
                    # TMA load SFA: contiguous (L=1), use coord_l_a=0
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sSFA_staged,
                        tma_sfa_desc.get_ptr(),
                        (0, coord_k_sf, coord_m_sf, coord_l_a),
                        ab_full_mbar_ptr.subview(current_ab_stage),
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sSFB_staged,
                        tma_sfb_desc.get_ptr(),
                        (0, coord_k_sf, coord_n_sf, coord_l_b),
                        ab_full_mbar_ptr.subview(current_ab_stage),
                    )

            # Wait for producer to write next work_tile
            if cutlass.const_expr(use_local_scheduler):
                work_linear_idx += local_stride
                work_tile = local_work_tile(
                    work_linear_idx,
                    local_actual_tiles_m,
                    local_n_tiles,
                    local_major_divisor,
                    tile_sched_params.raster_along_m,
                )
            else:
                work_tile, sched_full_phase = sched_load_work_tile(
                    s_work_tile,
                    sched_full_mbar_ptr,
                    sched_empty_mbar_ptr,
                    sched_full_phase,
                )

    # =========================================================================
    # MMA warp (warp 4): TMEM alloc, S2T copy, block-scaled tcgen05.mma compute
    # =========================================================================
    if warp_idx == mma_warp_id:
        prims.setmaxregister(num_regs_mma, prims.SetMaxRegisterAction.DECREASE)
        # Allocate TMEM using cutlass - MMA warp is the allocator (once)
        prims.tcgen05_alloc(tmem_ptr_i32, tmem_allocation_cols, group="cta_1")
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")
        # Sync with epilogue warps after allocation
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)
        # Retrieve TMEM pointer from smem buffer (returns cutlass.Array)
        tmem_raw_addr = tmem_ptr_i32.load()
        acc_tmem_ptr = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Float32)

        # Build instruction descriptor for the block-scaled tcgen05.mma (loop-invariant)
        # MMA instruction descriptor. These were hardcoded to the GPT-OSS mixed-precision
        # configuration in the reference, so they do not follow the module dtypes -- a
        # tensor-core-level mismatch that costs no memory error and no crash, just
        # silently wrong numerics. Driven off the module constants now.
        # scale_format is MxScaleFormat: UE4M3=0, E8M0=1, UE5M3=2. NVFP4 uses E4M3
        # scale factors, so 0; the reference's 1 is the MX format.
        # Tcgen05MxOmmaInstrDesc, not Tcgen05MxInstrDesc: NVFP4 uses the
        # kind::mxf4nvf4 form, whose descriptor has a *different bit layout* (it adds
        # FP4-specific k_dim / sparsity_version fields). The MX descriptor is for the
        # mxf8f6f4 mixed-precision path the GPT-OSS reference took. Filling the wrong
        # one does not fault at build time -- the MMA warp hits an illegal instruction.
        idesc = prims.Tcgen05MxOmmaInstrDesc.build(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            scale_format=int(prims.MxScaleFormat.UE4M3),
            n_dim=mma_tiler_mnk[1],
            m_dim=mma_tiler_mnk[0],
        )
        if cutlass.const_expr(use_mixed_k96):
            # Dense SM103 K96 instruction descriptor: k_dim=1, sparsity_version=0.
            idesc_k96 = prims.Tcgen05MxOmmaInstrDesc.build(
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                scale_format=int(prims.MxScaleFormat.UE4M3),
                n_dim=mma_tiler_mnk[1],
                m_dim=128,
                k_dim=1,
                sparsity_version=0,
            )
        s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4

        # TMEM base addresses (loop-invariant)
        base_col_id = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        num_smem_inc_per_sf_atom = num_elts_atom_sf_e8 >> 4
        num_kblocks = cute.ceil_div(mma_tiler_mnk[2], mma_inst_mnk[2])
        num_sf_ids = num_k_per_sf_atom * sf_vec_size // mma_inst_mnk[2]

        # Accumulator stage geometry: partition TMEM columns across acc_stages.
        # SFA/SFB are only used during the MMA K-loop and can be safely
        # overwritten on the next tile — no per-stage copies needed.
        cols_per_acc_stage = mma_tiler_mnk[1]
        sfa_tmem_cols_shared = rest_k_sf_per_ktile * rest_m_sf_per_ktile * num_tmem_cols_per_sf_atom
        # Retain the frozen constexpr binding for device-source/codegen equivalence.
        sfb_tmem_cols_shared = (  # noqa: F841
            rest_k_sf_per_ktile * rest_n_sf_per_ktile * num_tmem_cols_per_sf_atom
        )
        # SFA/SFB placed at fixed positions after accumulator stages.
        # Overlapping: SF at end of 512 cols (after two overlapping acc regions).
        # Non-overlapping: SF after both disjoint acc stages.
        if cutlass.const_expr(overlapping_accum):
            sfa_col_base = base_col_id + 2 * cols_per_acc_stage - sf_tmem_cols
        else:
            sfa_col_base = base_col_id + acc_stages * cols_per_acc_stage
        sfb_col_base = sfa_col_base + sfa_tmem_cols_shared
        # Fixed SFA/SFB TMEM addresses (not per acc_stage)
        sfa_tmem_addr_base = (base_row_id << 16) | sfa_col_base
        sfb_tmem_addr_base = (base_row_id << 16) | sfb_col_base

        # Build SMEM descriptors once (stage 0).
        desc_sfa_s2t_base = _compat48.tcgen05_smem_desc(
            sSFA,
            leading_byte_offset=16,
            stride_byte_offset=128,
            version=0,
            base_offset=0,
            layout=0,
        )
        desc_sfb_s2t_base = _compat48.tcgen05_smem_desc(
            sSFB,
            leading_byte_offset=16,
            stride_byte_offset=128,
            version=0,
            base_offset=0,
            layout=0,
        )
        desc_a_mma_base = _compat48.tcgen05_smem_desc(
            sA,
            leading_byte_offset=mma_leading,
            stride_byte_offset=mma_stride,
            version=0,
            base_offset=0,
            layout=mma_layout,
        )
        desc_b_mma_base = _compat48.tcgen05_smem_desc(
            sB,
            leading_byte_offset=mma_leading,
            stride_byte_offset=mma_stride,
            version=0,
            base_offset=0,
            layout=mma_layout,
        )
        stage_inc_a = num_bytes_a
        stage_inc_b = num_bytes_b
        stage_inc_sfa = num_bytes_sfa
        stage_inc_sfb = num_bytes_sfb

        # AB pipeline consumer state: stage index carries across tiles.
        ab_stage_idx = 0
        acc_stage_idx = 0

        # ---- Persistent loop: MMA warp ----
        # Read initial work_tile from SMEM
        if cutlass.const_expr(use_local_scheduler):
            work_linear_idx = cutlass.Int32(cute.arch.block_idx()[2])
            work_tile = local_work_tile(
                work_linear_idx,
                local_actual_tiles_m,
                local_n_tiles,
                local_major_divisor,
                tile_sched_params.raster_along_m,
            )
        else:
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

            # SFB tmem offset within an SF atom (depends on tile_coord_n)
            sfb_tmem_col_offset = 0
            if cutlass.const_expr(
                mma_tiler_mnk[1] % (num_m0_per_sf_atom * num_m1_per_sf_atom) != 0
            ):
                num_mma_n_per_sf_atom = (num_m0_per_sf_atom * num_m1_per_sf_atom) // mma_tiler_mnk[
                    1
                ]
                num_tmem_col_per_mma = num_tmem_cols_per_sf_atom // num_mma_n_per_sf_atom
                sfb_tmem_col_offset = (tile_coord_n % num_mma_n_per_sf_atom) * num_tmem_col_per_mma

            # TMEM column base for the accumulator write region.
            if cutlass.const_expr(overlapping_accum):
                # Phase ping-pong: MMA writes to opposite region of epilogue.
                mma_phys_stage = acc_empty_phase_bit ^ 1
                cur_base_col_id = base_col_id + mma_phys_stage * (cols_per_acc_stage - sf_tmem_cols)
            else:
                cur_base_col_id = base_col_id + acc_stage_idx * cols_per_acc_stage
            acc_tmem_raw_addr = (base_row_id << 16) | cur_base_col_id
            acc_tmem_ptr_stage = prims.make_tmem_ptr(acc_tmem_raw_addr, cutlass.Float32)

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
                desc_a_nvvm_s2t_base = desc_sfa_s2t_base.advance_start_address(
                    current_ab_stage * stage_inc_sfa
                )
                desc_b_nvvm_s2t_base = desc_sfb_s2t_base.advance_start_address(
                    current_ab_stage * stage_inc_sfb
                )
                desc_a_nvvm_mma_base = desc_a_mma_base.advance_start_address(
                    current_ab_stage * stage_inc_a
                )
                desc_b_nvvm_mma_base = desc_b_mma_base.advance_start_address(
                    current_ab_stage * stage_inc_b
                )

                # Copy SFA from SMEM to TMEM
                for s2t_idx in cutlass.range(
                    rest_m_sf_per_ktile * rest_k_sf_per_ktile, unroll_full=True
                ):
                    sfa_tmem_addr_s2t = sfa_tmem_addr_base + s2t_idx * num_tmem_cols_per_sf_atom
                    sfa_tmem_ptr_s2t = prims.make_tmem_ptr(sfa_tmem_addr_s2t, cutlass.Int32)
                    s2t_idx_k = s2t_idx // rest_m_sf_per_ktile
                    s2t_idx_m = s2t_idx % rest_m_sf_per_ktile
                    increment_s2t = (
                        num_smem_inc_per_sf_atom * s2t_idx_k
                        + num_smem_inc_per_sf_atom * rest_k_sf_per_ktile * s2t_idx_m
                    )
                    desc_a_nvvm_s2t = desc_a_nvvm_s2t_base + increment_s2t
                    if prims.elect_sync():
                        prims.tcgen05_cp(
                            s2t_shape,
                            sfa_tmem_ptr_s2t,
                            desc_a_nvvm_s2t,
                            multicast=s2t_multicast,
                        )

                # Copy SFB from SMEM to TMEM
                for s2t_idx in cutlass.range(
                    rest_n_sf_per_ktile * rest_k_sf_per_ktile, unroll_full=True
                ):
                    sfb_tmem_addr_s2t = sfb_tmem_addr_base + s2t_idx * num_tmem_cols_per_sf_atom
                    sfb_tmem_ptr_s2t = prims.make_tmem_ptr(sfb_tmem_addr_s2t, cutlass.Int32)
                    s2t_idx_k = s2t_idx // rest_n_sf_per_ktile
                    s2t_idx_n = s2t_idx % rest_n_sf_per_ktile
                    increment_s2t = (
                        num_smem_inc_per_sf_atom * s2t_idx_k
                        + num_smem_inc_per_sf_atom * rest_k_sf_per_ktile * s2t_idx_n
                    )
                    desc_b_nvvm_s2t = desc_b_nvvm_s2t_base + increment_s2t
                    if prims.elect_sync():
                        prims.tcgen05_cp(
                            s2t_shape,
                            sfb_tmem_ptr_s2t,
                            desc_b_nvvm_s2t,
                            multicast=s2t_multicast,
                        )

                # Execute MMA over the single full-K SMEM tile.
                sfa_mma_tmem_addr_base = sfa_tmem_addr_base
                sfb_mma_tmem_addr_base = sfb_tmem_addr_base + sfb_tmem_col_offset
                inst_k_bytes_16B = (mma_inst_mnk[2] * a_dtype.width // 8) >> 4
                if cutlass.const_expr(use_mixed_k96):
                    # K96 + K96 + K64 covers the SAME K256 stage exactly.
                    # A/B and all sixteen scales are already ready. No MMA
                    # crosses a stage boundary; the commit remains after all3.
                    for mixed_idx in cutlass.range_constexpr(3):
                        if cutlass.const_expr(mixed_idx < 2):
                            # SF starts 0 and 6: four-column base plus selector 2
                            # for the second instruction, using the SM103 rule.
                            sf_atom = mixed_idx
                            sf_id = mixed_idx * 2
                            idesc_mixed = idesc_k96.set_sf_ids(a_sf_id=sf_id, b_sf_id=sf_id)
                            # Bit 46: descriptor version 1. Bit 52: absolute
                            # leading-byte-offset mode, which holds the K96
                            # next-buffer address. Both K96 ranges remain in
                            # this stage, so next points to the SAME base.
                            desc_a_circular = prims.Tcgen05SmemDesc(
                                (desc_a_nvvm_mma_base | (1 << 52) | (1 << 46)) & -4294901761
                            )
                            desc_b_circular = prims.Tcgen05SmemDesc(
                                (desc_b_nvvm_mma_base | (1 << 52) | (1 << 46)) & -4294901761
                            )
                            desc_a_mixed = prims.Tcgen05SmemDesc(
                                desc_a_circular.advance_start_address(mixed_idx * 48)
                                | ((desc_a_nvvm_mma_base & 0xFFFF) << 16)
                            )
                            desc_b_mixed = prims.Tcgen05SmemDesc(
                                desc_b_circular.advance_start_address(mixed_idx * 48)
                                | ((desc_b_nvvm_mma_base & 0xFFFF) << 16)
                            )
                        else:
                            # Last64 consumes K192:256 / SF12:16 with the
                            # unchanged K64 relative descriptor and selector0.
                            sf_atom = 3
                            idesc_mixed = idesc
                            desc_a_mixed = desc_a_nvvm_mma_base + 6
                            desc_b_mixed = desc_b_nvvm_mma_base + 6
                        scale_a_mixed = prims.make_tmem_ptr(
                            sfa_mma_tmem_addr_base + sf_atom * num_tmem_cols_per_kblock_sfa,
                            cutlass.Int32,
                        )
                        scale_b_mixed = prims.make_tmem_ptr(
                            sfb_mma_tmem_addr_base + sf_atom * num_tmem_cols_per_kblock_sfb,
                            cutlass.Int32,
                        )
                        first_mixed = (k_tile == 0) and (mixed_idx == 0)
                        if prims.elect_sync():
                            prims.tcgen05_mma_block_scale(
                                prims.MMABlockScaleKind.MXF4NVF4,
                                prims.CTAGroup.CTA_1,
                                acc_tmem_ptr_stage,
                                desc_a_mixed,
                                desc_b_mixed,
                                idesc_mixed,
                                not first_mixed,
                                scale_a_mixed,
                                scale_b_mixed,
                                scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                            )
                else:
                    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                        sf_inside_tmem_col_idx = kblock_idx % num_sf_ids
                        sf_col_idx = kblock_idx // num_sf_ids

                        sfa_tmem_addr_mma = (
                            sfa_mma_tmem_addr_base + sf_col_idx * num_tmem_cols_per_kblock_sfa
                        )
                        sfb_tmem_addr_mma = (
                            sfb_mma_tmem_addr_base + sf_col_idx * num_tmem_cols_per_kblock_sfb
                        )
                        sfa_tmem_ptr_mma = prims.make_tmem_ptr(sfa_tmem_addr_mma, cutlass.Int32)
                        sfb_tmem_ptr_mma = prims.make_tmem_ptr(sfb_tmem_addr_mma, cutlass.Int32)

                        idesc_updated = idesc.set_sf_ids(
                            a_sf_id=sf_inside_tmem_col_idx,
                            b_sf_id=sf_inside_tmem_col_idx,
                        )

                        desc_offset = kblock_idx * inst_k_bytes_16B
                        desc_a_nvvm_mma = desc_a_nvvm_mma_base + desc_offset
                        desc_b_nvvm_mma = desc_b_nvvm_mma_base + desc_offset
                        first = (k_tile == 0) and (kblock_idx == 0)

                        if prims.elect_sync():
                            # NVFP4 is kind::mxf4nvf4 with scale_vec::4X, not the
                            # reference's kind::mxf8f6f4. mxf8f6f4 is the mixed-precision
                            # form and requires the 4-bit operand padded to 8 bits in SMEM
                            # (16384 B/stage for B); mxf4nvf4 takes it 4-bit packed
                            # (8192 B/stage), which is what num_bytes_b already
                            # allocates now that a_dtype is FP4. scale_vec::4X follows from
                            # the MMA instruction K of 64 against sf_vec_size 16: one
                            # instruction spans 64/16 = 4 scale factors.
                            prims.tcgen05_mma_block_scale(
                                prims.MMABlockScaleKind.MXF4NVF4,
                                prims.CTAGroup.CTA_1,
                                acc_tmem_ptr_stage,
                                desc_a_nvvm_mma,
                                desc_b_nvvm_mma,
                                idesc_updated,
                                not first,
                                sfa_tmem_ptr_mma,
                                sfb_tmem_ptr_mma,
                                scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                            )
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
            if cutlass.const_expr(use_local_scheduler):
                work_linear_idx += local_stride
                work_tile = local_work_tile(
                    work_linear_idx,
                    local_actual_tiles_m,
                    local_n_tiles,
                    local_major_divisor,
                    tile_sched_params.raster_along_m,
                )
            else:
                work_tile, sched_full_phase = sched_load_work_tile(
                    s_work_tile,
                    sched_full_mbar_ptr,
                    sched_empty_mbar_ptr,
                    sched_full_phase,
                )

    # =========================================================================
    # Epilogue warps (warps 0-3): T2R -> scale -> atomic scatter
    # =========================================================================
    if warp_idx < mma_warp_id:
        prims.setmaxregister(
            96 if dual_cta else num_regs_epilog, prims.SetMaxRegisterAction.INCREASE
        )
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
            sC_row_offset = epi_thread_idx * (
                (blkred_chunk_n or mma_tiler_mnk[1]) + blkred_row_padding
            )

        acc_stage_idx = 0

        # ---- Persistent loop: Epilogue warps ----
        # Read initial work_tile from SMEM
        if cutlass.const_expr(use_local_scheduler):
            work_linear_idx = cutlass.Int32(cute.arch.block_idx()[2])
            work_tile = local_work_tile(
                work_linear_idx,
                local_actual_tiles_m,
                local_n_tiles,
                local_major_divisor,
                tile_sched_params.raster_along_m,
            )
        else:
            work_tile, sched_full_phase = sched_load_work_tile(
                s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, 0
            )
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            # Scheduler returns (m, n, l) — raster order set at host
            tile_coord_m = cur_tile_coord[0]
            tile_coord_n = cur_tile_coord[1]
            # Per-tile metadata
            native_tile_idx = tile_coord_m // (native_metadata_m // 128)
            expert_idx = tile_idx_arr.subview(native_tile_idx).load()
            base_m = tile_coord_m * mma_tiler_mnk[0]
            base_n = tile_coord_n * mma_tiler_mnk[1]

            # Per-expert metadata
            bias_expert_base = bias_arr.subview(expert_idx * problem_n)
            mn_limit = mn_limit_arr.subview(native_tile_idx).load()

            abs_row = base_m + epi_thread_idx
            is_valid_row = abs_row < mn_limit
            expanded_idx = cutlass.Int32(0)
            token_idx = cutlass.Int32(0)
            token_scale_val = cutlass.Float32(0.0)
            if is_valid_row:
                expanded_idx = perm_idx_arr.subview(abs_row).load()
                token_idx = expanded_idx // topK
                token_scale_val = token_scales_arr.subview(expanded_idx).load()
                # Per-expert, matching production's alpha_tuple[0][expert_idx].
                token_scale_val = alpha_arr.subview(expert_idx).load() * token_scale_val

            # ---- Bias loading ----
            # Overlapping accum alternates subtile direction based on which
            # physical TMEM region is being read:
            #   Region 0 (phase=0): overlap at HIGH end → reverse (high N first)
            #   Region 1 (phase=1): overlap at LOW end  → forward (low N first)
            # This ensures the overlap region is always read FIRST, enabling
            # safe early release before MMA writes the other region.
            if cutlass.const_expr(overlapping_accum):
                # reverse_flag: 1 when phase=0 (reverse), 0 when phase=1 (forward)
                reverse_flag = cutlass.Int32(1) - acc_full_phase_bit
                _first_subtile_col = base_n + reverse_flag * (subtile_cnt - 1) * epi_subtile_n
            else:
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
                    if cutlass.const_expr(overlapping_accum):
                        if is_valid_row and _first_subtile_col < problem_n:
                            prefetched_bias = bias_expert_base.subview(_first_subtile_col).load(
                                vector_size=t2r_inst_repx,
                                alignment=t2r_inst_repx * 4,
                            )
                    else:
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
            if cutlass.const_expr(use_blkred and defer_bulk_wait):
                # A lane owns the same scratch row throughout the persistent
                # loop. Finish its previous async SMEM reader before any store
                # reuses that row, but overlap the read with this tile's
                # scheduler/metadata/accumulator waits. Invalid rows still wait:
                # they may have issued a reduction for the preceding tile.
                prims.cp_async_bulk_wait_group(0, read=True)

            if cutlass.const_expr(overlapping_accum):
                epi_phys_stage = acc_full_phase_bit
                cur_base_col_id = base_col_id + epi_phys_stage * (cols_per_acc_stage - sf_tmem_cols)
            else:
                cur_base_col_id = base_col_id + acc_stage_idx * cols_per_acc_stage

            # Whole-row blkred scratch geometry.
            if cutlass.const_expr(use_blkred):
                # Optional 16B-aligned row padding distributes per-lane stores
                # over SMEM banks while preserving a contiguous bulk-reduce row.
                # Host stage budgeting includes this padding; it can trade an
                # AB stage for fewer bank conflicts and therefore needs tuning.
                bulk_width = blkred_chunk_n or mma_tiler_mnk[1]
                sC_row_offset = epi_thread_idx * (bulk_width + blkred_row_padding)
                copy_size_bytes = bulk_width * c_dtype.width // 8
                bulk_tile_is_full = base_n + mma_tiler_mnk[1] <= problem_n

            # Epilogue tiling loop.
            # Overlapping accum: direction alternates per region so the overlap
            # region is always the FIRST subtile processed.
            #   reverse_flag=1 (phase=0, region 0): real = subtile_cnt-1-idx
            #   reverse_flag=0 (phase=1, region 1): real = idx
            _early_release_subtile = cute.ceil_div(sf_tmem_cols, epi_subtile_n) - 1
            for subtile_idx in cutlass.range(subtile_cnt):
                if cutlass.const_expr(overlapping_accum):
                    # Runtime select: reverse_flag * (subtile_cnt-1-idx) + (1-reverse_flag) * idx
                    _rev = subtile_cnt - 1 - subtile_idx  # compile-time
                    _fwd = subtile_idx  # compile-time
                    real_subtile_idx = reverse_flag * (_rev - _fwd) + _fwd
                else:
                    real_subtile_idx = subtile_idx
                col_offset = base_n + real_subtile_idx * epi_subtile_n
                # Wait for current subtile's bias data, then prefetch next
                if cutlass.const_expr(use_bias_smem):
                    prims.cp_async_wait_group(0)
                    prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)
                    # Prefetch NEXT subtile into the other half of dbuf.
                    # Overlapping: direction depends on region (reverse_flag).
                    if cutlass.const_expr(overlapping_accum):
                        _step = cutlass.Int32(1) - cutlass.Int32(2) * reverse_flag
                        _next_real = real_subtile_idx + _step
                        _next_col = base_n + _next_real * epi_subtile_n
                        _has_next = _next_col >= base_n
                    else:
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
                # Overlapping accum: early acc_empty right after T2R read of the
                # overlap region. This lets MMA start the next tile while epilogue
                # continues processing the remaining (non-overlapping) subtiles.
                if cutlass.const_expr(overlapping_accum):
                    if subtile_idx == _early_release_subtile:
                        cute.arch.fence_view_async_tmem_load()
                        prims.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage_idx))

                if cutlass.const_expr(not use_bias_smem):
                    if cutlass.const_expr(overlapping_accum):
                        _step_ldg = cutlass.Int32(1) - cutlass.Int32(2) * reverse_flag
                        _next_real_ldg = real_subtile_idx + _step_ldg
                        next_col_offset = base_n + _next_real_ldg * epi_subtile_n
                        _has_next_ldg = next_col_offset >= base_n
                    else:
                        next_col_offset = col_offset + epi_subtile_n
                        _has_next_ldg = next_col_offset < base_n + mma_tiler_mnk[1]
                    next_bias = cutlass.vector.full([t2r_inst_repx], 0.0, cutlass.Float32)
                    # Same guard as the first prefetch. Missing it here is what kept
                    # the corruption alive after has_bias was introduced: this is the
                    # *next* subtile's bias, and it feeds `prefetched_bias` for the
                    # following iteration, so the garbage arrived one subtile later --
                    # which is why every damaged sample carried sub=1.
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
                            if bulk_tile_is_full:
                                sC_elem = sC_blkred.subview(
                                    sC_row_offset + (real_subtile_idx * epi_subtile_n) % bulk_width
                                )
                                # Eight 16-byte chunks cover all eight bank
                                # groups, without row padding or AB-stage loss.
                                # XOR issue order is a per-lane permutation of
                                # the same contiguous 64-BF16 fragment.
                                packed = scaled_bf16.bitcast(cutlass.Int32)
                                for issue in cutlass.range(8, unroll_full=True):
                                    chunk = (epi_thread_idx ^ issue) & 7
                                    choose_odd = cutlass.vector.full(
                                        [4], (chunk & 1) != 0, cutlass.Boolean
                                    )
                                    choose_middle = cutlass.vector.full(
                                        [4], (chunk & 2) != 0, cutlass.Boolean
                                    )
                                    choose_high = cutlass.vector.full(
                                        [4], (chunk & 4) != 0, cutlass.Boolean
                                    )
                                    pair0 = cutlass.vector.where(
                                        choose_odd, packed[4:8], packed[0:4]
                                    )
                                    pair1 = cutlass.vector.where(
                                        choose_odd, packed[12:16], packed[8:12]
                                    )
                                    pair2 = cutlass.vector.where(
                                        choose_odd, packed[20:24], packed[16:20]
                                    )
                                    pair3 = cutlass.vector.where(
                                        choose_odd, packed[28:32], packed[24:28]
                                    )
                                    half0 = cutlass.vector.where(choose_middle, pair1, pair0)
                                    half1 = cutlass.vector.where(choose_middle, pair3, pair2)
                                    selected = cutlass.vector.where(choose_high, half1, half0)
                                    sC_elem.subview(chunk * 8).store(
                                        selected.bitcast(cutlass.BFloat16),
                                        vector_size=8,
                                        alignment=16,
                                    )
                            else:
                                # A bulk copy cannot be predicated per byte. Keep
                                # partial N tails on the existing bounded atomics.
                                for elem_idx in cutlass.range(
                                    0, epi_subtile_n, 8, unroll_full=True
                                ):
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
                        else:
                            for elem_idx in cutlass.range(0, epi_subtile_n, 8, unroll_full=True):
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

                if cutlass.const_expr(use_blkred and blkred_chunk_n != 0):
                    # A smaller padded scratch window trades more bulk issues
                    # for bank distribution and additional AB pipeline capacity.
                    # Each thread owns its row: wait for its async reader before
                    # reusing that window. Both accumulator traversal directions
                    # finish a contiguous chunk before issuing its reduction.
                    chunk_subtiles = blkred_chunk_n // epi_subtile_n
                    chunk_end = real_subtile_idx % chunk_subtiles == chunk_subtiles - 1
                    if cutlass.const_expr(overlapping_accum):
                        if reverse_flag != 0:
                            chunk_end = real_subtile_idx % chunk_subtiles == 0
                    if is_valid_row and bulk_tile_is_full and chunk_end:
                        chunk_col = (real_subtile_idx // chunk_subtiles) * blkred_chunk_n
                        row_out = out_arr.subview(token_idx * problem_n + base_n + chunk_col)
                        row_smem = sC_blkred.subview(sC_row_offset)
                        cute.arch.fence_view_async_shared()
                        prims.cp_reduce_async_bulk_global_shared_cta(
                            row_out,
                            row_smem,
                            copy_size_bytes,
                            op=prims.CpReduceOp.ADD,
                            type=prims.CpReduceType.BF16,
                            noftz=True,
                        )
                        prims.cp_async_bulk_commit_group()
                        prims.cp_async_bulk_wait_group(0, read=True)

                if cutlass.const_expr(not use_bias_smem):
                    prefetched_bias = next_bias

            if cutlass.const_expr(use_blkred and blkred_chunk_n == 0):
                if is_valid_row and bulk_tile_is_full:
                    row_out = out_arr.subview(token_idx * problem_n + base_n)
                    row_smem = sC_blkred.subview(sC_row_offset)
                    cute.arch.fence_view_async_shared()
                    prims.cp_reduce_async_bulk_global_shared_cta(
                        row_out,
                        row_smem,
                        copy_size_bytes,
                        op=prims.CpReduceOp.ADD,
                        type=prims.CpReduceType.BF16,
                        noftz=True,
                    )
                    prims.cp_async_bulk_commit_group()

            # Signal acc_empty: all T2R reads done, MMA can reuse this acc_stage.
            # Finalize writes only depend on RMEM / scratch, so release the acc
            # stage as soon as T2R is complete.
            # For overlapping accum, early release was already sent inside loop.
            if cutlass.const_expr(not overlapping_accum):
                prims.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage_idx))

            if cutlass.const_expr(use_blkred and not defer_bulk_wait):
                prims.cp_async_bulk_wait_group(0, read=True)

            acc_stage_idx = acc_stage_idx + 1
            if acc_stage_idx == acc_stages:
                acc_stage_idx = 0
                acc_full_phase_bit = acc_full_phase_bit ^ 1

            # Wait for producer to write next work_tile
            if cutlass.const_expr(use_local_scheduler):
                work_linear_idx += local_stride
                work_tile = local_work_tile(
                    work_linear_idx,
                    local_actual_tiles_m,
                    local_n_tiles,
                    local_major_divisor,
                    tile_sched_params.raster_along_m,
                )
            else:
                work_tile, sched_full_phase = sched_load_work_tile(
                    s_work_tile,
                    sched_full_mbar_ptr,
                    sched_empty_mbar_ptr,
                    sched_full_phase,
                )

        if cutlass.const_expr(use_blkred and defer_bulk_wait):
            # The final tile has no following iteration to drain its reader.
            # Keep scratch alive until every issuing lane has completed reads.
            prims.cp_async_bulk_wait_group(0, read=True)

        # Deallocate TMEM - all epilogue warps sync then one deallocates.
        # Only epilogue warps (warp_idx < mma_warp_id) reach this point, so
        # use the epilogue-scoped barrier rather than a full-CTA sync.
        prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)
        if warp_idx == 0:
            prims.tcgen05_dealloc(acc_tmem_ptr, tmem_allocation_cols, group="cta_1")

    # Release the successor kernel as early as this CTA can.
    if cutlass.const_expr(USE_PDL):
        prims.griddepcontrol("launch_dependents")

    return


moe_grouped_gemm_finalize_fusion_kernel.set_name_prefix(
    "glm52_moe_nvfp4_gg2_finalize", remove_cutlass_symbol=True
)


@cute.jit
def moe(
    a,
    b,
    sfa,
    sfb,
    out,
    tile_idx_to_expert_idx,
    bias,
    permuted_idx_to_expanded_idx,
    token_final_scales,
    alpha,
    mn_limit,
    problem_k: cutlass.Int32,
    problem_n: cutlass.Int32,
    topK: cutlass.Int32,
    num_tiles_m,
    max_tiles_m: cutlass.Int32,
    mma_tiler_n: cutlass.Constexpr[int],
    mma_tiler_k: cutlass.Constexpr[int],
    use_blkred: cutlass.Constexpr[bool],
    blkred_bufs: cutlass.Constexpr[int],
    has_bias: cutlass.Constexpr[bool],
    use_bias_smem: cutlass.Constexpr[bool],
    epi_subtile_n: cutlass.Constexpr[int],
    raster_along_m: cutlass.Constexpr[bool],
    num_sms: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
    blkred_row_padding: cutlass.Constexpr[int] = 0,
    blkred_chunk_n: cutlass.Constexpr[int] = 0,
    defer_bulk_wait: cutlass.Constexpr[bool] = False,
    dual_cta: cutlass.Constexpr[bool] = False,
    use_local_scheduler: cutlass.Constexpr[bool] = False,
    use_mixed_k96: cutlass.Constexpr[bool] = False,
    native_metadata_m: cutlass.Constexpr[int] = 128,
) -> None:
    """Torch-ABI adapter for packed NVFP4 grouped GEMM + finalize."""
    host_function(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[2, 1, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0])),
        cute.make_tensor(sfa.iterator, cute.select(sfa.layout, mode=[3, 2, 1, 0])),
        cute.make_tensor(sfb.iterator, cute.select(sfb.layout, mode=[3, 2, 1, 0])),
        out.iterator.toint(),
        tile_idx_to_expert_idx.iterator.toint(),
        bias.iterator.toint(),
        permuted_idx_to_expanded_idx.iterator.toint(),
        token_final_scales.iterator.toint(),
        alpha.iterator.toint(),
        mn_limit.iterator.toint(),
        problem_k,
        problem_n,
        topK,
        num_tiles_m.iterator.toint(),
        max_tiles_m,
        mma_tiler_n,
        mma_tiler_k,
        use_blkred,
        blkred_bufs,
        has_bias,
        use_bias_smem,
        epi_subtile_n,
        raster_along_m,
        num_sms,
        stream,
        blkred_row_padding,
        blkred_chunk_n,
        defer_bulk_wait,
        dual_cta,
        use_local_scheduler,
        use_mixed_k96,
        native_metadata_m,
    )
