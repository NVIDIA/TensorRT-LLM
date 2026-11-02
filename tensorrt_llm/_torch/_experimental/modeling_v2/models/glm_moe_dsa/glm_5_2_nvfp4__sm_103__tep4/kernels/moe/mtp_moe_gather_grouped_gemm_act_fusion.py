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
MTP-layer MoE FC1: in-kernel gather + grouped BF16 GEMM + SwiGLU, fused.

Replaces three production kernels with one:

    expandInputRowsKernel  ->  the cp.async warps gather A rows in-kernel
    GEMM1 (GemmUniversal)  ->  the mainloop
    doActivationKernel     ->  the epilogue

  For each expert i:
    acc       = A_gathered[i] x B[i]              // full N columns in TMEM
    C_rows[i] = up * gate * sigmoid(gate)         // N/2 columns out

  where  up   = acc[:, even subtile blocks]
         gate = acc[:, odd subtile blocks]

Data types: A, B and C are all BFloat16. No scale factors, no quantised output.

The gate/up layout, which is the one thing that does NOT carry over from the NVFP4
sibling. TRT-LLM's BF16 checkpoint stores W13 as two contiguous halves --
`load_expert_w3_w1_weight` writes w3 (up) into the first and w1 (gate) into the second,
and `doActivationKernel` reads the gated operand at `+ inter_size` -- so column j pairs
with column j + inter_size, 2048 apart. Those two columns fall in *different* N-tiles,
which a fused epilogue cannot pair without computing two tiles per CTA.

This kernel therefore requires W13 pre-permuted so up and gate interleave every
`epi_subtile_n` columns, exactly the layout the NVFP4 path already uses. That is not a
workaround invented here: the production CuTeDSL backend reorders its weights the same
way, at load time, for the same reason. The permutation is a one-off host-side copy with
no runtime cost, and `interleave_w13` below is the single definition of it. Feeding this
kernel the unpermuted BF16 checkpoint layout produces wrong numbers with correct shapes.

Constraints:
  - mma_tiler_n >= 128, i.e. at least 2 * epi_subtile_n, so a tile holds at least one
    complete up+gate pair for the SwiGLU epilogue. 64 is not valid.
  - N must be an even number of mma_tiler_n tiles
  - K divisible by the 64-element BF16 K granule

Runs on SM100+ (Blackwell) GPUs.
"""

from __future__ import annotations

import os

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import torch
from cutlass.experimental import primitives as prims

from ..common import compat48 as _compat48
from .device_bound_scheduler import DeviceBoundPersistentTileScheduler
from .glm_mtp_bf16_utils import (
    a_dtype,
    b_dtype,
    c_dtype,
    compute_ktile_config,
    mma_inst_k,
    mma_k_granule,
)
from .static_persistent_tile_scheduler import PersistentTileSchedulerParams, WorkTileInfo


def interleave_w13(w13: torch.Tensor, epi_subtile_n: int, dim: int = 0) -> torch.Tensor:
    """Reorder W13 along `dim` from [up | gate] halves to alternating blocks.

    The single definition of this kernel's weight-layout requirement. TRT-LLM's BF16
    checkpoint stores `w13[:inter] = up` and `w13[inter:] = gate`
    (`quantization.py::load_expert_w3_w1_weight`), while the fused epilogue needs the two
    operands of one SwiGLU inside one N-tile. So block b of up is followed by block b of
    gate:

        [up_0:g, gate_0:g, up_g:2g, gate_g:2g, ...]      g = epi_subtile_n

    Applied once at load time. Doing it in the kernel instead would mean each CTA
    computing two N-tiles 2048 columns apart, which buys nothing -- the weight traffic is
    identical either way and the weights are what this kernel is bound by.
    """
    n = w13.shape[dim]
    assert n % (2 * epi_subtile_n) == 0, (
        f"N={n} must be an even number of {epi_subtile_n}-column blocks"
    )
    half = n // 2
    blocks = half // epi_subtile_n
    order = torch.empty(n, dtype=torch.long, device=w13.device)
    base = torch.arange(epi_subtile_n, device=w13.device)
    for b in range(blocks):
        order[2 * b * epi_subtile_n : (2 * b + 1) * epi_subtile_n] = b * epi_subtile_n + base
        order[(2 * b + 1) * epi_subtile_n : (2 * b + 2) * epi_subtile_n] = (
            half + b * epi_subtile_n + base
        )
    return w13.index_select(dim, order)


# GLM's MoE has no per-expert FC1 bias, and the BF16 path confirms it: production passes
# `nullptr` for the GEMM1 biases (`moe_kernels.cu:3416`). The bias pipeline -- SMEM
# buffer, two mbarriers, a cp.async producer stage, an epilogue consumer stage -- is kept
# behind a const_expr rather than deleted, so the DSL folds it away at compile time and
# the runtime cost is zero.
HAS_BIAS = False

# Kernel configuration
smem_capacity = cutlass.memory.get_smem_capacity_in_bytes("sm_100")
num_mbar_bytes = 1024
num_tmem_alloc_cols = 512
epi_stages = 2  # Number of pipeline stages for epilog TMA stores

# CTA_2 collective MMA over a 2-CTA cluster. The pair splits M (each CTA supplies its own
# 128 A rows) and N (each CTA supplies its own half of B), so the collective tile is
# (2*128) x N x K while per-CTA SMEM per stage drops from A+B to A+B/2.
USE_2CTA = False


# 0 = use whatever SMEM allows. Diagnostic; see the note at its use site.
MAX_AB_STAGE = 0


# L2 eviction policies, one per operand stream. The pairing follows the reuse structure.
#
#   B (weights)  read exactly once. The 126 permuted rows spread over 57 activated experts
#                give every expert a single M-tile, so the N-tile loop reads each expert's
#                weights once and never returns to them. Nothing is gained by keeping them,
#                and keeping them evicts what would benefit.              -> evict_first
#   A (tokens)   re-read by every one of the 32 N-tiles per expert, and tiny: the gather
#                source is seq_len x hidden = 0.774 MB against 132.6 MB of L2. -> evict_last
#
# With cold weights and warm tokens, B's hint is the one that helps. A's hint has no effect
# here: the 0.774 MB gather source stays resident whatever the policy says. It is set anyway
# because it states the reuse structure and costs nothing, while evict_first on A is slower.
# On the FC2 sibling, whose A is much larger, the same hint helps on its own.
#
# The mechanism behind B's hint is not established. DRAM bytes and L2 hit rate are the same
# with and without it, and the gain shows only when launches run back to back.
L2_POLICY_A = "evict_last"
L2_POLICY_B = "evict_first"

# Programmatic dependent launch, matching what TRT-LLM sets for the production kernels.
# Defaults on so a model run overlaps the same way production does; the standalone
# benchmarks are unaffected because a single kernel has no predecessor to overlap with.
USE_PDL = os.environ.get("TRTLLM_ENABLE_PDL", "1").lower() in ("1", "true", "yes", "on")
# `no_allocate` is deliberately absent: it is a direct `.L2::no_allocate` qualifier on
# ld/st, not a `createpolicy.fractional` priority, and libNVVM rejects it there. Catching
# it here gives a readable message instead of "NVVM backend compilation failed".
for _name, _val in (
    ("L2_POLICY_A", L2_POLICY_A),
    ("L2_POLICY_B", L2_POLICY_B),
):
    assert _val in (
        "",
        "evict_first",
        "evict_last",
        "evict_normal",
    ), f"{_name}={_val!r} is not a valid createpolicy.fractional priority"

# Warp specialization: 12-warp with in-kernel cp.async gather + 2 idle warps
# Idle warps donate registers to epilogue warps via setmaxregister.
epilog_warp_id = (0, 1, 2, 3)
ldgsts_warp_id = (4, 5, 6, 7)
mma_warp_id = 8
tma_warp_id = 9
tile_sched_warp_id = 10  # scheduler producer warp
# Warp 11 was idle under CTA_1. Under CTA_2 it becomes the A-completion forwarder:
# `cp_async_mbarrier_arrive` is CTA-local, so the peer CTA's cp.async gather cannot signal
# the leader's ab_full barrier directly. Each CTA's gather lanes arrive on a local
# a_ready barrier; this warp waits that and re-arrives on the leader's ab_full through a
# `mapa` distributed-shared pointer.
fwd_warp_id = 11
threads_per_cta = 384  # 12 warps

# Register reconfig: decrease low-pressure warps, increase epilogue warps.
# Must be multiple of 8. Freed >= acquired per subpartition.
# SP0: w0(epilog→232) + w4(gather→80) + w8(mma→48) = 11520 regs ≤ 16384
# SP1: w1(epilog→232) + w5(gather→80) + w9(tma→40) = 11264 regs ≤ 16384
# SP2: w2(epilog→232) + w6(gather→80) + w10(sched→80) = 12544 regs ≤ 16384
# SP3: w3(epilog→232) + w7(gather→80) + w11(idle→24) = 10752 regs ≤ 16384
num_regs_epilog = 232
num_regs_ldgsts = 80
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

    Inserts fence.acq_rel.cta between the shared-memory load and mbarrier_arrive to prevent
    ptxas from scheduling the arrive before that load has completed.
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
    b_tensor: cute.Tensor,
    c_tensor: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    bias: cute.Tensor,
    # Gather-specific params (cute.Tensor — pointers extracted inside)
    a_original: cute.Tensor,
    token_mapping: cute.Tensor,
    mn_limit: cute.Tensor,
    problem_k: cutlass.Int32,
    bias_stride_n: cutlass.Int32,
    topK: cutlass.Constexpr[int],
    num_tiles_m_ptr: cutlass.Int64,
    max_tiles_m: cutlass.Int32,
    mma_tiler_n: cutlass.Constexpr[int],
    mma_tiler_k: cutlass.Constexpr[int],
    raster_along_m: cutlass.Constexpr[bool],
    # SM count, supplied by the caller rather than queried here: a
    # `torch.cuda.get_device_properties` call inside this function runs during
    # `cute.compile`, which would make ahead-of-time compilation require a GPU.
    # The caller passes the real value.
    num_sms: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
    round_gemm_to_bf16: cutlass.Constexpr[bool] = False,
    drain_epilogue: cutlass.Constexpr[bool] = False,
) -> None:
    """Host-side JIT for gather + grouped BF16 GEMM + SwiGLU.

    Sets up TMA descriptors and launches the GPU kernel. Runs on the host, JIT-compiled
    by the CuTe DSL.

    Args:
        b_tensor: Input matrix B, shape (K, N, L), BFloat16, L = number of experts.
            N carries up and gate interleaved every epi_subtile_n columns -- see
            `interleave_w13`; the checkpoint's two-halves layout is NOT accepted.
        c_tensor: Output matrix C, shape (N_out, valid_m, 1), BFloat16.
            N_out = N/2 because each up+gate pair collapses to one output column.
        tile_idx_to_expert_idx: Int32 (num_tiles_m,) mapping each M-tile to its expert
            index, used as B's L coordinate.
        bias: Float32 (L, N) per-expert bias in B-N space. Unused under HAS_BIAS=False,
            which is the production configuration.
        a_original: A in original token order, BF16, K-major. Gathered in-kernel; only
            the pointer and the K stride are used.
        token_mapping: Int32 token_id_mapping, the gather index.
        mn_limit: Int32 per-tile row validity limits.
        problem_k: The K dimension of the GEMM problem.
        bias_stride_n: N stride of the bias tensor.
        topK: Number of experts each token is routed to.
        num_tiles_m_ptr: Device pointer to the M-tile count, produced by the preceding
            kernel and therefore not readable on the host.
        mma_tiler_n: N dimension of the MMA tile (128 or 256). 64 is rejected: the
            SwiGLU epilogue consumes subtiles in up/gate pairs, so a tile narrower
            than 2 * epi_subtile_n cannot form one.
        stream: CUDA stream.
        round_gemm_to_bf16: Preserve a separate BF16 GEMM output before
            SwiGLU, as used by the production prefill activation kernel.
            False retains the existing fused/decode arithmetic.
        drain_epilogue: Explicitly finish the final TMA source read before
            the prefill CTA exits. False preserves the existing decode path.

    Note:
        - Grid is (num_tiles_m, ceil(N/tiler_N), 1) — L=1 since A and C are stacked.
        - TMA warp uses expert_idx to select B's slice per tile.
    """

    # MMA tile configuration for 1CTA
    mma_tiler_mnk = [128, mma_tiler_n, mma_tiler_k]  # Tile dimensions (M, N, K)
    # Per-CTA operand bytes. Under CTA_2 the pair splits N, so each CTA stages only half
    # of B; A is unchanged because the split along M gives each CTA its own 128 rows.
    # This is the whole source of the deeper pipeline: the stage shrinks by B/2.
    cta_tile_n = mma_tiler_mnk[1] // 2 if USE_2CTA else mma_tiler_mnk[1]
    # Per-instruction K granule: 16 for the dense F16/BF16 atom, against 64 for the
    # NVFP4 block-scaled one. num_kblocks and inst_k_bytes_16B both derive from this, so
    # carrying 64 over from the sibling kernel misprograms the whole K loop.
    mma_inst_mnk = [128, mma_tiler_n, mma_inst_k]
    # Epilogue subtile width == the up/gate interleave granularity, because the epilogue
    # pairs consecutive subtiles (even=up, odd=gate) into one SwiGLU output. Unlike the
    # NVFP4 path, where 64 is fixed by the checkpoint's own layout, this value is free
    # here -- the interleave is produced by `interleave_w13` and simply has to use the
    # same number. 64 is kept so both kernels have one answer to this question.
    epi_subtile_n = min(64, mma_tiler_n)

    # Rejected here rather than in the epilogue. mma_tiler_n=64 gives epi_subtile_n=64
    # and so a single subtile, which is half of an up/gate pair: the pair loop runs zero
    # times and the kernel emits nothing. The epilogue does assert this, but only once
    # tracing reaches it, which is a confusing place to learn that an argument was
    # invalid. This check covers every caller of host_function.
    assert mma_tiler_n >= 2 * epi_subtile_n, (
        f"mma_tiler_n ({mma_tiler_n}) must be at least 2 * epi_subtile_n "
        f"({epi_subtile_n}): the SwiGLU epilogue consumes subtiles in (up, gate) pairs, "
        f"so a narrower tile cannot form even one pair and would produce no output. "
        f"Valid values are 128 and 256."
    )

    # K-tile config
    _kc = compute_ktile_config(mma_tiler_k, mma_tiler_n)

    # Extract raw pointers from cute.Tensor for kernel consumption
    a_original_ptr = a_original.iterator.toint()
    token_mapping_ptr = token_mapping.iterator.toint()
    mn_limit_ptr = mn_limit.iterator.toint()
    # Byte stride between consecutive tokens in A. shape[0] is K in *elements*, so the
    # element width has to be applied -- 2 bytes here, against the NVFP4 sibling's half a
    # byte. Getting this wrong does not fault, it silently gathers the wrong rows.
    a_stride_k = a_original.shape[0] * a_dtype.width // 8

    # NO A TMA descriptor — A loaded by cp.async warps via in-kernel gather

    # B tile. K-major, L=num_experts. dtype/tma_format left to inference: BF16 has one
    # 16-bit format and no ambiguity, unlike the NVFP4 path where the default resolves to
    # an illegal padded encoding and has to be named explicitly.
    tma_b_desc = cuda.create_tensor_map_tiled_from_view(
        b_tensor,
        box_dims=(_kc["tma_b_box_k"], cta_tile_n, 1),  # (K_slice, per-CTA N, L_tile)
        stride_order=(0, 1, 2),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )

    # C TMA descriptor. The box's inner dimension is epi_subtile_n = 64 BF16 elements =
    # 128 B, so the swizzle is s128b and the epilogue's `store_swizzled` must use the
    # matching Swizzle(3,4,3). The NVFP4 sibling uses s32b / Swizzle(1,4,3) because its
    # 64 lanes are only 32 B. These two constants are a matched pair: disagreeing on them
    # scrambles the output within each subtile with no error reported anywhere.
    tma_c_desc = cuda.create_tensor_map_tiled_from_view(
        c_tensor,
        box_dims=(epi_subtile_n, mma_tiler_mnk[0], 1),
        stride_order=(0, 1, 2),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )

    # SMEM footprint. A and B are both BF16, so the SMEM and GMEM figures for B coincide;
    # the NVFP4 sibling keeps them separate only because TMA counts transactions against
    # the packed width.
    num_bytes_a = mma_tiler_mnk[0] * mma_tiler_mnk[2] * a_dtype.width // 8
    num_bytes_b_smem = cta_tile_n * mma_tiler_mnk[2] * b_dtype.width // 8
    num_bytes_b_gmem = num_bytes_b_smem
    # Persistent kernel: sC and sBias are separate allocations (not aliased).
    num_bytes_c_smem = epi_subtile_n * mma_tiler_mnk[0] * epi_stages * c_dtype.width // 8
    num_bytes_bias_smem = mma_tiler_mnk[1] * 4 if HAS_BIAS else 0  # Float32
    smem_for_ab = smem_capacity - num_mbar_bytes - num_bytes_c_smem - num_bytes_bias_smem
    ab_bytes_per_stage = num_bytes_a + num_bytes_b_smem
    num_ab_stage = smem_for_ab // ab_bytes_per_stage
    # Diagnostic cap on pipeline depth; see the same option in the FC2 sibling.
    # In the FC2 kernel 7 stages measured slower than 6, so the depth
    # that SMEM happens to allow is not automatically the right one.
    if cutlass.const_expr(MAX_AB_STAGE > 0):
        num_ab_stage = min(num_ab_stage, MAX_AB_STAGE)

    # Convert helper tensors to raw GPU pointers for kernel consumption
    tile_idx_ptr = tile_idx_to_expert_idx.iterator.toint()
    bias_ptr = bias.iterator.toint()

    # Persistent tile scheduling: tiles are distributed across a smaller grid.
    # Each CTA loops over multiple tiles via StaticPersistentTileScheduler.
    n_tiles = cute.ceil_div(b_tensor.shape[1], mma_tiler_mnk[1])
    tile_grid = (max_tiles_m, n_tiles, 1)
    # Under CTA_2 the launch unit is a 2-CTA cluster, so the scheduler hands out one work
    # tile per *cluster* and only half as many clusters fit on the same SM count.
    cluster_shape = (2, 1, 1) if USE_2CTA else (1, 1, 1)
    tile_sched_params = PersistentTileSchedulerParams(
        tile_grid, cluster_shape, raster_along_m=raster_along_m
    )
    max_active_clusters = num_sms // 2 if USE_2CTA else num_sms
    grid = DeviceBoundPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)
    if cutlass.const_expr(USE_2CTA):
        # get_grid_shape returns CTAs; the x extent must be an exact multiple of the
        # cluster x extent or the launch is rejected.
        assert grid[0] % 2 == 0, (
            f"grid {grid} is not compatible with cluster (2,1,1): the x extent must be "
            f"an even number of CTAs."
        )

    # TMA loads B only; A is gathered by the cp.async warps.
    num_tma_load_bytes = num_bytes_b_gmem

    # Launch the kernel
    mtp_moe_gather_grouped_gemm_act_fusion_kernel(
        tma_b_desc,
        tma_c_desc,
        tile_idx_ptr,
        bias_ptr,
        a_original_ptr,
        token_mapping_ptr,
        mn_limit_ptr,
        num_tiles_m_ptr,
        problem_k,
        bias_stride_n,
        a_stride_k,
        topK,
        num_tma_load_bytes,
        mma_tiler_mnk,
        mma_inst_mnk,
        epi_subtile_n,
        num_bytes_a,
        num_bytes_b_gmem,
        num_bytes_b_smem,
        num_ab_stage,
        tile_sched_params,
        DeviceBoundPersistentTileScheduler,
        round_gemm_to_bf16,
        drain_epilogue,
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        # The launch has to declare the cluster too, not just the scheduler: setting only
        # PersistentTileSchedulerParams(cluster_shape=...) changes how tiles are handed
        # out but leaves the launch at 1 CTA per cluster, and the driver then rejects the
        # kernel's cluster-scoped instructions with CUDA_ERROR_INVALID_CLUSTER_SIZE.
        cluster=(2, 1, 1) if USE_2CTA else (1, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
        use_pdl=USE_PDL,
    )
    return


# GPU device kernel.
#
# The kernel name matters. CuTe DSL mangles the cubin symbol as
# "kernel_cutlass_<python function name>_<repr of each argument>" and truncates at 180
# characters; the module and file names never appear. Both MoE kernels lead with
# `GridConstant[cuda.TensorMap]` arguments whose reprs are identical here and in
# moe_grouped_gemm_finalize_fusion.py (NVFP4, 128B swizzle, B4X16, box_dims (256,128,1)), and the cut lands
# 25 characters into the second descriptor -- inside the shared "TensorMapswizzleTensorMap"
# boilerplate, one character before the two would diverge. Named `kernel`, as the DSL
# examples conventionally are, the two emit the *same* symbol and every profiler merges
# them into one row. Keep the names distinct.
@cute.kernel
# Named distinctly from the NVFP4 sibling on purpose. CuTe DSL builds the cubin
# symbol from this function's name plus the argument reprs and truncates at 180
# characters, so two kernels called `kernel` with similar leading arguments share
# one symbol and nsys merges them into a single row -- which looks exactly like a
# correct single-kernel trace. The NVFP4 pair already occupies
# `moe_gather_grouped_gemm_act_fusion_kernel` and
# `moe_grouped_gemm_finalize_fusion_kernel`; these carry the `mtp_` prefix
# so a trace with both layers' MoE in it separates all four.
def mtp_moe_gather_grouped_gemm_act_fusion_kernel(
    tma_b_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_c_desc: cutlass.GridConstant[cuda.TensorMap],
    tile_idx_ptr: cutlass.Int64,
    bias_ptr: cutlass.Int64,
    a_original_ptr: cutlass.Int64,
    token_mapping_ptr: cutlass.Int64,
    mn_limit_ptr: cutlass.Int64,
    num_tiles_m_ptr: cutlass.Int64,
    problem_k: cutlass.Int32,
    bias_stride_n: cutlass.Int32,
    a_stride_k: cutlass.Int32,
    topK: cutlass.Constexpr[int],
    num_tma_load_bytes: cutlass.Constexpr[int],
    mma_tiler_mnk: cutlass.Constexpr[tuple[int, int, int]],
    mma_inst_mnk: cutlass.Constexpr[tuple[int, int, int]],
    epi_subtile_n: cutlass.Constexpr[int],
    num_bytes_a: cutlass.Constexpr[int],
    num_bytes_b_gmem: cutlass.Constexpr[int],
    num_bytes_b_smem: cutlass.Constexpr[int],
    num_ab_stage: cutlass.Constexpr[int],
    tile_sched_params: PersistentTileSchedulerParams,
    scheduler_type: cutlass.Constexpr,
    round_gemm_to_bf16: cutlass.Constexpr[bool] = False,
    drain_epilogue: cutlass.Constexpr[bool] = False,
) -> None:
    """GPU device kernel: in-kernel gather + contiguous grouped BF16 GEMM + SwiGLU.

    - tile_idx_to_expert_idx maps each M-tile to an expert index
    - TMA warp uses expert_idx to index B in the L dimension
    - cp.async warps gather A rows through token_mapping, replacing production's separate
      expandInputRowsKernel
    - the epilogue pairs even/odd subtiles into one SwiGLU output, replacing production's
      separate doActivationKernel

    Args:
        tma_b_desc: TMA descriptor for matrix B (BF16, K-major, L=num_experts).
        tma_c_desc: TMA descriptor for output C (BF16, L=1).
        tile_idx_ptr: GPU pointer to Int32 array mapping M-tile → expert index.
        bias_ptr: GPU pointer to Float32 array of per-expert bias in B-N space.
        a_original_ptr: Raw GPU pointer to A in original token order (BF16).
        token_mapping_ptr: Raw GPU pointer to Int32 token_id_mapping array.
        mn_limit_ptr: Raw GPU pointer to Int32 per-tile row validity limits.
        problem_k: K dimension of the GEMM.
        bias_stride_n: Bias tensor N stride.
        a_stride_k: Stride in bytes between consecutive tokens in A.
        topK: Number of experts each token is routed to.
        num_tiles_m: Total number of M-tiles (= len(tile_idx_to_expert_idx)).
        num_tma_load_bytes: Total bytes per TMA load (B only).
        mma_tiler_mnk: Tile dimensions [M, N, K].
        mma_inst_mnk: MMA instruction dimensions [M, N, K].
        epi_subtile_n: Epilog subtile N dimension.
        num_bytes_*: Byte sizes for SMEM allocation.
        num_ab_stage: A/B pipeline depth.

    Kernel Architecture (12-warp with cp.async gather + register reconfig):
        - Block: 384 threads (12 warps)
        - Warp roles: warp 0-3 (epilogue), warp 4-7 (cp.async gather A),
          warp 8 (MMA), warp 9 (TMA B), warp 10-11 (idle, donate regs)
        - Register reconfig: epilogue→232, gather→80, mma→48, tma→40, idle→24
        - Subtile pairs are walked forward and the accumulator is released after the
          last one. The NVFP4 sibling alternates direction and releases early at
          tiler_n=256, so the columns its two accumulators share are read before the
          MMA warp overwrites them; with disjoint accumulators there is nothing to
          read early.
    """
    #
    # K-tile config — computed inline from mma_tiler_mnk at compile time
    #
    # Columns of B this CTA stages: half the collective N tile under CTA_2.
    cta_tile_n: cutlass.Constexpr[int] = mma_tiler_mnk[1] // 2 if USE_2CTA else mma_tiler_mnk[1]
    _kc = compute_ktile_config(mma_tiler_mnk[2], mma_tiler_mnk[1])
    smem_row_bytes = _kc["smem_row_bytes"]
    slice_row_bytes = _kc["slice_row_bytes"]
    num_k_slices = _kc["num_k_slices"]
    tma_b_box_k = _kc["tma_b_box_k"]
    mma_layout = _kc["mma_layout"]
    mma_stride = _kc["mma_stride"]
    mma_leading_a = _kc["mma_leading_a"]
    # B's descriptor stride is the one k-tile constant that depends on N, and it must
    # describe the *per-CTA* tile: under CTA_2 each CTA stages only cta_tile_n columns.
    # It only bites when the K tile is sliced (num_k_slices > 1, i.e. mma_tiler_k=128) --
    # at mma_tiler_k=64 the value is the constant 16 and the mistake is invisible, which
    # is exactly the shape of the bug this replaced: k=64 bit-exact, k=128 wrong.
    if cutlass.const_expr(USE_2CTA):
        mma_leading_b = cta_tile_n * _kc["smem_row_bytes"] if _kc["num_k_slices"] > 1 else 16
    else:
        mma_leading_b = _kc["mma_leading_b"]
    k_chunks_per_slice = _kc["k_chunks_per_slice"]
    swizzle_mask = _kc["swizzle_mask"]
    slice_size_a = mma_tiler_mnk[0] * slice_row_bytes

    #
    # Thread / block identification
    #
    warp_idx = cute.arch.warp_idx()
    tidx, _, _ = cute.arch.thread_idx()
    k = problem_k

    #
    # Shared memory allocation (mbarriers + data tiles)
    #
    # acc_stages=2 enables MMA/epilogue overlap: MMA writes tile T+1 while the epilogue
    # reads tile T. One stage is mma_tiler_n TMEM columns of F32, so the constraint is
    # just 2*N <= num_tmem_alloc_cols.
    #
    # The accumulator is the only thing this kernel puts in TMEM -- there is no S2T stage
    # and no scale-factor region, because BF16 has no scale factors. That is what makes
    # the bound this simple, and it is why the NVFP4 siblings cannot use it: they need
    # 2*(N + SFA + SFB) columns, which forces one stage at N=256 and drives the
    # `overlapping_accum` machinery. At N=256 here the two accumulators are disjoint and
    # occupy exactly the 512 columns available.
    #
    # At N=256 the second accumulator stage does not change speed: the kernel is DRAM-bound,
    # so there is no exposed MMA time for the epilogue to hide behind. Two stages are allowed
    # because this kernel has no constraint against them, not for speed.
    acc_stages = 2 if 2 * mma_tiler_mnk[1] <= num_tmem_alloc_cols else 1
    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    # Guarded for the same reason as sBias below: with HAS_BIAS off these two are never
    # initialised, arrived on, or waited on, so allocating them only reserved SMEM.
    if cutlass.const_expr(HAS_BIAS):
        bias_full_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
        bias_empty_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    else:
        bias_full_mbar_ptr = None
        bias_empty_mbar_ptr = None
    num_tiles_m_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    # Warp-specialized scheduler: producer→consumer work_tile via SMEM
    sched_full_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    sched_empty_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
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
    )
    sB = cutlass.Array(
        cutlass.Int8,
        num_bytes_b_smem * num_ab_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )

    # Persistent kernel: sC and sBias are dedicated allocations (not aliased
    # to sA/sB) because mainloop of tile T+1 overlaps epilogue of tile T.
    num_bytes_c_smem = epi_subtile_n * mma_tiler_mnk[0] * epi_stages * c_dtype.width // 8
    sC = cutlass.Array(
        cutlass.Int8, num_bytes_c_smem, space=cutlass.AddressSpace.smem, alignment=1024
    )
    # Guarded so the allocation matches the SMEM budget above, which reserves for the bias
    # only when HAS_BIAS (`num_bytes_bias_smem = ... if HAS_BIAS else 0`). Allocating it
    # unconditionally took mma_tiler_n * 4 bytes the budget never accounted for -- 512 B at
    # mma_tiler_n=128, 1024 B at 256 -- which fitted only inside the remainder of the
    # `smem_for_ab // ab_bytes_per_stage` floor division, by arithmetic accident rather
    # than by construction.
    #
    # Binding None is safe: every sBias use is already inside `cutlass.const_expr(HAS_BIAS)`,
    # so the compile-time branch removes them together with the allocation. The FC2 kernel
    # guards sBias_dbuf the same way.
    if cutlass.const_expr(HAS_BIAS):
        sBias = cutlass.Array(
            cutlass.Float32,
            mma_tiler_mnk[1],
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
    else:
        sBias = None

    k_tile_cnt = cute.ceil_div(k, mma_tiler_mnk[2])

    # Rows a single work tile spans across the whole cluster: 2 * mma_tiler_m under
    # CTA_2 (the pair splits M), mma_tiler_m under CTA_1.
    cta_m_span: cutlass.Constexpr[int] = mma_tiler_mnk[0] * 2 if USE_2CTA else mma_tiler_mnk[0]

    # ---- Cluster identity -------------------------------------------------
    # Under CTA_1 there is one CTA per cluster and every predicate below folds away.
    if cutlass.const_expr(USE_2CTA):
        cluster_rank = cute.arch.block_idx_in_cluster()
        is_pair_leader = cluster_rank == 0
    else:
        cluster_rank = cutlass.Int32(0)
        is_pair_leader = True

    # A-completion relay, CTA_2 only. `cp_async_mbarrier_arrive` is CTA-local, so the
    # peer's gather lanes cannot signal the leader's ab_full. They signal this local
    # barrier instead and the forwarder warp re-arrives cluster-side.
    if cutlass.const_expr(USE_2CTA):
        a_ready_mbar_ptr = cutlass.Array(
            cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem, alignment=8
        )
    else:
        a_ready_mbar_ptr = None

    # Peer handshake before the collective TMEM dealloc. The two CTAs leave the persistent
    # loop at different times, and tcgen05_dealloc(group=CTA_2) is collective: one CTA
    # must not tear down TMEM while the other is still reading it.
    if cutlass.const_expr(USE_2CTA):
        tmem_dealloc_mbar_ptr = cutlass.Array(
            cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
        )
    else:
        tmem_dealloc_mbar_ptr = None

    # TMEM allocation synchronization barrier
    # MMA warp (8) and epilogue warps (0,1,2,3) = 5 warps * 32 threads = 160
    tmem_bar_id = 1
    tmem_bar_threads = 32 * len((mma_warp_id, *epilog_warp_id))

    # Epilog synchronization barrier
    epilog_sync_bar_id = 2
    threads_in_epilogue = 32 * len(epilog_warp_id)

    # Bias-full mbarrier: cp.async warps (producers) → epilogue warps (consumers)
    threads_ldgsts = 32 * len(ldgsts_warp_id)

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
                if cutlass.const_expr(USE_2CTA):
                    # Leader's ab_full collects: its own elected arrive_expect_tx (1,
                    # covering BOTH CTAs' TMA bytes -- group=CTA_2 routes the peer's
                    # complete_tx here) plus one forwarded A-ready arrive per CTA (2).
                    # The 128 gather lanes no longer arrive here; they arrive on the
                    # local a_ready relay, which the forwarder collapses into one.
                    prims.mbarrier_init(ab_full_mbar_ptr.subview(i), 3)
                    prims.mbarrier_init(a_ready_mbar_ptr.subview(i), 128)
                else:
                    # 129 producers: TMA warp (1) + 128 cp.async threads (4 warps × 32)
                    prims.mbarrier_init(ab_full_mbar_ptr.subview(i), 129)
                prims.mbarrier_init(ab_empty_mbar_ptr.subview(i), 1)
            for i in range(acc_stages):
                prims.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
                # acc_empty is a *peer-drain* barrier under CTA_2, not a local epilogue
                # barrier: acc_full wakes both CTAs and both epilogues drain the same
                # collective accumulator, so the leader must not reuse TMEM until the
                # peer has finished its tcgen05_ld too. One arrival per CTA (each CTA's
                # epilogue collapses its 4 warps into one mapped arrive).
                # A missing peer drain shows up as partial-K sums like 128/256/384.
                if cutlass.const_expr(USE_2CTA):
                    prims.mbarrier_init(acc_empty_mbar_ptr.subview(i), 2)
                else:
                    # 4 epilogue warps signal acc_empty via elect_sync (4 arrivals)
                    prims.mbarrier_init(acc_empty_mbar_ptr.subview(i), 4)
            if cutlass.const_expr(HAS_BIAS):
                prims.mbarrier_init(bias_full_mbar_ptr, threads_ldgsts)
                # bias_empty: 4 epilogue warps signal via elect_sync (4 arrivals)
                prims.mbarrier_init(bias_empty_mbar_ptr, 4)
            # Warp-specialized scheduler: producer (warp 10) → consumers (warps 0-9)
            # sched_full: 1 arrival (producer via elect_sync)
            prims.mbarrier_init(sched_full_mbar_ptr, 1)
            # sched_empty: every consumer thread arrives (sched_load_work_tile calls
            # mbarrier_arrive unguarded, so 32 per warp). Warps 0-9 always consume; under
            # CTA_2 warp 11 becomes the A-completion forwarder and consumes work tiles
            # too, so it has to be counted or the producer stalls one arrival short.
            _sched_consumer_warps: cutlass.Constexpr[int] = 11 if USE_2CTA else 10
            prims.mbarrier_init(sched_empty_mbar_ptr, 32 * _sched_consumer_warps)
            if cutlass.const_expr(USE_2CTA):
                # One elected lane on the peer arrives here; match the count to the
                # signalling lanes, not to the number of CTAs.
                prims.mbarrier_init(tmem_dealloc_mbar_ptr, 1)

    # This fence orders the mbar init operations with respect to the barrier below
    prims.fence_mbarrier_init()
    if cutlass.const_expr(USE_2CTA):
        # barrier_cta_sync is CTA-local and is NOT a cluster-formation fence. Without the
        # cluster pair the peer's mapa-addressed arrive can land on an mbar this CTA has
        # not yet observed as initialised, and the kernel deadlocks at the wait.
        prims.barrier_cluster_arrive_relaxed()
        prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)

    # Programmatic dependent launch. Production runs GLM-5.2 with PDL on, so without
    # this our kernel serializes fully behind its predecessor while the kernel it is
    # being compared against overlaps -- an e2e comparison would be measuring the
    # missing feature, not the kernel. Placed after the init barrier, matching
    # production: mbarriers must be constructed before any dependence is honoured.
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
    bias_full_phase_bit = 0
    bias_empty_phase_bit = 1

    # ---- Construct typed GMEM arrays from Int64 pointers ----
    tile_idx_arr = cutlass.Array(
        cutlass.inttoptr(tile_idx_ptr, mem_space=1, dtype=cutlass.Int32),
        dtype=cutlass.Int32,
    )
    token_map_arr = cutlass.Array(
        cutlass.inttoptr(token_mapping_ptr, mem_space=1, dtype=cutlass.Int32),
        dtype=cutlass.Int32,
    )
    mn_limit_arr = cutlass.Array(
        cutlass.inttoptr(mn_limit_ptr, mem_space=1, dtype=cutlass.Int32),
        dtype=cutlass.Int32,
    )

    # =========================================================================
    # Warp 11: idle under CTA_1; A-completion forwarder under CTA_2.
    #
    # The gather is cp.async, not TMA, and `cp_async_mbarrier_arrive` is CTA-local -- the
    # peer CTA physically cannot signal the leader's ab_full with it. So each CTA's 128
    # gather lanes arrive on the local a_ready relay, and this warp collapses that into a
    # single cluster-side arrive on the *leader's* ab_full through a `mapa` pointer.
    # B needs no such relay: its TMA carries group=CTA_2, which routes complete_tx to the
    # leader's mbar in hardware.
    #
    # One arrive per CTA per stage, matching the count of 3 the leader's ab_full is
    # initialised with (1 elected arrive_expect_tx + 2 forwarded A-ready).
    # =========================================================================
    if warp_idx == fwd_warp_id:
        if cutlass.const_expr(USE_2CTA):
            prims.setmaxregister(num_regs_sched, prims.SetMaxRegisterAction.DECREASE)
            fwd_stage_idx = 0
            fwd_phase = 0
            work_tile, sched_full_phase = sched_load_work_tile(
                s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, 0
            )
            while work_tile.is_valid_tile:
                for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                    cur = fwd_stage_idx
                    cur_phase = fwd_phase
                    fwd_stage_idx = fwd_stage_idx + 1
                    if fwd_stage_idx == num_ab_stage:
                        fwd_stage_idx = 0
                        fwd_phase = fwd_phase ^ 1
                    while not prims.mbarrier_try_wait_parity(
                        a_ready_mbar_ptr.subview(cur),
                        cur_phase,
                        time_limit=10_000_000,
                    ):
                        pass
                    if prims.elect_sync():
                        # Translate the *element* pointer, not the array: mapa returns a
                        # distributed-shared (addrspace 7) handle and subviewing that
                        # handle afterwards is not a supported form. Arrive on it
                        # directly -- downcasting to plain .shared loses the remote
                        # address-space contract and faults on SM100.
                        prims.mbarrier_arrive(
                            prims.mapa(ab_full_mbar_ptr.subview(cur), 0),
                            scope=prims.MemScope.CTA,
                        )
                work_tile, sched_full_phase = sched_load_work_tile(
                    s_work_tile,
                    sched_full_mbar_ptr,
                    sched_empty_mbar_ptr,
                    sched_full_phase,
                )
        else:
            prims.setmaxregister(num_regs_idle, prims.SetMaxRegisterAction.DECREASE)

    # =========================================================================
    # Warp 10: scheduler producer (tile_sched lives here only).
    # Writes work_tile to SMEM; consumer warps (0-9) read via mbarrier.
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
            # Compute next work_tile first, then wait for consumers to finish
            # reading previous — overlaps advance with consumer processing.
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
    # cp.async warps (warps 4-7): GATHER A from original token order → SMEM.
    #
    # This is the fusion that removes production's `expandInputRowsKernel`: instead of a
    # separate pass writing the gathered rows to DRAM for the GEMM to read back, the rows
    # are pulled straight into the mainloop's SMEM stage.
    #
    # The mapping is derived from the dtype rather than fixed: at BF16 a 64-element K tile
    # is 128 B, so 8 threads cover one token row and the 128 threads cover 16 rows per
    # issue, 8 issues per thread for the 128-row tile.
    # =========================================================================
    if warp_idx >= ldgsts_warp_id[0] and warp_idx <= ldgsts_warp_id[-1]:
        prims.setmaxregister(num_regs_ldgsts, prims.SetMaxRegisterAction.DECREASE)

        # A's L2 policy. The gather uses `cp.async`, not TMA, so the hint rides on the
        # PTX instruction's optional `.L2::cache_hint` operand rather than on a
        # `l2_cache_hint=` keyword. Built once, outside the tile loop.
        if cutlass.const_expr(L2_POLICY_A):
            a_policy = prims.inline_ptx_hl(
                f"createpolicy.fractional.L2::{L2_POLICY_A}.b64 {{$w0}}, 1.0;",
                write_only_types=[cutlass.Int64],
            )
        else:
            a_policy = None

        tidx_in_wg = tidx % 128
        threads_for_gather = 128
        bytes_per_load = 16
        # Thread-to-row mapping: 128 threads across 4 warps load 128 rows x K bytes
        threads_per_token = mma_tiler_mnk[2] * a_dtype.width // 8 // bytes_per_load
        rows_per_load = threads_for_gather // threads_per_token
        loads_per_thread = mma_tiler_mnk[0] // rows_per_load
        k_thread_offset = (tidx_in_wg % threads_per_token) * bytes_per_load

        # Tile-invariant: smem_off only depends on local_row, not tile coord.
        k_idx = tidx_in_wg % threads_per_token
        a_row_token_off = cutlass.Array(cutlass.Int32, loads_per_thread)
        a_row_smem_off = cutlass.Array(cutlass.Int32, loads_per_thread)
        for i in range(loads_per_thread):
            local_row = (tidx_in_wg // threads_per_token) + i * rows_per_load
            # SMEM address: XOR swizzle (k_chunk ^ row) prevents bank conflicts
            slice_idx = k_idx // k_chunks_per_slice
            k_chunk_in_slice = k_idx % k_chunks_per_slice
            swizzled_k = (k_chunk_in_slice ^ (local_row & swizzle_mask)) * bytes_per_load
            smem_off = slice_idx * slice_size_a + local_row * smem_row_bytes + swizzled_k
            a_row_smem_off.subview(i).store(smem_off)

        ldg_tidx = (warp_idx - ldgsts_warp_id[0]) * 32 + (tidx % 32)

        # ---- Persistent loop: cp.async warps ----
        # Read initial work_tile from SMEM (written by sched warp)
        work_tile, sched_full_phase = sched_load_work_tile(
            s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, 0
        )
        # AB pipeline producer state: stage index carries across tiles.
        ab_stage_idx = 0
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            tile_coord_m = cur_tile_coord[0]
            tile_coord_n = cur_tile_coord[1]
            # Under CTA_2 a work tile spans 2 * mma_tiler_m rows and this CTA gathers its
            # own half. mn_limit is an absolute bound in the same padded row space, so the
            # existing `abs_row < mn_limit` test keeps masking correctly and the peer's
            # half simply finds nothing valid whenever the expert has <= 128 rows.
            # tile_coord_m is already this CTA's own tile: under CTA_2 the scheduler
            # returns cluster_coord * cluster_shape_m + cta_id_in_cluster, so the pair
            # gets consecutive M-tiles. Adding a rank offset on top would double-count it.
            # N is the opposite case -- both CTAs get the same tile_coord_n, so the B
            # fetch below *does* apply a rank offset.
            coord_m = tile_coord_m * mma_tiler_mnk[0]
            expert_idx = tile_idx_arr.subview(tile_coord_m).load()
            mn_limit = mn_limit_arr.subview(tile_coord_m).load()

            # Re-compute per-thread gather addresses for this tile
            for i in range(loads_per_thread):
                local_row = (tidx_in_wg // threads_per_token) + i * rows_per_load
                abs_row = coord_m + local_row
                a_row_token_off.subview(i).store(0)
                if abs_row < mn_limit:
                    token_id = token_map_arr.subview(abs_row).load()
                    original_token = token_id // topK
                    a_row_token_off.subview(i).store(original_token * a_stride_k)

            # cp.async pipeline: acquire-and-advance pattern (stage carries across tiles)
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

                # Bytes, not elements. `a_original_ptr`, `token_byte_off` and
                # `k_thread_offset` are all byte quantities, so a K offset in elements
                # advanced A by 2x for 4-bit data -- and by exactly 0 on the first
                # k_tile, which is why every single-K-tile configuration passed. Same
                # unit mix as the earlier kblocks_per_slice defect.
                k_base_bytes = k_tile * mma_tiler_mnk[2] * a_dtype.width // 8
                sA_stage_base = sA.subview(current_ab_stage * num_bytes_a)

                # Gather: combine original token offset (from routing) with K-tile offset
                for i in range(loads_per_thread):
                    local_row = (tidx_in_wg // threads_per_token) + i * rows_per_load
                    token_byte_off = a_row_token_off.subview(i).load()
                    smem_off = a_row_smem_off.subview(i).load()
                    smem_ir = sA_stage_base.subview(smem_off).data_ptr()
                    gmem_ir = a_original_ptr + token_byte_off + k_base_bytes + k_thread_offset
                    gmem_ir = cutlass.inttoptr(gmem_ir, mem_space=1, dtype=a_dtype)
                    is_valid_m = coord_m + local_row < mn_limit
                    if cutlass.const_expr(mma_tiler_mnk[2] > mma_k_granule):
                        # A K tile wider than one granule can be partial at the tail:
                        # M OOB → skip (predicate off), K OOB with valid M → zero-fill
                        # (cp_size=0). Both sides in bytes; k_thread_offset is a byte
                        # offset, and using an element offset there advanced A by the
                        # wrong factor for any dtype that is not one byte wide.
                        is_valid_k = k_base_bytes + k_thread_offset < k * a_dtype.width // 8
                        cp_sz = cutlass.Int32(is_valid_k * bytes_per_load)
                        # Predicated cp.async has no NVVM op form — emit @p cp.async inline.
                        if cutlass.const_expr(L2_POLICY_A):
                            prims.inline_ptx_hl(
                                f"cp.async.cg.shared.global.L2::cache_hint "
                                f"[{{$r0}}], [{{$r1}}], {bytes_per_load}, {{$r2}}, {{$r3}};",
                                read_only_args=[smem_ir, gmem_ir, cp_sz, a_policy],
                                pred=is_valid_m,
                            )
                        else:
                            prims.inline_ptx_hl(
                                f"cp.async.cg.shared.global [{{$r0}}], [{{$r1}}], {bytes_per_load}, {{$r2}};",
                                read_only_args=[smem_ir, gmem_ir, cp_sz],
                                pred=is_valid_m,
                            )
                    else:
                        # K always aligned: M OOB → skip entirely.
                        # No zero-fill needed — epilogue masks invalid rows.
                        if cutlass.const_expr(L2_POLICY_A):
                            prims.inline_ptx_hl(
                                f"cp.async.cg.shared.global.L2::cache_hint "
                                f"[{{$r0}}], [{{$r1}}], {bytes_per_load}, {{$r2}};",
                                read_only_args=[smem_ir, gmem_ir, a_policy],
                                pred=is_valid_m,
                            )
                        else:
                            prims.inline_ptx_hl(
                                f"cp.async.cg.shared.global [{{$r0}}], [{{$r1}}], {bytes_per_load};",
                                read_only_args=[smem_ir, gmem_ir],
                                pred=is_valid_m,
                            )

                # CTA_1: signal ab_full directly. CTA_2: signal the CTA-local relay --
                # this arrive cannot cross to the leader's SMEM, which is the whole
                # reason the forwarder warp exists.
                if cutlass.const_expr(USE_2CTA):
                    prims.cp_async_mbarrier_arrive(
                        a_ready_mbar_ptr.subview(current_ab_stage), noinc=True
                    )
                else:
                    prims.cp_async_mbarrier_arrive(
                        ab_full_mbar_ptr.subview(current_ab_stage), noinc=True
                    )

            # Wait for epilogue to finish reading previous bias before overwriting.
            if cutlass.const_expr(HAS_BIAS):
                while not prims.mbarrier_try_wait_parity(bias_empty_mbar_ptr, bias_empty_phase_bit):
                    pass
                bias_empty_phase_bit = bias_empty_phase_bit ^ 1

            if cutlass.const_expr(HAS_BIAS):
                # Bias preload for this tile into dedicated sBias buffer via cp.async.
                # Uses cp.async + cp.async.mbarrier.arrive to signal bias_full mbarrier.
                coordc_n_in_ldg = tile_coord_n * mma_tiler_mnk[1]
                bias_expert_byte_base = (
                    bias_ptr + cutlass.Int64(expert_idx) * cutlass.Int64(bias_stride_n) * 4
                )
                # Bias load: N>=128 -> each thread loads N/128 elements, N<128 -> one element per thread
                if cutlass.const_expr(mma_tiler_mnk[1] >= 128):
                    # N>=128: each of 128 threads copies bias_vec_size*4 bytes
                    bias_vec_size = mma_tiler_mnk[1] // 128
                    bias_cp_bytes = bias_vec_size * 4
                    bias_base = ldg_tidx * bias_vec_size
                    bias_col = coordc_n_in_ldg + bias_base
                    bias_smem_ir = sBias.subview(bias_base)
                    bias_gmem_ir = (
                        bias_expert_byte_base + cutlass.Int64(coordc_n_in_ldg + bias_base) * 4
                    )
                    bias_gmem_ir = cutlass.inttoptr(
                        bias_gmem_ir, mem_space=1, dtype=cutlass.Float32
                    )
                    bias_valid = bias_col + bias_vec_size <= bias_stride_n
                    bias_cp_sz = bias_valid * bias_cp_bytes
                    prims.cp_async_shared_global(
                        bias_smem_ir,
                        bias_gmem_ir,
                        size=bias_cp_bytes,
                        modifier="ca",
                        cp_size=bias_cp_sz,
                    )
                else:
                    # N<128: each thread copies 4 bytes (1 Float32) if in range.
                    # Clamp index to stay within sBias allocation for OOB threads
                    # (cp_size=0 still needs a valid SMEM address).
                    bias_idx = (ldg_tidx < mma_tiler_mnk[1]) * ldg_tidx
                    bias_smem_ir = sBias.subview(bias_idx)
                    bias_gmem_ir = (
                        bias_expert_byte_base + cutlass.Int64(coordc_n_in_ldg + bias_idx) * 4
                    )
                    bias_gmem_ir = cutlass.inttoptr(
                        bias_gmem_ir, mem_space=1, dtype=cutlass.Float32
                    )
                    bias_valid = (ldg_tidx < mma_tiler_mnk[1]) & (
                        coordc_n_in_ldg + ldg_tidx < bias_stride_n
                    )
                    bias_cp_sz = bias_valid * 4
                    prims.cp_async_shared_global(
                        bias_smem_ir,
                        bias_gmem_ir,
                        size=4,
                        modifier="ca",
                        cp_size=bias_cp_sz,
                    )
                # cp.async.mbarrier.arrive: signal bias_full mbarrier when cp.async completes.
                prims.cp_async_mbarrier_arrive(bias_full_mbar_ptr, noinc=True)

            # Wait for producer to write next work_tile
            work_tile, sched_full_phase = sched_load_work_tile(
                s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, sched_full_phase
            )

    # =========================================================================
    # TMA warp (warp 9): GMEM -> SMEM loads for B (A is gathered by the cp.async warps)
    # =========================================================================
    if warp_idx == tma_warp_id:
        prims.setmaxregister(num_regs_tma, prims.SetMaxRegisterAction.DECREASE)

        # L2 eviction policy for the B stream. Measured B-tile reuse is 1 and the L2 hit
        # rate is 1.5-2.5%: every weight byte is read exactly once, so allocating L2
        # lines for it can only displace the one tensor that *is* reused -- A, whose
        # ~393 KB per expert serves all 32 n-tiles.
        #
        # There is no `createpolicy` wrapper in the primitives package; the 64-bit
        # descriptor `l2_cache_hint` wants is built with one PTX instruction.
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
        # Read initial work_tile from SMEM
        work_tile, sched_full_phase = sched_load_work_tile(
            s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, 0
        )
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            tile_coord_m = cur_tile_coord[0]
            tile_coord_n = cur_tile_coord[1]
            # Under CTA_2 the pair splits N: this CTA fetches its own cta_tile_n columns
            # of the collective N tile.
            coord_n = tile_coord_n * mma_tiler_mnk[1] + cluster_rank * cta_tile_n
            expert_idx = tile_idx_arr.subview(tile_coord_m).load()
            coord_l_b = expert_idx

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
                # Only the pair leader sets the transaction budget, and it must cover
                # BOTH CTAs' bytes: group=CTA_2 masks bit 24 of the mbar pointer so every
                # CTA's complete_tx is redirected to the leader's barrier. Setting the
                # single-CTA total here leaves the bar pending on the missing half
                # forever. Under CTA_1 this is the plain local case.
                if is_pair_leader:
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            ab_full_mbar_ptr.subview(current_ab_stage),
                            num_tma_load_bytes * (2 if cutlass.const_expr(USE_2CTA) else 1),
                        )

                coord_k = k_tile * mma_tiler_mnk[2]
                sB_staged = sB.subview(num_bytes_b_smem * current_ab_stage)

                if prims.elect_sync():
                    # B may be K-sliced: each slice = tma_b_box_k K-elements into SMEM
                    slice_size_b = cta_tile_n * slice_row_bytes
                    for slice_i in cutlass.range_constexpr(num_k_slices):
                        coord_k_slice = coord_k + slice_i * tma_b_box_k
                        sB_slice = sB_staged.subview(slice_i * slice_size_b)
                        if cutlass.const_expr(USE_2CTA):
                            # Per-CTA unicast: every CTA issues its own slice, so
                            # multicast_mask is OMITTED. Passing 1 << cluster_rank would
                            # deliver the same bytes but still emit a multicast TMA load and pay
                            # the routing tax; passing 3 with both CTAs issuing doubles
                            # delivery and overruns the leader's tx budget (hang).
                            # Note the argument order differs from the _shared_cta_
                            # variant: coords come BEFORE the mbar here.
                            prims.cp_async_bulk_tensor_shared_cluster_global(
                                sB_slice,
                                tma_b_desc.get_ptr(),
                                (coord_k_slice, coord_n, coord_l_b),
                                ab_full_mbar_ptr.subview(current_ab_stage),
                                [],
                                group=prims.CTAGroup.CTA_2,
                                l2_cache_hint=b_policy,
                            )
                        else:
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
    # MMA warp (warp 8): TMEM alloc and MMA issue. No S2T stage -- that exists in the
    # NVFP4 sibling to copy scale factors SMEM -> TMEM, and there are none here.
    # =========================================================================
    if warp_idx == mma_warp_id:
        prims.setmaxregister(num_regs_mma, prims.SetMaxRegisterAction.DECREASE)
        # Allocate TMEM using cutlass - MMA warp is the allocator (once, before loop)
        # Both CTAs run this -- tcgen05_alloc with group=CTA_2 is collective over the
        # pair, and the group here must match the group at tcgen05_dealloc or the
        # lifecycle is undefined. Only the MMA *issue* below is leader-only.
        if cutlass.const_expr(USE_2CTA):
            prims.tcgen05_alloc(tmem_ptr_i32, num_tmem_alloc_cols, group=prims.CTAGroup.CTA_2)
            prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_2)
        else:
            prims.tcgen05_alloc(tmem_ptr_i32, num_tmem_alloc_cols)
            prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
        # Sync with epilogue warps after allocation
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)
        # Retrieve TMEM pointer from smem buffer (returns cutlass.Array)
        tmem_raw_addr = tmem_ptr_i32.load()
        acc_tmem_ptr = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Float32)

        # MMA instruction descriptor, loop-invariant.
        #
        # `Tcgen05InstrDesc`, not the Mx/Omma variants: those carry scale-format and FP4
        # k_dim fields in a different bit layout and exist for the block-scaled kinds.
        # `c_dtype=Float32` selects the F32 accumulator; the field's default of 0 is the
        # F16-accumulator layout, which would not fault -- it would quietly accumulate a
        # K of 6144 in half precision.
        #
        # a_major / b_major stay 0 (K-major) and must agree with the s128b SMEM
        # descriptors below; a mismatch reads valid SMEM as if transposed.
        # n_dim / m_dim describe the *collective* tile, not this CTA's slice. Under
        # CTA_2 the pair computes 2*mma_tiler_m rows; N is unchanged because each CTA
        # contributes half of B and the accumulator spans the full N in both CTAs' TMEM.
        idesc = prims.Tcgen05InstrDesc.build(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=cutlass.Float32,
            n_dim=mma_tiler_mnk[1],
            m_dim=cta_m_span if cutlass.const_expr(USE_2CTA) else mma_tiler_mnk[0],
        )

        # Set ACCUMULATE field to False for the first k_tile iteration
        scale_d = False

        # TMEM addresses packed as (row_id << 16) | col_id. With acc_stages=2 the two
        # accumulators are simply adjacent: [ACC_0][ACC_1], one F32 column per output
        # column. The NVFP4 sibling has to interleave SFA/SFB regions here and, at
        # tiler_n 256, overlap the two accumulators to make them fit -- neither applies
        # once the scale factors are gone.
        tmem_base_col = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        cols_per_acc_stage = mma_tiler_mnk[1]
        num_kblocks = cute.ceil_div(mma_tiler_mnk[2], mma_inst_mnk[2])

        # Build SMEM descriptors once (stage 0) — only start address changes per stage.
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
        # Stage offsets in 16B granularity for tcgen05_update_smem_start_address.
        stage_inc_a = num_bytes_a >> 4
        stage_inc_b = num_bytes_b_smem >> 4

        # Only the pair leader issues the collective MMA. The peer CTA supplies its A
        # rows and B columns through SMEM and its warp 8 must NOT run this loop: it has
        # no arrives on its local ab_full (group=CTA_2 routes them all to the leader) so
        # it would hang on the first wait, and a second issuer would double the commit
        # arrives and overflow the leader's barrier.
        # It still has to drain the scheduler, because sched_empty counts warp 8 in both
        # CTAs and the producer stalls otherwise.
        if cutlass.const_expr(USE_2CTA) and not is_pair_leader:
            work_tile, sched_full_phase = sched_load_work_tile(
                s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, 0
            )
            while work_tile.is_valid_tile:
                work_tile, sched_full_phase = sched_load_work_tile(
                    s_work_tile,
                    sched_full_mbar_ptr,
                    sched_empty_mbar_ptr,
                    sched_full_phase,
                )
        else:
            # ---- Persistent loop: MMA warp ----
            # Read initial work_tile from SMEM
            work_tile, sched_full_phase = sched_load_work_tile(
                s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, 0
            )
            # AB pipeline consumer state: stage index carries across tiles.
            ab_stage_idx = 0
            acc_stage_idx = 0
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                tile_coord_n = cur_tile_coord[1]

                # Wait for acc buffer empty (from epilogue)
                while not prims.mbarrier_try_wait_parity(
                    acc_empty_mbar_ptr.subview(acc_stage_idx),
                    acc_empty_phase_bit,
                    time_limit=10_000_000,
                ):
                    pass
                base_col_id = tmem_base_col + acc_stage_idx * cols_per_acc_stage
                # Update acc_tmem_ptr for MMA output
                acc_tmem_ptr = prims.make_tmem_ptr(
                    (base_row_id << 16) | base_col_id, cutlass.Float32
                )

                scale_d = False
                for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                    current_ab_stage = ab_stage_idx
                    current_ab_full_phase = ab_mbar_full_phase_bit
                    ab_stage_idx = ab_stage_idx + 1
                    if ab_stage_idx == num_ab_stage:
                        ab_stage_idx = 0
                        ab_mbar_full_phase_bit = ab_mbar_full_phase_bit ^ 1

                    while not prims.mbarrier_try_wait_parity(
                        ab_full_mbar_ptr.subview(current_ab_stage),
                        current_ab_full_phase,
                        time_limit=10_000_000,
                    ):
                        pass

                    desc_a_nvvm_mma_base = desc_a_mma_base + cutlass.Int64(
                        current_ab_stage * stage_inc_a
                    )
                    desc_b_nvvm_mma_base = desc_b_mma_base + cutlass.Int64(
                        current_ab_stage * stage_inc_b
                    )

                    # K-blocks: partition SMEM slices into MMA instruction blocks. This is a
                    # count of instructions, so both operands must be in elements --
                    # mma_tiler_k / mma_inst_k, never a byte quantity over an element one.
                    kblocks_per_slice = num_kblocks // num_k_slices
                    inst_k_bytes_16B = (mma_inst_mnk[2] * a_dtype.width // 8) >> 4
                    slice_size_a_16B = mma_tiler_mnk[0] * smem_row_bytes // 16
                    # Per-CTA, like mma_leading_b above: under CTA_2 this CTA's B slice
                    # is cta_tile_n rows, not the collective N. Both constants are only
                    # read when the K tile is sliced, so getting them wrong is invisible
                    # at mma_tiler_k=64 and corrupts every result at 128.
                    slice_size_b_16B = cta_tile_n * smem_row_bytes // 16
                    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                        slice_idx = kblock_idx // kblocks_per_slice
                        kblock_in_slice = kblock_idx % kblocks_per_slice
                        inc_a = slice_idx * slice_size_a_16B + kblock_in_slice * inst_k_bytes_16B
                        inc_b = slice_idx * slice_size_b_16B + kblock_in_slice * inst_k_bytes_16B

                        desc_a_nvvm_mma = desc_a_nvvm_mma_base + inc_a
                        desc_b_nvvm_mma = desc_b_nvvm_mma_base + inc_b

                        if prims.elect_sync():
                            # kind::f16 covers both F16 and BF16 operands; the operand format
                            # is carried by the instruction descriptor, not by the kind. K is
                            # 16 elements per instruction.
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16,
                                # Must match the group used at tcgen05_alloc: libNVVM
                                # rejects a kernel that mixes CTA_1 and CTA_2 tcgen05
                                # instructions, and the error it gives says only
                                # "NVVM backend compilation failed".
                                prims.CTAGroup.CTA_2
                                if cutlass.const_expr(USE_2CTA)
                                else prims.CTAGroup.CTA_1,
                                acc_tmem_ptr,
                                desc_a_nvvm_mma,
                                desc_b_nvvm_mma,
                                idesc,
                                scale_d,
                            )
                        # scale_d=True: accumulate (first kblock overwrites, rest add)
                        scale_d = True

                    if prims.elect_sync():
                        # mask=3 broadcasts the arrive to both CTAs of the pair. The peer's
                        # TMA/gather warps wait on their own local ab_empty copy, and only
                        # this multicast delivers it -- a narrow 1<<rank mask deadlocks them.
                        if cutlass.const_expr(USE_2CTA):
                            prims.tcgen05_commit(
                                ab_empty_mbar_ptr.subview(current_ab_stage),
                                multicast_mask=3,
                                group=prims.CTAGroup.CTA_2,
                            )
                        else:
                            prims.tcgen05_commit(ab_empty_mbar_ptr.subview(current_ab_stage))

                # Signal acc ready for epilogue
                if prims.elect_sync():
                    # Both CTAs' epilogues drain this accumulator, so both must be woken.
                    if cutlass.const_expr(USE_2CTA):
                        prims.tcgen05_commit(
                            acc_full_mbar_ptr.subview(acc_stage_idx),
                            multicast_mask=3,
                            group=prims.CTAGroup.CTA_2,
                        )
                    else:
                        prims.tcgen05_commit(acc_full_mbar_ptr.subview(acc_stage_idx))

                acc_stage_idx = acc_stage_idx + 1
                if acc_stage_idx == acc_stages:
                    acc_stage_idx = 0
                    acc_empty_phase_bit = acc_empty_phase_bit ^ 1

                # Wait for producer to write next work_tile
                work_tile, sched_full_phase = sched_load_work_tile(
                    s_work_tile,
                    sched_full_mbar_ptr,
                    sched_empty_mbar_ptr,
                    sched_full_phase,
                )
    # =========================================================================
    # Epilogue warps (warps 0-3): T2R -> SwiGLU -> BF16 -> SMEM -> TMA store
    # =========================================================================
    if warp_idx < ldgsts_warp_id[0]:
        prims.setmaxregister(num_regs_epilog, prims.SetMaxRegisterAction.INCREASE)
        # Sync with MMA warp after TMEM allocation (once, before loop)
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)

        # Retrieve TMEM pointer from smem buffer (same as MMA warp)
        tmem_raw_addr = tmem_ptr_i32.load()
        acc_tmem_ptr = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Float32)

        tmem_base_col = tmem_raw_addr & 0xFFFF
        # Must match the MMA warp exactly: two disjoint accumulators, mma_tiler_n F32
        # columns each.
        cols_per_acc_stage = mma_tiler_mnk[1]

        # T2R instruction configuration (tile-invariant)
        t2r_inst_shape = "32x32b"
        t2r_inst_repx = epi_subtile_n
        # Bytes per epilogue stage in sC. Must agree with num_bytes_c_smem, which carries
        # the same c_dtype.width // 8. At BF16 with epi_subtile_n = 64 this is 16384 B a
        # stage, four times the NVFP4 sibling's figure -- which is why the SMEM budget
        # above has less room for A/B stages than that kernel's.
        epi_subtile_size = mma_tiler_mnk[0] * epi_subtile_n * c_dtype.width // 8
        subtile_cnt = cute.ceil_div(mma_tiler_mnk[1], epi_subtile_n)
        # SwiGLU: process pairs of subtiles → output count is half
        num_output_subtiles = subtile_cnt // 2

        assert num_output_subtiles >= 1, (
            f"mma_tiler_n ({mma_tiler_mnk[1]}) < 2 * epi_subtile_n ({epi_subtile_n}): a "
            f"tile this narrow holds a single interleave block and can never form an "
            f"up/gate pair, so the epilogue would silently emit nothing. Production "
            f"rejects the same configuration -- only mma_tiler_n 128 and 256 are valid."
        )

        # ---- Persistent loop: Epilogue warps ----
        # Read initial work_tile from SMEM
        work_tile, sched_full_phase = sched_load_work_tile(
            s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, 0
        )
        acc_stage_idx = 0
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            tile_coord_m = cur_tile_coord[0]
            tile_coord_n = cur_tile_coord[1]
            # Compute per-tile coordinates
            # Same row base the gather used -- this CTA's own M-tile.
            coordc_m = tile_coord_m * mma_tiler_mnk[0]
            coordc_n_out = tile_coord_n * (mma_tiler_mnk[1] // 2)
            expert_idx = tile_idx_arr.subview(tile_coord_m).load()
            mn_limit = mn_limit_arr.subview(tile_coord_m).load()

            # Subtile pairs are always walked forward and the accumulator released after
            # the last one. The NVFP4 sibling alternates the direction at tiler_n >= 256
            # and releases early, so the columns its two accumulators share get read
            # before the MMA warp overwrites them; with disjoint accumulators there is
            # nothing to read early.
            release_idx = num_output_subtiles - 1

            # Wait for cp.async warps to finish loading bias to sBias.
            if cutlass.const_expr(HAS_BIAS):
                while not prims.mbarrier_try_wait_parity(bias_full_mbar_ptr, bias_full_phase_bit):
                    pass

            # Wait for acc ready from MMA warp
            while not prims.mbarrier_try_wait_parity(
                acc_full_mbar_ptr.subview(acc_stage_idx), acc_full_phase_bit
            ):
                pass

            # Per-stage TMEM col offset for T2R reads
            base_col_id = tmem_base_col + acc_stage_idx * cols_per_acc_stage

            epi_stage_idx = 0

            # Epilogue tiling loop — process pairs: (up_subtile, gate_subtile)
            for pair_idx in cutlass.range(num_output_subtiles):
                real_pair_idx = pair_idx
                up_subtile_idx = real_pair_idx * 2
                up_tmem_ptr = prims.make_tmem_ptr_from_warp_row_col(
                    tmem_raw_addr,
                    warp_idx,
                    base_col_id + up_subtile_idx * t2r_inst_repx,
                    cutlass.Float32,
                )
                up_f32 = prims.tcgen05_ld(t2r_inst_shape, up_tmem_ptr, num=t2r_inst_repx)

                gate_subtile_idx = real_pair_idx * 2 + 1
                gate_tmem_ptr = prims.make_tmem_ptr_from_warp_row_col(
                    tmem_raw_addr,
                    warp_idx,
                    base_col_id + gate_subtile_idx * t2r_inst_repx,
                    cutlass.Float32,
                )
                gate_f32 = prims.tcgen05_ld(t2r_inst_shape, gate_tmem_ptr, num=t2r_inst_repx)
                # NOTE: the SwiGLU below adds no bias, because GLM's FC1 has no bias
                # (`moe_kernels.cu:3416` passes nullptr) and HAS_BIAS is False. Setting
                # HAS_BIAS to True only stages a bias in SMEM; the epilogue has to be
                # completed before that flag means anything.

                # Release acc buffer: at release_idx (early for overlap,
                # last pair for non-overlap).
                if pair_idx == release_idx:
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    if cutlass.const_expr(USE_2CTA):
                        # Peer-drain, not a local release. Both CTAs' epilogues read the
                        # same collective accumulator, so the leader must not start the
                        # next tile until BOTH have finished tcgen05_ld. Converge this
                        # CTA's 4 epilogue warps first, then one elected thread arrives
                        # once on the *leader's* acc_empty through mapa. A local count-4
                        # release would let the leader overwrite TMEM while the peer is
                        # still issuing loads -- the documented failure signature is
                        # partial-K sums (128/256/384) instead of the full-K result.
                        prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)
                        if warp_idx == epilog_warp_id[0]:
                            if prims.elect_sync():
                                prims.mbarrier_arrive(
                                    prims.mapa(acc_empty_mbar_ptr.subview(acc_stage_idx), 0),
                                    scope=prims.MemScope.CTA,
                                )
                    else:
                        if prims.elect_sync():
                            prims.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage_idx))
                    if cutlass.const_expr(HAS_BIAS and mma_tiler_mnk[1] < 256):
                        if prims.elect_sync():
                            prims.mbarrier_arrive(bias_empty_mbar_ptr)

                swiglu_out = cutlass.vector.full_like(up_f32, 0.0)
                abs_row_epi = coordc_m + tidx
                if abs_row_epi < mn_limit:
                    # A separate BF16 GEMM followed by FP32 activation rounds
                    # both operands here. Keep that boundary in registers so
                    # prefill fusion preserves it without a global temporary.
                    if cutlass.const_expr(round_gemm_to_bf16):
                        up_f32 = up_f32.to(cutlass.BFloat16).to(cutlass.Float32)
                        gate_f32 = gate_f32.to(cutlass.BFloat16).to(cutlass.Float32)
                    # SwiGLU, exactly production's BF16 form:
                    #   out = up * gate * sigmoid(gate)
                    # from `GLUAdaptor<SiLu>` (moe_kernels.cuh:46-59) with the SiLu
                    # alpha/beta/limit at their defaults, so no clamp and no `+1` on up.
                    # The NVFP4 sibling additionally multiplies both operands by the
                    # global dequant scale; BF16 has no such scale -- production's
                    # activation applies none at all, `fc2_fp8_quant` being asserted null
                    # (moe_kernels.cu:4195-4196).
                    #
                    # exp2(-x*log2e) == exp(-x); exp2 is the cheaper GPU primitive.
                    log2e = cutlass.vector.full_like(gate_f32, 1.4426950408889634)
                    exp_neg = cute.math.exp2(-gate_f32 * log2e, fastmath=True)
                    one = cutlass.vector.full_like(gate_f32, 1.0)
                    sigmoid = cute.math.rcp(one + exp_neg, approx=True, ftz=True)
                    if cutlass.const_expr(round_gemm_to_bf16):
                        # GLUAdaptor<SiLu> computes SiLu(gate) * up. Preserve
                        # its FP32 multiply association as well as the BF16
                        # GEMM boundary in the opt-in prefill specialization.
                        swiglu_out = (sigmoid * gate_f32) * up_f32
                    else:
                        swiglu_out = up_f32 * gate_f32 * sigmoid

                epi_stage_idx = (epi_stage_idx + 1) % epi_stages
                smem_subtile_ptr = sC.subview(epi_stage_idx * epi_subtile_size)
                # Byte offset of this thread's slice within the stage. t2r_inst_repx is
                # an element count, so it needs the element width; at BF16 that is 2
                # bytes per element, so threads are 128 B apart.
                smem_subtile_thr_ptr = smem_subtile_ptr.subview(
                    tidx * t2r_inst_repx * c_dtype.width // 8
                )

                # Straight to BF16 -- no quantisation and no scale factors, so the whole
                # SFC round trip the NVFP4 sibling performs here is absent.
                swiglu_bf16 = swiglu_out.to(c_dtype)
                # Bitcast to Int8: store_swizzled's internal `.to(ptr.dtype)` works
                # against the Int8-typed sC buffer, and a pure reinterpret keeps the
                # byte count honest -- t2r_inst_repx BF16 lanes are 2*t2r_inst_repx bytes.
                swiglu_i8 = swiglu_bf16.bitcast(cutlass.Int8)
                smem_ptr = smem_subtile_thr_ptr.data_ptr()
                # Swizzle(3,4,3) is the 128-byte pattern, matched to the s128b C TMA
                # descriptor. The NVFP4 sibling uses Swizzle(1,4,3)/s32b because its 64
                # lanes are 32 B; these two constants must always be changed together.
                smem_ptr.store_swizzled(swiglu_i8, alignment=128, swizzle=cutlass.Swizzle(3, 4, 3))

                prims.fence_proxy(kind=prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)

                if warp_idx == epilog_warp_id[0]:
                    coord_n = coordc_n_out + real_pair_idx * epi_subtile_n
                    prims.cp_async_bulk_tensor_global_shared_cta(
                        tma_c_desc.get_ptr(),
                        smem_subtile_ptr,
                        (coord_n, coordc_m, 0),
                    )
                    prims.cp_async_bulk_commit_group()
                    prims.cp_async_bulk_wait_group(epi_stages - 1, read=True)

                prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)

            # Overlap: bias_empty signaled after loop (separate from acc_empty)
            if cutlass.const_expr(HAS_BIAS and mma_tiler_mnk[1] >= 256):
                if prims.elect_sync():
                    prims.mbarrier_arrive(bias_empty_mbar_ptr)

            acc_stage_idx = acc_stage_idx + 1
            if acc_stage_idx == acc_stages:
                acc_stage_idx = 0
                acc_full_phase_bit = acc_full_phase_bit ^ 1
            bias_full_phase_bit = bias_full_phase_bit ^ 1

            # Wait for producer to write next work_tile
            work_tile, sched_full_phase = sched_load_work_tile(
                s_work_tile, sched_full_mbar_ptr, sched_empty_mbar_ptr, sched_full_phase
            )

        if cutlass.const_expr(drain_epilogue):
            # The steady-state wait permits the newest TMA store to remain
            # pending. Drain its SMEM source read before the CTA can exit;
            # this is distinct from destination visibility, still supplied
            # by the caller's ordinary completion/PDL dependency contract.
            # Only the issuing warp owns these bulk groups. The following
            # epilogue barrier holds all peer writers until this read drains.
            if warp_idx == epilog_warp_id[0]:
                prims.cp_async_bulk_wait_group(0, read=True)

        # Deallocate TMEM after persistent loop — epilogue warps sync
        prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)
        if cutlass.const_expr(USE_2CTA):
            # Prove the peer has finished reading TMEM before the collective teardown.
            # Signal the peer on its own barrier, then wait for the peer's signal on ours.
            # Parity 0: the bar is fresh, and the peer's arrive flips it to phase 1.
            if warp_idx == 0:
                if prims.elect_sync():
                    prims.mbarrier_arrive(
                        prims.mapa(tmem_dealloc_mbar_ptr, 1 - cluster_rank),
                        scope=prims.MemScope.CTA,
                    )
                while not prims.mbarrier_try_wait_parity(
                    tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000
                ):
                    pass
                prims.tcgen05_dealloc(acc_tmem_ptr, num_tmem_alloc_cols, group=prims.CTAGroup.CTA_2)
        else:
            if warp_idx == 0:
                prims.tcgen05_dealloc(acc_tmem_ptr, num_tmem_alloc_cols, group=prims.CTAGroup.CTA_1)

    # Release the successor kernel as early as this CTA can.
    if cutlass.const_expr(USE_PDL):
        prims.griddepcontrol("launch_dependents")

    return


mtp_moe_gather_grouped_gemm_act_fusion_kernel.set_name_prefix(
    "glm52_moe_mtp_bf16_gg1_swiglu", remove_cutlass_symbol=True
)
