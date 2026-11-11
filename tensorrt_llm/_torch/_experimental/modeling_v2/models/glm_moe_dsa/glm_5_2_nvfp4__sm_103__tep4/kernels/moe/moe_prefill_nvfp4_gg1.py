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

"""GLM-5.2 prefill NVFP4 GG1: gather, grouped GEMM, SwiGLU and quantization.

Compact-M128 variant for SM103a, not a serving default.
Compile through moe_prefill_nvfp4_pair.compile_gg1: H=6144, intermediate=2048,
64 local experts, top-K=8, K256 stages and N128/N256. Each stage issues mixed
K96+K96+K64 MMA. Consumer roles derive the persistent schedule locally; a
separate warp loads activation scales. FP32 accumulation feeds SwiGLU and
NVFP4 output with E4M3 scales (one scale per 16 values).

Inputs use packed FP4x2 Torch views; input scales have the actual, unpadded T
extent. Routing supplies expert IDs, token mapping and valid-row limits for
128-row expert-padded tiles. Output scales zero padding inside active tiles.
The caller owns buffers, current routing metadata and stream dependencies.
No TRT-LLM frontend, routing, zeroing or shared-expert work is included here.

GG1 uses the tested evict-last weight policy; GG2 uses evict-first. PDL is fixed
at module import by TRTLLM_ENABLE_PDL and must match the explicit compile API.
SwiGLU uses the native FP32 multiply association, which the numerical tests
cover. Other compile-time branches in this file are not exposed by the compile
API.
See moe_prefill_nvfp4_pair.py for the callable ABI.
"""

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass._mlir.dialects import vector as _vector_dialect
from cutlass._mlir.extras import types as T_  # for T_.f32() in reduction
from cutlass._mlir_helpers.dialect_proxy import DialectAutoConvertProxy
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
from .static_persistent_tile_scheduler import PersistentTileSchedulerParams, WorkTileInfo

_vector = DialectAutoConvertProxy(_vector_dialect)

# ---------------------------------------------------------------------------
# GLM-5.2 data types -- deliberately NOT imported from moe_utils.
#
# moe_utils defines the GPT-OSS configuration (A=FP8E4M3, SF=E8M0 MX vec=32).
# GLM-5.2 is NVFP4 end to end: both operands 4-bit with an FP8E4M3 scale every
# 16 elements. Overriding here keeps the shared moe_utils untouched so the
# GPT-OSS kernels next to this file keep working.
#
#   moe_utils (GPT-OSS)                 this file (GLM-5.2)
#   a_dtype     = Float8E4M3FN     ->   Float4E2M1FN     (8 bit -> 4 bit)
#   b_dtype     = Float4E2M1FN     ->   Float4E2M1FN     (unchanged)
#   sf_dtype    = Float8E8M0FNU    ->   Float8E4M3FN     (MX -> NVFP4)
#   sf_vec_size = 32               ->   16
# ---------------------------------------------------------------------------

# Output C: NVFP4 + generated SFC, matching the production cute-DSL kernel whose
# FC1 output feeds FC2 as NVFP4. The GPT-OSS reference emitted FP8E4M3 here.
c_dtype = cutlass.Float4E2M1FN

# SFC quantization: normalize amax to the max representable value of c_dtype.
# GPT-OSS reference used 448.0 (FP8E4M3 max); NVFP4 E2M1 tops out at 6.0.
c_dtype_amax = 6.0

# GLM-5.2 SwiGLU is the plain form -- out = (alpha*up) * silu(alpha*gate) -- with
# no +1 bias and no clamping, unlike GPT-OSS's
# (up+1)*gate*sigmoid(gate*swiglu_alpha) with gate/up clamped to +/-swiglu_limit.
# `alpha` here is the NVFP4 global dequant scale (fc1_global_scale), a different
# quantity from GPT-OSS's swiglu_alpha sigmoid steepness -- do not conflate them.
# swiglu_alpha / swiglu_limit are intentionally NOT imported.

# Kernel configuration
smem_capacity = cutlass.memory.get_smem_capacity_in_bytes("sm_100")
num_mbar_bytes = 1024
num_tmem_alloc_cols = 512
epi_stages = 2  # Number of pipeline stages for epilog TMA stores

# Fixed per-module policy. Do not inherit a
# shared environment policy: the companion GG2 deliberately uses evict_first.
L2_POLICY_B = "evict_last"

# Programmatic dependent launch, matching what TRT-LLM sets for the production kernels.
# Defaults on so a model run overlaps the same way production does; the standalone
# benchmarks are unaffected because a single kernel has no predecessor to overlap with.
USE_PDL = bool(_TRTLLM_ENABLE_PDL)
# `no_allocate` is deliberately absent: it is a direct `.L2::no_allocate` qualifier on
# ld/st, not a `createpolicy.fractional` priority, and libNVVM rejects it there. Catching
# it here gives a readable message instead of "NVVM backend compilation failed".
assert L2_POLICY_B in ("", "evict_first", "evict_last", "evict_normal"), (
    f"L2_POLICY_B={L2_POLICY_B!r} is not a valid createpolicy.fractional priority"
)

# Warp specialization: 12-warp with in-kernel cp.async gather + 2 idle warps
# Idle warps donate registers to epilogue warps via setmaxregister.
epilog_warp_id = (0, 1, 2, 3)
ldgsts_warp_id = (4, 5, 6, 7)
mma_warp_id = 8
tma_warp_id = 9
tile_sched_warp_id = 10  # scheduler producer warp
idle_warp_id = 11
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


@cute.jit
def local_work_tile(
    linear_idx,
    actual_tiles_m,
    n_tiles,
    major_divisor,
    raster_along_m: cutlass.Constexpr[bool],
):
    """Decode the existing unswizzled cluster1 order independently per role.

    A/MMA/SFA/TMA/epilogue retain their original stage/arrival/read-completion
    barriers. Only scheduler metadata handshakes are removed. Invalid sentinel
    coordinates are not consumed; the empty-M divisor is clamped to one.
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
def host_function(
    b_tensor: cute.Tensor,
    sfb_tensor: cute.Tensor,
    c_tensor: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    # Canonical token-order inputs (pointers extracted inside for indirect gather)
    a: cute.Tensor,
    a_scales: cute.Tensor,
    sfc_tensor: cute.Tensor,
    # Device pointers, not values: both scalars are produced on the GPU by the kernel
    # that runs before this one, so the host cannot read them. Taking them by value
    # forced the caller to dereference device memory on the host -- a host-side SIGSEGV
    # inside the compiled `host_function` (vmovss from a device address). Tests that
    # pass Python constants do not hit this.
    norm_const_ptr: cutlass.Int64,
    alpha_ptr: cutlass.Int64,
    token_mapping: cute.Tensor,
    mn_limit: cute.Tensor,
    problem_k: cutlass.Int32,
    topK: cutlass.Constexpr[int],
    num_tiles_m_ptr: cutlass.Int64,
    max_tiles_m: cutlass.Int32,
    rest_n_sfc: cutlass.Int32,
    mma_tiler_n: cutlass.Constexpr[int],
    mma_tiler_k: cutlass.Constexpr[int],
    raster_along_m: cutlass.Constexpr[bool],
    # SM count, supplied by the caller rather than queried here: a
    # `torch.cuda.get_device_properties` call inside this function runs during
    # `cute.compile`, which would make compiling without a GPU impossible. The caller
    # passes the device SM count.
    num_sms: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
    pack_sfc_store: cutlass.Constexpr[bool] = False,
    zero_fill_gather: cutlass.Constexpr[bool] = False,
    native_gather: cutlass.Constexpr[bool] = False,
    typed_gather_ptrs: cutlass.Constexpr[bool] = False,
) -> None:
    """Host-side JIT for gather + grouped GEMM + SwiGLU + SFC.

    Sets up TMA descriptors for all input tensors and launches the GPU kernel.
    This function runs on the host and is JIT-compiled by the CUTE DSL.

    Args:
        b_tensor: Input matrix B, shape (K/2, N, L), Float4E2M1FNx2.
            L = number of experts.
        sfb_tensor: Scale factors for B, shape (512, K//128, N//128, L), E4M3.
        c_tensor: Output matrix C, shape (N_out, valid_m, 1), Float4E2M1FN (quantized by SFC).
            N_out = N/2 due to SwiGLU fusion (up+gate pairs → single output).
        tile_idx_to_expert_idx: Int32 tensor of shape (num_tiles_m,) mapping
            each M-tile index to the expert index (L coordinate for B/SFB).
        a: Canonical A input in token order, shape (K/2, tokens, 1),
            NVFP4x2 packed-K-major.
        a_scales: Canonical linear A scales in token order, shape
            (padded_tokens, K/16, 1), E4M3.
        sfc_tensor: SFC output tensor, shape (1, rest_m, rest_n, 512), uint8.
            Pointer extracted inside via .iterator.toint().
        norm_const_ptr: GPU pointer to the Float32 normalization constant for SFC.
        alpha_ptr: GPU pointer to per-expert Float32 NVFP4 global dequant scales
            applied to the accumulator
            (production `fc1_global_scale`). NOT the GPT-OSS `swiglu_alpha`
            sigmoid-steepness parameter -- different quantity, same name.
        token_mapping: Int32 token_id_mapping tensor.
        mn_limit: Int32 per-tile row validity tensor.
        problem_k: The K dimension of the GEMM problem.
        topK: Number of experts each token is routed to.
        num_tiles_m_ptr: GPU pointer to the number of active M-tiles.
        max_tiles_m: Upper bound used to construct the persistent tile grid.
        rest_n_sfc: Number of scale-factor atoms along the output N dimension.
        mma_tiler_n: N dimension of the MMA tile (64, 128, or 256).
        mma_tiler_k: K dimension of the MMA tile (128 or 256).
        raster_along_m: Whether the persistent scheduler rasterizes along M.
        num_sms: Number of SMs used to size the persistent grid.
        stream: CUDA stream for asynchronous execution.
        pack_sfc_store: Experimental prefill schedule using aligned 32-bit SFC stores.
        zero_fill_gather: Experimental prefill schedule using zero-fill rather than
            outer predication for invalid rows in the A/SFA gather.

    Note:
        - Grid is (num_tiles_m, ceil(N/tiler_N), 1) — L=1 since A/C are stacked.
        - TMA warp uses expert_idx to select B/SFB slice per tile.
    """

    # MMA tile configuration for 1CTA
    mma_tiler_mnk = [128, mma_tiler_n, mma_tiler_k]  # Tile dimensions (M, N, K)
    # Retain the original K64 storage/SF atom accounting. Issued instructions
    # below are K96+K96+K64 per K256 stage, not four K64 instructions. Passing
    # 96 to the inherited storage helper would truncate its instK//64 ratio;
    # the mixed instruction loop explicitly selects every SF atom and ID.
    mma_inst_mnk = [128, mma_tiler_n, 64]
    # Epilogue subtile width == the up/gate interleave granularity, because the
    # epilogue pairs consecutive subtiles (even=up, odd=gate) into one SwiGLU output.
    # GPT-OSS interleaves every 32 columns; GLM-5.2's w3_w1_weight interleaves every 64
    #   [up_0:64, gate_64:128, up_128:192, gate_192:256, ...]
    # (production blockscaled_contiguous_gather_grouped_gemm_act_fusion.py L86/L624).
    # Getting this wrong yields numerically wrong output with correct shapes.
    epi_subtile_n = min(64, mma_tiler_n)

    # Extract raw pointers from cute.Tensor for kernel consumption
    a_ptr = a.iterator.toint()
    a_scales_ptr = a_scales.iterator.toint()
    token_mapping_ptr = token_mapping.iterator.toint()
    mn_limit_ptr = mn_limit.iterator.toint()
    sfc_ptr = sfc_tensor.iterator.toint()
    # A is already expressed in packed FP4x2 elements, one byte per shape unit.
    # Its leading K/2 extent is therefore also the byte stride between tokens.
    a_row_stride_bytes = a.shape[0] * fp4x2_dtype.width // 8

    # NO A TMA descriptor — A loaded by cp.async warps via in-kernel gather

    # B is a packed FP4x2 tensor, so the descriptor and its K box use packed-pair
    # coordinates. TMA copies the same packed bytes consumed by the scalar-FP4 MMA.
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

    # C TMA descriptor. s32b swizzle (Swizzle<1,4,3>) matches what
    # sm100_utils.make_smem_layout_epi derives for (Float4E2M1FN, ROW_MAJOR, (128,64)).
    #
    # tma_format must be given explicitly. Left to inference, Float4E2M1FN resolves to the
    # padded B4X16_P64 (align16b) encoding, which _PADDED_SUBBYTE_BOX_BYTES pins to a box of
    # exactly 64B -- unreachable here, since box_dims[0] is epi_subtile_n = 64 lanes x 4 bit
    # = 32B. B4X16 (align8b) carries no such constraint, and per the (dtype, format) table in
    # tensor_map.py it is the only format legal for a *scalar* Float4E2M1FN element type
    # (B4X16_P64 is only paired with the packed Float4E2M1FNx2).
    #
    # epi_subtile_n is not free to grow: it is the up/gate interleave granularity of the B
    # weights (module docstring), fixed at 64 by the production layout
    # [up_0:64, gate_64:128, ...].
    #
    # The epilogue matches this descriptor. It quantises straight to NVFP4 with
    # `swiglu_out.to(c_dtype)` (c_dtype is Float4E2M1FN) and only then bitcasts to Int8,
    # which is a byte container for the packed 4-bit values, not FP8 data. The sizes are on
    # the same footing: num_bytes_c_smem and epi_subtile_size both take the width from
    # c_dtype rather than hardcoding one byte per element. The C path is 4-bit throughout.
    tma_c_desc = cuda.create_tensor_map_tiled_from_view(
        c_tensor,
        box_dims=(epi_subtile_n, mma_tiler_mnk[0], 1),
        stride_order=(0, 1, 2),
        swizzle=cuda.TensorMapSwizzle.s32b,
        tma_format=cuda.TensorMapDataFormat.B4X16,
    )

    # Compute SF atom counts per tile and TMEM columns per MMA kblock
    tile_sf, num_tmem_cols_per_kblock_sfa, num_tmem_cols_per_kblock_sfb = compute_sf_rest(
        mma_tiler_mnk, mma_inst_mnk
    )
    rest_k_sf_per_ktile = tile_sf["k"]
    rest_m_sf_per_ktile = tile_sf["m"]
    rest_n_sf_per_ktile = tile_sf["n"]

    # SFB tensor has flat layout (512, rest_k_sf, rest_n_sf, L) as 8-bit
    # Recast to 16-bit (256 elements) for TMA descriptor
    sfb_tensor_fp16 = cute.recast_tensor(sfb_tensor, cutlass.Uint16)

    # SFB TMA descriptor
    # Built from explicit dims rather than `..._from_view`, because the view helper
    # auto-detects the global dims and strides from the tensor and needs them known at
    # trace time. That holds when the tensor comes from a torch array but not when it
    # is built over raw pointers with runtime extents, which is
    # what the TRT-LLM op passes -- there the shape is `Int32(?)` and the helper
    # segfaults rather than diagnosing it.
    #
    # `create_tensor_map_tiled` takes `global_dims: Int32 | int` and
    # `global_strides: Int64 | int`, so the shape may be dynamic; `box_dims` stays
    # `Int8 | int`, i.e. compile-time, which is what the descriptor actually requires.
    #
    # TMA is column-major: global_dims[0] is the contiguous axis, and global_strides
    # covers dims 1.. only, in 16-byte units. This tensor is already fastest-dim-first
    # (512-byte SF atoms, recast to 256 x Uint16), so no reordering is needed.
    _sfb_rest_k = sfb_tensor_fp16.shape[1]
    _sfb_rest_n = sfb_tensor_fp16.shape[2]
    _sfb_l = sfb_tensor_fp16.shape[3]
    _u16 = cutlass.Uint16.width // 8  # 2 bytes
    _sfb_addr = sfb_tensor_fp16.iterator.toint()
    tma_sfb_desc = cuda.create_tensor_map_tiled(
        _sfb_addr,
        cutlass.Uint16,
        global_dims=[num_elts_atom_sf_fp16, _sfb_rest_k, _sfb_rest_n, _sfb_l],
        global_strides=[
            num_elts_atom_sf_fp16 * _u16 // 16,
            num_elts_atom_sf_fp16 * _u16 * _sfb_rest_k // 16,
            num_elts_atom_sf_fp16 * _u16 * _sfb_rest_k * _sfb_rest_n // 16,
        ],
        box_dims=(num_elts_atom_sf_fp16, rest_k_sf_per_ktile, rest_n_sf_per_ktile, 1),
        swizzle=cuda.TensorMapSwizzle.none,
    )

    # SMEM footprint. GLM is NVFP4 end to end, so A and B are both packed at
    # 4 bits per scalar lane. The FP4x2 TMA descriptor transfers exactly these bytes.
    num_bytes_a = mma_tiler_mnk[0] * mma_tiler_mnk[2] * a_dtype.width // 8
    num_bytes_b = mma_tiler_mnk[1] * mma_tiler_mnk[2] * b_dtype.width // 8
    num_bytes_sfa = (
        num_elts_atom_sf_e8 * rest_k_sf_per_ktile * rest_m_sf_per_ktile * sf_dtype.width // 8
    )
    num_bytes_sfb = (
        num_elts_atom_sf_e8 * rest_k_sf_per_ktile * rest_n_sf_per_ktile * sf_dtype.width // 8
    )
    # Reserve SMEM for the dedicated epilogue C buffer.
    # c_dtype is NVFP4 (4 bit) here, not the reference's FP8 (8 bit). Width is taken
    # from the dtype rather than hardcoded. Note the two changes cancel: the subtile
    # doubled 32->64 while the element halved 8->4 bit, so this budget is unchanged.
    num_bytes_c_smem = epi_subtile_n * mma_tiler_mnk[0] * epi_stages * c_dtype.width // 8
    smem_for_ab = smem_capacity - num_mbar_bytes - num_bytes_c_smem
    ab_bytes_per_stage = num_bytes_a + num_bytes_b + num_bytes_sfa + num_bytes_sfb
    num_ab_stage = smem_for_ab // ab_bytes_per_stage

    # Convert helper tensors to raw GPU pointers for kernel consumption
    tile_idx_ptr = tile_idx_to_expert_idx.iterator.toint()

    # Persistent tile scheduling: tiles are distributed across a smaller grid.
    # Each CTA loops over multiple tiles via StaticPersistentTileScheduler.
    n_tiles = cute.ceil_div(b_tensor.shape[1], mma_tiler_mnk[1])
    tile_grid = (max_tiles_m, n_tiles, 1)
    cluster_shape = (1, 1, 1)
    tile_sched_params = PersistentTileSchedulerParams(
        tile_grid, cluster_shape, raster_along_m=raster_along_m
    )
    grid = DeviceBoundPersistentTileScheduler.get_grid_shape(tile_sched_params, num_sms)

    # TMA loads B + SFB only (A and SFA loaded by cp.async gather)
    num_tma_load_bytes = num_bytes_b + num_bytes_sfb

    # Launch the kernel
    moe_gather_grouped_gemm_act_fusion_kernel(
        tma_b_desc,
        tma_sfb_desc,
        tma_c_desc,
        tile_idx_ptr,
        a_ptr,
        a_scales_ptr,
        sfc_ptr,
        norm_const_ptr,
        alpha_ptr,
        token_mapping_ptr,
        mn_limit_ptr,
        num_tiles_m_ptr,
        problem_k,
        a_row_stride_bytes,
        topK,
        rest_n_sfc,
        num_tma_load_bytes,
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
        tile_sched_params,
        pack_sfc_store,
        zero_fill_gather,
        native_gather,
        typed_gather_ptrs,
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(1, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
        use_pdl=USE_PDL,
    )
    return


# GPU device kernel.
#
# The name is load-bearing, not decorative. CuTe DSL mangles the cubin symbol as
# "kernel_cutlass_<python function name>_<repr of each argument>" and truncates at 180
# characters; the module and file names never appear. Both MoE kernels lead with
# `GridConstant[cuda.TensorMap]` arguments whose reprs are identical here and in
# moe_grouped_gemm_finalize_fusion.py (NVFP4x2, 128B swizzle, BYTE tensor maps),
# and the cut lands 25 characters into the second descriptor -- inside the shared
# "TensorMapswizzleTensorMap" boilerplate, one character before the two would diverge.
# Named `kernel`, as the DSL examples conventionally are, the two emit the *same* symbol
# and every profiler merges them into one row. Keep the names distinct.
@cute.kernel
def moe_gather_grouped_gemm_act_fusion_kernel(
    tma_b_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfb_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_c_desc: cutlass.GridConstant[cuda.TensorMap],
    tile_idx_ptr: cutlass.Int64,
    a_ptr: cutlass.Int64,
    a_scales_ptr: cutlass.Int64,
    sfc_ptr: cutlass.Int64,
    norm_const_ptr: cutlass.Int64,
    alpha_ptr: cutlass.Int64,
    token_mapping_ptr: cutlass.Int64,
    mn_limit_ptr: cutlass.Int64,
    num_tiles_m_ptr: cutlass.Int64,
    problem_k: cutlass.Int32,
    a_row_stride_bytes: cutlass.Int32,
    topK: cutlass.Constexpr[int],
    rest_n_sfc: cutlass.Int32,
    num_tma_load_bytes: cutlass.Constexpr[int],
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
    tile_sched_params: PersistentTileSchedulerParams,
    pack_sfc_store: cutlass.Constexpr[bool] = False,
    zero_fill_gather: cutlass.Constexpr[bool] = False,
    native_gather: cutlass.Constexpr[bool] = False,
    typed_gather_ptrs: cutlass.Constexpr[bool] = False,
) -> None:
    """GPU device kernel for contiguous grouped mixed precision block-scaled GEMM.

    Extends the dense block-scaled GEMM kernel with contiguous grouped GEMM support:
    - tile_idx_to_expert_idx maps each M-tile to an expert index
    - TMA warp uses expert_idx to index B/SFB in the L dimension

    Args:
        tma_b_desc: TMA descriptor for matrix B (FP4, K-major, L=num_experts).
        tma_sfb_desc: TMA descriptor for B's scale factors (L=num_experts).
        tma_c_desc: TMA descriptor for output C (NVFP4, L=1).
        tile_idx_ptr: GPU pointer to Int32 array mapping M-tile → expert index.
        a_ptr: Raw GPU pointer to canonical A in token order (NVFP4x2).
        a_scales_ptr: Raw GPU pointer to canonical linear A scales in token order.
        sfc_ptr: Raw GPU pointer to E4M3 SFC output tensor.
        norm_const: Float32 normalization constant for SFC.
        token_mapping_ptr: Raw GPU pointer to Int32 token_id_mapping array.
        mn_limit_ptr: Raw GPU pointer to Int32 per-tile row validity limits.
        problem_k: K dimension of the GEMM.
        a_row_stride_bytes: Stride in bytes between consecutive tokens in A.
        topK: Number of experts each token is routed to.
        num_tiles_m: Total number of M-tiles (= len(tile_idx_to_expert_idx)).
        num_tma_load_bytes: Total bytes per TMA load (B + SFB).
        mma_tiler_mnk: Tile dimensions [M, N, K].
        mma_inst_mnk: MMA instruction dimensions [M, N, K].
        epi_subtile_n: Epilog subtile N dimension.
        num_bytes_*: Byte sizes for SMEM allocation.
        num_ab_stage: Pipeline depth.
        rest_*_sf_per_ktile: SF atom counts per tile dimension.
        num_tmem_cols_per_kblock_*: TMEM columns per kblock for scale factors.

    Kernel Architecture (12-warp with cp.async gather + register reconfig):
        - Block: 384 threads (12 warps)
        - Warp roles: warp 0-3 (epilogue), warp 4-7 (cp.async gather A/SFA),
          warp 8 (MMA+S2T), warp 9 (TMA B/SFB), warp 10-11 (idle, donate regs)
        - Register reconfig: epilogue→232, gather→80, mma→48, tma→40, idle→24
        - tiler_n=256: overlap path with early acc release and pair reversal
    """
    #
    # K-tile config — computed inline from mma_tiler_mnk at compile time
    #
    _kc = compute_ktile_config(mma_tiler_mnk[2])
    smem_row_bytes = _kc["smem_row_bytes"]
    mma_layout = _kc["mma_layout"]
    mma_stride = _kc["mma_stride"]
    mma_leading = _kc["mma_leading"]
    swizzle_mask = _kc["swizzle_mask"]
    swizzle_row_shift = 0  # This family fixes K256 / SW128.

    #
    # Thread / block identification
    #
    warp_idx = cute.arch.warp_idx()
    tidx, _, _ = cute.arch.thread_idx()
    k = problem_k

    #
    # Shared memory allocation (mbarriers + data tiles)
    #
    # acc_stages=2 enables MMA/epilogue overlap: MMA writes tile T+1 while
    # epilogue reads tile T. Requires 2x TMEM for acc+SFA+SFB.
    # tiler_n=256 doesn't fit: 2*(256+SFA+SFB) > 512 TMEM cols.
    acc_stages = 1 if mma_tiler_mnk[1] >= 256 else 2
    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    num_tiles_m_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    # Warp-specialized scheduler: producer→consumer work_tile via SMEM
    sched_full_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    sched_empty_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    # TMEM address buffer - will hold the allocated TMEM address
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    s_num_tiles_m = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # Work tile SMEM: [is_valid_i32, tile_coord_m, tile_coord_n, tile_coord_l] — 4×Int32 for one 128-bit ld.shared
    s_work_tile = cutlass.Array(  # noqa: F841 -- keeps the SMEM layout unchanged.
        cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16
    )
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

    # Persistent kernel: sC is not aliased to sA/sB because the mainloop of tile
    # T+1 overlaps the epilogue of tile T.
    num_bytes_c_smem = epi_subtile_n * mma_tiler_mnk[0] * epi_stages * c_dtype.width // 8
    sC = cutlass.Array(
        cutlass.Int8, num_bytes_c_smem, space=cutlass.AddressSpace.smem, alignment=1024
    )
    k_tile_cnt = cute.ceil_div(k, mma_tiler_mnk[2])

    # TMEM allocation synchronization barrier
    # MMA warp (8) and epilogue warps (0,1,2,3) = 5 warps * 32 threads = 160
    tmem_bar_id = 1
    tmem_bar_threads = 32 * len((mma_warp_id, *epilog_warp_id))

    # Epilog synchronization barrier
    epilog_sync_bar_id = 2
    threads_in_epilogue = 32 * len(epilog_warp_id)

    # Bias-full mbarrier: cp.async warps (producers) → epilogue warps (consumers)
    threads_ldgsts = 32 * len(ldgsts_warp_id)  # noqa: F841 -- inherited role geometry.

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
                # 161 producers: TMA (1), A gather (128), independent SFA gather (32)
                prims.mbarrier_init(ab_full_mbar_ptr.subview(i), 161)
                prims.mbarrier_init(ab_empty_mbar_ptr.subview(i), 1)
            for i in range(acc_stages):
                prims.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
                # 4 epilogue warps signal acc_empty via elect_sync (4 arrivals)
                prims.mbarrier_init(acc_empty_mbar_ptr.subview(i), 4)
            # Warp-specialized scheduler: producer (warp 10) → consumers (warps 0-9)
            # sched_full: 1 arrival (producer via elect_sync)
            prims.mbarrier_init(sched_full_mbar_ptr, 1)
            # sched_empty: A/MMA/TMA/epilogue plus SFA warp = 352 consumers
            prims.mbarrier_init(sched_empty_mbar_ptr, 32 * 11)

    # This fence orders the mbar init operations with respect to the barrier below
    prims.fence_mbarrier_init()
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

    # The preceding routing write is visible only after PDL WAIT and this
    # asynchronous scalar load. All local scheduler values dominate role scopes.
    # Physical cluster is fixed1; only valid coordinates are ever dereferenced.
    while not prims.mbarrier_try_wait_parity(num_tiles_m_mbar_ptr, 0):
        pass
    local_actual_tiles_m = cute.arch.make_warp_uniform(s_num_tiles_m.load())
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
    token_map_arr = cutlass.Array(
        cutlass.inttoptr(token_mapping_ptr, mem_space=1, dtype=cutlass.Int32),
        dtype=cutlass.Int32,
    )
    mn_limit_arr = cutlass.Array(
        cutlass.inttoptr(mn_limit_ptr, mem_space=1, dtype=cutlass.Int32),
        dtype=cutlass.Int32,
    )
    # Read the epilogue scalars on the device rather than on the host. `norm_const` is
    # a genuine 1-element tensor; `alpha` is per-expert (production computes it as
    # a_global_sf * b_global_sf, with b_global_sf reduced per expert), so it is indexed
    # by expert_idx in the epilogue rather than read here.
    norm_const = cutlass.Array(
        cutlass.inttoptr(norm_const_ptr, mem_space=1, dtype=cutlass.Float32),
        dtype=cutlass.Float32,
    )[0]
    alpha_arr = cutlass.Array(
        cutlass.inttoptr(alpha_ptr, mem_space=1, dtype=cutlass.Float32),
        dtype=cutlass.Float32,
    )

    # =========================================================================
    # Warp 11: independent SFA gather, retaining 80 registers.
    # =========================================================================
    if warp_idx == idle_warp_id:
        prims.setmaxregister(80, prims.SetMaxRegisterAction.DECREASE)
        # One warp visits the same four row groups previously owned by four
        # A-loader warps. Every source/destination pair is unchanged; only the
        # issue owner and completion arrival count change.
        lane = tidx % 32
        sfa_offsets = cutlass.Array(cutlass.Int32, 4)
        sfa_row_bytes = k // sf_vec_size
        sfa_rest_k_total = k // (sf_vec_size * num_k_per_sf_atom)
        work_linear_idx = cutlass.Int32(cute.arch.block_idx()[2])
        work_tile = local_work_tile(
            work_linear_idx,
            local_actual_tiles_m,
            local_n_tiles,
            local_major_divisor,
            tile_sched_params.raster_along_m,
        )
        sfa_stage_idx = 0
        sfa_empty_phase = 1
        while work_tile.is_valid_tile:
            coord_m = work_tile.tile_idx[0] * mma_tiler_mnk[0]
            limit = mn_limit_arr.subview(work_tile.tile_idx[0]).load()
            for row_group in cutlass.range_constexpr(4):
                local_row = 8 * row_group + 32 * (lane // 8) + lane % 8
                source_offset = cutlass.Int32(0)
                if coord_m + local_row < limit:
                    expanded = token_map_arr.subview(coord_m + local_row).load()
                    source_offset = (expanded // topK) * sfa_row_bytes
                sfa_offsets.subview(row_group).store(source_offset)
            for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                stage = sfa_stage_idx
                phase = sfa_empty_phase
                sfa_stage_idx += 1
                if sfa_stage_idx == num_ab_stage:
                    sfa_stage_idx = 0
                    sfa_empty_phase ^= 1
                while not prims.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(stage), phase):
                    pass
                for row_group in cutlass.range_constexpr(4):
                    local_row = 8 * row_group + 32 * (lane // 8) + lane % 8
                    atom_offset = (local_row % 32) * 16 + (local_row // 32) * 4
                    source_offset = sfa_offsets.subview(row_group).load()
                    for atom in cutlass.range_constexpr(rest_k_sf_per_ktile):
                        issue = atom ^ ((lane // 8) % rest_k_sf_per_ktile)
                        scale_k = k_tile * rest_k_sf_per_ktile + issue
                        source = cutlass.inttoptr(
                            a_scales_ptr + source_offset + scale_k * num_k_per_sf_atom,
                            mem_space=1,
                            dtype=sf_dtype,
                        )
                        target = sSFA.subview(
                            stage * num_bytes_sfa + issue * 512 + atom_offset
                        ).data_ptr()
                        copy_bytes = cutlass.Int32(scale_k < sfa_rest_k_total) * 4
                        prims.inline_ptx_hl(
                            "cp.async.ca.shared.global [{$r0}], [{$r1}], 4, {$r2};",
                            read_only_args=[target, source, copy_bytes],
                            pred=coord_m + local_row < limit,
                        )
                # No speculative completion: all 16 copies per lane must finish
                # before this warp's arrival permits MMA to consume the stage.
                prims.cp_async_mbarrier_arrive(ab_full_mbar_ptr.subview(stage), noinc=True)
            work_linear_idx += local_stride
            work_tile = local_work_tile(
                work_linear_idx,
                local_actual_tiles_m,
                local_n_tiles,
                local_major_divisor,
                tile_sched_params.raster_along_m,
            )

    # =========================================================================
    # Warp 10: retains its register donation, with no scheduler producer work.
    # =========================================================================
    if warp_idx == tile_sched_warp_id:
        # Preserve register donation; no work-tile handshake remains.
        prims.setmaxregister(num_regs_sched, prims.SetMaxRegisterAction.DECREASE)

    # =========================================================================
    # cp.async warps (warps 4-7): gather canonical token-order A → SMEM
    # 128 threads × 8 cp.async.ca.shared.global (16B each) = 16384 bytes
    # = 128 rows × 256 K (NVFP4 = 4 bit/element)
    #
    # Thread mapping (128 threads across 4 warps):
    #   row = (tidx_in_warpgroup // 8) + i * 16    (i=0..7, 8 rows/thread)
    #   k_offset = (tidx_in_warpgroup % 8) * 16    (16 bytes per cp.async)
    # =========================================================================
    if warp_idx >= ldgsts_warp_id[0] and warp_idx <= ldgsts_warp_id[-1]:
        prims.setmaxregister(num_regs_ldgsts, prims.SetMaxRegisterAction.DECREASE)

        tidx_in_wg = tidx % 128
        if cutlass.const_expr(typed_gather_ptrs):
            # Preserve pointer + byte-offset IR instead of reconstructing a
            # pointer from a lane-varying integer address for every copy.
            a_gather_base = cutlass.Array(
                cutlass.inttoptr(a_ptr, mem_space=1, dtype=cutlass.Int8),
                dtype=cutlass.Int8,
            )
            # Unused; kept so the generated code stays unchanged.
            sfa_gather_base = cutlass.Array(  # noqa: F841
                cutlass.inttoptr(a_scales_ptr, mem_space=1, dtype=cutlass.Int8),
                dtype=cutlass.Int8,
            )
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
        sfa_local_gmem = cutlass.Array(cutlass.Int32, 1)
        sfa_local_smem = cutlass.Array(cutlass.Int32, 1)
        for i in range(loads_per_thread):
            local_row = (tidx_in_wg // threads_per_token) + i * rows_per_load
            # SMEM address: XOR swizzle (k_chunk ^ row) prevents bank conflicts
            swizzled_k = (
                k_idx ^ ((local_row >> swizzle_row_shift) & swizzle_mask)
            ) * bytes_per_load
            smem_off = local_row * smem_row_bytes + swizzled_k
            a_row_smem_off.subview(i).store(smem_off)

        # SFA thread mapping: rearrange 128 threads into SF atom (m0, m1) layout
        sfa_tidx = 8 * (tidx_in_wg // 32) + 32 * ((tidx_in_wg % 32) // 8) + (tidx_in_wg % 8)
        sfa_rest_k_total = k // (sf_vec_size * num_k_per_sf_atom)
        # Bytes per row of the linear SFA buffer: one E4M3 scale per sf_vec_size elements.
        sfa_row_bytes = k // sf_vec_size
        sfa_m0_smem = sfa_tidx % 32
        sfa_m1_smem = sfa_tidx // 32
        sfa_local_smem.subview(0).store(sfa_m0_smem * 16 + sfa_m1_smem * 4)

        # Unused; kept so the generated code stays unchanged.
        ldg_tidx = (warp_idx - ldgsts_warp_id[0]) * 32 + (tidx % 32)  # noqa: F841

        # ---- Persistent loop: cp.async warps ----
        # Initialize the deterministic per-role work index.
        work_linear_idx = cutlass.Int32(cute.arch.block_idx()[2])
        work_tile = local_work_tile(
            work_linear_idx,
            local_actual_tiles_m,
            local_n_tiles,
            local_major_divisor,
            tile_sched_params.raster_along_m,
        )
        # AB pipeline producer state: stage index carries across tiles.
        ab_stage_idx = 0
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            tile_coord_m = cur_tile_coord[0]
            tile_coord_n = cur_tile_coord[1]
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
                    source_token = token_id // topK
                    a_row_token_off.subview(i).store(source_token * a_row_stride_bytes)

            # SFA gather addresses for this tile
            sfa_abs_row = coord_m + sfa_tidx
            sfa_local_gmem.subview(0).store(0)
            if sfa_abs_row < mn_limit:
                sfa_token_id = token_map_arr.subview(sfa_abs_row).load()
                a_scale_token = sfa_token_id // topK
                # A scales arrive in LINEAR layout: row = source token,
                # `scale_k` bytes per row. `fp4_quantize(..., is_sf_swizzled_layout=False)`
                # produces it that way deliberately -- the unswizzled form is what the
                # alltoall/allgather path requires -- and the production kernel reads it
                # linearly too. The reference this port came from used the swizzled
                # 512-byte atom layout, which is why the addressing here used to be
                # atom-based; against a linear buffer that walks off the end.
                #
                # Only the *source* address changes. The SMEM destination stays in atom
                # order, so the linear->atom transpose still happens implicitly in the
                # addressing, with no extra pass and the same 4-byte copy -- exactly the
                # structure production uses.
                sfa_local_gmem.subview(0).store(a_scale_token * sfa_row_bytes)

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

                # Bytes, not elements. `a_ptr`, `token_byte_off` and
                # `k_thread_offset` are all byte quantities, so a K offset in elements
                # advanced A by 2x for 4-bit data -- and by exactly 0 on the first
                # k_tile, which is why every single-K-tile configuration passed.
                k_base_bytes = k_tile * mma_tiler_mnk[2] * a_dtype.width // 8
                sA_stage_base = sA.subview(current_ab_stage * num_bytes_a)

                # Gather: combine source-token offset (from routing) with K-tile offset.
                for i in range(loads_per_thread):
                    local_row = (tidx_in_wg // threads_per_token) + i * rows_per_load
                    token_byte_off = a_row_token_off.subview(i).load()
                    smem_off = a_row_smem_off.subview(i).load()
                    smem_ir = sA_stage_base.subview(smem_off).data_ptr()
                    if cutlass.const_expr(typed_gather_ptrs):
                        gmem_ir = a_gather_base.subview(
                            token_byte_off + k_base_bytes + k_thread_offset
                        ).data_ptr()
                    else:
                        gmem_ir = a_ptr + token_byte_off + k_base_bytes + k_thread_offset
                        gmem_ir = cutlass.inttoptr(gmem_ir, mem_space=1, dtype=a_dtype)
                    is_valid_m = coord_m + local_row < mn_limit
                    if cutlass.const_expr(native_gather):
                        # Native async-copy IR under the same M predicate lets
                        # lowering share memory descriptors without writing
                        # otherwise-unused padding rows.
                        if is_valid_m:
                            prims.cp_async_shared_global(
                                smem_ir,
                                gmem_ir,
                                size=bytes_per_load,
                                modifier="cg",
                                cp_size=cutlass.Int32(
                                    k_base_bytes + k_thread_offset < k * a_dtype.width // 8
                                )
                                * bytes_per_load,
                            )
                    elif cutlass.const_expr(zero_fill_gather):
                        # A zero-size source copy reads no GMEM and writes a
                        # defined zero to padding. Keeping the instruction outside
                        # an outer lane predicate lets the compiler share its
                        # uniform memory descriptor across all gather copies.
                        is_valid_k = k_base_bytes + k_thread_offset < k * a_dtype.width // 8
                        prims.cp_async_shared_global(
                            smem_ir,
                            gmem_ir,
                            size=bytes_per_load,
                            modifier="cg",
                            cp_size=cutlass.Int32(is_valid_m & is_valid_k) * bytes_per_load,
                        )
                    elif cutlass.const_expr(mma_tiler_mnk[2] > 128):
                        # K may be partial: M OOB → skip (predicate off),
                        # K OOB with valid M → zero-fill (cp_size=0).
                        # Both sides in bytes: k_thread_offset is a byte offset.
                        is_valid_k = k_base_bytes + k_thread_offset < k * a_dtype.width // 8
                        cp_sz = cutlass.Int32(is_valid_k * bytes_per_load)
                        # Predicated cp.async has no NVVM op form — emit @p cp.async inline.
                        prims.inline_ptx_hl(
                            f"cp.async.cg.shared.global [{{$r0}}], [{{$r1}}], {bytes_per_load}, {{$r2}};",
                            read_only_args=[smem_ir, gmem_ir, cp_sz],
                            pred=is_valid_m,
                        )
                    else:
                        # K always aligned: M OOB → skip entirely.
                        # No zero-fill needed — epilogue masks invalid rows.
                        prims.inline_ptx_hl(
                            f"cp.async.cg.shared.global [{{$r0}}], [{{$r1}}], {bytes_per_load};",
                            read_only_args=[smem_ir, gmem_ir],
                            pred=is_valid_m,
                        )

                prims.cp_async_mbarrier_arrive(
                    ab_full_mbar_ptr.subview(current_ab_stage), noinc=True
                )

            # Advance the same local worker index; data barriers remain unchanged.
            work_linear_idx += local_stride
            work_tile = local_work_tile(
                work_linear_idx,
                local_actual_tiles_m,
                local_n_tiles,
                local_major_divisor,
                tile_sched_params.raster_along_m,
            )

    # =========================================================================
    # TMA warp (warp 9): GMEM -> SMEM loads for B, SFB (A and SFA by cp.async)
    # =========================================================================
    if warp_idx == tma_warp_id:
        prims.setmaxregister(num_regs_tma, prims.SetMaxRegisterAction.DECREASE)

        # L2 eviction policy for the B stream. Measured B-tile reuse is 1 and the L2 hit
        # rate is 1.5-2.5%: every weight byte is read exactly once, so allocating L2
        # lines for it can only displace the one tensor that *is* reused -- A, whose
        # ~393 KB per expert serves all 32 n-tiles. Both this kernel and production
        # currently leave the policy at default, so this is untested on either side.
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
        # Initialize the deterministic per-role work index.
        work_linear_idx = cutlass.Int32(cute.arch.block_idx()[2])
        work_tile = local_work_tile(
            work_linear_idx,
            local_actual_tiles_m,
            local_n_tiles,
            local_major_divisor,
            tile_sched_params.raster_along_m,
        )
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            tile_coord_m = cur_tile_coord[0]
            tile_coord_n = cur_tile_coord[1]
            coord_n = tile_coord_n * mma_tiler_mnk[1]
            # SF N coordinate is coarser: each SF atom covers 128 N elements
            coord_n_sf = coord_n // (num_m0_per_sf_atom * num_m1_per_sf_atom)
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
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(
                        ab_full_mbar_ptr.subview(current_ab_stage), num_tma_load_bytes
                    )

                coord_k_packed = k_tile * mma_tiler_mnk[2] // fp4_lanes_per_storage
                coord_k_sf = k_tile * rest_k_sf_per_ktile

                sB_staged = sB.subview(num_bytes_b * current_ab_stage)
                sSFB_staged = sSFB.subview(num_bytes_sfb * current_ab_stage)

                if prims.elect_sync():
                    # TMA coordinates count packed FP4x2 elements; the descriptor
                    # covers the complete logical K tile in one transaction.
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sB_staged,
                        tma_b_desc.get_ptr(),
                        (coord_k_packed, coord_n, coord_l_b),
                        ab_full_mbar_ptr.subview(current_ab_stage),
                        l2_cache_hint=b_policy,
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sSFB_staged,
                        tma_sfb_desc.get_ptr(),
                        (0, coord_k_sf, coord_n_sf, coord_l_b),
                        ab_full_mbar_ptr.subview(current_ab_stage),
                    )

            # Advance the same local worker index; data barriers remain unchanged.
            work_linear_idx += local_stride
            work_tile = local_work_tile(
                work_linear_idx,
                local_actual_tiles_m,
                local_n_tiles,
                local_major_divisor,
                tile_sched_params.raster_along_m,
            )

    # =========================================================================
    # MMA warp (warp 8): TMEM alloc, S2T copy, block-scaled tcgen05.mma compute
    # =========================================================================
    if warp_idx == mma_warp_id:
        prims.setmaxregister(num_regs_mma, prims.SetMaxRegisterAction.DECREASE)
        # Allocate TMEM using cutlass - MMA warp is the allocator (once, before loop)
        prims.tcgen05_alloc(tmem_ptr_i32, num_tmem_alloc_cols)
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
        # Sync with epilogue warps after allocation
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)
        # Retrieve TMEM pointer from smem buffer (returns cutlass.Array)
        tmem_raw_addr = tmem_ptr_i32.load()
        acc_tmem_ptr = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Float32)

        # ISA descriptor: a_format=0 (E4M3), b_format=5 (E2M1), scale=1 (E8M0)
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
        idesc_k96 = prims.Tcgen05MxOmmaInstrDesc.build(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            scale_format=int(prims.MxScaleFormat.UE4M3),
            n_dim=mma_tiler_mnk[1],
            m_dim=mma_tiler_mnk[0],
            k_dim=1,
            sparsity_version=0,
        )

        # S2T copy mode (32x128b with warpx4 broadcast)
        s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4

        # TMEM addresses packed as (row_id << 16) | col_id
        # With acc_stages=2, TMEM is partitioned into 2 slots:
        #   [ACC_0 | SFA_0 | SFB_0] [ACC_1 | SFA_1 | SFB_1]
        tmem_base_col = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        sfa_cols = rest_k_sf_per_ktile * rest_m_sf_per_ktile * num_tmem_cols_per_sf_atom
        sfb_cols = rest_k_sf_per_ktile * rest_n_sf_per_ktile * num_tmem_cols_per_sf_atom
        # Each acc stage occupies N + SFA + SFB TMEM columns
        cols_per_acc_stage = mma_tiler_mnk[1] + sfa_cols + sfb_cols
        num_smem_inc_per_sf_atom = num_elts_atom_sf_e8 >> 4  # 16B granularity

        # Build SMEM descriptors once (stage 0) — only start address changes per stage.
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
        # Stage offsets in 16B granularity for tcgen05_update_smem_start_address.
        stage_inc_a = num_bytes_a >> 4
        stage_inc_b = num_bytes_b >> 4
        stage_inc_sfa = num_bytes_sfa >> 4
        stage_inc_sfb = num_bytes_sfb >> 4

        # Overlap (tiler_n>=256): SF columns shared between stages, acc overlaps by sf_cols
        if cutlass.const_expr(mma_tiler_mnk[1] >= 256):
            sf_cols = sfa_cols + sfb_cols
            acc_stage_stride = mma_tiler_mnk[1] - sf_cols
            num_acc_tmem_cols = mma_tiler_mnk[1] * 2 - sf_cols

        # ---- Persistent loop: MMA warp ----
        # Initialize the deterministic per-role work index.
        work_linear_idx = cutlass.Int32(cute.arch.block_idx()[2])
        work_tile = local_work_tile(
            work_linear_idx,
            local_actual_tiles_m,
            local_n_tiles,
            local_major_divisor,
            tile_sched_params.raster_along_m,
        )
        # AB pipeline consumer state: stage index carries across tiles.
        ab_stage_idx = 0
        acc_stage_idx = 0
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            tile_coord_n = cur_tile_coord[1]
            # SFB tmem offset within an SF atom (tile-dependent)
            sfb_tmem_col_offset = 0
            if cutlass.const_expr(
                mma_tiler_mnk[1] % (num_m0_per_sf_atom * num_m1_per_sf_atom) != 0
            ):
                num_mma_n_per_sf_atom = (num_m0_per_sf_atom * num_m1_per_sf_atom) // mma_tiler_mnk[
                    1
                ]
                num_tmem_col_per_mma = num_tmem_cols_per_sf_atom // num_mma_n_per_sf_atom
                sfb_tmem_col_offset = (tile_coord_n % num_mma_n_per_sf_atom) * num_tmem_col_per_mma

            # Wait for acc buffer empty (from epilogue)
            if cutlass.const_expr(mma_tiler_mnk[1] >= 256):
                # Overlap: MMA writes stage = phase^1
                acc_stage_mma = acc_empty_phase_bit ^ 1
                while not prims.mbarrier_try_wait_parity(acc_empty_mbar_ptr, acc_empty_phase_bit):
                    pass
                # ACC TMEM: overlapping stride; SF at fixed offset
                base_col_id = tmem_base_col + acc_stage_mma * acc_stage_stride
                sfa_col_id = tmem_base_col + num_acc_tmem_cols
                sfb_col_id = sfa_col_id + sfa_cols
            else:
                while not prims.mbarrier_try_wait_parity(
                    acc_empty_mbar_ptr.subview(acc_stage_idx),
                    acc_empty_phase_bit,
                    time_limit=10_000_000,
                ):
                    pass
                # Per-stage TMEM addresses (acc_stages=2: stage 0 and stage 1 interleaved)
                acc_stage_col_offset = acc_stage_idx * cols_per_acc_stage
                base_col_id = tmem_base_col + acc_stage_col_offset
                sfa_col_id = base_col_id + mma_tiler_mnk[1]
                sfb_col_id = sfa_col_id + sfa_cols
            sfa_tmem_addr_base = (base_row_id << 16) | sfa_col_id
            sfb_tmem_addr_base = (base_row_id << 16) | sfb_col_id
            # Update acc_tmem_ptr for MMA output
            acc_tmem_ptr = prims.make_tmem_ptr((base_row_id << 16) | base_col_id, cutlass.Float32)

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

                desc_a_nvvm_s2t_base = desc_sfa_s2t_base + cutlass.Int64(
                    current_ab_stage * stage_inc_sfa
                )
                desc_b_nvvm_s2t_base = desc_sfb_s2t_base + cutlass.Int64(
                    current_ab_stage * stage_inc_sfb
                )
                desc_a_nvvm_mma_base = desc_a_mma_base + cutlass.Int64(
                    current_ab_stage * stage_inc_a
                )
                desc_b_nvvm_mma_base = desc_b_mma_base + cutlass.Int64(
                    current_ab_stage * stage_inc_b
                )

                # S2T: copy SFA from SMEM to TMEM, linearized (M, K) SF atom indexing
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

                # S2T: copy SFB from SMEM to TMEM, linearized (N, K) SF atom indexing
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

                sfa_mma_tmem_addr_base = sfa_tmem_addr_base
                sfb_mma_tmem_addr_base = sfb_tmem_addr_base + sfb_tmem_col_offset
                # SM103 mixed instruction schedule within an UNCHANGED K256
                # stage: K96 + K96 + K64. No instruction reads the next AB
                # stage, so the original completion/reuse point stays valid.
                # SF storage still uses four K64 atoms. K96 consumes six scale
                # bytes: starts 0/6 use atom+ID=(0,0)/(1,2); final K64 starts 12.
                for kblock_idx in cutlass.range_constexpr(3):
                    if cutlass.const_expr(kblock_idx < 2):
                        start_k = kblock_idx * 96
                        start_sf = kblock_idx * 6
                        current_idesc = idesc_k96
                    else:
                        start_k = 192
                        start_sf = 12
                        current_idesc = idesc
                    sf_atom_idx = start_sf // 4
                    sf_inside_atom = start_sf % 4
                    sfa_tmem_ptr_mma = prims.make_tmem_ptr(
                        sfa_mma_tmem_addr_base + sf_atom_idx * num_tmem_cols_per_kblock_sfa,
                        cutlass.Int32,
                    )
                    sfb_tmem_ptr_mma = prims.make_tmem_ptr(
                        sfb_mma_tmem_addr_base + sf_atom_idx * num_tmem_cols_per_kblock_sfb,
                        cutlass.Int32,
                    )
                    idesc_updated = current_idesc.set_sf_ids(
                        a_sf_id=sf_inside_atom, b_sf_id=sf_inside_atom
                    )
                    desc_offset = (start_k // 2) >> 4
                    desc_a_nvvm_mma = desc_a_nvvm_mma_base + desc_offset
                    desc_b_nvvm_mma = desc_b_nvvm_mma_base + desc_offset
                    if cutlass.const_expr(kblock_idx < 2):
                        # SM103 absolute-next-buffer mode. The next address is
                        # this SAME stage: neither K96 issue crosses K256.
                        # Current inherited roots use version0; K96 explicitly
                        # uses version1, leaving the K64 descriptor unchanged.
                        desc_a_nvvm_mma = prims.Tcgen05SmemDesc(
                            ((desc_a_nvvm_mma | (1 << 52) | (1 << 46)) & -4294901761)
                            | ((desc_a_nvvm_mma_base & 0xFFFF) << 16)
                        )
                        desc_b_nvvm_mma = prims.Tcgen05SmemDesc(
                            ((desc_b_nvvm_mma | (1 << 52) | (1 << 46)) & -4294901761)
                            | ((desc_b_nvvm_mma_base & 0xFFFF) << 16)
                        )
                    first = (k_tile == 0) and (kblock_idx == 0)
                    if prims.elect_sync():
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF4NVF4,
                            prims.CTAGroup.CTA_1,
                            acc_tmem_ptr,
                            desc_a_nvvm_mma,
                            desc_b_nvvm_mma,
                            idesc_updated,
                            not first,
                            sfa_tmem_ptr_mma,
                            sfb_tmem_ptr_mma,
                            scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                        )
                if prims.elect_sync():
                    prims.tcgen05_commit(ab_empty_mbar_ptr.subview(current_ab_stage))

            # Signal acc ready for epilogue
            if prims.elect_sync():
                prims.tcgen05_commit(acc_full_mbar_ptr.subview(acc_stage_idx))

            acc_stage_idx = acc_stage_idx + 1
            if acc_stage_idx == acc_stages:
                acc_stage_idx = 0
                acc_empty_phase_bit = acc_empty_phase_bit ^ 1

            # Advance the same local worker index; data barriers remain unchanged.
            work_linear_idx += local_stride
            work_tile = local_work_tile(
                work_linear_idx,
                local_actual_tiles_m,
                local_n_tiles,
                local_major_divisor,
                tile_sched_params.raster_along_m,
            )

    # =========================================================================
    # Epilogue warps (warps 0-3): T2R -> FP32->NVFP4 -> SMEM -> TMA store
    # =========================================================================
    if warp_idx < ldgsts_warp_id[0]:
        prims.setmaxregister(num_regs_epilog, prims.SetMaxRegisterAction.INCREASE)
        # Sync with MMA warp after TMEM allocation (once, before loop)
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)

        # Retrieve TMEM pointer from smem buffer (same as MMA warp)
        tmem_raw_addr = tmem_ptr_i32.load()
        acc_tmem_ptr = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Float32)

        tmem_base_col = tmem_raw_addr & 0xFFFF
        # TMEM stride for double-buffered acc (same as MMA warp)
        sfa_cols = rest_k_sf_per_ktile * rest_m_sf_per_ktile * num_tmem_cols_per_sf_atom
        sfb_cols = rest_k_sf_per_ktile * rest_n_sf_per_ktile * num_tmem_cols_per_sf_atom
        if cutlass.const_expr(mma_tiler_mnk[1] >= 256):
            sf_cols = sfa_cols + sfb_cols
            acc_stage_stride = mma_tiler_mnk[1] - sf_cols
            iter_early_release_pairs = cute.ceil_div(sf_cols, 2 * epi_subtile_n) - 1
        else:
            cols_per_acc_stage = mma_tiler_mnk[1] + sfa_cols + sfb_cols

        # T2R instruction configuration (tile-invariant)
        t2r_inst_shape = "32x32b"
        t2r_inst_repx = epi_subtile_n
        # Bytes per epilogue stage in sC. The reference could write rows*cols because its
        # C was FP8 at 1 byte/element; C is 4-bit here, so the element width applies.
        # Must agree with num_bytes_c_smem, which already carries c_dtype.width // 8 --
        # sm100_utils.make_smem_layout_epi(Float4E2M1FN, ROW_MAJOR, (128,64), 1) confirms
        # 4096 B/stage.
        epi_subtile_size = mma_tiler_mnk[0] * epi_subtile_n * c_dtype.width // 8
        subtile_cnt = cute.ceil_div(mma_tiler_mnk[1], epi_subtile_n)
        # SwiGLU: process pairs of subtiles → output count is half
        num_output_subtiles = subtile_cnt // 2

        # SFC granularity is sf_vec_size, NOT epi_subtile_n. The two were both 32 in the
        # GPT-OSS reference, so one amax per output subtile happened to be exactly one
        # scale factor and the conflation was invisible. GLM has epi_subtile_n = 64
        # (the up/gate interleave, fixed by the weight layout) against sf_vec_size = 16
        # (fixed by NVFP4), so each output subtile needs sfc_group = 4 scale factors.
        sfc_group = epi_subtile_n // sf_vec_size
        num_sfc_per_tile = num_output_subtiles * sfc_group
        assert num_output_subtiles >= 1, (
            f"mma_tiler_n ({mma_tiler_mnk[1]}) < 2 * epi_subtile_n ({epi_subtile_n}): a "
            f"tile this narrow holds a single interleave block and can never form an "
            f"up/gate pair, so the epilogue would silently emit nothing. Production "
            f"rejects the same configuration -- only mma_tiler_n 128 and 256 are valid."
        )
        # SFC: collect FP32 scale factors during the loop, convert to E4M3 after.
        # Local array (rmem) for indexed storage across loop iterations.
        sfc_local = cutlass.Array(cutlass.Float32, num_sfc_per_tile)
        sfc_n_blocks = cute.ceil_div(mma_tiler_mnk[1] // 2, sf_vec_size)

        lane_id = tidx % 32
        thread_row = warp_idx * 32 + lane_id
        m0 = thread_row % 32
        m1 = thread_row // 32
        sfc_gmem = cutlass.Array(
            cutlass.inttoptr(sfc_ptr, mem_space=1, dtype=cutlass.Int8),
            dtype=cutlass.Int8,
        )

        # ---- Persistent loop: Epilogue warps ----
        # Initialize the deterministic per-role work index.
        work_linear_idx = cutlass.Int32(cute.arch.block_idx()[2])
        work_tile = local_work_tile(
            work_linear_idx,
            local_actual_tiles_m,
            local_n_tiles,
            local_major_divisor,
            tile_sched_params.raster_along_m,
        )
        acc_stage_idx = 0
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            tile_coord_m = cur_tile_coord[0]
            tile_coord_n = cur_tile_coord[1]
            # Compute per-tile coordinates
            coordc_m = tile_coord_m * mma_tiler_mnk[0]
            coordc_n_out = tile_coord_n * (mma_tiler_mnk[1] // 2)
            # Unused; kept so the generated code stays unchanged.
            coordc_n_in = tile_coord_n * mma_tiler_mnk[1]  # noqa: F841
            expert_idx = tile_idx_arr.subview(tile_coord_m).load()
            mn_limit = mn_limit_arr.subview(tile_coord_m).load()
            # Per-expert, matching production's `alpha_tuple[0][expert_idx]`. Reading
            # element 0 for every expert is correct only when all experts share a
            # scale, which the op-level test does not.
            alpha = alpha_arr.subview(expert_idx).load()

            # Overlap: determine pair order and release threshold
            if cutlass.const_expr(mma_tiler_mnk[1] >= 256):
                acc_stage_epi = acc_full_phase_bit
                reverse_pairs = acc_stage_epi == 0
                release_idx = iter_early_release_pairs
            else:
                release_idx = num_output_subtiles - 1

            # Wait for acc ready from MMA warp
            while not prims.mbarrier_try_wait_parity(
                acc_full_mbar_ptr.subview(acc_stage_idx), acc_full_phase_bit
            ):
                pass

            # Per-stage TMEM col offset for T2R reads
            if cutlass.const_expr(mma_tiler_mnk[1] >= 256):
                base_col_id = tmem_base_col + acc_stage_epi * acc_stage_stride
            else:
                base_col_id = tmem_base_col + acc_stage_idx * cols_per_acc_stage

            epi_stage_idx = 0
            # Invalid rows skip arithmetic but still publish every scale entry.
            # Initialize the entire tile: N256 owns eight entries, not four.
            sfc_local.store(cutlass.vector.full((num_sfc_per_tile,), 0.0, cutlass.Float32))

            # Epilogue tiling loop — process pairs: (up_subtile, gate_subtile)
            for pair_idx in cutlass.range(num_output_subtiles):
                real_pair_idx = pair_idx
                if cutlass.const_expr(mma_tiler_mnk[1] >= 256):
                    if reverse_pairs:
                        real_pair_idx = num_output_subtiles - 1 - pair_idx
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
                # Release acc buffer: at release_idx (early for overlap,
                # last pair for non-overlap).
                if pair_idx == release_idx:
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage_idx))

                swiglu_out = cutlass.vector.full_like(up_f32, 0.0)
                # Unused; kept so the generated code stays unchanged.
                sfc_scalar = cutlass.Float32(0.0)  # noqa: F841
                abs_row_epi = coordc_m + tidx
                if abs_row_epi < mn_limit:
                    # GLM-5.2 SwiGLU: out = (alpha*up) * silu(alpha*gate),
                    #   silu(x) = x * sigmoid(x) = x * rcp(1 + exp(-x))
                    # `alpha` is the NVFP4 global dequant scale applied to the
                    # accumulator. This differs from the GPT-OSS reference in three
                    # ways, all deliberate:
                    #   - no per-expert bias add (GLM has no MoE bias)
                    #   - no +1 on `up`     (GPT-OSS used (up+1)*gate)
                    #   - no clamp to +/-swiglu_limit
                    alpha_vec = cutlass.vector.full_like(gate_f32, alpha)
                    up_s = up_f32 * alpha_vec
                    gate_s = gate_f32 * alpha_vec
                    # exp2(-x*log2e) == exp(-x); exp2 is the cheaper GPU primitive.
                    log2e = cutlass.vector.full_like(gate_f32, 1.4426950408889634)
                    exp_neg = cute.math.exp2(-gate_s * log2e, fastmath=True)
                    one = cutlass.vector.full_like(gate_f32, 1.0)
                    sigmoid = cute.math.rcp(one + exp_neg, approx=True, ftz=True)
                    # Match native SwiGLU's FP32 multiply order before NVFP4
                    # quantization; reassociation can change values and scales.
                    swiglu_out = (sigmoid * gate_s) * up_s

                    # SFC: amax per sf_vec_size-wide group, normalized to c_dtype's max
                    # representable value. GPT-OSS emitted FP8E4M3 (max 448); GLM emits
                    # NVFP4 E2M1 (max 6.0) -- see c_dtype_amax at module scope.
                    #
                    # This thread owns one full row of epi_subtile_n columns (128 threads
                    # x 64 cols = the 128x64 subtile), so all sfc_group reductions are
                    # local to its own vector -- no cross-thread communication. The
                    # reference reduced the whole vector to a single scalar, which was
                    # correct only because its epi_subtile_n equalled its sf_vec_size.
                    abs_swiglu = cute.math.abs(swiglu_out)
                    for g in cutlass.range_constexpr(sfc_group):
                        seg = abs_swiglu[g * sf_vec_size : (g + 1) * sf_vec_size]
                        amax_g = cutlass.Float32(
                            _vector.reduction(T_.f32(), _vector.CombiningKind.MAXIMUMF, seg)
                        )
                        # Store by real_pair_idx, not the iteration index: stage 0 walks
                        # pairs in reverse, and indexing by the real one keeps sfc_local
                        # in N order so the GMEM store needs no reversal shuffle.
                        # Round-trip through E4M3 before storing, so the reciprocal
                        # applied to the values below is the scale that actually reaches
                        # GMEM. Production does the same (`rcp_approx` of the quantised
                        # `qpvscale_up`); dividing by the unrounded f32 scale leaves the
                        # consumer unable to reconstruct the value, since it can only see
                        # the E4M3 one. Invisible at k=128, where amax happens to be
                        # exactly representable -- it shows up at k=256 as C code errors
                        # of 1-2 with SFC itself still matching.
                        sfc_q = cutlass.Float32(
                            (amax_g * cutlass.Float32(1.0 / c_dtype_amax) * norm_const)
                            .to(sf_dtype)
                            .to(cutlass.Float32)
                        )
                        sfc_local.subview(real_pair_idx * sfc_group + g).store(sfc_q)

                    # Normalise each group by its own reciprocal. Build the epi_subtile_n
                    # wide reciprocal vector by broadcasting the sfc_group values through
                    # a shuffle mask [0]*vec + [1]*vec + ... rather than one scalar.
                    from cutlass._mlir.dialects import vector as _vec_dialect

                    sfc_grp_vec = sfc_local.subview(real_pair_idx * sfc_group).load(
                        vector_size=sfc_group
                    )
                    rcp_grp = cute.math.rcp(sfc_grp_vec, approx=True, ftz=True)
                    # Match production's finite reciprocal clamp. Quantized SF=0
                    # otherwise makes zero activations evaluate as 0*Inf before
                    # FP4 conversion. The stored SFA stays zero and nonzero/tiny
                    # values retain production's finite, saturating normalization.
                    rcp_grp = rcp_grp * cutlass.vector.full_like(rcp_grp, norm_const)
                    fp32_max = cutlass.vector.full_like(
                        rcp_grp, cutlass.Float32(3.4028234663852886e38)
                    )
                    rcp_grp = cutlass.vector.where(rcp_grp > fp32_max, fp32_max, rcp_grp)
                    sfc_rcp_vec = type(swiglu_out)(
                        _vec_dialect.shuffle(
                            rcp_grp,
                            rcp_grp,
                            [i // sf_vec_size for i in range(epi_subtile_n)],
                        ),
                        dtype=cutlass.Float32,
                    )
                    # The `norm_const` factor is not redundant with the one already in
                    # SFC: production computes SFC = amax * rcp_limit * norm_const and
                    # then scales values by `norm_const * rcp(SFC)`, so the two cancel
                    # to value / (amax * rcp_limit) and the row maximum lands exactly on
                    # the dtype limit. Dropping it divides every value by norm_const.
                    # With norm_const = 1.0 the omission is invisible; with
                    # norm_const = 1/global_sf the output saturates to 0x77 everywhere.
                    # norm_const is already incorporated in the clamped reciprocal.
                    swiglu_out = swiglu_out * sfc_rcp_vec

                epi_stage_idx = (epi_stage_idx + 1) % epi_stages
                smem_subtile_ptr = sC.subview(epi_stage_idx * epi_subtile_size)
                # Byte offset of this thread's slice within the stage. t2r_inst_repx is an
                # element count, so it needs the element width -- FP8 in the reference made
                # the two numerically equal. memcheck showed the un-scaled form striding
                # threads 64 B apart instead of 32 B.
                smem_subtile_thr_ptr = smem_subtile_ptr.subview(
                    tidx * t2r_inst_repx * c_dtype.width // 8
                )

                # Quantise straight to NVFP4. The GPT-OSS reference emitted FP8E4M3
                # (1 byte/element); GLM's C is 4-bit, so this vector is half the bytes
                # and epi_subtile_size / the per-thread stride below are scaled to match.
                # Production does the same one-step `.to(c_dtype)` conversion.
                swiglu_fp4 = swiglu_out.to(c_dtype)
                # Bitcast to Int8 for the same reason the reference did it for FP8:
                # store_swizzled's internal .to(ptr.dtype) cannot handle the narrow float
                # type and libNVVM rejects the IR. t2r_inst_repx FP4 lanes are
                # t2r_inst_repx/2 bytes -- same total bits, so this is a pure reinterpret.
                swiglu_i8 = swiglu_fp4.bitcast(cutlass.Int8)
                smem_ptr = smem_subtile_thr_ptr.data_ptr()
                smem_ptr.store_swizzled(swiglu_i8, alignment=32, swizzle=cutlass.Swizzle(1, 4, 3))

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

            # ---- Vectorized SFC store to GMEM (after epilog tiling loop) ----
            # sfc_local is already in N order (stored by real_pair_idx), so the
            # reference's reverse-shuffle for the stage-0 reverse walk is not needed.
            #
            # num_sfc_per_tile = num_output_subtiles * sfc_group is always a multiple of
            # 4 under GLM (sfc_group == 4), and within a 512-byte atom the k mode is
            # contiguous with stride 1, so every group of 4 is one aligned vec4 store.
            # That collapses the reference's 4 / 2 / 1 packing branches, which keyed off
            # num_output_subtiles and no longer track the number of scale factors.
            #
            # sfc_n_blocks is a multiple of 4 for both legal mma_tiler_n, so
            # k_rest_base is always 0 and each group of 4 lands atom-aligned.
            global_sfc_n_base = tile_coord_n * sfc_n_blocks
            k_rest_base = global_sfc_n_base % 4
            rest_n_atom = global_sfc_n_base // 4
            # Atom offset: m0(0-31)*16 + m1(0-3)*4 + k_rest(0-3) within 512-byte atom
            atom_base = m0 * 16 + m1 * 4 + k_rest_base

            for j in cutlass.range_constexpr(num_sfc_per_tile // 4):
                sfc_f32_vec = sfc_local.subview(j * 4).load(vector_size=4)
                # E4M3, not the reference's E8M0: GLM-5.2's weight_scale tensors are
                # F8_E4M3 (NVFP4), and the kernel's SFC has to match what FC2 reads.
                sfc_e4m3_vec = sfc_f32_vec.to(sf_dtype)
                # Store narrow floats through the equivalent integer representation.
                sfc_packed = sfc_e4m3_vec.bitcast(cutlass.Int8)
                atom_idx = tile_coord_m * rest_n_sfc + rest_n_atom + j
                if cutlass.const_expr(pack_sfc_store):
                    # The base allocation is 16B-aligned and every atom_base is
                    # a multiple of four. An explicit 32-bit bitcast preserves
                    # the four E4M3 bytes while preventing scalar byte stores.
                    prims.store_ext(
                        sfc_packed.bitcast(cutlass.Int32),
                        sfc_gmem.subview(atom_idx * 512 + atom_base).data_ptr(),
                    )
                else:
                    sfc_gmem.subview(atom_idx * 512 + atom_base).store(sfc_packed)

            acc_stage_idx = acc_stage_idx + 1
            if acc_stage_idx == acc_stages:
                acc_stage_idx = 0
                acc_full_phase_bit = acc_full_phase_bit ^ 1
            # Advance the same local worker index; data barriers remain unchanged.
            work_linear_idx += local_stride
            work_tile = local_work_tile(
                work_linear_idx,
                local_actual_tiles_m,
                local_n_tiles,
                local_major_divisor,
                tile_sched_params.raster_along_m,
            )

        # Deallocate TMEM after persistent loop — epilogue warps sync
        prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)
        if warp_idx == 0:
            prims.tcgen05_dealloc(acc_tmem_ptr, num_tmem_alloc_cols, group=prims.CTAGroup.CTA_1)

    # Release the successor kernel as early as this CTA can.
    if cutlass.const_expr(USE_PDL):
        prims.griddepcontrol("launch_dependents")

    return


moe_gather_grouped_gemm_act_fusion_kernel.set_name_prefix(
    "glm52_prefill_gg1_local_scheduler", remove_cutlass_symbol=True
)


@cute.jit
def moe(
    b,
    sfb,
    c,
    tile_idx_to_expert_idx,
    a,
    a_scales,
    sfc,
    norm_const,
    alpha,
    token_mapping,
    mn_limit,
    problem_k: cutlass.Int32,
    topK: cutlass.Constexpr[int],
    num_tiles_m,
    max_tiles_m: cutlass.Int32,
    rest_n_sfc: cutlass.Int32,
    mma_tiler_n: cutlass.Constexpr[int],
    mma_tiler_k: cutlass.Constexpr[int],
    raster_along_m: cutlass.Constexpr[bool],
    num_sms: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
    pack_sfc_store: cutlass.Constexpr[bool] = False,
    zero_fill_gather: cutlass.Constexpr[bool] = False,
    native_gather: cutlass.Constexpr[bool] = False,
    typed_gather_ptrs: cutlass.Constexpr[bool] = False,
) -> None:
    """Torch-ABI adapter for the model's packed NVFP4 tensors.

    The public boundary stays in model-natural Torch order and uses
    ``torch.float4_e2m1fn_x2`` storage. Packed A/B TMA views, scalar-lane C
    recasting, and mode reversal are kernel details, so callers do not construct
    ``cute.Tensor`` objects or import ``cutlass.torch`` utilities.
    """
    c_fp4 = cute.recast_tensor(c, c_dtype)

    host_function(
        # Torch B is [expert, K/2, N], with K/2 physically contiguous.
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0])),
        cute.make_tensor(sfb.iterator, cute.select(sfb.layout, mode=[3, 2, 1, 0])),
        # Torch A/C are [batch, row, packed-column].
        cute.make_tensor(c_fp4.iterator, cute.select(c_fp4.layout, mode=[2, 1, 0])),
        tile_idx_to_expert_idx,
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[2, 1, 0])),
        a_scales,
        sfc,
        norm_const.iterator.toint(),
        alpha.iterator.toint(),
        token_mapping,
        mn_limit,
        problem_k,
        topK,
        num_tiles_m.iterator.toint(),
        max_tiles_m,
        rest_n_sfc,
        mma_tiler_n,
        mma_tiler_k,
        raster_along_m,
        num_sms,
        stream,
        pack_sfc_store,
        zero_fill_gather,
        native_gather,
        typed_gather_ptrs,
    )
