/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Translation unit for the skip_softmax sm_120 / sm_121 warp-specialized FMHA
// (TMA-load + sync-MMA). This kernel is hand-written rather than emitted by
// fmha_v2/setup.py code-gen, so it is compiled directly into the
// _context_attention_kernels_120 target (see the sibling CMakeLists.txt). It
// only ever builds for the sm_120 family (sm_120 / sm_121), which is the only
// hardware that provides the TMA + sync-MMA combination this kernel targets.
//
// Supported shapes: BF16 in/out, head_dim == head_dim_v in {128, 256}, causal
// mask, PACKED_QKV layout. The runner dispatches here only when those
// constraints are met and the use_skip_softmax_fmha flag is set; see
// fused_multihead_attention_v2.cpp.
//
// The design rationale lives in
// cpp/kernels/fmha_v2/src/fmha/warpspec_sm120/README.md.

#include <cstdlib>

#include <cuda.h> // CUtensorMap

#include <fmha/traits.h>
#include <fmha/warpspec_sm120/kernel_traits.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_attention_kernel.h>

#include "fused_multihead_flash_attention_kernel_ws_sm120.h"

namespace fmha_skip_softmax
{

// NOTE: the `S` template arg is the *kv loop step* (per-iter KV tile size),
// not the runtime maximum sequence length. The runtime kv seqlen is read from
// binfo.actual_kv_seqlen and the kv loop iterates in chunks of S. With the TMA
// box size capped at 256 elements per axis, S must be <= 256 (we load
// STEP_KV = Cta_tile_p::N = S elements per TMA box call).
//
// HEAD_DIM must be a multiple of the 64-element (= 128-byte BF16) TMA chunk
// width so the Q/K head-dim chunks and the 64-wide V dv-chunks keep 128-byte
// smem rows -- the layout that matches the TMA 128B hardware swizzle. head_dim
// 128 and 256 both satisfy this; a non-multiple-of-64 head dim would break the
// swizzle invariant.
template <int HEAD_DIM>
using Skip_softmax_ktraits = fmha::ws_sm120::Kernel_traits_skip_softmax_sm120<
    /*Traits_=*/fmha::Ampere_hmma_bf16_traits,
    /*S=*/128,
    /*VALID_D_=*/HEAD_DIM,
    /*VALID_DV_=*/HEAD_DIM,
    /*STEP_Q_=*/64,
    /*WARPS_M_=*/4,
    /*WARPS_N_=*/1,
    /*VERSION_=*/2,
    /*MASK_VERSION_=*/3, // 3 = causal
    /*RING_DEPTH_=*/1,
    /*ENABLE_SKIP_SOFTMAX_=*/true,
    /*NUM_PRODUCER_WARPS_=*/1>;

} // namespace fmha_skip_softmax

// Templated entry kernel -- one instantiation per head dim (128, 256). THREADS
// is head-dim-independent (1 producer + WARPS_M*WARPS_N consumer warps), so the
// launch_bounds value is identical across instantiations.
template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS, 1) void skip_softmax_kernel(
    bert::Fused_multihead_attention_params_v2 const params, __grid_constant__ const CUtensorMap tma_q,
    __grid_constant__ const CUtensorMap tma_k, __grid_constant__ const CUtensorMap tma_v)
{
    // The CUtensorMaps live in const/param space (grid_constant); passing
    // their addresses to cp.async.bulk.tensor is a valid tensormap operand.
    fused_multihead_attention::device_flash_attention_ws_sm120<Ktraits>(params, &tma_q, &tma_k, &tma_v);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host-side launcher for the skip_softmax kernel.
//
// Sets up the TMA descriptors via DMA::Host::init_params, sizes shared memory,
// configures the launch attribute that lets the kernel use >48KB smem, and
// launches the kernel on the given stream.
//
// Returns the cudaError_t from the launch -- the caller is responsible for
// checking it (and running a separate sync to surface launch-time aborts).
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Ktraits>
static cudaError_t launch_skip_softmax(bert::Fused_multihead_attention_params_v2 params,
    bert::Fused_multihead_attention_launch_params const& launch_params, cudaStream_t stream)
{
    // 1. Build the three TMA descriptors host-side (cuTensorMapEncodeTiled).
    //    These are passed to the kernel as __grid_constant__ params.
    CUtensorMap tma_q{}, tma_k{}, tma_v{};
    typename fmha::ws_sm120::DMA<Ktraits>::Host dma_host;
    dma_host.init_params(params, launch_params, tma_q, tma_k, tma_v);

    // 2. Size the smem allocation. If it exceeds the 48 KB default, raise the
    //    cudaFuncAttributeMaxDynamicSharedMemorySize cap before launch. (head_dim
    //    128 needs roughly half the head_dim 256 footprint.)
    constexpr int smem_bytes = static_cast<int>(Ktraits::BYTES_PER_SMEM);
    if (smem_bytes >= 48 * 1024)
    {
        cudaError_t err = cudaFuncSetAttribute(
            skip_softmax_kernel<Ktraits>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess)
        {
            return err;
        }
    }

    // 3. Grid = (Q-tiles, H, B). One CTA per (Q-tile, head, batch). Each CTA
    //    has THREADS threads (1 producer warp + WARPS_M*WARPS_N consumer
    //    warps = 5 warps total = 160 threads).
    int const q_tiles = (params.s + Ktraits::STEP_Q - 1) / Ktraits::STEP_Q;
    dim3 const grid(q_tiles, params.h, params.b);
    dim3 const block(Ktraits::THREADS);

    skip_softmax_kernel<Ktraits><<<grid, block, smem_bytes, stream>>>(params, tma_q, tma_k, tma_v);
    return cudaGetLastError();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// In-engine dispatch bridges.
//
// The production fmhaRunner uses `tensorrt_llm::kernels::Fused_multihead_attention_params_v2`
// (defined in contextFusedMultiHeadAttention/fused_multihead_attention_common.h),
// which is a separate but ABI-compatible struct from the fmha_v2 `bert::` one --
// the generated kernels bridge them with reinterpret_cast, and we do the same.
// One bridge per supported head dim; both share the templated launch_skip_softmax<>
// path. The bert launch_params is NOT ABI-identical to kernels::Launch_params,
// so we copy the fields Host::init_params reads rather than reinterpret_cast it.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../fused_multihead_attention_common.h"
#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void run_skip_softmax_bf16_d256_causal_sm120(
    Fused_multihead_attention_params_v2& params, Launch_params const& launch_params, cudaStream_t stream)
{
    bert::Fused_multihead_attention_launch_params blp{};
    blp.total_q_seqlen = launch_params.total_q_seqlen;
    blp.attention_input_layout = fmha::Attention_input_layout::PACKED_QKV;

    ::launch_skip_softmax<::fmha_skip_softmax::Skip_softmax_ktraits<256>>(
        reinterpret_cast<bert::Fused_multihead_attention_params_v2&>(params), blp, stream);
}

void run_skip_softmax_bf16_d128_causal_sm120(
    Fused_multihead_attention_params_v2& params, Launch_params const& launch_params, cudaStream_t stream)
{
    bert::Fused_multihead_attention_launch_params blp{};
    blp.total_q_seqlen = launch_params.total_q_seqlen;
    blp.attention_input_layout = fmha::Attention_input_layout::PACKED_QKV;

    ::launch_skip_softmax<::fmha_skip_softmax::Skip_softmax_ktraits<128>>(
        reinterpret_cast<bert::Fused_multihead_attention_params_v2&>(params), blp, stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
