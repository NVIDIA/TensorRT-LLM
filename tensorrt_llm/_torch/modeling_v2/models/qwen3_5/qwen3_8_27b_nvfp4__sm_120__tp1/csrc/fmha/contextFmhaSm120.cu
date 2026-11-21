/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Launcher of the target's SM120 e4m3 head-256 context FMHA (see contextFmhaSm120.h).
//
// It replaces, for this configuration, TensorRT-LLM's generated launcher of
// fmha_v2_flash_attention_e4m3_fp32_64_128_S_q_paged_kv_256_causal_output_bf16_sm120_kernel_nl_tiled (same kernel
// traits, grid and shared memory), with the target's kernel body, traits, softmax, O store and split kernels (namespace
// TRTLLM_MV2_NS::fmha_sm120) compiled against the stock fmha_v2 headers. It takes
// bert::Fused_multihead_attention_params_v2 directly and runs the tiled kernel for every s, including max KV lengths <=
// 64, where TensorRT-LLM runs a generated non-tiled kernel.
//
// That kernel runs 2 CTAs/SM with one CTA per (64-row q-tile, head, sequence), and all CTAs of a long-prefix prefill
// do about the same work. A 1 x 2048-token prefill with 24 heads is 768 CTAs = 2 x 376 + 16 on 188 SMs: the last 16
// CTAs run as a tail wave. For 1 to 3 sequences, when dropping the last q-tile from the grid saves a whole wave:
//   A) the kernel runs q-tiles 0 .. Q-2 (a params copy with s - 64; bitwise the same rows), or is skipped when the
//      caller reads only each sequence's last row (skipLeadingQTiles);
//   B) the last q-tile of every (head, sequence) runs as `pieces` CTAs over contiguous KV ranges (one wave);
//   C) the pieces are merged in a fixed order and O is stored with the unchanged epilogue.
// The device code of B and C is in contextFmhaLastTileSplit.cuh.

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/fmha/contextFmhaSm120.h"

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/fmha/contextFmhaLastTileSplit.cuh"
#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/fmha/fmhaSm120KernelTraits.cuh"
#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/fmha/fmhaSm120NoloopTiled.cuh"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

// The causal kernel traits of the generated kernel (setup.py flash_attention_kernel_template,
// Kernel_traits_nl_tiled_causal; flags 0x1027: LDGSTS for Q/K/V, not head-interleaved, tiled; 0x200: no loop) with
// USE_EXP2_SOFTMAX (0x4000).
using Kt = fmha_sm120::Kernel_traits_v2_paged_kv_cache<fmha::Ada_qmma_e4m3_fp32_traits, 128, 256, 0, 64, 4, 1, 1,
    0x1027u | 0x200u | 0x4000u, 3, true, fmha::bf16_t>;

using Ws = fmha_sm120::last_tile_split::Workspace<Kt>;

// The q-tile height of the kernel (the generated launcher's noloop_step).
constexpr int kStepQ = 64;
// At most this many pieces per (head, sequence).
constexpr int kMaxPieces = 16;
constexpr int kMaxPiecesTotal = kernels::kContextFmhaSm120MaxPiecesTotal;
constexpr int kMaxBatch = kernels::kContextFmhaSm120MaxBatch;

static_assert(Kt::STORE_O_FROM_REGS, "");
static_assert(Kt::USE_EXP2_SOFTMAX, "");
static_assert(Kt::Q_RESIDENT, "");
static_assert(Kt::Cta_tile_p::M == kStepQ, "");
static_assert(sizeof(float4) * (Ws::ACC_FLOAT4S_PER_PIECE + Ws::STATS_FLOAT4S_PER_PIECE) * kMaxPiecesTotal
        == kernels::kContextFmhaSm120SplitWorkspaceBytes,
    "kContextFmhaSm120SplitWorkspaceBytes does not match the split workspace");

__global__ void contextFmhaE4m3D256CausalPagedKvSm120Kernel(bert::Fused_multihead_attention_params_v2 params)
{
    fmha_sm120::device_flash_attention_nl_tiled<Kt>(params);
}

// The generated launcher's causal branch, for this kernel.
void launchKernel(bert::Fused_multihead_attention_params_v2 const& params, cudaStream_t stream)
{
    using Cta_tile_o = typename Kt::Cta_tile_o;
    int const ctasPerORow = (params.d + Cta_tile_o::N - 1) / Cta_tile_o::N;
    int const loopIters = (params.s + kStepQ - 1) / kStepQ;
    dim3 const grid(loopIters * ctasPerORow, params.h, params.b);
    constexpr int kSmemSize = Kt::BYTES_PER_SMEM;
    if (kSmemSize >= 48 * 1024)
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(
            contextFmhaE4m3D256CausalPagedKvSm120Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }
    contextFmhaE4m3D256CausalPagedKvSm120Kernel<<<grid, Kt::THREADS, Kt::BYTES_PER_SMEM, stream>>>(params);
}

// CTAs/SM of the piece kernel (the kernel's body with the same traits and shared memory, so the same occupancy), per
// device.
int pieceKernelOccupancy()
{
    constexpr int kMaxDevices = 64;
    static int sOccupancy[kMaxDevices] = {};
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices)
    {
        return 0;
    }
    if (sOccupancy[device] == 0)
    {
        constexpr int kSmemSize = Kt::BYTES_PER_SMEM;
        auto pieceKernel = fmha_sm120::last_tile_split::fmha_v2_nl_tiled_last_tile_piece_kernel<Kt>;
        if (kSmemSize >= 48 * 1024)
        {
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(pieceKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
        int occupancy = 0;
        TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, pieceKernel, Kt::THREADS, kSmemSize));
        sOccupancy[device] = occupancy;
    }
    return sOccupancy[device];
}

} // namespace

namespace kernels
{

void launchContextFmhaSm120(bert::Fused_multihead_attention_params_v2 const& params, int multiProcessorCount,
    void* splitWorkspace, bool skipLeadingQTiles, cudaStream_t stream)
{
    namespace lts = fmha_sm120::last_tile_split;

    TLLM_CHECK_WITH_INFO(params.d == 256 && params.dv == 256,
        "SM120 e4m3 context FMHA: head size 256 only, got d=%d dv=%d", params.d, params.dv);
    if (params.b <= 0 || params.s <= 0)
    {
        return;
    }

    // The last q-tile split: only for the plain causal prefill of 1 to kMaxBatch unpadded sequences, and only with
    // the split workspace.
    bool const splittable = splitWorkspace != nullptr && params.b >= 1 && params.b <= kMaxBatch && !params.is_s_padded
        && !params.has_alibi && params.softcapping_scale_bmm1 == 0 && params.softmax_stats_ptr == nullptr
        && params.attention_sinks == nullptr && params.sage.q.scales == nullptr && params.sage.k.scales == nullptr
        && params.sage.v.scales == nullptr && params.skip_softmax_threshold_scale_factor == 0;
    int pieces = 0;
    if (splittable)
    {
        int const slots = multiProcessorCount * pieceKernelOccupancy();
        pieces = lts::choose_num_pieces(
            params.b, params.h, params.s, kStepQ, slots, kMaxPiecesTotal, kMaxPieces, kMaxBatch);
    }
    if (pieces == 0)
    {
        launchKernel(params, stream);
        TLLM_CUDA_CHECK(cudaGetLastError());
        return;
    }

    // A) q-tiles 0 .. Q-2, unless the caller reads only each sequence's last row and has checked that every
    // sequence's last row lies in q-tile Q-1: then those rows are left unwritten and B + C below write the last q-tile
    // exactly as they always do.
    if (!skipLeadingQTiles)
    {
        bert::Fused_multihead_attention_params_v2 paramsA = params;
        paramsA.s -= kStepQ;
        launchKernel(paramsA, stream);
    }

    // B) + C) The last q-tile.
    auto* const wsAcc = static_cast<float4*>(splitWorkspace);
    auto* const wsStats = wsAcc + static_cast<size_t>(Ws::ACC_FLOAT4S_PER_PIECE) * kMaxPiecesTotal;
    int const qLoopLast = (params.s + kStepQ - 1) / kStepQ - 1;
    cudaError_t const err = lts::launch_last_tile_split<Kt>(params, qLoopLast, pieces, wsAcc, wsStats, stream);
    TLLM_CHECK_WITH_INFO(
        err == cudaSuccess, "SM120 e4m3 context FMHA last q-tile KV split launch failed: %s", cudaGetErrorString(err));
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
