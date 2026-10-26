/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Shared ABI enums and parameter structs of the tile-wise NVFP4 V2 MoE
 * all-to-all combine family (used by the SM107 Push-Combine DSO).
 */
#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"

#include <cuda_runtime_api.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

enum class MoeA2ACombineQuantFormat : std::uint32_t
{
    kNone = 0,
    kFp8E4M3 = 1,
    kTileNvfp4 = 2,
};

enum class MoeA2ACombinePayloadLayout : std::uint32_t
{
    kLegacy = 0,
    kTileNvfp4SplitPlane = 1,
    kTileNvfp4Compact = 2,
};

enum class MoeA2ACombineScaleLayout : std::uint32_t
{
    kNone = 0,
    kTile64Block16E4M3PlusF32LE = 1,
};

// Fixed-width, pointer-free layout record. It is safe to hash/serialize field by
// field, but callers must not hash compiler padding or a raw struct image.
struct alignas(8) MoeA2ACombineLayoutV1
{
    std::uint32_t abi_version;
    std::uint32_t quant_format;
    std::uint32_t format_version;
    std::uint32_t payload_layout;
    std::uint32_t scale_layout;
    std::uint32_t hidden_elements;
    std::uint32_t tile_elements;
    std::uint32_t scale_block_elements;
    std::uint32_t ep_size;
    std::uint32_t runtime_max_tokens_per_rank;
    std::uint32_t graph_static_max_tokens_per_rank;
    // Reserved on the split-plane path (must be 0). When payload_layout ==
    // kTileNvfp4Compact (packed-interleaved [Q|SF]), reserved0 carries the
    // q_aligned byte offset of the SF start within each packed record (3584).
    // No ABI change: the field is repurposed, not appended.
    std::uint32_t reserved0;
    std::uint64_t q_plane_offset_bytes;
    std::uint64_t sf_plane_offset_bytes;
    std::uint64_t src_token_stride_bytes;
    std::uint64_t q_token_stride_bytes;
    std::uint64_t sf_token_stride_bytes;
    std::uint64_t q_payload_bytes_per_token;
    std::uint64_t sf_payload_bytes_per_token;
    std::uint64_t workspace_size_per_rank_bytes;
};

static_assert(sizeof(MoeA2ACombineLayoutV1) == 112, "V2 layout ABI must remain 112 bytes");
static_assert(alignof(MoeA2ACombineLayoutV1) == 8, "V2 layout ABI must remain 8-byte aligned");

// tileNvfp4-local combine-ABI rank capacity. Only the V2 combine structs
// (MoeA2ACombineParamsV2 and TileNvfp4CombinePointersV2) are sized by it so
// the unified combine path can address EP up to 128. The shared dispatch-side
// kMaxRanks (moeAlltoAllKernels.h, 64) is deliberately untouched.
constexpr int kMaxRanksTile = 128;
static_assert(kMaxRanksTile >= kMaxRanks, "tile combine rank capacity must cover the shared dispatch capacity");

// V2 is a new type and new launcher symbol. Never append these fields to
// MoeA2ACombineParams or bind this object to a legacy launcher.
struct MoeA2ACombineParamsV2
{
    bool one_block_per_token;
    bool payload_in_workspace;

    int ep_size;
    int ep_rank;
    int local_num_tokens;
    int runtime_max_tokens_per_rank;
    int graph_static_max_tokens_per_rank;
    int top_k;

    // External mode: contiguous BF16 [E, R, 7168]. In-place mode: must be
    // null; source is the local q_recv_buffers[ep_rank] BF16 view.
    void const* prepare_payload;
    void* output_data;

    MoeA2ACombineLayoutV1 layout;

    std::uint32_t* flag_val;
    int* topk_target_ranks;
    int* topk_send_indices;
    int const* recv_counters;
    std::uint32_t* completion_flags[kMaxRanksTile];

    // Each pointer is already rebased to the corresponding peer's plane.
    // Prepare writes only index ep_rank; combine will consume every peer.
    std::uint8_t* q_recv_buffers[kMaxRanksTile];
    std::uint8_t* sf_recv_buffers[kMaxRanksTile];

    cudaStream_t stream;
};

// Phase-A implementation in the companion .cu. It accepts only the frozen
// tile-wise V1 BF16/H7168 split-plane descriptor.
void moe_a2a_prepare_combine_launch_v2(MoeA2ACombineParamsV2 const& params);

// Reserved production symbol. The prepare-only review patch must not register
// moe_a2a_combine_v2 until this decoder/readiness launcher has a definition and
// independent correctness coverage.
void moe_a2a_combine_launch_v2(MoeA2ACombineParamsV2 const& params);

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
