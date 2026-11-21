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

// The attention op's context FMHA call: the kernel parameters TensorRT-LLM's FMHA runner fills for this configuration
// (fmhaRunner.cpp setupKernelParams; Q_PAGED_KV, e4m3 in, BF16 out, causal), then the target's launcher.

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/contextFmhaSm120Run.h"
#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/fmha/contextFmhaSm120.h"

#include <cstdint>
#include <cstring>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int kNumHeads = 24;
constexpr int kNumKvHeads = 4;
constexpr int kHeadSize = 256;

uint32_t floatBits(float x)
{
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

} // namespace

size_t contextFmhaSm120SplitWorkspaceBytes()
{
    return kContextFmhaSm120SplitWorkspaceBytes;
}

void runContextFmhaSm120(ContextFmhaSm120Args const& a, cudaStream_t stream)
{
    bert::Fused_multihead_attention_params_v2 params{};
    params.b = a.batch;
    params.s = a.maxQSeqLen;
    params.d = kHeadSize;
    params.dv = kHeadSize;
    params.h = kNumHeads;
    params.h_kv = kNumKvHeads;
    params.h_q_per_kv = kNumHeads / kNumKvHeads;
    params.sliding_window_size = a.slidingWindowSize;
    params.log2_chunked_attention_size = 0;
    params.is_s_padded = false;
    params.num_grouped_heads = 1;

    params.q_ptr = const_cast<void*>(a.qFp8);
    params.q_stride_in_bytes = kNumHeads * kHeadSize; // e4m3
    params.cu_q_seqlens = const_cast<int*>(a.cuQSeqlens);
    params.cu_kv_seqlens = const_cast<int*>(a.cuKvSeqlens);

    tensorrt_llm::kernels::KVBlockArrayForContextFMHA const kvArray = a.kvCache.copyKVBlockArrayForContextFMHA();
    static_assert(sizeof(kvArray) == sizeof(params.paged_kv_cache), "KVBlockArrayForContextFMHA != Kv_block_array");
    std::memcpy(&params.paged_kv_cache, &kvArray, sizeof(kvArray));
    // One head's slab inside a page.
    params.k_stride_in_bytes = static_cast<int64_t>(a.kvCache.mTokensPerBlock) * kHeadSize;
    params.v_stride_in_bytes = params.k_stride_in_bytes;

    params.o_ptr = a.output;
    params.o_stride_in_bytes = kNumHeads * kHeadSize * 2; // BF16
    params.softmax_stats_ptr = nullptr;
    params.softmax_stats_stride_in_bytes = sizeof(float) * 2 * kNumHeads;

    params.scale_bmm1 = floatBits(a.hostBmm1Scale);
    params.scale_softmax = floatBits(1.f);
    params.scale_bmm2 = floatBits(1.f);
    params.softcapping_scale_bmm1 = 0;
    params.scale_bmm1_d = reinterpret_cast<uint32_t*>(const_cast<float*>(a.bmm1ScaleDevice));
    params.scale_bmm2_d = reinterpret_cast<uint32_t*>(const_cast<float*>(a.bmm2ScaleDevice));
    params.tile_id_counter_ptr = a.tileCounter;
    params.has_alibi = false;
    params.attention_sinks = nullptr;

    launchContextFmhaSm120(params, a.multiProcessorCount, a.splitWorkspace, a.skipLeadingQTiles, stream);
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
