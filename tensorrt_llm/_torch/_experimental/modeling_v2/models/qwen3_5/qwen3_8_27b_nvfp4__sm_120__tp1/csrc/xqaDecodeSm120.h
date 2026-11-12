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

#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

// Decode attention of the 16 attention layers: the XQA HMMA kernel (csrc/xqa, a copy of TensorRT-LLM's
// cpp/kernels/xqa with this target's SM120 changes) compiled ahead of time for exactly this configuration:
// head size 256, 24 query / 4 KV heads, BF16 Q and output, FP8 (e4m3) paged KV cache with
// kXqaSm120TokensPerPage tokens per page, beam width 1, one query token per request.

constexpr uint32_t kXqaSm120TokensPerPage = 32;
constexpr uint32_t kXqaSm120KvHeads = 4;

// Bytes of the multi-block scratch xqaSm120Decode needs: the stock XQA multi-block budget (6000 sub-sequences).
// The shared-prefix pass carves its records from the same buffer.
size_t xqaSm120ScratchBytes();

// Semaphore words xqaSm120Decode needs for maxBatch requests. They must be zeroed once and stay alive; the kernel
// leaves them at zero.
size_t xqaSm120SemaphoreWords(uint32_t maxBatch, int multiProcessorCount);

// Sub-sequences per (request, KV head): TensorRT-LLM's computeMultiBlockCount for maxPastKvLength (the longest KV
// length of the batch, including the new token), honoring TRTLLM_XQA_BLOCKS_PER_SEQUENCE, then capped at one CTA wave
// (multiProcessorCount / (batch * kvHeads)) unless the environment variable is set.
uint32_t xqaSm120MultiBlockCount(uint32_t batch, uint32_t maxPastKvLength, int multiProcessorCount);

struct XqaSm120DecodeArgs
{
    void* output;              // [batch][24][256] BF16
    void const* q;             // [batch][24][256] BF16, RoPE applied
    void* kvPool;              // the layer's primary KV pool: FP8 pages [page][K|V][4 heads][32 tokens][256]
    int32_t const* kvPageList; // [batch][2][maxNbPagesPerSeq] page indices (KVBlockArray layout)
    uint32_t const* seqLens;   // [batch] KV length including the new token
    uint32_t maxNbPagesPerSeq;
    uint32_t batch;
    uint32_t slidingWindowSize;    // the KV capacity (max_seq_len): no window
    float qScale;                  // q_scaling
    float const* kvCacheScale;     // device [1]: dequantization scale of K and V
    uint32_t* semaphores;          // xqaSm120SemaphoreWords words, zeroed once
    void* scratch;                 // xqaSm120ScratchBytes bytes
    uint32_t multiBlock;           // xqaSm120MultiBlockCount
    uint32_t sharedPrefixMinBatch; // run the shared-prefix pass from this batch on (0: never)
    int multiProcessorCount;
    bool enablePdl;
};

// Launches [kernel_mha_prefix +] kernel_mha. With batch >= max(2, sharedPrefixMinBatch) and a multi-block launch,
// kernel_mha_prefix first attends every request's queries against the leading KV pages all requests share (found on
// the device, graph safe) and kernel_mha then walks only each request's suffix.
void xqaSm120Decode(XqaSm120DecodeArgs const& args, cudaStream_t stream);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
