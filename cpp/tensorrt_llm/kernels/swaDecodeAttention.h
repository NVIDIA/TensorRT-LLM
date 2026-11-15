/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"

#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Dedicated generation-phase attention for short sliding windows (window + maxSeqLenQ - 1 <= 160 keys): E4M3 Q and
// paged E4M3 KV (HND pages), head dim 64, BF16 output, optional attention sinks, linear (causal) multi-token queries.
// One CTA of 128 threads per (request, KV head) stages the window in shared memory, so several requests are in flight
// per SM. Same mask, sinks and device scales as the trtllm-gen generation FMHA; QK^T and PV run on FP16 mma.sync
// with FP32 accumulation and an FP32 softmax (P in FP16).
struct SwaDecodeAttentionParams
{
    void const* q;                 // [numTokens, numHeadsQ, 64] E4M3
    void* out;                     // [numTokens, numHeadsQ, 64] BF16
    KVBlockArray kv;               // the cache the preprocessing kernel just appended to
    int const* seqLensKv;          // [batch], including the current tokens
    int const* cuSeqLensQ;         // [batch + 1] or nullptr
    float const* scaleSoftmaxLog2; // device scalar
    float const* outputScale;      // device scalar
    float const* sinks;            // [numHeadsQ] or nullptr
    int batch;
    int numHeadsQ;
    int numHeadsKv;
    int maxSeqLenQ;
    int window;
};

// Whether the kernel covers the shape (head dim, Q heads per KV head, query tokens per request, window).
bool swaDecodeAttentionSupported(int headDim, int headsPerKv, int maxSeqLenQ, int window);

void launchSwaDecodeAttention(SwaDecodeAttentionParams const& params, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
