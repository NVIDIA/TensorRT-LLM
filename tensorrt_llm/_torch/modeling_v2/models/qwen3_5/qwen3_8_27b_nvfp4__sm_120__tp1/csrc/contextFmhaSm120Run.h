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

#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

// The prompt (context) attention of the 16 attention layers: the target's SM120 e4m3 head-256 causal paged-KV flash
// kernel (csrc/fmha) over FP8 Q and the paged FP8 KV cache (cached prefix + this chunk), BF16 output. 1 to 3 sequences
// run their last 64-row q-tile split along KV when that saves a whole wave.
struct ContextFmhaSm120Args
{
    void const* qFp8;           // [tokens][24][256] e4m3
    void* output;               // [tokens][24][256] BF16
    int32_t const* cuQSeqlens;  // [batch + 1]
    int32_t const* cuKvSeqlens; // [batch + 1]: full KV lengths (cached + this chunk)
    tensorrt_llm::kernels::KVBlockArray kvCache;
    int batch;
    int maxQSeqLen;               // longest chunk of the batch
    int slidingWindowSize;        // the KV capacity (max_seq_len): no window
    float hostBmm1Scale;          // 1 / (sqrt(256) * q_scaling)
    float const* bmm1ScaleDevice; // [2]: qk scale, qk scale * log2(e)
    float const* bmm2ScaleDevice; // [1]
    uint32_t* tileCounter;
    void* splitWorkspace;         // contextFmhaSm120SplitWorkspaceBytes() bytes, or nullptr: never split
    bool skipLeadingQTiles;       // with the split, leave the rows before each sequence's last q-tile unwritten
    int multiProcessorCount;
};

size_t contextFmhaSm120SplitWorkspaceBytes();

void runContextFmhaSm120(ContextFmhaSm120Args const& args, cudaStream_t stream);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
