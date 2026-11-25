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

#include "tensorrt_llm/common/config.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3
{
//! Fused greedy Eagle3 verification of the MiniMax-M3 NVFP4 TP4 decode path (sm_103a): the cross-rank argmax of the
//! target logits shard and the strict acceptance in one kernel. See shardAccept.cu.
namespace shardAccept
{

constexpr int kMaxRows = 48;  // verify rows of one launch
constexpr int kThreads = 512; // threads per CTA

// Greedy-acceptance epilogue run by the last CTA once the global per-row argmax is known (rows = nreq requests x
// (draft_len + 1) verify positions, request-major). Mirrors the strict path of SpecWorkerBase: accepted = the target
// tokens, count = 1 + length of the matching draft prefix, optionally forced; plus the first draft pass's row index.
struct AcceptArgs
{
    int const* draft;      // [nreq, draft_len] previous draft tokens
    int const* batch_idx;  // [nreq] request row in the verify layout
    int* acc_tokens;       // [nreq, draft_len + 1]
    int* num_acc;          // [nreq]
    long long* gather_ids; // [nreq] last accepted row
    int draft_len;
    int nreq;
    int force_total;      // > 0 overrides the accepted count
    int* next_new;        // optional [nreq, next_stride]: column 0 = last accepted token (next step's input)
    int next_stride;
    int const* force_vec; // optional [nreq] per-request accepted count; overrides force_total
};

//! Greedy verification from this rank's BF16 logits shard [M, Vs] (row stride ld), M <= kMaxRows, over numCtas
//! kThreads-thread CTAs, followed by the strict acceptance epilogue.
cudaError_t invokeShardAccept(__nv_bfloat16 const* logits, int M, int Vs, long long ld, int vocabOffset,
    unsigned long long* keys, unsigned int* counters, unsigned long long* wsUc, unsigned long long* wsMc, int rank,
    int world, AcceptArgs const& acc, int numCtas, cudaStream_t stream);

} // namespace shardAccept
} // namespace minimaxM3
} // namespace kernels

TRTLLM_NAMESPACE_END
