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
#include "tensorrt_llm/kernels/speculativeDecoding/chainSpeculativeSamplingKernels.h"

#include <cstddef>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! The widest vocab invokeChainSpecSamplingMc takes (64 tiles of 4096).
constexpr int kChainSpecMcMaxVocab = 64 * 4096;

//! The most draft tokens per request invokeChainSpecSamplingMc takes (its acceptance probabilities sit in shared
//! memory).
constexpr int kChainSpecMcMaxDraft = 32;

//! ST-01-REJ: chain speculative sampling (kernels/speculativeDecoding/chainSpeculativeSamplingKernels.cu, the
//! VEC_SIZE 4 kernel behind trtllm::chain_speculative_sampling) as ONE launch of batch x ctasPerRow CTAs (1024 threads
//! each) instead of one CTA per request.
//!
//! Semantics: the stock kernel's (see ChainSpeculativeSamplingParams): per request, the Philox acceptance loop, then
//! the corrected (or bonus) token by an inverse-CDF search over max(0, target - draft) (or the target), the outputs
//! [batch, numDraft + 1] tokens and [batch] counts.
//!
//! Bitwise contract vs the stock kernel (tokens and counts, every input of its vectorized path):
//!   1. every CTA of a request runs the stock tid-0 acceptance loop unchanged (same seed, subsequence
//!      kPerRowSubsequenceBase | offset, same draws), so all agree on the rejected position and the uniform;
//!   2. the (target, residual) mass: the stock kernel's 1024 per-thread chains (thread t adds the elements at
//!      4t + 4096k in k order) are spread over the request's CTAs unchanged, and one CTA reduces them with the same
//!      cub BlockReduce<float2, 1024> and Float2SumOp;
//!   3. each 4096-element tile's sum is the stock cub BlockReduce<float, 1024>::Sum over the same per-thread vec4, and
//!      each tile's last index with mass is a max (order-free);
//!   4. one CTA walks the tile sums in tile order with the stock fp32 prefix adds, and in the tile where the threshold
//!      falls runs the stock cub BlockScan<float, 1024>::InclusiveSum and atomicMin search; the fallback (no winner) is
//!      the last index with mass over the tiles the stock loop visits.
//! Validated bitwise (tokens + counts) vs trtllm::chain_speculative_sampling on GB300, K7, vocab 200064, batch
//! 1 / 2 / 4 / 8 x 5 cases (realistic, same, far, ties, onehot) x 64 seeds at ctasPerRow 4 / 8 / 16, eager and graph
//! 31.2 vs 41.2 us per call at batch 1 with ctasPerRow 16.
//!
//! Compiled with the repo's default flags, as the stock kernel (only adds, max and compares on the probs; no
//! --use_fast_math).
//!
//! Contract: vocabSize % 4 == 0, vocabSize <= kChainSpecMcMaxVocab, draftProbs / targetProbs 16-byte aligned (the
//! stock kernel's VEC_SIZE 4 path); 1024 % ctasPerRow == 0. The request's CTAs synchronize through global-memory
//! barriers, so all batch x ctasPerRow CTAs must be co-resident (one 1024-thread CTA per SM): pick ctasPerRow with
//! batch x ctasPerRow <= the SM count (ctasPerRow 1 needs no co-residency).
//!
//! Workspace: chainSpecSamplingMcWorkspaceBytesPerRow() x batch bytes, 16-byte aligned, ZERO-initialized once before
//! the first launch and then reused as is: every launch leaves it ready for the next (arrival counters reset, the
//! barrier generation only increases), so it is CUDA-graph safe without an init kernel. Launches that share one
//! workspace must be stream-ordered.
//! Slot-indexed draft rows (VR REJ_SLOTS): with rowIds set, request row r's draft distribution at position i is read at
//! draftProbs + rowIds[r] * rowStride + i * posStride (a [slots, maxDraft, >= vocab] buffer) instead of the contiguous
//! [batch, numDraft, vocab] layout; the same floats as the caller's gather, without the copy. rowIds null (the default)
//! keeps the contiguous layout.
struct ChainSpecMcDraftRows
{
    int64_t const* rowIds = nullptr;
    int64_t rowStride = 0;
    int64_t posStride = 0;
};

cudaError_t invokeChainSpecSamplingMc(speculative_decoding::ChainSpeculativeSamplingParams const& params,
    int ctasPerRow, void* workspace, cudaStream_t stream, ChainSpecMcDraftRows const& rows = ChainSpecMcDraftRows{});

//! Workspace bytes per request of invokeChainSpecSamplingMc.
size_t chainSpecSamplingMcWorkspaceBytesPerRow();

//! The kernel invokeChainSpecSamplingMc launches: 3 (REJ v3: staged chains, both tile variants before one row barrier,
//! the same adds and order; the default) or 2 (the v2 kernel, with TRTLLM_M31_DISABLE_REJ_V3=1, read once per process).
int chainSpecSamplingMcKernelVersion();

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
