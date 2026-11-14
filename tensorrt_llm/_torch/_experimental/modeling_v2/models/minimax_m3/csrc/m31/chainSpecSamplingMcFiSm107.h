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

#include "minimax_m3/csrc/m31/chainSpecSamplingMcSm103.h"
#include "tensorrt_llm/kernels/speculativeDecoding/chainSpeculativeSamplingKernels.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstddef>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! REJ_MC_FI: FlashInfer's chain speculative sampling (include/flashinfer/sampling.cuh,
//! ChainSpeculativeSampling<1024, BLOCK_SCAN_WARP_SCANS, BLOCK_REDUCE_WARP_REDUCTIONS, 4, DETERMINISTIC = true>, the
//! kernel behind flashinfer.sampling.chain_speculative_sampling(deterministic=True) with tensor seed / offset) as ONE
//! launch of batch x ctasPerRow CTAs (1024 threads each) instead of one CTA per request. Same inputs and outputs as
//! invokeChainSpecSamplingMc ([batch, numDraft + 1] tokens padded with -1, [batch] counts = accepted + 1).
//!
//! Bitwise contract vs FlashInfer's kernel (tokens and counts):
//!   1. Philox: every CTA of a request replays FlashInfer's stream, curand_init(seed[request], request,
//!      offset[request]) (one seed / offset entry per request), the acceptance draws up to the first rejection, the
//!      counting loop's draws from that position on, then the resampling draw u;
//!   2. CTA c of G takes FlashInfer's thread lanes [c x 1024 / G, (c + 1) x 1024 / G) (whole warps) over every
//!      4096-element chunk of max(target - draft, 0) (draft 0 for the bonus row). Per lane it adds the lane's 4 values
//!      chunk after chunk, FlashInfer's per-thread total; per chunk and warp it takes cub's warp Sum of the lane's
//!      ThreadReduce and the warp's last index with mass, the pieces of FlashInfer's per-chunk
//!      BlockReduce<float, 1024, WARP_REDUCTIONS>::Sum;
//!   3. the last CTA to arrive reduces the 1024 lane totals with that BlockReduce (FlashInfer's total), adds each
//!      chunk's 32 warp sums in warp order (its chunk sums), takes u x total, walks the chunk prefix to the first
//!      crossing with FlashInfer's fp32 adds, runs FlashInfer's DeterministicInclusiveSum (copied) in that chunk and
//!      takes its first element over the threshold, else the last index with mass of the chunks visited (-1 when the
//!      residual has no mass at all, as FlashInfer's ChainSpeculativeSampling returns).
//! Validated bitwise (tokens + counts) vs flashinfer.sampling.chain_speculative_sampling on VR (sm_107), K7, vocab
//! 200064: served-like rows at batches 1 / 12 / 48, every rejection position and the bonus row, padding rows
//! (uniform and one-hot), denormals, rows without residual mass, slot-indexed rows, ctasPerRow 1..32 and CUDA-graph
//! replay.
//!
//! Compiled with --use_fast_math, as FlashInfer's JIT compiles it: flush-to-zero decides denormal probabilities.
//!
//! Contract: vocabSize % 4 == 0, vocabSize <= kChainSpecMcMaxVocab, numDraft <= kChainSpecMcMaxDraft, draftProbs /
//! targetProbs 16-byte aligned (slot rows: strides divisible by 4), ctasPerRow a power of two <= 32. No CTA waits for
//! another (the last CTA of a request to finish its lanes runs the tail), so the grid needs no co-residency.
//!
//! Workspace: chainSpecSamplingMcFiWorkspaceBytesPerRow() x batch bytes, 16-byte aligned, ZERO-initialized once before
//! the first launch and then reused as is (each launch adds 1024 to a request's arrival counter whatever its
//! ctasPerRow, so the counter needs no reset; CUDA-graph safe). Launches that share one workspace must be
//! stream-ordered.
cudaError_t invokeChainSpecSamplingMcFi(
    tensorrt_llm::kernels::speculative_decoding::ChainSpeculativeSamplingParams const& params, int ctasPerRow,
    void* workspace, cudaStream_t stream, ChainSpecMcDraftRows const& rows = ChainSpecMcDraftRows{});

//! Workspace bytes per request of invokeChainSpecSamplingMcFi.
size_t chainSpecSamplingMcFiWorkspaceBytesPerRow();

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
