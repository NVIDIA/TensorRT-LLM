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
namespace minimaxM31
{

//! The largest vocabulary invokeSamplingFromProbsMcFi takes (64 chunks of 4096).
constexpr int32_t kSamplingFromProbsMcMaxVocab = 64 * 4096;

//! SFP_MC_FI: FlashInfer's categorical sampling from probabilities (include/flashinfer/sampling.cuh,
//! SamplingFromProbKernel<1024, BLOCK_SCAN_WARP_SCANS, BLOCK_REDUCE_WARP_REDUCTIONS, 4, DETERMINISTIC = true>, the
//! kernel behind flashinfer.sampling.sampling_from_probs(deterministic=True) with tensor seed / offset) as ONE launch
//! of rows x ctasPerRow CTAs (1024 threads each) instead of one CTA per row: one int32 token per row.
//!
//! FlashInfer's kernel scans a row's 4096-element chunks in order and stops at the chunk that holds the drawn token,
//! so its time follows the token's position (one chunk to the whole vocabulary). This kernel reads every chunk once,
//! split over the row's CTAs, so its time does not depend on the draw.
//!
//! Bitwise contract vs FlashInfer's kernel (tokens):
//!   1. u = curand_uniform of curand_init(seed[row * seedStride], row, offset[row * offsetStride]), unscaled: as
//!      FlashInfer's SamplingFromProbKernel reads them, a stride is 1 for a tensor with one entry per row and 0 for a
//!      one-entry tensor broadcast to every row;
//!   2. each chunk's sum of (p > 0 ? p : 0) is FlashInfer's cub BlockReduce<float, 1024, WARP_REDUCTIONS>::Sum over the
//!      same per-thread vec4; each chunk's last index with p > 0 is a max (order-free);
//!   3. one CTA walks the chunks in order with FlashInfer's fp32 adds up to the first chunk where aggregate + sum > u,
//!      runs FlashInfer's DeterministicInclusiveSum (copied) in that chunk and takes its first element over u, else the
//!      last index with p > 0 of the chunks visited, else token 0.
//! Validated bitwise vs flashinfer.sampling.sampling_from_probs on VR (sm_107), vocab 200064: rows 1 / 8 / 12 / 16 /
//! 24 / 48 x 32 seeds, the token in the first / last chunk, FlashInfer's fallbacks, padding rows, denormals and
//! ctasPerRow 1..64 (microbenchmark, 420 / 420 calls); 8.2-10.1 us vs 5.6 (token in the first chunk) to 40.2 us (in the
//! last) for FlashInfer at 12 rows with ctasPerRow 16.
//!
//! Compiled with --use_fast_math, as FlashInfer's JIT compiles it: flush-to-zero decides denormal probabilities.
//!
//! Contract: vocabSize % 4 == 0, vocabSize <= kSamplingFromProbsMcMaxVocab, probs 16-byte aligned with a row stride
//! divisible by 4, 1024 % ctasPerRow == 0. No CTA waits for another (the last CTA of a row to finish its chunks runs
//! the walk), so the grid needs no co-residency.
//!
//! Workspace: samplingFromProbsMcFiWorkspaceBytesPerRow() x rows bytes, 16-byte aligned, ZERO-initialized once before
//! the first launch and then reused as is (each launch adds 1024 to a row's arrival counter whatever its ctasPerRow, so
//! the counter needs no reset; CUDA-graph safe). Launches that share one workspace must be stream-ordered.
cudaError_t invokeSamplingFromProbsMcFi(float const* probs, int64_t rowStride, int64_t const* seed, int32_t seedStride,
    int64_t const* offset, int32_t offsetStride, int32_t* outputTokenIds, int32_t rows, int32_t vocabSize,
    int ctasPerRow, void* workspace, cudaStream_t stream);

//! Workspace bytes per row of invokeSamplingFromProbsMcFi.
size_t samplingFromProbsMcFiWorkspaceBytesPerRow();

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
