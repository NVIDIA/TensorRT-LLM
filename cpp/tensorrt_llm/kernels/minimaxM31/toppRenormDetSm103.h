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

#include <cstddef>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! ST-01: the spec-decoding sampler's top-p renorm (FlashInfer top_p_renorm_probs) as ONE persistent launch of
//! rows x ctasPerRow CTAs (1024 threads each), flat in the kept count.
//!
//! Semantics: probs fp32 [rows, d] (contiguous, d % 4 == 0, 16-byte aligned), topP fp32 [rows]. Per row, the
//! threshold thr is the value FlashInfer's AirTopPRenormProb<IsDeterministic = true, float> finds with its three
//! radix passes (11 / 11 / 10 bits of the descending twiddled key); out[i] = probs[i] * norm where probs[i] >= thr,
//! else 0, with norm = 1 / sum (1 when sum <= 1e-8) and sum the kept values' total in FlashInfer's order.
//!
//! Bitwise contract vs FlashInfer top_p_renorm_probs(probs, top_p, is_deterministic=True) (air_top_p.cuh, FlashInfer
//! 0.6.15 == 0.6.18), for every row with top_p < 1:
//!   - the det histograms are INTEGER (uint64 mantissa sums + int counts), so the warp-aggregated updates here give
//!     FlashInfer's exact histograms in any order;
//!   - each pass's bucket selection is FlashInfer's last-block code (calcFloatValue, cooperative_groups warp / block
//!     sums, the sequential scan) run on integer-identical histograms with the same lane <-> bucket mapping;
//!   - the kept sum: thread chain t adds the kept values at i = t (mod 1024) in order (FlashInfer's per-thread input
//!     to cub BlockReduce<float, 1024>); the chains are spread over the row's CTAs and one CTA runs the same
//!     cub BlockReduce<float, 1024>::Sum on them.
//! Validated bitwise (renormed probs and the sampled token) on GB300 over R 7 / 8 / 56 / 64 x 8 row classes x
//! 4 top_p modes x 16 seeds, eager and graph (microbenchmark); v1.2's binning (each pass's histogram adds:
//! one aggregated atomic for a warp whose 32 lanes share a bucket, else per-lane atomics into one of 4 privatized
//! shared-memory copies; v1.1's cached bucket counts) is bitwise v1 on every row of the MB grid at 54.3 vs v1's 85.7 us
//! per 7-row call.
//!
//! Exception: a row with top_p >= 1 skips the radix search and keeps EVERY element
//! (threshold -inf), then the same sum and write. FlashInfer itself drops tail buckets at top_p 1 in ~43% of rows,
//! in BOTH its default and its deterministic mode; keep-all is the intended law. The choice is row-local (all CTAs of
//! a row take the same branch), so the other rows of a batch stay bitwise.
//!
//! MUST be compiled with --use_fast_math (FlashInfer's JIT flags: flush-to-zero and the approximate 1 / x of the norm);
//! without it the output is not bitwise FlashInfer's. cpp/tensorrt_llm/kernels/CMakeLists.txt sets the flag on this
//! one source.
//!
//! Workspace: toppRenormDetWorkspaceBytesPerRow() x rows bytes, ZERO-initialized once before the first launch and
//! then reused as is: every launch leaves it ready for the next (histograms cleared, arrival counters reset, the
//! barrier generation only increases), so it is CUDA-graph safe without an init kernel. Launches that share one
//! workspace must be stream-ordered. The row's CTAs synchronize through global-memory barriers, so all
//! rows x ctasPerRow CTAs must be co-resident (one 1024-thread CTA per SM): pick ctasPerRow with
//! rows x ctasPerRow <= the SM count (ctasPerRow 1 needs no co-residency). 1024 % ctasPerRow == 0.
//! keptOut (optional, int [rows]) receives the per-row count of elements >= the threshold.
cudaError_t invokeToppRenormDet(float const* probs, float* out, float const* topP, int rows, int d, int ctasPerRow,
    void* workspace, int* keptOut, cudaStream_t stream);

//! Workspace bytes per row of invokeToppRenormDet.
size_t toppRenormDetWorkspaceBytesPerRow();

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
