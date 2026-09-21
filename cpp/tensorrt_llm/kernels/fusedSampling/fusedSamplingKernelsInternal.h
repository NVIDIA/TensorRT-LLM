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

#include "fusedSamplingKernels.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace fusedSampling
{

//! One block owns one row in the single-CTA family, so the best block size depends on
//! whether rows can fill the GPU or each row should use all available warp slots.
constexpr int kNarrowBlock = 512;
constexpr int kWideBlock = 1024;
constexpr int kWideBlockMaxRows = 128;

//! Large-vocabulary probability-producing small batches use several CTAs per row.
constexpr int kSmallBatchSplitBlock = 512;
constexpr int kSmallBatchSplits = 8;
constexpr int kSmallBatchSplitMaxRows = 32;
constexpr int kSmallBatchSplitMinVocab = 65536;

//! Architectural limit used by the single-CTA occupancy bound.
constexpr int kMaxThreadsPerSm = 2048;

[[nodiscard]] inline bool shouldUseMultiCta(FusedSamplingParams const& params)
{
    return params.outputProbs != nullptr && params.numRows <= kSmallBatchSplitMaxRows
        && params.vocabSize >= kSmallBatchSplitMinVocab;
}

template <typename T>
void launchFusedSamplingSingleCta(FusedSamplingParams const& params, cudaStream_t stream);

template <typename T>
void launchFusedSamplingMultiCta(FusedSamplingParams const& params, cudaStream_t stream);

} // namespace fusedSampling
} // namespace kernels
} // namespace tensorrt_llm
