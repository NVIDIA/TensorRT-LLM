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

#include <cstdint>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

//! \brief Inputs and outputs of the fused sampler.
//!
//! One fused invocation applies temperature, min-p, top-k and top-p together and decides
//! per row, on device, which of them to do any work for. The launcher selects a
//! single-CTA or multi-CTA execution family without changing this interface. A row that
//! enables no filter costs the softmax it would have paid anyway; the disable sentinels
//! below are what it is recognized by.
//!
//! The unifying observation: after the softmax's own max-reduce, w = exp(l/T - max) is
//! exactly p / p_max, and all three filters are thresholds on w --
//!   min-p keeps w >= minP                       (a constant, so it is free)
//!   top-k keeps the k largest                   (a threshold found by rank)
//!   top-p keeps the minimal mass-ordered prefix  (a threshold found by mass)
//! -- so the whole filter is a single comparison w >= t against the strictest of them.
struct FusedSamplingParams
{
    //! Input [numRows, vocabSize] of type T. Not modified.
    void const* logits{nullptr};
    //! Input [numRows]. Per-row temperature. The greedy sentinel (a very small value)
    //! makes the softmax one-hot, which is how a greedy row returns its argmax natively.
    float const* temperatures{nullptr};
    //! Input [numRows], optional. Per-row k. Any value >= vocabSize (including the
    //! INT32_MAX disable sentinel) or <= 0 disables top-k for that row.
    int32_t const* topKs{nullptr};
    //! Input [numRows], optional. Per-row p. Any value >= 1 disables top-p for that row.
    float const* topPs{nullptr};
    //! Input [numRows], optional. Per-row min-p. Any value <= 0 disables it for that row.
    //! 1.0 keeps only the argmax, matching SamplingParams' explicit-greedy semantics.
    float const* minPs{nullptr};

    //! Input [1] or [numRows]. Philox seed/offset. Tensors rather than scalars so the
    //! call is legal inside a captured CUDA graph.
    uint64_t const* seed{nullptr};
    uint64_t const* offset{nullptr};
    //! Whether seed/offset carry one entry per row rather than a single shared entry.
    bool perRowRng{false};

    //! Output [numRows], optional. Sampled token ids.
    int32_t* outputTokens{nullptr};
    //! Output [numRows, vocabSize] float32, optional. The filtered, renormalized
    //! distribution -- what the rejection path needs and what a sort-free sampler
    //! cannot hand back.
    float* outputProbs{nullptr};

    int32_t numRows{0};
    int32_t vocabSize{0};
};

//! \brief Launch the fused sampler. Writes whichever of outputTokens / outputProbs is set.
template <typename T>
void invokeFusedSampling(FusedSamplingParams const& params, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
