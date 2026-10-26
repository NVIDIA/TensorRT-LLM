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

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding
{

//! Chain speculative sampling: verifies each request's linear draft against the target.
//!
//! Draft token i is accepted with probability min(1, p_target / p_draft); the first rejected
//! position is replaced by a sample from norm(max(0, p_target - p_draft)), and when every
//! draft token is accepted a bonus token is sampled from the last target distribution. The
//! output distribution is the target's.
//!
//! Each request draws from the Philox subsequence its own offset names, so its result does not
//! depend on its row or on the other requests.
struct ChainSpeculativeSamplingParams
{
    //! [batchSize, numDraftTokens, vocabSize]
    float const* draftProbs{nullptr};
    //! [batchSize, numDraftTokens]
    int32_t const* draftTokenIds{nullptr};
    //! [batchSize, numDraftTokens + 1, vocabSize]
    float const* targetProbs{nullptr};
    //! [batchSize]
    int64_t const* seed{nullptr};
    //! [batchSize]
    int64_t const* offset{nullptr};
    //! Output [batchSize, numDraftTokens + 1]: the accepted draft tokens, then the corrected or
    //! bonus token, then -1.
    int32_t* outputTokenIds{nullptr};
    //! Output [batchSize]: accepted draft tokens + 1.
    int32_t* outputNumTokens{nullptr};
    int32_t batchSize{0};
    int32_t numDraftTokens{0};
    int32_t vocabSize{0};
};

void invokeChainSpeculativeSampling(ChainSpeculativeSamplingParams const& params, cudaStream_t stream);

} // namespace kernels::speculative_decoding

TRTLLM_NAMESPACE_END
