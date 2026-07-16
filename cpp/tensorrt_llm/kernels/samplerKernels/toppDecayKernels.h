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
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! \brief Fused post-sample runtime top-p update for the PyTorch TorchSampler's
//! Top-P Decay feature. One thread per sampled row; the decay-active gate is
//! applied on-device via \p isDecaySlot so no host-side filtering is needed.
//!
//! Unlike the legacy invokeComputeToppDecay (samplingTopPKernels.h), this variant
//! gathers the sampled token in-kernel from a slot-indexed strided view of the
//! new-tokens buffer (\p stepTokens with element stride \p stepTokenStride, i.e.
//! new_tokens[step, slot, beam] for a fixed step/beam), applies a gated reset
//! (reset_id < 0 disables the reset), and filters decay-active slots on-device
//! via \p isDecaySlot.
//!
//! For each sampled row i with slot s = sampledSlots[i] where isDecaySlot[s]:
//!   runtimeTopP[s] = (resetIds[s] >= 0 && stepTokens[s * stride] == resetIds[s])
//!                      ? initialTopP[s]
//!                      : max(runtimeTopP[s] * topPDecay[s], topPMin[s])
//!
//! All per-slot arrays are length numSlots; \p sampledSlots is length numSampled.
//! \p runtimeTopP is updated in place.
void invokeToppDecayUpdate(float* runtimeTopP, float const* initialTopP, float const* topPDecay, float const* topPMin,
    int32_t const* resetIds, bool const* isDecaySlot, int32_t const* stepTokens, int64_t stepTokenStride,
    int64_t const* sampledSlots, int32_t numSampled, cudaStream_t stream);

//! \brief Fused pre-sample per-row top-p gather for the Top-P Decay feature.
//! Replaces the eager index_select(runtime) + index_select(gate) + where(static)
//! chain with a single launch:
//!   rowTopP[i] = isDecaySlot[slots[i]] ? runtimeTopP[slots[i]] : staticTopP[i]
//! \p rowTopP / \p staticTopP / \p slots are length numRows (per-step group rows);
//! \p runtimeTopP / \p isDecaySlot are per-slot arrays.
void invokeToppDecayGather(float* rowTopP, float const* runtimeTopP, bool const* isDecaySlot, float const* staticTopP,
    int64_t const* slots, int32_t numRows, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
