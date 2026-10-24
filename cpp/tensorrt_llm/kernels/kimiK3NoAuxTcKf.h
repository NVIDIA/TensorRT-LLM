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

namespace kernels
{

// Kimi K3 routing on FP32 logits: sigmoid scores plus the correction bias select the top 16 of 896 experts
// (n_group = topk_group = 1), whose scores are renormalized to sum to 1 (routed_scaling_factor 1). The outputs
// are bit-identical to invokeNoAuxTc<float, float, float, int32_t>(..., 896, 1, 1, 16, 1.0): same experts, same
// order, same weights. scores [numTokens, 896] must be contiguous and 16-byte aligned, bias [896] contiguous;
// topkValues and topkIndices are [numTokens, 16]. SM10x only.
void invokeKimiK3NoAuxTcKf(float const* scores, float const* bias, float* topkValues, int32_t* topkIndices,
    int64_t numTokens, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
