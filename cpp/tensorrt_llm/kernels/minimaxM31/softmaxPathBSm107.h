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
#include <cuda_runtime.h>
TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
constexpr int kSoftmaxPathBMinRows = 129;
constexpr int kSoftmaxPathBMaxRows = 448;
constexpr int kSoftmaxPathBVocab = 200064;
//! SM107 fp32 softmax for contiguous [129..448, 200064] tensors. Inputs and output must not overlap.
//! A scalar temperature or one fp32 temperature per row; inverse temperature is zero for temperature zero.
//! Preserves FlashInfer Path B's ordered denominator recurrence and final CUB sum. Compile with --use_fast_math.
//! Uses one stream-ordered CTA per row, no workspace and no PDL. All pointers belong to the stream's device.
cudaError_t invokeSoftmaxPathB(
    float* logits, float* output, float* temperature, float temperatureValue, int rows, cudaStream_t stream);
} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
