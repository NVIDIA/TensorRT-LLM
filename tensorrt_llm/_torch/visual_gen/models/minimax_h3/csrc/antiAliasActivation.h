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

#include <cuda_runtime.h>

namespace tensorrt_llm::visual_gen::minimax_h3
{

//! Taps of each Kaiser-sinc filter of the anti-aliased activation.
constexpr int kAntiAliasTaps = 12;

//! Fused anti-aliased SnakeBeta over contiguous fp32 x and out [rows, length], row = batch * channels + channel;
//! upFilter and downFilter hold kAntiAliasTaps taps, logAlpha and logBeta one value per channel.
cudaError_t invokeAntiAliasSnakeBeta(float* out, float const* x, float const* upFilter, float const* downFilter,
    float const* logAlpha, float const* logBeta, int rows, int channels, int length, cudaStream_t stream);

} // namespace tensorrt_llm::visual_gen::minimax_h3
