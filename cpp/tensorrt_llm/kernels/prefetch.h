/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

namespace tensorrt_llm
{
namespace kernels
{

template <typename T, int CTA_M, int CTA_K>
void cute_host_prefetch(T const* data, int M, int K, int strideM, int throttle_time, bool pdl, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
