/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::kernels::dsv3MinLatencyKernels
{

template <typename T, int kHdIn, int kHdOut, int kTileN>
void invokeFusedAGemm(T* output, T const* mat_a, T const* mat_b, int num_tokens, cudaStream_t const stream);

} // namespace tensorrt_llm::kernels::dsv3MinLatencyKernels
