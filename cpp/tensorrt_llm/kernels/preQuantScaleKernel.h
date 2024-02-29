/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#if defined(ENABLE_BF16)
#include <cuda_bf16.h>
#endif

#include <type_traits>
#include <vector>

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void apply_per_channel_scale_kernel_launcher(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream = 0);

} // namespace kernels
} // namespace tensorrt_llm
