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

#include <cstdint>

namespace tensorrt_llm::kernels::nccl_device
{

// CUDA and kernel constants
constexpr int kWarpSize = 32;
constexpr int kMaxThreadsPerBlock = 1024;      // Maximum block size configurable for performance.
constexpr int kMinThreadsPerBlock = kWarpSize; // Minimum block size is a warp.
constexpr int kMaxUnrollFactor = 8; // We require manual instantiation and switches. Changing the number is not good
                                    // enough, see launcher function for details
} // namespace tensorrt_llm::kernels::nccl_device
