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

#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

//! SM versions (e.g. 90 for compute capability 9.0) this build was configured for, i.e. the entries of
//! CMAKE_CUDA_ARCHITECTURES. Only these GPUs have the optimized kernels the runtime relies on.
std::vector<int> const& getBuiltCudaArchitectures();

//! Whether a GPU of SM version `smVersion` is one this build was configured for.
bool isCudaArchitectureBuilt(int smVersion);

//! SM versions TensorRT-LLM provides optimized kernels for, i.e. the values CMAKE_CUDA_ARCHITECTURES may select
//! from. A GPU listed here but not built for is supported after rebuilding with it included.
std::vector<int> const& getSupportedCudaArchitectures();

//! Whether a GPU of SM version `smVersion` is one TensorRT-LLM provides optimized kernels for.
bool isCudaArchitectureSupported(int smVersion);

//! Throws if the GPU `device` is not one this build was configured for.
void checkCudaArchitectureSupported(int device);

} // namespace common

TRTLLM_NAMESPACE_END
