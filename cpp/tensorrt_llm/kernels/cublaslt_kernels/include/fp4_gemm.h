/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <vector>
#include <cstddef>  // for size_t

namespace tensorrt_llm
{
namespace kernels
{
namespace cublaslt_kernels
{

// Forward declarations
class CublasLtFp4GemmRunnerInterface;
template <typename T>
class CublasLtFp4GemmRunner;

} // namespace cublaslt_kernels
} // namespace kernels
} // namespace tensorrt_llm
