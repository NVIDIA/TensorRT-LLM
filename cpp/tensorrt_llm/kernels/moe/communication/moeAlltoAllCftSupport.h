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

// CFT depends on CUDA 13.4+ Logical Endpoint headers and matching fabric PTX support.
// Older CUDA toolchains are intentionally unsupported and compile only CFT stubs/traps.
#if defined(CUDART_VERSION)
#define TLLM_CFT_HAS_CUDA_13_4_SUPPORT (CUDART_VERSION >= 13040)
#elif defined(CUDA_VERSION)
#define TLLM_CFT_HAS_CUDA_13_4_SUPPORT (CUDA_VERSION >= 13040)
#elif defined(__CUDACC_VER_MAJOR__)                                                                                    \
    && ((__CUDACC_VER_MAJOR__ > 13) || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
#define TLLM_CFT_HAS_CUDA_13_4_SUPPORT 1
#else
#define TLLM_CFT_HAS_CUDA_13_4_SUPPORT 0
#endif
