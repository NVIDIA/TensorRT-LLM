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

#ifdef TLLM_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////

#define TLLM_CHECK_CUDA(call)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError status_ = call;                                                                                      \
        if (status_ != cudaSuccess)                                                                                    \
        {                                                                                                              \
            std::cerr << "CUDA error at file " << __FILE__ << " line " << __LINE__ << " call " << #call << ":"         \
                      << cudaGetErrorName(status_) << " -> " << cudaGetErrorString(status_) << std::endl;              \
        }                                                                                                              \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////

#define TLLM_CHECK_RET_CUDA(call)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError status_ = call;                                                                                      \
        if (status_ != cudaSuccess)                                                                                    \
        {                                                                                                              \
            std::cerr << "CUDA error at file " << __FILE__ << " line " << __LINE__ << " call " << #call << ":"         \
                      << cudaGetErrorName(status_) << " -> " << cudaGetErrorString(status_) << std::endl;              \
            return (int) status_;                                                                                      \
        }                                                                                                              \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////
