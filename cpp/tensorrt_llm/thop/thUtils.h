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

#include "tensorrt_llm/runtime/iTensor.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/script.h>

#include <cstdio>

#define CHECK_TYPE(x, st)                                                                                              \
    TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), ", while ", st, " is expected")
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st)                                                                                             \
    CHECK_TH_CUDA(x);                                                                                                  \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_TYPE(x, st)
#define CHECK_CPU_INPUT(x, st)                                                                                         \
    CHECK_CPU(x);                                                                                                      \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_TYPE(x, st)
#define CHECK_OPTIONAL_INPUT(x, st)                                                                                    \
    if (x.has_value())                                                                                                 \
    {                                                                                                                  \
        CHECK_INPUT(x.value(), st);                                                                                    \
    }
#define CHECK_OPTIONAL_CPU_INPUT(x, st)                                                                                \
    if (x.has_value())                                                                                                 \
    {                                                                                                                  \
        CHECK_CPU_INPUT(x.value(), st);                                                                                \
    }
#define PRINT_TENSOR(x) std::cout << #x << ":\n" << x << std::endl
#define PRINT_TENSOR_SIZE(x) std::cout << "size of " << #x << ": " << x.sizes() << std::endl

namespace torch_ext
{

// TODO: switch to use torch native fp4 dtype when ready
constexpr auto FLOAT4_E2M1X2 = torch::ScalarType::Byte; // uint8_t
constexpr auto SF_DTYPE = torch::ScalarType::Byte;      // uint8_t

constexpr auto FP8_BLOCK_SCALING_SF_DTYPE = torch::ScalarType::Float;
constexpr auto FP8_ROWWISE_SF_DTYPE = torch::ScalarType::Float;

template <typename T>
inline T* get_ptr(torch::Tensor& t)
{
    return reinterpret_cast<T*>(t.data_ptr());
}

template <typename T>
inline T get_val(torch::Tensor& t, int idx)
{
    assert(idx < t.numel());
    return reinterpret_cast<T*>(t.data_ptr())[idx];
}

tensorrt_llm::runtime::ITensor::Shape convert_shape(torch::Tensor tensor);

template <typename T>
tensorrt_llm::runtime::ITensor::UniquePtr convert_tensor(torch::Tensor tensor);

size_t sizeBytes(torch::Tensor tensor);

// from: cpp/tensorrt_llm/plugins/common/gemmPluginProfiler.h
int nextPowerOfTwo(int v);

// from: cpp/tensorrt_llm/plugins/lowLatencyGemmPlugin/lowLatencyGemmPlugin.cpp
std::optional<float> getFloatEnv(char const* name);

cudaDataType_t convert_torch_dtype(torch::ScalarType dtype);

} // namespace torch_ext
