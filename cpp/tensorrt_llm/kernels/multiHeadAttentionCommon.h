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

#include "tensorrt_llm/common/assert.h"
#include <limits.h>
#include <stdint.h>

namespace tensorrt_llm
{
namespace kernels
{

// Change to this enum must sync with nvrtcWrapper.cpp in internal kernel repo
enum Data_type
{
    DATA_TYPE_BOOL,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT32,
    DATA_TYPE_BF16,
    DATA_TYPE_E2M1,
    DATA_TYPE_E4M3,
    DATA_TYPE_E5M2
};

static inline std::string data_type_to_string(Data_type dtype)
{
    switch (dtype)
    {
    case DATA_TYPE_BOOL: return "bool";
    case DATA_TYPE_FP16: return "fp16";
    case DATA_TYPE_FP32: return "fp32";
    case DATA_TYPE_INT4: return "int4";
    case DATA_TYPE_INT8: return "int8";
    case DATA_TYPE_INT32: return "int32";
    case DATA_TYPE_BF16: return "bf16";
    case DATA_TYPE_E2M1: return "e2m1";
    case DATA_TYPE_E4M3: return "e4m3";
    case DATA_TYPE_E5M2: return "e5m2";
    default: return std::to_string(static_cast<int>(dtype)) + " (unknown)";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline size_t get_size_in_bits(Data_type dtype)
{
    switch (dtype)
    {
    case DATA_TYPE_FP32: return 32;
    case DATA_TYPE_FP16: return 16;
    case DATA_TYPE_INT32: return 32;
    case DATA_TYPE_INT8: return 8;
    case DATA_TYPE_BF16: return 16;
    case DATA_TYPE_E2M1: return 4;
    case DATA_TYPE_E4M3: return 8;
    case DATA_TYPE_E5M2: return 8;
    default: TLLM_CHECK_WITH_INFO(false, "FMHA Data Type is not supported."); return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline size_t get_size_in_bytes(size_t n, Data_type dtype)
{
    switch (dtype)
    {
    case DATA_TYPE_FP32: return n * 4;
    case DATA_TYPE_FP16: return n * 2;
    case DATA_TYPE_INT32: return n * 4;
    case DATA_TYPE_INT8: return n;
    case DATA_TYPE_BF16: return n * 2;
    case DATA_TYPE_E2M1: TLLM_CHECK_WITH_INFO(n % 2 == 0, "Not supported."); return n / 2;
    case DATA_TYPE_E4M3: return n;
    case DATA_TYPE_E5M2: return n;
    default: TLLM_CHECK_WITH_INFO(false, "FMHA Data Type is not supported."); return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline size_t get_size_in_bytes(Data_type dtype)
{
    return get_size_in_bytes(1, dtype);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int32_t kSM_70 = 70;
constexpr int32_t kSM_72 = 72;
constexpr int32_t kSM_75 = 75;
constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;
constexpr int32_t kSM_89 = 89;
constexpr int32_t kSM_90 = 90;
constexpr int32_t kSM_100 = 100;
constexpr int32_t kSM_100f = 10100;
constexpr int32_t kSM_103 = 103;
constexpr int32_t kSM_120 = 120;
constexpr int32_t kSM_121 = 121;

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int kIdxScaleSoftmaxPtr = 0;
static constexpr int kIdxScaleSoftmaxLog2Ptr = 1;

} // namespace kernels
} // namespace tensorrt_llm
