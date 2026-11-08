/*
 * Copyright (c) 1993-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/tllmException.h"

#include "tensorrt_llm/common/tllmDataType.h"
#include <map>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

constexpr static size_t getDTypeSize(tensorrt_llm::DataType type)
{
    switch (type)
    {
    case tensorrt_llm::DataType::kINT64: return 8;
    case tensorrt_llm::DataType::kINT32: [[fallthrough]];
    case tensorrt_llm::DataType::kFLOAT: return 4;
    case tensorrt_llm::DataType::kBF16: [[fallthrough]];
    case tensorrt_llm::DataType::kHALF: return 2;
    case tensorrt_llm::DataType::kBOOL: [[fallthrough]];
    case tensorrt_llm::DataType::kUINT8: [[fallthrough]];
    case tensorrt_llm::DataType::kINT8: [[fallthrough]];
    case tensorrt_llm::DataType::kFP8: return 1;
    case tensorrt_llm::DataType::kINT4: TLLM_THROW("Cannot determine size of INT4 data type");
    case tensorrt_llm::DataType::kFP4: TLLM_THROW("Cannot determine size of FP4 data type");
    default: TLLM_THROW("Unknown dtype %d", static_cast<int>(type));
    }
    return 0;
}

constexpr static size_t getDTypeSizeInBits(tensorrt_llm::DataType type)
{
    switch (type)
    {
    case tensorrt_llm::DataType::kINT64: return 64;
    case tensorrt_llm::DataType::kINT32: [[fallthrough]];
    case tensorrt_llm::DataType::kFLOAT: return 32;
    case tensorrt_llm::DataType::kBF16: [[fallthrough]];
    case tensorrt_llm::DataType::kHALF: return 16;
    case tensorrt_llm::DataType::kBOOL: [[fallthrough]];
    case tensorrt_llm::DataType::kUINT8: [[fallthrough]];
    case tensorrt_llm::DataType::kINT8: [[fallthrough]];
    case tensorrt_llm::DataType::kFP8: return 8;
    case tensorrt_llm::DataType::kINT4: [[fallthrough]];
    case tensorrt_llm::DataType::kFP4: return 4;
    default: TLLM_THROW("Unknown dtype %d", static_cast<int>(type));
    }
    return 0;
}

[[maybe_unused]] static std::string getDtypeString(tensorrt_llm::DataType type)
{
    switch (type)
    {
    case tensorrt_llm::DataType::kFLOAT: return "fp32"; break;
    case tensorrt_llm::DataType::kHALF: return "fp16"; break;
    case tensorrt_llm::DataType::kINT8: return "int8"; break;
    case tensorrt_llm::DataType::kINT32: return "int32"; break;
    case tensorrt_llm::DataType::kBOOL: return "bool"; break;
    case tensorrt_llm::DataType::kUINT8: return "uint8"; break;
    case tensorrt_llm::DataType::kFP8: return "fp8"; break;
    case tensorrt_llm::DataType::kBF16: return "bf16"; break;
    case tensorrt_llm::DataType::kINT64: return "int64"; break;
    case tensorrt_llm::DataType::kINT4: return "int4"; break;
    case tensorrt_llm::DataType::kFP4: return "fp4"; break;
    default: throw std::runtime_error("Unsupported data type"); break;
    }

    return "";
}

} // namespace common

TRTLLM_NAMESPACE_END
