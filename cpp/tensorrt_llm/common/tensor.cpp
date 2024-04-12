/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/stringUtils.h"

#include "stdlib.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <numeric>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <dirent.h>
#endif // !defined(_WIN32)

namespace tensorrt_llm
{
namespace common
{

Tensor::Tensor()
    : // a none tensor.
    where(MEMORY_CPU)
    , type(TYPE_INVALID)
    , shape({})
    , data(nullptr)
{
}

Tensor::Tensor(MemoryType _where, DataType _type, std::vector<size_t> const& _shape, void const* _data)
    : where(_where)
    , type(_type)
    , shape(_shape)
    , data(_data)
{
}

size_t Tensor::size() const
{
    if (data == nullptr || shape.size() == 0)
    {
        return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), (size_t) 1, std::multiplies<size_t>());
}

size_t Tensor::sizeBytes() const
{
    return size() * Tensor::getTypeSize(type);
}

std::string Tensor::whereToString() const
{
    static const std::unordered_map<MemoryType, std::string> mem_to_string{
        {MEMORY_CPU, "CPU"}, {MEMORY_CPU_PINNED, "CPU_PINNED"}, {MEMORY_GPU, "GPU"}};
    return mem_to_string.at(where);
}

std::string Tensor::toString() const
{
    std::string memtype_str = whereToString();

    static const std::unordered_map<DataType, std::string> type_to_string{
        {TYPE_BOOL, "BOOL"},
        {TYPE_UINT8, "UINT8"},
        {TYPE_UINT16, "UINT16"},
        {TYPE_UINT32, "UINT32"},
        {TYPE_UINT64, "UINT64"},
        {TYPE_INT8, "INT8"},
        {TYPE_INT16, "INT16"},
        {TYPE_INT32, "INT32"},
        {TYPE_INT64, "INT64"},
        {TYPE_BF16, "BF16"},
        {TYPE_FP16, "FP16"},
        {TYPE_FP32, "FP32"},
        {TYPE_FP64, "FP64"},
        {TYPE_BYTES, "BYTES"},
        {TYPE_INVALID, "INVALID"},
        {TYPE_FP8_E4M3, "E4M3"},
        {TYPE_VOID, "VOID"},
    };
    return fmtstr("Tensor[where=%s, type=%s, shape=%s, data=%p]", memtype_str.c_str(), type_to_string.at(type).c_str(),
        vec2str(shape).c_str(), data);
}

size_t Tensor::getTypeSize(DataType type)
{
    static const std::unordered_map<DataType, size_t> type_map{{TYPE_BOOL, sizeof(bool)}, {TYPE_BYTES, sizeof(char)},
        {TYPE_UINT8, sizeof(uint8_t)}, {TYPE_UINT16, sizeof(uint16_t)}, {TYPE_UINT32, sizeof(uint32_t)},
        {TYPE_UINT64, sizeof(uint64_t)}, {TYPE_INT8, sizeof(int8_t)}, {TYPE_INT16, sizeof(int16_t)},
        {TYPE_INT32, sizeof(int32_t)}, {TYPE_INT64, sizeof(int64_t)},
#ifdef ENABLE_BF16
        {TYPE_BF16, sizeof(__nv_bfloat16)},
#endif
#ifdef ENABLE_FP8
        {TYPE_FP8_E4M3, sizeof(__nv_fp8_e4m3)},
#endif
        {TYPE_FP16, sizeof(half)}, {TYPE_FP32, sizeof(float)}, {TYPE_FP64, sizeof(double)}};
    return type_map.at(type);
}

std::string Tensor::getNumpyTypeDesc(DataType type) const
{
    static const std::unordered_map<DataType, std::string> type_map{{TYPE_INVALID, "x"}, {TYPE_BOOL, "?"},
        {TYPE_BYTES, "b"}, {TYPE_UINT8, "u1"}, {TYPE_UINT16, "u2"}, {TYPE_UINT32, "u4"}, {TYPE_UINT64, "u8"},
        {TYPE_INT8, "i1"}, {TYPE_INT16, "i2"}, {TYPE_INT32, "i4"}, {TYPE_INT64, "i8"}, {TYPE_FP16, "f2"},
        {TYPE_FP32, "f4"}, {TYPE_FP64, "f8"}};

    if (type == TYPE_BF16)
    {
        TLLM_LOG_WARNING(
            "getNumpyTypeDesc(TYPE_BF16) returns an invalid type 'x' since Numpy doesn't "
            "support bfloat16 as of now, it will be properly extended if numpy supports. "
            "Please refer for the discussions https://github.com/numpy/numpy/issues/19808.");
    }

    return type_map.count(type) > 0 ? type_map.at(type) : "x";
}

Tensor Tensor::slice(std::vector<size_t> shape, size_t offset) const
{
    if (this->data != nullptr)
    {
        size_t n_elts = this->size();
        size_t n_sliced_elts = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
        TLLM_CHECK_WITH_INFO(n_sliced_elts + offset <= n_elts,
            fmtstr("The number (%ld) of elements of sliced tensor exceeds that (%ld) of the original tensor",
                n_sliced_elts + offset, n_elts));
    }
    return Tensor(this->where, this->type, shape, this->getPtrWithOffset(offset));
}

TensorMap::TensorMap(std::unordered_map<std::string, Tensor> const& tensor_map)
{
    for (auto& kv : tensor_map)
    {
        if (kv.second.isValid())
        {
            insert(kv.first, kv.second);
        }
        else
        {
            TLLM_LOG_DEBUG(fmtstr("%s is not a valid tensor, skipping insert into TensorMap", kv.first.c_str()));
        }
    }
}

TensorMap::TensorMap(std::vector<Tensor> const& tensor_map)
{
    for (size_t i = 0; i < tensor_map.size(); i++)
    {
        insert(std::to_string(i), tensor_map[i]);
    }
}

TensorMap::TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensor_map)
{
    for (auto& pair : tensor_map)
    {
        if (pair.second.isValid())
        {
            insert(pair.first, pair.second);
        }
        else
        {
            TLLM_LOG_DEBUG(fmtstr("%s is not a valid tensor, skipping insert into TensorMap", pair.first.c_str()));
        }
    }
}

TensorMap::~TensorMap()
{
    tensor_map_.clear();
}

std::vector<std::string> TensorMap::keys() const
{
    std::vector<std::string> key_names;
    for (auto& kv : tensor_map_)
    {
        key_names.push_back(kv.first);
    }
    return key_names;
}

std::string TensorMap::toString()
{
    std::stringstream ss;
    ss << "{";
    std::vector<std::string> key_names = keys();
    for (size_t i = 0; i < tensor_map_.size(); ++i)
    {
        ss << key_names[i] << ": " << at(key_names[i]).toString();
        if (i < tensor_map_.size() - 1)
        {
            ss << ", ";
        }
    }
    ss << "}";
    return ss.str();
}

} // namespace common
} // namespace tensorrt_llm
