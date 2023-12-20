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

#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/stringUtils.h"

#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm
{
namespace common
{

typedef enum datatype_enum
{
    TYPE_INVALID,
    TYPE_BOOL,
    TYPE_UINT8,
    TYPE_UINT16,
    TYPE_UINT32,
    TYPE_UINT64,
    TYPE_INT8,
    TYPE_INT16,
    TYPE_INT32,
    TYPE_INT64,
    TYPE_FP16,
    TYPE_FP32,
    TYPE_FP64,
    TYPE_BYTES,
    TYPE_BF16,
    TYPE_FP8_E4M3,
    TYPE_STR,
    TYPE_VOID,
    TYPE_INT32_PTR,
} DataType;

template <typename T>
struct TensorDataType
{
};

template <>
struct TensorDataType<bool>
{
    static constexpr DataType value = TYPE_BOOL;
};

template <>
struct TensorDataType<std::uint8_t>
{
    static constexpr DataType value = TYPE_UINT8;
};

template <>
struct TensorDataType<std::uint16_t>
{
    static constexpr DataType value = TYPE_UINT16;
};

template <>
struct TensorDataType<std::uint32_t>
{
    static constexpr DataType value = TYPE_UINT32;
};

template <>
struct TensorDataType<std::uint64_t>
{
    static constexpr DataType value = TYPE_UINT64;
};

#if !defined(_WIN32)
template <>
struct TensorDataType<unsigned long long>
{
    static constexpr DataType value = TYPE_UINT64;
};
#endif // !defined(_WIN32)

static_assert(sizeof(std::uint64_t) == sizeof(unsigned long long), "");

template <>
struct TensorDataType<std::int8_t>
{
    static constexpr DataType value = TYPE_INT8;
};

template <>
struct TensorDataType<std::int16_t>
{
    static constexpr DataType value = TYPE_INT16;
};

template <>
struct TensorDataType<std::int32_t>
{
    static constexpr DataType value = TYPE_INT32;
};

template <>
struct TensorDataType<std::int64_t>
{
    static constexpr DataType value = TYPE_INT64;
};

template <>
struct TensorDataType<half>
{
    static constexpr DataType value = TYPE_FP16;
};

template <>
struct TensorDataType<float>
{
    static constexpr DataType value = TYPE_FP32;
};

template <>
struct TensorDataType<double>
{
    static constexpr DataType value = TYPE_FP64;
};

template <>
struct TensorDataType<char>
{
    static constexpr DataType value = TYPE_BYTES;
};

#ifdef ENABLE_BF16
template <>
struct TensorDataType<__nv_bfloat16>
{
    static constexpr DataType value = TYPE_BF16;
};
#endif

#ifdef ENABLE_FP8
template <>
struct TensorDataType<__nv_fp8_e4m3>
{
    static constexpr DataType value = TYPE_FP8_E4M3;
};
#endif

template <>
struct TensorDataType<std::string>
{
    static constexpr DataType value = TYPE_STR;
};

template <>
struct TensorDataType<void>
{
    static constexpr DataType value = TYPE_VOID;
};

template <>
struct TensorDataType<int*>
{
    static constexpr DataType value = TYPE_INT32_PTR;
};

template <>
struct TensorDataType<const int*>
{
    static constexpr DataType value = TYPE_INT32_PTR;
};

template <typename T>
DataType getTensorType()
{
    return TensorDataType<typename std::remove_cv<T>::type>::value;
}

typedef enum memorytype_enum
{
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU
} MemoryType;

class Tensor
{
public:
    // Do not write to these variables directly. Use copy / move constructors instead.
    MemoryType where;
    DataType type;
    std::vector<size_t> shape;
    void const* data; // TODO modify from const void* to void* const

    Tensor();
    Tensor(MemoryType _where, DataType _type, std::vector<size_t> const& _shape, void const* _data);

    std::size_t size() const;
    std::size_t sizeBytes() const;

    std::string whereToString() const;
    std::string toString() const;
    std::string getNumpyTypeDesc(DataType type) const;

    static size_t getTypeSize(DataType type);

    template <typename T>
    inline T getVal(size_t index) const
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        TLLM_CHECK(where == MEMORY_CPU);
        TLLM_CHECK(data != nullptr);
        TLLM_CHECK_WITH_INFO(index < size(), "index is larger than buffer size");

        if (getTensorType<T>() != type)
        {
            TLLM_LOG_DEBUG("getVal with type %s, but data type is: %s", getNumpyTypeDesc(getTensorType<T>()).c_str(),
                getNumpyTypeDesc(type).c_str());
        }
        return ((T*) data)[index];
    }

    template <typename T>
    inline T getVal() const
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type)
        {
            TLLM_LOG_DEBUG("getVal with type %s, but data type is: %s", getNumpyTypeDesc(getTensorType<T>()).c_str(),
                getNumpyTypeDesc(type).c_str());
        }
        return getVal<T>(0);
    }

    template <typename T>
    inline T* getPtr() const
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type)
        {
            TLLM_LOG_DEBUG("getPtr with type %s, but data type is: %s", getNumpyTypeDesc(getTensorType<T>()).c_str(),
                getNumpyTypeDesc(type).c_str());
        }
        return (T*) data;
    }

    inline void* getPtrWithOffset(size_t offset) const
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        if (data == nullptr)
        {
            return (void*) data;
        }
        else
        {
            TLLM_CHECK_WITH_INFO(offset < size(), "offset is larger than buffer size");
            return (void*) ((char*) data + offset * Tensor::getTypeSize(type));
        }
    }

    template <typename T>
    inline T* getPtrWithOffset(size_t offset) const
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type)
        {
            TLLM_LOG_DEBUG("getVal with type %s, but data type is: %s", getNumpyTypeDesc(getTensorType<T>()).c_str(),
                getNumpyTypeDesc(type).c_str());
        }
        if (data == nullptr)
        {
            return (T*) data;
        }
        else
        {
            TLLM_CHECK_WITH_INFO(
                offset < size(), fmtstr("offset (%lu) is larger than buffer size (%lu)", offset, size()));
            return ((T*) data) + offset;
        }
    }

    template <typename T>
    T max() const
    {
        if (getTensorType<T>() != type)
        {
            TLLM_LOG_DEBUG("getVal with type %s, but data type is: %s", getNumpyTypeDesc(getTensorType<T>()).c_str(),
                getNumpyTypeDesc(type).c_str());
        }
        TLLM_CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
        TLLM_CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
            "max() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        size_t max_idx = 0;
        T max_val = getVal<T>(max_idx);
        for (size_t i = 1; i < size(); ++i)
        {
            T val = getVal<T>(i);
            if (val > max_val)
            {
                max_idx = i;
                max_val = val;
            }
        }
        return max_val;
    }

    template <typename T>
    T min() const
    {
        if (getTensorType<T>() != type)
        {
            TLLM_LOG_DEBUG("getVal with type %s, but data type is: %s", getNumpyTypeDesc(getTensorType<T>()).c_str(),
                getNumpyTypeDesc(type).c_str());
        }
        TLLM_CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
        TLLM_CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
            "min() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        size_t min_idx = 0;
        T min_val = getVal<T>(min_idx);
        for (size_t i = 1; i < size(); ++i)
        {
            T val = getVal<T>(i);
            if (val < min_val)
            {
                min_idx = i;
                min_val = val;
            }
        }
        return min_val;
    }

    template <typename T>
    T any(T val) const
    {
        if (getTensorType<T>() != type)
        {
            TLLM_LOG_DEBUG("getVal with type %s, but data type is: %s", getNumpyTypeDesc(getTensorType<T>()).c_str(),
                getNumpyTypeDesc(type).c_str());
        }
        TLLM_CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
        TLLM_CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
            "any() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        for (size_t i = 0; i < size(); ++i)
        {
            if (getVal<T>(i) == val)
            {
                return true;
            }
        }
        return false;
    }

    template <typename T>
    T all(T val) const
    {
        if (getTensorType<T>() != type)
        {
            TLLM_LOG_DEBUG("getVal with type %s, but data type is: %s", getNumpyTypeDesc(getTensorType<T>()).c_str(),
                getNumpyTypeDesc(type).c_str());
        }
        TLLM_CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
        TLLM_CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
            "all() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        for (size_t i = 0; i < size(); ++i)
        {
            if (getVal<T>(i) != val)
            {
                return false;
            }
        }
        return true;
    }

    void updateShape(size_t idx, size_t val)
    {
        // TODO: find a better way to update the shape
        std::vector<size_t>& shape_ref = const_cast<std::vector<size_t>&>(shape);
        shape_ref[idx] = val;
    }

    inline bool isValid() const
    {
        return size() > 0 && data != nullptr;
    }

    Tensor slice(std::vector<size_t> shape, size_t offset = 0) const;
};

class TensorMap
{
private:
    std::unordered_map<std::string, Tensor> tensor_map_;

public:
    TensorMap() = default;
    TensorMap(const std::unordered_map<std::string, Tensor>& tensor_map);
    TensorMap(const std::vector<Tensor>& tensor_map);
    TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensor_map);
    ~TensorMap();

    inline size_t size() const
    {
        return tensor_map_.size();
    }

    inline bool contains(const std::string& key) const
    {
        TLLM_LOG_TRACE("%s for key: %s", __PRETTY_FUNCTION__, key.c_str());
        return tensor_map_.find(key) != tensor_map_.end();
    }

    std::vector<std::string> keys() const;

    inline void insert(const std::string& key, const Tensor& value)
    {
        TLLM_CHECK_WITH_INFO(!contains(key), fmtstr("Duplicated key %s", key.c_str()));
        TLLM_CHECK_WITH_INFO(
            value.isValid(), fmtstr("A none tensor or nullptr is not allowed (key is %s)", key.c_str()));
        tensor_map_.insert({key, value});
    }

    inline void insertIfValid(const std::string& key, const Tensor& value)
    {
        if (value.isValid())
        {
            insert({key, value});
        }
    }

    inline void insert(std::pair<std::string, Tensor> p)
    {
        tensor_map_.insert(p);
    }

    // prevent converting int or size_t to string automatically
    Tensor at(int tmp) = delete;
    Tensor at(size_t tmp) = delete;

    inline Tensor& at(const std::string& key)
    {
        TLLM_LOG_TRACE("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        TLLM_CHECK_WITH_INFO(contains(key),
            fmtstr(
                "Cannot find a tensor of name %s in the tensor map (keys: %s)", key.c_str(), vec2str(keys()).c_str()));
        return tensor_map_.at(key);
    }

    inline Tensor at(const std::string& key) const
    {
        TLLM_CHECK_WITH_INFO(contains(key),
            fmtstr(
                "Cannot find a tensor of name %s in the tensor map (keys: %s)", key.c_str(), vec2str(keys()).c_str()));
        return tensor_map_.at(key);
    }

    inline std::optional<Tensor> atOpt(const std::string& key) const
    {
        if (contains(key))
            return tensor_map_.at(key);
        else
            return std::nullopt;
    }

    inline Tensor& at(const std::string& key, Tensor& default_tensor)
    {
        TLLM_LOG_TRACE("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        if (contains(key))
        {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    inline Tensor at(const std::string& key, Tensor& default_tensor) const
    {
        TLLM_LOG_TRACE("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        if (contains(key))
        {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    inline Tensor& at(const std::string& key, Tensor&& default_tensor)
    {
        TLLM_LOG_TRACE("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        if (contains(key))
        {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    inline Tensor at(const std::string& key, Tensor&& default_tensor) const
    {
        if (contains(key))
        {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    template <typename T>
    inline T getVal(const std::string& key) const
    {
        TLLM_CHECK_WITH_INFO(contains(key),
            fmtstr(
                "Cannot find a tensor of name %s in the tensor map (keys: %s)", key.c_str(), vec2str(keys()).c_str()));
        return tensor_map_.at(key).getVal<T>();
    }

    template <typename T>
    inline std::optional<T> getValOpt(const std::string& key) const
    {
        if (contains(key))
        {
            return tensor_map_.at(key).getVal<T>();
        }
        else
        {
            return std::nullopt;
        }
    }

    template <typename T>
    inline T getVal(const std::string& key, T default_value) const
    {
        if (contains(key))
        {
            return tensor_map_.at(key).getVal<T>();
        }
        return default_value;
    }

    template <typename T>
    inline T getValWithOffset(const std::string& key, size_t index) const
    {
        TLLM_CHECK_WITH_INFO(contains(key),
            fmtstr(
                "Cannot find a tensor of name %s in the tensor map (keys: %s)", key.c_str(), vec2str(keys()).c_str()));
        return tensor_map_.at(key).getVal<T>(index);
    }

    template <typename T>
    inline T getValWithOffset(const std::string& key, size_t index, T default_value) const
    {
        if (contains(key))
        {
            return tensor_map_.at(key).getVal<T>(index);
        }
        return default_value;
    }

    template <typename T>
    inline T* getPtr(const std::string& key) const
    {
        TLLM_CHECK_WITH_INFO(contains(key),
            fmtstr(
                "Cannot find a tensor of name %s in the tensor map (keys: %s)", key.c_str(), vec2str(keys()).c_str()));
        return tensor_map_.at(key).getPtr<T>();
    }

    template <typename T>
    inline T* getPtr(const std::string& key, T* default_ptr) const
    {
        if (contains(key))
        {
            return tensor_map_.at(key).getPtr<T>();
        }
        return default_ptr;
    }

    template <typename T>
    inline T* getPtrWithOffset(const std::string& key, size_t index) const
    {
        TLLM_CHECK_WITH_INFO(contains(key),
            fmtstr(
                "Cannot find a tensor of name %s in the tensor map (keys: %s)", key.c_str(), vec2str(keys()).c_str()));
        return tensor_map_.at(key).getPtrWithOffset<T>(index);
    }

    template <typename T>
    inline T* getPtrWithOffset(const std::string& key, size_t index, T* default_ptr) const
    {
        if (contains(key))
        {
            return tensor_map_.at(key).getPtrWithOffset<T>(index);
        }
        return default_ptr;
    }

    inline std::unordered_map<std::string, Tensor> getMap() const
    {
        return tensor_map_;
    }

    inline std::unordered_map<std::string, Tensor>::iterator begin()
    {
        return tensor_map_.begin();
    }

    inline std::unordered_map<std::string, Tensor>::iterator end()
    {
        return tensor_map_.end();
    }

    std::string toString();
};

} // namespace common
} // namespace tensorrt_llm
