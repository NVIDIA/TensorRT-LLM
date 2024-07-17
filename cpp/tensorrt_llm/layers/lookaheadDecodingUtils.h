/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tensorView.h"

namespace tensorrt_llm::layers
{

inline nvinfer1::DataType toTrtDataType(tensorrt_llm::common::DataType type)
{
    using namespace tensorrt_llm;
    switch (type)
    {
    case common::DataType::TYPE_FP32: return nvinfer1::DataType::kFLOAT;
    case common::DataType::TYPE_FP16: return nvinfer1::DataType::kHALF;
    case common::DataType::TYPE_BF16: return nvinfer1::DataType::kBF16;
    case common::DataType::TYPE_FP8_E4M3: return nvinfer1::DataType::kFP8;
    case common::DataType::TYPE_INT8: return nvinfer1::DataType::kINT8;
    case common::DataType::TYPE_UINT8: return nvinfer1::DataType::kUINT8;
    case common::DataType::TYPE_INT32: return nvinfer1::DataType::kINT32;
    case common::DataType::TYPE_INT64: return nvinfer1::DataType::kINT64;
    case common::DataType::TYPE_BOOL: return nvinfer1::DataType::kBOOL;
    default: TLLM_THROW("Unsupported data type: %d", static_cast<int>(type));
    }
}

inline runtime::ITensor::UniquePtr wrap(tensorrt_llm::common::Tensor tensor)
{
    using namespace tensorrt_llm::runtime;
    auto type = toTrtDataType(tensor.type);
    ITensor::Shape shape{};
    shape.nbDims = tensor.shape.size();
    std::copy(tensor.shape.begin(), tensor.shape.end(), shape.d);
    return ITensor::wrap(const_cast<void*>(tensor.data), type, shape);
}

template <typename T>
class BufferLocation : public runtime::BufferRange<T>
{
public:
    using typename runtime::BufferRange<T>::size_type;
    using runtime::BufferRange<T>::begin;
    using runtime::BufferRange<T>::operator[];

    BufferLocation(T* data, size_type size)
        : runtime::BufferRange<T>{data, size}
    {
    }

    template <typename U = T, std::enable_if_t<!std::is_const_v<U>, bool> = true>
    explicit BufferLocation(runtime::ITensor& tensor)
        : BufferLocation(runtime::bufferCast<U>(tensor), tensor.getSize())
    {
        mStrides = runtime::ITensor::strides(tensor.getShape());
    }

    template <typename U = T, std::enable_if_t<std::is_const_v<U>, bool> = true>
    explicit BufferLocation(runtime::ITensor const& tensor)
        : BufferLocation(runtime::bufferCast<U>(tensor), tensor.getSize())
    {
        mStrides = runtime::ITensor::strides(tensor.getShape());
    }

    inline T& at(runtime::ITensor::Shape const& dims)
    {
        return *ptr(dims);
    }

    inline T& at(std::initializer_list<runtime::ITensor::DimType64> const& dims)
    {
        return *ptr(dims);
    }

    template <typename... Args>
    inline T& at(Args... args)
    {
        runtime::ITensor::DimType64 offset = 0;
        runtime::ITensor::DimType64 dims = 0;
        atHelper(offset, dims, args...);
        return *(begin() + offset);
    }

    inline T& operator[](runtime::ITensor::Shape const& dims)
    {
        return *ptr(dims);
    }

    inline T& operator[](std::initializer_list<runtime::ITensor::DimType64> const& dims)
    {
        return *ptr(dims);
    }

    inline T* ptr(runtime::ITensor::Shape const& dims)
    {
        return begin() + offset(dims);
    }

    inline T* ptr(std::initializer_list<runtime::ITensor::DimType64> const& dims)
    {
        return ptr(runtime::ITensor::makeShape(dims));
    }

    runtime::ITensor::DimType64 offset(runtime::ITensor::Shape const& dims)
    {
        TLLM_CHECK(mStrides.nbDims == dims.nbDims);
        runtime::ITensor::DimType64 result = 0;
        for (runtime::ITensor::DimType64 di = 0; di < mStrides.nbDims; di++)
        {
            result += dims.d[di] * mStrides.d[di];
        }
        return result;
    }

    runtime::ITensor::DimType64 offset(std::initializer_list<runtime::ITensor::DimType64> const& dims)
    {
        return offset(runtime::ITensor::makeShape(dims));
    }

private:
    inline void atHelper(runtime::ITensor::DimType64& offset, runtime::ITensor::DimType64& dims) {}

    template <typename... Args>
    inline void atHelper(runtime::ITensor::DimType64& offset, runtime::ITensor::DimType64& dims, int dim, Args... args)
    {
        offset += dim * mStrides.d[dims++];
        atHelper(offset, dims, args...);
    }

private:
    runtime::ITensor::Shape mStrides;
};

class DebugTensor
{
public:
    DebugTensor(runtime::ITensor const& tensor, char const* name)
        : mTensor(tensor)
        , mName(name)
    {
    }

    DebugTensor(runtime::ITensor::SharedConstPtr tensor, char const* name)
        : DebugTensor(*tensor, name)
    {
    }

    uint8_t const& u8(std::initializer_list<runtime::ITensor::DimType64> const& dims)
    {
        return (BufferLocation<uint8_t const>(mTensor))[dims];
    }

    uint8_t const& u8(int32_t idx)
    {
        return (BufferLocation<uint8_t const>(mTensor))[idx];
    }

    int8_t const& i8(std::initializer_list<runtime::ITensor::DimType64> const& dims)
    {
        return (BufferLocation<int8_t const>(mTensor))[dims];
    }

    int8_t const& i8(int32_t idx)
    {
        return (BufferLocation<int8_t const>(mTensor))[idx];
    }

    int32_t const& i32(std::initializer_list<runtime::ITensor::DimType64> const& dims)
    {
        return (BufferLocation<int32_t const>(mTensor))[dims];
    }

    int32_t const& i32(int32_t idx)
    {
        return (BufferLocation<int32_t const>(mTensor))[idx];
    }

    int64_t const& i64(std::initializer_list<runtime::ITensor::DimType64> const& dims)
    {
        return (BufferLocation<int64_t const>(mTensor))[dims];
    }

    int64_t const& i64(int32_t idx)
    {
        return (BufferLocation<int64_t const>(mTensor))[idx];
    }

    float const& f(std::initializer_list<runtime::ITensor::DimType64> const& dims)
    {
        return (BufferLocation<float const>(mTensor))[dims];
    }

    float const& f(int32_t idx)
    {
        return (BufferLocation<float const>(mTensor))[idx];
    }

    std::string string(void)
    {
        runtime::BufferRange<runtime::TokenIdType const> range(mTensor);
        std::string result(range.size(), '\0');
        std::copy(range.begin(), range.end(), result.begin());
        return result;
    }

    std::string tokens(void)
    {
        using namespace tensorrt_llm::runtime;
        std::ostringstream buf;
        auto shape = mTensor.getShape();
        runtime::BufferRange<TokenIdType const> tensorRange(mTensor);
        buf << mName << ": " << shape;
        auto line = [&buf](TokenIdType const* array, SizeType32 size)
        {
            buf << '[';
            for (SizeType32 i = 0; i < size; i++)
            {
                auto token = array[i];
                if (token >= ' ' && token <= '~')
                {
                    buf << '\'' << static_cast<char>(token) << '\'';
                }
                else
                {
                    buf << token;
                }
                if (i != size - 1)
                {
                    buf << ',';
                }
            }
            buf << ']';
        };
        if (shape.nbDims == 0)
        {
            buf << "[]";
        }
        else if (shape.nbDims == 1)
        {
            line(tensorRange.begin(), shape.d[0]);
        }
        else if (shape.nbDims == 2)
        {
            buf << '[';
            for (runtime::SizeType32 i = 0; i < shape.d[0]; i++)
            {
                buf << "\n " << i << ": ";
                line(tensorRange.begin() + i * shape.d[1], shape.d[1]);
            }
            buf << ']';
        }
        else
        {
            buf << "Too Large to be printed";
        }
        return buf.str();
    }

    template <typename T>
    std::string values(void)
    {
        using namespace tensorrt_llm::runtime;
        std::ostringstream buf;
        auto shape = mTensor.getShape();
        runtime::BufferRange<T const> tensorRange(mTensor);
        buf << mName << ": " << shape;
        auto line = [&buf](T const* array, SizeType32 size)
        {
            buf << '[';
            for (SizeType32 i = 0; i < size; i++)
            {
                buf << array[i];
                if (i != size - 1)
                {
                    buf << ',';
                }
            }
            buf << ']';
        };
        if (shape.nbDims == 0)
        {
            buf << "[]";
        }
        else if (shape.nbDims == 1)
        {
            line(tensorRange.begin(), shape.d[0]);
        }
        else if (shape.nbDims == 2)
        {
            buf << '[';
            for (runtime::SizeType32 i = 0; i < shape.d[0]; i++)
            {
                buf << "\n " << i << ": ";
                line(tensorRange.begin() + i * shape.d[1], shape.d[1]);
            }
            buf << ']';
        }
        else
        {
            buf << "Too Large to be printed";
        }
        return buf.str();
    }

    std::string values(void)
    {
        switch (mTensor.getDataType())
        {
        case nvinfer1::DataType::kBOOL: return values<bool>();
        case nvinfer1::DataType::kFLOAT: return values<float>();
        case nvinfer1::DataType::kINT8: return values<std::int8_t>();
        case nvinfer1::DataType::kINT32: return values<std::int32_t>();
        case nvinfer1::DataType::kINT64: return values<std::int64_t>();
        case nvinfer1::DataType::kUINT8: return values<std::uint8_t>();
        default: return std::string(mName + ": Unsupported data type");
        }
    }

    std::string shape(void)
    {
        using namespace tensorrt_llm::runtime;
        std::ostringstream buf;
        buf << mName << ": " << mTensor.getShape();
        return buf.str();
    }

    void print_tokens(void)
    {
        TLLM_LOG_DEBUG(tokens());
    }

    void print_values(void)
    {
        TLLM_LOG_DEBUG(values());
    }

    void print_shape(void)
    {
        TLLM_LOG_DEBUG(shape());
    }

private:
    runtime::ITensor const& mTensor;
    std::string mName;
};

#define D(x) tensorrt_llm::layers::DebugTensor(x, #x)
#define PRINT_TOKENS(x) D(x).print_tokens()
#define PRINT_VALUES(x) D(x).print_values()
#define PRINT_SHAPE(x) D(x).print_shape()

} // namespace tensorrt_llm::layers
