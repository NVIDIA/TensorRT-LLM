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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::layers
{

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
    DebugTensor(runtime::ITensor const& tensor, char const* name,
        std::shared_ptr<runtime::BufferManager> bufferManager = nullptr,
        std::shared_ptr<runtime::CudaStream> stream = nullptr)
        : mTensor(tensor)
        , mName(name)
        , mBufferManager(bufferManager)
        , mStream(stream)
    {
    }

    DebugTensor(runtime::ITensor::SharedConstPtr tensor, char const* name,
        std::shared_ptr<runtime::BufferManager> bufferManager = nullptr,
        std::shared_ptr<runtime::CudaStream> stream = nullptr)
        : DebugTensor(*tensor, name, bufferManager, stream)
    {
        isEmpty = (tensor.get() == nullptr);
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

    runtime::BufferManager::ITensorPtr copyToHostOptional()
    {
        runtime::BufferManager::ITensorPtr hostPtr{nullptr};
        if (mTensor.getMemoryType() == runtime::MemoryType::kGPU)
        {
            auto theManager = mBufferManager
                ? mBufferManager
                : std::make_shared<runtime::BufferManager>(mStream ? mStream : std::make_shared<runtime::CudaStream>());
            hostPtr = theManager->copyFrom(mTensor, runtime::MemoryType::kCPU);
            theManager->getStream().synchronize();
        }
        return hostPtr;
    }

    std::string string(void)
    {
        runtime::BufferManager::ITensorPtr hostPtr = copyToHostOptional();
        runtime::BufferRange<runtime::TokenIdType const> range(hostPtr ? (*hostPtr) : mTensor);
        std::string result(range.size(), '\0');
        std::copy(range.begin(), range.end(), result.begin());
        return result;
    }

    template <typename T, bool IsTokens = false>
    std::string value()
    {
        using namespace tensorrt_llm::runtime;
        std::ostringstream buf;
        auto shape = mTensor.getShape();
        runtime::BufferManager::ITensorPtr hostPtr = copyToHostOptional();
        runtime::BufferRange<T const> tensorRange(hostPtr ? (*hostPtr) : mTensor);

        buf << mName << ": " << mTensor.getMemoryTypeName() << ',' << mTensor.getDataTypeName() << ',' << shape;
        auto line = [&buf](T const* array, SizeType32 size)
        {
            buf << '[';
            for (SizeType32 i = 0; i < size; i++)
            {
                if constexpr (std::is_same_v<T, runtime::TokenIdType> && IsTokens)
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
                }
                else
                {
                    buf << static_cast<unsigned long long>(array[i]);
                }
                if (i != size - 1)
                {
                    buf << ',';
                }
            }
            buf << ']';
        };
        switch (shape.nbDims)
        {
        case 0: buf << "[]"; break;
        case 1: line(tensorRange.begin(), shape.d[0]); break;
        case 2:
        {
            for (auto i = 0; i < shape.d[0]; i++)
            {
                buf << "\n [" << std::setw(3) << i << "]: ";
                line(tensorRange.begin() + i * shape.d[1], shape.d[1]);
            }
            break;
        }
        case 3:
        {
            for (auto i = 0; i < shape.d[0]; i++)
            {
                for (auto j = 0; j < shape.d[1]; j++)
                {
                    buf << "\n [" << std::setw(3) << i << "," << std::setw(3) << j << "]: ";
                    line(tensorRange.begin() + (i * shape.d[0] + j) * shape.d[1], shape.d[2]);
                }
            }
            break;
        }
        default: buf << "More than 3 dimensions";
        }
        return buf.str();
    }

    std::string value(bool const isTokens = false)
    {
        if (isEmpty)
        {
            return mName + " is empty";
        }
        switch (mTensor.getDataType())
        {
        case nvinfer1::DataType::kFLOAT: return value<float>();
        case nvinfer1::DataType::kHALF: return value<half>();
        case nvinfer1::DataType::kINT8: return value<std::int8_t>();
        case nvinfer1::DataType::kINT32: return isTokens ? value<std::int32_t, true>() : value<std::int32_t, false>();
        case nvinfer1::DataType::kBOOL: return value<bool>();
        case nvinfer1::DataType::kUINT8: return value<std::uint8_t>();
        // FP8 is not supported
        // BF16 is not supported
        case nvinfer1::DataType::kINT64: return value<std::int64_t>();
        // INT4 is not supported
        // FP4 is not supported
        default: return std::string(mName + ": Unsupported data type");
        }
    }

    std::string shape(void)
    {
        if (isEmpty)
        {
            return mName + " is empty";
        }
        using namespace tensorrt_llm::runtime;
        std::ostringstream buf;
        buf << mName << ": " << mTensor.getShape();
        return buf.str();
    }

    void print_value(bool const isTokens)
    {
        TLLM_LOG_DEBUG(value(isTokens));
    }

    void print_shape(void)
    {
        TLLM_LOG_DEBUG(shape());
    }

    template <typename T>
    void randomize(runtime::SizeType32 vtype)
    {
        runtime::BufferRange<T> tensorRange(const_cast<runtime::ITensor&>(mTensor));
        for (auto& item : tensorRange)
        {
            item = (vtype == 0 ? 0 : (vtype == 1 ? 1 : rand()));
        }
    }

    void randomize(void)
    {
        if (mTensor.getMemoryType() == runtime::MemoryType::kGPU)
        {
            runtime::ITensor& nonConstTensor = const_cast<runtime::ITensor&>(mTensor);
            runtime::BufferManager manager{std::make_shared<runtime::CudaStream>()};
            runtime::ITensor::SharedConstPtr cpuBuffer = manager.cpu(mTensor.getShape(), mTensor.getDataType());
            DebugTensor(cpuBuffer, "cpuBuffer").randomize();
            manager.copy(*cpuBuffer, nonConstTensor);
            manager.getStream().synchronize();
        }
        else
        {
            switch (mTensor.getDataType())
            {
            case nvinfer1::DataType::kBOOL: return randomize<bool>(3);
            case nvinfer1::DataType::kFLOAT: return randomize<float>(3);
            case nvinfer1::DataType::kINT8: return randomize<std::int8_t>(3);
            case nvinfer1::DataType::kINT32: return randomize<std::int32_t>(3);
            case nvinfer1::DataType::kINT64: return randomize<std::int64_t>(3);
            case nvinfer1::DataType::kUINT8: return randomize<std::uint8_t>(3);
            default: return;
            }
        }
    }

    void setZeros(void)
    {
        switch (mTensor.getDataType())
        {
        case nvinfer1::DataType::kBOOL: return randomize<bool>(0);
        case nvinfer1::DataType::kFLOAT: return randomize<float>(0);
        case nvinfer1::DataType::kINT8: return randomize<std::int8_t>(0);
        case nvinfer1::DataType::kINT32: return randomize<std::int32_t>(0);
        case nvinfer1::DataType::kINT64: return randomize<std::int64_t>(0);
        case nvinfer1::DataType::kUINT8: return randomize<std::uint8_t>(0);
        default: return;
        }
    }

    void setOnes(void)
    {
        switch (mTensor.getDataType())
        {
        case nvinfer1::DataType::kBOOL: return randomize<bool>(1);
        case nvinfer1::DataType::kFLOAT: return randomize<float>(1);
        case nvinfer1::DataType::kINT8: return randomize<std::int8_t>(1);
        case nvinfer1::DataType::kINT32: return randomize<std::int32_t>(1);
        case nvinfer1::DataType::kINT64: return randomize<std::int64_t>(1);
        case nvinfer1::DataType::kUINT8: return randomize<std::uint8_t>(1);
        default: return;
        }
    }

private:
    runtime::ITensor const& mTensor;
    std::string mName;
    std::shared_ptr<runtime::BufferManager> mBufferManager;
    std::shared_ptr<runtime::CudaStream> mStream;
    bool isEmpty{false};
};

#define D(x) tensorrt_llm::layers::DebugTensor(x, #x)
#define Db(x, bufferManager) tensorrt_llm::layers::DebugTensor(x, #x, bufferManager, nullptr)
#define Ds(x, stream) tensorrt_llm::layers::DebugTensor(x, #x, nullptr, stream)
#define Dbs(x, bufferManager, stream) tensorrt_llm::layers::DebugTensor(x, #x, bufferManager, stream)
#define PRINT_TOKEN(x) D(x).print_value(true)
#define PRINT_VALUE(x) D(x).print_value(false)
#define PRINT_SHAPE(x) D(x).print_shape()

} // namespace tensorrt_llm::layers
