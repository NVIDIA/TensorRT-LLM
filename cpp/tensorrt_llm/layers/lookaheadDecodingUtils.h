#pragma once

#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tensorView.h"

namespace tensorrt_llm::layers
{

//! Syntax sugar for a tensor being squeezed as an argument.
inline runtime::ITensor::SharedPtr squeezed(runtime::ITensor::SharedPtr tensor, runtime::SizeType32 dim = 0)
{
    TLLM_CHECK(tensor->getShape().d[dim] == 1);
    tensor->squeeze(dim);
    return tensor;
}

runtime::ITensor::UniquePtr slice(runtime::ITensor::SharedPtr tensor,
    std::initializer_list<runtime::SizeType32> const& offsetDims, size_t const sizeDim);

template <typename T>
class BufferLocation : public runtime::BufferRange<T>
{
public:
    using runtime::BufferRange<T>::begin;

    BufferLocation(runtime::ITensor& t)
        : runtime::BufferRange<T>(t)
        , volumes(t.getShape().nbDims)
    {
        auto shape = t.getShape();
        TLLM_CHECK(shape.nbDims > 0);
        volumes[shape.nbDims - 1] = 1;
        for (runtime::SizeType32 i = shape.nbDims - 2; i >= 0; i--)
        {
            volumes[i] = shape.d[i + 1] * volumes[i + 1];
        }
    }

    T& operator()(std::initializer_list<runtime::SizeType32> const& dims)
    {
        return *ptr(dims);
    }

    T* ptr(std::initializer_list<runtime::SizeType32> const& dims)
    {
        return begin() + idx(dims);
    }

    runtime::SizeType32 idx(std::initializer_list<runtime::SizeType32> const& dims)
    {
        TLLM_CHECK(volumes.size() == dims.size());
        runtime::SizeType32 offset = 0;
        auto itd = dims.begin();
        auto itv = volumes.begin();
        for (; itd != dims.end() && itv != volumes.end(); itd++, itv++)
        {
            offset += (*itd) * (*itv);
        }
        return offset;
    }

private:
    std::vector<runtime::SizeType32> volumes;
};

class DebugTensor
{
public:
    DebugTensor(runtime::ITensor& tensor, char const* name)
        : mTensor(tensor)
        , mName(name)
    {
    }

    DebugTensor(runtime::ITensor::SharedPtr tensor, char const* name)
        : DebugTensor(*tensor, name)
    {
    }

    uint8_t& u8(std::initializer_list<runtime::SizeType32> const& dims)
    {
        return (BufferLocation<uint8_t>(mTensor))(dims);
    }

    uint8_t& u8(int32_t idx)
    {
        return (BufferLocation<uint8_t>(mTensor))[idx];
    }

    int8_t& i8(std::initializer_list<runtime::SizeType32> const& dims)
    {
        return (BufferLocation<int8_t>(mTensor))(dims);
    }

    int8_t& i8(int32_t idx)
    {
        return (BufferLocation<int8_t>(mTensor))[idx];
    }

    int32_t& i32(std::initializer_list<runtime::SizeType32> const& dims)
    {
        return (BufferLocation<int32_t>(mTensor))(dims);
    }

    int32_t& i32(int32_t idx)
    {
        return (BufferLocation<int32_t>(mTensor))[idx];
    }

    int64_t& i64(std::initializer_list<runtime::SizeType32> const& dims)
    {
        return (BufferLocation<int64_t>(mTensor))(dims);
    }

    int64_t& i64(int32_t idx)
    {
        return (BufferLocation<int64_t>(mTensor))[idx];
    }

    float& f(std::initializer_list<runtime::SizeType32> const& dims)
    {
        return (BufferLocation<float>(mTensor))(dims);
    }

    float& f(int32_t idx)
    {
        return (BufferLocation<float>(mTensor))[idx];
    }

    std::string string(void)
    {
        runtime::BufferRange<runtime::TokenIdType> range(mTensor);
        std::string result(range.size(), '\0');
        std::copy(range.begin(), range.end(), result.begin());
        return result;
    }

    std::string tokens(void)
    {
        using namespace tensorrt_llm::runtime;
        std::ostringstream buf;
        auto shape = mTensor.getShape();
        BufferRange<TokenIdType> tensorRange(mTensor);
        buf << mName << ": " << shape;
        auto line = [&buf](TokenIdType* array, SizeType32 size)
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
                buf << (i == size - 1 ? ']' : ',');
            }
        };
        if (shape.nbDims == 1)
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
        BufferRange<T> tensorRange(mTensor);
        buf << mName << ": " << shape;
        auto line = [&buf](T* array, SizeType32 size)
        {
            buf << '[';
            for (SizeType32 i = 0; i < size; i++)
            {
                buf << array[i] << (i == size - 1 ? ']' : ',');
            }
        };
        if (shape.nbDims == 1)
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

    void print_tokens(void)
    {
        TLLM_LOG_DEBUG(tokens());
    }

    void print_values(void)
    {
        switch (mTensor.getDataType())
        {
        case nvinfer1::DataType::kFLOAT: TLLM_LOG_DEBUG(values<float>()); break;
        case nvinfer1::DataType::kINT8: TLLM_LOG_DEBUG(values<std::int8_t>()); break;
        case nvinfer1::DataType::kINT32: TLLM_LOG_DEBUG(values<std::int32_t>()); break;
        case nvinfer1::DataType::kINT64: TLLM_LOG_DEBUG(values<std::int64_t>()); break;
        case nvinfer1::DataType::kUINT8: TLLM_LOG_DEBUG(values<std::uint8_t>()); break;
        }
    }

private:
    runtime::ITensor& mTensor;
    std::string mName;
};

#define D(x) tensorrt_llm::layers::DebugTensor(x, #x)
#define PRINT_TOKENS(x) D(x).print_tokens()
#define PRINT_TENSOR(x) D(x).print_values()
#define PRINT_VALUES(x) D(x).print_values()

} // namespace tensorrt_llm::layers
