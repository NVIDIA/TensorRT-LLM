/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <NvInferRuntime.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace nvinfer1
{
class IExecutionContext;
}

namespace tensorrt_llm::runtime
{

class ITensor : virtual public IBuffer
{
public:
    using UniquePtr = std::unique_ptr<ITensor>;
    using SharedPtr = std::shared_ptr<ITensor>;
    using UniqueConstPtr = std::unique_ptr<ITensor const>;
    using SharedConstPtr = std::shared_ptr<ITensor const>;
    using Shape = nvinfer1::Dims;
    using DimType = std::remove_reference_t<decltype(Shape::d[0])>;

    ~ITensor() override = default;

    //!
    //! \brief Returns the tensor dimensions.
    //!
    [[nodiscard]] virtual Shape const& getShape() const = 0;

    //!
    //! \brief Sets the tensor dimensions. The new size of the tensor will be `volume(dims)`
    //!
    virtual void reshape(Shape const& dims) = 0;

    void resize(std::size_t newSize) override
    {
        if (newSize == getSize())
            return;

        reshape(makeShape({castSize(newSize)}));
    }

    //!
    //! \brief Not allowed to copy.
    //!
    ITensor(ITensor const&) = delete;

    //!
    //! \brief Not allowed to copy.
    //!
    ITensor& operator=(ITensor const&) = delete;

    //!
    //! \brief Returns the volume of the dimensions. Returns -1 if `d.nbDims < 0`.
    //!
    static std::int64_t volume(Shape const& dims)
    {
        {
            return dims.nbDims < 0 ? -1
                : dims.nbDims == 0
                ? 0
                : std::accumulate(dims.d, dims.d + dims.nbDims, std::int64_t{1}, std::multiplies<>{});
        }
    }

    //!
    //! \brief Returns the volume of the dimensions. Throws if `d.nbDims < 0`.
    //!
    static std::size_t volumeNonNegative(Shape const& shape)
    {
        auto const vol = volume(shape);
        TLLM_CHECK_WITH_INFO(0 <= vol, "Invalid tensor shape");
        return static_cast<std::size_t>(vol);
    }

    //!
    //! \brief Removes the given *unit* dimension from `shape`.
    //!
    //! \param shape The shape to squeeze.
    //! \param dim The dimension that should be removed ("squeezed").
    //! \return A new shape without the unit dimension.
    //!
    static Shape squeeze(Shape const& shape, SizeType dim);

    //!
    //! \brief Add a *unit* dimension to `shape` at the specified position.
    //!
    //! \param shape The shape to unsqueeze.
    //! \param dim The dimension where unit dimension should be added.
    //! \return A new shape with the added unit dimension.
    //!
    static Shape unsqueeze(Shape const& shape, SizeType dim);

    //!
    //! \brief Removes the given *unit* dimensions from this tensor.
    //!
    void squeeze(SizeType dim)
    {
        reshape(squeeze(getShape(), dim));
    }

    //!
    //! \brief Adds a *unit* dimension at the specified position
    //!
    void unsqueeze(SizeType dim)
    {
        reshape(unsqueeze(getShape(), dim));
    }

    //!
    //! \brief Creates a sliced view on the underlying `tensor`. The view will have the same data type as `tensor`.
    //!
    //! \param tensor The tensor to view.
    //! \param offset The offset of the view w.r.t. dimension 0 of the tensor.
    //! \param size The size of the view w.r.t. dimension 0 of the tensor.
    //! \return A view on the `buffer`.
    //!
    static UniquePtr slice(SharedPtr tensor, std::size_t offset, std::size_t size);

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(TConstPtr&& tensor, std::size_t offset, std::size_t size)
    {
        return ITensor::slice(constPointerCast(std::forward<TConstPtr>(tensor)), offset, size);
    }

    static UniquePtr slice(SharedPtr tensor, std::size_t offset)
    {
        auto const dims = tensor->getShape();
        auto const size = (dims.nbDims > 0 ? dims.d[0] : 0) - offset;
        return ITensor::slice(std::move(tensor), offset, size);
    }

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(TConstPtr&& tensor, std::size_t offset)
    {
        return ITensor::slice(constPointerCast(std::forward<TConstPtr>(tensor)), offset);
    }

    //!
    //! \brief Returns a view on the underlying `buffer` (or tensor) with the given shape.
    //!
    //! \param tensor The tensor to view.
    //! \param shape The shape of the view.
    //! \return A view on the `tensor`.
    //!
    static UniquePtr view(IBuffer::SharedPtr buffer, Shape const& dims);

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr view(TConstPtr&& tensor, Shape const& dims)
    {
        return ITensor::view(constPointerCast(std::forward<TConstPtr>(tensor)), dims);
    }

    //!
    //! \brief Returns a view on the underlying `tensor` which can be independently reshaped.
    //!
    //! \param tensor The tensor to view.
    //! \return A view on the `tensor`.
    //!
    static UniquePtr view(SharedPtr tensor)
    {
        auto shapes = tensor->getShape();
        return ITensor::view(std::move(tensor), shapes);
    }

    //!
    //! \brief Wraps the given `data` in an `ITensor`. The `ITensor` will not own the underlying `data` and cannot
    //! be reshaped beyond `capacity`.
    //!
    //! \param data The data to wrap.
    //! \param type The data type of the `data`.
    //! \param shape The shape of the tensor.
    //! \param capacity The capacity of the buffer.
    //! \return An `ITensor`.
    static UniquePtr wrap(void* data, nvinfer1::DataType type, Shape const& shape, std::size_t capacity);

    static UniquePtr wrap(void* data, nvinfer1::DataType type, Shape const& shape)
    {
        return wrap(data, type, shape, volumeNonNegative(shape));
    }

    template <typename T>
    static UniquePtr wrap(T* data, Shape const& shape, std::size_t capacity)
    {
        return wrap(data, TRTDataType<T>::value, shape, capacity);
    }

    template <typename T>
    static UniquePtr wrap(T* data, Shape const& shape)
    {
        return wrap<T>(data, shape, volumeNonNegative(shape));
    }

    template <typename T>
    static UniquePtr wrap(std::vector<T>& v, Shape const& shape)
    {
        return wrap<T>(v.data(), shape, v.capacity());
    }

    //!
    //! \brief A convenience function to create a tensor shape with the given dimensions.
    //!
    static Shape makeShape(std::initializer_list<SizeType> const& dims);

    //!
    //! \brief A convenience function for converting a tensor shape to a `string`.
    //!
    static std::string toString(Shape const& dims);

    //!
    //! \brief A convenience function to compare shapes.
    //!
    static bool shapeEquals(Shape const& lhs, Shape const& rhs)
    {
        return shapeEquals(lhs, rhs.d, rhs.nbDims);
    }

    //!
    //! \brief A convenience function to compare shapes.
    //!
    template <typename T>
    static bool shapeEquals(Shape const& lhs, T const* dims, SizeType count)
    {
        return lhs.nbDims == count && std::equal(lhs.d, lhs.d + lhs.nbDims, dims);
    }

    bool shapeEquals(Shape const& other) const
    {
        return shapeEquals(getShape(), other);
    }

    bool shapeEquals(std::initializer_list<SizeType> const& other) const
    {
        return shapeEquals(getShape(), other.begin(), other.size());
    }

    template <typename T>
    bool shapeEquals(T const* dims, SizeType count) const
    {
        return shapeEquals(getShape(), dims, count);
    }

protected:
    ITensor() = default;

    static DimType castSize(size_t newSize)
    {
        TLLM_CHECK_WITH_INFO(
            newSize <= std::numeric_limits<DimType>::max(), "New size is too large. Use reshape() instead.");
        return static_cast<DimType>(newSize);
    }
};

//! \brief Utility function to print a shape.
inline std::ostream& operator<<(std::ostream& output, ITensor::Shape const& dims)
{
    return output << ITensor::toString(dims);
}

//! \brief Utility function to print a tensor with its shape.
std::ostream& operator<<(std::ostream& output, ITensor const& tensor);

} // namespace tensorrt_llm::runtime
