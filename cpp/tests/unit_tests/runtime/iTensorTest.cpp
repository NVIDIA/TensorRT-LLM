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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

using namespace tensorrt_llm::runtime;

TEST(ITensorTest, SqueezeTensor)
{
    auto dims = ITensor::makeShape({16, 1, 4});
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    ITensor::SharedPtr tensor{BufferManager::cpu(dims, dataType)};

    auto squeezeDim = 0;
    EXPECT_THROW(tensor->squeeze(squeezeDim), std::runtime_error);
    squeezeDim = 1;
    auto squeezed = ITensor::view(tensor, ITensor::squeeze(dims, squeezeDim));

    EXPECT_EQ(squeezed->getSize(), tensor->getSize());
    EXPECT_EQ(squeezed->getShape().nbDims, tensor->getShape().nbDims - 1);
    EXPECT_EQ(squeezed->getShape().d[0], tensor->getShape().d[0]);
    EXPECT_EQ(squeezed->getShape().d[1], tensor->getShape().d[2]);

    EXPECT_NO_THROW(squeezed->release());
    EXPECT_EQ(squeezed->data(), nullptr);
    EXPECT_NE(tensor->data(), nullptr);
}

TEST(ITensorTest, UnsqueezeShape)
{
    auto oldShape = ITensor::makeShape({2, 3, 4, 5});
    {
        auto shape = ITensor::unsqueeze(oldShape, 0);

        EXPECT_EQ(shape.nbDims, 5);
        EXPECT_EQ(shape.d[0], 1);
        EXPECT_EQ(shape.d[1], 2);
        EXPECT_EQ(shape.d[2], 3);
        EXPECT_EQ(shape.d[3], 4);
        EXPECT_EQ(shape.d[4], 5);
    }
    {
        auto shape = ITensor::unsqueeze(oldShape, 1);

        EXPECT_EQ(shape.nbDims, 5);
        EXPECT_EQ(shape.d[0], 2);
        EXPECT_EQ(shape.d[1], 1);
        EXPECT_EQ(shape.d[2], 3);
        EXPECT_EQ(shape.d[3], 4);
        EXPECT_EQ(shape.d[4], 5);
    }

    {
        auto shape = ITensor::unsqueeze(oldShape, 4);

        EXPECT_EQ(shape.nbDims, 5);
        EXPECT_EQ(shape.d[0], 2);
        EXPECT_EQ(shape.d[1], 3);
        EXPECT_EQ(shape.d[2], 4);
        EXPECT_EQ(shape.d[3], 5);
        EXPECT_EQ(shape.d[4], 1);
    }

    std::vector<int> invalidDims{-1, 5, 10};
    for (auto invalidDim : invalidDims)
    {
        try
        {
            ITensor::unsqueeze(oldShape, invalidDim);
            FAIL() << "Expected failure";
        }
        catch (tensorrt_llm::common::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("Invalid dim"));
        }
        catch (...)
        {
            FAIL() << "Expected TllmException";
        }
    }
}

TEST(ITensorTest, UnsqueezeTensor)
{
    auto oldShape = ITensor::makeShape({2, 3, 4, 5});

    {
        auto tensor = BufferManager::cpu(oldShape, nvinfer1::DataType::kINT32);
        tensor->unsqueeze(0);
        auto shape = tensor->getShape();

        EXPECT_EQ(shape.nbDims, 5);
        EXPECT_EQ(shape.d[0], 1);
        EXPECT_EQ(shape.d[1], 2);
        EXPECT_EQ(shape.d[2], 3);
        EXPECT_EQ(shape.d[3], 4);
        EXPECT_EQ(shape.d[4], 5);
    }
    {
        auto tensor = BufferManager::cpu(oldShape, nvinfer1::DataType::kINT32);
        tensor->unsqueeze(1);
        auto shape = tensor->getShape();

        EXPECT_EQ(shape.nbDims, 5);
        EXPECT_EQ(shape.d[0], 2);
        EXPECT_EQ(shape.d[1], 1);
        EXPECT_EQ(shape.d[2], 3);
        EXPECT_EQ(shape.d[3], 4);
        EXPECT_EQ(shape.d[4], 5);
    }

    {
        auto tensor = BufferManager::cpu(oldShape, nvinfer1::DataType::kINT32);
        tensor->unsqueeze(4);
        auto shape = tensor->getShape();

        EXPECT_EQ(shape.nbDims, 5);
        EXPECT_EQ(shape.d[0], 2);
        EXPECT_EQ(shape.d[1], 3);
        EXPECT_EQ(shape.d[2], 4);
        EXPECT_EQ(shape.d[3], 5);
        EXPECT_EQ(shape.d[4], 1);
    }

    std::vector<int> invalidDims{-1, 5, 10};
    for (auto invalidDim : invalidDims)
    {
        try
        {
            auto tensor = BufferManager::cpu(oldShape, nvinfer1::DataType::kINT32);
            tensor->unsqueeze(invalidDim);
            FAIL() << "Expected failure";
        }
        catch (tensorrt_llm::common::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("Invalid dim"));
        }
        catch (...)
        {
            FAIL() << "Expected TllmException";
        }
    }
}

TEST(ITensorTest, TensorView)
{
    auto const dims = ITensor::makeShape({16, 1, 4});
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    ITensor::SharedPtr tensor = BufferManager::cpu(dims, dataType);

    auto const viewDims = ITensor::makeShape({16, 1, 2});

    auto view = ITensor::view(tensor, viewDims);
    EXPECT_EQ(view->getSize(), tensor->getSize() / 2);
    EXPECT_EQ(view->getShape().nbDims, tensor->getShape().nbDims);
    EXPECT_EQ(view->getShape().d[2], tensor->getShape().d[2] / 2);

    EXPECT_NO_THROW(view->release());
    EXPECT_EQ(view->data(), nullptr);
    EXPECT_NE(tensor->data(), nullptr);
}

TEST(ITensorTest, TensorSlice)
{
    auto dims = ITensor::makeShape({16, 8, 4});
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    ITensor::SharedPtr tensor{BufferManager::cpu(dims, dataType)};
    auto offset = dims.d[0] / 4;
    auto slice = ITensor::slice(tensor, offset);
    auto const sizeSlice = 3 * tensor->getSize() / 4;
    EXPECT_EQ(slice->getShape().d[0], dims.d[0] - offset);
    EXPECT_EQ(slice->getSize(), sizeSlice);
    EXPECT_EQ(slice->getCapacity(), sizeSlice);
    EXPECT_EQ(static_cast<std::uint8_t*>(slice->data()) - static_cast<std::uint8_t*>(tensor->data()),
        offset * ITensor::volume(dims) / dims.d[0] * BufferDataType(dataType).getSize());

    auto dimsNew = ITensor::makeShape({12, 32});
    EXPECT_EQ(ITensor::volume(dimsNew), sizeSlice);
    EXPECT_NO_THROW(slice->reshape(dimsNew));
    EXPECT_EQ(slice->getShape().d[1], dimsNew.d[1]);
    dimsNew.d[0] = 6;
    EXPECT_LT(ITensor::volume(dimsNew), sizeSlice);
    EXPECT_NO_THROW(slice->reshape(dimsNew));
    EXPECT_EQ(slice->getShape().d[0], dimsNew.d[0]);
    dimsNew.d[0] = 16;
    EXPECT_GT(ITensor::volume(dimsNew), sizeSlice);
    EXPECT_THROW(slice->reshape(dimsNew), std::runtime_error);

    EXPECT_NO_THROW(slice->resize(sizeSlice));
    EXPECT_NO_THROW(slice->resize(sizeSlice / 2));
    EXPECT_EQ(slice->getShape().d[0], sizeSlice / 2);
    EXPECT_THROW(slice->resize(sizeSlice * 2), std::runtime_error);
    EXPECT_NO_THROW(slice->release());
    EXPECT_EQ(slice->data(), nullptr);
    EXPECT_NE(tensor->data(), nullptr);

    std::shared_ptr<ITensor const> constTensor{tensor};
    auto constSlice = ITensor::slice(constTensor, offset);
    EXPECT_EQ(constSlice->getShape().d[0], dims.d[0] - offset);
    auto uniqueSlice = ITensor::slice(std::move(constSlice), 1);
    EXPECT_EQ(uniqueSlice->getShape().d[0], dims.d[0] - offset - 1);
}

TEST(ITensorTest, TensorDimsSliceAtManual)
{
    auto shape = ITensor::makeShape({5, 5, 5, 5, 5});
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    ITensor::SharedPtr tensor(BufferManager::cpu(shape, dataType));
    auto offsetDims = ITensor::makeShape({4, 3, 3});
    auto sizeDim = 2;
    auto sliced = ITensor::slice(tensor, offsetDims, sizeDim);
    EXPECT_TRUE(sliced->shapeEquals({2, 5, 5}));

    auto getVolume = [](std::initializer_list<ITensor::DimType64> const&& dims)
    { return ITensor::volume(ITensor::makeShape(dims)); };
    auto slicedVolume = ITensor::volume(sliced->getShape());
    auto offset = 4 * getVolume({5, 5}) + 3 * getVolume({5}) + 3;
    EXPECT_EQ(static_cast<std::uint8_t*>(sliced->data()) - static_cast<std::uint8_t*>(tensor->data()),
        offset * (slicedVolume / sizeDim) * BufferDataType(dataType).getSize());

    EXPECT_EQ(ITensor::volume(shape), getVolume({5, 5, 5}) * (slicedVolume / sizeDim));

    EXPECT_THROW(ITensor::slice(tensor, {5, 5}, 2), std::runtime_error);

    EXPECT_THROW(ITensor::slice(tensor, {4, 3, 4}, 3), std::runtime_error);

    sliced = ITensor::slice(tensor, {4, 3, 3}, 0);
    EXPECT_TRUE(sliced->shapeEquals({0, 5, 5}));

    sliced = ITensor::slice(tensor, {3});
    EXPECT_TRUE(sliced->shapeEquals({2, 5, 5, 5, 5}));

    sliced = ITensor::slice(tensor, {4, 3, 3});
    EXPECT_TRUE(sliced->shapeEquals({2, 5, 5}));

    auto theOne = ITensor::at(tensor, ITensor::makeShape({4, 3, 3}));
    EXPECT_TRUE(theOne->shapeEquals({5, 5}));

    theOne = ITensor::at(tensor, {4, 3});
    EXPECT_TRUE(theOne->shapeEquals({5, 5, 5}));

    theOne = ITensor::at(tensor, {4, 4, 4, 4, 4});
    EXPECT_TRUE(theOne->shapeEquals({1}));

    ITensor::SharedConstPtr constTensor = tensor;

    auto constSliced = ITensor::slice(constTensor, {4, 3, 3}, 0);
    EXPECT_TRUE(constSliced->shapeEquals({0, 5, 5}));

    constSliced = ITensor::slice(tensor, {1});
    EXPECT_TRUE(constSliced->shapeEquals({4, 5, 5, 5, 5}));

    constSliced = ITensor::slice(tensor, {4, 3, 2});
    EXPECT_TRUE(constSliced->shapeEquals({3, 5, 5}));

    auto theConstOne = ITensor::at(constTensor, ITensor::makeShape({4, 3, 3}));
    EXPECT_TRUE(theConstOne->shapeEquals({5, 5}));

    theConstOne = ITensor::at(constTensor, {4, 3});
    EXPECT_TRUE(theConstOne->shapeEquals({5, 5, 5}));

    theConstOne = ITensor::at(constTensor, {4, 4, 4, 4, 4});
    EXPECT_TRUE(theConstOne->shapeEquals({1}));
}

TEST(ITensorTest, TensorDimsSliceAtExtrame)
{
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    {
        auto shape = ITensor::makeShape({5, 5, 5, 5, 5});
        ITensor::SharedPtr tensor(BufferManager::cpu(shape, dataType));

        EXPECT_TRUE(ITensor::slice(tensor, {}, 0)->shapeEquals(ITensor::makeShape({0, 5, 5, 5, 5, 5})));
        EXPECT_TRUE(ITensor::slice(tensor, {}, 1)->shapeEquals(ITensor::makeShape({1, 5, 5, 5, 5, 5})));
        EXPECT_TRUE(ITensor::slice(tensor, {5}, 0)->shapeEquals(ITensor::makeShape({0, 5, 5, 5, 5})));
        EXPECT_TRUE(ITensor::slice(tensor, {4, 5}, 0)->shapeEquals(ITensor::makeShape({0, 5, 5, 5})));
        EXPECT_TRUE(ITensor::slice(tensor, {4, 4, 5}, 0)->shapeEquals(ITensor::makeShape({0, 5, 5})));
        EXPECT_TRUE(ITensor::slice(tensor, {4, 4, 4, 5}, 0)->shapeEquals(ITensor::makeShape({0, 5})));
        EXPECT_TRUE(ITensor::slice(tensor, {4, 4, 4, 4, 5}, 0)->shapeEquals(ITensor::makeShape({0})));
        EXPECT_TRUE(ITensor::slice(tensor, {4, 4, 4, 4, 4}, 0)->shapeEquals(ITensor::makeShape({0})));
        EXPECT_TRUE(ITensor::slice(tensor, {4, 4, 4, 4, 4}, 1)->shapeEquals(ITensor::makeShape({1})));
        EXPECT_THROW(ITensor::slice(tensor, {}, 2), std::runtime_error);

        EXPECT_TRUE(ITensor::at(tensor, {})->shapeEquals(ITensor::makeShape({5, 5, 5, 5, 5})));
        EXPECT_TRUE(ITensor::at(tensor, {4})->shapeEquals(ITensor::makeShape({5, 5, 5, 5})));
        EXPECT_TRUE(ITensor::at(tensor, {4, 4})->shapeEquals(ITensor::makeShape({5, 5, 5})));
        EXPECT_TRUE(ITensor::at(tensor, {4, 4, 4})->shapeEquals(ITensor::makeShape({5, 5})));
        EXPECT_TRUE(ITensor::at(tensor, {4, 4, 4, 4})->shapeEquals(ITensor::makeShape({5})));
        EXPECT_TRUE(ITensor::at(tensor, {4, 4, 4, 4, 4})->shapeEquals(ITensor::makeShape({1})));
    }

    {
        ITensor::SharedPtr tensor(BufferManager::cpu(ITensor::makeShape({}), dataType));

        EXPECT_TRUE(ITensor::slice(tensor, 0, 0)->shapeEquals(ITensor::makeShape({})));
        EXPECT_TRUE(ITensor::slice(tensor, {}, 0)->shapeEquals(ITensor::makeShape({0}))); // {0,{}} ==> {0}
        EXPECT_THROW(ITensor::slice(tensor, {}, 1), std::runtime_error);                  // (1,{}} /=> {1}
        EXPECT_THROW(ITensor::slice(tensor, {}, 2), std::runtime_error);
        EXPECT_THROW(ITensor::slice(tensor, {0}, 0), std::runtime_error);

        EXPECT_THROW(ITensor::at(tensor, {}), std::runtime_error); // due illegal slice(tensor, {}, 1)
        EXPECT_THROW(ITensor::at(tensor, {0}), std::runtime_error);
    }
    {
        ITensor::SharedPtr tensor(BufferManager::cpu(ITensor::makeShape({0}), dataType));

        EXPECT_TRUE(ITensor::slice(tensor, 0, 0)->shapeEquals(ITensor::makeShape({0})));
        EXPECT_TRUE(ITensor::slice(tensor, {}, 0)->shapeEquals(ITensor::makeShape({0, 0})));
        EXPECT_TRUE(ITensor::slice(tensor, {}, 1)->shapeEquals(ITensor::makeShape({1, 0})));
        EXPECT_TRUE(ITensor::slice(tensor, {0}, 0)->shapeEquals(ITensor::makeShape({0})));
        EXPECT_THROW(ITensor::slice(tensor, {}, 2), std::runtime_error);

        EXPECT_TRUE(ITensor::at(tensor, {})->shapeEquals(ITensor::makeShape({0})));
        EXPECT_THROW(ITensor::at(tensor, {0}), std::runtime_error);
    }
    {
        ITensor::SharedPtr tensor(BufferManager::cpu(ITensor::makeShape({0, 0}), dataType));

        EXPECT_TRUE(ITensor::slice(tensor, 0, 0)->shapeEquals(ITensor::makeShape({0, 0})));
        EXPECT_TRUE(ITensor::slice(tensor, {}, 0)->shapeEquals(ITensor::makeShape({0, 0, 0})));
        EXPECT_TRUE(ITensor::slice(tensor, {}, 1)->shapeEquals(ITensor::makeShape({1, 0, 0})));
        EXPECT_TRUE(ITensor::slice(tensor, {0}, 0)->shapeEquals(ITensor::makeShape({0, 0})));
        EXPECT_THROW(ITensor::slice(tensor, {}, 2), std::runtime_error);
        EXPECT_THROW(ITensor::slice(tensor, {0, 0}, 0), std::runtime_error);

        EXPECT_TRUE(ITensor::at(tensor, {})->shapeEquals(ITensor::makeShape({0, 0})));
        EXPECT_THROW(ITensor::at(tensor, {0}), std::runtime_error);
        EXPECT_THROW(ITensor::at(tensor, {0, 0}), std::runtime_error);
    }
    {
        ITensor::SharedPtr tensor(BufferManager::cpu(ITensor::makeShape({5, 0, 5}), dataType));

        EXPECT_TRUE(ITensor::slice(tensor, 0, 0)->shapeEquals(ITensor::makeShape({0, 0, 5})));
        EXPECT_TRUE(ITensor::slice(tensor, {}, 0)->shapeEquals(ITensor::makeShape({0, 5, 0, 5})));
        EXPECT_TRUE(ITensor::slice(tensor, {}, 1)->shapeEquals(ITensor::makeShape({1, 5, 0, 5})));
        EXPECT_TRUE(ITensor::slice(tensor, {0}, 0)->shapeEquals(ITensor::makeShape({0, 0, 5})));
        EXPECT_TRUE(ITensor::slice(tensor, {0, 0}, 0)->shapeEquals(ITensor::makeShape({0, 5})));
        EXPECT_THROW(ITensor::slice(tensor, {}, 2), std::runtime_error);
        EXPECT_THROW(ITensor::slice(tensor, {0, 0, 0}, 0), std::runtime_error);

        EXPECT_TRUE(ITensor::at(tensor, {})->shapeEquals(ITensor::makeShape({5, 0, 5})));
        EXPECT_TRUE(ITensor::at(tensor, {0})->shapeEquals(ITensor::makeShape({0, 5})));
        EXPECT_THROW(ITensor::at(tensor, {0, 0}), std::runtime_error);
    }
}

//! \brief Range shape in [begin, end).
class ShapeRange
{
public:
    ShapeRange(ITensor::Shape const& begin, ITensor::Shape const& end)
        : mBegin(begin)
        , mEnd(end)
    {
        TLLM_CHECK(mBegin.nbDims == mEnd.nbDims);
        for (int i = 0; i < mEnd.nbDims; i++)
        {
            TLLM_CHECK(mBegin.d[i] <= mEnd.d[i]);
        }
    }

    ShapeRange(ITensor::Shape end)
        : ShapeRange(
            [](auto dims)
            {
                for (int i = 0; i < dims.nbDims; i++)
                {
                    dims.d[i] = 0;
                }
                return dims;
            }(end),
            end)
    {
    }

    ShapeRange(
        std::initializer_list<ITensor::DimType64> const& begin, std::initializer_list<ITensor::DimType64> const& end)
        : ShapeRange(ITensor::makeShape(begin), ITensor::makeShape(end))
    {
    }

    ShapeRange(std::initializer_list<ITensor::DimType64> const& end)
        : ShapeRange(ITensor::makeShape(end))
    {
    }

    class Iterator : public std::iterator<std::input_iterator_tag, ITensor::Shape, ITensor::Shape,
                         ITensor::Shape const*, ITensor::Shape>
    {
        friend ShapeRange;

    protected:
        explicit Iterator(ITensor::Shape const& value, ShapeRange const& range)
            : mValue(value)
            , mRange(range)
        {
        }

    public:
        Iterator& operator++()
        {
            auto counter = [](ITensor::DimType64& value, bool& carry, ITensor::DimType64 min, ITensor::DimType64 max)
            {
                value += carry ? 1 : 0;
                carry = value == max;
            };
            if (mValue.nbDims == 0)
            {
                return *this;
            }
            bool carry = true;
            int i = mValue.nbDims;
            do
            {
                i--;
                counter(mValue.d[i], carry, mRange.mBegin.d[i], mRange.mEnd.d[i]);
            } while (i > 0 && carry);

            if (!carry)
            {
                i++;
                for (; i < mValue.nbDims; i++)
                {
                    mValue.d[i] = mRange.mBegin.d[i];
                }
            }
            return *this;
        }

        Iterator operator++(int)
        {
            Iterator retval = *this;
            ++(*this);
            return retval;
        }

        bool operator==(Iterator const& other) const
        {
            return ITensor::shapeEquals(mValue, other.mValue);
        }

        bool operator!=(Iterator const& other) const
        {
            return !(*this == other);
        }

        reference operator*() const
        {
            return mValue;
        }

    private:
        ITensor::Shape mValue;
        ShapeRange const& mRange;
    };

    Iterator begin() const
    {
        return Iterator(mBegin, *this);
    }

    Iterator end() const
    {
        return Iterator(mEnd, *this);
    }

private:
    ITensor::Shape const mBegin;
    ITensor::Shape const mEnd;
};

TEST(ShapeRange, test)
{
    {
        ITensor::Shape a = ITensor::makeShape({});
        ITensor::Shape b = ITensor::makeShape({});
        ShapeRange range(a, b);
        EXPECT_TRUE(range.begin() == range.end());
        EXPECT_TRUE(ITensor::shapeEquals(*range.begin(), ITensor::makeShape({})));
        int count = 0;
        for (auto const& v : range)
        {
            count++;
        }
        EXPECT_EQ(count, 0);
    }
    {
        int count = 0;
        for (auto const& v : ShapeRange({1, 1, 1}, {1, 1, 1}))
        {
            count++;
        }
        EXPECT_EQ(count, 0);
    }
    {
        ITensor::Shape a = ITensor::makeShape({1, 1, 1});
        ITensor::Shape b = ITensor::makeShape({3, 3, 3});
        ShapeRange range(a, b);
        EXPECT_TRUE(range.begin() != range.end());
        EXPECT_TRUE(ITensor::shapeEquals(a, *range.begin()));
        EXPECT_TRUE(ITensor::shapeEquals(b, *range.end()));
        int count = 0;
        for (auto const& v : range)
        {
            count++;
        }
        EXPECT_EQ(count, 8);
    }
    {
        int count = 0;
        for (auto const& v : ShapeRange({2, 2, 2, 2}))
        {
            count++;
        }
        EXPECT_EQ(count, 16);
    }
    {
        EXPECT_THROW(ShapeRange({0}, {1, 1}), std::runtime_error);
        EXPECT_THROW(ShapeRange({2, 2}, {1, 1}), std::runtime_error);
    }
}

TEST(ITensorTest, TensorDimsSliceAt)
{
    auto shape = ITensor::makeShape({5, 5, 5, 5});
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    ITensor::SharedPtr tensor(BufferManager::cpu(shape, dataType));

    auto verify = [&shape, &tensor, &dataType](ITensor::Shape const& index)
    {
        auto blockAt = ITensor::at(tensor, index);
        auto blockSliceRest = ITensor::slice(tensor, index);
        auto blockSliceZero = ITensor::slice(tensor, index, 0);
        auto blockSliceOne = ITensor::slice(tensor, index, 1);
        auto blockSliceTwo = (shape.d[index.nbDims - 1] - index.d[index.nbDims - 1] >= 2)
            ? std::make_optional(ITensor::slice(tensor, index, 2))
            : [&tensor, &index]()
        {
            EXPECT_THROW(ITensor::slice(tensor, index, 2), std::runtime_error);
            return std::nullopt;
        }();

        {
            auto strides = ITensor::strides(tensor->getShape());
            ITensor::DimType64 offset = 0;
            for (SizeType32 i = 0; i < index.nbDims; i++)
            {
                offset += index.d[i] * strides.d[i];
            }
            offset *= BufferDataType(dataType).getSize();
            auto base = static_cast<std::uint8_t*>(tensor->data());
            EXPECT_EQ(static_cast<std::uint8_t*>(blockAt->data()) - base, offset);
            EXPECT_EQ(static_cast<std::uint8_t*>(blockSliceRest->data()) - base, offset);
            EXPECT_EQ(static_cast<std::uint8_t*>(blockSliceOne->data()) - base, offset);
            if (blockSliceTwo)
            {
                EXPECT_EQ(static_cast<std::uint8_t*>(blockSliceTwo.value()->data()) - base, offset);
            }
        }
        {
            auto blockShape = blockAt->getShape();
            ITensor::Shape goldenShape = ITensor::makeShape({1});
            if (shape.nbDims > index.nbDims)
            {
                goldenShape.nbDims = shape.nbDims - index.nbDims;
                for (SizeType32 i = 0; i < goldenShape.nbDims; i++)
                {
                    goldenShape.d[i] = shape.d[i + index.nbDims];
                }
            }
            EXPECT_TRUE(ITensor::shapeEquals(blockShape, goldenShape));
        }
        {
            auto blockShape = blockSliceRest->getShape();
            ITensor::Shape goldenShape;
            goldenShape.nbDims = shape.nbDims - index.nbDims + 1;
            goldenShape.d[0] = shape.d[0 + index.nbDims - 1] - index.d[0 + index.nbDims - 1];
            for (SizeType32 i = 1; i < goldenShape.nbDims; i++)
            {
                goldenShape.d[i] = shape.d[i + index.nbDims - 1];
            }
            EXPECT_TRUE(ITensor::shapeEquals(blockShape, goldenShape));
        }
        {
            auto blockShape = blockSliceZero->getShape();
            ITensor::Shape goldenShape;
            goldenShape.nbDims = shape.nbDims - index.nbDims + 1;
            goldenShape.d[0] = 0;
            for (SizeType32 i = 1; i < goldenShape.nbDims; i++)
            {
                goldenShape.d[i] = shape.d[i + index.nbDims - 1];
            }
            EXPECT_TRUE(ITensor::shapeEquals(blockShape, goldenShape));
        }
        {
            auto blockShape = blockSliceOne->getShape();
            ITensor::Shape goldenShape;
            goldenShape.nbDims = shape.nbDims - index.nbDims + 1;
            goldenShape.d[0] = 1;
            for (SizeType32 i = 1; i < goldenShape.nbDims; i++)
            {
                goldenShape.d[i] = shape.d[i + index.nbDims - 1];
            }
            EXPECT_TRUE(ITensor::shapeEquals(blockShape, goldenShape));
        }
        if (blockSliceTwo)
        {
            auto blockShape = blockSliceTwo.value()->getShape();
            ITensor::Shape goldenShape;
            goldenShape.nbDims = shape.nbDims - index.nbDims + 1;
            goldenShape.d[0] = 2;
            for (SizeType32 i = 1; i < goldenShape.nbDims; i++)
            {
                goldenShape.d[i] = shape.d[i + index.nbDims - 1];
            }
            EXPECT_TRUE(ITensor::shapeEquals(blockShape, goldenShape));
        }
    };

    for (auto const& range : {ShapeRange({5}), ShapeRange({5, 5}), ShapeRange({5, 5, 5}), ShapeRange({5, 5, 5, 5})})
    {
        for (auto const& index : range)
        {
            verify(index);
        }
    }

    for (auto& range : {ShapeRange({4}, {7}), ShapeRange({4, 4}, {7, 7}), ShapeRange({4, 4, 4}, {7, 7, 7}),
             ShapeRange({4, 4, 4, 4}, {7, 7, 7, 7})})
    {
        auto it = range.begin();
        for (it++; it != range.end(); ++it)
        {
            EXPECT_THROW(ITensor::at(tensor, *it), std::runtime_error);
            EXPECT_THROW(ITensor::slice(tensor, *it, 1), std::runtime_error);
        }
    }
}

TEST(BufferRangeTest, ConstType)
{
    auto shape = ITensor::makeShape({5, 5, 5, 5, 5});
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    ITensor::SharedPtr tensor(BufferManager::cpu(shape, dataType));
    ITensor::SharedConstPtr tensorConst = tensor;

    //// 1 ////////////////////////////////////
    BufferRange<float const> tensor_RangeConst_WONT_ASSIGN(*tensor);
    // tensorRange_WONT_ASSIGN[0] = 3.14159;

    //// 2 ////////////////////////////////////
    BufferRange<float> tensorRange(*tensor);
    tensorRange[0] = 3.14159;

    //// 3 ////////////////////////////////////
    // BufferRange<float> TensorConst_Range_WONT_COMPILE(*tensorConst);

    //// 4 ////////////////////////////////////
    BufferRange<float const> tensorConst_RangeConst_WONT_ASSIGN(*tensorConst);
    // theConstOnerange_WONT_ASSIGN[0] = 1.1;

    BufferRange<float const> tensor_RangeConst(*tensor);
    BufferRange<float const> tensorConst_RangeConst(*tensorConst);

    float acc = 3.14159;
    for (auto& v : tensorRange)
    {
        v = acc * acc + 1.0;
    }
    for (SizeType32 i = 0; i < tensorRange.size(); i++)
    {
        EXPECT_EQ(tensorRange[i], tensor_RangeConst[i]);
        EXPECT_EQ(tensorRange[i], tensorConst_RangeConst[i]);
    }
}

TEST(ITensorTest, GetDimension)
{
    auto shape = ITensor::makeShape({10, 11, 12});
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    ITensor::SharedPtr tensor(BufferManager::cpu(shape, dataType));

    auto firstDimensionFromStart = tensor->getDimension<0>();
    EXPECT_EQ(firstDimensionFromStart, 10);

    auto firstDimensionFromEnd = tensor->getDimension<-3>();
    EXPECT_EQ(firstDimensionFromEnd, firstDimensionFromStart);

    auto lastDimensionFromEnd = tensor->getDimension<-1>();
    EXPECT_EQ(lastDimensionFromEnd, 12);

    auto lastDimensionFromStart = tensor->getDimension<2>();
    EXPECT_EQ(lastDimensionFromStart, lastDimensionFromEnd);
}
