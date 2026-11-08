/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <tensorrt_llm/runtime/bufferView.h>

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"

using namespace tensorrt_llm::runtime;

TEST(IBufferTest, getBufferPtr)
{
    auto buffer = BufferManager::cpu(16, TRTDataType<float>::value);
    auto* ptr = buffer->getBufferPtr<float>();
    static_assert(std::is_same_v<decltype(ptr), float*>);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(ptr, buffer->data());
    ptr[0] = 3.0F;
    EXPECT_EQ(buffer->getBufferRange<float>()[0], 3.0F);
}

TEST(IBufferTest, getConstBufferPtr)
{
    IBuffer::SharedConstPtr buffer = BufferManager::cpu(16, TRTDataType<float>::value);
    auto* ptr = buffer->getBufferPtr<float>();
    static_assert(std::is_same_v<decltype(ptr), float const*>);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(ptr, buffer->data());
}

TEST(IBufferTest, getBufferPtrWithConstElementType)
{
    auto buffer = BufferManager::cpu(16, TRTDataType<float>::value);
    auto* ptr = buffer->getBufferPtr<float const>();
    static_assert(std::is_same_v<decltype(ptr), float const*>);
    EXPECT_EQ(ptr, buffer->data());
}

TEST(IBufferTest, typedAccessRejectsMismatchedDataType)
{
    IBuffer::SharedPtr buffer = BufferManager::cpu(16, TRTDataType<float>::value);
    IBuffer::SharedConstPtr constBuffer = buffer;
    EXPECT_THROW(static_cast<void>(buffer->getBufferPtr<int32_t>()), std::bad_cast);
    EXPECT_THROW(static_cast<void>(constBuffer->getBufferPtr<int32_t>()), std::bad_cast);
    EXPECT_THROW(static_cast<void>(buffer->getBufferRange<int32_t>()), std::bad_cast);
    EXPECT_THROW(static_cast<void>(constBuffer->getBufferRange<int32_t>()), std::bad_cast);
}

TEST(IBufferTest, getBufferRange)
{
    auto buffer = BufferManager::cpu(16, TRTDataType<float>::value);
    auto range = buffer->getBufferRange<float>();
    static_assert(std::is_same_v<decltype(range), std::span<float>>);
    EXPECT_EQ(range.size(), buffer->getSize());
    EXPECT_EQ(range.data(), buffer->getBufferPtr<float>());
    range.back() = 7.0F;
    EXPECT_EQ(buffer->getBufferPtr<float>()[buffer->getSize() - 1], 7.0F);
}

TEST(IBufferTest, getConstBufferRange)
{
    IBuffer::SharedPtr buffer = BufferManager::cpu(16, TRTDataType<float>::value);
    IBuffer::SharedConstPtr constBuffer = buffer;
    auto range = constBuffer->getBufferRange<float>();
    auto constElementRange = buffer->getBufferRange<float const>();
    static_assert(std::is_same_v<decltype(range), std::span<float const>>);
    static_assert(std::is_same_v<decltype(constElementRange), std::span<float const>>);
    EXPECT_EQ(range.size(), buffer->getSize());
    EXPECT_EQ(range.data(), constElementRange.data());
    buffer->getBufferPtr<float>()[0] = 5.0F;
    EXPECT_EQ(range[0], 5.0F);
}

TEST(IBufferTest, getPointerBufferRange)
{
    IBuffer::SharedPtr buffer = BufferManager::cpu(2, TRTDataType<void*>::value);
    IBuffer::SharedConstPtr constBuffer = buffer;
    auto range = buffer->getBufferRange<void*>();
    auto constRange = constBuffer->getBufferRange<void*>();
    static_assert(std::is_same_v<decltype(range), std::span<void*>>);
    static_assert(std::is_same_v<decltype(constRange), std::span<void* const>>);
    int value = 42;
    range[0] = &value;
    EXPECT_EQ(constRange[0], &value);
    EXPECT_EQ(range.size(), buffer->getSize());
}

TEST(IBufferTest, typedAccessEmptyBuffer)
{
    IBuffer::SharedPtr buffer = BufferManager::cpu(0, TRTDataType<float>::value);
    IBuffer::SharedConstPtr constBuffer = buffer;
    EXPECT_EQ(buffer->getBufferPtr<float>(), nullptr);
    EXPECT_EQ(constBuffer->getBufferPtr<float>(), nullptr);
    EXPECT_TRUE(buffer->getBufferRange<float>().empty());
    EXPECT_TRUE(constBuffer->getBufferRange<float>().empty());

    buffer->resize(16);
    EXPECT_NE(buffer->getBufferPtr<float>(), nullptr);
    EXPECT_EQ(buffer->getBufferRange<float>().size(), 16);
    buffer->resize(0);
    EXPECT_EQ(buffer->getBufferPtr<float>(), nullptr);
    EXPECT_TRUE(buffer->getBufferRange<float>().empty());
}

TEST(IBufferTest, accessDataPtrEmptyInit)
{
    IBuffer::SharedPtr buffer = BufferManager::cpu(0, TRTDataType<float>::value);
    auto data = buffer->data();
    ASSERT_TRUE(data == nullptr);

    buffer->resize(16);
    data = buffer->data();
    ASSERT_TRUE(data != nullptr);
}

TEST(IBufferTest, accessDataPtr)
{
    IBuffer::SharedPtr buffer = BufferManager::cpu(16, TRTDataType<float>::value);
    auto data = buffer->data();
    ASSERT_TRUE(data != nullptr);

    buffer->resize(0);
    data = buffer->data();
    ASSERT_TRUE(data == nullptr);
}

TEST(IBufferTest, accessDataConstPtr)
{
    IBuffer::SharedConstPtr buffer = BufferManager::cpu(16, TRTDataType<float>::value);
    auto data = buffer->data();
    ASSERT_TRUE(data != nullptr);
}

TEST(IBufferTest, BufferView)
{
    IBuffer::SharedPtr buffer = BufferManager::cpu(16, TRTDataType<float>::value);
    auto view = std::make_unique<BufferView>(buffer, 0, 8);

    EXPECT_EQ(view->getSize(), buffer->getSize() / 2);

    EXPECT_NE(view->data(), nullptr);
    EXPECT_NE(buffer->data(), nullptr);

    EXPECT_NO_THROW(view->release());
    EXPECT_EQ(view->data(), nullptr);
    EXPECT_NE(buffer->data(), nullptr);
}

TEST(IBufferTest, BufferEmptyView)
{
    IBuffer::SharedPtr buffer = BufferManager::cpu(16, TRTDataType<float>::value);
    auto view = std::make_unique<BufferView>(buffer, 0, 0);

    EXPECT_EQ(view->getSize(), 0);

    EXPECT_EQ(view->data(), nullptr);
    EXPECT_NE(buffer->data(), nullptr);
}
