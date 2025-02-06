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
#include <tensorrt_llm/runtime/bufferView.h>

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"

using namespace tensorrt_llm::runtime;

TEST(IBufferTest, bufferCastOrNullForOptionalWithValue)
{
    std::optional<IBuffer::SharedPtr> buffer = std::make_optional(BufferManager::cpu(16, TRTDataType<float>::value));
    auto ptr = bufferCastOrNull<float>(buffer);
    static_assert(std::is_same<decltype(ptr), float*>::value);
    ASSERT_TRUE(ptr != nullptr);
}

TEST(IBufferTest, bufferCastOrNullForOptionalWithoutValue)
{
    std::optional<IBuffer::SharedPtr> buffer = std::nullopt;
    auto ptr = bufferCastOrNull<float>(buffer);
    static_assert(std::is_same<decltype(ptr), float*>::value);
    ASSERT_TRUE(ptr == nullptr);
}

TEST(IBufferTest, bufferCastOrNullForSharedPtrWithValue)
{
    IBuffer::SharedPtr buffer = BufferManager::cpu(16, TRTDataType<float>::value);
    auto ptr = bufferCastOrNull<float>(buffer);
    static_assert(std::is_same<decltype(ptr), float*>::value);
    ASSERT_TRUE(ptr != nullptr);
}

TEST(IBufferTest, bufferCastOrNullForSharedPtrWithoutValue)
{
    IBuffer::SharedPtr buffer{nullptr};
    auto ptr = bufferCastOrNull<float>(buffer);
    static_assert(std::is_same<decltype(ptr), float*>::value);
    ASSERT_TRUE(ptr == nullptr);
}

TEST(IBufferTest, bufferCastOrNullForOptionalWithConstValue)
{
    std::optional<IBuffer::SharedConstPtr> buffer
        = std::make_optional(BufferManager::cpu(16, TRTDataType<float>::value));
    auto ptr = bufferCastOrNull<float>(buffer);
    static_assert(std::is_same<decltype(ptr), float const*>::value);
    ASSERT_TRUE(ptr != nullptr);
}

TEST(IBufferTest, bufferCastOrNullForOptionalWithoutConstValue)
{
    std::optional<IBuffer::SharedConstPtr> buffer = std::nullopt;
    auto ptr = bufferCastOrNull<float>(buffer);
    static_assert(std::is_same<decltype(ptr), float const*>::value);
    ASSERT_TRUE(ptr == nullptr);
}

TEST(IBufferTest, bufferCastOrNullForSharedPtrWithConstValue)
{
    IBuffer::SharedConstPtr buffer = BufferManager::cpu(16, TRTDataType<float>::value);
    auto ptr = bufferCastOrNull<float>(buffer);
    static_assert(std::is_same<decltype(ptr), float const*>::value);
    ASSERT_TRUE(ptr != nullptr);
}

TEST(IBufferTest, bufferCastOrNullForSharedPtrWithoutConstValue)
{
    IBuffer::SharedConstPtr buffer{nullptr};
    auto ptr = bufferCastOrNull<float>(buffer);
    static_assert(std::is_same<decltype(ptr), float const*>::value);
    ASSERT_TRUE(ptr == nullptr);
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
