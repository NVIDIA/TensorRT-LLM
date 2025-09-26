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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/decodingLayerWorkspace.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include <gtest/gtest.h>
#include <memory_resource>
#include <random>
#include <vector>

namespace tensorrt_llm::tests::layers
{

using namespace tensorrt_llm::runtime;

class MineFieldAllocator : public std::pmr::memory_resource
{
    static constexpr std::size_t kMinPadding = 256;

    void* do_allocate(std::size_t bytes, std::size_t alignment) override
    {
        alignment = std::max(alignment, sizeof(uint64_t));
        bytes = common::roundUp(bytes, alignment);
        auto const padding = common::roundUp(kMinPadding, alignment);

        auto const allocSize = bytes + 2 * padding;
        void* p = std::pmr::new_delete_resource()->allocate(allocSize, alignment);

        std::generate_n(static_cast<uint64_t*>(p), allocSize / sizeof(uint64_t),
            [engine = std::mt19937_64(reinterpret_cast<uint64_t>(p))]() mutable { return engine(); });

        return static_cast<uint8_t*>(p) + padding;
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override
    {
        auto allocAlignment = std::max(alignment, sizeof(uint64_t));
        auto allocBytes = common::roundUp(bytes, allocAlignment);
        auto const padding = common::roundUp(kMinPadding, alignment);
        void* allocP = static_cast<uint8_t*>(p) - padding;
        auto const allocSize = allocBytes + 2 * padding;

        auto engine = std::mt19937_64(reinterpret_cast<uint64_t>(allocP));
        auto* verifyP = static_cast<uint64_t*>(allocP);
        for (size_t i = 0; i < padding / sizeof(uint64_t); ++i)
        {
            ASSERT_EQ(verifyP[i], engine());
        }

        engine.discard(allocBytes / sizeof(uint64_t));

        verifyP += (allocBytes + padding) / sizeof(uint64_t);
        for (size_t i = 0; i < padding / sizeof(uint64_t); ++i)
        {
            ASSERT_EQ(verifyP[i], engine());
        }

        std::pmr::new_delete_resource()->deallocate(allocP, allocSize, allocAlignment);
    }

    bool do_is_equal(std::pmr::memory_resource const& other) const noexcept override
    {
        return *this == other;
    }
};

MineFieldAllocator mineFieldAllocator{};

template <typename T>
class CopyToWorkspaceFixture : public testing::Test
{
public:
    CopyToWorkspaceFixture()
    {
        bufferManager = std::make_unique<BufferManager>(std::make_shared<CudaStream>());
    }

    void SetUp() override
    {
        std::pmr::set_default_resource(&mineFieldAllocator);
    }

    void fillData(std::pmr::vector<T>& vec)
    {
        if constexpr (std::is_pointer_v<T>)
        {
            std::iota(vec.begin(), vec.end(), static_cast<T>(nullptr));
        }
        else
        {
            std::iota(vec.begin(), vec.end(), 0);
        }
    }

    std::unique_ptr<BufferManager> bufferManager;
    static T value_;
};

using DataTypes = ::testing::Types<int8_t, half, float, double, int32_t, int64_t, int32_t*>;

TYPED_TEST_SUITE(CopyToWorkspaceFixture, DataTypes);

TYPED_TEST(CopyToWorkspaceFixture, DataTooLarge_Throws)
{
    using dataType = decltype(this->value_);
    constexpr size_t numElements = 1024;
    auto const data = std::pmr::vector<dataType>(numElements);
    auto const workspaceSizeInBytes = numElements * sizeof(dataType) / 2;
    IBuffer::SharedPtr workspace = this->bufferManager->gpu(workspaceSizeInBytes);
    ASSERT_THROW(DecodingLayerWorkspace::copyToWorkspace(*this->bufferManager, data, workspace), common::TllmException);
}

TYPED_TEST(CopyToWorkspaceFixture, DataMuchTooLarge_Throws)
{
    using dataType = decltype(this->value_);
    constexpr size_t numElements = 1 << 15;
    auto const data = std::pmr::vector<dataType>(numElements);
    auto const workspaceSizeInBytes = numElements * sizeof(dataType) / 1000;
    IBuffer::SharedPtr workspace = this->bufferManager->gpu(workspaceSizeInBytes);
    ASSERT_THROW(DecodingLayerWorkspace::copyToWorkspace(*this->bufferManager, data, workspace), common::TllmException);
}

TYPED_TEST(CopyToWorkspaceFixture, DataFitsExactly_Succeeds)
{
    using dataType = decltype(this->value_);
    constexpr size_t numElements = 2048;
    auto data = std::pmr::vector<dataType>(numElements);
    this->fillData(data);
    auto const workspaceSizeInBytes = numElements * sizeof(dataType);
    IBuffer::SharedPtr workspace = this->bufferManager->gpu(workspaceSizeInBytes);
    DecodingLayerWorkspace::copyToWorkspace(*this->bufferManager, data, workspace);
    sync_check_cuda_error(this->bufferManager->getStream().get());

    // Copy back and check data integrity.
    auto dataCopy = std::pmr::vector<dataType>(numElements);
    auto const dataSizeInBytes = numElements * sizeof(dataType);
    this->bufferManager->copy(*IBuffer::slice(workspace, 0, dataSizeInBytes), dataCopy.data(), MemoryType::kCPU);
    sync_check_cuda_error(this->bufferManager->getStream().get());

    for (auto i = 0; i < numElements; i++)
    {
        ASSERT_EQ(dataCopy[i], data[i]);
    }
}

TYPED_TEST(CopyToWorkspaceFixture, DataSmallerThanWorkspace_Succeeds)
{
    using dataType = decltype(this->value_);
    constexpr size_t numElements = 2048;
    auto data = std::pmr::vector<dataType>(numElements);
    this->fillData(data);
    auto const workspaceSizeInBytes = numElements * sizeof(dataType) * 4;
    IBuffer::SharedPtr workspace = this->bufferManager->gpu(workspaceSizeInBytes);
    DecodingLayerWorkspace::copyToWorkspace(*this->bufferManager, data, workspace);
    sync_check_cuda_error(this->bufferManager->getStream().get());

    // Copy back and check data integrity.
    auto dataCopy = std::pmr::vector<dataType>(numElements);
    auto const dataSizeInBytes = numElements * sizeof(dataType);
    this->bufferManager->copy(*IBuffer::slice(workspace, 0, dataSizeInBytes), dataCopy.data(), MemoryType::kCPU);
    sync_check_cuda_error(this->bufferManager->getStream().get());

    for (auto i = 0; i < numElements; i++)
    {
        ASSERT_EQ(dataCopy[i], data[i]);
    }
}

TYPED_TEST(CopyToWorkspaceFixture, DataMuchSmallerThanWorkspace_Succeeds)
{
    using dataType = decltype(this->value_);
    constexpr size_t numElements = 2048;
    auto data = std::pmr::vector<dataType>(numElements);
    this->fillData(data);
    auto const workspaceSizeInBytes = numElements * sizeof(dataType) * 1000;
    IBuffer::SharedPtr workspace = this->bufferManager->gpu(workspaceSizeInBytes);
    DecodingLayerWorkspace::copyToWorkspace(*this->bufferManager, data, workspace);
    sync_check_cuda_error(this->bufferManager->getStream().get());

    // Copy back and check data integrity.
    auto dataCopy = std::pmr::vector<dataType>(numElements);
    auto const dataSizeInBytes = numElements * sizeof(dataType);
    this->bufferManager->copy(*IBuffer::slice(workspace, 0, dataSizeInBytes), dataCopy.data(), MemoryType::kCPU);
    sync_check_cuda_error(this->bufferManager->getStream().get());

    for (auto i = 0; i < numElements; i++)
    {
        ASSERT_EQ(dataCopy[i], data[i]);
    }
}

TYPED_TEST(CopyToWorkspaceFixture, TypedWorkspaceBuffer_Succeeds)
{
    using dataType = decltype(this->value_);
    if constexpr (std::is_same_v<dataType, double>)
    {
        // There's no TRTDataType<double>
    }
    else
    {
        constexpr size_t numElements = 2048;
        auto data = std::pmr::vector<dataType>(numElements);
        this->fillData(data);
        IBuffer::SharedPtr workspace = this->bufferManager->gpu(numElements, TRTDataType<dataType>::value);
        DecodingLayerWorkspace::copyToWorkspace(*this->bufferManager, data, workspace);
        sync_check_cuda_error(this->bufferManager->getStream().get());

        // Copy back and check data integrity.
        auto dataCopy = std::pmr::vector<dataType>(numElements);
        this->bufferManager->copy(*IBuffer::slice(workspace, 0, numElements), dataCopy.data(), MemoryType::kCPU);
        sync_check_cuda_error(this->bufferManager->getStream().get());

        for (auto i = 0; i < numElements; i++)
        {
            ASSERT_EQ(dataCopy[i], data[i]);
        }
    }
}

TYPED_TEST(CopyToWorkspaceFixture, MismatchBufferType_Throws)
{
    using dataType = decltype(this->value_);
    if constexpr (sizeof(dataType) == 1 || std::is_same_v<dataType, double>)
    {
        // Allow copy mismatch type into int8_t workspace buffer
        // There's no TRTDataType<double>
    }
    else
    {
        constexpr size_t numElements = 2048;
        using differentType = std::pair<dataType, dataType>;
        auto data = std::pmr::vector<differentType>(numElements);
        IBuffer::SharedPtr workspace = this->bufferManager->gpu(numElements, TRTDataType<dataType>::value);
        ASSERT_THROW(
            DecodingLayerWorkspace::copyToWorkspace(*this->bufferManager, data, workspace), common::TllmException);
    }
}

} // namespace tensorrt_llm::tests::layers
