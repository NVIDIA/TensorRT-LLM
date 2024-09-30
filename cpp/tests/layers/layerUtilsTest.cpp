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
#include <gtest/gtest.h>
#include <vector>

namespace tensorrt_llm::tests::layers
{

using namespace tensorrt_llm::runtime;

template <typename T>
class CopyToWorkspaceFixture : public testing::Test
{
public:
    CopyToWorkspaceFixture()
    {
        bufferManager = std::make_unique<BufferManager>(std::make_shared<CudaStream>());
    }

    std::unique_ptr<BufferManager> bufferManager;
    T value_;
};

using DataTypes = ::testing::Types<int8_t, half, float, double, int, long, int*>;

TYPED_TEST_SUITE(CopyToWorkspaceFixture, DataTypes);

TYPED_TEST(CopyToWorkspaceFixture, DataTooLarge_Throws)
{
    using dataType = decltype(this->value_);
    constexpr size_t numElements = 1024;
    auto const data = std::vector<dataType>(numElements);
    auto const workspaceSizeInBytes = numElements * sizeof(dataType) / 2;
    IBuffer::SharedPtr workspace = this->bufferManager->gpu(workspaceSizeInBytes);
    ASSERT_THROW(DecodingLayerWorkspace::copyToWorkspace(*this->bufferManager, data, workspace), common::TllmException);
}

TYPED_TEST(CopyToWorkspaceFixture, DataMuchTooLarge_Throws)
{
    using dataType = decltype(this->value_);
    constexpr size_t numElements = 1 << 15;
    auto const data = std::vector<dataType>(numElements);
    auto const workspaceSizeInBytes = numElements * sizeof(dataType) / 1000;
    IBuffer::SharedPtr workspace = this->bufferManager->gpu(workspaceSizeInBytes);
    ASSERT_THROW(DecodingLayerWorkspace::copyToWorkspace(*this->bufferManager, data, workspace), common::TllmException);
}

TYPED_TEST(CopyToWorkspaceFixture, DataFitsExactly_Succeeds)
{
    using dataType = decltype(this->value_);
    constexpr size_t numElements = 2048;
    auto const data = std::vector<dataType>(numElements);
    auto const workspaceSizeInBytes = numElements * sizeof(dataType);
    IBuffer::SharedPtr workspace = this->bufferManager->gpu(workspaceSizeInBytes);
    DecodingLayerWorkspace::copyToWorkspace(*this->bufferManager, data, workspace);
    sync_check_cuda_error();

    // Copy back and check data integrity.
    auto dataCopy = std::vector<dataType>(numElements);
    auto const dataSizeInBytes = numElements * sizeof(dataType);
    this->bufferManager->copy(*IBuffer::slice(workspace, 0, dataSizeInBytes), dataCopy.data(), MemoryType::kCPU);
    sync_check_cuda_error();

    for (auto i = 0; i < numElements; i++)
    {
        ASSERT_EQ(dataCopy[i], data[i]);
    }
}

TYPED_TEST(CopyToWorkspaceFixture, DataSmallerThanWorkspace_Succeeds)
{
    using dataType = decltype(this->value_);
    constexpr size_t numElements = 2048;
    auto const data = std::vector<dataType>(numElements);
    auto const workspaceSizeInBytes = numElements * sizeof(dataType) * 4;
    IBuffer::SharedPtr workspace = this->bufferManager->gpu(workspaceSizeInBytes);
    DecodingLayerWorkspace::copyToWorkspace(*this->bufferManager, data, workspace);
    sync_check_cuda_error();

    // Copy back and check data integrity.
    auto dataCopy = std::vector<dataType>(numElements);
    auto const dataSizeInBytes = numElements * sizeof(dataType);
    this->bufferManager->copy(*IBuffer::slice(workspace, 0, dataSizeInBytes), dataCopy.data(), MemoryType::kCPU);
    sync_check_cuda_error();

    for (auto i = 0; i < numElements; i++)
    {
        ASSERT_EQ(dataCopy[i], data[i]);
    }
}

TYPED_TEST(CopyToWorkspaceFixture, DataMuchSmallerThanWorkspace_Succeeds)
{
    using dataType = decltype(this->value_);
    constexpr size_t numElements = 2048;
    auto const data = std::vector<dataType>(numElements);
    auto const workspaceSizeInBytes = numElements * sizeof(dataType) * 1000;
    IBuffer::SharedPtr workspace = this->bufferManager->gpu(workspaceSizeInBytes);
    DecodingLayerWorkspace::copyToWorkspace(*this->bufferManager, data, workspace);
    sync_check_cuda_error();

    // Copy back and check data integrity.
    auto dataCopy = std::vector<dataType>(numElements);
    auto const dataSizeInBytes = numElements * sizeof(dataType);
    this->bufferManager->copy(*IBuffer::slice(workspace, 0, dataSizeInBytes), dataCopy.data(), MemoryType::kCPU);
    sync_check_cuda_error();

    for (auto i = 0; i < numElements; i++)
    {
        ASSERT_EQ(dataCopy[i], data[i]);
    }
}

} // namespace tensorrt_llm::tests::layers
