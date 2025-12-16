/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/ncclUtils.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <gtest/gtest.h>
#include <nccl.h>
#include <thread>
#include <vector>

#if ENABLE_MULTI_DEVICE && BUILD_PYT
#include <torch/extension.h>
#endif

#if ENABLE_MULTI_DEVICE

namespace mpi = tensorrt_llm::mpi;
namespace tr = tensorrt_llm::runtime;
namespace nccl_util = tensorrt_llm::common::nccl_util;

using tensorrt_llm::getComm;

// Helper function to create a split communicator for testing
// This allows us to test cleanup behavior explicitly by controlling the lifetime
std::shared_ptr<ncclComm_t> createSplitComm(ncclComm_t parentComm, int color, int key)
{
    ncclComm_t newComm;
    ncclResult_t result = ncclCommSplit(parentComm, color, key, &newComm, nullptr);
    if (result != ncclSuccess)
    {
        TLLM_THROW("ncclCommSplit failed with error: %d", result);
    }

    // Create a shared_ptr with custom deleter that cleans up resources first
    return std::shared_ptr<ncclComm_t>(new ncclComm_t(newComm),
        [](ncclComm_t* comm)
        {
            if (comm && *comm)
            {
                // STEP 1: Clean up all registered resources FIRST
                tensorrt_llm::common::nccl_util::NcclCommResourceManager::getInstance().cleanupResources(*comm);

                // STEP 2: Now destroy the NCCL communicator
                ncclResult_t result = ncclCommDestroy(*comm);
                if (result != ncclSuccess)
                {
                    TLLM_LOG_WARNING("ncclCommDestroy failed with error: %d", result);
                }

                // STEP 3: Free the memory
                delete comm;
            }
        });
}

//==============================================================================
// NcclCommResourceManager Tests
//==============================================================================

class NcclCommResourceManagerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto& comm = mpi::MpiComm::world();
        mWorldSize = comm.getSize();
        mRank = comm.getRank();

        if (mWorldSize < 2)
        {
            GTEST_SKIP() << "Requires at least 2 ranks (got " << mWorldSize << ")";
        }

        // Set CUDA device for this rank (required before NCCL initialization)
        int deviceCount = 0;
        TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount > 0)
        {
            int deviceId = mRank % deviceCount;
            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        }

        // Create a communicator for testing
        std::set<int> group;
        for (int i = 0; i < mWorldSize; ++i)
        {
            group.insert(i);
        }
        mComm = getComm(group);
    }

    void TearDown() override
    {
        // Communicator cleanup happens automatically via shared_ptr deleter
        mComm.reset();
    }

    int mWorldSize;
    int mRank;
    std::shared_ptr<ncclComm_t> mComm;
};

TEST_F(NcclCommResourceManagerTest, ResourceRegistration)
{
    auto& manager = nccl_util::NcclCommResourceManager::getInstance();

    // Create a separate comm using split for this test
    auto testComm = createSplitComm(*mComm, 0, mRank);

    // Register a resource
    bool cleanupCalled = false;
    manager.registerResource(
        *testComm, [&cleanupCalled]() { cleanupCalled = true; }, "TestResource");

    EXPECT_TRUE(manager.hasResources(*testComm));
    EXPECT_EQ(manager.getResourceCount(*testComm), 1);
    EXPECT_FALSE(cleanupCalled); // Cleanup not called yet

    // Store the raw comm value before destruction
    ncclComm_t rawComm = *testComm;

    // Cleanup should be called when comm is destroyed
    testComm.reset();

    // Verify cleanup was called
    EXPECT_TRUE(cleanupCalled);

    // Verify cleanup: check that the old comm (now destroyed) no longer has resources
    // Note: The comm is destroyed, but we can still check the manager's internal state
    // The cleanup should have removed all resources for this comm
    EXPECT_FALSE(manager.hasResources(rawComm));
    EXPECT_EQ(manager.getResourceCount(rawComm), 0);
}

TEST_F(NcclCommResourceManagerTest, MultipleResources)
{
    auto& manager = nccl_util::NcclCommResourceManager::getInstance();

    // Create a separate comm using split for this test
    auto testComm = createSplitComm(*mComm, 0, mRank);

    std::vector<int> cleanupOrder;
    manager.registerResource(
        *testComm, [&cleanupOrder]() { cleanupOrder.push_back(1); }, "Resource1");
    manager.registerResource(
        *testComm, [&cleanupOrder]() { cleanupOrder.push_back(2); }, "Resource2");
    manager.registerResource(
        *testComm, [&cleanupOrder]() { cleanupOrder.push_back(3); }, "Resource3");

    EXPECT_EQ(manager.getResourceCount(*testComm), 3);

    // Cleanup order should be preserved - destroy comm and verify order
    testComm.reset();

    // Verify cleanup order was preserved (1, 2, 3)
    EXPECT_EQ(cleanupOrder.size(), 3);
    EXPECT_EQ(cleanupOrder[0], 1);
    EXPECT_EQ(cleanupOrder[1], 2);
    EXPECT_EQ(cleanupOrder[2], 3);
}

TEST_F(NcclCommResourceManagerTest, ResourceCount)
{
    auto& manager = nccl_util::NcclCommResourceManager::getInstance();

    // Create a separate comm using split for this test
    auto testComm = createSplitComm(*mComm, 0, mRank);

    EXPECT_FALSE(manager.hasResources(*testComm));
    EXPECT_EQ(manager.getResourceCount(*testComm), 0);

    manager.registerResource(
        *testComm, []() {}, "Test1");
    EXPECT_EQ(manager.getResourceCount(*testComm), 1);

    manager.registerResource(
        *testComm, []() {}, "Test2");
    EXPECT_EQ(manager.getResourceCount(*testComm), 2);

    testComm.reset();
}

//==============================================================================
// NCCLWindowAllocator Tests
//==============================================================================

class NCCLWindowAllocatorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto& comm = mpi::MpiComm::world();
        mWorldSize = comm.getSize();
        mRank = comm.getRank();

        if (mWorldSize < 2)
        {
            GTEST_SKIP() << "Requires at least 2 ranks (got " << mWorldSize << ")";
        }

        // Set CUDA device for this rank (required before NCCL initialization)
        int deviceCount = 0;
        TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount > 0)
        {
            int deviceId = mRank % deviceCount;
            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        }

        // Check if NCCL symmetric is supported
        auto& ncclHelper = nccl_util::NCCLHelper::getInstance();
        if (!ncclHelper.isLoaded())
        {
            GTEST_SKIP() << "NCCL library with symmetric memory support is not available";
        }

        std::set<int> group;
        for (int i = 0; i < mWorldSize; ++i)
        {
            group.insert(i);
        }
        mComm = getComm(group);
    }

    void TearDown() override
    {
        // Cleanup happens automatically
        mComm.reset();
    }

    int mWorldSize;
    int mRank;
    std::shared_ptr<ncclComm_t> mComm;
};

TEST_F(NCCLWindowAllocatorTest, BasicAllocation)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    const size_t bufferSize = 1024 * 1024; // 1MB
    auto buffer = allocator.requestBuffer(*mComm, bufferSize);

    EXPECT_TRUE(buffer.isValid());
    EXPECT_NE(buffer.ptr, nullptr);
    EXPECT_NE(buffer.window, nullptr);
    EXPECT_EQ(buffer.size, bufferSize);
    EXPECT_GE(buffer.handle, 0);

    // Verify we can search for it
    auto found = allocator.searchBuffer(*mComm, buffer.ptr);
    EXPECT_TRUE(found.isValid());
    EXPECT_EQ(found.ptr, buffer.ptr);

    // Release the buffer
    allocator.releaseBuffer(*mComm, buffer.ptr);
}

TEST_F(NCCLWindowAllocatorTest, BufferReuse)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    const size_t bufferSize = 512 * 1024; // 512KB

    // Allocate first buffer
    auto buffer1 = allocator.requestBuffer(*mComm, bufferSize);
    EXPECT_TRUE(buffer1.isValid());
    void* ptr1 = buffer1.ptr;

    // Release it
    allocator.releaseBuffer(*mComm, ptr1);

    // Request another buffer of the same size - should reuse
    auto buffer2 = allocator.requestBuffer(*mComm, bufferSize);
    EXPECT_TRUE(buffer2.isValid());
    EXPECT_EQ(buffer2.ptr, ptr1); // Should be the same buffer

    allocator.releaseBuffer(*mComm, buffer2.ptr);
}

TEST_F(NCCLWindowAllocatorTest, BestFitReuse)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    // Allocate buffers of different sizes
    auto buffer1MB = allocator.requestBuffer(*mComm, 1024 * 1024);
    auto buffer2MB = allocator.requestBuffer(*mComm, 2 * 1024 * 1024);
    auto buffer512KB = allocator.requestBuffer(*mComm, 512 * 1024);

    void* ptr1MB = buffer1MB.ptr;
    void* ptr2MB = buffer2MB.ptr;
    void* ptr512KB = buffer512KB.ptr;

    // Release all
    allocator.releaseBuffer(*mComm, ptr1MB);
    allocator.releaseBuffer(*mComm, ptr2MB);
    allocator.releaseBuffer(*mComm, ptr512KB);

    // Request 768KB - should reuse 1MB (best fit, smallest that fits)
    auto buffer768KB = allocator.requestBuffer(*mComm, 768 * 1024);
    EXPECT_TRUE(buffer768KB.isValid());
    EXPECT_EQ(buffer768KB.ptr, ptr1MB);       // Should reuse 1MB buffer
    EXPECT_EQ(buffer768KB.size, 1024 * 1024); // Original size

    allocator.releaseBuffer(*mComm, buffer768KB.ptr);
}

TEST_F(NCCLWindowAllocatorTest, MultipleBuffers)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    const size_t bufferSize = 256 * 1024;
    std::vector<void*> ptrs;

    // Allocate multiple buffers
    for (int i = 0; i < 5; ++i)
    {
        auto buffer = allocator.requestBuffer(*mComm, bufferSize);
        EXPECT_TRUE(buffer.isValid());
        ptrs.push_back(buffer.ptr);
    }

    EXPECT_EQ(allocator.getBufferCount(*mComm), 5);
    EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 5);

    // Release all
    for (auto* ptr : ptrs)
    {
        allocator.releaseBuffer(*mComm, ptr);
    }

    EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 0);
    EXPECT_EQ(allocator.getBufferCount(*mComm), 5); // Buffers still exist, just not in use
}

TEST_F(NCCLWindowAllocatorTest, SearchBuffer)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    const size_t bufferSize = 128 * 1024;
    auto buffer = allocator.requestBuffer(*mComm, bufferSize);

    // Test searchBuffer
    auto found = allocator.searchBuffer(*mComm, buffer.ptr);
    EXPECT_TRUE(found.isValid());
    EXPECT_EQ(found.ptr, buffer.ptr);
    // Compare against actual allocated size (ncclMemAlloc may allocate more than requested)
    EXPECT_EQ(found.size, buffer.size);
    EXPECT_GE(found.size, bufferSize); // At least the requested size

    // Test search for non-existent buffer
    void* fakePtr = reinterpret_cast<void*>(0xDEADBEEF);
    auto notFound = allocator.searchBuffer(*mComm, fakePtr);
    EXPECT_FALSE(notFound.isValid());

    allocator.releaseBuffer(*mComm, buffer.ptr);
}

TEST_F(NCCLWindowAllocatorTest, GetWindowAndSize)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    const size_t bufferSize = 64 * 1024;
    auto buffer = allocator.requestBuffer(*mComm, bufferSize);

    // Test getWindow
    auto window = allocator.getWindow(*mComm, buffer.ptr);
    EXPECT_NE(window, nullptr);
    EXPECT_EQ(window, buffer.window);

    // Test getSize - compare against actual allocated size (ncclMemAlloc may allocate more than requested)
    auto size = allocator.getSize(*mComm, buffer.ptr);
    EXPECT_EQ(size, buffer.size);
    EXPECT_GE(size, bufferSize); // At least the requested size

    // Test with invalid pointer
    void* fakePtr = reinterpret_cast<void*>(0xDEADBEEF);
    EXPECT_EQ(allocator.getWindow(*mComm, fakePtr), nullptr);
    EXPECT_EQ(allocator.getSize(*mComm, fakePtr), 0);

    allocator.releaseBuffer(*mComm, buffer.ptr);
}

TEST_F(NCCLWindowAllocatorTest, GetBufferInfo)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    const size_t bufferSize = 32 * 1024;
    auto buffer = allocator.requestBuffer(*mComm, bufferSize);

    auto info = allocator.getBufferInfo(*mComm, buffer.ptr);
    EXPECT_TRUE(info.isValid());
    EXPECT_EQ(info.ptr, buffer.ptr);
    EXPECT_EQ(info.size, buffer.size);
    EXPECT_EQ(info.handle, buffer.handle);
    EXPECT_EQ(info.window, buffer.window);

    allocator.releaseBuffer(*mComm, buffer.ptr);
}

TEST_F(NCCLWindowAllocatorTest, ScopedBuffer)
{
    const size_t bufferSize = 16 * 1024;

    {
        nccl_util::ScopedNCCLWindowBuffer scopedBuffer(*mComm, bufferSize);
        EXPECT_TRUE(scopedBuffer.getBuffer().isValid());
        EXPECT_NE(scopedBuffer.getPtr(), nullptr);
        // Compare against actual allocated size (ncclMemAlloc may allocate more than requested)
        EXPECT_EQ(scopedBuffer.getSize(), scopedBuffer.getBuffer().size);
        EXPECT_GE(scopedBuffer.getSize(), bufferSize); // At least the requested size
        EXPECT_NE(scopedBuffer.getWindow(), nullptr);

        // Buffer should be in use
        auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
        EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 1);
    }

    // Buffer should be released when scoped buffer goes out of scope
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
    EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 0);
}

TEST_F(NCCLWindowAllocatorTest, CleanupOnCommDestroy)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    // Create a separate comm using split for this test
    auto testComm = createSplitComm(*mComm, 0, mRank);

    // Store the raw comm value before destruction
    ncclComm_t rawComm = *testComm;

    // Allocate some buffers
    const size_t bufferSize = 8 * 1024;
    auto buffer1 = allocator.requestBuffer(*testComm, bufferSize);
    auto buffer2 = allocator.requestBuffer(*testComm, bufferSize * 2);

    EXPECT_EQ(allocator.getBufferCount(*testComm), 2);
    EXPECT_EQ(allocator.getBufferInUseCount(*testComm), 2);

    // Verify buffers are valid
    EXPECT_TRUE(buffer1.isValid());
    EXPECT_TRUE(buffer2.isValid());

    // Manually release buffers before cleanup to avoid warnings
    allocator.releaseBuffer(*testComm, buffer1.ptr);
    allocator.releaseBuffer(*testComm, buffer2.ptr);

    // Verify buffers are released but still exist in pool
    EXPECT_EQ(allocator.getBufferInUseCount(*testComm), 0);
    EXPECT_EQ(allocator.getBufferCount(*testComm), 2); // Buffers still exist, just not in use

    // Destroy the communicator - buffers should be cleaned up automatically
    testComm.reset();

    // Verify cleanup: check that the old comm (now destroyed) no longer has buffers
    // Note: The comm is destroyed, but we can still check the allocator's internal state
    // The cleanup should have removed all buffers for this comm
    EXPECT_EQ(allocator.getBufferCount(rawComm), 0);
    EXPECT_EQ(allocator.getBufferInUseCount(rawComm), 0);
    // Note: isCommValid only checks for null, not cleaned-up state, because NCCL can reuse addresses
    // The real check is that buffers are gone, which we verify above
}

TEST_F(NCCLWindowAllocatorTest, CommValidity)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    // Valid comm should be valid
    EXPECT_TRUE(allocator.isCommValid(*mComm));

    // Null comm should be invalid
    EXPECT_FALSE(allocator.isCommValid(nullptr));
}

//==============================================================================
// Integration Tests
//==============================================================================

TEST_F(NCCLWindowAllocatorTest, MultipleComms)
{
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    // Create two different communicators using split (different colors)
    auto comm1 = createSplitComm(*mComm, 0, mRank);
    auto comm2 = createSplitComm(*mComm, 1, mRank);

    const size_t bufferSize = 4 * 1024;

    // Allocate buffers from both comms
    auto buffer1 = allocator.requestBuffer(*comm1, bufferSize);
    auto buffer2 = allocator.requestBuffer(*comm2, bufferSize);

    EXPECT_TRUE(buffer1.isValid());
    EXPECT_TRUE(buffer2.isValid());

    // Buffers should be tracked separately per comm
    EXPECT_EQ(allocator.getBufferCount(*comm1), 1);
    EXPECT_EQ(allocator.getBufferCount(*comm2), 1);
    EXPECT_NE(buffer1.ptr, buffer2.ptr); // Different buffers from different comms

    allocator.releaseBuffer(*comm1, buffer1.ptr);
    allocator.releaseBuffer(*comm2, buffer2.ptr);

    // Clean up comms
    comm1.reset();
    comm2.reset();
}

#if ENABLE_MULTI_DEVICE && BUILD_PYT
//==============================================================================
// createNCCLWindowTensor Tests
//==============================================================================

class CreateNCCLWindowTensorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto& comm = mpi::MpiComm::world();
        mWorldSize = comm.getSize();
        mRank = comm.getRank();

        if (mWorldSize < 2)
        {
            GTEST_SKIP() << "Requires at least 2 ranks (got " << mWorldSize << ")";
        }

        // Set CUDA device for this rank (required before NCCL initialization)
        int deviceCount = 0;
        TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount > 0)
        {
            int deviceId = mRank % deviceCount;
            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        }

        // Check if NCCL symmetric is supported
        auto& ncclHelper = nccl_util::NCCLHelper::getInstance();
        if (!ncclHelper.isLoaded())
        {
            GTEST_SKIP() << "NCCL library with symmetric memory support is not available";
        }

        std::set<int> group;
        for (int i = 0; i < mWorldSize; ++i)
        {
            group.insert(i);
        }
        mComm = getComm(group);
    }

    void TearDown() override
    {
        mComm.reset();
    }

    int mWorldSize;
    int mRank;
    std::shared_ptr<ncclComm_t> mComm;
};

TEST_F(CreateNCCLWindowTensorTest, BasicTensorCreation)
{
    using nccl_util::createNCCLWindowTensor;

    // Create a tensor with shape [4, 8] and float32 dtype
    std::vector<int64_t> shape = {4, 8};
    auto [tensor, buffer] = createNCCLWindowTensor(*mComm, shape, torch::kFloat32);

    // Verify tensor properties
    EXPECT_TRUE(tensor.defined());
    EXPECT_EQ(tensor.dtype(), torch::kFloat32);
    EXPECT_EQ(tensor.device().type(), torch::kCUDA);
    EXPECT_EQ(tensor.dim(), 2);
    EXPECT_EQ(tensor.size(0), 4);
    EXPECT_EQ(tensor.size(1), 8);
    EXPECT_EQ(tensor.numel(), 4 * 8);

    // Verify buffer properties
    EXPECT_TRUE(buffer.isValid());
    EXPECT_NE(buffer.ptr, nullptr);
    // ncclMemAlloc may allocate more than requested, so check at least the requested size
    EXPECT_GE(buffer.size, 4 * 8 * sizeof(float));
    EXPECT_NE(buffer.window, nullptr);

    // Verify tensor data pointer matches buffer pointer
    EXPECT_EQ(tensor.data_ptr(), buffer.ptr);

    // Tensor should be in use
    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();
    EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 1);
}

TEST_F(CreateNCCLWindowTensorTest, DifferentDtypes)
{
    using nccl_util::createNCCLWindowTensor;

    std::vector<int64_t> shape = {10};

    // Test float32
    {
        auto [tensor, buffer] = createNCCLWindowTensor(*mComm, shape, torch::kFloat32);
        EXPECT_EQ(tensor.dtype(), torch::kFloat32);
        // ncclMemAlloc may allocate more than requested, so check at least the requested size
        EXPECT_GE(buffer.size, 10 * sizeof(float));
        EXPECT_EQ(tensor.data_ptr(), buffer.ptr);
    }

    // Test float16
    {
        auto [tensor, buffer] = createNCCLWindowTensor(*mComm, shape, torch::kFloat16);
        EXPECT_EQ(tensor.dtype(), torch::kFloat16);
        // ncclMemAlloc may allocate more than requested, so check at least the requested size
        EXPECT_GE(buffer.size, 10 * sizeof(at::Half));
        EXPECT_EQ(tensor.data_ptr(), buffer.ptr);
    }

    // Test int32
    {
        auto [tensor, buffer] = createNCCLWindowTensor(*mComm, shape, torch::kInt32);
        EXPECT_EQ(tensor.dtype(), torch::kInt32);
        // ncclMemAlloc may allocate more than requested, so check at least the requested size
        EXPECT_GE(buffer.size, 10 * sizeof(int32_t));
        EXPECT_EQ(tensor.data_ptr(), buffer.ptr);
    }
}

TEST_F(CreateNCCLWindowTensorTest, DifferentShapes)
{
    using nccl_util::createNCCLWindowTensor;

    // 1D tensor
    {
        std::vector<int64_t> shape = {100};
        auto [tensor, buffer] = createNCCLWindowTensor(*mComm, shape, torch::kFloat32);
        EXPECT_EQ(tensor.dim(), 1);
        EXPECT_EQ(tensor.size(0), 100);
        // ncclMemAlloc may allocate more than requested, so check at least the requested size
        EXPECT_GE(buffer.size, 100 * sizeof(float));
    }

    // 3D tensor
    {
        std::vector<int64_t> shape = {2, 3, 4};
        auto [tensor, buffer] = createNCCLWindowTensor(*mComm, shape, torch::kFloat32);
        EXPECT_EQ(tensor.dim(), 3);
        EXPECT_EQ(tensor.size(0), 2);
        EXPECT_EQ(tensor.size(1), 3);
        EXPECT_EQ(tensor.size(2), 4);
        // ncclMemAlloc may allocate more than requested, so check at least the requested size
        EXPECT_GE(buffer.size, 2 * 3 * 4 * sizeof(float));
    }

    // 4D tensor
    {
        std::vector<int64_t> shape = {1, 2, 3, 4};
        auto [tensor, buffer] = createNCCLWindowTensor(*mComm, shape, torch::kFloat32);
        EXPECT_EQ(tensor.dim(), 4);
        EXPECT_EQ(tensor.numel(), 1 * 2 * 3 * 4);
        // ncclMemAlloc may allocate more than requested, so check at least the requested size
        EXPECT_GE(buffer.size, 1 * 2 * 3 * 4 * sizeof(float));
    }
}

TEST_F(CreateNCCLWindowTensorTest, TensorDeleterReleasesBuffer)
{
    using nccl_util::createNCCLWindowTensor;

    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    {
        std::vector<int64_t> shape = {16, 16};
        auto [tensor, buffer] = createNCCLWindowTensor(*mComm, shape, torch::kFloat32);

        EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 1);
        EXPECT_TRUE(buffer.isValid());
        void* bufferPtr = buffer.ptr;

        // Tensor goes out of scope - deleter should release the buffer
    }

    // Buffer should be released (not in use anymore)
    EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 0);

    // Buffer should still exist in the pool (for reuse)
    EXPECT_GE(allocator.getBufferCount(*mComm), 1);
}

TEST_F(CreateNCCLWindowTensorTest, MultipleTensors)
{
    using nccl_util::createNCCLWindowTensor;

    auto& allocator = nccl_util::NCCLWindowAllocator::getInstance();

    std::vector<int64_t> shape = {8, 8};
    auto [tensor1, buffer1] = createNCCLWindowTensor(*mComm, shape, torch::kFloat32);
    auto [tensor2, buffer2] = createNCCLWindowTensor(*mComm, shape, torch::kFloat32);
    auto [tensor3, buffer3] = createNCCLWindowTensor(*mComm, shape, torch::kFloat32);

    EXPECT_EQ(allocator.getBufferInUseCount(*mComm), 3);
    EXPECT_NE(buffer1.ptr, buffer2.ptr);
    EXPECT_NE(buffer2.ptr, buffer3.ptr);
    EXPECT_NE(buffer1.ptr, buffer3.ptr);

    // All tensors should be valid
    EXPECT_TRUE(tensor1.defined());
    EXPECT_TRUE(tensor2.defined());
    EXPECT_TRUE(tensor3.defined());
}

TEST_F(CreateNCCLWindowTensorTest, TensorStrides)
{
    using nccl_util::createNCCLWindowTensor;

    std::vector<int64_t> shape = {3, 4, 5};
    auto [tensor, buffer] = createNCCLWindowTensor(*mComm, shape, torch::kFloat32);

    // Verify strides are correct (row-major order)
    EXPECT_EQ(tensor.stride(0), 4 * 5); // stride for first dimension
    EXPECT_EQ(tensor.stride(1), 5);     // stride for second dimension
    EXPECT_EQ(tensor.stride(2), 1);     // stride for third dimension
}

#endif // ENABLE_MULTI_DEVICE && BUILD_PYT

#endif // ENABLE_MULTI_DEVICE
