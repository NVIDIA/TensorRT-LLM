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

#include <gtest/gtest.h>
#include <nvml.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/tllmBuffers.h"
#include "tensorrt_llm/runtime/virtualMemory.h"

#include <cstdint>
#include <random>
#include <unistd.h>
#include <vector>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

struct DummyException : std::runtime_error
{
    DummyException()
        : runtime_error("dummy exception")
    {
    }
};

class VirtualMemoryTestBase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (tc::getDeviceCount() == 0)
        {
            GTEST_SKIP() << "This test suite cannot run on systems with no devices.";
        }

        TLLM_CU_CHECK(cuInit(0));

        CUdevice dev;
        TLLM_CU_CHECK(cuDeviceGet(&dev, 0));

        CUcontext ctx;
        TLLM_CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
        TLLM_CU_CHECK(cuCtxSetCurrent(ctx));

        // Initialize NVML
        nvmlReturn_t nvmlResult = nvmlInit();
        TLLM_CHECK_WITH_INFO(nvmlResult == NVML_SUCCESS, "Failed to initialize NVML: %s", nvmlErrorString(nvmlResult));

        if (!memoryInfoAvailable())
        {
            TLLM_LOG_WARNING("Per process memory information unavailable.");
        }

        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
    }

    void TearDown() override
    {
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
    }

    static bool memoryInfoAvailable()
    {
        static bool available = []
        {
            auto blob = BufferManager::gpuSync(4096);
            auto usage = getCurrentProcessMemoryInfo();
            return usage != 0;
        }();

        return available;
    }

    static size_t getCurrentProcessMemoryInfo()
    {
        // Get current process ID
        uint32_t currentPid = static_cast<uint32_t>(getpid());

        // Get device handle for GPU 0
        nvmlDevice_t device;
        auto nvmlResult = nvmlDeviceGetHandleByIndex(0, &device);
        TLLM_CHECK_WITH_INFO(
            nvmlResult == NVML_SUCCESS, "Failed to get device handle: %s", nvmlErrorString(nvmlResult));

        // Get running processes
        unsigned int processCount = 1;
        std::vector<nvmlProcessInfo_v2_t> processes(processCount);
        nvmlResult = NVML_ERROR_INSUFFICIENT_SIZE;
        while (nvmlResult == NVML_ERROR_INSUFFICIENT_SIZE)
        {
            nvmlResult = nvmlDeviceGetComputeRunningProcesses_v3(device, &processCount, processes.data());
            TLLM_CHECK_WITH_INFO(nvmlResult == NVML_SUCCESS || nvmlResult == NVML_ERROR_INSUFFICIENT_SIZE,
                "Failed to get process count: %s", nvmlErrorString(nvmlResult));
            processes.resize(processCount);
        }

        // Find current process
        for (auto const& process : processes)
        {
            if (process.pid == currentPid)
            {
                return process.usedGpuMemory;
            }
        }

        return 0;
    }
};

class VirtualMemoryTest : public VirtualMemoryTestBase
{
};

// Test CUDAVirtualMemoryChunk materialize and release memory correctly
TEST_F(VirtualMemoryTest, TestBasic)
{
    CUdeviceptr address{};
    std::size_t constexpr size = 256 * 1024 * 1024;
    TLLM_CU_CHECK(cuMemAddressReserve(&address, size, 0, {}, 0));

    CUDAVirtualMemoryChunk::CreatorPtr creator
        = std::make_unique<LocalCreator<>>(CUmemAllocationProp{CU_MEM_ALLOCATION_TYPE_PINNED, CU_MEM_HANDLE_TYPE_NONE,
                                               {
                                                   CU_MEM_LOCATION_TYPE_DEVICE,
                                                   0,
                                               }},
            size);

    CUDAVirtualMemoryChunk::Configurators configurators;
    configurators.push_back(std::make_unique<UnicastConfigurator>(address, size,
        CUmemAccessDesc{{
                            CU_MEM_LOCATION_TYPE_DEVICE,
                            0,
                        },
            CU_MEM_ACCESS_FLAGS_PROT_READWRITE}));

    CUDAVirtualMemoryChunk vm(std::move(creator), std::move(configurators));
    ASSERT_EQ(vm.status(), CUDAVirtualMemoryChunk::RELEASED);

    auto memoryBegin = getCurrentProcessMemoryInfo();
    vm.materialize();
    ASSERT_EQ(vm.status(), CUDAVirtualMemoryChunk::MATERIALIZED);

    auto memoryMaterialized = getCurrentProcessMemoryInfo();
    if (memoryInfoAvailable())
    {
        ASSERT_EQ(memoryBegin + size, memoryMaterialized) << "materialize does not allocate memory";
    }

    auto result = cuMemsetD8_v2(address, 255, size);
    ASSERT_EQ(result, CUDA_SUCCESS) << "Accessing memory returned failure (first materialize)";
    TLLM_CU_CHECK(cuStreamSynchronize(nullptr));

    vm.release();
    ASSERT_EQ(vm.status(), CUDAVirtualMemoryChunk::RELEASED);
    auto memoryReleased = getCurrentProcessMemoryInfo();
    if (memoryInfoAvailable())
    {
        ASSERT_EQ(memoryBegin, memoryReleased) << "release does not release memory";
    }

    vm.materialize();
    ASSERT_EQ(vm.status(), CUDAVirtualMemoryChunk::MATERIALIZED);
    result = cuMemsetD8_v2(address, 255, size);
    ASSERT_EQ(result, CUDA_SUCCESS) << "Accessing memory returned failure (second materialize)";
    TLLM_CU_CHECK(cuStreamSynchronize(nullptr));

    vm.release();
    ASSERT_EQ(vm.status(), CUDAVirtualMemoryChunk::RELEASED);
    memoryReleased = getCurrentProcessMemoryInfo();
    if (memoryInfoAvailable())
    {
        ASSERT_EQ(memoryBegin, memoryReleased) << "release does not release memory";
    }
}

// Test BackedConfigurator refills memory correctly for both CPU and PINNED memory types
class VirtualMemoryOffloadConfigurator : public VirtualMemoryTest, public ::testing::WithParamInterface<MemoryType>
{
};

TEST_P(VirtualMemoryOffloadConfigurator, Test)
{
    MemoryType backType = GetParam();
    CUdeviceptr address{};
    std::size_t constexpr size = 4 * 1024 * 1024;
    TLLM_CU_CHECK(cuMemAddressReserve(&address, size, 0, {}, 0));

    CudaStream stream;

    CUDAVirtualMemoryChunk::CreatorPtr creator
        = std::make_unique<LocalCreator<>>(CUmemAllocationProp{CU_MEM_ALLOCATION_TYPE_PINNED, CU_MEM_HANDLE_TYPE_NONE,
                                               {
                                                   CU_MEM_LOCATION_TYPE_DEVICE,
                                                   0,
                                               }},
            size);

    CUDAVirtualMemoryChunk::Configurators configurators;
    configurators.push_back(std::make_unique<UnicastConfigurator>(address, size,
        CUmemAccessDesc{{
                            CU_MEM_LOCATION_TYPE_DEVICE,
                            0,
                        },
            CU_MEM_ACCESS_FLAGS_PROT_READWRITE}));
    configurators.push_back(std::make_unique<OffloadConfigurator>(address, size, backType, stream.get(), false));

    CUDAVirtualMemoryChunk vm(std::move(creator), std::move(configurators));

    std::vector<uint64_t> data(size / sizeof(uint64_t), 0);
    std::generate(data.begin(), data.end(), [engine = std::mt19937_64(address)]() mutable { return engine(); });

    vm.materialize();

    auto pointer = reinterpret_cast<void*>(address);
    auto result = cudaMemcpyAsync(pointer, data.data(), size, cudaMemcpyHostToDevice, stream.get());
    ASSERT_EQ(result, CUDA_SUCCESS) << "Copying memory returned failure";

    vm.release();

    vm.materialize();

    std::fill(data.begin(), data.end(), 0);
    result = cudaMemcpyAsync(data.data(), pointer, size, cudaMemcpyDeviceToHost, stream.get());
    stream.synchronize();
    ASSERT_EQ(result, CUDA_SUCCESS) << "Copying memory returned failure";

    auto engine = std::mt19937_64(static_cast<uint64_t>(address));
    for (size_t i = 0; i < data.size(); ++i)
    {
        ASSERT_EQ(data[i], engine()) << "Mismatched at index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Backends, VirtualMemoryOffloadConfigurator, ::testing::Values(MemoryType::kCPU, MemoryType::kPINNED));

// Test CUDAVirtualMemoryChunk calls creator and configurators in correct order
TEST_F(VirtualMemoryTest, TestOrder)
{
    // Order tracking - local counter to track call sequence
    int callOrder = 0;

    // OrderTrackingCreator that records when its methods are called
    class OrderTrackingCreator : public CUDAVirtualMemoryChunk::Creator
    {
    public:
        int& mCallOrder;
        int createOrder = -1;
        int releaseOrder = -1;
        CUmemGenericAllocationHandle createdHandle = 0;

        OrderTrackingCreator(int& callOrder)
            : mCallOrder(callOrder)
        {
        }

        CUmemGenericAllocationHandle create() override
        {
            createOrder = ++mCallOrder;
            createdHandle = 0xbaadf00dbaadf00d;
            return createdHandle;
        }

        void release(CUmemGenericAllocationHandle handle, bool destructing) override
        {
            releaseOrder = ++mCallOrder;
            ASSERT_EQ(handle, createdHandle);
        }
    };

    // OrderTrackingConfigurator that records when its methods are called
    class OrderTrackingConfigurator : public CUDAVirtualMemoryChunk::Configurator
    {
    public:
        int& mCallOrder;
        std::string name;
        int setupOrder = -1;
        int teardownOrder = -1;

        OrderTrackingConfigurator(int& callOrder, std::string n)
            : mCallOrder(callOrder)
            , name(std::move(n))
        {
        }

        void setup(CUmemGenericAllocationHandle handle) override
        {
            setupOrder = ++mCallOrder;
        }

        void teardown(CUmemGenericAllocationHandle handle, bool destructing) override
        {
            teardownOrder = ++mCallOrder;
        }
    };

    // Create creator and configurators
    auto creator = std::make_unique<OrderTrackingCreator>(callOrder);
    auto* creatorPtr = creator.get();

    auto config1 = std::make_unique<OrderTrackingConfigurator>(callOrder, "config1");
    auto config2 = std::make_unique<OrderTrackingConfigurator>(callOrder, "config2");
    auto config3 = std::make_unique<OrderTrackingConfigurator>(callOrder, "config3");
    auto* config1Ptr = config1.get();
    auto* config2Ptr = config2.get();
    auto* config3Ptr = config3.get();

    CUDAVirtualMemoryChunk::Configurators configurators;
    configurators.push_back(std::move(config1));
    configurators.push_back(std::move(config2));
    configurators.push_back(std::move(config3));

    CUDAVirtualMemoryChunk vm(std::move(creator), std::move(configurators));

    // Test materialize() order: creator.create() first, then configurators.setup() in order
    vm.materialize();

    // Verify materialize order
    EXPECT_EQ(creatorPtr->createOrder, 1); // creator.create() should be called first
    EXPECT_EQ(config1Ptr->setupOrder, 2);  // config1.setup() should be called second
    EXPECT_EQ(config2Ptr->setupOrder, 3);  // config2.setup() should be called third
    EXPECT_EQ(config3Ptr->setupOrder, 4);  // config3.setup() should be called fourth

    // Verify release() hasn't been called yet
    EXPECT_EQ(creatorPtr->releaseOrder, -1);
    EXPECT_EQ(config1Ptr->teardownOrder, -1);
    EXPECT_EQ(config2Ptr->teardownOrder, -1);
    EXPECT_EQ(config3Ptr->teardownOrder, -1);

    // Test release() order: configurators.teardown() in reverse order, then creator.release()
    vm.release();

    // Verify release order
    EXPECT_EQ(config3Ptr->teardownOrder, 5); // config3.teardown() should be called first (reverse order)
    EXPECT_EQ(config2Ptr->teardownOrder, 6); // config2.teardown() should be called second
    EXPECT_EQ(config1Ptr->teardownOrder, 7); // config1.teardown() should be called third
    EXPECT_EQ(creatorPtr->releaseOrder, 8);  // creator.release() should be called last
}

// Test CUDAVirtualMemoryChunk behaves correctly when exceptions were thrown
TEST_F(VirtualMemoryTest, TestException)
{
    // Dummy Creator that can be configured to throw on create() or release()
    class DummyCreator : public CUDAVirtualMemoryChunk::Creator
    {
    public:
        bool throwOnCreate = false;
        bool throwOnRelease = false;
        bool createCalled = false;
        bool releaseCalled = false;
        CUmemGenericAllocationHandle createdHandle = 0;

        CUmemGenericAllocationHandle create() override
        {
            createCalled = true;
            if (throwOnCreate)
            {
                throw DummyException();
            }
            createdHandle = 0xbaadf00dbaadf00d;
            return createdHandle;
        }

        void release(CUmemGenericAllocationHandle handle, bool destructing) override
        {
            releaseCalled = true;
            ASSERT_EQ(handle, createdHandle);
            if (throwOnRelease)
            {
                throw DummyException();
            }
        }
    };

    // Dummy Configurator that can be configured to throw on setup() or teardown()
    class DummyConfigurator : public CUDAVirtualMemoryChunk::Configurator
    {
    public:
        bool throwOnSetup = false;
        bool throwOnTeardown = false;
        bool setupCalled = false;
        bool teardownCalled = false;
        std::string name;

        DummyConfigurator(std::string n)
            : name(std::move(n))
        {
        }

        void setup(CUmemGenericAllocationHandle) override
        {
            setupCalled = true;
            if (throwOnSetup)
            {
                throw DummyException();
            }
        }

        void teardown(CUmemGenericAllocationHandle handle, bool destructing) override
        {
            teardownCalled = true;
            if (throwOnTeardown)
            {
                throw DummyException();
            }
        }
    };

    // Test 1: Exception in creator->create()
    {
        auto creator = std::make_unique<DummyCreator>();
        creator->throwOnCreate = true;
        auto* creatorPtr = creator.get();

        auto config1 = std::make_unique<DummyConfigurator>("config1");
        auto config2 = std::make_unique<DummyConfigurator>("config2");
        auto* config1Ptr = config1.get();
        auto* config2Ptr = config2.get();

        CUDAVirtualMemoryChunk::Configurators configurators;
        configurators.push_back(std::move(config1));
        configurators.push_back(std::move(config2));

        CUDAVirtualMemoryChunk vm(std::move(creator), std::move(configurators));

        // materialize() should throw due to creator->create() exception
        EXPECT_THROW(vm.materialize(), DummyException);

        // Verify creator->create() was called but no configurators were setup
        EXPECT_TRUE(creatorPtr->createCalled);
        EXPECT_FALSE(config1Ptr->setupCalled);
        EXPECT_FALSE(config2Ptr->setupCalled);

        // Internal state is still valid.
        // If the failure from creator is temporary, materialize() can be reattempted.
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::RELEASED);
    }

    // Test 2: Exception in first configurator setup()
    {
        auto creator = std::make_unique<DummyCreator>();
        auto* creatorPtr = creator.get();

        auto config1 = std::make_unique<DummyConfigurator>("config1");
        auto config2 = std::make_unique<DummyConfigurator>("config2");
        config1->throwOnSetup = true;
        auto* config1Ptr = config1.get();
        auto* config2Ptr = config2.get();

        CUDAVirtualMemoryChunk::Configurators configurators;
        configurators.push_back(std::move(config1));
        configurators.push_back(std::move(config2));

        CUDAVirtualMemoryChunk vm(std::move(creator), std::move(configurators));

        // materialize() should throw due to first configurator exception
        EXPECT_THROW(vm.materialize(), DummyException);

        // Verify creator->create() was called and first configurator setup() was called
        EXPECT_TRUE(creatorPtr->createCalled);
        EXPECT_TRUE(config1Ptr->setupCalled);
        EXPECT_FALSE(config2Ptr->setupCalled);

        // Status should be ERRORED
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::ERRORED);

        // release() should still work and only teardown what was set up
        vm.release();
        EXPECT_TRUE(creatorPtr->releaseCalled);
        EXPECT_FALSE(config1Ptr->teardownCalled); // Failed setup, so no teardown
        EXPECT_FALSE(config2Ptr->teardownCalled); // Never setup
    }

    // Test 3: Exception in second configurator setup()
    {
        auto creator = std::make_unique<DummyCreator>();
        auto* creatorPtr = creator.get();

        auto config1 = std::make_unique<DummyConfigurator>("config1");
        auto config2 = std::make_unique<DummyConfigurator>("config2");
        config2->throwOnSetup = true;
        auto* config1Ptr = config1.get();
        auto* config2Ptr = config2.get();

        CUDAVirtualMemoryChunk::Configurators configurators;
        configurators.push_back(std::move(config1));
        configurators.push_back(std::move(config2));

        CUDAVirtualMemoryChunk vm(std::move(creator), std::move(configurators));

        // materialize() should throw due to second configurator exception
        EXPECT_THROW(vm.materialize(), DummyException);

        // Verify both creator and first configurator were called
        EXPECT_TRUE(creatorPtr->createCalled);
        EXPECT_TRUE(config1Ptr->setupCalled);
        EXPECT_TRUE(config2Ptr->setupCalled);

        // Status should be ERRORED
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::ERRORED);

        // release() should teardown the first configurator (successful setup) but not the second
        vm.release();
        EXPECT_TRUE(creatorPtr->releaseCalled);
        EXPECT_TRUE(config1Ptr->teardownCalled);  // Successful setup, so teardown called
        EXPECT_FALSE(config2Ptr->teardownCalled); // Failed setup, so no teardown
    }

    // Test 4: Exception in configurator teardown() during release()
    {
        auto creator = std::make_unique<DummyCreator>();
        auto* creatorPtr = creator.get();

        auto config1 = std::make_unique<DummyConfigurator>("config1");
        auto config2 = std::make_unique<DummyConfigurator>("config2");
        config2->throwOnTeardown = true;
        auto* config1Ptr = config1.get();
        auto* config2Ptr = config2.get();

        CUDAVirtualMemoryChunk::Configurators configurators;
        configurators.push_back(std::move(config1));
        configurators.push_back(std::move(config2));

        CUDAVirtualMemoryChunk vm(std::move(creator), std::move(configurators));

        // materialize() should succeed
        vm.materialize();
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::MATERIALIZED);

        // release() should throw due to teardown exception but still complete cleanup
        EXPECT_THROW(vm.release(), DummyException);

        // Verify all teardown methods were called despite exception
        EXPECT_TRUE(config1Ptr->teardownCalled);
        EXPECT_TRUE(config2Ptr->teardownCalled);
        EXPECT_TRUE(creatorPtr->releaseCalled);

        // Status should be ERRORED due to exception
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::ERRORED);
    }

    // Test 5: Exception in creator->release()
    {
        auto creator = std::make_unique<DummyCreator>();
        creator->throwOnRelease = true;
        auto* creatorPtr = creator.get();

        auto config1 = std::make_unique<DummyConfigurator>("config1");
        auto* config1Ptr = config1.get();

        CUDAVirtualMemoryChunk::Configurators configurators;
        configurators.push_back(std::move(config1));

        CUDAVirtualMemoryChunk vm(std::move(creator), std::move(configurators));

        // materialize() should succeed
        vm.materialize();
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::MATERIALIZED);

        // release() should throw due to creator exception but still complete configurator cleanup
        EXPECT_THROW(vm.release(), DummyException);

        // Verify configurator teardown was called despite creator exception
        EXPECT_TRUE(config1Ptr->teardownCalled);
        EXPECT_TRUE(creatorPtr->releaseCalled);

        // Status should be ERRORED due to exception
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::ERRORED);
    }
}

// Test various class facilities
TEST_F(VirtualMemoryTest, TestFacilities)
{
    // Test default constructed CUDAVirtualMemoryChunk
    {
        CUDAVirtualMemoryChunk defaultVm;

        // Should be invalid
        EXPECT_FALSE(defaultVm);
        EXPECT_EQ(defaultVm.status(), CUDAVirtualMemoryChunk::INVALID);
    }

    CUdeviceptr address{};
    std::size_t constexpr size = 64 * 1024 * 1024;
    TLLM_CU_CHECK(cuMemAddressReserve(&address, size, 0, {}, 0));
    // Test move semantic
    {

        // Create original CUDAVirtualMemoryChunk
        CUDAVirtualMemoryChunk::CreatorPtr creator
            = std::make_unique<LocalCreator<>>(CUmemAllocationProp{CU_MEM_ALLOCATION_TYPE_PINNED,
                                                   CU_MEM_HANDLE_TYPE_NONE, {CU_MEM_LOCATION_TYPE_DEVICE, 0}},
                size);

        CUDAVirtualMemoryChunk::Configurators configurators;
        configurators.push_back(std::make_unique<UnicastConfigurator>(
            address, size, CUmemAccessDesc{{CU_MEM_LOCATION_TYPE_DEVICE, 0}, CU_MEM_ACCESS_FLAGS_PROT_READWRITE}));

        CUDAVirtualMemoryChunk original(std::move(creator), std::move(configurators));
        original.materialize();
        EXPECT_EQ(original.status(), CUDAVirtualMemoryChunk::MATERIALIZED);

        // Test move constructor
        CUDAVirtualMemoryChunk moved{std::move(original)};
        EXPECT_FALSE(original); // Original should be invalid after move
        EXPECT_TRUE(moved);     // Moved-to object should be valid
        EXPECT_EQ(moved.status(), CUDAVirtualMemoryChunk::MATERIALIZED);

        // Test move assignment
        CUDAVirtualMemoryChunk assigned;
        EXPECT_FALSE(assigned); // Default constructed, should be invalid

        assigned = std::move(moved);
        EXPECT_FALSE(moved);   // moved should be invalid after move
        EXPECT_TRUE(assigned); // assigned should be valid
        EXPECT_EQ(assigned.status(), CUDAVirtualMemoryChunk::MATERIALIZED);

        // Clean up
        assigned.release();
    }
}

// Test destructor
TEST_F(VirtualMemoryTest, TestDestructor)
{

    // Dummy Creator for testing destructor behavior
    class DummyCreator : public CUDAVirtualMemoryChunk::Creator
    {
    public:
        bool& createCalledRef;
        bool& releaseCalledRef;
        CUmemGenericAllocationHandle createdHandle = 0;

        DummyCreator(bool& createRef, bool& releaseRef)
            : createCalledRef(createRef)
            , releaseCalledRef(releaseRef)
        {
        }

        CUmemGenericAllocationHandle create() override
        {
            createCalledRef = true;
            createdHandle = 0xbaadf00dbaadf00d;
            return createdHandle;
        }

        void release(CUmemGenericAllocationHandle handle, bool destructing) override
        {
            releaseCalledRef = true;
            ASSERT_EQ(handle, createdHandle);
        }
    };

    // Dummy Configurator for testing destructor behavior
    class DummyConfigurator : public CUDAVirtualMemoryChunk::Configurator
    {
    public:
        bool& setupCalledRef;
        bool& teardownCalledRef;
        std::string name;

        DummyConfigurator(std::string n, bool& setupRef, bool& teardownRef)
            : setupCalledRef(setupRef)
            , teardownCalledRef(teardownRef)
            , name(std::move(n))
        {
        }

        void setup(CUmemGenericAllocationHandle) override
        {
            setupCalledRef = true;
        }

        void teardown(CUmemGenericAllocationHandle, bool) override
        {
            teardownCalledRef = true;
        }
    };

    // Test destructor calls release automatically for materialized memory
    {
        bool createCalled = false;
        bool releaseCalled = false;
        bool setupCalled = false;
        bool teardownCalled = false;

        auto creator = std::make_unique<DummyCreator>(createCalled, releaseCalled);
        auto config1 = std::make_unique<DummyConfigurator>("config1", setupCalled, teardownCalled);

        CUDAVirtualMemoryChunk::Configurators configurators;
        configurators.push_back(std::move(config1));

        alignas(CUDAVirtualMemoryChunk) std::byte storage[sizeof(CUDAVirtualMemoryChunk)];
        CUDAVirtualMemoryChunk* vm = new (storage) CUDAVirtualMemoryChunk(std::move(creator), std::move(configurators));

        vm->materialize();

        // Verify materialize was called
        EXPECT_TRUE(createCalled);
        EXPECT_TRUE(setupCalled);
        EXPECT_FALSE(releaseCalled);
        EXPECT_FALSE(teardownCalled);
        EXPECT_EQ(vm->status(), CUDAVirtualMemoryChunk::MATERIALIZED);

        vm->~CUDAVirtualMemoryChunk();

        // Verify destructor called release
        EXPECT_TRUE(releaseCalled);
        EXPECT_TRUE(teardownCalled);
    }

    // Test destructor doesn't double-release for manually released memory
    {
        // Local variables to track calls (persist after object destruction)
        bool createCalled = false;
        bool releaseCalled = false;
        bool setupCalled = false;
        bool teardownCalled = false;

        auto creator = std::make_unique<DummyCreator>(createCalled, releaseCalled);
        auto config1 = std::make_unique<DummyConfigurator>("config1", setupCalled, teardownCalled);

        CUDAVirtualMemoryChunk::Configurators configurators;
        configurators.push_back(std::move(config1));

        alignas(CUDAVirtualMemoryChunk) std::byte storage[sizeof(CUDAVirtualMemoryChunk)];
        auto* vm = new (storage) CUDAVirtualMemoryChunk(std::move(creator), std::move(configurators));

        vm->materialize();
        vm->release(); // Manual release

        // Verify manual release was called
        EXPECT_TRUE(releaseCalled);
        EXPECT_TRUE(teardownCalled);
        EXPECT_EQ(vm->status(), CUDAVirtualMemoryChunk::RELEASED);

        // Reset flags to verify destructor doesn't call release again
        releaseCalled = false;
        teardownCalled = false;

        vm->~CUDAVirtualMemoryChunk();

        // Verify destructor did NOT call release again (no double-release)
        EXPECT_FALSE(releaseCalled);
        EXPECT_FALSE(teardownCalled);
    }

    // Test destructor behavior with ERRORED state
    {
        // Local variables to track calls (persist after object destruction)
        bool createCalled = false;
        bool releaseCalled = false;
        bool config1SetupCalled = false;
        bool config1TeardownCalled = false;
        bool throwingSetupCalled = false;
        bool throwingTeardownCalled = false;

        class ThrowingConfigurator : public CUDAVirtualMemoryChunk::Configurator
        {
        public:
            bool& setupCalledRef;
            bool& teardownCalledRef;

            ThrowingConfigurator(bool& setupRef, bool& teardownRef)
                : setupCalledRef(setupRef)
                , teardownCalledRef(teardownRef)
            {
            }

            void setup(CUmemGenericAllocationHandle) override
            {
                setupCalledRef = true;
                throw DummyException();
            }

            void teardown(CUmemGenericAllocationHandle, bool) override
            {
                teardownCalledRef = true;
            }
        };

        auto creator = std::make_unique<DummyCreator>(createCalled, releaseCalled);
        auto config1 = std::make_unique<DummyConfigurator>("config1", config1SetupCalled, config1TeardownCalled);
        auto throwingConfig = std::make_unique<ThrowingConfigurator>(throwingSetupCalled, throwingTeardownCalled);

        CUDAVirtualMemoryChunk::Configurators configurators;
        configurators.push_back(std::move(config1));
        configurators.push_back(std::move(throwingConfig));

        alignas(CUDAVirtualMemoryChunk) std::byte storage[sizeof(CUDAVirtualMemoryChunk)];
        auto* vm = new (storage) CUDAVirtualMemoryChunk(std::move(creator), std::move(configurators));

        // Materialize should throw and leave VM in ERRORED state
        EXPECT_THROW(vm->materialize(), DummyException);
        EXPECT_EQ(vm->status(), CUDAVirtualMemoryChunk::ERRORED);

        // Verify partial setup occurred
        EXPECT_TRUE(createCalled);
        EXPECT_TRUE(config1SetupCalled);
        EXPECT_TRUE(throwingSetupCalled);
        EXPECT_FALSE(releaseCalled);

        vm->~CUDAVirtualMemoryChunk();

        // Verify destructor called release to clean up the errored state
        EXPECT_TRUE(releaseCalled);
        EXPECT_TRUE(config1TeardownCalled);
        // throwingConfig's teardown should NOT be called since setup failed
        EXPECT_FALSE(throwingTeardownCalled);
    }
}

// Test edge cases and error scenarios
TEST_F(VirtualMemoryTest, TestEdgeCases)
{
    // Dummy Creator for testing edge cases
    class DummyCreator : public CUDAVirtualMemoryChunk::Creator
    {
    public:
        CUmemGenericAllocationHandle createdHandle = 0xbaadf00dbaadf00d;

        CUmemGenericAllocationHandle create() override
        {
            return createdHandle;
        }

        void release(CUmemGenericAllocationHandle handle, bool destructing) override
        {
            ASSERT_EQ(handle, createdHandle);
        }
    };

    // Test multiple materialize calls (should throw)
    {
        auto creator = std::make_unique<DummyCreator>();
        CUDAVirtualMemoryChunk vm(std::move(creator), {});

        vm.materialize();
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::MATERIALIZED);

        // Second materialize should throw
        EXPECT_THROW(vm.materialize(), tc::TllmException);
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::MATERIALIZED);

        vm.release();
    }

    // Test multiple release calls (should throw)
    {
        auto creator = std::make_unique<DummyCreator>();
        CUDAVirtualMemoryChunk vm(std::move(creator), {});

        vm.materialize();
        vm.release();
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::RELEASED);

        // Second release should throw
        EXPECT_THROW(vm.release(), tc::TllmException);
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::RELEASED);
    }

    // Test release on RELEASED state (should throw)
    {
        auto creator = std::make_unique<DummyCreator>();
        CUDAVirtualMemoryChunk vm(std::move(creator), {});

        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::RELEASED);
        EXPECT_THROW(vm.release(), tc::TllmException); // Should throw on RELEASED state
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::RELEASED);
    }

    // Test materialize on ERRORED state after exception recovery
    {
        // Create a VM that will go into ERRORED state
        class ThrowingConfigurator : public CUDAVirtualMemoryChunk::Configurator
        {
        public:
            bool shouldThrow = true;

            void setup(CUmemGenericAllocationHandle) override
            {
                if (shouldThrow)
                {
                    throw DummyException();
                }
            }

            void teardown(CUmemGenericAllocationHandle, bool) override {}
        };

        auto creator = std::make_unique<DummyCreator>();
        auto throwingConfig = std::make_unique<ThrowingConfigurator>();

        CUDAVirtualMemoryChunk::Configurators configurators;
        configurators.push_back(std::move(throwingConfig));

        CUDAVirtualMemoryChunk vm(std::move(creator), std::move(configurators));

        // First materialize should throw and leave VM in ERRORED state
        EXPECT_THROW(vm.materialize(), DummyException);
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::ERRORED);

        // Should be able to release from ERRORED state
        vm.release();
        EXPECT_EQ(vm.status(), CUDAVirtualMemoryChunk::RELEASED);
    }
}

class VirtualMemoryManagerTest : public VirtualMemoryTestBase // NOLINT(cppcoreguidelines-pro-type-member-init)
{
    using Base = VirtualMemoryTestBase;

protected:
    auto& entries()
    {
        return mVMManager->mEntries;
    }

    auto& memories()
    {
        return mVMManager->mMemories;
    }

    auto& badHandles()
    {
        return mVMManager->mBadHandles;
    }

    void SetUp() override
    {
        this->Base::SetUp();
        mVMManager = std::make_unique<CudaVirtualMemoryManager>();
    }

    void TearDown() override
    {
        this->Base::TearDown();
        ASSERT_TRUE(!mVMManager || entries().size() == 0) << "Leftover memory in manager";
    }

    std::unique_ptr<CudaVirtualMemoryManager> mVMManager = nullptr;
};

TEST_F(VirtualMemoryManagerTest, TestBasic)
{
    CUdeviceptr address{};
    std::size_t constexpr size = 256 * 1024 * 1024;
    TLLM_CU_CHECK(cuMemAddressReserve(&address, size, 0, {}, 0));

    uintptr_t handle = static_cast<uintptr_t>(address);
    std::string tag = "test_tag";

    CUDAVirtualMemoryChunk::CreatorPtr creator
        = std::make_unique<LocalCreator<>>(CUmemAllocationProp{CU_MEM_ALLOCATION_TYPE_PINNED, CU_MEM_HANDLE_TYPE_NONE,
                                               {
                                                   CU_MEM_LOCATION_TYPE_DEVICE,
                                                   0,
                                               }},
            size);

    CUDAVirtualMemoryChunk::Configurators configurators;
    configurators.push_back(std::make_unique<UnicastConfigurator>(address, size,
        CUmemAccessDesc{{
                            CU_MEM_LOCATION_TYPE_DEVICE,
                            0,
                        },
            CU_MEM_ACCESS_FLAGS_PROT_READWRITE}));

    auto memoryBegin = getCurrentProcessMemoryInfo();

    // Add to manager - this automatically materializes
    mVMManager->add(handle, tag, std::move(creator), std::move(configurators));

    auto memoryMaterialized = getCurrentProcessMemoryInfo();
    if (memoryInfoAvailable())
    {
        ASSERT_EQ(memoryBegin + size, memoryMaterialized) << "add/materialize does not allocate memory";
    }

    // Test memory access after materialization
    auto result = cuMemsetD8_v2(address, 255, size);
    ASSERT_EQ(result, CUDA_SUCCESS) << "Accessing memory returned failure (first materialize)";
    TLLM_CU_CHECK(cuStreamSynchronize(nullptr));

    // Release memory through manager
    auto releaseCount = mVMManager->releaseWithTag(tag);
    ASSERT_EQ(releaseCount, 1) << "Expected to release 1 memory object";

    auto memoryReleased = getCurrentProcessMemoryInfo();
    if (memoryInfoAvailable())
    {
        ASSERT_EQ(memoryBegin, memoryReleased) << "releaseWithTag does not release memory";
    }

    // Materialize again through manager
    auto materializeCount = mVMManager->materializeWithTag(tag);
    ASSERT_EQ(materializeCount, 1) << "Expected to materialize 1 memory object";

    auto memoryRematerialized = getCurrentProcessMemoryInfo();
    if (memoryInfoAvailable())
    {
        ASSERT_EQ(memoryBegin + size, memoryRematerialized) << "materializeWithTag does not allocate memory";
    }

    // Test memory access after rematerialization
    result = cuMemsetD8_v2(address, 255, size);
    ASSERT_EQ(result, CUDA_SUCCESS) << "Accessing memory returned failure (second materialize)";
    TLLM_CU_CHECK(cuStreamSynchronize(nullptr));

    // Clean up - remove from manager
    {
        auto removedMemory = mVMManager->remove(handle);
        ASSERT_TRUE(removedMemory) << "Expected to successfully remove memory from manager";
    }

    auto memoryAfterRemove = getCurrentProcessMemoryInfo();
    if (memoryInfoAvailable())
    {
        ASSERT_EQ(memoryBegin, memoryAfterRemove) << "remove does not release memory";
    }

    auto unknownMemory = mVMManager->remove(0);
    ASSERT_FALSE(unknownMemory) << "Expect invalid memory for unknown handle";
}

TEST_F(VirtualMemoryManagerTest, TestTags)
{
    // Dummy Creator for testing tag functionality
    class DummyCreator : public CUDAVirtualMemoryChunk::Creator
    {
    public:
        bool createCalled = false;
        bool releaseCalled = false;
        CUmemGenericAllocationHandle createdHandle = 0xbaadf00dbaadf00d;

        CUmemGenericAllocationHandle create() override
        {
            createCalled = true;
            return createdHandle;
        }

        void release(CUmemGenericAllocationHandle handle, bool destructing) override
        {
            releaseCalled = true;
            ASSERT_EQ(handle, createdHandle);
        }
    };

    // Create creators for different virtual memories
    auto creator1 = std::make_unique<DummyCreator>();
    auto creator2 = std::make_unique<DummyCreator>();
    auto creator3 = std::make_unique<DummyCreator>();
    auto creator4 = std::make_unique<DummyCreator>();

    // Keep pointers to track state
    auto* creator1Ptr = creator1.get();
    auto* creator2Ptr = creator2.get();
    auto* creator3Ptr = creator3.get();
    auto* creator4Ptr = creator4.get();

    mVMManager->add(0x1000, "tag_A", std::move(creator1), {});
    mVMManager->add(0x2000, "tag_B", std::move(creator2), {});
    mVMManager->add(0x3000, "tag_A", std::move(creator3), {});
    mVMManager->add(0x4000, "tag_C", std::move(creator4), {});

    // All should be materialized initially (since add() materializes automatically)
    EXPECT_TRUE(creator1Ptr->createCalled);
    EXPECT_TRUE(creator2Ptr->createCalled);
    EXPECT_TRUE(creator3Ptr->createCalled);
    EXPECT_TRUE(creator4Ptr->createCalled);

    // Reset create flags to test materializeWithTag later
    creator1Ptr->createCalled = false;
    creator2Ptr->createCalled = false;
    creator3Ptr->createCalled = false;
    creator4Ptr->createCalled = false;

    // Test releaseWithTag - should release only memories with "tag_A"
    auto releaseCount = mVMManager->releaseWithTag("tag_A");
    EXPECT_EQ(releaseCount, 2); // Should release 2 memories with tag_A

    // Verify only tag_A memories were released
    EXPECT_TRUE(creator1Ptr->releaseCalled);  // tag_A
    EXPECT_FALSE(creator2Ptr->releaseCalled); // tag_B
    EXPECT_TRUE(creator3Ptr->releaseCalled);  // tag_A
    EXPECT_FALSE(creator4Ptr->releaseCalled); // tag_C

    // Test materializeWithTag - should materialize only memories with "tag_A"
    auto materializeCount = mVMManager->materializeWithTag("tag_A");
    EXPECT_EQ(materializeCount, 2); // Should materialize 2 memories with tag_A

    // Verify only tag_A memories were materialized
    EXPECT_TRUE(creator1Ptr->createCalled);  // tag_A
    EXPECT_FALSE(creator2Ptr->createCalled); // tag_B
    EXPECT_TRUE(creator3Ptr->createCalled);  // tag_A
    EXPECT_FALSE(creator4Ptr->createCalled); // tag_C

    // Reset flags and test releasing with a different tag
    creator2Ptr->releaseCalled = false;
    releaseCount = mVMManager->releaseWithTag("tag_B");
    EXPECT_EQ(releaseCount, 1);              // Should release 1 memory with tag_B
    EXPECT_TRUE(creator2Ptr->releaseCalled); // tag_B should now be released

    // Test with non-existent tag
    releaseCount = mVMManager->releaseWithTag("nonexistent_tag");
    EXPECT_EQ(releaseCount, 0); // Should release 0 memories

    materializeCount = mVMManager->materializeWithTag("nonexistent_tag");
    EXPECT_EQ(materializeCount, 0); // Should materialize 0 memories

    // Clean up - remove all memories
    mVMManager->remove(0x1000);
    mVMManager->remove(0x2000);
    mVMManager->remove(0x3000);
    mVMManager->remove(0x4000);
}

TEST_F(VirtualMemoryManagerTest, TestAddException)
{
    // Dummy Creator that succeeds
    class DummyCreator : public CUDAVirtualMemoryChunk::Creator
    {
    public:
        CUmemGenericAllocationHandle createdHandle = 0xbaadf00dbaadf00d;

        CUmemGenericAllocationHandle create() override
        {
            return createdHandle;
        }

        void release(CUmemGenericAllocationHandle handle, bool destructing) override
        {
            ASSERT_EQ(handle, createdHandle);
        }
    };

    // Dummy Configurator that throws during setup
    class ThrowingConfigurator : public CUDAVirtualMemoryChunk::Configurator
    {
    public:
        void setup(CUmemGenericAllocationHandle) override
        {
            throw DummyException();
        }

        void teardown(CUmemGenericAllocationHandle, bool) override
        {
            ASSERT_TRUE(false) << "Unreachable";
        }
    };

    uintptr_t handle = 0x12345678;
    std::string tag = "test_tag";

    // Verify initial state is clean
    EXPECT_TRUE(memories().empty());
    EXPECT_TRUE(entries().empty());
    EXPECT_TRUE(badHandles().empty());

    auto creator = std::make_unique<DummyCreator>();
    CUDAVirtualMemoryChunk::Configurators configurators;
    configurators.push_back(std::make_unique<ThrowingConfigurator>());

    // add() should throw because materialize() will fail due to ThrowingConfigurator
    EXPECT_THROW(mVMManager->add(handle, tag, std::move(creator), std::move(configurators)), DummyException);

    // Verify that the manager state is clean after the exception
    // The ScopeGuards in add() should have cleaned up properly
    EXPECT_TRUE(memories().empty()) << "mMemories should be empty after failed add()";
    EXPECT_TRUE(entries().empty()) << "mEntries should be empty after failed add()";
    EXPECT_TRUE(badHandles().empty()) << "mBadHandles should be empty after failed add()";

    // Test that we can successfully add a memory with the same handle after the failure
    auto successCreator = std::make_unique<DummyCreator>();
    CUDAVirtualMemoryChunk::Configurators successConfigurators; // Empty configurators should work

    // This should succeed without throwing
    EXPECT_NO_THROW(mVMManager->add(handle, tag, std::move(successCreator), std::move(successConfigurators)));

    // Verify that the manager now has the entry
    EXPECT_EQ(memories().size(), 1);
    EXPECT_EQ(entries().size(), 1);
    EXPECT_TRUE(badHandles().empty());

    // Clean up
    auto removedMemory = mVMManager->remove(handle);
    EXPECT_TRUE(removedMemory);
}

TEST_F(VirtualMemoryManagerTest, TestMaterializeException)
{
    // State structure to track create/release order and can throw on a specific call
    struct CreatorState
    {
        int& createCounter;       // Reference to shared counter
        int throwOnCreateIdx = 0; // 1-based index to throw on create
        int myCreateIdx = INT_MAX;
        bool createCalled = false;
        bool releaseCalled = false;
        CUmemGenericAllocationHandle createdHandle = 0xbaadf00dbaadf00d;

        CreatorState(int& sharedCounter)
            : createCounter(sharedCounter)
        {
        }
    };

    // Dummy Creator that uses external state
    class TestMatEx_DummyCreator : public CUDAVirtualMemoryChunk::Creator
    {
    public:
        CreatorState& state;

        TestMatEx_DummyCreator(CreatorState& state)
            : state(state)
        {
        }

        CUmemGenericAllocationHandle create() override
        {
            state.createCalled = true;
            state.myCreateIdx = ++state.createCounter;
            if (state.throwOnCreateIdx > 0 && state.myCreateIdx == state.throwOnCreateIdx)
            {
                throw DummyException();
            }
            return state.createdHandle;
        }

        void release(CUmemGenericAllocationHandle handle, bool destructing) override
        {
            state.releaseCalled = true;
            ASSERT_EQ(handle, state.createdHandle);
        }
    };

    // Create shared counter
    int sharedCreateCounter = 0;

    // Create state objects for each creator
    CreatorState state1(sharedCreateCounter);
    CreatorState state2(sharedCreateCounter);
    CreatorState state3(sharedCreateCounter);

    // We want the second memory (by create order) to throw
    state1.throwOnCreateIdx = 2;
    state2.throwOnCreateIdx = 2;
    state3.throwOnCreateIdx = 2;

    // Create creators and configurators
    auto creator1 = std::make_unique<TestMatEx_DummyCreator>(state1);
    auto creator2 = std::make_unique<TestMatEx_DummyCreator>(state2);
    auto creator3 = std::make_unique<TestMatEx_DummyCreator>(state3);

    // Add memories to manager in RELEASED state (don't auto-materialize by constructing manually)
    CUDAVirtualMemoryChunk vm1(std::move(creator1), {});
    CUDAVirtualMemoryChunk vm2(std::move(creator2), {});
    CUDAVirtualMemoryChunk vm3(std::move(creator3), {});

    mVMManager->add(0x1000, "test_tag", std::move(vm1));
    mVMManager->add(0x2000, "test_tag", std::move(vm2));
    mVMManager->add(0x3000, "test_tag", std::move(vm3));

    // Verify initial state is clean
    EXPECT_TRUE(badHandles().empty());

    // materializeWithTag should stop at the first exception (second memory by create order)
    // and attempt to rollback the first memory that succeeded
    EXPECT_THROW(mVMManager->materializeWithTag("test_tag"), DummyException);

    // Find which creators were called and in what order
    std::vector<std::pair<uintptr_t, CreatorState*>> creators
        = {{0x1000, &state1}, {0x2000, &state2}, {0x3000, &state3}};
    // Sort by myCreateIdx (nonzero means create was called)
    std::sort(creators.begin(), creators.end(),
        [](auto const& a, auto const& b) { return a.second->myCreateIdx < b.second->myCreateIdx; });

    // The first memory (by create order) should have been materialized then released during rollback
    auto* first = creators[0].second;
    EXPECT_TRUE(first->createCalled);
    EXPECT_TRUE(first->releaseCalled); // Rolled back
    // The second memory should have thrown during setup, so creator was called but setup failed
    auto* second = creators[1].second;
    EXPECT_TRUE(second->createCalled);
    EXPECT_FALSE(second->releaseCalled);
    // The third memory should not have been touched (myCreateIdx == 0)
    auto* third = creators[2].second;
    EXPECT_FALSE(third->createCalled);
    EXPECT_FALSE(third->releaseCalled);

    // The handle of the memory that threw should be the second one's handle
    uintptr_t thrownHandle = creators[1].first;

    // Verify bad handles tracking - memories that threw exceptions should be removed
    auto badHandles = mVMManager->retrieveBadHandles();
    EXPECT_EQ(badHandles.size(), 1);
    EXPECT_EQ(badHandles[0], thrownHandle);

    // Verify the memory that threw was removed from the manager
    auto removedMem = mVMManager->remove(thrownHandle);
    EXPECT_FALSE(removedMem); // Should have been removed due to exception

    // The other two memories should still be in manager
    for (int i = 0; i < 3; ++i)
    {
        if (creators[i].first != thrownHandle)
        {
            auto removed = mVMManager->remove(creators[i].first);
            EXPECT_TRUE(removed);
        }
    }
}

TEST_F(VirtualMemoryManagerTest, TestReleaseException)
{
    // State structure to track create/release calls
    struct CreatorState
    {
        bool createCalled = false;
        bool releaseCalled = false;
        int& releaseCounter;
        int throwOnReleaseCount;
        CUmemGenericAllocationHandle createdHandle = 0xbaadf00dbaadf00d;

        CreatorState(int& counter, int throwCount)
            : releaseCounter(counter)
            , throwOnReleaseCount(throwCount)
        {
        }
    };

    // State structure to track setup/teardown calls
    struct ConfiguratorState
    {
        bool setupCalled = false;
        bool teardownCalled = false;
        int& teardownCounter;
        int throwOnTeardownCount;

        ConfiguratorState(int& counter, int throwCount)
            : teardownCounter(counter)
            , throwOnTeardownCount(throwCount)
        {
        }
    };

    // Dummy Creator that succeeds
    class DummyCreator : public CUDAVirtualMemoryChunk::Creator
    {
    public:
        CreatorState& state;

        DummyCreator(CreatorState& state)
            : state(state)
        {
        }

        CUmemGenericAllocationHandle create() override
        {
            state.createCalled = true;
            return state.createdHandle;
        }

        void release(CUmemGenericAllocationHandle handle, bool destructing) override
        {
            state.releaseCalled = true;
            ASSERT_EQ(handle, state.createdHandle);
            if (++state.releaseCounter == state.throwOnReleaseCount)
            {
                throw DummyException();
            }
        }
    };

    // Dummy Configurator that succeeds
    class DummyConfigurator : public CUDAVirtualMemoryChunk::Configurator
    {
    public:
        ConfiguratorState& state;

        DummyConfigurator(ConfiguratorState& state)
            : state(state)
        {
        }

        void setup(CUmemGenericAllocationHandle) override
        {
            state.setupCalled = true;
        }

        void teardown(CUmemGenericAllocationHandle, bool) override
        {
            state.teardownCalled = true;
            if (++state.teardownCounter == state.throwOnTeardownCount)
            {
                throw DummyException();
            }
        }
    };

    // Create counters for tracking release/teardown calls
    int releaseCounter = 0;
    int teardownCounter = 0;

    // Create state objects for each creator and configurator
    CreatorState state1(releaseCounter, 2);             // Throw on 2nd release
    CreatorState state2(releaseCounter, 2);             // Throw on 2nd release
    CreatorState state3(releaseCounter, 2);             // Throw on 2nd release
    CreatorState state4(releaseCounter, 2);             // Throw on 2nd release

    ConfiguratorState configState1(teardownCounter, 3); // Throw on 3rd teardown
    ConfiguratorState configState2(teardownCounter, 3); // Throw on 3rd teardown
    ConfiguratorState configState3(teardownCounter, 3); // Throw on 3rd teardown
    ConfiguratorState configState4(teardownCounter, 3); // Throw on 3rd teardown

    // Create creators and configurators
    auto creator1 = std::make_unique<DummyCreator>(state1);
    auto creator2 = std::make_unique<DummyCreator>(state2);
    auto creator3 = std::make_unique<DummyCreator>(state3);
    auto creator4 = std::make_unique<DummyCreator>(state4);

    auto config1 = std::make_unique<DummyConfigurator>(configState1);
    auto config2 = std::make_unique<DummyConfigurator>(configState2);
    auto config3 = std::make_unique<DummyConfigurator>(configState3);
    auto config4 = std::make_unique<DummyConfigurator>(configState4);

    CUDAVirtualMemoryChunk::Configurators configurators1;
    configurators1.push_back(std::move(config1));

    CUDAVirtualMemoryChunk::Configurators configurators2;
    configurators2.push_back(std::move(config2));

    CUDAVirtualMemoryChunk::Configurators configurators3;
    configurators3.push_back(std::move(config3));

    CUDAVirtualMemoryChunk::Configurators configurators4;
    configurators4.push_back(std::move(config4));

    mVMManager->add(0x1000, "test_tag", std::move(creator1), std::move(configurators1));
    mVMManager->add(0x2000, "test_tag", std::move(creator2), std::move(configurators2));
    mVMManager->add(0x3000, "test_tag", std::move(creator3), std::move(configurators3));
    mVMManager->add(0x4000, "other_tag", std::move(creator4), std::move(configurators4));

    // Verify initial state
    EXPECT_TRUE(badHandles().empty());

    // releaseWithTag should call release on all memories with "test_tag"
    // and continue despite exceptions
    EXPECT_THROW(mVMManager->releaseWithTag("test_tag"), DummyException);

    // Verify behavior:
    // - All memories with "test_tag" should have had release() attempted
    EXPECT_TRUE(state1.releaseCalled);
    EXPECT_TRUE(configState1.teardownCalled);

    EXPECT_TRUE(state2.releaseCalled);
    EXPECT_TRUE(configState2.teardownCalled);

    EXPECT_TRUE(state3.releaseCalled);
    EXPECT_TRUE(configState3.teardownCalled);

    // - Memory with different tag should not be affected
    EXPECT_FALSE(state4.releaseCalled);
    EXPECT_FALSE(configState4.teardownCalled);

    // Verify bad handles tracking - memories that threw exceptions should be removed
    auto badHandles = mVMManager->retrieveBadHandles();
    EXPECT_EQ(badHandles.size(), 2);
    EXPECT_NE(std::find(badHandles.begin(), badHandles.end(), 0x2000), badHandles.end());
    EXPECT_NE(std::find(badHandles.begin(), badHandles.end(), 0x3000), badHandles.end());

    // Verify the memories were removed from the manager
    auto removedMem1 = mVMManager->remove(0x1000);
    auto removedMem2 = mVMManager->remove(0x2000);
    auto removedMem3 = mVMManager->remove(0x3000);
    auto removedMem4 = mVMManager->remove(0x4000);

    EXPECT_TRUE(removedMem1);  // Should have been removed due to exception
    EXPECT_FALSE(removedMem2); // Should have been removed due to exception
    EXPECT_FALSE(removedMem3); // Should have been removed due to exception
    EXPECT_TRUE(removedMem4);  // Should still be in manager (different tag, not affected)
}

TEST_F(VirtualMemoryManagerTest, TestCudaVirtualMemoryAllocator)
{
    std::size_t constexpr size = 64 * 1024 * 1024; // 64 MB
    std::string tag = "test_allocator_tag";

    // Create a CUDA stream for the allocator
    CudaStream stream;
    auto streamPtr = std::make_shared<CudaStream>(std::move(stream));

    // Create configuration for the virtual address allocator
    auto config = std::make_shared<CudaVirtualMemoryAllocator::Configuration>(
        *mVMManager.get(), tag, CudaVirtualMemoryAllocator::RestoreMode::NONE, streamPtr);

    auto memoryBegin = getCurrentProcessMemoryInfo();

    // Create a buffer using the virtual address allocator
    auto buffer = std::make_unique<VirtualAddressDeviceBuffer>(
        size, nvinfer1::DataType::kINT8, CudaVirtualMemoryAllocator{config});

    auto memoryAfterAllocation = getCurrentProcessMemoryInfo();
    if (memoryInfoAvailable())
    {
        ASSERT_EQ(memoryBegin + size, memoryAfterAllocation) << "Buffer allocation does not allocate memory";
    }

    // Test that we can access the buffer data
    ASSERT_NE(buffer->data(), nullptr) << "Buffer data should not be null";
    ASSERT_EQ(buffer->getSize(), size) << "Buffer size should match requested size";
    ASSERT_EQ(buffer->getDataType(), nvinfer1::DataType::kINT8) << "Buffer data type should be INT8";
    ASSERT_EQ(buffer->getMemoryType(), MemoryType::kGPU) << "Buffer memory type should be GPU";

    // Test memory access by setting memory to a known pattern
    auto devicePtr = reinterpret_cast<CUdeviceptr>(buffer->data());
    auto result = cuMemsetD8_v2(devicePtr, 0xAB, size);
    ASSERT_EQ(result, CUDA_SUCCESS) << "Memory access should succeed";
    TLLM_CU_CHECK(cuStreamSynchronize(nullptr));

    // Test releasing memory with tag - this should free the virtual memory
    auto releaseCount = mVMManager->releaseWithTag(tag);
    ASSERT_EQ(releaseCount, 1) << "Expected to release 1 memory object";

    auto memoryAfterRelease = getCurrentProcessMemoryInfo();
    if (memoryInfoAvailable())
    {
        ASSERT_EQ(memoryBegin, memoryAfterRelease) << "Release should free the memory";
    }

    // Test materializing memory with tag - this should re-allocate the virtual memory
    auto materializeCount = mVMManager->materializeWithTag(tag);
    ASSERT_EQ(materializeCount, 1) << "Expected to materialize 1 memory object";

    auto memoryAfterMaterialize = getCurrentProcessMemoryInfo();
    if (memoryInfoAvailable())
    {
        ASSERT_EQ(memoryBegin + size, memoryAfterMaterialize) << "Materialize should allocate memory";
    }

    // Test memory access again after rematerialization
    result = cuMemsetD8_v2(devicePtr, 0xCD, size);
    ASSERT_EQ(result, CUDA_SUCCESS) << "Memory access should succeed after rematerialization";
    TLLM_CU_CHECK(cuStreamSynchronize(nullptr));

    // Clean up by destroying the buffer (this should automatically clean up the virtual memory)
    buffer.reset();

    auto memoryAfterCleanup = getCurrentProcessMemoryInfo();
    if (memoryInfoAvailable())
    {
        ASSERT_EQ(memoryBegin, memoryAfterCleanup) << "Buffer destruction should free memory";
    }
}
