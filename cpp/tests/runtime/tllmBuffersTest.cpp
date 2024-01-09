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

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/batch_manager/namedTensor.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/tllmBuffers.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;
namespace tb = tensorrt_llm::batch_manager;

class TllmBuffersTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();
    }

    void TearDown() override {}

    int mDeviceCount;
};

TEST_F(TllmBuffersTest, Stream)
{
    if (mDeviceCount == 0)
        GTEST_SKIP();

    CudaStream stream{};
    EXPECT_NE(stream.get(), nullptr);
    auto ptr = std::make_shared<CudaStream>();
    EXPECT_NE(ptr->get(), nullptr);
    EXPECT_GE(ptr->getDevice(), 0);
    CudaStream lease{ptr->get(), ptr->getDevice(), false};
    EXPECT_EQ(lease.get(), ptr->get());
}

TEST_F(TllmBuffersTest, CudaAllocator)
{
    auto constexpr size = 1024;
    CudaAllocator allocator{};
    auto& counters = MemoryCounters::getInstance();
    EXPECT_EQ(counters.getGpu(), 0);
    auto ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(counters.getGpu(), size);
    EXPECT_EQ(counters.getGpuDiff(), size);
    EXPECT_NO_THROW(allocator.deallocate(ptr, size));
    EXPECT_EQ(counters.getGpu(), 0);
    EXPECT_EQ(counters.getGpuDiff(), -size);
    EXPECT_EQ(allocator.getMemoryType(), MemoryType::kGPU);
    EXPECT_THROW(allocator.deallocate(ptr, size), std::runtime_error);
}

TEST_F(TllmBuffersTest, PinnedAllocator)
{
    auto constexpr size = 1024;
    PinnedAllocator allocator{};
    auto& counters = MemoryCounters::getInstance();
    EXPECT_EQ(counters.getPinned(), 0);
    auto ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(counters.getPinned(), size);
    EXPECT_EQ(counters.getPinnedDiff(), size);
    EXPECT_NO_THROW(allocator.deallocate(ptr, size));
    EXPECT_EQ(counters.getPinned(), 0);
    EXPECT_EQ(counters.getPinnedDiff(), -size);
    EXPECT_EQ(allocator.getMemoryType(), MemoryType::kPINNED);
    EXPECT_THROW(allocator.deallocate(ptr, size), std::runtime_error);
}

TEST_F(TllmBuffersTest, HostAllocator)
{
    auto constexpr size = 1024;
    HostAllocator allocator{};
    auto& counters = MemoryCounters::getInstance();
    EXPECT_EQ(counters.getCpu(), 0);
    auto ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(counters.getCpu(), size);
    EXPECT_EQ(counters.getCpuDiff(), size);
    EXPECT_NO_THROW(allocator.deallocate(ptr, size));
    EXPECT_EQ(counters.getCpu(), 0);
    EXPECT_EQ(counters.getCpuDiff(), -size);
    EXPECT_EQ(allocator.getMemoryType(), MemoryType::kCPU);
}

TEST_F(TllmBuffersTest, CudaAllocatorAsync)
{
    if (mDeviceCount == 0)
        GTEST_SKIP();

    auto streamPtr = std::make_shared<CudaStream>();
    auto constexpr size = 1024;
    CudaAllocatorAsync allocator{streamPtr};
    auto& counters = MemoryCounters::getInstance();
    EXPECT_EQ(counters.getGpu(), 0);
    auto ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(counters.getGpu(), size);
    EXPECT_EQ(counters.getGpuDiff(), size);
    EXPECT_NO_THROW(allocator.deallocate(ptr, size));
    EXPECT_EQ(counters.getGpu(), 0);
    EXPECT_EQ(counters.getGpuDiff(), -size);
    EXPECT_EQ(allocator.getMemoryType(), MemoryType::kGPU);
    streamPtr->synchronize();
    CudaAllocatorAsync allocatorCopy = allocator;
    EXPECT_EQ(allocatorCopy.getCudaStream(), streamPtr);
    CudaAllocatorAsync allocatorMove = std::move(allocatorCopy);
    EXPECT_EQ(allocatorMove.getCudaStream(), streamPtr);
    EXPECT_THROW(allocator.deallocate(ptr, size), std::runtime_error);
}

namespace
{
void testBuffer(IBuffer& buffer, std::int32_t typeSize)
{
    auto const size = buffer.getSize();
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.getSizeInBytes(), size * typeSize);
    EXPECT_EQ(buffer.getCapacity(), size);
    buffer.resize(size / 2);
    EXPECT_EQ(buffer.getSize(), size / 2);
    EXPECT_EQ(buffer.getCapacity(), size);
    buffer.resize(size * 2);
    EXPECT_EQ(buffer.getSize(), size * 2);
    EXPECT_EQ(buffer.getCapacity(), size * 2);
    buffer.release();
    EXPECT_EQ(buffer.getSize(), 0);
    EXPECT_EQ(buffer.data(), nullptr);
    buffer.resize(size / 2);
    EXPECT_EQ(buffer.getCapacity(), size / 2);
    auto bufferWrapped = IBuffer::wrap(buffer.data(), buffer.getDataType(), buffer.getSize(), buffer.getCapacity());
    EXPECT_EQ(bufferWrapped->data(), buffer.data());
    EXPECT_EQ(bufferWrapped->getSize(), buffer.getSize());
    EXPECT_EQ(bufferWrapped->getCapacity(), buffer.getCapacity());
    EXPECT_EQ(bufferWrapped->getDataType(), buffer.getDataType());
    EXPECT_EQ(bufferWrapped->getMemoryType(), buffer.getMemoryType());
    EXPECT_NO_THROW(bufferWrapped->resize(buffer.getCapacity() / 2));
    EXPECT_THROW(bufferWrapped->resize(buffer.getCapacity() * 2), std::bad_alloc);
    auto byteBuffer = IBuffer::wrap(static_cast<std::uint8_t*>(buffer.data()), buffer.getSizeInBytes());
    EXPECT_EQ(byteBuffer->getSizeInBytes(), buffer.getSizeInBytes());
    EXPECT_EQ(byteBuffer->getCapacity(), buffer.getSizeInBytes());
    auto tensorWrapped = ITensor::wrap(buffer.data(), buffer.getDataType(),
        ITensor::makeShape({static_cast<SizeType>(buffer.getSize())}), buffer.getCapacity());
    EXPECT_EQ(tensorWrapped->getSize(), buffer.getSize());
    EXPECT_EQ(tensorWrapped->getCapacity(), buffer.getCapacity());
    EXPECT_EQ(tensorWrapped->getDataType(), buffer.getDataType());
    EXPECT_EQ(tensorWrapped->getMemoryType(), buffer.getMemoryType());
    EXPECT_NO_THROW(tensorWrapped->reshape(ITensor::makeShape({static_cast<SizeType>(buffer.getCapacity()) / 2})));
    EXPECT_THROW(
        tensorWrapped->reshape(ITensor::makeShape({static_cast<SizeType>(buffer.getCapacity()) * 2})), std::bad_alloc);
}
} // namespace

TEST_F(TllmBuffersTest, DeviceBuffer)
{
    if (mDeviceCount == 0)
        GTEST_SKIP();

    auto streamPtr = std::make_shared<CudaStream>();
    auto constexpr size = 1024;
    CudaAllocatorAsync allocator{streamPtr};
    {
        DeviceBuffer buffer{size, nvinfer1::DataType::kFLOAT, allocator};
        testBuffer(buffer, sizeof(float));
    }
    streamPtr->synchronize();

    static_assert(!std::is_copy_constructible<DeviceBuffer>::value);
    static_assert(!std::is_copy_assignable<DeviceBuffer>::value);
}

TEST_F(TllmBuffersTest, DeviceTensor)
{
    if (mDeviceCount == 0)
        GTEST_SKIP();

    auto streamPtr = std::make_shared<CudaStream>();
    nvinfer1::Dims constexpr dims{3, 16, 8, 4};
    CudaAllocatorAsync allocator{streamPtr};
    {
        DeviceTensor tensor{dims, nvinfer1::DataType::kFLOAT, allocator};
        EXPECT_EQ(tensor.getSize(), ITensor::volume(dims));
        testBuffer(tensor, sizeof(float));
        EXPECT_EQ(tensor.getSize(), ITensor::volume(tensor.getShape()));
    }
    streamPtr->synchronize();

    static_assert(!std::is_copy_constructible<DeviceBuffer>::value);
    static_assert(!std::is_copy_assignable<DeviceBuffer>::value);
}

TEST_F(TllmBuffersTest, BufferSlice)
{
    auto constexpr size = 1024;
    HostAllocator allocator{};
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    auto buffer = std::make_shared<HostBuffer>(size, dataType, allocator);
    auto offset = size / 8;
    auto slice = IBuffer::slice(buffer, offset);
    auto const sizeSlice = size - offset;
    EXPECT_EQ(slice->getSize(), sizeSlice);
    EXPECT_EQ(slice->getCapacity(), sizeSlice);
    EXPECT_EQ(static_cast<std::uint8_t*>(slice->data()) - static_cast<std::uint8_t*>(buffer->data()),
        offset * BufferDataType(dataType).getSize());

    EXPECT_NO_THROW(slice->resize(sizeSlice));
    EXPECT_NO_THROW(slice->resize(sizeSlice / 2));
    EXPECT_THROW(slice->resize(sizeSlice * 2), std::runtime_error);
    EXPECT_NO_THROW(slice->release());
    EXPECT_EQ(slice->data(), nullptr);

    std::shared_ptr<HostBuffer const> constBuffer{buffer};
    auto constSlice = IBuffer::slice(constBuffer, offset);
    EXPECT_EQ(constSlice->getSize(), sizeSlice);
    auto uniqueSlice = IBuffer::slice(std::move(constSlice), 1);
    EXPECT_EQ(uniqueSlice->getSize(), sizeSlice - 1);
}

TEST_F(TllmBuffersTest, BufferOutput)
{
    if (mDeviceCount == 0)
        GTEST_SKIP();

    auto streamPtr = std::make_shared<CudaStream>();
    CudaAllocatorAsync allocator{streamPtr};
    for (std::size_t size : {0, 16})
    {
        DeviceBuffer buffer{size, nvinfer1::DataType::kFLOAT, allocator};
        TLLM_CUDA_CHECK(cudaMemsetAsync(buffer.data(), 0, buffer.getSizeInBytes(), streamPtr->get()));
        streamPtr->synchronize();
        std::stringstream ss;
        ss << buffer;
        auto str = ss.str();
        EXPECT_THAT(str, ::testing::HasSubstr(std::string("shape: (") + std::to_string(size) + ")"));
        EXPECT_THAT(str, ::testing::HasSubstr(tc::vec2str(std::vector<int>(size, 0))));
    }
    streamPtr->synchronize();
}

TEST_F(TllmBuffersTest, TensorOutput)
{
    if (mDeviceCount == 0)
        GTEST_SKIP();

    auto streamPtr = std::make_shared<CudaStream>();
    nvinfer1::Dims constexpr dims{3, 16, 8, 4};
    CudaAllocatorAsync allocator{streamPtr};
    for (auto dataType :
        {nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF, nvinfer1::DataType::kBOOL, nvinfer1::DataType::kINT8,
            nvinfer1::DataType::kINT32, nvinfer1::DataType::kINT64, nvinfer1::DataType::kUINT8})
    {
        DeviceTensor tensor{dims, dataType, allocator};
        TLLM_CUDA_CHECK(cudaMemsetAsync(tensor.data(), 0, tensor.getSizeInBytes(), streamPtr->get()));
        streamPtr->synchronize();
        std::stringstream ss;
        ss << tensor;
        auto str = ss.str();
        EXPECT_THAT(str, ::testing::HasSubstr(std::string("shape: ") + ITensor::toString(dims)));
        EXPECT_THAT(str, ::testing::HasSubstr("i=15 j=7: (0, 0, 0, 0)"))
            << "dataType: " << static_cast<std::int32_t>(dataType);
    }
    streamPtr->synchronize();
}

namespace
{
template <typename T>
void testBufferType()
{
    auto constexpr size = 1024;
    HostAllocator allocator{};
    BufferDataType constexpr dataType{TRTDataType<T>::value};
    using limits = std::numeric_limits<T>;
    static_assert(dataType.isPointer() || dataType.isUnsigned() != limits::is_signed);
    static_assert(std::is_same_v<T,
        typename DataTypeTraits<dataType.getDataType(), dataType.isUnsigned(), dataType.isPointer()>::type>);
    IBuffer::SharedPtr buffer{std::make_shared<HostBuffer>(size, dataType, allocator)};
    auto bufferPtr = bufferCast<T>(*buffer);
    auto constexpr max = limits::max();
    bufferPtr[0] = max;
    EXPECT_EQ(bufferPtr[0], max);
    auto constexpr min = limits::min();
    bufferPtr[size - 1] = min;
    EXPECT_EQ(bufferPtr[size - 1], min);
    EXPECT_EQ(buffer->data(size), bufferPtr + size);
}
} // namespace

TEST_F(TllmBuffersTest, ExtendedTypes)
{
    testBufferType<bool>();
    testBufferType<bool*>();
    testBufferType<std::int8_t>();
    testBufferType<std::int8_t*>();
    testBufferType<std::uint8_t>();
    testBufferType<std::uint8_t*>();
    testBufferType<std::int32_t>();
    testBufferType<std::int32_t*>();
    testBufferType<std::uint32_t>();
    testBufferType<std::uint32_t*>();
    testBufferType<std::int64_t>();
    testBufferType<std::int64_t*>();
    testBufferType<std::uint64_t>();
    testBufferType<std::uint64_t*>();
}

TEST_F(TllmBuffersTest, BytesToString)
{
    auto constexpr precision = 2;
    MemoryCounters::SizeType size;
    MemoryCounters::DiffType diff;

    size = (1ul << 10) - 1;
    EXPECT_EQ(MemoryCounters::bytesToString(size, precision), "1023.00 B");
    size = 1ul << 10;
    EXPECT_EQ(MemoryCounters::bytesToString(size, precision), "1.00 KB");
    size = (1ul << 10) + (1ul << 9);
    EXPECT_EQ(MemoryCounters::bytesToString(size, precision), "1.50 KB");
    size = (1ul << 20) - (1ul << 10);
    EXPECT_EQ(MemoryCounters::bytesToString(size, precision), "1023.00 KB");
    size = 1ul << 20;
    EXPECT_EQ(MemoryCounters::bytesToString(size, precision), "1.00 MB");
    size = (1ul << 20) + (1ul << 19);
    EXPECT_EQ(MemoryCounters::bytesToString(size, precision), "1.50 MB");
    size = (1ul << 30) - (1ul << 20);
    EXPECT_EQ(MemoryCounters::bytesToString(size, precision), "1023.00 MB");
    size = 1ul << 30;
    EXPECT_EQ(MemoryCounters::bytesToString(size, precision), "1.00 GB");
    size = (1ul << 30) + (1ul << 29);
    EXPECT_EQ(MemoryCounters::bytesToString(size, precision), "1.50 GB");
    size = (1ull << 40) - (1ull << 30);
    EXPECT_EQ(MemoryCounters::bytesToString(size, precision), "1023.00 GB");
    size = 1ull << 40;
    EXPECT_EQ(MemoryCounters::bytesToString(size, precision), "1.00 TB");
    size = (1ull << 40) + (1ull << 39);
    EXPECT_EQ(MemoryCounters::bytesToString(size, precision), "1.50 TB");

    diff = -(1l << 10) + 1;
    EXPECT_EQ(MemoryCounters::bytesToString(diff, precision), "-1023.00 B");
    diff = -(1l << 10);
    EXPECT_EQ(MemoryCounters::bytesToString(diff, precision), "-1.00 KB");
    diff = -(1l << 10) - (1l << 9);
    EXPECT_EQ(MemoryCounters::bytesToString(diff, precision), "-1.50 KB");
    diff = -(1l << 20) + (1l << 10);
    EXPECT_EQ(MemoryCounters::bytesToString(diff, precision), "-1023.00 KB");
    diff = -(1l << 20);
    EXPECT_EQ(MemoryCounters::bytesToString(diff, precision), "-1.00 MB");
    diff = -(1l << 20) - (1l << 19);
    EXPECT_EQ(MemoryCounters::bytesToString(diff, precision), "-1.50 MB");
    diff = -(1l << 30) + (1l << 20);
    EXPECT_EQ(MemoryCounters::bytesToString(diff, precision), "-1023.00 MB");
    diff = -(1l << 30);
    EXPECT_EQ(MemoryCounters::bytesToString(diff, precision), "-1.00 GB");
    diff = -(1l << 30) - (1l << 29);
    EXPECT_EQ(MemoryCounters::bytesToString(diff, precision), "-1.50 GB");
    diff = -(1ll << 40) + (1ll << 30);
    EXPECT_EQ(MemoryCounters::bytesToString(diff, precision), "-1023.00 GB");
    diff = -(1ll << 40);
    EXPECT_EQ(MemoryCounters::bytesToString(diff, precision), "-1.00 TB");
    diff = -(1ll << 40) - (1ll << 39);
    EXPECT_EQ(MemoryCounters::bytesToString(diff, precision), "-1.50 TB");
}

TEST_F(TllmBuffersTest, PinnedPoolAllocator)
{
    if (mDeviceCount == 0)
        GTEST_SKIP();

    using MemPool = MemoryPool<PinnedAllocator>;
    auto expectedSize = [](const auto& tensor)
    {
        auto s = tensor()->getSizeInBytes();
        constexpr auto alignment = MemPool::kAlignment;
        s = s + alignment - 1 - ((s + alignment - 1) % alignment);
        return s;
    };

    auto& pool = PinnedPoolAllocator::getPool();
    auto& segments = pool.getMemorySegments();
    pool.logSegments();
    EXPECT_EQ(segments.size(), 0);

    {
        auto a = tb::NamedTensor{nvinfer1::DataType::kFLOAT, {512, 4, 4}, "a", nullptr};
        auto b = tb::NamedTensor{nvinfer1::DataType::kHALF, {512, 10}, "b", nullptr};
        pool.logSegments();
        auto it = std::begin(segments);
        EXPECT_NE(it->tag, nullptr);
        EXPECT_EQ(it->size, expectedSize(a));
        it = std::next(it);
        EXPECT_NE(it->tag, nullptr);
        EXPECT_EQ(it->size, expectedSize(b));
        it = std::next(it);
        EXPECT_EQ(it->tag, nullptr);
        it = std::next(it);
        EXPECT_EQ(it, std::end(segments));
    }

    auto const chunkSize = pool.getChunkSize();
    auto constexpr initChunkSize = MemPool::kInitialChunkSize;
    EXPECT_EQ(chunkSize, initChunkSize);

    {
        pool.logSegments();
        auto it = std::begin(segments);
        EXPECT_EQ(it->tag, nullptr);
        EXPECT_EQ(it->size, chunkSize);
    }

    std::size_t secondChunkSize;
    {
        // Test creating a new chunk
        auto c = tb::NamedTensor{nvinfer1::DataType::kUINT8, {initChunkSize + 1}, "c", nullptr};
        pool.logSegments();
        auto it = std::begin(segments);
        EXPECT_EQ(it->tag, nullptr);
        EXPECT_EQ(it->size, chunkSize);
        it = std::next(it);
        EXPECT_NE(it->tag, nullptr);
        EXPECT_EQ(it->size, expectedSize(c));
        it = std::next(it);
        EXPECT_EQ(it->tag, nullptr);
        secondChunkSize = expectedSize(c) + it->size;
        EXPECT_EQ(secondChunkSize, pool.getChunkSize());
        it = std::next(it);
        EXPECT_EQ(it, std::end(segments));
    }

    {
        pool.logSegments();
        auto it = std::begin(segments);
        EXPECT_EQ(it->tag, nullptr);
        EXPECT_EQ(it->size, chunkSize);
        it = std::next(it);
        EXPECT_EQ(it->tag, nullptr);
        EXPECT_EQ(it->size, secondChunkSize);
        it = std::next(it);
        EXPECT_EQ(it, std::end(segments));
    }
}

TEST_F(TllmBuffersTest, MemoryPool)
{
    using MemPool = MemoryPool<HostAllocator>;
    auto constexpr alignment = MemPool::kAlignment;
    auto constexpr chunkSize = alignment * 4;
    auto& memCounters = MemoryCounters::getInstance();
    auto const initMemory = memCounters.getCpu();
    {
        MemPool pool{chunkSize};
        EXPECT_EQ(pool.getChunkSize(), chunkSize);
        EXPECT_EQ(memCounters.getCpu(), initMemory);
        auto constexpr sizeBytes = alignment / 4;
        auto ptr_0 = pool.allocate(sizeBytes);
        auto const oneChunk = initMemory + chunkSize;
        EXPECT_EQ(memCounters.getCpu(), oneChunk);
        auto ptr_1 = pool.allocate(0);
        EXPECT_EQ(memCounters.getCpu(), oneChunk);
        auto ptr_2 = pool.allocate(sizeBytes);
        EXPECT_EQ(memCounters.getCpu(), oneChunk);
        pool.deallocate(ptr_0, sizeBytes);
        EXPECT_EQ(memCounters.getCpu(), oneChunk);
        pool.deallocate(ptr_1, 0);
        EXPECT_EQ(memCounters.getCpu(), oneChunk);
        pool.deallocate(ptr_2, sizeBytes);
        EXPECT_EQ(memCounters.getCpu(), oneChunk);
        EXPECT_EQ(static_cast<std::uint8_t*>(ptr_1) - static_cast<std::uint8_t*>(ptr_0), alignment);
        EXPECT_EQ(static_cast<std::uint8_t*>(ptr_2) - static_cast<std::uint8_t*>(ptr_1), alignment);
    }
    EXPECT_EQ(memCounters.getCpu(), initMemory);
}

TEST_F(TllmBuffersTest, PinnedPoolStressTest)
{
    if (mDeviceCount == 0)
        GTEST_SKIP();

    using Allocator = PinnedPoolAllocator;
    using MemPool = Allocator::PoolType;
    auto& memCounters = MemoryCounters::getInstance();
    auto const initMemory = memCounters.getPinned();
    auto constexpr chunkSize = MemPool::kInitialChunkSize / 4;
    std::mt19937 rnd{42};                               // mersenne_twister_engine seeded with 42 NOLINT(*-msc51-cpp)
    auto constexpr expectedSize = std::size_t{1} << 20; // 1 MiB
    std::poisson_distribution distribution{expectedSize};
    auto constexpr numberOfAllocations = chunkSize * 2 / expectedSize;
    std::vector<std::tuple<void*, std::size_t>> allocations;
    allocations.reserve(numberOfAllocations);

    Allocator allocator{};
    auto& pool = Allocator::getPool();
    pool.setChunkSize(chunkSize);
    EXPECT_EQ(pool.getChunkSize(), chunkSize);
    EXPECT_EQ(memCounters.getPinned(), initMemory);
    auto const poolReservedSize = pool.getReservedSize();
    auto const poolUsedSize = pool.getUsedSize();
    std::size_t totalUsedSize{0};
    for (std::size_t i = 0; i < numberOfAllocations; ++i)
    {
        auto const size = distribution(rnd);
        auto const ptr = allocator.allocate(size);
        allocations.emplace_back(ptr, size);
        totalUsedSize += size;
    }
    EXPECT_GE(pool.getUsedSize(), poolUsedSize + totalUsedSize);

    std::shuffle(allocations.begin(), allocations.end(), rnd);
    auto const deallocIdx = allocations.size() / 2;
    auto const deallocSize = allocations.size() - deallocIdx;
    for (auto const& [ptr, size] : BufferRange{allocations.data() + deallocIdx, deallocSize})
    {
        allocator.deallocate(ptr, size);
        totalUsedSize -= size;
    }
    allocations.resize(deallocIdx);
    EXPECT_GE(pool.getUsedSize(), poolUsedSize + totalUsedSize);
    EXPECT_EQ(memCounters.getPinned() - initMemory, pool.getReservedSize() - poolReservedSize);

    std::thread thread(
        [&]()
        {
            for (std::size_t i = 0; i < deallocSize; ++i)
            {
                auto const size = distribution(rnd);
                auto const ptr = allocator.allocate(size);
                allocations.emplace_back(ptr, size);
                totalUsedSize += size;
            }
            EXPECT_GE(pool.getUsedSize(), poolUsedSize + totalUsedSize);

            std::shuffle(allocations.begin() + static_cast<std::ptrdiff_t>(deallocIdx), allocations.end(), rnd);
            for (auto const& [ptr, size] : allocations)
            {
                allocator.deallocate(ptr, size);
                totalUsedSize -= size;
            }
            EXPECT_EQ(totalUsedSize, 0u);
            EXPECT_EQ(pool.getUsedSize(), poolUsedSize);
        });
    thread.join();
}
