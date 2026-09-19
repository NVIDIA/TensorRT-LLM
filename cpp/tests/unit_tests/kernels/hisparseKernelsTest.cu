/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/kernels/hisparseKernels.cuh"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{

namespace hk = tensorrt_llm::kernels::hisparse;

void checkCuda(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename T, bool Pinned>
struct CudaDeleter
{
    void operator()(T* pointer) const
    {
        EXPECT_EQ(Pinned ? cudaFreeHost(pointer) : cudaFree(pointer), cudaSuccess);
    }
};

template <typename T, bool Pinned = false>
class CudaBuffer
{
public:
    explicit CudaBuffer(std::vector<T> const& values)
        : mSize(values.size())
    {
        T* pointer = nullptr;
        if constexpr (Pinned)
        {
            checkCuda(cudaHostAlloc(&pointer, mSize * sizeof(T), cudaHostAllocMapped));
        }
        else
        {
            checkCuda(cudaMalloc(&pointer, mSize * sizeof(T)));
        }
        mPointer.reset(pointer);
        assign(values);
    }

    void assign(std::vector<T> const& values)
    {
        ASSERT_EQ(values.size(), mSize);
        if constexpr (Pinned)
        {
            std::copy(values.begin(), values.end(), mPointer.get());
        }
        else
        {
            checkCuda(cudaMemcpy(mPointer.get(), values.data(), mSize * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    T* deviceData() const
    {
        if constexpr (Pinned)
        {
            T* alias = nullptr;
            checkCuda(cudaHostGetDevicePointer(&alias, mPointer.get(), 0));
            return alias;
        }
        return mPointer.get();
    }

    std::vector<T> read() const
    {
        std::vector<T> result(mSize);
        if constexpr (Pinned)
        {
            std::copy_n(mPointer.get(), mSize, result.begin());
        }
        else
        {
            checkCuda(cudaMemcpy(result.data(), mPointer.get(), mSize * sizeof(T), cudaMemcpyDeviceToHost));
        }
        return result;
    }

private:
    size_t mSize;
    std::unique_ptr<T, CudaDeleter<T, Pinned>> mPointer;
};

class CudaLaunch
{
public:
    CudaLaunch()
    {
        checkCuda(cudaStreamCreate(&mStream));
    }

    ~CudaLaunch()
    {
        if (mExecutable != nullptr)
        {
            EXPECT_EQ(cudaGraphExecDestroy(mExecutable), cudaSuccess);
        }
        if (mGraph != nullptr)
        {
            EXPECT_EQ(cudaGraphDestroy(mGraph), cudaSuccess);
        }
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    CudaLaunch(CudaLaunch const&) = delete;
    CudaLaunch& operator=(CudaLaunch const&) = delete;

    template <typename Launcher>
    void run(Launcher const& launch, bool graph)
    {
        if (graph)
        {
            if (mExecutable == nullptr)
            {
                checkCuda(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal));
                launch(mStream);
                checkCuda(cudaStreamEndCapture(mStream, &mGraph));
                checkCuda(cudaGraphInstantiate(&mExecutable, mGraph, nullptr, nullptr, 0));
            }
            checkCuda(cudaGraphLaunch(mExecutable, mStream));
        }
        else
        {
            launch(mStream);
        }
        checkCuda(cudaGetLastError());
        checkCuda(cudaStreamSynchronize(mStream));
    }

private:
    cudaStream_t mStream = nullptr;
    cudaGraph_t mGraph = nullptr;
    cudaGraphExec_t mExecutable = nullptr;
};

struct CacheState
{
    int batchSize = 3;
    int realRequests = 2;
    int topK;
    int hotSize;
    int itemBytes;
    int bufferStride;
    int hostStride;
    int lruStride;
    int inputStride;
    int outputStride;
    std::vector<int32_t> selections;
    std::vector<int32_t> tokens;
    std::vector<int32_t> deviceLocations;
    std::vector<int64_t> hostLocations;
    std::vector<int16_t> lru;
    std::vector<int64_t> requestIds;
    std::vector<int64_t> seqLens;
    std::vector<int32_t> attentionIndices;
    std::vector<uint8_t> hostK;
    std::vector<uint8_t> hostV;
    std::vector<uint8_t> deviceK;
    std::vector<uint8_t> deviceV;
};

void copyItem(CacheState& state, int request, int token, int slot)
{
    int64_t const src = state.hostLocations[request * state.hostStride + token] * state.itemBytes;
    int64_t const dst = state.deviceLocations[request * state.bufferStride + slot] * state.itemBytes;
    std::copy_n(state.hostK.begin() + src, state.itemBytes, state.deviceK.begin() + dst);
    std::copy_n(state.hostV.begin() + src, state.itemBytes, state.deviceV.begin() + dst);
}

CacheState makeState(int topK, int hotSize, int itemBytes = 32)
{
    constexpr int kPoolSize = 3;
    CacheState state;
    state.topK = topK;
    state.hotSize = hotSize;
    state.itemBytes = itemBytes;
    state.bufferStride = hotSize + 4;
    state.hostStride = 3 * hotSize + topK + 5;
    state.lruStride = hotSize + 3;
    state.inputStride = topK + 2;
    state.outputStride = topK + 7;
    state.requestIds = {2, 0, -12345}; // Padding must not dereference request IDs.
    state.seqLens = {state.hostStride - 1, state.hostStride - 3, -12345};
    state.selections.resize(state.batchSize * state.inputStride, -77);
    state.attentionIndices.resize(state.batchSize * state.outputStride, -99);
    state.tokens.resize(kPoolSize * state.bufferStride, -1);
    state.deviceLocations.resize(kPoolSize * state.bufferStride, -1);
    state.hostLocations.resize(kPoolSize * state.hostStride);
    state.lru.resize(kPoolSize * state.lruStride, -1);
    state.hostK.resize(kPoolSize * state.hostStride * itemBytes);
    state.hostV.resize(state.hostK.size());
    // Reverse physical locations with guard items between slots and after the pool.
    int const deviceItems = 2 * kPoolSize * (hotSize + 1) + 3;
    state.deviceK.resize(deviceItems * itemBytes, 0xA5);
    state.deviceV.resize(deviceItems * itemBytes, 0x5A);
    std::mt19937 random(42);
    for (size_t i = 0; i < state.hostK.size(); ++i)
    {
        state.hostK[i] = static_cast<uint8_t>(random());
        state.hostV[i] = static_cast<uint8_t>(random());
    }
    for (int request = 0; request < kPoolSize; ++request)
    {
        for (int token = 0; token < state.hostStride; ++token)
        {
            state.hostLocations[request * state.hostStride + token]
                = kPoolSize * state.hostStride - 1 - (request * state.hostStride + token);
        }
        for (int slot = 0; slot <= hotSize; ++slot)
        {
            state.deviceLocations[request * state.bufferStride + slot]
                = 2 * (kPoolSize * (hotSize + 1) - 1 - (request * (hotSize + 1) + slot)) + 1;
            if (slot < hotSize)
            {
                int const token = (slot + 1) % hotSize;
                state.tokens[request * state.bufferStride + slot] = token;
                state.lru[request * state.lruStride + slot] = static_cast<int16_t>(slot);
                copyItem(state, request, token, slot);
            }
        }
        auto const first = state.lru.begin() + request * state.lruStride;
        std::shuffle(first, first + hotSize, random);
    }
    for (int batch = 0; batch < state.realRequests; ++batch)
    {
        int const request = state.requestIds[batch];
        copyItem(state, request, state.seqLens[batch] - 1, hotSize);
        for (int i = 0; i < topK; ++i)
        {
            state.selections[batch * state.inputStride + i] = i % 2 == 0 ? i : hotSize + i;
        }
        state.selections[batch * state.inputStride + topK - 1] = state.seqLens[batch] - 1;
    }
    return state;
}

// Deliberately scalar reference: search the selected list, partition the old LRU,
// replace misses, then concatenate untouched slots, replacements, and hits.
void referenceStep(CacheState& state, bool isMla)
{
    auto const originalV = state.deviceV;
    for (int batch = 0; batch < state.batchSize; ++batch)
    {
        auto output = state.attentionIndices.begin() + batch * state.outputStride;
        std::fill_n(output, state.topK, -1);
        if (batch >= state.realRequests)
        {
            continue;
        }
        int const request = state.requestIds[batch];
        auto const locations = state.deviceLocations.begin() + request * state.bufferStride;
        auto tokens = state.tokens.begin() + request * state.bufferStride;
        auto lru = state.lru.begin() + request * state.lruStride;
        auto const selected = state.selections.begin() + batch * state.inputStride;
        if (state.seqLens[batch] <= state.hotSize)
        {
            for (int i = 0; i < std::min<int64_t>(state.seqLens[batch], state.topK); ++i)
            {
                if (selected[i] >= 0 && selected[i] < state.seqLens[batch])
                {
                    output[i] = locations[selected[i]];
                }
            }
            continue;
        }
        std::vector<int16_t> hits;
        std::vector<int16_t> evictable;
        for (int i = 0; i < state.hotSize; ++i)
        {
            int16_t const slot = lru[i];
            auto const found = std::find(selected, selected + state.topK, tokens[slot]);
            if (tokens[slot] >= 0 && found != selected + state.topK)
            {
                output[found - selected] = locations[slot];
                hits.push_back(slot);
            }
            else
            {
                evictable.push_back(slot);
            }
        }
        size_t misses = 0;
        for (int i = 0; i < state.topK; ++i)
        {
            if (selected[i] == state.seqLens[batch] - 1)
            {
                output[i] = locations[state.hotSize];
            }
            else if (output[i] < 0)
            {
                int16_t const slot = evictable.at(misses++);
                tokens[slot] = selected[i];
                output[i] = locations[slot];
                copyItem(state, request, selected[i], slot);
            }
        }
        auto next = std::copy(evictable.begin() + misses, evictable.end(), lru);
        next = std::copy(evictable.begin(), evictable.begin() + misses, next);
        std::copy(hits.begin(), hits.end(), next);
    }
    if (isMla)
    {
        state.deviceV = originalV;
    }
}

template <int BlockSize, int TopK, int HotSize, bool IsMla = true, typename SeqT = int32_t, typename RequestT = int64_t>
void runAndCompare(CacheState state, int steps = 1, bool graph = false)
{
    ASSERT_EQ(state.topK, TopK);
    ASSERT_EQ(state.hotSize, HotSize);
    CudaBuffer<int32_t> selections(state.selections), tokens(state.tokens), locations(state.deviceLocations);
    CudaBuffer<int64_t> hostLocations(state.hostLocations);
    CudaBuffer<int16_t> lru(state.lru);
    CudaBuffer<RequestT> requests(std::vector<RequestT>(state.requestIds.begin(), state.requestIds.end()));
    CudaBuffer<SeqT> lengths(std::vector<SeqT>(state.seqLens.begin(), state.seqLens.end()));
    CudaBuffer<int32_t> realRequests({state.realRequests}), output(state.attentionIndices);
    CudaBuffer<uint8_t, true> hostK(state.hostK), hostV(state.hostV);
    CudaBuffer<uint8_t> deviceK(state.deviceK), deviceV(state.deviceV);
    auto const* hostKAlias = hostK.deviceData();
    auto const* hostVAlias = IsMla ? nullptr : hostV.deviceData();
    auto* deviceVAlias = IsMla ? nullptr : deviceV.deviceData();
    constexpr size_t kSmemBytes = hk::SmemLayout<TopK, HotSize>::kBytes;
    if constexpr (kSmemBytes > 48 * 1024)
    {
        checkCuda(
            cudaFuncSetAttribute(hk::loadCacheToDeviceBufferKernel<BlockSize, TopK, HotSize, IsMla, SeqT, RequestT>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    }
    CudaLaunch launcher;
    auto const launch = [&](cudaStream_t stream)
    {
        hk::loadCacheToDeviceBufferKernel<BlockSize, TopK, HotSize, IsMla, SeqT, RequestT>
            <<<state.batchSize, BlockSize, kSmemBytes, stream>>>(selections.deviceData(), tokens.deviceData(),
                hostLocations.deviceData(), locations.deviceData(), hostKAlias, hostVAlias, deviceK.deviceData(),
                deviceVAlias, output.deviceData(), requests.deviceData(), lengths.deviceData(), lru.deviceData(),
                realRequests.deviceData(), state.bufferStride, state.hostStride, state.lruStride, state.inputStride,
                state.outputStride, state.itemBytes);
    };
    std::mt19937 random(123);
    for (int step = 0; step < steps; ++step)
    {
        SCOPED_TRACE(step);
        if (step > 0)
        {
            for (int batch = 0; batch < state.realRequests; ++batch)
            {
                std::vector<int32_t> candidates(state.seqLens[batch]);
                std::iota(candidates.begin(), candidates.end(), 0);
                std::shuffle(candidates.begin(), candidates.end(), random);
                std::copy_n(candidates.begin(), TopK, state.selections.begin() + batch * state.inputStride);
            }
            selections.assign(state.selections);
        }
        launcher.run(launch, graph);
        referenceStep(state, IsMla);
        EXPECT_EQ(output.read(), state.attentionIndices);
        EXPECT_EQ(tokens.read(), state.tokens);
        EXPECT_EQ(lru.read(), state.lru);
        EXPECT_EQ(deviceK.read(), state.deviceK);
        EXPECT_EQ(deviceV.read(), state.deviceV);
    }
    EXPECT_EQ(hostK.read(), state.hostK);
    EXPECT_EQ(hostV.read(), state.hostV);
}

TEST(HiSparseKernelsTest, HitsMissesNewestAndRequestMapping)
{
    auto const state = makeState(3, 4);
    runAndCompare<256, 3, 4, true, int32_t, int32_t>(state);
    runAndCompare<256, 3, 4, true, int32_t, int64_t>(state);
    runAndCompare<256, 3, 4, true, int64_t, int32_t>(state);
    runAndCompare<256, 3, 4, true, int64_t, int64_t>(state);
}

TEST(HiSparseKernelsTest, AllHitsAndAllMisses)
{
    auto state = makeState(4, 4);
    for (int batch = 0; batch < state.realRequests; ++batch)
    {
        std::iota(state.selections.begin() + batch * state.inputStride,
            state.selections.begin() + batch * state.inputStride + state.topK, 0);
    }
    runAndCompare<256, 4, 4>(state);
    for (int batch = 0; batch < state.realRequests; ++batch)
    {
        std::iota(state.selections.begin() + batch * state.inputStride,
            state.selections.begin() + batch * state.inputStride + state.topK, state.hotSize);
    }
    runAndCompare<256, 4, 4, false>(state);
    std::fill(state.tokens.begin(), state.tokens.end(), -1);
    runAndCompare<256, 4, 4>(state);
}

TEST(HiSparseKernelsTest, ShortSequenceBoundaryAndPadding)
{
    auto state = makeState(4, 4);
    state.seqLens = {4, 2, -12345};
    std::copy_n(std::vector<int32_t>{3, 0, 2, 1}.begin(), 4, state.selections.begin());
    std::copy_n(std::vector<int32_t>{1, -1, 0, 0}.begin(), 4, state.selections.begin() + state.inputStride);
    runAndCompare<256, 4, 4>(state);
    state.seqLens[0] = 0;
    runAndCompare<256, 4, 4, true, int64_t, int32_t>(state);
    state.realRequests = 0;
    runAndCompare<256, 4, 4>(state);
}

TEST(HiSparseKernelsTest, NewestOnly)
{
    runAndCompare<32, 1, 1>(makeState(1, 1));
}

TEST(HiSparseKernelsTest, RefetchIsByteExact)
{
    for (int bytes : {1, 7, 8, 15, 16, 20, 24, 32, 584, 1024, 1028, 1152})
    {
        SCOPED_TRACE(bytes);
        runAndCompare<128, 3, 4, false>(makeState(3, 4, bytes));
    }
}

TEST(HiSparseKernelsTest, RepeatedReplacementAndGraphReplay)
{
    runAndCompare<256, 33, 65>(makeState(33, 65), 10, true);
}

TEST(HiSparseKernelsTest, MultipleScanIterationsAndPartialWarps)
{
    runAndCompare<32, 67, 131, false>(makeState(67, 131), 3);
    runAndCompare<256, 513, 1031>(makeState(513, 1031), 3);
    runAndCompare<1024, 513, 1031>(makeState(513, 1031), 3);
}

TEST(HiSparseKernelsTest, HashCollisions)
{
    auto state = makeState(4, 4);
    state.seqLens = {19, 19, -12345};
    // Keys 0, 8, and 16 collide in the eight-entry table; 18 uses the newest slot.
    for (int batch = 0; batch < state.realRequests; ++batch)
    {
        std::copy_n(
            std::vector<int32_t>{0, 8, 16, 18}.begin(), 4, state.selections.begin() + batch * state.inputStride);
    }
    runAndCompare<64, 4, 4>(state);
}

TEST(HiSparseKernelsTest, DynamicSharedMemoryOptIn)
{
    static_assert(hk::SmemLayout<2048, 8192>::kBytes > 48 * 1024);
    runAndCompare<256, 2048, 8192>(makeState(2048, 8192, 8));
}

} // namespace
