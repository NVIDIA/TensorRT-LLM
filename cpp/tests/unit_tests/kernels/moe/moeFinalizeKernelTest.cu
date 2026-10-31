/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

// finalizeMoeRoutingKernelLauncher runs finalizeMoeRoutingWarpKernel when the rows are whole 128-bit vectors, the
// pointers are vector aligned and top_k <= 32, and the block-per-row kernels otherwise. Both must produce bitwise
// identical outputs. The tests run the launcher several times on the same data, with the buffers it reads or writes
// either 16-byte aligned (warp kernel) or shifted by 4, 8 or 12 bytes (block-per-row kernels, which access memory in
// 32-bit words), compare the outputs bit for bit and check, with a stream capture, which kernel each launch runs.

#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moe/cutlass/include/moe_kernels.h"
#include "tensorrt_llm/kernels/moe/cutlass/include/moe_util_kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorrt_llm::tests::kernels::moe_finalize
{

namespace
{

namespace ck = tensorrt_llm::kernels::cutlass_kernels;

struct CudaFree
{
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

using DeviceMemory = std::unique_ptr<void, CudaFree>;

DeviceMemory allocateDevice(size_t bytes)
{
    void* ptr = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(bytes, 1)));
    return DeviceMemory(ptr);
}

//! Device copy of a host vector, starting `offset` elements into the allocation.
template <class T>
struct DeviceArray
{
    DeviceMemory memory;
    T* data = nullptr;
};

template <class T>
DeviceArray<T> toDevice(std::vector<T> const& host, size_t offset = 0)
{
    DeviceArray<T> array{allocateDevice((host.size() + offset) * sizeof(T))};
    array.data = static_cast<T*>(array.memory.get()) + offset;
    TLLM_CUDA_CHECK(cudaMemcpy(array.data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return array;
}

template <class T>
std::vector<T> toHost(T const* device, size_t count)
{
    std::vector<T> host(count);
    TLLM_CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

template <class T>
using Bits = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;

//! Raw bit patterns of NaNs (several payloads, both signs), infinities, signed zeros, subnormals and extreme
//! finite values.
template <class T>
std::vector<uint32_t> specialBits()
{
    if constexpr (std::is_same_v<T, float>)
    {
        return {0x7FC00000, 0x7F800001, 0xFFC00001, 0x7F800000, 0xFF800000, 0x00000000, 0x80000000, 0x00000001,
            0x807FFFFF, 0x7F7FFFFF, 0xFF7FFFFF};
    }
    else if constexpr (std::is_same_v<T, half>)
    {
        return {0x7E00, 0x7C01, 0xFE01, 0x7C00, 0xFC00, 0x0000, 0x8000, 0x0001, 0x83FF, 0x7BFF, 0xFBFF};
    }
    else
    {
        return {0x7FC0, 0x7F81, 0xFFC1, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x0001, 0x807F, 0x7F7F, 0xFF7F};
    }
}

template <class T>
T fromBits(uint32_t bits)
{
    Bits<T> const raw = static_cast<Bits<T>>(bits);
    T value;
    std::memcpy(&value, &raw, sizeof(T));
    return value;
}

template <class T>
T fromFloat(float value)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return value;
    }
    else if constexpr (std::is_same_v<T, half>)
    {
        return __float2half_rn(value);
    }
    else
    {
        return __float2bfloat16_rn(value);
    }
}

//! Byte offsets from a 16-byte boundary of the buffers the launcher reads or writes.
struct BufferOffsets
{
    size_t gemmOutput = 0;
    size_t output = 0;
    size_t bias = 0;
};

//! Every buffer aligned (warp kernel when the shape allows it), then every buffer shifted by 4 bytes.
std::vector<BufferOffsets> const kAlignedAndShifted{{}, {4, 4, 4}};

struct FinalizeCase
{
    int64_t numRows = 0;
    int64_t hidden = 0;
    int64_t paddedHidden = 0;
    int64_t topK = 0;
    int numExperts = 0;
    int epSize = 1;
    int epRank = 0;
    int tpRank = 0;
    bool enableAlltoall = false;
    bool withScales = true;
    bool withBias = false;
    bool specialValues = false;
    uint32_t seed = 0;
};

template <class T>
class MoeFinalizeKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        TLLM_CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    void TearDown() override
    {
        TLLM_CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    template <class V>
    std::vector<V> randomValues(size_t count, bool special, std::mt19937& rng)
    {
        std::normal_distribution<float> normal(0.F, 1.F);
        std::uniform_real_distribution<float> uniform(0.F, 1.F);
        auto const specials = specialBits<V>();
        std::uniform_int_distribution<size_t> pick(0, specials.size() - 1);
        std::vector<V> values(count);
        for (auto& value : values)
        {
            value = (special && uniform(rng) < 0.03F) ? fromBits<V>(specials[pick(rng)]) : fromFloat<V>(normal(rng));
        }
        return values;
    }

    //! Names of the kernels that `launch` enqueues on mStream, recorded with a stream capture (nothing runs).
    template <class Launch>
    std::vector<std::string> capturedKernelNames(Launch const& launch)
    {
        cudaGraph_t graph{};
        TLLM_CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal));
        launch();
        TLLM_CUDA_CHECK(cudaStreamEndCapture(mStream, &graph));
        size_t numNodes = 0;
        TLLM_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &numNodes));
        std::vector<cudaGraphNode_t> nodes(numNodes);
        TLLM_CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &numNodes));
        std::vector<std::string> names;
        for (auto const node : nodes)
        {
            cudaGraphNodeType type{};
            TLLM_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
            if (type == cudaGraphNodeTypeKernel)
            {
                cudaKernelNodeParams params{};
                TLLM_CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &params));
                char const* name = nullptr;
                TLLM_CUDA_CHECK(cudaFuncGetName(&name, params.func));
                names.emplace_back(name);
            }
        }
        TLLM_CUDA_CHECK(cudaGraphDestroy(graph));
        return names;
    }

    //! Runs the launcher on the same data with each set of buffer offsets, checks which finalize kernel it launches
    //! and compares the outputs bit for bit with those of the first run.
    void runCase(FinalizeCase const& c, std::vector<BufferOffsets> const& offsetsList = kAlignedAndShifted)
    {
        std::mt19937 rng(c.seed);
        int64_t const paddedHidden = c.paddedHidden > 0 ? c.paddedHidden : c.hidden;
        int64_t const numExpertsPerNode = c.numExperts / c.epSize;
        int64_t const startExpert = numExpertsPerNode * c.epRank;
        int64_t const expandedRows = c.numRows * c.topK;

        // Distinct experts per token; about one slot in ten has an id that belongs to no rank.
        std::vector<int> tokenSelectedExperts(expandedRows);
        std::vector<int> experts(c.numExperts);
        std::iota(experts.begin(), experts.end(), 0);
        std::uniform_real_distribution<float> uniform(0.F, 1.F);
        for (int64_t row = 0; row < c.numRows; ++row)
        {
            std::shuffle(experts.begin(), experts.end(), rng);
            for (int64_t k = 0; k < c.topK; ++k)
            {
                tokenSelectedExperts[row * c.topK + k] = uniform(rng) < 0.1F ? -1 : experts[k];
            }
        }

        // Expert maps as the MoE sort kernels build them: the slots with an expert on this rank are sorted by expert
        // into permuted rows [0, numValid); unpermuted row = k * numRows + token. The entries of the other slots are
        // never read; a huge value makes sure a read would fault.
        constexpr int kNeverRead = 0x3FFFFFFF;
        std::vector<int> unpermutedToPermuted(expandedRows, kNeverRead);
        std::vector<int> permutedToUnpermuted(expandedRows, kNeverRead);
        std::vector<int64_t> expertFirstTokenOffset(numExpertsPerNode + 1, 0);
        std::vector<std::pair<int64_t, int64_t>> localSlots; // (local expert, unpermuted row)
        for (int64_t row = 0; row < c.numRows; ++row)
        {
            for (int64_t k = 0; k < c.topK; ++k)
            {
                int64_t const expert = tokenSelectedExperts[row * c.topK + k] - startExpert;
                if (expert >= 0 && expert < numExpertsPerNode)
                {
                    localSlots.emplace_back(expert, k * c.numRows + row);
                }
            }
        }
        std::sort(localSlots.begin(), localSlots.end());
        for (size_t permuted = 0; permuted < localSlots.size(); ++permuted)
        {
            unpermutedToPermuted[localSlots[permuted].second] = static_cast<int>(permuted);
            permutedToUnpermuted[permuted] = static_cast<int>(localSlots[permuted].second);
            ++expertFirstTokenOffset[localSlots[permuted].first + 1];
        }
        std::partial_sum(expertFirstTokenOffset.begin(), expertFirstTokenOffset.end(), expertFirstTokenOffset.begin());

        auto const gemmOutput = randomValues<T>(expandedRows * paddedHidden, c.specialValues, rng);
        auto const bias = randomValues<T>(c.withBias ? numExpertsPerNode * paddedHidden : 0, c.specialValues, rng);
        std::vector<float> scales(c.numRows * c.topK);
        auto const specialScales = specialBits<float>();
        std::uniform_int_distribution<size_t> pickSpecialScale(0, specialScales.size() - 1);
        for (auto& scale : scales)
        {
            scale = uniform(rng) * 2.F - 0.5F;
            if (c.specialValues && uniform(rng) < 0.05F)
            {
                scale = fromBits<float>(specialScales[pickSpecialScale(rng)]);
            }
        }
        // Rows that the no-filling (all-to-all) variant does not write keep this pattern.
        std::vector<Bits<T>> outputInit(c.numRows * c.hidden);
        std::uniform_int_distribution<uint32_t> anyBits;
        for (auto& bits : outputInit)
        {
            bits = static_cast<Bits<T>>(anyBits(rng));
        }

        auto const dScales = toDevice(scales);
        auto const dUnpermutedToPermuted = toDevice(unpermutedToPermuted);
        auto const dPermutedToUnpermuted = toDevice(permutedToUnpermuted);
        auto const dTokenSelectedExperts = toDevice(tokenSelectedExperts);
        auto const dExpertFirstTokenOffset = toDevice(expertFirstTokenOffset);
        ck::MOEParallelismConfig const parallelismConfig(c.tpRank + 1, c.tpRank, c.epSize, c.epRank);

        constexpr int64_t kElemsPerVector = 16 / sizeof(T);
        bool const shapeFitsWarpKernel
            = c.topK >= 1 && c.topK <= 32 && c.hidden % kElemsPerVector == 0 && paddedHidden % kElemsPerVector == 0;
        bool const biasIsRead = c.withBias && c.tpRank == 0;
        std::vector<std::vector<Bits<T>>> outputs;
        for (auto const& offsets : offsetsList)
        {
            bool const expectWarpKernel = shapeFitsWarpKernel && offsets.gemmOutput % 16 == 0
                && offsets.output % 16 == 0 && (!biasIsRead || offsets.bias % 16 == 0);
            if (c.numRows == 0 && !expectWarpKernel)
            {
                continue; // The block-per-row kernels cannot be launched with an empty grid.
            }
            auto const dGemmOutput = toDevice(gemmOutput, offsets.gemmOutput / sizeof(T));
            auto const dBias = toDevice(bias, offsets.bias / sizeof(T));
            auto const dOutput = toDevice(outputInit, offsets.output / sizeof(T));
            auto const launch = [&]()
            {
                ck::finalizeMoeRoutingKernelLauncher<T, T, T>(dGemmOutput.data, reinterpret_cast<T*>(dOutput.data),
                    c.withBias ? dBias.data : nullptr, c.withScales ? dScales.data : nullptr,
                    dUnpermutedToPermuted.data, dPermutedToUnpermuted.data, dTokenSelectedExperts.data,
                    dExpertFirstTokenOffset.data, c.numRows, paddedHidden, c.hidden, c.topK, numExpertsPerNode,
                    parallelismConfig, c.enableAlltoall, mStream);
            };
            launch();
            TLLM_CUDA_CHECK(cudaGetLastError());
            TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));
            outputs.push_back(toHost(dOutput.data, outputInit.size()));

            auto const names = capturedKernelNames(launch);
            if (c.numRows == 0)
            {
                EXPECT_TRUE(names.empty()) << "a kernel was launched for an empty batch: " << names.front();
                continue;
            }
            ASSERT_EQ(names.size(), 1U);
            EXPECT_NE(names.front().find("finalizeMoeRouting"), std::string::npos) << names.front();
            EXPECT_EQ(names.front().find("finalizeMoeRoutingWarpKernel") != std::string::npos, expectWarpKernel)
                << names.front() << " (offsets " << offsets.gemmOutput << ", " << offsets.output << ", " << offsets.bias
                << ", topK=" << c.topK << ", hidden=" << c.hidden << ")";
        }

        for (size_t i = 1; i < outputs.size(); ++i)
        {
            auto const mismatch = std::mismatch(outputs[0].begin(), outputs[0].end(), outputs[i].begin());
            EXPECT_EQ(mismatch.first, outputs[0].end())
                << "first difference at element " << (mismatch.first - outputs[0].begin()) << " of "
                << outputs[0].size() << ": bits " << std::hex << static_cast<uint32_t>(*mismatch.first) << " vs "
                << static_cast<uint32_t>(*mismatch.second) << std::dec << " (run " << i << ", numRows=" << c.numRows
                << ", hidden=" << c.hidden << ", topK=" << c.topK << ", epSize=" << c.epSize
                << ", enableAlltoall=" << c.enableAlltoall << ")";
        }
    }

    cudaStream_t mStream{};
};

using FinalizeTypes = ::testing::Types<half, float, __nv_bfloat16>;
TYPED_TEST_SUITE(MoeFinalizeKernelTest, FinalizeTypes);

TYPED_TEST(MoeFinalizeKernelTest, MatchesBlockKernels)
{
    for (int64_t const hidden : {2048, 2880, 7168})
    {
        for (int64_t const topK : {1, 2, 8})
        {
            for (auto const& [epSize, enableAlltoall] : {std::pair{1, false}, {8, false}, {8, true}})
            {
                this->runCase({.numRows = 257,
                    .hidden = hidden,
                    .topK = topK,
                    .numExperts = 128,
                    .epSize = epSize,
                    .epRank = epSize / 2,
                    .enableAlltoall = enableAlltoall,
                    .seed = static_cast<uint32_t>(hidden + topK)});
            }
        }
    }
}

TYPED_TEST(MoeFinalizeKernelTest, SpecialValuesBiasAndScales)
{
    for (auto const& [epSize, enableAlltoall] : {std::pair{1, false}, {4, false}, {4, true}})
    {
        for (bool const withScales : {true, false})
        {
            for (int const tpRank : {0, 1})
            {
                this->runCase({.numRows = 300,
                    .hidden = 2880,
                    .topK = 8,
                    .numExperts = 64,
                    .epSize = epSize,
                    .epRank = epSize - 1,
                    .tpRank = tpRank,
                    .enableAlltoall = enableAlltoall,
                    .withScales = withScales,
                    .withBias = true,
                    .specialValues = true,
                    .seed = static_cast<uint32_t>(tpRank * 2 + withScales)});
            }
        }
    }
}

TYPED_TEST(MoeFinalizeKernelTest, TopKUpToWarpSizeAndBeyond)
{
    // top_k = 32 is the largest the warp kernel takes; top_k = 33 runs the block-per-row kernels in both launches.
    for (int64_t const topK : {3, 31, 32, 33})
    {
        for (bool const enableAlltoall : {false, true})
        {
            this->runCase({.numRows = 111,
                .hidden = 1024,
                .topK = topK,
                .numExperts = 64,
                .epSize = 2,
                .epRank = 1,
                .enableAlltoall = enableAlltoall,
                .withBias = true,
                .seed = static_cast<uint32_t>(topK)});
        }
    }
}

TYPED_TEST(MoeFinalizeKernelTest, PaddedAndNarrowRows)
{
    for (auto const& [hidden, paddedHidden] : {std::pair<int64_t, int64_t>{8, 8}, {24, 24}, {2880, 3072}, {4096, 4224}})
    {
        for (bool const enableAlltoall : {false, true})
        {
            this->runCase({.numRows = 129,
                .hidden = hidden,
                .paddedHidden = paddedHidden,
                .topK = 4,
                .numExperts = 32,
                .epSize = 2,
                .epRank = 1,
                .enableAlltoall = enableAlltoall,
                .withBias = true,
                .specialValues = true,
                .seed = static_cast<uint32_t>(hidden)});
        }
    }
}

TYPED_TEST(MoeFinalizeKernelTest, TokenCounts)
{
    // numRows = 0 launches nothing.
    for (int64_t const numRows : {0, 1, 3, 32, 4099})
    {
        for (bool const enableAlltoall : {false, true})
        {
            this->runCase({.numRows = numRows,
                .hidden = 2048,
                .topK = 8,
                .numExperts = 128,
                .epSize = 8,
                .epRank = 3,
                .enableAlltoall = enableAlltoall,
                .seed = static_cast<uint32_t>(numRows)});
        }
    }
}

TYPED_TEST(MoeFinalizeKernelTest, EachMisalignedBufferFallsBack)
{
    // Aligned, then the GEMM output, the output and the bias shifted one at a time, then all three.
    std::vector<BufferOffsets> const offsetsList{{}, {4, 0, 0}, {0, 8, 0}, {0, 0, 12}, {4, 8, 12}};
    for (bool const enableAlltoall : {false, true})
    {
        this->runCase({.numRows = 97,
                          .hidden = 2048,
                          .topK = 8,
                          .numExperts = 64,
                          .epSize = 4,
                          .epRank = 2,
                          .enableAlltoall = enableAlltoall,
                          .withBias = true,
                          .seed = 11},
            offsetsList);
    }
}

TYPED_TEST(MoeFinalizeKernelTest, MisalignedBiasIgnoredOffTpRank0)
{
    // The bias is only added on TP rank 0, so on other ranks its alignment does not keep the warp kernel from running.
    this->runCase(
        {.numRows = 97, .hidden = 2048, .topK = 8, .numExperts = 64, .tpRank = 1, .withBias = true, .seed = 5},
        {{}, {0, 0, 12}, {4, 4, 4}});
}

} // namespace

} // namespace tensorrt_llm::tests::kernels::moe_finalize
