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

#include "tensorrt_llm/common/attentionWorkspace.h"

#include "tensorrt_llm/common/sageScaleGeometry.h"
#include "tensorrt_llm/common/workspace.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace tc = tensorrt_llm::common;
namespace tcop = tensorrt_llm::common::op;

namespace
{

constexpr size_t kAlignment = tc::kCudaMemAlign;

void expectSlice(char const* name, tcop::WorkspaceSlice const& slice, size_t expectedOffset, size_t expectedSize)
{
    SCOPED_TRACE(name);
    EXPECT_EQ(slice.offset, expectedOffset);
    EXPECT_EQ(slice.size, expectedSize);
}

template <typename T>
void expectPtrAt(T* ptr, std::uint8_t* base, tcop::WorkspaceSlice const& slice)
{
    EXPECT_EQ(static_cast<void*>(ptr), static_cast<void*>(base + slice.offset));
}

void expectPtrAt(void* ptr, std::uint8_t* base, tcop::WorkspaceSlice const& slice)
{
    EXPECT_EQ(ptr, static_cast<void*>(base + slice.offset));
}

template <typename Slice>
void expectNextSlice(char const* name, Slice const& slice, size_t size, size_t& expectedOffset)
{
    expectSlice(name, slice, expectedOffset, size);
    expectedOffset += tc::alignSize(size, kAlignment);
}

tcop::AttentionContextWorkspaceParams fusedContextParams()
{
    return {.elementSize = 2,
        .batchSize = 3,
        .querySequenceLength = 64,
        .kvSequenceLength = 96,
        .crossKvLength = 96,
        .packedTokenCount = 131,
        .numQueryHeads = 6,
        .numAttnKvHeads = 2,
        .headSize = 64,
        .totalKvTokenCount = 197,
        .rotaryEmbeddingDim = 32,
        .enableContextFmha = true};
}

tcop::AttentionContextWorkspaceSizes fusedContextSizes()
{
    tcop::AttentionContextWorkspaceSizes sizes{};
    sizes.cuQSeqlens = 16;
    sizes.cuKvSeqlens = 16;
    sizes.cuMaskRows = 16;
    sizes.rotaryInvFreq = 192;
    sizes.tokensInfo = 1048;
    sizes.fmhaTileCounter = 4;
    return sizes;
}

void expectContextSizes(
    tcop::AttentionContextWorkspaceSizes const& actual, tcop::AttentionContextWorkspaceSizes const& expected)
{
    EXPECT_EQ(actual.cublasWorkspace, expected.cublasWorkspace) << "cublasWorkspace";
    EXPECT_EQ(actual.attentionMask, expected.attentionMask) << "attentionMask";
    EXPECT_EQ(actual.cuQSeqlens, expected.cuQSeqlens) << "cuQSeqlens";
    EXPECT_EQ(actual.cuKvSeqlens, expected.cuKvSeqlens) << "cuKvSeqlens";
    EXPECT_EQ(actual.cuMaskRows, expected.cuMaskRows) << "cuMaskRows";
    EXPECT_EQ(actual.rotaryInvFreq, expected.rotaryInvFreq) << "rotaryInvFreq";
    EXPECT_EQ(actual.qBuf, expected.qBuf) << "qBuf";
    EXPECT_EQ(actual.kBuf, expected.kBuf) << "kBuf";
    EXPECT_EQ(actual.vBuf, expected.vBuf) << "vBuf";
    EXPECT_EQ(actual.qkBuf, expected.qkBuf) << "qkBuf";
    EXPECT_EQ(actual.qkvBuf, expected.qkvBuf) << "qkvBuf";
    EXPECT_EQ(actual.qkFloatBuf, expected.qkFloatBuf) << "qkFloatBuf";
    EXPECT_EQ(actual.fp8QkvBuf, expected.fp8QkvBuf) << "fp8QkvBuf";
    EXPECT_EQ(actual.fp8QBuf, expected.fp8QBuf) << "fp8QBuf";
    EXPECT_EQ(actual.fp8KBuf, expected.fp8KBuf) << "fp8KBuf";
    EXPECT_EQ(actual.fp8VBuf, expected.fp8VBuf) << "fp8VBuf";
    EXPECT_EQ(actual.paddingOffset, expected.paddingOffset) << "paddingOffset";
    EXPECT_EQ(actual.encoderPaddingOffset, expected.encoderPaddingOffset) << "encoderPaddingOffset";
    EXPECT_EQ(actual.tokensInfo, expected.tokensInfo) << "tokensInfo";
    EXPECT_EQ(actual.fmhaTileCounter, expected.fmhaTileCounter) << "fmhaTileCounter";
    EXPECT_EQ(actual.fmhaBmm1Scale, expected.fmhaBmm1Scale) << "fmhaBmm1Scale";
    EXPECT_EQ(actual.fmhaBmm2Scale, expected.fmhaBmm2Scale) << "fmhaBmm2Scale";
    EXPECT_EQ(actual.sageQScale, expected.sageQScale) << "sageQScale";
    EXPECT_EQ(actual.sageKScale, expected.sageKScale) << "sageKScale";
    EXPECT_EQ(actual.sageVScale, expected.sageVScale) << "sageVScale";
    EXPECT_EQ(actual.sageKMean, expected.sageKMean) << "sageKMean";
    EXPECT_EQ(actual.sageKMeanSum, expected.sageKMeanSum) << "sageKMeanSum";
    EXPECT_EQ(actual.fmhaMultiCtasKvScratch, expected.fmhaMultiCtasKvScratch) << "fmhaMultiCtasKvScratch";
}

} // namespace

TEST(AttentionWorkspaceManagerTest, RaggedUnfusedSelfAttentionUsesPaddedTokenCounts)
{
    auto const sizes = tcop::AttentionWorkspaceManager::buildContextSizes({.elementSize = sizeof(half),
        .batchSize = 4,
        .querySequenceLength = 64,
        .kvSequenceLength = 64,
        .crossKvLength = 0,
        .packedTokenCount = 100,
        .numQueryHeads = 8,
        .numAttnKvHeads = 2,
        .headSize = 64});

    EXPECT_EQ(sizes.attentionMask, 32768);
    EXPECT_EQ(sizes.cuQSeqlens, 20);
    EXPECT_EQ(sizes.cuKvSeqlens, 20);
    EXPECT_EQ(sizes.cuMaskRows, 20);
    EXPECT_EQ(sizes.qBuf, 262144);
    EXPECT_EQ(sizes.kBuf, 65536);
    EXPECT_EQ(sizes.vBuf, 65536);
    EXPECT_EQ(sizes.qkBuf, 262144);
    EXPECT_EQ(sizes.qkvBuf, 262144);
    EXPECT_EQ(sizes.qkFloatBuf, 524288);
    EXPECT_EQ(sizes.paddingOffset, 1024);
    EXPECT_EQ(sizes.encoderPaddingOffset, 0);
    EXPECT_EQ(sizes.tokensInfo, 800);
}

TEST(AttentionWorkspaceManagerTest, RaggedUnfusedCrossAttentionUsesPaddedTokenCounts)
{
    auto const sizes = tcop::AttentionWorkspaceManager::buildContextSizes({.elementSize = sizeof(half),
        .batchSize = 4,
        .querySequenceLength = 64,
        .kvSequenceLength = 32,
        .crossKvLength = 32,
        .packedTokenCount = 100,
        .numQueryHeads = 8,
        .numAttnKvHeads = 2,
        .headSize = 64});

    EXPECT_EQ(sizes.attentionMask, 16384);
    EXPECT_EQ(sizes.cuQSeqlens, 20);
    EXPECT_EQ(sizes.cuKvSeqlens, 20);
    EXPECT_EQ(sizes.cuMaskRows, 20);
    EXPECT_EQ(sizes.qBuf, 262144);
    EXPECT_EQ(sizes.kBuf, 32768);
    EXPECT_EQ(sizes.vBuf, 32768);
    EXPECT_EQ(sizes.qkBuf, 131072);
    EXPECT_EQ(sizes.qkvBuf, 262144);
    EXPECT_EQ(sizes.qkFloatBuf, 262144);
    EXPECT_EQ(sizes.paddingOffset, 1024);
    EXPECT_EQ(sizes.encoderPaddingOffset, 512);
    EXPECT_EQ(sizes.tokensInfo, 800);
}

TEST(AttentionWorkspaceManagerTest, UnfusedGroupedQuerySizesDistinguishQueryAndKvHeads)
{
    tcop::AttentionContextWorkspaceParams params{.elementSize = sizeof(half),
        .batchSize = 3,
        .querySequenceLength = 5,
        .kvSequenceLength = 2,
        .crossKvLength = 2,
        .packedTokenCount = 7,
        .numQueryHeads = 4,
        .numAttnKvHeads = 1,
        .headSize = 8};
    auto const sizes = tcop::AttentionWorkspaceManager::buildContextSizes(params);
    EXPECT_EQ(sizes.attentionMask, 60);
    EXPECT_EQ(sizes.qBuf, 960);
    EXPECT_EQ(sizes.qkvBuf, 960);
    EXPECT_EQ(sizes.kBuf, 96);
    EXPECT_EQ(sizes.vBuf, 96);
    EXPECT_EQ(sizes.qkBuf, 240);
    EXPECT_EQ(sizes.qkFloatBuf, 480);
    EXPECT_EQ(sizes.paddingOffset, 60);
    EXPECT_EQ(sizes.encoderPaddingOffset, 24);
    EXPECT_EQ(sizes.tokensInfo, sizeof(int2) * 7);

    params.packedTokenCount = 15;
    auto const denseSizes = tcop::AttentionWorkspaceManager::buildContextSizes(params);
    EXPECT_EQ(denseSizes.qBuf, sizes.qBuf);
    EXPECT_EQ(denseSizes.kBuf, sizes.kBuf);
    EXPECT_EQ(denseSizes.qkBuf, sizes.qkBuf);
    EXPECT_EQ(denseSizes.paddingOffset, sizes.paddingOffset);
    EXPECT_EQ(denseSizes.tokensInfo, sizeof(int2) * 15);
}

TEST(AttentionWorkspaceManagerTest, UnfusedFloatContextKeepsAccumulatorAndMetadataWidths)
{
    auto params = fusedContextParams();
    params.elementSize = 4;
    params.enableContextFmha = false;
    auto expected = fusedContextSizes();
    expected.attentionMask = 73728;
    expected.qBuf = 294912;
    expected.kBuf = 147456;
    expected.vBuf = 147456;
    expected.qkBuf = 442368;
    expected.qkvBuf = 294912;
    expected.qkFloatBuf = 442368;
    expected.paddingOffset = 768;
    expected.encoderPaddingOffset = 1152;
    expected.fmhaTileCounter = 0;
    expectContextSizes(tcop::AttentionWorkspaceManager::buildContextSizes(params), expected);
}

TEST(AttentionWorkspaceManagerTest, FusedPackedContextNeedsOnlyMetadata)
{
    expectContextSizes(tcop::AttentionWorkspaceManager::buildContextSizes(fusedContextParams()), fusedContextSizes());
}

TEST(AttentionWorkspaceManagerTest, FusedSeparateContextStagesQueryAtActivationWidth)
{
    auto params = fusedContextParams();
    params.separateQAndKvInput = true;
    auto expected = fusedContextSizes();
    expected.qBuf = 100608;
    expectContextSizes(tcop::AttentionWorkspaceManager::buildContextSizes(params), expected);

    params.elementSize = 4;
    expected.qBuf = 201216;
    expectContextSizes(tcop::AttentionWorkspaceManager::buildContextSizes(params), expected);
}

TEST(AttentionWorkspaceManagerTest, Fp8PackedContextStagesCombinedQkv)
{
    auto params = fusedContextParams();
    params.fp8ContextFmha = true;
    auto expected = fusedContextSizes();
    expected.fp8QkvBuf = 83840;
    expected.fmhaBmm1Scale = 8;
    expected.fmhaBmm2Scale = 4;
    expectContextSizes(tcop::AttentionWorkspaceManager::buildContextSizes(params), expected);
}

TEST(AttentionWorkspaceManagerTest, Fp8SeparateContextStagesOneByteQueries)
{
    auto params = fusedContextParams();
    params.fp8ContextFmha = true;
    params.separateQAndKvInput = true;
    auto expected = fusedContextSizes();
    expected.qBuf = 50304;
    expected.fmhaBmm1Scale = 8;
    expected.fmhaBmm2Scale = 4;
    expectContextSizes(tcop::AttentionWorkspaceManager::buildContextSizes(params), expected);
}

TEST(AttentionWorkspaceManagerTest, SageContiguousContextStagesPackedKvAndSmoothing)
{
    auto params = fusedContextParams();
    params.fp8ContextFmha = true;
    params.separateQAndKvInput = true;
    params.sageAttention = true;
    params.sageSmoothK = true;
    params.sageQBlockSize = 16;
    params.sageKBlockSize = 32;
    params.sageVBlockSize = 1;
    auto expected = fusedContextSizes();
    expected.qBuf = 50304;
    expected.fp8QBuf = 50304;
    expected.fp8KBuf = 25216;
    expected.fp8VBuf = 25216;
    expected.fmhaBmm1Scale = 8;
    expected.fmhaBmm2Scale = 4;
    expected.sageQScale = 264;
    expected.sageKScale = 72;
    expected.sageVScale = 512;
    expected.sageKMean = 512;
    expected.sageKMeanSum = 1024;
    expectContextSizes(tcop::AttentionWorkspaceManager::buildContextSizes(params), expected);
}

TEST(AttentionWorkspaceManagerTest, SageHopperContextUsesAlignedPerThreadScaleGeometry)
{
    auto params = fusedContextParams();
    params.totalKvTokenCount = 213;
    params.fp8ContextFmha = true;
    params.separateQAndKvInput = true;
    params.sageAttention = true;
    params.hopperSage = true;
    params.sageQBlockSize = 2;
    params.sageKBlockSize = 16;
    params.sageVBlockSize = 1;
    auto expected = fusedContextSizes();
    expected.qBuf = 50304;
    expected.fp8QBuf = 50304;
    expected.fp8KBuf = 27264;
    expected.fp8VBuf = 27264;
    expected.fmhaBmm1Scale = 8;
    expected.fmhaBmm2Scale = 4;
    expected.sageQScale = 3864;
    expected.sageKScale = 608;
    expected.sageVScale = 512;
    expectContextSizes(tcop::AttentionWorkspaceManager::buildContextSizes(params), expected);
}

TEST(AttentionWorkspaceManagerTest, SageScaleAndSmoothingSizesDoNotRequireSeparateStaging)
{
    auto params = fusedContextParams();
    params.sageQBlockSize = 16;
    params.sageKBlockSize = 32;
    params.sageVBlockSize = 3;
    params.sageSmoothK = true;
    auto expected = fusedContextSizes();
    expected.sageQScale = 264;
    expected.sageKScale = 72;
    expected.sageVScale = 172;
    expected.sageKMean = 512;
    expected.sageKMeanSum = 1024;
    expectContextSizes(tcop::AttentionWorkspaceManager::buildContextSizes(params), expected);
}

TEST(AttentionWorkspaceManagerTest, SparseContextScratchDependsOnSmCountAndHeadSize)
{
    auto params = fusedContextParams();
    params.sparseFmha = true;
    params.multiProcessorCount = 108;
    auto expected = fusedContextSizes();
    expected.fmhaMultiCtasKvScratch = 3760128;
    expectContextSizes(tcop::AttentionWorkspaceManager::buildContextSizes(params), expected);

    params.multiProcessorCount = 120;
    params.headSize = 128;
    expected.fmhaMultiCtasKvScratch = 8110080;
    expectContextSizes(tcop::AttentionWorkspaceManager::buildContextSizes(params), expected);

    params.sparseFmha = false;
    expected.fmhaMultiCtasKvScratch = 0;
    expectContextSizes(tcop::AttentionWorkspaceManager::buildContextSizes(params), expected);
}

TEST(AttentionWorkspaceManagerTest, SageScaleStridesHandleTileBoundariesAndDisabledQuantization)
{
    using Partition = tc::SageScalePartition;

    struct Case
    {
        Partition partition;
        int blockSize;
        int tokens;
        int batchSize;
        int expected;
    };

    std::array const cases{Case{Partition::Contiguous, 1, 65, 3, 67}, Case{Partition::Contiguous, 4, 65, 3, 19},
        Case{Partition::Contiguous, 16, 65, 3, 7}, Case{Partition::Contiguous, 32, 65, 3, 5},
        Case{Partition::HopperQ, 16, 65, 3, 128}, Case{Partition::HopperK, 32, 213, 3, 76},
        Case{Partition::Contiguous, 8, 65, 3, 0}, Case{Partition::HopperQ, 0, 65, 3, 0},
        Case{Partition::Contiguous, -1, 65, 3, 0}, Case{Partition::Contiguous, 16, 0, 3, 0},
        Case{Partition::HopperK, 32, 213, 0, 0}};
    for (auto const& test : cases)
    {
        EXPECT_EQ(
            tc::getSageScaleHeadStride(test.partition, test.blockSize, test.tokens, test.batchSize), test.expected)
            << static_cast<int>(test.partition) << ": block " << test.blockSize << ", tokens " << test.tokens
            << ", batch " << test.batchSize;
    }
}

TEST(AttentionWorkspaceManagerTest, ContextLayoutMatchesAttentionOpOrdering)
{
    tcop::AttentionContextWorkspaceSizes sizes{};
    sizes.cublasWorkspace = 13;
    sizes.attentionMask = 17;
    sizes.cuQSeqlens = 19;
    sizes.cuKvSeqlens = 23;
    sizes.cuMaskRows = 29;
    sizes.rotaryInvFreq = 31;
    sizes.qBuf = 37;
    sizes.kBuf = 41;
    sizes.vBuf = 43;
    sizes.qkBuf = 47;
    sizes.qkvBuf = 53;
    sizes.qkFloatBuf = 59;
    sizes.fp8QkvBuf = 61;
    sizes.fp8QBuf = 67;
    sizes.fp8KBuf = 71;
    sizes.fp8VBuf = 0;
    sizes.paddingOffset = 73;
    sizes.encoderPaddingOffset = 0;
    sizes.tokensInfo = 79;
    sizes.fmhaTileCounter = 83;
    sizes.fmhaBmm1Scale = 89;
    sizes.fmhaBmm2Scale = 97;
    sizes.sageQScale = 101;
    sizes.sageKScale = 103;
    sizes.sageVScale = 107;
    sizes.sageKMean = 109;
    sizes.sageKMeanSum = 113;

    auto const layout = tcop::AttentionWorkspaceManager::buildContextLayout(sizes);

    size_t expectedOffset = 0;
    expectNextSlice("cublasWorkspace", layout.cublasWorkspace, sizes.cublasWorkspace, expectedOffset);
    expectNextSlice("attentionMask", layout.attentionMask, sizes.attentionMask, expectedOffset);
    expectNextSlice("cuQSeqlens", layout.cuQSeqlens, sizes.cuQSeqlens, expectedOffset);
    expectNextSlice("cuKvSeqlens", layout.cuKvSeqlens, sizes.cuKvSeqlens, expectedOffset);
    expectNextSlice("cuMaskRows", layout.cuMaskRows, sizes.cuMaskRows, expectedOffset);
    expectNextSlice("rotaryInvFreq", layout.rotaryInvFreq, sizes.rotaryInvFreq, expectedOffset);
    expectNextSlice("qBuf", layout.qBuf, sizes.qBuf, expectedOffset);
    expectNextSlice("kBuf", layout.kBuf, sizes.kBuf, expectedOffset);
    expectNextSlice("vBuf", layout.vBuf, sizes.vBuf, expectedOffset);
    expectNextSlice("qkBuf", layout.qkBuf, sizes.qkBuf, expectedOffset);
    expectNextSlice("qkvBuf", layout.qkvBuf, sizes.qkvBuf, expectedOffset);
    expectNextSlice("qkFloatBuf", layout.qkFloatBuf, sizes.qkFloatBuf, expectedOffset);
    expectNextSlice("fp8QkvBuf", layout.fp8QkvBuf, sizes.fp8QkvBuf, expectedOffset);
    expectNextSlice("fp8QBuf", layout.fp8QBuf, sizes.fp8QBuf, expectedOffset);
    expectNextSlice("fp8KBuf", layout.fp8KBuf, sizes.fp8KBuf, expectedOffset);
    expectNextSlice("fp8VBuf", layout.fp8VBuf, sizes.fp8VBuf, expectedOffset);
    expectNextSlice("paddingOffset", layout.paddingOffset, sizes.paddingOffset, expectedOffset);
    expectNextSlice("encoderPaddingOffset", layout.encoderPaddingOffset, sizes.encoderPaddingOffset, expectedOffset);
    expectNextSlice("tokensInfo", layout.tokensInfo, sizes.tokensInfo, expectedOffset);
    expectNextSlice("fmhaTileCounter", layout.fmhaTileCounter, sizes.fmhaTileCounter, expectedOffset);
    expectNextSlice("fmhaBmm1Scale", layout.fmhaBmm1Scale, sizes.fmhaBmm1Scale, expectedOffset);
    expectNextSlice("fmhaBmm2Scale", layout.fmhaBmm2Scale, sizes.fmhaBmm2Scale, expectedOffset);
    expectNextSlice("sageQScale", layout.sageQScale, sizes.sageQScale, expectedOffset);
    expectNextSlice("sageKScale", layout.sageKScale, sizes.sageKScale, expectedOffset);
    expectNextSlice("sageVScale", layout.sageVScale, sizes.sageVScale, expectedOffset);
    expectNextSlice("sageKMean", layout.sageKMean, sizes.sageKMean, expectedOffset);
    expectNextSlice("sageKMeanSum", layout.sageKMeanSum, sizes.sageKMeanSum, expectedOffset);
    EXPECT_EQ(layout.totalSize, expectedOffset);
}

TEST(AttentionWorkspaceManagerTest, MaterializeContextReturnsTypedViewsAndNullZeroSlices)
{

    tcop::AttentionContextWorkspaceSizes sizes{};
    sizes.cublasWorkspace = 0;
    sizes.attentionMask = sizeof(float) * 4;
    sizes.cuQSeqlens = sizeof(int) * 3;
    sizes.cuKvSeqlens = sizeof(int) * 3;
    sizes.rotaryInvFreq = 0;
    sizes.qBuf = sizeof(float) * 8;
    sizes.fp8QBuf = sizeof(__nv_fp8_e4m3) * 5;
    sizes.tokensInfo = sizeof(int2) * 2;
    sizes.fmhaTileCounter = sizeof(uint32_t);
    sizes.fmhaBmm1Scale = sizeof(float) * 2;

    auto const layout = tcop::AttentionWorkspaceManager::buildContextLayout(sizes);

    constexpr size_t kWorkspaceSize = 8192;
    alignas(kAlignment) std::array<std::uint8_t, kWorkspaceSize> workspace{};
    auto* base = workspace.data();

    auto const views = tcop::AttentionWorkspaceManager::materializeContext<float>(workspace.data(), layout);

    EXPECT_EQ(views.cublasWorkspace, nullptr);
    expectPtrAt(views.attentionMask, base, layout.attentionMask);
    expectPtrAt(views.cuQSeqlens, base, layout.cuQSeqlens);
    expectPtrAt(views.cuKvSeqlens, base, layout.cuKvSeqlens);
    EXPECT_EQ(views.rotaryInvFreq, nullptr);
    expectPtrAt(views.qBuf, base, layout.qBuf);
    expectPtrAt(views.fp8QBuf, base, layout.fp8QBuf);
    expectPtrAt(views.tokensInfo, base, layout.tokensInfo);
    expectPtrAt(views.fmhaTileCounter, base, layout.fmhaTileCounter);
    expectPtrAt(views.fmhaBmm1Scale, base, layout.fmhaBmm1Scale);
}

TEST(AttentionWorkspaceManagerTest, GenerationLayoutPlacesCpWorkspaceBeforePartialBuffers)
{

    tcop::AttentionGenerationWorkspaceSizes sizes{};
    sizes.partialOut = 33;
    sizes.partialSum = 9;
    sizes.partialMax = 0;
    sizes.shiftKCache = 5;

    auto const layout = tcop::AttentionWorkspaceManager::buildGenerationLayout(sizes);

    size_t expectedOffset = 0;
    expectNextSlice("partialOut", layout.partialOut, sizes.partialOut, expectedOffset);
    expectNextSlice("partialSum", layout.partialSum, sizes.partialSum, expectedOffset);
    expectNextSlice("partialMax", layout.partialMax, sizes.partialMax, expectedOffset);
    expectNextSlice("shiftKCache", layout.shiftKCache, sizes.shiftKCache, expectedOffset);
    EXPECT_EQ(layout.totalSize, expectedOffset);

    constexpr size_t kWorkspaceSize = 4096;
    alignas(kAlignment) std::array<std::uint8_t, kWorkspaceSize> workspace{};
    auto* base = workspace.data();

    auto const views = tcop::AttentionWorkspaceManager::materializeGeneration<float>(workspace.data(), layout);

    expectPtrAt(views.partialOut, base, layout.partialOut);
    expectPtrAt(views.partialSum, base, layout.partialSum);
    EXPECT_EQ(views.partialMax, nullptr);
    expectPtrAt(views.shiftKCache, base, layout.shiftKCache);
}

TEST(AttentionWorkspaceManagerTest, XqaLayoutUsesSingleSparseCacheSlice)
{
    tcop::AttentionXqaWorkspaceSizes sizes{};
    sizes.cuSeqlens = 4;
    sizes.cuKvSeqlens = 8;
    sizes.rotaryInvFreq = 0;
    sizes.tokensInfo = 16;
    sizes.bmm1Scale = 8;
    sizes.bmm2Scale = 4;
    sizes.sparseAttnCache = 31;
    sizes.kernelWorkspace = 64;

    auto const layout = tcop::AttentionWorkspaceManager::buildXqaLayout(sizes);

    size_t expectedOffset = 0;
    expectNextSlice("cuSeqlens", layout.cuSeqlens, sizes.cuSeqlens, expectedOffset);
    expectNextSlice("cuKvSeqlens", layout.cuKvSeqlens, sizes.cuKvSeqlens, expectedOffset);
    expectNextSlice("rotaryInvFreq", layout.rotaryInvFreq, sizes.rotaryInvFreq, expectedOffset);
    expectNextSlice("tokensInfo", layout.tokensInfo, sizes.tokensInfo, expectedOffset);
    expectNextSlice("bmm1Scale", layout.bmm1Scale, sizes.bmm1Scale, expectedOffset);
    expectNextSlice("bmm2Scale", layout.bmm2Scale, sizes.bmm2Scale, expectedOffset);

    auto const sparseCacheOffset = expectedOffset;
    expectNextSlice("sparseAttnCache", layout.sparseAttnCache, sizes.sparseAttnCache, expectedOffset);
    EXPECT_EQ(layout.kernelWorkspace.offset, sparseCacheOffset + tc::alignSize(sizes.sparseAttnCache, kAlignment));
    expectNextSlice("kernelWorkspace", layout.kernelWorkspace, sizes.kernelWorkspace, expectedOffset);
    EXPECT_EQ(layout.totalSize, expectedOffset);
}

TEST(AttentionWorkspaceManagerTest, FlashMlaLayoutKeepsAccumulatorOrder)
{
    tcop::AttentionFlashMlaWorkspaceSizes sizes{};
    sizes.tileSchedulerMetadata = 8;
    sizes.numSplits = 12;
    sizes.softmaxLse = 20;
    sizes.softmaxLseAccum = 24;
    sizes.outAccum = 28;

    auto const layout = tcop::AttentionWorkspaceManager::buildFlashMlaLayout(sizes);

    size_t expectedOffset = 0;
    expectNextSlice("tileSchedulerMetadata", layout.tileSchedulerMetadata, sizes.tileSchedulerMetadata, expectedOffset);
    expectNextSlice("numSplits", layout.numSplits, sizes.numSplits, expectedOffset);
    expectNextSlice("softmaxLse", layout.softmaxLse, sizes.softmaxLse, expectedOffset);
    expectNextSlice("softmaxLseAccum", layout.softmaxLseAccum, sizes.softmaxLseAccum, expectedOffset);
    expectNextSlice("outAccum", layout.outAccum, sizes.outAccum, expectedOffset);
    EXPECT_EQ(layout.totalSize, expectedOffset);
}
