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

} // namespace

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
    sizes.cpWorkspace = 109;

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
    expectNextSlice("cpWorkspace", layout.cpWorkspace, sizes.cpWorkspace, expectedOffset);
    EXPECT_EQ(layout.totalSize, expectedOffset);
}

TEST(AttentionWorkspaceManagerTest, MaterializeContextReturnsTypedViewsAndNullZeroSlices)
{
    constexpr size_t kCpMaxPaddedSequenceLength = 2;
    constexpr int kHeadSize = 4;
    constexpr int kNumHeads = 2;
    constexpr int kNumKvHeads = 1;
    constexpr size_t kCpBufferElements = kCpMaxPaddedSequenceLength * kHeadSize * (kNumHeads + 2 * kNumKvHeads);

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
    sizes.cpWorkspace = 2 * kCpBufferElements * sizeof(float) + sizeof(int) * 3;

    auto const layout = tcop::AttentionWorkspaceManager::buildContextLayout(sizes);

    constexpr size_t kWorkspaceSize = 8192;
    alignas(kAlignment) std::array<std::uint8_t, kWorkspaceSize> workspace{};
    auto* base = workspace.data();

    auto const views = tcop::AttentionWorkspaceManager::materializeContext<float>(
        workspace.data(), layout, kCpMaxPaddedSequenceLength, kHeadSize, kNumHeads, kNumKvHeads);

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
    expectPtrAt(views.gatherInBuffer, base, layout.cpWorkspace);

    auto* const expectedGatherOutBuffer
        = reinterpret_cast<float*>(base + layout.cpWorkspace.offset) + kCpBufferElements;
    auto* const expectedCuCpPartialSeqlens = reinterpret_cast<int*>(expectedGatherOutBuffer + kCpBufferElements);
    EXPECT_EQ(static_cast<void*>(views.gatherOutBuffer), static_cast<void*>(expectedGatherOutBuffer));
    EXPECT_EQ(static_cast<void*>(views.cuCpPartialSeqlens), static_cast<void*>(expectedCuCpPartialSeqlens));
}

TEST(AttentionWorkspaceManagerTest, GenerationLayoutPlacesCpWorkspaceBeforePartialBuffers)
{
    constexpr size_t kCpMaxPaddedSequenceLength = 3;
    constexpr int kNumHeads = 2;
    constexpr int kNumKvHeads = 1;
    constexpr int kHeadSize = 4;
    constexpr size_t kCpBufferElements = kCpMaxPaddedSequenceLength * (kNumHeads + 2 * kNumKvHeads) * kHeadSize;

    tcop::AttentionGenerationWorkspaceSizes sizes{};
    sizes.cpWorkspace = 2 * kCpBufferElements * sizeof(float);
    sizes.partialOut = 33;
    sizes.partialSum = 9;
    sizes.partialMax = 0;
    sizes.shiftKCache = 5;

    auto const layout = tcop::AttentionWorkspaceManager::buildGenerationLayout(sizes);

    size_t expectedOffset = 0;
    expectNextSlice("cpWorkspace", layout.cpWorkspace, sizes.cpWorkspace, expectedOffset);
    expectNextSlice("partialOut", layout.partialOut, sizes.partialOut, expectedOffset);
    expectNextSlice("partialSum", layout.partialSum, sizes.partialSum, expectedOffset);
    expectNextSlice("partialMax", layout.partialMax, sizes.partialMax, expectedOffset);
    expectNextSlice("shiftKCache", layout.shiftKCache, sizes.shiftKCache, expectedOffset);
    EXPECT_EQ(layout.totalSize, expectedOffset);

    constexpr size_t kWorkspaceSize = 4096;
    alignas(kAlignment) std::array<std::uint8_t, kWorkspaceSize> workspace{};
    auto* base = workspace.data();

    auto const views = tcop::AttentionWorkspaceManager::materializeGeneration<float>(
        workspace.data(), layout, kCpMaxPaddedSequenceLength, kNumHeads, kNumKvHeads, kHeadSize);

    expectPtrAt(views.mhaOutput, base, layout.cpWorkspace);
    auto* const expectedMhaInput = reinterpret_cast<float*>(base + layout.cpWorkspace.offset) + kCpBufferElements;
    EXPECT_EQ(static_cast<void*>(views.mhaInput), static_cast<void*>(expectedMhaInput));
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
