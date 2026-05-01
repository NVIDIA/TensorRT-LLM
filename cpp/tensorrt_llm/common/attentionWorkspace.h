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

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/workspace.h"

#include <cstddef>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace common::op
{

struct WorkspaceSlice
{
    size_t offset{};
    size_t size{};
};

struct AttentionContextWorkspaceSizes
{
    size_t cublasWorkspace{CUBLAS_WORKSPACE_SIZE};
    size_t attentionMask{};
    size_t cuQSeqlens{};
    size_t cuKvSeqlens{};
    size_t cuMaskRows{};
    size_t rotaryInvFreq{};
    size_t qBuf{};
    size_t kBuf{};
    size_t vBuf{};
    size_t qkBuf{};
    size_t qkvBuf{};
    size_t qkFloatBuf{};
    size_t fp8QkvBuf{};
    size_t fp8QBuf{};
    size_t fp8KBuf{};
    size_t fp8VBuf{};
    size_t paddingOffset{};
    size_t encoderPaddingOffset{};
    size_t tokensInfo{};
    size_t fmhaTileCounter{};
    size_t fmhaBmm1Scale{};
    size_t fmhaBmm2Scale{};
    size_t sageQScale{};
    size_t sageKScale{};
    size_t sageVScale{};
    size_t cpWorkspace{};
    size_t fmhaMultiCtasKvScratch{};
};

struct AttentionContextWorkspaceLayout
{
    WorkspaceSlice cublasWorkspace{};
    WorkspaceSlice attentionMask{};
    WorkspaceSlice cuQSeqlens{};
    WorkspaceSlice cuKvSeqlens{};
    WorkspaceSlice cuMaskRows{};
    WorkspaceSlice rotaryInvFreq{};
    WorkspaceSlice qBuf{};
    WorkspaceSlice kBuf{};
    WorkspaceSlice vBuf{};
    WorkspaceSlice qkBuf{};
    WorkspaceSlice qkvBuf{};
    WorkspaceSlice qkFloatBuf{};
    WorkspaceSlice fp8QkvBuf{};
    WorkspaceSlice fp8QBuf{};
    WorkspaceSlice fp8KBuf{};
    WorkspaceSlice fp8VBuf{};
    WorkspaceSlice paddingOffset{};
    WorkspaceSlice encoderPaddingOffset{};
    WorkspaceSlice tokensInfo{};
    WorkspaceSlice fmhaTileCounter{};
    WorkspaceSlice fmhaBmm1Scale{};
    WorkspaceSlice fmhaBmm2Scale{};
    WorkspaceSlice sageQScale{};
    WorkspaceSlice sageKScale{};
    WorkspaceSlice sageVScale{};
    WorkspaceSlice cpWorkspace{};
    WorkspaceSlice fmhaMultiCtasKvScratch{};
    size_t totalSize{};
};

template <typename T>
struct AttentionContextWorkspaceViews
{
    void* cublasWorkspace{};
    T* attentionMask{};
    int* cuQSeqlens{};
    int* cuKvSeqlens{};
    int* cuMaskRows{};
    float* rotaryInvFreq{};
    T* qBuf{};
    T* kBuf{};
    T* vBuf{};
    T* qkBuf{};
    T* qkvBuf{};
    float* qkFloatBuf{};
    __nv_fp8_e4m3* fp8QkvBuf{};
    __nv_fp8_e4m3* fp8QBuf{};
    __nv_fp8_e4m3* fp8KBuf{};
    __nv_fp8_e4m3* fp8VBuf{};
    int* paddingOffset{};
    int* encoderPaddingOffset{};
    int2* tokensInfo{};
    uint32_t* fmhaTileCounter{};
    float* fmhaBmm1Scale{};
    float* fmhaBmm2Scale{};
    float* sageQScale{};
    float* sageKScale{};
    float* sageVScale{};
    T* gatherInBuffer{};
    T* gatherOutBuffer{};
    int* cuCpPartialSeqlens{};
    void* fmhaMultiCtasKvScratch{};
};

struct AttentionGenerationWorkspaceSizes
{
    size_t cpWorkspace{};
    size_t partialOut{};
    size_t partialSum{};
    size_t partialMax{};
    size_t shiftKCache{};
};

struct AttentionGenerationWorkspaceLayout
{
    WorkspaceSlice cpWorkspace{};
    WorkspaceSlice partialOut{};
    WorkspaceSlice partialSum{};
    WorkspaceSlice partialMax{};
    WorkspaceSlice shiftKCache{};
    size_t totalSize{};
};

template <typename T>
struct AttentionGenerationWorkspaceViews
{
    T* mhaOutput{};
    T* mhaInput{};
    T* partialOut{};
    float* partialSum{};
    float* partialMax{};
    T* shiftKCache{};
};

struct AttentionFlashMlaWorkspaceSizes
{
    size_t tileSchedulerMetadata{};
    size_t numSplits{};
    size_t softmaxLse{};
    size_t softmaxLseAccum{};
    size_t outAccum{};
};

struct AttentionFlashMlaWorkspaceLayout
{
    WorkspaceSlice tileSchedulerMetadata{};
    WorkspaceSlice numSplits{};
    WorkspaceSlice softmaxLse{};
    WorkspaceSlice softmaxLseAccum{};
    WorkspaceSlice outAccum{};
    size_t totalSize{};
};

struct AttentionXqaWorkspaceSizes
{
    size_t cuSeqlens{};
    size_t cuKvSeqlens{};
    size_t rotaryInvFreq{};
    size_t tokensInfo{};
    size_t bmm1Scale{};
    size_t bmm2Scale{};
    size_t sparseAttnCache{};
    size_t kernelWorkspace{};
};

struct AttentionXqaWorkspaceLayout
{
    WorkspaceSlice cuSeqlens{};
    WorkspaceSlice cuKvSeqlens{};
    WorkspaceSlice rotaryInvFreq{};
    WorkspaceSlice tokensInfo{};
    WorkspaceSlice bmm1Scale{};
    WorkspaceSlice bmm2Scale{};
    WorkspaceSlice sparseAttnCache{};
    WorkspaceSlice kernelWorkspace{};
    size_t totalSize{};
};

class AttentionWorkspaceManager
{
public:
    static AttentionContextWorkspaceLayout buildContextLayout(
        AttentionContextWorkspaceSizes const& sizes, uintptr_t alignment = common::kCudaMemAlign)
    {
        AttentionContextWorkspaceLayout layout{};
        size_t offset = 0;
        layout.cublasWorkspace = nextSlice(offset, sizes.cublasWorkspace, alignment);
        layout.attentionMask = nextSlice(offset, sizes.attentionMask, alignment);
        layout.cuQSeqlens = nextSlice(offset, sizes.cuQSeqlens, alignment);
        layout.cuKvSeqlens = nextSlice(offset, sizes.cuKvSeqlens, alignment);
        layout.cuMaskRows = nextSlice(offset, sizes.cuMaskRows, alignment);
        layout.rotaryInvFreq = nextSlice(offset, sizes.rotaryInvFreq, alignment);
        layout.qBuf = nextSlice(offset, sizes.qBuf, alignment);
        layout.kBuf = nextSlice(offset, sizes.kBuf, alignment);
        layout.vBuf = nextSlice(offset, sizes.vBuf, alignment);
        layout.qkBuf = nextSlice(offset, sizes.qkBuf, alignment);
        layout.qkvBuf = nextSlice(offset, sizes.qkvBuf, alignment);
        layout.qkFloatBuf = nextSlice(offset, sizes.qkFloatBuf, alignment);
        layout.fp8QkvBuf = nextSlice(offset, sizes.fp8QkvBuf, alignment);
        layout.fp8QBuf = nextSlice(offset, sizes.fp8QBuf, alignment);
        layout.fp8KBuf = nextSlice(offset, sizes.fp8KBuf, alignment);
        layout.fp8VBuf = nextSlice(offset, sizes.fp8VBuf, alignment);
        layout.paddingOffset = nextSlice(offset, sizes.paddingOffset, alignment);
        layout.encoderPaddingOffset = nextSlice(offset, sizes.encoderPaddingOffset, alignment);
        layout.tokensInfo = nextSlice(offset, sizes.tokensInfo, alignment);
        layout.fmhaTileCounter = nextSlice(offset, sizes.fmhaTileCounter, alignment);
        layout.fmhaBmm1Scale = nextSlice(offset, sizes.fmhaBmm1Scale, alignment);
        layout.fmhaBmm2Scale = nextSlice(offset, sizes.fmhaBmm2Scale, alignment);
        layout.sageQScale = nextSlice(offset, sizes.sageQScale, alignment);
        layout.sageKScale = nextSlice(offset, sizes.sageKScale, alignment);
        layout.sageVScale = nextSlice(offset, sizes.sageVScale, alignment);
        layout.cpWorkspace = nextSlice(offset, sizes.cpWorkspace, alignment);
        layout.fmhaMultiCtasKvScratch = nextSlice(offset, sizes.fmhaMultiCtasKvScratch, alignment);
        layout.totalSize = offset;
        return layout;
    }

    template <typename T>
    static AttentionContextWorkspaceViews<T> materializeContext(void* workspace,
        AttentionContextWorkspaceLayout const& layout, size_t cpMaxPaddedSequenceLength, int headSize, int numHeads,
        int numKvHeads)
    {
        AttentionContextWorkspaceViews<T> views{};
        views.cublasWorkspace = ptr<void>(workspace, layout.cublasWorkspace);
        views.attentionMask = ptr<T>(workspace, layout.attentionMask);
        views.cuQSeqlens = ptr<int>(workspace, layout.cuQSeqlens);
        views.cuKvSeqlens = ptr<int>(workspace, layout.cuKvSeqlens);
        views.cuMaskRows = ptr<int>(workspace, layout.cuMaskRows);
        views.rotaryInvFreq = ptr<float>(workspace, layout.rotaryInvFreq);
        views.qBuf = ptr<T>(workspace, layout.qBuf);
        views.kBuf = ptr<T>(workspace, layout.kBuf);
        views.vBuf = ptr<T>(workspace, layout.vBuf);
        views.qkBuf = ptr<T>(workspace, layout.qkBuf);
        views.qkvBuf = ptr<T>(workspace, layout.qkvBuf);
        views.qkFloatBuf = ptr<float>(workspace, layout.qkFloatBuf);
        views.fp8QkvBuf = ptr<__nv_fp8_e4m3>(workspace, layout.fp8QkvBuf);
        views.fp8QBuf = ptr<__nv_fp8_e4m3>(workspace, layout.fp8QBuf);
        views.fp8KBuf = ptr<__nv_fp8_e4m3>(workspace, layout.fp8KBuf);
        views.fp8VBuf = ptr<__nv_fp8_e4m3>(workspace, layout.fp8VBuf);
        views.paddingOffset = ptr<int>(workspace, layout.paddingOffset);
        views.encoderPaddingOffset = ptr<int>(workspace, layout.encoderPaddingOffset);
        views.tokensInfo = ptr<int2>(workspace, layout.tokensInfo);
        views.fmhaTileCounter = ptr<uint32_t>(workspace, layout.fmhaTileCounter);
        views.fmhaBmm1Scale = ptr<float>(workspace, layout.fmhaBmm1Scale);
        views.fmhaBmm2Scale = ptr<float>(workspace, layout.fmhaBmm2Scale);
        views.sageQScale = ptr<float>(workspace, layout.sageQScale);
        views.sageKScale = ptr<float>(workspace, layout.sageKScale);
        views.sageVScale = ptr<float>(workspace, layout.sageVScale);

        views.gatherInBuffer = ptr<T>(workspace, layout.cpWorkspace);
        if (views.gatherInBuffer != nullptr)
        {
            auto const cpBufferElements
                = cpMaxPaddedSequenceLength * static_cast<size_t>(headSize) * (numHeads + 2 * numKvHeads);
            views.gatherOutBuffer = views.gatherInBuffer + cpBufferElements;
            views.cuCpPartialSeqlens = reinterpret_cast<int*>(views.gatherOutBuffer + cpBufferElements);
        }
        views.fmhaMultiCtasKvScratch = ptr<void>(workspace, layout.fmhaMultiCtasKvScratch);
        return views;
    }

    static AttentionGenerationWorkspaceLayout buildGenerationLayout(
        AttentionGenerationWorkspaceSizes const& sizes, uintptr_t alignment = common::kCudaMemAlign)
    {
        AttentionGenerationWorkspaceLayout layout{};
        size_t offset = 0;
        layout.cpWorkspace = nextSlice(offset, sizes.cpWorkspace, alignment);
        layout.partialOut = nextSlice(offset, sizes.partialOut, alignment);
        layout.partialSum = nextSlice(offset, sizes.partialSum, alignment);
        layout.partialMax = nextSlice(offset, sizes.partialMax, alignment);
        layout.shiftKCache = nextSlice(offset, sizes.shiftKCache, alignment);
        layout.totalSize = offset;
        return layout;
    }

    template <typename T>
    static AttentionGenerationWorkspaceViews<T> materializeGeneration(void* workspace,
        AttentionGenerationWorkspaceLayout const& layout, size_t cpMaxPaddedSequenceLength, int numHeads,
        int numKvHeads, int headSize)
    {
        AttentionGenerationWorkspaceViews<T> views{};
        views.mhaOutput = ptr<T>(workspace, layout.cpWorkspace);
        if (views.mhaOutput != nullptr)
        {
            auto const cpBufferElements
                = cpMaxPaddedSequenceLength * (numHeads + 2 * numKvHeads) * static_cast<size_t>(headSize);
            views.mhaInput = views.mhaOutput + cpBufferElements;
        }
        views.partialOut = ptr<T>(workspace, layout.partialOut);
        views.partialSum = ptr<float>(workspace, layout.partialSum);
        views.partialMax = ptr<float>(workspace, layout.partialMax);
        views.shiftKCache = ptr<T>(workspace, layout.shiftKCache);
        return views;
    }

    static AttentionFlashMlaWorkspaceLayout buildFlashMlaLayout(
        AttentionFlashMlaWorkspaceSizes const& sizes, uintptr_t alignment = common::kCudaMemAlign)
    {
        AttentionFlashMlaWorkspaceLayout layout{};
        size_t offset = 0;
        layout.tileSchedulerMetadata = nextSlice(offset, sizes.tileSchedulerMetadata, alignment);
        layout.numSplits = nextSlice(offset, sizes.numSplits, alignment);
        layout.softmaxLse = nextSlice(offset, sizes.softmaxLse, alignment);
        layout.softmaxLseAccum = nextSlice(offset, sizes.softmaxLseAccum, alignment);
        layout.outAccum = nextSlice(offset, sizes.outAccum, alignment);
        layout.totalSize = offset;
        return layout;
    }

    static AttentionXqaWorkspaceLayout buildXqaLayout(
        AttentionXqaWorkspaceSizes const& sizes, uintptr_t alignment = common::kCudaMemAlign)
    {
        AttentionXqaWorkspaceLayout layout{};
        size_t offset = 0;
        layout.cuSeqlens = nextSlice(offset, sizes.cuSeqlens, alignment);
        layout.cuKvSeqlens = nextSlice(offset, sizes.cuKvSeqlens, alignment);
        layout.rotaryInvFreq = nextSlice(offset, sizes.rotaryInvFreq, alignment);
        layout.tokensInfo = nextSlice(offset, sizes.tokensInfo, alignment);
        layout.bmm1Scale = nextSlice(offset, sizes.bmm1Scale, alignment);
        layout.bmm2Scale = nextSlice(offset, sizes.bmm2Scale, alignment);
        layout.sparseAttnCache = nextSlice(offset, sizes.sparseAttnCache, alignment);
        layout.kernelWorkspace = nextSlice(offset, sizes.kernelWorkspace, alignment);
        layout.totalSize = offset;
        return layout;
    }

    template <typename T>
    static T* ptr(void* workspace, WorkspaceSlice const& slice)
    {
        if (workspace == nullptr || slice.size == 0)
        {
            return nullptr;
        }
        return reinterpret_cast<T*>(reinterpret_cast<int8_t*>(workspace) + slice.offset);
    }

private:
    static WorkspaceSlice nextSlice(size_t& offset, size_t size, uintptr_t alignment)
    {
        auto const slice = WorkspaceSlice{offset, size};
        offset += common::alignSize(size, alignment);
        return slice;
    }
};

} // namespace common::op

TRTLLM_NAMESPACE_END
