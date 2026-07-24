/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "dora.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/doraScaling.h"

#include <NvInferRuntime.h>
#include <numeric>
#include <vector>

using tensorrt_llm::kernels::DoraImpl;

DoraImpl::DoraImpl(std::vector<int> const& outHiddenSizes, nvinfer1::DataType type)
    : mType(type)
{
    mCumModuleSizes.resize(outHiddenSizes.size());
    // TODO(oargov): reserve max size for mHostBuf and make it pinned memory for better h2d copy
    std::partial_sum(outHiddenSizes.cbegin(), outHiddenSizes.cend(), mCumModuleSizes.begin());
}

size_t DoraImpl::getWorkspaceElemCount(int64_t const numTokens) const
{
    auto const numModules = mCumModuleSizes.size();
    // numModules pointers per token, plus numModules for cumModuleSizes
    return (numModules + numTokens * numModules);
}

size_t DoraImpl::getWorkspaceSize(int64_t const numTokens) const
{
    return getWorkspaceElemCount(numTokens) * sizeof(int64_t);
}

int DoraImpl::run(int64_t numTokens, void const* input, void const* const* loraWeightsPtr, void* const* outputs,
    void* workspace, cudaStream_t stream)
{
    if (numTokens == 0)
    {
        return 0;
    }

    auto const numModules = mCumModuleSizes.size();
    auto const workspaceElems = getWorkspaceElemCount(numTokens);
    auto const numel = mCumModuleSizes.back() * numTokens;

    // put all host data in a single host buffer to make only one h2d copy
    mHostBuf.clear();
    mHostBuf.reserve(workspaceElems);
    // push module sizes
    mHostBuf.assign(mCumModuleSizes.cbegin(), mCumModuleSizes.cend());
    // push scale ptrs
    std::for_each(loraWeightsPtr, loraWeightsPtr + numTokens * numModules,
        [this](auto const ptr) { this->mHostBuf.push_back(reinterpret_cast<int64_t>(ptr)); });

    tensorrt_llm::common::cudaAutoCpy(
        (int8_t*) workspace, (int8_t*) mHostBuf.data(), workspaceElems * sizeof(int64_t), stream);

    auto const* deviceCumModuleSizes = reinterpret_cast<int64_t const*>(workspace);
    auto const* deviceScalePtrs = reinterpret_cast<void const* const*>((&deviceCumModuleSizes[numModules]));

    if (mType == nvinfer1::DataType::kHALF)
    {
        tokenPerChannelScale<half>(numel, numModules, numTokens, deviceCumModuleSizes,
            reinterpret_cast<half const*>(input), reinterpret_cast<half const* const*>(deviceScalePtrs),
            reinterpret_cast<half*>(outputs[0]), stream);
    }
#ifdef ENABLE_BF16
    else if (mType == nvinfer1::DataType::kBF16)
    {
        tokenPerChannelScale<nv_bfloat16>(numel, numModules, numTokens, deviceCumModuleSizes,
            reinterpret_cast<nv_bfloat16 const*>(input), reinterpret_cast<nv_bfloat16 const* const*>(deviceScalePtrs),
            reinterpret_cast<nv_bfloat16*>(outputs[0]), stream);
    }
#endif
    else
    {
        TLLM_CHECK_WITH_INFO(false, "DoRA is not implemented for the selected dtype");
    }

    return 0;
}
