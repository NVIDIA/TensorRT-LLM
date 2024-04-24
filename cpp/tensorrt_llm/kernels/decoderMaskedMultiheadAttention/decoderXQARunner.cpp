/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "decoderXQARunner.h"

#include <assert.h>
#include <string.h>

#include <mutex>
#include <unordered_map>

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/cubin/xqa_kernel_cubin.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAConstants.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImpl.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

namespace tensorrt_llm
{
namespace kernels
{

DecoderXQARunner::DecoderXQARunner(
    const XQADataType data_type, int num_heads, int num_kv_heads, int head_size, bool multi_block_mode)
    : mPrepareCalled(false)
    , mDataType(data_type)
    , mNumHeads(num_heads)
    , mNumKVHeads(num_kv_heads)
    , mHeadSize(head_size)
    , mMultiBlockMode(multi_block_mode)
{
    mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
    // The initialization of mImpl must be the last line because *this needs to be fully initialized before calling
    // DecoderXQAImpl::create().
    mImpl = DecoderXQAImpl::create(this, DecoderXQAImpl::ImplType::kPrecompiled);
}

DecoderXQARunner::~DecoderXQARunner() = default;

namespace
{

template <typename T>
constexpr inline T divUp(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
constexpr inline T roundUp(T a, T b)
{
    return divUp(a, b) * b;
}

} // namespace

size_t DecoderXQARunner::getWorkspaceSize(int max_batch_beam_size)
{
    size_t workspace_size = 0;
    if (mMultiBlockMode)
    {
        int workspaces[4];
        int const max_num_request = max_batch_beam_size;
        uint32_t const nbSeq = mNumKVHeads * max_num_request;
        uint32_t const nbSubSeq = xqaMaxNbCtaPerKVHeadFactor() * nbSeq;
        int group_size = mNumHeads / mNumKVHeads;
        workspaces[0] = sizeof(uint32_t) * nbSeq;
        workspaces[1] = sizeof(float) * roundUp(group_size, 32) * nbSubSeq;
        workspaces[2] = sizeof(float) * roundUp(group_size, 32) * nbSubSeq;
        int32_t const multi_block_workspace_alignment
            = roundUp<int32_t>(sizeof(__half) * kMaxBeamWidth * group_size * mHeadSize, 128);
        workspaces[3] = multi_block_workspace_alignment * xqaMaxNbCtaPerKVHeadFactor() * mNumKVHeads
            * divUp(max_batch_beam_size, kMaxBeamWidth);
        workspace_size = roundUp(workspaces[0], multi_block_workspace_alignment)
            + roundUp(workspaces[1], multi_block_workspace_alignment)
            + roundUp(workspaces[2], multi_block_workspace_alignment)
            + roundUp(workspaces[3], multi_block_workspace_alignment)
            + multi_block_workspace_alignment; // extra space reserved for alignment
    }
    return workspace_size;
}

bool DecoderXQARunner::shouldUseImpl(XQAParams const& xqaParams)
{
    return mImpl->shouldUse(xqaParams);
}

void DecoderXQARunner::prepareForRun(XQAParams const& xqa_params)
{
    return mImpl->prepare(xqa_params);
}

template <typename KVCacheBuffer>
void DecoderXQARunner::run(
    XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream)
{
    return mImpl->run(xqa_params, kv_cache_buffer, stream);
}

template void DecoderXQARunner::run(
    XQAParams const& xqa_params, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream);
template void DecoderXQARunner::run(
    XQAParams const& xqa_params, KVBlockArray const& kv_block_array, cudaStream_t const& stream);

} // namespace kernels

} // namespace tensorrt_llm
