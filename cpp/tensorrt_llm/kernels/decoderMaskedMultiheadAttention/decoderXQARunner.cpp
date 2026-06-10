/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAConstants.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

DecoderXQARunner::DecoderXQARunner(
    const XQADataType data_type, int num_heads, int num_kv_heads, int head_size, bool multi_block_mode)
    : mDataType(data_type)
    , mNumHeads(num_heads)
    , mNumKVHeads(num_kv_heads)
    , mHeadSize(head_size)
    , mMultiBlockMode(multi_block_mode)
{
    mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

    // This assignment must be the last line of this constructor. DecoderXQAImplJIT::create() relies on *this being
    // fully initialized.
    mJITImpl = DecoderXQAImplJIT::create(this);
}

DecoderXQARunner::~DecoderXQARunner() = default;

DecoderXQAImplJIT* DecoderXQARunner::getImpl()
{
    return mJITImpl.get();
}

bool DecoderXQARunner::shouldUse(XQAParams const& xqa_params, bool for_configure_plugin)
{
    return getImpl()->shouldUse(xqa_params, for_configure_plugin);
}

void DecoderXQARunner::prepareForRun(XQAParams const& xqa_params)
{
    return getImpl()->prepare(xqa_params);
}

template <typename KVCacheBuffer>
void DecoderXQARunner::run(
    XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream)
{
    return getImpl()->run(xqa_params, kv_cache_buffer, stream);
}

std::shared_ptr<DecoderXQARunnerResource> DecoderXQARunner::getResourceGlobal()
{
    static std::mutex sMutex;
    static std::weak_ptr<DecoderXQARunnerResource> sResource;
    std::lock_guard<std::mutex> lock(sMutex);
    auto ret = sResource.lock();
    if (ret != nullptr)
    {
        return ret;
    }
    ret = std::make_shared<DecoderXQARunnerResource>();
    sResource = ret;
    return ret;
}

template void DecoderXQARunner::run(
    XQAParams const& xqa_params, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream);
template void DecoderXQARunner::run(
    XQAParams const& xqa_params, KVBlockArray const& kv_block_array, cudaStream_t const& stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
