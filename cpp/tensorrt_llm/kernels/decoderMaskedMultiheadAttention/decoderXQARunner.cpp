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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/workspace.h"
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
    : mDataType(data_type)
    , mNumHeads(num_heads)
    , mNumKVHeads(num_kv_heads)
    , mHeadSize(head_size)
    , mMultiBlockMode(multi_block_mode)
{
    mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

    // TODO: needs both impls because medusa kernels haven't been migrated to JIT yet (which should be).
    // mJITImpl/mPrecompiledImpl assignments must be the last lines of this constructor. DecoderXQAImpl::create() relies
    // on *this being fully initialized.
    mJITImpl = DecoderXQAImpl::create(this, DecoderXQAImpl::ImplType::kJIT);
    mPrecompiledImpl = DecoderXQAImpl::create(this, DecoderXQAImpl::ImplType::kPrecompiled);
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

DecoderXQAImpl* DecoderXQARunner::getImplFromXQAParams(XQAParams const& xqaParams, bool for_configure_plugin)
{
    int const smVersion = tensorrt_llm::common::getSMVersion();

    std::optional<bool> envEnableXQAJIT = tensorrt_llm::common::getEnvEnableXQAJIT();
    if (envEnableXQAJIT.has_value())
    {
        return envEnableXQAJIT.value() ? mJITImpl.get() : mPrecompiledImpl.get();
    }
    else
    {
        if (xqaParams.multi_query_tokens)
        {
            // Some multi_query kernels are not ported to JIT yet.
            auto const grpSize = xqaParams.num_q_heads / xqaParams.num_kv_heads;
            // Hopper XQA supports spec dec with JIT, but only for E4M3 kv cache data type. Only allow 64%grpSize==0 for
            // now.
            bool const supportedByHopperXqa
                = (smVersion == 90 && xqaParams.kv_cache_data_type == XQADataType::DATA_TYPE_E4M3 && grpSize <= 64);
            bool const supportedBySm120Mla = (smVersion == 120 && xqaParams.isMLA()
                && xqaParams.kv_cache_data_type == XQADataType::DATA_TYPE_E4M3);
            bool const supportedByAmpereXqa = (!xqaParams.isMLA() && (64 % grpSize == 0));

            return (supportedByHopperXqa || supportedBySm120Mla || supportedByAmpereXqa) ? mJITImpl.get()
                                                                                         : mPrecompiledImpl.get();
        }
        else
        {
            // regular decoding kernels uses JIT by default
            return mJITImpl.get();
        }
    }
}

bool DecoderXQARunner::shouldUse(XQAParams const& xqa_params, bool for_configure_plugin)
{
    return getImplFromXQAParams(xqa_params, for_configure_plugin)->shouldUse(xqa_params, for_configure_plugin);
}

void DecoderXQARunner::prepareForRun(XQAParams const& xqa_params)
{
    return getImplFromXQAParams(xqa_params, true)->prepare(xqa_params);
}

template <typename KVCacheBuffer>
void DecoderXQARunner::run(
    XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream)
{
    return getImplFromXQAParams(xqa_params, false)->run(xqa_params, kv_cache_buffer, stream);
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

//// DecoderXQARunner::Resource
DecoderXQARunnerResource::DecoderXQARunnerResource()
    : mCubinObjRegistry(std::make_unique<jit::CubinObjRegistry>())
{
}

DecoderXQARunnerResource::DecoderXQARunnerResource(DecoderXQARunnerResource const& other)
    : mCubinObjRegistry(other.mCubinObjRegistry->clone())
{
}

DecoderXQARunnerResource& DecoderXQARunnerResource::operator=(DecoderXQARunnerResource const& other)
{
    if (this == &other)
    {
        return *this;
    }
    mCubinObjRegistry = other.mCubinObjRegistry->clone();
    return *this;
}

DecoderXQARunnerResource::DecoderXQARunnerResource(void const* buffer, size_t buffer_size)
    : mCubinObjRegistry(std::make_unique<jit::CubinObjRegistry>(buffer, buffer_size))
{
}

size_t DecoderXQARunnerResource::getSerializationSize() const noexcept
{
    return mCubinObjRegistry->getSerializationSize();
}

void DecoderXQARunnerResource::serialize(void* buffer, size_t buffer_size) const noexcept
{
    mCubinObjRegistry->serialize(buffer, buffer_size);
}

} // namespace kernels

} // namespace tensorrt_llm
