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
    if (xqaParams.multi_query_tokens)
    {
        auto const grpSize = xqaParams.num_q_heads / xqaParams.num_kv_heads;
        // Ampere XQA supports spec dec with pre-compiled cubins (may also work with JIT but not implemented yet)
        // Hopper XQA supports spec dec with JIT, but only for E4M3 kv cache data type. Only allow 64%grpSize==0 for
        // now.
        return (tensorrt_llm::common::getSMVersion() == 90
                   && xqaParams.kv_cache_data_type == XQADataType::DATA_TYPE_E4M3 && 64 % grpSize == 0)
            ? mJITImpl.get()
            : mPrecompiledImpl.get();
    }

    std::optional<bool> envEnableXQAJIT = tensorrt_llm::common::getEnvEnableXQAJIT();

    if (envEnableXQAJIT.has_value())
    {
        return envEnableXQAJIT.value() ? mJITImpl.get() : mPrecompiledImpl.get();
    }
    else
    {
        // If no env var set, default to precompiled impl for sm120, otherwise default to JIT.
        return tensorrt_llm::common::getSMVersion() == 120 ? mPrecompiledImpl.get() : mJITImpl.get();
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

DecoderXQARunner::Resource* DecoderXQARunner::getResourceGlobal()
{
    static DecoderXQARunner::Resource sResource;
    return &sResource;
}

template void DecoderXQARunner::run(
    XQAParams const& xqa_params, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream);
template void DecoderXQARunner::run(
    XQAParams const& xqa_params, KVBlockArray const& kv_block_array, cudaStream_t const& stream);

//// DecoderXQARunner::Resource
DecoderXQARunner::Resource::Resource()
    : mCubinObjRegistry(std::make_unique<jit::CubinObjRegistry>())
{
}

DecoderXQARunner::Resource::Resource(DecoderXQARunner::Resource const& other)
    : mCubinObjRegistry(other.mCubinObjRegistry->clone())
{
}

DecoderXQARunner::Resource& DecoderXQARunner::Resource::operator=(DecoderXQARunner::Resource const& other)
{
    if (this == &other)
    {
        return *this;
    }
    mCubinObjRegistry = other.mCubinObjRegistry->clone();
    return *this;
}

DecoderXQARunner::Resource::Resource(void const* buffer, size_t buffer_size)
    : mCubinObjRegistry(std::make_unique<jit::CubinObjRegistry>(buffer, buffer_size))
{
}

size_t DecoderXQARunner::Resource::getSerializationSize() const noexcept
{
    return mCubinObjRegistry->getSerializationSize();
}

void DecoderXQARunner::Resource::serialize(void* buffer, size_t buffer_size) const noexcept
{
    mCubinObjRegistry->serialize(buffer, buffer_size);
}

} // namespace kernels

} // namespace tensorrt_llm
