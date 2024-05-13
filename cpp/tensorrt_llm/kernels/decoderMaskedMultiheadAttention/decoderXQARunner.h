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

#pragma once

#include <NvInferRuntime.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/cubinObjRegistry.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/decoderXQAImplJIT.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplPrecompiled.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/xqaParams.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T, typename KVCacheBuffer>
struct XQADispatchHelper
{
    static constexpr bool CanSupport = false;
};

template <>
struct XQADispatchHelper<__half, KVLinearBuffer>
{
    static constexpr bool CanSupport = true;
};

template <>
struct XQADispatchHelper<__half, KVBlockArray>
{
    static constexpr bool CanSupport = true;
};

#ifdef ENABLE_BF16
template <>
struct XQADispatchHelper<__nv_bfloat16, KVLinearBuffer>
{
    static constexpr bool CanSupport = true;
};

template <>
struct XQADispatchHelper<__nv_bfloat16, KVBlockArray>
{
    static constexpr bool CanSupport = true;
};
#endif

#define SUPPORT_RETURN_FALSE(X)                                                                                        \
    {                                                                                                                  \
        return false;                                                                                                  \
    }

class DecoderXQARunner
{
public:
    // Resources for constructing a DecoderXQARunner object.
    class Resource;
    DecoderXQARunner(Resource* resource, const XQADataType data_type, int num_heads, int num_kv_heads, int head_size,
        bool multi_block_mode);
    ~DecoderXQARunner();

    /**
     * \param[in] xqaParams the xqaParams to be tested against.
     * \param[in] forConfigurePlugin indicates whether this method is called in configurePlugin, or in
     * enqueueGeneration.
     */
    template <typename T>
    bool shouldUse(XQAParams const& xqaParams, bool forConfigurePlugin)
    {
        if (!(xqaParams.data_type == DATA_TYPE_FP16 || xqaParams.data_type == DATA_TYPE_BF16))
        {
            SUPPORT_RETURN_FALSE("data type");
        }
        bool const isGPTJBeam4Kernel = (xqaParams.head_size == 256 && xqaParams.beam_width == 4
            && xqaParams.paged_kv_cache && (xqaParams.tokens_per_block == 64 || xqaParams.tokens_per_block == 128));
        if (xqaParams.head_size != 128 && xqaParams.head_size != 256 && !isGPTJBeam4Kernel)
        {
            SUPPORT_RETURN_FALSE("head_size");
        }
        if (xqaParams.unidirectional != 1)
        {
            SUPPORT_RETURN_FALSE("unidirectional");
        }
        if (xqaParams.q_scaling != 1.0f)
        {
            SUPPORT_RETURN_FALSE("q_scaling");
        }
        if (xqaParams.mask_type != tensorrt_llm::kernels::AttentionMaskType::CAUSAL)
        {
            SUPPORT_RETURN_FALSE("mask_type");
        }
        if (xqaParams.cross_attention)
        {
            SUPPORT_RETURN_FALSE("cross_attention");
        }
        // Only support 64/128 tokens per block.
        if (xqaParams.paged_kv_cache && xqaParams.tokens_per_block != 64 && xqaParams.tokens_per_block != 128)
        {
            SUPPORT_RETURN_FALSE("paged_kv_cache");
        }
        if (!forConfigurePlugin && xqaParams.host_past_key_value_lengths == nullptr)
        {
            SUPPORT_RETURN_FALSE("host_past_key_value_lengths");
        }
        if (xqaParams.beam_width != 1 && !isGPTJBeam4Kernel)
        {
            SUPPORT_RETURN_FALSE("beam_width");
        }
        if (xqaParams.cyclic_attention_window_size != xqaParams.max_attention_window_size)
        {
            SUPPORT_RETURN_FALSE("cyclic_attention_window_size != max_attention_window_size");
        }
        if (xqaParams.position_shift_enabled || xqaParams.sink_token_length > 0)
        {
            SUPPORT_RETURN_FALSE("streaming-llm");
        }

        // OPTIMIZE: For the standard generation-phase MHA, there are still extra limitations.
        // NOTE: Medusa mode = Multi_query_tokens > 1.
        int const nbQHeads = xqaParams.num_q_heads;
        int const nbKVHeads = xqaParams.num_kv_heads;
        int const nbQHeadsPerKV = nbQHeads / nbKVHeads;
        if (!xqaParams.multi_query_tokens)
        {
            if (nbQHeadsPerKV != 8 && nbQHeadsPerKV != 1)
            {
                SUPPORT_RETURN_FALSE("nbHeads");
            }
        }
        else
        {
            // Number of Q heads Per KV needs to be power of 2 or 1.
            if (!(nbQHeadsPerKV % 2 == 0 || nbQHeadsPerKV == 1))
            {
                SUPPORT_RETURN_FALSE("nbHeads");
            }
        }
        return shouldUseImpl(xqaParams, forConfigurePlugin);
    }

    size_t getWorkspaceSize(int max_batch_beam_size);

    void prepare(XQAParams const& xqa_params)
    {
        this->prepareForRun(xqa_params);
    }

    template <typename KVCacheBuffer>
    void dispatch(XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream)
    {
        sync_check_cuda_error();
        this->run(xqa_params, kv_cache_buffer, stream);
    }

private:
    bool shouldUseImpl(XQAParams const& xqa_params, bool for_configure_plugin);
    void prepareForRun(XQAParams const& xqa_params);

    template <typename KVCacheBuffer>
    void run(XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream);

    static constexpr int kMaxBeamWidth = 4;

    Resource* mResource;

    XQADataType mDataType;
    int mNumHeads;
    int mNumKVHeads;
    int mHeadSize;
    bool mMultiBlockMode;
    int mMultiProcessorCount;

    std::unique_ptr<DecoderXQAImpl> mJITImpl, mPrecompiledImpl;
    DecoderXQAImpl* getImplFromXQAParams(XQAParams const& params);

    friend DecoderXQAImplPrecompiled;
    friend DecoderXQAImplJIT;
};

class DecoderXQARunner::Resource
{
public:
    Resource();
    Resource(Resource const& other);
    Resource& operator=(Resource const& other);
    Resource(Resource&& other) = default;
    Resource& operator=(Resource&& other) = default;
    // Construct from a serialized buffer.
    Resource(void const* buffer, size_t buffer_size);
    ~Resource() = default;

    jit::CubinObjRegistry* getCubinObjRegistry()
    {
        return mCubinObjRegistry.get();
    }

    size_t getSerializationSize() const noexcept;
    void serialize(void* buffer, size_t buffer_size) const noexcept;

private:
    std::unique_ptr<jit::CubinObjRegistry> mCubinObjRegistry;
};

} // namespace kernels
} // namespace tensorrt_llm
