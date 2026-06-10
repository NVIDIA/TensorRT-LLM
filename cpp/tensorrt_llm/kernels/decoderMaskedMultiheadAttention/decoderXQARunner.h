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

#pragma once

#include <NvInferRuntime.h>
#include <cuda_fp16.h>

#include "decoderXQARunnerResource.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/cubinObjRegistry.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/xqaParams.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

class DecoderXQARunner
{
public:
    DecoderXQARunner(
        const XQADataType data_type, int num_heads, int num_kv_heads, int head_size, bool multi_block_mode);
    ~DecoderXQARunner();

    /**
     * \param[in] xqaParams the xqaParams to be tested against.
     * \param[in] forConfigurePlugin indicates whether this method is called in configurePlugin, or in
     * enqueueGeneration.
     */
    bool shouldUse(XQAParams const& xqaParams, bool forConfigurePlugin);

    void prepare(XQAParams const& xqa_params)
    {
        this->prepareForRun(xqa_params);
    }

    template <typename KVCacheBuffer>
    void dispatch(XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream)
    {
        sync_check_cuda_error(stream);
        this->run(xqa_params, kv_cache_buffer, stream);
    }

    static std::shared_ptr<DecoderXQARunnerResource> getResourceGlobal();

private:
    void prepareForRun(XQAParams const& xqa_params);

    template <typename KVCacheBuffer>
    void run(XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream);

    static constexpr int kMaxBeamWidth = 4;

    XQADataType mDataType;
    int mNumHeads;
    int mNumKVHeads;
    int mHeadSize;
    bool mMultiBlockMode;
    int mMultiProcessorCount;

    std::unique_ptr<DecoderXQAImpl> mJITImpl;
    DecoderXQAImpl* getImpl();

    friend DecoderXQAImplJIT;
};

} // namespace kernels

TRTLLM_NAMESPACE_END
