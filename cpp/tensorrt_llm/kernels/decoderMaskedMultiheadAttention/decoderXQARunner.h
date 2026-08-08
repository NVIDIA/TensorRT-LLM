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

#include "decoderXQARunnerResource.h"

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/cubinObjRegistry.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/xqaParams.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

class DecoderXQARunner
{
public:
    DecoderXQARunner(XQADataType const dataType, int numHeads, int numKVHeads, int headSize, bool multiBlockMode);
    ~DecoderXQARunner();

    /**
     * \param[in] xqaParams the xqaParams to be tested against.
     * \param[in] forConfigurePlugin indicates whether this method is called in configurePlugin, or in
     * enqueueGeneration.
     * TODO: shouldUse()/prepare() should be templated with KVCacheBuffer.
     * Whether it is beneficial to use this XQA codepath.
     */
    bool shouldUse(XQAParams const& xqaParams, bool forConfigurePlugin);
    // Prepares for the kernel running. Must be called before calling run.
    void prepare(XQAParams const& xqaParams);

    template <typename KVCacheBuffer>
    void dispatch(XQAParams const& xqaParams, KVCacheBuffer const& kvCacheBuffer, cudaStream_t const& stream)
    {
        sync_check_cuda_error(stream);
        this->run(xqaParams, kvCacheBuffer, stream);
    }

    static std::shared_ptr<DecoderXQARunnerResource> getResourceGlobal();

private:
    template <typename KVCacheBuffer>
    void run(XQAParams const& xqaParams, KVCacheBuffer const& kvCacheBuffer, cudaStream_t const& stream);

    //! Whether DecoderXQARunner needs to compile 2 sets (tilesize = 16, 32) kernels for spec-dec
    bool needHMMASpecDec(XQAParams const& xqaParams, bool forConfigurePlugin) const;

    //! Whether DecoderXQARunner supports xqaParams.
    bool supportConfig(XQAParams const& xqaParams, bool forConfigurePlugin) const;
    //! Whether DecoderXQARunner has perf gain over the default (non-XQA-optimized) implementation.
    bool mayHavePerfGain(XQAParams const& xqaParams) const;

    jit::CubinObjKey getCubinObjKeyFromXQAParams(XQAParams const& xqaParams) const;

    void prepareForActualXQAParams(XQAParams const& xqaParams);

    template <typename KVCacheBuffer>
    void runDispatchKVCacheBuffer(
        XQAParams const& xqaParams, KVCacheBuffer const& kvCacheBuffer, cudaStream_t const& stream);

    template <typename T, typename KVCacheBuffer>
    void runImpl(XQAParams const& xqaParams, KVCacheBuffer const& kvCacheBuffer, int multiprocessorCount,
        cudaStream_t const& stream) const;

    static constexpr int kMaxBeamWidth = 4;

    XQADataType mDataType;
    int mNumHeads;
    int mNumKVHeads;
    int mHeadSize;
    bool mMultiBlockMode;
    int mMultiProcessorCount;
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;
    std::shared_ptr<DecoderXQARunnerResource> mResource;
    bool mForceXQA;
    int mSM;
};

} // namespace kernels

TRTLLM_NAMESPACE_END
