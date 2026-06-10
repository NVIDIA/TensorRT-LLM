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
#include "tensorrt_llm/common/config.h"

#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplCommon.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/compileEngine.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/cubinObjRegistry.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include <unordered_set>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Forward declaration to avoid cyclic dependency.
class DecoderXQARunner;
class DecoderXQARunnerResource;

/**
 * The underlying XQA implementation called from DecoderXQARunner.
 *
 * XQA kernels are compiled on the fly via NVRTC.
 */
class DecoderXQAImplJIT
{
public:
    // TODO: shouldUse()/prepare() should be templated with KVCacheBuffer.
    // Whether it is beneficial to use this XQA codepath.
    //
    // forConfigurePlugin: whether this method is called in configure plugin phase.
    DecoderXQAImplJIT(DecoderXQARunner* runner);

    bool shouldUse(XQAParams const& xqaParams, bool forConfigurePlugin);
    // Prepares for the kernel running. Must be called before calling run.
    void prepare(XQAParams const& xqaParams);

    // Run XQA kernel with KVCacheBuffer.
    //
    // Sub-classes should implement runWithKVLinearBuffer and runWithKVBlockArray.
    template <typename KVCacheBuffer>
    void run(XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream);

    // Needs runner pointer for accessing resources in DecoderXQARunner class.
    static std::unique_ptr<DecoderXQAImplJIT> create(DecoderXQARunner* runner);

    ~DecoderXQAImplJIT() = default;

protected:
    void runWithKVLinearBuffer(
        XQAParams const& xqaParams, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream);
    void runWithKVBlockArray(
        XQAParams const& xqaParams, KVBlockArray const& kv_block_array, cudaStream_t const& stream);

private:
    DecoderXQARunner* mRunner;

    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;
    std::shared_ptr<DecoderXQARunnerResource> mResource;

    //! Whether DecoderXQAImplJIT needs to compile 2 sets (tilesize = 16, 32) kernels for spec-dec
    bool needHMMASpecDec(XQAParams const& xqaParams, bool forConfigurePlugin) const;

    //! Whether DecoderXQAImplJIT supports xqaParams.
    bool supportConfig(XQAParams const& xqaParams, bool forConfigurePlugin) const;
    //! Whether DecoderXQAImplJIT has perf gain over the default (non-XQA-optimized) implementation.
    bool mayHavePerfGain(XQAParams const& xqaParams) const;

    void prepareForActualXQAParams(XQAParams const& xqaParams);

    template <typename T, typename KVCacheBuffer>
    void runImpl(XQAParams const& xqaParams, KVCacheBuffer const& kv_cache_buffer, int multiprocessor_count,
        cudaStream_t const& stream) const;

    template <typename KVCacheBuffer>
    void runDispatchKVCacheBuffer(
        XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream);

    bool mForceXQA;
    int mSM;

    jit::CubinObjKey getCubinObjKeyFromXQAParams(XQAParams const& xqaParams) const;
};

} // namespace kernels

TRTLLM_NAMESPACE_END
