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
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImpl.h"

#include "compileEngine.h"
#include "cubinObjRegistry.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplCommon.h"
#include <unordered_set>

namespace tensorrt_llm
{
namespace kernels
{

class DecoderXQAImplJIT : public DecoderXQAImpl
{
public:
    DecoderXQAImplJIT(DecoderXQARunner* runner);

    bool shouldUse(XQAParams const& xqaParams, bool forConfigurePlugin) override;
    void prepare(XQAParams const& xqaParams) override;

protected:
    void runWithKVLinearBuffer(
        XQAParams const& xqaParams, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream) override;
    void runWithKVBlockArray(
        XQAParams const& xqaParams, KVBlockArray const& kv_block_array, cudaStream_t const& stream) override;

private:
    void initSupportedConfigs();
    //! Whether DecoderXQAImplJIT supports xqaParams.
    bool supportConfig(XQAParams const& xqaParams) const;
    //! Whether DecoderXQAImplJIT has perf gain over the default (non-XQA-optimized) implementation.
    bool mayHavePerfGain(XQAParams const& xqaParams) const;

    template <typename T, typename KVCacheBuffer>
    void runImpl(XQAParams const& xqaParams, KVCacheBuffer const& kv_cache_buffer, int multiprocessor_count,
        cudaStream_t const& stream);

    template <typename KVCacheBuffer>
    void runDispatchKVCacheBuffer(
        XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream);

    bool mForceXQA;
    int mSM;

    jit::CubinObjRegistry* mCubinObjRegistry;
    jit::CubinObjKey getCubinObjKeyFromXQAParams(XQAParams const& xqaParams) const;

    //! The first prototype just takes whatever available from the Precompiled cubins.
    std::unordered_set<XQAKernelRuntimeHashKey, XQAKernelRuntimeHasher> mSupportedConfigs;
};

} // namespace kernels
} // namespace tensorrt_llm
