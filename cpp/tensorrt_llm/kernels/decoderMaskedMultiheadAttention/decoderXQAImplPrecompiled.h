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

namespace tensorrt_llm
{
namespace kernels
{

class DecoderXQAImplPrecompiled : public DecoderXQAImpl
{
public:
    DecoderXQAImplPrecompiled(DecoderXQARunner* runner)
        : DecoderXQAImpl(runner)
    {
    }

    bool shouldUse(XQAParams const& xqaParams, bool forConfigurePlugin) override;
    void prepare(XQAParams const& xqa_params) override;

    ~DecoderXQAImplPrecompiled() override = default;

protected:
    void runWithKVLinearBuffer(
        XQAParams const& xqa_params, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream) override;
    void runWithKVBlockArray(
        XQAParams const& xqa_params, KVBlockArray const& kv_block_array, cudaStream_t const& stream) override;

private:
    template <typename KVCacheBuffer>
    void runDispatchBuffer(
        XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream);
};

} // namespace kernels
} // namespace tensorrt_llm
