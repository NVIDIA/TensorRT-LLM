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

    bool shouldUse(const XQAParams& xqaParams) override;
    void prepare(const XQAParams& xqa_params) override;

protected:
    void runWithKVLinearBuffer(const XQAParams& xqa_params, KVLinearBuffer& kv_linear_buffer,
        int2& rotary_kernel_launch_cache, const cudaStream_t& stream) override;
    void runWithKVBlockArray(const XQAParams& xqa_params, KVBlockArray& kv_block_array,
        int2& rotary_kernel_launch_cache, const cudaStream_t& stream) override;

private:
    template <typename KVCacheBuffer>
    void runDispatchBuffer(const XQAParams& xqa_params, KVCacheBuffer& kv_cache_buffer,
        int2& rotary_kernel_launch_cache, const cudaStream_t& stream);
};

} // namespace kernels
} // namespace tensorrt_llm
