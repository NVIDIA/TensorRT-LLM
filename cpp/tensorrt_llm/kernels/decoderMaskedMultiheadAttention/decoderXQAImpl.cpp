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
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImpl.h"

#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplPrecompiled.h"

#include <cassert>
#include <functional>
#include <memory>

namespace tensorrt_llm
{
namespace kernels
{

template <>
void DecoderXQAImpl::run(const XQAParams& xqa_params, KVLinearBuffer& kv_linear_buffer,
    int2& rotary_kernel_launch_cache, const cudaStream_t& stream)
{
    runWithKVLinearBuffer(xqa_params, kv_linear_buffer, rotary_kernel_launch_cache, stream);
}

template <>
void DecoderXQAImpl::run(const XQAParams& xqa_params, KVBlockArray& kv_block_array, int2& rotary_kernel_launch_cache,
    const cudaStream_t& stream)
{
    runWithKVBlockArray(xqa_params, kv_block_array, rotary_kernel_launch_cache, stream);
}

std::unique_ptr<DecoderXQAImpl> DecoderXQAImpl::create(DecoderXQARunner* runner, ImplType implType)
{
    switch (implType)
    {
    case ImplType::kPrecompiled: return std::unique_ptr<DecoderXQAImpl>(new DecoderXQAImplPrecompiled(runner));
    // TODO(minwei): JIT impl.
    case ImplType::kJIT: return nullptr;
    }
    // Shouldn't reach here.
    assert(false);
    return nullptr;
}

} // namespace kernels
} // namespace tensorrt_llm
