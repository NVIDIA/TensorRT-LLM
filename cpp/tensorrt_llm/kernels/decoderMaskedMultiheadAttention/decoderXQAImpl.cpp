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

#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/decoderXQAImplJIT.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplPrecompiled.h"

#include <cassert>
#include <functional>
#include <memory>

namespace tensorrt_llm
{
namespace kernels
{

template <>
void DecoderXQAImpl::run(
    XQAParams const& xqa_params, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream)
{
    runWithKVLinearBuffer(xqa_params, kv_linear_buffer, stream);
}

template <>
void DecoderXQAImpl::run(XQAParams const& xqa_params, KVBlockArray const& kv_block_array, cudaStream_t const& stream)
{
    runWithKVBlockArray(xqa_params, kv_block_array, stream);
}

std::unique_ptr<DecoderXQAImpl> DecoderXQAImpl::create(DecoderXQARunner* runner, ImplType implType)
{
    switch (implType)
    {
    case ImplType::kPrecompiled: return std::make_unique<DecoderXQAImplPrecompiled>(runner);
    case ImplType::kJIT: return std::make_unique<DecoderXQAImplJIT>(runner);
    }
    // Shouldn't reach here.
    TLLM_THROW("Unknown DecoderXQAImpl::ImplType");
}

} // namespace kernels
} // namespace tensorrt_llm
