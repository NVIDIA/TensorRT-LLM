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
#include <memory>

#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/xqaParams.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"

namespace tensorrt_llm
{
namespace kernels
{

// Forward declaration to avoid cyclic dependency.
class DecoderXQARunner;

/**
 * The underlying XQA implementation called from DecoderXQARunner.
 *
 * We need this layer of abstraction for abstracting out implementation details. Two possible implementations:
 *   1. Precompiled, i.e. kernels are compiled and saved as cubins in advance.
 *   2. JIT, i.e. kernels are compiled on the fly via NVRTC.
 *
 * This class is written as Composition over Inheritance, primarily because C++ does not support virtual template
 * functions.
 */
class DecoderXQAImpl
{
public:
    // Whether it is beneficial to use this XQA codepath.
    virtual bool shouldUse(const XQAParams& xqaParams) = 0;
    // Prepares for the kernel running. Must be called before calling run.
    virtual void prepare(const XQAParams& xqa_params) = 0;
    // Run XQA kernel with KVCacheBuffer.
    //
    // Sub-classes should implement runWithKVLinearBuffer and runWithKVBlockArray.
    template <typename KVCacheBuffer>
    void run(const XQAParams& xqa_params, KVCacheBuffer& kv_cache_buffer, int2& rotary_kernel_launch_cache,
        const cudaStream_t& stream);

    enum class ImplType
    {
        kPrecompiled = 0,
        kJIT = 1,
    };
    // Needs runner pointer for accessing resources in DecoderXQARunner class.
    static std::unique_ptr<DecoderXQAImpl> create(DecoderXQARunner* runner, ImplType implType);

protected:
    DecoderXQAImpl(DecoderXQARunner* runner)
        : mRunner(runner)
    {
    }

    virtual void runWithKVLinearBuffer(const XQAParams& xqa_params, KVLinearBuffer& kv_linear_buffer,
        int2& rotary_kernel_launch_cache, const cudaStream_t& stream)
        = 0;
    virtual void runWithKVBlockArray(const XQAParams& xqa_params, KVBlockArray& kv_block_array,
        int2& rotary_kernel_launch_cache, const cudaStream_t& stream)
        = 0;

    DecoderXQARunner* mRunner;
};

} // namespace kernels
} // namespace tensorrt_llm
