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
#include "compileEngine.h"

#include "cubinObj.h"
#include "nvrtcWrapper/include/nvrtcWrapper.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/kernelUtils.h"
#include <string>
#include <vector>

namespace
{

void CHECK_TLLM_XQA_JIT_ERROR_(tllmXqaJitStatus result, char const* const func, char const* const file, int const line)
{
    if (result != TLLM_XQA_JIT_SUCCESS)
    {
        std::vector<char> log(tllmXqaJitGetLastErrorStringSize());
        tllmXqaJitGetLastErrorString(log.data());
        throw tensorrt_llm::common::TllmException(file, line,
            tensorrt_llm::common::fmtstr("[TensorRT-LLM][ERROR] TllmXqaJit runtime error in %s: %s", func, log.data()));
    }
}

#define CHECK_TLLM_XQA_JIT_ERROR(val) CHECK_TLLM_XQA_JIT_ERROR_((val), #val, __FILE__, __LINE__)

} // anonymous namespace

namespace tensorrt_llm
{
namespace kernels
{
namespace jit
{

CubinObj CompileEngine::compile() const
{
    tllmXqaJitProgram program;
    bool useQGMMAKernel = supportConfigQGMMA(mXqaParams, mSM, true);
    tllmXqaJitContext context{/*sm=*/mSM,
        /*head_size=*/static_cast<uint32_t>(mXqaParams.head_size),
        /*num_q_heads=*/static_cast<uint32_t>(mXqaParams.num_q_heads),
        /*num_kv_heads=*/static_cast<uint32_t>(mXqaParams.num_kv_heads),
        /*beam_width=*/static_cast<uint32_t>(mXqaParams.beam_width),
        /*tokens_per_block=*/static_cast<uint32_t>(mXqaParams.tokens_per_block),
        /*multi_query_tokens=*/mXqaParams.multi_query_tokens,
        /*paged_kv_cache=*/mXqaParams.paged_kv_cache,
        /*data_type=*/static_cast<int>(mXqaParams.data_type),
        /*kv_cache_data_type=*/static_cast<int>(mXqaParams.kv_cache_data_type),
        /*kernel_type=*/useQGMMAKernel ? TLLM_XQA_JIT_QGMMA : TLLM_XQA_JIT_HMMA};

    CHECK_TLLM_XQA_JIT_ERROR(tllmXqaJitCreateAndCompileProgram(&program, &context));

    size_t cubinSize;
    CHECK_TLLM_XQA_JIT_ERROR(tllmXqaJitGetCUBINSize(program, &cubinSize));
    std::string cubinContent(cubinSize, ' ');
    CHECK_TLLM_XQA_JIT_ERROR(tllmXqaJitGetCUBIN(program, const_cast<char*>(cubinContent.c_str())));

    CHECK_TLLM_XQA_JIT_ERROR(tllmXqaJitDestroyProgram(&program));

    return CubinObj(cubinContent);
}

CompileEngine::CompileEngine(int SM, XQAParams const& xqaParams)
    : mSM(SM)
    , mXqaParams(xqaParams)
{
}

} // namespace jit
} // namespace kernels
} // namespace tensorrt_llm
