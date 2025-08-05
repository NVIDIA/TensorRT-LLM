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
#include "tensorrt_llm/common/utils.h"
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
            tensorrt_llm::common::fmtstr("[TensorRT-LLM][ERROR] TllmXqaJit runtime error in %s: %s", func, log.data())
                .c_str());
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
    bool const useQGMMAKernel = supportConfigQGMMA(mXqaParams, mSM, true);
    tllmXqaJitRopeStyle ropeStyle = tllmXqaJitRopeStyle::TLLM_XQA_JIT_ROPE_NONE;
    bool const applyRoPEInXqaKernel = !mXqaParams.multi_query_tokens && useQGMMAKernel
        && tensorrt_llm::common::contains({PositionEmbeddingType::kLONG_ROPE, PositionEmbeddingType::kROPE_GPT_NEOX,
                                              PositionEmbeddingType::kROPE_GPTJ},
            mXqaParams.position_embedding_type);
    if (applyRoPEInXqaKernel)
    {
        TLLM_CHECK(useQGMMAKernel);
        switch (mXqaParams.position_embedding_type)
        {
        case PositionEmbeddingType::kROPE_GPTJ: ropeStyle = tllmXqaJitRopeStyle::TLLM_XQA_JIT_ROPE_GPTJ; break;
        case PositionEmbeddingType::kROPE_GPT_NEOX:
        case PositionEmbeddingType::kLONG_ROPE: ropeStyle = tllmXqaJitRopeStyle::TLLM_XQA_JIT_ROPE_NEOX; break;
        default: TLLM_THROW("TllmXqaJit: Bad RoPE type");
        }
    }
    else
    {
        // Make it explicit that Ampere-style kernel doesn't apply RoPE in the kernel.
        // For kROPE_M, set ropeStyle to TLLM_XQA_JIT_ROPE_NONE to let XQA kernel not apply RoPE.
        // At runtime, a separate kernel (see invokeQKVPreprocessing) will be launched to apply RoPE.
        ropeStyle = tllmXqaJitRopeStyle::TLLM_XQA_JIT_ROPE_NONE;
    }
    if (applyRoPEInXqaKernel)
    {
        TLLM_CHECK(ropeStyle != tllmXqaJitRopeStyle::TLLM_XQA_JIT_ROPE_NONE);
    }
    else
    {
        TLLM_CHECK(ropeStyle == tllmXqaJitRopeStyle::TLLM_XQA_JIT_ROPE_NONE);
    }
    tllmXqaJitContext context{/*sm=*/mSM,
        /*head_size=*/static_cast<uint32_t>(mXqaParams.head_size),
        /*num_q_heads=*/static_cast<uint32_t>(mXqaParams.num_q_heads),
        /*num_kv_heads=*/static_cast<uint32_t>(mXqaParams.num_kv_heads),
        /*beam_width=*/static_cast<uint32_t>(mXqaParams.beam_width),
        /*tokens_per_block=*/static_cast<uint32_t>(mXqaParams.tokens_per_block),
        /*multi_query_tokens=*/mXqaParams.multi_query_tokens,
        /*q_seq_len=*/static_cast<uint32_t>(mXqaParams.generation_input_length),
        /*paged_kv_cache=*/mXqaParams.paged_kv_cache,
        /*data_type=*/static_cast<int>(mXqaParams.data_type),
        /*kv_cache_data_type=*/static_cast<int>(mXqaParams.kv_cache_data_type),
        /*kernel_type=*/mXqaParams.isMLA() ? TLLM_XQA_JIT_MLA
                                           : (useQGMMAKernel ? TLLM_XQA_JIT_QGMMA : TLLM_XQA_JIT_HMMA),
        /*fp8_output=*/mXqaParams.is_fp8_output,
        // If applyRoPEInXqaKernel, no scratch is needed for storing intermediate RoPE result. Use input KV instead of
        // scratch in this case.
        /*use_input_kv=*/applyRoPEInXqaKernel,
        /*rope_style=*/ropeStyle,
        /*is_spec_dec_tree=*/mXqaParams.is_spec_dec_tree};
    if (context.kernel_type == TLLM_XQA_JIT_MLA)
    {
        auto const& c = context;
        TLLM_CHECK(c.head_size == 576 && c.num_q_heads == 128 && c.num_kv_heads == 1 && c.beam_width == 1
            && c.data_type == DATA_TYPE_E4M3 && c.kv_cache_data_type == DATA_TYPE_E4M3 && c.fp8_output == false
            && !c.use_input_kv && ropeStyle == TLLM_XQA_JIT_ROPE_NONE);
    }

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
