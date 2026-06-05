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
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAConstants.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/kernelUtils.h"
#include <chrono>
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

TRTLLM_NAMESPACE_BEGIN

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
    bool const isMlaKernel = mXqaParams.isMLA();
    uint32_t const numQHeadsOverKv = static_cast<uint32_t>(mXqaParams.num_q_heads / mXqaParams.num_kv_heads);
    uint32_t const kernelMlaHeadGrpSize = getXqaMlaRuntimeKernelHeadGrpSize(numQHeadsOverKv);
    // For MLA, the v-head dim differs from head_size. mlaHeadSizeV() returns 512 for DSV3 and 448
    // for DSV4. For non-MLA kernels we pass 0 and the JIT wrapper falls back to head_size.
    uint32_t const headSizeV = isMlaKernel ? static_cast<uint32_t>(mXqaParams.mlaHeadSizeV()) : 0;
    tllmXqaJitContext context{/*sm=*/mSM,
        /*head_size=*/static_cast<uint32_t>(mXqaParams.head_size),
        /*head_size_v=*/headSizeV,
        /*num_q_heads=*/static_cast<uint32_t>(isMlaKernel ? kernelMlaHeadGrpSize : mXqaParams.num_q_heads),
        /*num_kv_heads=*/static_cast<uint32_t>(isMlaKernel ? 1 : mXqaParams.num_kv_heads),
        /*beam_width=*/static_cast<uint32_t>(mXqaParams.beam_width),
        /*tokens_per_block=*/static_cast<uint32_t>(mXqaParams.tokens_per_block),
        /*multi_query_tokens=*/mXqaParams.multi_query_tokens,
        /*q_seq_len=*/static_cast<uint32_t>(mXqaParams.generation_input_length),
        /*paged_kv_cache=*/mXqaParams.paged_kv_cache,
        /*data_type=*/static_cast<int>(mXqaParams.data_type),
        /*kv_cache_data_type=*/static_cast<int>(mXqaParams.kv_cache_data_type),
        /*kernel_type=*/isMlaKernel ? TLLM_XQA_JIT_MLA : (useQGMMAKernel ? TLLM_XQA_JIT_QGMMA : TLLM_XQA_JIT_HMMA),
        /*fp8_output=*/mXqaParams.is_fp8_output,
        // If applyRoPEInXqaKernel, no scratch is needed for storing intermediate RoPE result. Use input KV instead of
        // scratch in this case.
        /*use_input_kv=*/applyRoPEInXqaKernel,
        /*rope_style=*/ropeStyle,
        /*is_spec_dec_tree=*/mXqaParams.is_spec_dec_tree,
        /*use_skip_softmax_attn=*/mXqaParams.skip_softmax_threshold_scale_factor != 0};
    if (context.kernel_type == TLLM_XQA_JIT_MLA)
    {
        auto const& c = context;
        // DSV3: head_size=576, head_size_v=512.   DSV4: head_size=512, head_size_v=448.
        bool const isDsv3Shape = (c.head_size == 576 && c.head_size_v == 512);
        bool const isDsv4Shape = (c.head_size == 512 && c.head_size_v == 448);
        TLLM_CHECK((isDsv3Shape || isDsv4Shape) && (c.num_q_heads == 32 || c.num_q_heads == 64 || c.num_q_heads == 128)
            && c.num_kv_heads == 1
            && c.beam_width == 1 && c.data_type == DATA_TYPE_E4M3 && c.kv_cache_data_type == DATA_TYPE_E4M3
            && c.fp8_output == false && !c.use_input_kv && ropeStyle == TLLM_XQA_JIT_ROPE_NONE);
    }

    auto const compileStart = std::chrono::steady_clock::now();
    TLLM_LOG_DEBUG(
        "Compiling JIT XQA cubin: sm=%d kernel_type=%d head_size=%u runtime_head_grp=%u kernel_head_grp=%u "
        "tokens_per_block=%u q_seq_len=%u beam_width=%u",
        mSM, static_cast<int>(context.kernel_type), context.head_size, numQHeadsOverKv, context.num_q_heads,
        context.tokens_per_block, context.q_seq_len, context.beam_width);
    CHECK_TLLM_XQA_JIT_ERROR(tllmXqaJitCreateAndCompileProgram(&program, &context));

    size_t cubinSize;
    CHECK_TLLM_XQA_JIT_ERROR(tllmXqaJitGetCUBINSize(program, &cubinSize));
    std::string cubinContent(cubinSize, ' ');
    CHECK_TLLM_XQA_JIT_ERROR(tllmXqaJitGetCUBIN(program, const_cast<char*>(cubinContent.c_str())));

    CHECK_TLLM_XQA_JIT_ERROR(tllmXqaJitDestroyProgram(&program));
    auto const compileEnd = std::chrono::steady_clock::now();
    auto const compileMs = std::chrono::duration_cast<std::chrono::milliseconds>(compileEnd - compileStart).count();
    TLLM_LOG_DEBUG(
        "Compiled JIT XQA cubin: sm=%d kernel_type=%d kernel_head_grp=%u cubin_size=%zu compile_ms=%lld",
        mSM, static_cast<int>(context.kernel_type), context.num_q_heads, cubinSize, static_cast<long long>(compileMs));

    return CubinObj(cubinContent);
}

CompileEngine::CompileEngine(int SM, XQAParams const& xqaParams)
    : mSM(SM)
    , mXqaParams(xqaParams)
{
}

} // namespace jit
} // namespace kernels

TRTLLM_NAMESPACE_END
