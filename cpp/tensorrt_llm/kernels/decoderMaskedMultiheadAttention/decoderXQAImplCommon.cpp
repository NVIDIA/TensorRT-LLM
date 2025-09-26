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
 *
 * Common utils to be shared between Precompiled and JIT implementation.
 */
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplCommon.h"

namespace tensorrt_llm::kernels
{

uint32_t getKernelMTileSize(
    uint32_t headGrpSize, bool isSpecDec, uint32_t qSeqLen, bool isXqaJit, bool supportQGMMA, bool supportMLA)
{
    if (!isSpecDec)
    {
        return headGrpSize;
    }
    if (isXqaJit && (supportQGMMA || supportMLA)) // HMMA (mha.cu) goes to the heuristic below
    {
        return 64;
    }
    uint32_t const gemmM = qSeqLen * headGrpSize;
    return gemmM < 16 ? 16 : 32;
}

XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParams(XQAParams const& xqaParams, bool isXqaJit, int SM)
{
    unsigned int head_size = xqaParams.head_size;
    unsigned int num_q_heads = xqaParams.num_q_heads;
    unsigned int num_kv_heads = xqaParams.num_kv_heads;
    TLLM_CHECK_WITH_INFO(num_q_heads % num_kv_heads == 0, "numQHeads should be multiple of numKVHeads.");
    unsigned int num_q_heads_over_kv = num_q_heads / num_kv_heads;
    unsigned int beam_width = xqaParams.beam_width;

    unsigned int qSeqLen = static_cast<unsigned int>(xqaParams.generation_input_length);
    // MultiQueryToken kernels can support any num_q_heads_over_kv that is power of 2.
    unsigned int kernel_num_q_heads_over_kv = xqaParams.multi_query_tokens ? 0 : num_q_heads_over_kv;
    bool supportQGMMA = jit::supportConfigQGMMA(xqaParams, SM, true);
    bool supportMLA = jit::supportConfigMLA(xqaParams, SM, true);
    unsigned int kernel_m_tilesize = getKernelMTileSize(
        num_q_heads_over_kv, xqaParams.multi_query_tokens, qSeqLen, isXqaJit, supportQGMMA, supportMLA);

    // precompiled XQA does not use is_fp8_output as hashing key
    return {xqaParams.kv_cache_data_type, head_size, beam_width, kernel_num_q_heads_over_kv, kernel_m_tilesize,
        xqaParams.paged_kv_cache ? static_cast<unsigned int>(xqaParams.tokens_per_block) : 0, xqaParams.paged_kv_cache,
        xqaParams.multi_query_tokens, isXqaJit ? xqaParams.is_fp8_output : false,
        isXqaJit ? std::optional(xqaParams.position_embedding_type) : std::nullopt};
}

} // namespace tensorrt_llm::kernels
