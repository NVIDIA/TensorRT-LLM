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
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/kernelUtils.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace jit
{

namespace
{

template <typename T>
bool contains(std::initializer_list<T> const& c, T const& v)
{
    return std::find(c.begin(), c.end(), v) != c.end();
}

bool supportConfigCommon(XQAParams const& xqaParams, bool forConfigurePlugin)
{
    if (xqaParams.unidirectional != 1)
    {
        return false;
    }
    if (xqaParams.q_scaling != 1.0f)
    {
        return false;
    }
    if (xqaParams.mask_type != tensorrt_llm::kernels::AttentionMaskType::CAUSAL)
    {
        return false;
    }
    if (xqaParams.cross_attention)
    {
        return false;
    }
    if (xqaParams.cyclic_attention_window_size != xqaParams.max_attention_window_size)
    {
        return false;
    }
    if (xqaParams.position_shift_enabled || xqaParams.sink_token_length > 0)
    {
        return false;
    }
    if (xqaParams.num_kv_heads != 0 && xqaParams.num_q_heads % xqaParams.num_kv_heads != 0)
    {
        return false;
    }
    bool is_vanilla_mha = xqaParams.num_kv_heads == 0 || xqaParams.num_q_heads == xqaParams.num_kv_heads;
    if (is_vanilla_mha && xqaParams.beam_width == 1)
    {
        // Do not use XQA kernel for vanilla MHA case for performance reasons.
        return false;
    }
    if (is_vanilla_mha && xqaParams.head_size <= 128)
    {
        // TODO(yaoy): remove this when the kernel bug for num_kv_heads <= 128 gets fixed.
        return false;
    }
    if (!contains({PositionEmbeddingType::kROPE_GPTJ, PositionEmbeddingType::kROPE_GPT_NEOX,
                      PositionEmbeddingType::kLONG_ROPE},
            xqaParams.position_embedding_type))
    {
        return false;
    }
    if (!forConfigurePlugin)
    {
        // Inference time checks.
        if (xqaParams.host_past_key_value_lengths == nullptr)
        {
            return false;
        }
        for (int i = 0; i < xqaParams.batch_size; ++i)
        {
            // Only checks for non-medusa case, because medusa may not accept all tokens in host_past_key_value_lengths.
            if (!xqaParams.multi_query_tokens
                && xqaParams.host_past_key_value_lengths[i] + 1 > xqaParams.max_attention_window_size)
            {
                return false;
            }
        }
    }
    return true;
}

} // anonymous namespace

bool supportConfigQGMMA(XQAParams const& xqaParams, int SM, bool forConfigurePlugin)
{
    if (!supportConfigCommon(xqaParams, forConfigurePlugin))
    {
        return false;
    }
    if (SM != kSM_90)
    {
        return false;
    }
    if (!contains({DATA_TYPE_FP16, DATA_TYPE_BF16}, xqaParams.data_type))
    {
        return false;
    }
    if (xqaParams.kv_cache_data_type != DATA_TYPE_E4M3)
    {
        return false;
    }
    if (xqaParams.beam_width != 1)
    {
        return false;
    }
    if (xqaParams.head_size % 16 != 0 || xqaParams.head_size < 16 || xqaParams.head_size > 256)
    {
        return false;
    }
    int32_t head_grp_size = xqaParams.num_kv_heads == 0 ? 1 : xqaParams.num_q_heads / xqaParams.num_kv_heads;
    if (head_grp_size * xqaParams.beam_width > 32)
    {
        return false;
    }
    if (xqaParams.paged_kv_cache && !contains({16, 32, 64, 128}, xqaParams.tokens_per_block))
    {
        return false;
    }
    return true;
}

bool supportConfigHMMA(XQAParams const& xqaParams, int SM, bool forConfigurePlugin)
{
    if (!supportConfigCommon(xqaParams, forConfigurePlugin))
    {
        return false;
    }
    if (SM < kSM_80)
    {
        return false;
    }
    if (!contains({DATA_TYPE_FP16, DATA_TYPE_BF16}, xqaParams.data_type))
    {
        return false;
    }
    if (!contains({DATA_TYPE_FP16, DATA_TYPE_BF16, DATA_TYPE_INT8, DATA_TYPE_E4M3}, xqaParams.kv_cache_data_type))
    {
        return false;
    }
    if (xqaParams.beam_width != 1 && xqaParams.beam_width != 4)
    {
        return false;
    }
    if (xqaParams.head_size % 16 != 0 || xqaParams.head_size < 16 || xqaParams.head_size > 256)
    {
        return false;
    }
    int32_t head_grp_size = xqaParams.num_kv_heads == 0 ? 1 : xqaParams.num_q_heads / xqaParams.num_kv_heads;
    if (head_grp_size * xqaParams.beam_width > 32)
    {
        return false;
    }
    if (xqaParams.paged_kv_cache && !contains({16, 32, 64, 128}, xqaParams.tokens_per_block))
    {
        return false;
    }
    return true;
}

} // namespace jit
} // namespace kernels
} // namespace tensorrt_llm
