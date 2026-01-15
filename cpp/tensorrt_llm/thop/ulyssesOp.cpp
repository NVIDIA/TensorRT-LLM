/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>

namespace tk = tensorrt_llm::kernels;

namespace tensorrt_llm::torch_ext
{

/**
 * @brief Context preprocess transpose for Ulysses attention.
 *
 * Transforms: [partialTokenNum, numHeads, headSize]
 *          -> [cpSize, partialTokenNum, partialHeads, headSize]
 *
 * The data for rank==cpRank goes to dst_my_rank, others go to dst_other_ranks.
 *
 * @param src Input tensor [partialTokenNum, numHeads, headSize]
 * @param dst_other_ranks Output for other ranks [cpSize-1, partialTokenNum, partialHeads, headSize]
 * @param dst_my_rank Output for my rank [partialTokenNum, partialHeads, headSize]
 * @param partial_token_num Number of partial tokens
 * @param cp_size Context parallelism size
 * @param num_q_heads Number of Q attention heads
 * @param num_kv_heads Number of KV attention heads
 * @param mqa_broadcast MQA broadcast factor
 * @param head_size Size per head
 * @param cp_rank Current rank in context parallelism
 */
void ulyssesCpTranspose(torch::Tensor const& src, torch::Tensor& dst_other_ranks, torch::Tensor& dst_my_rank,
    int64_t partial_token_num, int64_t cp_size, int64_t num_q_heads, int64_t num_kv_heads, int64_t mqa_broadcast,
    int64_t head_size, int64_t cp_rank)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto const dtype = src.scalar_type();

    if (dtype == at::ScalarType::Half)
    {
        tk::invokeCpTranspose<half>(static_cast<half*>(dst_other_ranks.data_ptr()),
            static_cast<half*>(dst_my_rank.data_ptr()), static_cast<half const*>(src.data_ptr()), partial_token_num,
            cp_size, num_q_heads, num_kv_heads, mqa_broadcast, head_size, cp_rank, stream);
    }
#ifdef ENABLE_BF16
    else if (dtype == at::ScalarType::BFloat16)
    {
        tk::invokeCpTranspose<__nv_bfloat16>(static_cast<__nv_bfloat16*>(dst_other_ranks.data_ptr()),
            static_cast<__nv_bfloat16*>(dst_my_rank.data_ptr()), static_cast<__nv_bfloat16 const*>(src.data_ptr()),
            partial_token_num, cp_size, num_q_heads, num_kv_heads, mqa_broadcast, head_size, cp_rank, stream);
    }
#endif
    else if (dtype == at::ScalarType::Float)
    {
        tk::invokeCpTranspose<float>(static_cast<float*>(dst_other_ranks.data_ptr()),
            static_cast<float*>(dst_my_rank.data_ptr()), static_cast<float const*>(src.data_ptr()), partial_token_num,
            cp_size, num_q_heads, num_kv_heads, mqa_broadcast, head_size, cp_rank, stream);
    }
    else
    {
        TLLM_THROW("Unsupported dtype for ulyssesCpTranspose");
    }
}

/**
 * @brief Context preprocess transpose2 for Ulysses attention (after all-to-all).
 *
 * Transforms: [cpSize, partialTokenNum, partialHeads, headSize] (scattered layout)
 *          -> [numTokens, partialHeads, headSize] (sequence major)
 *
 * Handles variable sequence lengths with cu_q_seqlens and cu_cp_partial_seqlens.
 *
 * @param src Input tensor after all-to-all
 * @param dst Output tensor in sequence major layout
 * @param q_seq_lengths Sequence lengths per batch [batch_size]
 * @param cu_q_seqlens Cumulative sequence lengths [batch_size + 1]
 * @param cu_cp_partial_seqlens Cumulative partial sequence lengths [batch_size + 1]
 * @param cp_size Context parallelism size
 * @param max_partial_length Maximum partial sequence length
 * @param batch_size Batch size
 * @param partial_heads Number of partial heads (Q + 2*KV)
 * @param head_size Size per head
 */
void ulyssesCpTranspose2(torch::Tensor const& src, torch::Tensor& dst, torch::Tensor const& q_seq_lengths,
    torch::Tensor const& cu_q_seqlens, torch::Tensor const& cu_cp_partial_seqlens, int64_t cp_size,
    int64_t max_partial_length, int64_t batch_size, int64_t partial_heads, int64_t head_size)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto const dtype = src.scalar_type();

    if (dtype == at::ScalarType::Half)
    {
        tk::invokeCpTranspose2<half>(static_cast<half*>(dst.data_ptr()), static_cast<half const*>(src.data_ptr()),
            q_seq_lengths.data_ptr<int32_t>(), cu_q_seqlens.data_ptr<int32_t>(),
            cu_cp_partial_seqlens.data_ptr<int32_t>(), cp_size, max_partial_length, batch_size, partial_heads,
            head_size, stream);
    }
#ifdef ENABLE_BF16
    else if (dtype == at::ScalarType::BFloat16)
    {
        tk::invokeCpTranspose2<__nv_bfloat16>(static_cast<__nv_bfloat16*>(dst.data_ptr()),
            static_cast<__nv_bfloat16 const*>(src.data_ptr()), q_seq_lengths.data_ptr<int32_t>(),
            cu_q_seqlens.data_ptr<int32_t>(), cu_cp_partial_seqlens.data_ptr<int32_t>(), cp_size, max_partial_length,
            batch_size, partial_heads, head_size, stream);
    }
#endif
    else if (dtype == at::ScalarType::Float)
    {
        tk::invokeCpTranspose2<float>(static_cast<float*>(dst.data_ptr()), static_cast<float const*>(src.data_ptr()),
            q_seq_lengths.data_ptr<int32_t>(), cu_q_seqlens.data_ptr<int32_t>(),
            cu_cp_partial_seqlens.data_ptr<int32_t>(), cp_size, max_partial_length, batch_size, partial_heads,
            head_size, stream);
    }
    else
    {
        TLLM_THROW("Unsupported dtype for ulyssesCpTranspose2");
    }
}

/**
 * @brief Context postprocess transpose for Ulysses attention (before all-to-all).
 *
 * Transforms: [numTokens, partialHeads, headSize] (sequence major)
 *          -> [cpSize, partialTokens, partialHeads, headSize]
 *
 * @param src Input tensor in sequence major layout
 * @param dst Output tensor for all-to-all
 * @param q_seq_lengths Sequence lengths per batch [batch_size]
 * @param cu_q_seqlens Cumulative sequence lengths [batch_size + 1]
 * @param cu_cp_partial_seqlens Cumulative partial sequence lengths [batch_size + 1]
 * @param cp_size Context parallelism size
 * @param max_partial_length Maximum partial sequence length
 * @param batch_size Batch size
 * @param partial_heads Number of partial heads
 * @param head_size Size per head
 */
void ulyssesCpTransposeToSeqMajor2(torch::Tensor const& src, torch::Tensor& dst, torch::Tensor const& q_seq_lengths,
    torch::Tensor const& cu_q_seqlens, torch::Tensor const& cu_cp_partial_seqlens, int64_t cp_size,
    int64_t max_partial_length, int64_t batch_size, int64_t partial_heads, int64_t head_size)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto const dtype = src.scalar_type();

    if (dtype == at::ScalarType::Half)
    {
        tk::invokeCpTransposeToSeqMajor2<half>(static_cast<half*>(dst.data_ptr()),
            static_cast<half const*>(src.data_ptr()), q_seq_lengths.data_ptr<int32_t>(),
            cu_q_seqlens.data_ptr<int32_t>(), cu_cp_partial_seqlens.data_ptr<int32_t>(), cp_size, max_partial_length,
            batch_size, partial_heads, head_size, stream);
    }
#ifdef ENABLE_BF16
    else if (dtype == at::ScalarType::BFloat16)
    {
        tk::invokeCpTransposeToSeqMajor2<__nv_bfloat16>(static_cast<__nv_bfloat16*>(dst.data_ptr()),
            static_cast<__nv_bfloat16 const*>(src.data_ptr()), q_seq_lengths.data_ptr<int32_t>(),
            cu_q_seqlens.data_ptr<int32_t>(), cu_cp_partial_seqlens.data_ptr<int32_t>(), cp_size, max_partial_length,
            batch_size, partial_heads, head_size, stream);
    }
#endif
    else if (dtype == at::ScalarType::Float)
    {
        tk::invokeCpTransposeToSeqMajor2<float>(static_cast<float*>(dst.data_ptr()),
            static_cast<float const*>(src.data_ptr()), q_seq_lengths.data_ptr<int32_t>(),
            cu_q_seqlens.data_ptr<int32_t>(), cu_cp_partial_seqlens.data_ptr<int32_t>(), cp_size, max_partial_length,
            batch_size, partial_heads, head_size, stream);
    }
    else if (dtype == at::ScalarType::Float8_e4m3fn)
    {
        tk::invokeCpTransposeToSeqMajor2<__nv_fp8_e4m3>(static_cast<__nv_fp8_e4m3*>(dst.data_ptr()),
            static_cast<__nv_fp8_e4m3 const*>(src.data_ptr()), q_seq_lengths.data_ptr<int32_t>(),
            cu_q_seqlens.data_ptr<int32_t>(), cu_cp_partial_seqlens.data_ptr<int32_t>(), cp_size, max_partial_length,
            batch_size, partial_heads, head_size, stream);
    }
    else
    {
        TLLM_THROW("Unsupported dtype for ulyssesCpTransposeToSeqMajor2");
    }
}

/**
 * @brief Context/Generation postprocess transpose for Ulysses attention (after all-to-all).
 *
 * Transforms: [cpSize, partialTokens, partialHeads, headSize]
 *          -> [partialTokens, numHeads, headSize]
 *
 * Combines data from my_rank and other_ranks after all-to-all.
 *
 * @param dst Output tensor [partialTokens, numHeads, headSize]
 * @param src_my_rank Input from my rank [partialTokens, partialHeads, headSize]
 * @param src_other_ranks Input from other ranks [cpSize-1, partialTokens, partialHeads, headSize]
 * @param partial_length Number of partial tokens
 * @param cp_size Context parallelism size
 * @param partial_heads Number of partial heads (numHeads / cpSize)
 * @param head_size Size per head
 * @param cp_rank Current rank in context parallelism
 */
void ulyssesCpTransposeToSeqMajor(torch::Tensor& dst, torch::Tensor const& src_my_rank,
    torch::Tensor const& src_other_ranks, int64_t partial_length, int64_t cp_size, int64_t partial_heads,
    int64_t head_size, int64_t cp_rank)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto const dtype = dst.scalar_type();

    if (dtype == at::ScalarType::Half)
    {
        tk::invokeCpTransposeToSeqMajor<half>(static_cast<half*>(dst.data_ptr()),
            static_cast<half const*>(src_my_rank.data_ptr()), static_cast<half const*>(src_other_ranks.data_ptr()),
            partial_length, cp_size, partial_heads, head_size, cp_rank, stream);
    }
#ifdef ENABLE_BF16
    else if (dtype == at::ScalarType::BFloat16)
    {
        tk::invokeCpTransposeToSeqMajor<__nv_bfloat16>(static_cast<__nv_bfloat16*>(dst.data_ptr()),
            static_cast<__nv_bfloat16 const*>(src_my_rank.data_ptr()),
            static_cast<__nv_bfloat16 const*>(src_other_ranks.data_ptr()), partial_length, cp_size, partial_heads,
            head_size, cp_rank, stream);
    }
#endif
    else if (dtype == at::ScalarType::Float)
    {
        tk::invokeCpTransposeToSeqMajor<float>(static_cast<float*>(dst.data_ptr()),
            static_cast<float const*>(src_my_rank.data_ptr()), static_cast<float const*>(src_other_ranks.data_ptr()),
            partial_length, cp_size, partial_heads, head_size, cp_rank, stream);
    }
    else if (dtype == at::ScalarType::Float8_e4m3fn)
    {
        tk::invokeCpTransposeToSeqMajor<__nv_fp8_e4m3>(static_cast<__nv_fp8_e4m3*>(dst.data_ptr()),
            static_cast<__nv_fp8_e4m3 const*>(src_my_rank.data_ptr()),
            static_cast<__nv_fp8_e4m3 const*>(src_other_ranks.data_ptr()), partial_length, cp_size, partial_heads,
            head_size, cp_rank, stream);
    }
    else
    {
        TLLM_THROW("Unsupported dtype for ulyssesCpTransposeToSeqMajor");
    }
}

} // namespace tensorrt_llm::torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "ulysses_cp_transpose("
        "Tensor src"
        ", Tensor(a!) dst_other_ranks"
        ", Tensor(b!) dst_my_rank"
        ", int partial_token_num"
        ", int cp_size"
        ", int num_q_heads"
        ", int num_kv_heads"
        ", int mqa_broadcast"
        ", int head_size"
        ", int cp_rank"
        ") -> ()",
        &tensorrt_llm::torch_ext::ulyssesCpTranspose);

    m.def(
        "ulysses_cp_transpose2("
        "Tensor src"
        ", Tensor(a!) dst"
        ", Tensor q_seq_lengths"
        ", Tensor cu_q_seqlens"
        ", Tensor cu_cp_partial_seqlens"
        ", int cp_size"
        ", int max_partial_length"
        ", int batch_size"
        ", int partial_heads"
        ", int head_size"
        ") -> ()",
        &tensorrt_llm::torch_ext::ulyssesCpTranspose2);

    m.def(
        "ulysses_cp_transpose_to_seq_major2("
        "Tensor src"
        ", Tensor(a!) dst"
        ", Tensor q_seq_lengths"
        ", Tensor cu_q_seqlens"
        ", Tensor cu_cp_partial_seqlens"
        ", int cp_size"
        ", int max_partial_length"
        ", int batch_size"
        ", int partial_heads"
        ", int head_size"
        ") -> ()",
        &tensorrt_llm::torch_ext::ulyssesCpTransposeToSeqMajor2);

    m.def(
        "ulysses_cp_transpose_to_seq_major("
        "Tensor(a!) dst"
        ", Tensor src_my_rank"
        ", Tensor src_other_ranks"
        ", int partial_length"
        ", int cp_size"
        ", int partial_heads"
        ", int head_size"
        ", int cp_rank"
        ") -> ()",
        &tensorrt_llm::torch_ext::ulyssesCpTransposeToSeqMajor);
}
