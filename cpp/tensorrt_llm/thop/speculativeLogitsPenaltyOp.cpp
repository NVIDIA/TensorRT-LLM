/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kernels/speculativeDecoding/logitsPenaltyKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

template <typename T, typename TokenT>
void invokeTypedApplyTokenPenalties(th::Tensor& logits, th::Tensor const& tokenIds, th::Tensor const& penaltyValues)
{
    auto stream = at::cuda::getCurrentCUDAStream(logits.get_device()).stream();
    tk::invokeApplySpeculativeTokenPenalties<T, TokenT>(reinterpret_cast<T*>(logits.data_ptr()),
        reinterpret_cast<TokenT const*>(tokenIds.data_ptr()), reinterpret_cast<float const*>(penaltyValues.data_ptr()),
        static_cast<int32_t>(logits.size(0)), static_cast<int32_t>(tokenIds.size(1)),
        static_cast<int32_t>(logits.size(1)), stream);
}

template <typename T>
void dispatchTokenType(th::Tensor& logits, th::Tensor const& tokenIds, th::Tensor const& penaltyValues)
{
    switch (tokenIds.scalar_type())
    {
    case torch::kInt:
        invokeTypedApplyTokenPenalties<T, int32_t>(logits, tokenIds, penaltyValues);
        break;
    case torch::kLong:
        invokeTypedApplyTokenPenalties<T, int64_t>(logits, tokenIds, penaltyValues);
        break;
    default:
        TORCH_CHECK(false, "token_ids dtype must be int32 or int64.");
    }
}

template <typename T>
void invokeTypedApplyCountFrequencyPenalty(
    th::Tensor& logits, th::Tensor const& tokenCounts, th::Tensor const& rowSlots, th::Tensor const& frequencyPenalties)
{
    auto stream = at::cuda::getCurrentCUDAStream(logits.get_device()).stream();
    tk::invokeApplySpeculativeCountFrequencyPenalty<T>(reinterpret_cast<T*>(logits.data_ptr()),
        reinterpret_cast<int32_t const*>(tokenCounts.data_ptr()), reinterpret_cast<int32_t const*>(rowSlots.data_ptr()),
        reinterpret_cast<float const*>(frequencyPenalties.data_ptr()), static_cast<int32_t>(logits.size(0)),
        static_cast<int32_t>(logits.size(1)), stream);
}

template <typename T>
void invokeTypedApplySparseCountFrequencyPenalty(th::Tensor& logits, th::Tensor const& tokenIds,
    th::Tensor const& tokenCounts, th::Tensor const& countLens, th::Tensor const& rowSlots,
    th::Tensor const& frequencyPenalties)
{
    auto stream = at::cuda::getCurrentCUDAStream(logits.get_device()).stream();
    tk::invokeApplySpeculativeSparseCountFrequencyPenalty<T>(reinterpret_cast<T*>(logits.data_ptr()),
        reinterpret_cast<int32_t const*>(tokenIds.data_ptr()), reinterpret_cast<int32_t const*>(tokenCounts.data_ptr()),
        reinterpret_cast<int32_t const*>(countLens.data_ptr()), reinterpret_cast<int32_t const*>(rowSlots.data_ptr()),
        reinterpret_cast<float const*>(frequencyPenalties.data_ptr()), static_cast<int32_t>(logits.size(0)),
        static_cast<int32_t>(tokenIds.size(1)), static_cast<int32_t>(logits.size(1)), stream);
}

} // namespace

void speculativeApplyTokenPenalties(th::Tensor& logits, th::Tensor const& tokenIds, th::Tensor const& penaltyValues)
{
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor.");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous.");
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor.");
    TORCH_CHECK(tokenIds.is_cuda(), "token_ids must be a CUDA tensor.");
    TORCH_CHECK(tokenIds.is_contiguous(), "token_ids must be contiguous.");
    TORCH_CHECK(tokenIds.dim() == 2, "token_ids must be a 2D tensor.");
    TORCH_CHECK(penaltyValues.is_cuda(), "penalty_values must be a CUDA tensor.");
    TORCH_CHECK(penaltyValues.is_contiguous(), "penalty_values must be contiguous.");
    TORCH_CHECK(penaltyValues.dim() == 2, "penalty_values must be a 2D tensor.");
    TORCH_CHECK(penaltyValues.scalar_type() == torch::kFloat, "penalty_values dtype must be float32.");
    TORCH_CHECK(tokenIds.size(0) == logits.size(0), "token_ids and logits must have the same row count.");
    TORCH_CHECK(penaltyValues.size(0) == logits.size(0), "penalty_values and logits must have the same row count.");
    TORCH_CHECK(penaltyValues.size(1) == tokenIds.size(1), "penalty_values and token_ids widths must match.");

    if (logits.size(0) == 0 || tokenIds.size(1) == 0)
    {
        return;
    }

    switch (logits.scalar_type())
    {
    case torch::kFloat32:
        dispatchTokenType<float>(logits, tokenIds, penaltyValues);
        break;
    case torch::kFloat16:
        dispatchTokenType<half>(logits, tokenIds, penaltyValues);
        break;
    case torch::kBFloat16:
        dispatchTokenType<__nv_bfloat16>(logits, tokenIds, penaltyValues);
        break;
    default:
        TORCH_CHECK(false, "logits dtype must be float32, float16, or bfloat16.");
    }
}

void speculativeApplyHistoryFrequencyPenalty(th::Tensor& logits, th::Tensor const& historyTokens,
    th::Tensor const& historyLens, th::Tensor const& rowSlots, th::Tensor const& frequencyPenalties)
{
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor.");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous.");
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor.");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits dtype must be float32.");
    TORCH_CHECK(historyTokens.is_cuda(), "history_tokens must be a CUDA tensor.");
    TORCH_CHECK(historyTokens.is_contiguous(), "history_tokens must be contiguous.");
    TORCH_CHECK(historyTokens.dim() == 2, "history_tokens must be a 2D tensor.");
    TORCH_CHECK(historyTokens.scalar_type() == torch::kInt, "history_tokens dtype must be int32.");
    TORCH_CHECK(historyLens.is_cuda(), "history_lens must be a CUDA tensor.");
    TORCH_CHECK(historyLens.is_contiguous(), "history_lens must be contiguous.");
    TORCH_CHECK(historyLens.dim() == 1, "history_lens must be a 1D tensor.");
    TORCH_CHECK(historyLens.scalar_type() == torch::kInt, "history_lens dtype must be int32.");
    TORCH_CHECK(rowSlots.is_cuda(), "row_slots must be a CUDA tensor.");
    TORCH_CHECK(rowSlots.is_contiguous(), "row_slots must be contiguous.");
    TORCH_CHECK(rowSlots.dim() == 1, "row_slots must be a 1D tensor.");
    TORCH_CHECK(rowSlots.scalar_type() == torch::kInt, "row_slots dtype must be int32.");
    TORCH_CHECK(frequencyPenalties.is_cuda(), "frequency_penalties must be a CUDA tensor.");
    TORCH_CHECK(frequencyPenalties.is_contiguous(), "frequency_penalties must be contiguous.");
    TORCH_CHECK(frequencyPenalties.dim() == 1, "frequency_penalties must be a 1D tensor.");
    TORCH_CHECK(frequencyPenalties.scalar_type() == torch::kFloat, "frequency_penalties dtype must be float32.");
    TORCH_CHECK(rowSlots.size(0) == logits.size(0), "row_slots and logits must have the same row count.");
    TORCH_CHECK(
        frequencyPenalties.size(0) == logits.size(0), "frequency_penalties and logits must have the same row count.");
    TORCH_CHECK(historyLens.size(0) == historyTokens.size(0), "history_lens and history_tokens slot count mismatch.");

    if (logits.size(0) == 0 || historyTokens.size(1) == 0)
    {
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream(logits.get_device()).stream();
    tk::invokeApplySpeculativeHistoryFrequencyPenalty(reinterpret_cast<float*>(logits.data_ptr()),
        reinterpret_cast<int32_t const*>(historyTokens.data_ptr()), reinterpret_cast<int32_t const*>(historyLens.data_ptr()),
        reinterpret_cast<int32_t const*>(rowSlots.data_ptr()),
        reinterpret_cast<float const*>(frequencyPenalties.data_ptr()), static_cast<int32_t>(logits.size(0)),
        static_cast<int32_t>(historyTokens.size(1)), static_cast<int32_t>(logits.size(1)), stream);
}

void speculativeAppendAcceptedTokens(th::Tensor& historyTokens, th::Tensor& historyLens, th::Tensor const& seqSlots,
    th::Tensor const& acceptedTokens, th::Tensor const& acceptedLens)
{
    TORCH_CHECK(historyTokens.is_cuda(), "history_tokens must be a CUDA tensor.");
    TORCH_CHECK(historyTokens.is_contiguous(), "history_tokens must be contiguous.");
    TORCH_CHECK(historyTokens.dim() == 2, "history_tokens must be a 2D tensor.");
    TORCH_CHECK(historyTokens.scalar_type() == torch::kInt, "history_tokens dtype must be int32.");
    TORCH_CHECK(historyLens.is_cuda(), "history_lens must be a CUDA tensor.");
    TORCH_CHECK(historyLens.is_contiguous(), "history_lens must be contiguous.");
    TORCH_CHECK(historyLens.dim() == 1, "history_lens must be a 1D tensor.");
    TORCH_CHECK(historyLens.scalar_type() == torch::kInt, "history_lens dtype must be int32.");
    TORCH_CHECK(seqSlots.is_cuda(), "seq_slots must be a CUDA tensor.");
    TORCH_CHECK(seqSlots.is_contiguous(), "seq_slots must be contiguous.");
    TORCH_CHECK(seqSlots.dim() == 1, "seq_slots must be a 1D tensor.");
    TORCH_CHECK(seqSlots.scalar_type() == torch::kInt, "seq_slots dtype must be int32.");
    TORCH_CHECK(acceptedTokens.is_cuda(), "accepted_tokens must be a CUDA tensor.");
    TORCH_CHECK(acceptedTokens.is_contiguous(), "accepted_tokens must be contiguous.");
    TORCH_CHECK(acceptedTokens.dim() == 2, "accepted_tokens must be a 2D tensor.");
    TORCH_CHECK(acceptedTokens.scalar_type() == torch::kInt, "accepted_tokens dtype must be int32.");
    TORCH_CHECK(acceptedLens.is_cuda(), "accepted_lens must be a CUDA tensor.");
    TORCH_CHECK(acceptedLens.is_contiguous(), "accepted_lens must be contiguous.");
    TORCH_CHECK(acceptedLens.dim() == 1, "accepted_lens must be a 1D tensor.");
    TORCH_CHECK(acceptedLens.scalar_type() == torch::kInt, "accepted_lens dtype must be int32.");
    TORCH_CHECK(seqSlots.size(0) == acceptedTokens.size(0), "seq_slots and accepted_tokens row count mismatch.");
    TORCH_CHECK(acceptedLens.size(0) == acceptedTokens.size(0), "accepted_lens and accepted_tokens row count mismatch.");
    TORCH_CHECK(historyLens.size(0) == historyTokens.size(0), "history_lens and history_tokens slot count mismatch.");

    if (acceptedTokens.size(0) == 0 || acceptedTokens.size(1) == 0 || historyTokens.size(1) == 0)
    {
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream(historyTokens.get_device()).stream();
    tk::invokeAppendSpeculativeAcceptedTokens(reinterpret_cast<int32_t*>(historyTokens.data_ptr()),
        reinterpret_cast<int32_t*>(historyLens.data_ptr()), reinterpret_cast<int32_t const*>(seqSlots.data_ptr()),
        reinterpret_cast<int32_t const*>(acceptedTokens.data_ptr()), reinterpret_cast<int32_t const*>(acceptedLens.data_ptr()),
        static_cast<int32_t>(acceptedTokens.size(0)), static_cast<int32_t>(acceptedTokens.size(1)),
        static_cast<int32_t>(historyTokens.size(1)), stream);
}

void speculativeApplyCountFrequencyPenalty(th::Tensor& logits, th::Tensor const& tokenCounts,
    th::Tensor const& rowSlots, th::Tensor const& frequencyPenalties)
{
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor.");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous.");
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor.");
    TORCH_CHECK(tokenCounts.is_cuda(), "token_counts must be a CUDA tensor.");
    TORCH_CHECK(tokenCounts.is_contiguous(), "token_counts must be contiguous.");
    TORCH_CHECK(tokenCounts.dim() == 2, "token_counts must be a 2D tensor.");
    TORCH_CHECK(tokenCounts.scalar_type() == torch::kInt, "token_counts dtype must be int32.");
    TORCH_CHECK(rowSlots.is_cuda(), "row_slots must be a CUDA tensor.");
    TORCH_CHECK(rowSlots.is_contiguous(), "row_slots must be contiguous.");
    TORCH_CHECK(rowSlots.dim() == 1, "row_slots must be a 1D tensor.");
    TORCH_CHECK(rowSlots.scalar_type() == torch::kInt, "row_slots dtype must be int32.");
    TORCH_CHECK(frequencyPenalties.is_cuda(), "frequency_penalties must be a CUDA tensor.");
    TORCH_CHECK(frequencyPenalties.is_contiguous(), "frequency_penalties must be contiguous.");
    TORCH_CHECK(frequencyPenalties.dim() == 1, "frequency_penalties must be a 1D tensor.");
    TORCH_CHECK(frequencyPenalties.scalar_type() == torch::kFloat, "frequency_penalties dtype must be float32.");
    TORCH_CHECK(rowSlots.size(0) == logits.size(0), "row_slots and logits must have the same row count.");
    TORCH_CHECK(
        frequencyPenalties.size(0) == logits.size(0), "frequency_penalties and logits must have the same row count.");
    TORCH_CHECK(tokenCounts.size(1) == logits.size(1), "token_counts and logits vocab size mismatch.");

    if (logits.size(0) == 0 || logits.size(1) == 0)
    {
        return;
    }

    switch (logits.scalar_type())
    {
    case torch::kFloat32:
        invokeTypedApplyCountFrequencyPenalty<float>(logits, tokenCounts, rowSlots, frequencyPenalties);
        break;
    case torch::kFloat16:
        invokeTypedApplyCountFrequencyPenalty<half>(logits, tokenCounts, rowSlots, frequencyPenalties);
        break;
    case torch::kBFloat16:
        invokeTypedApplyCountFrequencyPenalty<__nv_bfloat16>(logits, tokenCounts, rowSlots, frequencyPenalties);
        break;
    default:
        TORCH_CHECK(false, "logits dtype must be float32, float16, or bfloat16.");
    }
}

void speculativeAppendAcceptedTokenCounts(th::Tensor& tokenCounts, th::Tensor const& seqSlots,
    th::Tensor const& acceptedTokens, th::Tensor const& acceptedLens)
{
    TORCH_CHECK(tokenCounts.is_cuda(), "token_counts must be a CUDA tensor.");
    TORCH_CHECK(tokenCounts.is_contiguous(), "token_counts must be contiguous.");
    TORCH_CHECK(tokenCounts.dim() == 2, "token_counts must be a 2D tensor.");
    TORCH_CHECK(tokenCounts.scalar_type() == torch::kInt, "token_counts dtype must be int32.");
    TORCH_CHECK(seqSlots.is_cuda(), "seq_slots must be a CUDA tensor.");
    TORCH_CHECK(seqSlots.is_contiguous(), "seq_slots must be contiguous.");
    TORCH_CHECK(seqSlots.dim() == 1, "seq_slots must be a 1D tensor.");
    TORCH_CHECK(seqSlots.scalar_type() == torch::kInt, "seq_slots dtype must be int32.");
    TORCH_CHECK(acceptedTokens.is_cuda(), "accepted_tokens must be a CUDA tensor.");
    TORCH_CHECK(acceptedTokens.is_contiguous(), "accepted_tokens must be contiguous.");
    TORCH_CHECK(acceptedTokens.dim() == 2, "accepted_tokens must be a 2D tensor.");
    TORCH_CHECK(acceptedTokens.scalar_type() == torch::kInt, "accepted_tokens dtype must be int32.");
    TORCH_CHECK(acceptedLens.is_cuda(), "accepted_lens must be a CUDA tensor.");
    TORCH_CHECK(acceptedLens.is_contiguous(), "accepted_lens must be contiguous.");
    TORCH_CHECK(acceptedLens.dim() == 1, "accepted_lens must be a 1D tensor.");
    TORCH_CHECK(acceptedLens.scalar_type() == torch::kInt, "accepted_lens dtype must be int32.");
    TORCH_CHECK(seqSlots.size(0) == acceptedTokens.size(0), "seq_slots and accepted_tokens row count mismatch.");
    TORCH_CHECK(acceptedLens.size(0) == acceptedTokens.size(0), "accepted_lens and accepted_tokens row count mismatch.");

    if (acceptedTokens.size(0) == 0 || acceptedTokens.size(1) == 0 || tokenCounts.size(1) == 0)
    {
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream(tokenCounts.get_device()).stream();
    tk::invokeAppendSpeculativeAcceptedTokenCounts(reinterpret_cast<int32_t*>(tokenCounts.data_ptr()),
        reinterpret_cast<int32_t const*>(seqSlots.data_ptr()), reinterpret_cast<int32_t const*>(acceptedTokens.data_ptr()),
        reinterpret_cast<int32_t const*>(acceptedLens.data_ptr()), static_cast<int32_t>(acceptedTokens.size(0)),
        static_cast<int32_t>(acceptedTokens.size(1)), static_cast<int32_t>(tokenCounts.size(1)), stream);
}

void speculativeApplySparseCountFrequencyPenalty(th::Tensor& logits, th::Tensor const& tokenIds,
    th::Tensor const& tokenCounts, th::Tensor const& countLens, th::Tensor const& rowSlots,
    th::Tensor const& frequencyPenalties)
{
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor.");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous.");
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor.");
    TORCH_CHECK(tokenIds.is_cuda(), "token_ids must be a CUDA tensor.");
    TORCH_CHECK(tokenIds.is_contiguous(), "token_ids must be contiguous.");
    TORCH_CHECK(tokenIds.dim() == 2, "token_ids must be a 2D tensor.");
    TORCH_CHECK(tokenIds.scalar_type() == torch::kInt, "token_ids dtype must be int32.");
    TORCH_CHECK(tokenCounts.is_cuda(), "token_counts must be a CUDA tensor.");
    TORCH_CHECK(tokenCounts.is_contiguous(), "token_counts must be contiguous.");
    TORCH_CHECK(tokenCounts.dim() == 2, "token_counts must be a 2D tensor.");
    TORCH_CHECK(tokenCounts.scalar_type() == torch::kInt, "token_counts dtype must be int32.");
    TORCH_CHECK(countLens.is_cuda(), "count_lens must be a CUDA tensor.");
    TORCH_CHECK(countLens.is_contiguous(), "count_lens must be contiguous.");
    TORCH_CHECK(countLens.dim() == 1, "count_lens must be a 1D tensor.");
    TORCH_CHECK(countLens.scalar_type() == torch::kInt, "count_lens dtype must be int32.");
    TORCH_CHECK(rowSlots.is_cuda(), "row_slots must be a CUDA tensor.");
    TORCH_CHECK(rowSlots.is_contiguous(), "row_slots must be contiguous.");
    TORCH_CHECK(rowSlots.dim() == 1, "row_slots must be a 1D tensor.");
    TORCH_CHECK(rowSlots.scalar_type() == torch::kInt, "row_slots dtype must be int32.");
    TORCH_CHECK(frequencyPenalties.is_cuda(), "frequency_penalties must be a CUDA tensor.");
    TORCH_CHECK(frequencyPenalties.is_contiguous(), "frequency_penalties must be contiguous.");
    TORCH_CHECK(frequencyPenalties.dim() == 1, "frequency_penalties must be a 1D tensor.");
    TORCH_CHECK(frequencyPenalties.scalar_type() == torch::kFloat, "frequency_penalties dtype must be float32.");
    TORCH_CHECK(tokenIds.size(0) == tokenCounts.size(0), "token_ids and token_counts slot count mismatch.");
    TORCH_CHECK(tokenIds.size(1) == tokenCounts.size(1), "token_ids and token_counts capacity mismatch.");
    TORCH_CHECK(countLens.size(0) == tokenIds.size(0), "count_lens and token_ids slot count mismatch.");
    TORCH_CHECK(rowSlots.size(0) == logits.size(0), "row_slots and logits must have the same row count.");
    TORCH_CHECK(
        frequencyPenalties.size(0) == logits.size(0), "frequency_penalties and logits must have the same row count.");

    if (logits.size(0) == 0 || tokenIds.size(1) == 0 || logits.size(1) == 0)
    {
        return;
    }

    switch (logits.scalar_type())
    {
    case torch::kFloat32:
        invokeTypedApplySparseCountFrequencyPenalty<float>(
            logits, tokenIds, tokenCounts, countLens, rowSlots, frequencyPenalties);
        break;
    case torch::kFloat16:
        invokeTypedApplySparseCountFrequencyPenalty<half>(
            logits, tokenIds, tokenCounts, countLens, rowSlots, frequencyPenalties);
        break;
    case torch::kBFloat16:
        invokeTypedApplySparseCountFrequencyPenalty<__nv_bfloat16>(
            logits, tokenIds, tokenCounts, countLens, rowSlots, frequencyPenalties);
        break;
    default:
        TORCH_CHECK(false, "logits dtype must be float32, float16, or bfloat16.");
    }
}

void speculativeAppendSparseTokenCounts(th::Tensor& tokenIds, th::Tensor& tokenCounts, th::Tensor& countLens,
    th::Tensor const& seqSlots, th::Tensor const& acceptedTokens, th::Tensor const& acceptedLens, int64_t vocabSize)
{
    TORCH_CHECK(tokenIds.is_cuda(), "token_ids must be a CUDA tensor.");
    TORCH_CHECK(tokenIds.is_contiguous(), "token_ids must be contiguous.");
    TORCH_CHECK(tokenIds.dim() == 2, "token_ids must be a 2D tensor.");
    TORCH_CHECK(tokenIds.scalar_type() == torch::kInt, "token_ids dtype must be int32.");
    TORCH_CHECK(tokenCounts.is_cuda(), "token_counts must be a CUDA tensor.");
    TORCH_CHECK(tokenCounts.is_contiguous(), "token_counts must be contiguous.");
    TORCH_CHECK(tokenCounts.dim() == 2, "token_counts must be a 2D tensor.");
    TORCH_CHECK(tokenCounts.scalar_type() == torch::kInt, "token_counts dtype must be int32.");
    TORCH_CHECK(countLens.is_cuda(), "count_lens must be a CUDA tensor.");
    TORCH_CHECK(countLens.is_contiguous(), "count_lens must be contiguous.");
    TORCH_CHECK(countLens.dim() == 1, "count_lens must be a 1D tensor.");
    TORCH_CHECK(countLens.scalar_type() == torch::kInt, "count_lens dtype must be int32.");
    TORCH_CHECK(seqSlots.is_cuda(), "seq_slots must be a CUDA tensor.");
    TORCH_CHECK(seqSlots.is_contiguous(), "seq_slots must be contiguous.");
    TORCH_CHECK(seqSlots.dim() == 1, "seq_slots must be a 1D tensor.");
    TORCH_CHECK(seqSlots.scalar_type() == torch::kInt, "seq_slots dtype must be int32.");
    TORCH_CHECK(acceptedTokens.is_cuda(), "accepted_tokens must be a CUDA tensor.");
    TORCH_CHECK(acceptedTokens.is_contiguous(), "accepted_tokens must be contiguous.");
    TORCH_CHECK(acceptedTokens.dim() == 2, "accepted_tokens must be a 2D tensor.");
    TORCH_CHECK(acceptedTokens.scalar_type() == torch::kInt, "accepted_tokens dtype must be int32.");
    TORCH_CHECK(acceptedLens.is_cuda(), "accepted_lens must be a CUDA tensor.");
    TORCH_CHECK(acceptedLens.is_contiguous(), "accepted_lens must be contiguous.");
    TORCH_CHECK(acceptedLens.dim() == 1, "accepted_lens must be a 1D tensor.");
    TORCH_CHECK(acceptedLens.scalar_type() == torch::kInt, "accepted_lens dtype must be int32.");
    TORCH_CHECK(tokenIds.size(0) == tokenCounts.size(0), "token_ids and token_counts slot count mismatch.");
    TORCH_CHECK(tokenIds.size(1) == tokenCounts.size(1), "token_ids and token_counts capacity mismatch.");
    TORCH_CHECK(countLens.size(0) == tokenIds.size(0), "count_lens and token_ids slot count mismatch.");
    TORCH_CHECK(seqSlots.size(0) == acceptedTokens.size(0), "seq_slots and accepted_tokens row count mismatch.");
    TORCH_CHECK(acceptedLens.size(0) == acceptedTokens.size(0), "accepted_lens and accepted_tokens row count mismatch.");
    TORCH_CHECK(vocabSize > 0, "vocab_size must be positive.");

    if (acceptedTokens.size(0) == 0 || acceptedTokens.size(1) == 0 || tokenIds.size(1) == 0)
    {
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream(tokenIds.get_device()).stream();
    tk::invokeAppendSpeculativeSparseTokenCounts(reinterpret_cast<int32_t*>(tokenIds.data_ptr()),
        reinterpret_cast<int32_t*>(tokenCounts.data_ptr()), reinterpret_cast<int32_t*>(countLens.data_ptr()),
        reinterpret_cast<int32_t const*>(seqSlots.data_ptr()), reinterpret_cast<int32_t const*>(acceptedTokens.data_ptr()),
        reinterpret_cast<int32_t const*>(acceptedLens.data_ptr()), static_cast<int32_t>(acceptedTokens.size(0)),
        static_cast<int32_t>(acceptedTokens.size(1)), static_cast<int32_t>(tokenIds.size(1)),
        static_cast<int32_t>(vocabSize), stream);
}

void speculativeInitSparseTokenCounts(th::Tensor& tokenIds, th::Tensor& tokenCounts, th::Tensor& countLens,
    th::Tensor const& promptTokenIds, th::Tensor const& promptTokenCounts, th::Tensor const& promptLens,
    th::Tensor const& seqSlots, int64_t vocabSize)
{
    TORCH_CHECK(tokenIds.is_cuda(), "token_ids must be a CUDA tensor.");
    TORCH_CHECK(tokenIds.is_contiguous(), "token_ids must be contiguous.");
    TORCH_CHECK(tokenIds.dim() == 2, "token_ids must be a 2D tensor.");
    TORCH_CHECK(tokenIds.scalar_type() == torch::kInt, "token_ids dtype must be int32.");
    TORCH_CHECK(tokenCounts.is_cuda(), "token_counts must be a CUDA tensor.");
    TORCH_CHECK(tokenCounts.is_contiguous(), "token_counts must be contiguous.");
    TORCH_CHECK(tokenCounts.dim() == 2, "token_counts must be a 2D tensor.");
    TORCH_CHECK(tokenCounts.scalar_type() == torch::kInt, "token_counts dtype must be int32.");
    TORCH_CHECK(countLens.is_cuda(), "count_lens must be a CUDA tensor.");
    TORCH_CHECK(countLens.is_contiguous(), "count_lens must be contiguous.");
    TORCH_CHECK(countLens.dim() == 1, "count_lens must be a 1D tensor.");
    TORCH_CHECK(countLens.scalar_type() == torch::kInt, "count_lens dtype must be int32.");
    TORCH_CHECK(promptTokenIds.is_cuda(), "prompt_token_ids must be a CUDA tensor.");
    TORCH_CHECK(promptTokenIds.is_contiguous(), "prompt_token_ids must be contiguous.");
    TORCH_CHECK(promptTokenIds.dim() == 2, "prompt_token_ids must be a 2D tensor.");
    TORCH_CHECK(promptTokenIds.scalar_type() == torch::kInt, "prompt_token_ids dtype must be int32.");
    TORCH_CHECK(promptTokenCounts.is_cuda(), "prompt_token_counts must be a CUDA tensor.");
    TORCH_CHECK(promptTokenCounts.is_contiguous(), "prompt_token_counts must be contiguous.");
    TORCH_CHECK(promptTokenCounts.dim() == 2, "prompt_token_counts must be a 2D tensor.");
    TORCH_CHECK(promptTokenCounts.scalar_type() == torch::kInt, "prompt_token_counts dtype must be int32.");
    TORCH_CHECK(promptLens.is_cuda(), "prompt_lens must be a CUDA tensor.");
    TORCH_CHECK(promptLens.is_contiguous(), "prompt_lens must be contiguous.");
    TORCH_CHECK(promptLens.dim() == 1, "prompt_lens must be a 1D tensor.");
    TORCH_CHECK(promptLens.scalar_type() == torch::kInt, "prompt_lens dtype must be int32.");
    TORCH_CHECK(seqSlots.is_cuda(), "seq_slots must be a CUDA tensor.");
    TORCH_CHECK(seqSlots.is_contiguous(), "seq_slots must be contiguous.");
    TORCH_CHECK(seqSlots.dim() == 1, "seq_slots must be a 1D tensor.");
    TORCH_CHECK(seqSlots.scalar_type() == torch::kInt, "seq_slots dtype must be int32.");
    TORCH_CHECK(tokenIds.size(0) == tokenCounts.size(0), "token_ids and token_counts slot count mismatch.");
    TORCH_CHECK(tokenIds.size(1) == tokenCounts.size(1), "token_ids and token_counts capacity mismatch.");
    TORCH_CHECK(countLens.size(0) == tokenIds.size(0), "count_lens and token_ids slot count mismatch.");
    TORCH_CHECK(promptTokenIds.size(0) == promptTokenCounts.size(0),
        "prompt_token_ids and prompt_token_counts row count mismatch.");
    TORCH_CHECK(promptTokenIds.size(1) == promptTokenCounts.size(1),
        "prompt_token_ids and prompt_token_counts capacity mismatch.");
    TORCH_CHECK(promptLens.size(0) == promptTokenIds.size(0), "prompt_lens and prompt_token_ids row count mismatch.");
    TORCH_CHECK(seqSlots.size(0) == promptTokenIds.size(0), "seq_slots and prompt_token_ids row count mismatch.");
    TORCH_CHECK(vocabSize > 0, "vocab_size must be positive.");

    if (promptTokenIds.size(0) == 0 || promptTokenIds.size(1) == 0 || tokenIds.size(1) == 0)
    {
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream(tokenIds.get_device()).stream();
    tk::invokeInitSpeculativeSparseTokenCounts(reinterpret_cast<int32_t*>(tokenIds.data_ptr()),
        reinterpret_cast<int32_t*>(tokenCounts.data_ptr()), reinterpret_cast<int32_t*>(countLens.data_ptr()),
        reinterpret_cast<int32_t const*>(promptTokenIds.data_ptr()),
        reinterpret_cast<int32_t const*>(promptTokenCounts.data_ptr()),
        reinterpret_cast<int32_t const*>(promptLens.data_ptr()), reinterpret_cast<int32_t const*>(seqSlots.data_ptr()),
        static_cast<int32_t>(promptTokenIds.size(0)), static_cast<int32_t>(promptTokenIds.size(1)),
        static_cast<int32_t>(tokenIds.size(1)), static_cast<int32_t>(vocabSize), stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("speculative_apply_token_penalties(Tensor(a!) logits, Tensor token_ids, Tensor penalty_values) -> ()");
    m.def(
        "speculative_apply_history_frequency_penalty(Tensor(a!) logits, Tensor history_tokens, Tensor history_lens, Tensor row_slots, Tensor frequency_penalties) -> ()");
    m.def(
        "speculative_append_accepted_tokens(Tensor(a!) history_tokens, Tensor(b!) history_lens, Tensor seq_slots, Tensor accepted_tokens, Tensor accepted_lens) -> ()");
    m.def(
        "speculative_apply_count_frequency_penalty(Tensor(a!) logits, Tensor token_counts, Tensor row_slots, Tensor frequency_penalties) -> ()");
    m.def(
        "speculative_append_accepted_token_counts(Tensor(a!) token_counts, Tensor seq_slots, Tensor accepted_tokens, Tensor accepted_lens) -> ()");
    m.def(
        "speculative_apply_sparse_count_frequency_penalty(Tensor(a!) logits, Tensor token_ids, Tensor token_counts, Tensor count_lens, Tensor row_slots, Tensor frequency_penalties) -> ()");
    m.def(
        "speculative_append_sparse_token_counts(Tensor(a!) token_ids, Tensor(b!) token_counts, Tensor(c!) count_lens, Tensor seq_slots, Tensor accepted_tokens, Tensor accepted_lens, int vocab_size) -> ()");
    m.def(
        "speculative_init_sparse_token_counts(Tensor(a!) token_ids, Tensor(b!) token_counts, Tensor(c!) count_lens, Tensor prompt_token_ids, Tensor prompt_token_counts, Tensor prompt_lens, Tensor seq_slots, int vocab_size) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("speculative_apply_token_penalties", &tensorrt_llm::torch_ext::speculativeApplyTokenPenalties);
    m.impl("speculative_apply_history_frequency_penalty",
        &tensorrt_llm::torch_ext::speculativeApplyHistoryFrequencyPenalty);
    m.impl("speculative_append_accepted_tokens", &tensorrt_llm::torch_ext::speculativeAppendAcceptedTokens);
    m.impl("speculative_apply_count_frequency_penalty",
        &tensorrt_llm::torch_ext::speculativeApplyCountFrequencyPenalty);
    m.impl("speculative_append_accepted_token_counts",
        &tensorrt_llm::torch_ext::speculativeAppendAcceptedTokenCounts);
    m.impl("speculative_apply_sparse_count_frequency_penalty",
        &tensorrt_llm::torch_ext::speculativeApplySparseCountFrequencyPenalty);
    m.impl("speculative_append_sparse_token_counts",
        &tensorrt_llm::torch_ext::speculativeAppendSparseTokenCounts);
    m.impl("speculative_init_sparse_token_counts", &tensorrt_llm::torch_ext::speculativeInitSparseTokenCounts);
}
