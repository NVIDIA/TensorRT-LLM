/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/runtime/common.h"
#include <cuda_runtime.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding
{

//! \brief Tree mask mode for dynamic tree building
enum class TreeMaskMode : int32_t
{
    QLEN_ONLY = 1,            //! Query length only (bool array) [bs, draftTokenNum, draftTokenNum]
    QLEN_ONLY_BITPACKING = 2, //! Query length with bit packing (32 bits per int32)
};

//! \brief Build dynamic tree structure efficiently
//! Ported from SGLang's eagle_utils.cu for high-performance tree construction.
//! \param parentList [batchSize, topK * (depth - 1) + 1], on GPU. int64.
//! Parent indices for each token in the tree (layer-local relative indices).
//! \param selectedIndex [batchSize, numDraftTokens - 1], on GPU. int64.
//! Selected token indices (excluding root node).
//! \param treeMask output buffer [varies by mode], on GPU. int32.
//! Attention mask for tree structure. Shape depends on treeMaskMode.
//! \param positions output buffer [batchSize * numDraftTokens], on GPU. int32.
//! Position IDs for each draft token.
//! \param retrieveIndex output buffer [batchSize, numDraftTokens], on GPU. int32.
//! Local indices for retrieving tokens (0, 1, ..., numDraftTokens-1).
//! \param retrieveNextToken output buffer [batchSize, numDraftTokens], on GPU. int32.
//! Index of the first child token for each node.
//! \param retrieveNextSibling output buffer [batchSize, numDraftTokens], on GPU. int32.
//! Index of the next sibling token for each node.
//! \param batchSize runtime::SizeType32. Batch size.
//! \param topK runtime::SizeType32. Number of top-K tokens per node.
//! \param depth runtime::SizeType32. Tree depth.
//! \param numDraftTokens runtime::SizeType32. Total number of draft tokens.
//! \param treeMaskMode TreeMaskMode. Attention mask mode.
//! \param stream cuda stream
//! \param numInt32PerRow For QLEN_ONLY_BITPACKING: treeMask row stride in int32s (must be > 0). Ignored otherwise.
void invokeBuildDynamicTree(int64_t const* parentList, int64_t const* selectedIndex, void* treeMask, int32_t* positions,
    int32_t* retrieveIndex, int32_t* retrieveNextToken, int32_t* retrieveNextSibling, runtime::SizeType32 batchSize,
    runtime::SizeType32 topK, runtime::SizeType32 depth, runtime::SizeType32 numDraftTokens, TreeMaskMode treeMaskMode,
    cudaStream_t stream, runtime::SizeType32 numInt32PerRow);

//! \brief Build tree-position -> unique draft-prob row mapping for rejection sampling.
//! \param topkScoreIndices [batchSize, numDraftTokens], on GPU. int64.
//! History-buffer indices selected for each final tree position.
//! \param draftProbIndices output [batchSize, numDraftTokens + 1], on GPU. int32.
//! Column 0 is reserved for the root and set to 0.
//! \param batchSize runtime::SizeType32. Batch size.
//! \param topK runtime::SizeType32. Tree top-K branching factor.
//! \param numDraftTokens runtime::SizeType32. Total number of non-root draft positions.
//! \param stream cuda stream.
void invokeBuildDraftProbIndices(int64_t const* topkScoreIndices, int32_t* draftProbIndices,
    runtime::SizeType32 batchSize, runtime::SizeType32 topK, runtime::SizeType32 numDraftTokens, cudaStream_t stream);

//! \brief Verify dynamic tree using greedy strategy
//! Verifies draft tokens against target model predictions using tree traversal.
//! All index/token pointer parameters use int64.
//! \param predicts output buffer [seqLensSum], on GPU. int64.
//! Verified token predictions.
//! \param acceptIndex output buffer [batchSize, numSpecStep], on GPU. int64.
//! Indices of accepted tokens.
//! \param acceptTokenNum output buffer [batchSize], on GPU. int64.
//! Number of accepted tokens per request.
//! \param acceptToken output buffer [batchSize, numSpecStep], on GPU. int64.
//! Contiguous accepted token ids along the accepted path (root prediction,
//! accepted draft predictions, bonus token). Entry count per request =
//! acceptTokenNum + 1.
//! \param candidates [batchSize, numDraftTokens], on GPU. int64.
//! Candidate draft tokens.
//! \param retrieveIndex [batchSize, numDraftTokens], on GPU. int32.
//! Indices for retrieving tokens.
//! \param retrieveNextToken [batchSize, numDraftTokens], on GPU. int32.
//! Index of the first child token.
//! \param retrieveNextSibling [batchSize, numDraftTokens], on GPU. int32.
//! Index of the next sibling token.
//! \param targetPredict [batchSize, numDraftTokens], on GPU. int64.
//! Target model predictions.
//! \param batchSize runtime::SizeType32. Batch size.
//! \param numDraftTokens runtime::SizeType32. Total number of draft tokens.
//! \param numSpecStep runtime::SizeType32. Number of speculative steps.
//! \param stream cuda stream
void invokeVerifyDynamicTreeGreedy(int64_t* predicts, int64_t* acceptIndex, int64_t* acceptTokenNum,
    int64_t* acceptToken, int64_t const* candidates, int32_t const* retrieveIndex, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, int64_t const* targetPredict, bool const* treeValid,
    runtime::SizeType32 batchSize, runtime::SizeType32 numDraftTokens, runtime::SizeType32 numSpecStep,
    cudaStream_t stream);

//! \brief Verify dynamic tree using rejection sampling.
//! For each request, traverses the tree depth-by-depth. At each depth, siblings are tried
//! in order; the first sibling accepted by rejection sampling (p_target/p_draft) continues
//! the path. If all siblings are rejected, a correction token is sampled from (target-draft)_+.
//! \param acceptIndex    output [batchSize, numSpecStep] int64 — tree positions of accepted tokens.
//! \param acceptTokenNum output [batchSize] int64 — # accepted draft tokens (excl. root).
//! \param acceptToken    output [batchSize, numSpecStep] int64 — accepted/correction token ids.
//! \param candidates     [batchSize, numDraftTokens] int64; col 0 = root (target sample).
//! \param draftProbs     [batchSize, numDraftProbRows, vocabSize] float32.
//!        Unique draft probability rows per request; tree positions map into this
//!        tensor via draftProbIndices.
//! \param targetProbs    [batchSize, numDraftTokens, vocabSize] float32; index 0 = root.
//! \param targetSupportIndices [batchSize, numDraftTokens, maxTargetSupportSize] int32; compact token ids that
//!        survive top-k/top-p filtering for each row, padded with -1. May be empty when no filtering is active.
//! \param targetSupportLengths [batchSize, numDraftTokens] int32; valid support length per row. May be empty when
//!        no filtering is active.
//! \param draftProbIndices [batchSize, numDraftTokens] int32; maps each tree position to the
//!        corresponding row in draftProbs. Root is unused.
//! \param retrieveNextToken   [batchSize, numDraftTokens] int32 first-child pointer, -1=none.
//! \param retrieveNextSibling [batchSize, numDraftTokens] int32 next-sibling pointer, -1=none.
//! \param treeValid      [batchSize] bool; false means no valid tree exists for this request.
//! \param batchSize      runtime::SizeType32.
//! \param numDraftProbRows runtime::SizeType32. Number of unique draft-prob rows per request.
//! \param maxTargetSupportSize runtime::SizeType32. Third dim of targetSupportIndices. Can be zero.
//! \param numDraftTokens runtime::SizeType32. Total tree nodes per request (including root).
//! \param numSpecStep    runtime::SizeType32. Second dim of acceptIndex/acceptToken.
//! \param vocabSize      runtime::SizeType32. Vocabulary size.
//! \param seed           [1] int64 on GPU. Philox RNG seed.
//! \param offset         [1] int64 on GPU. Philox RNG offset.
//! \param stream         cudaStream_t.
void invokeVerifyDynamicTreeRejection(int64_t* acceptIndex, int64_t* acceptTokenNum, int64_t* acceptToken,
    int64_t const* candidates, float const* draftProbs, float const* targetProbs, int32_t const* targetSupportIndices,
    int32_t const* targetSupportLengths, int32_t const* draftProbIndices, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, bool const* treeValid, runtime::SizeType32 batchSize,
    runtime::SizeType32 numDraftProbRows, runtime::SizeType32 maxTargetSupportSize, runtime::SizeType32 numDraftTokens,
    runtime::SizeType32 numSpecStep, runtime::SizeType32 vocabSize, int64_t const* seed, int64_t const* offset,
    cudaStream_t stream);

//! \brief Compute draft probabilities for dynamic-tree rejection sampling from logits.
//! \param draftLogits [batchSize * numDraftProbRows, draftVocabSize], on GPU.
//! \param temperatures [batchSize], on GPU.
//! \param numDraftProbRows runtime::SizeType32. Unique draft-prob rows per request.
//! \param topK Optional [batchSize], on GPU.
//! \param topP Optional [batchSize], on GPU.
//! \param targetVocabSize runtime::SizeType32. Output vocabulary size after optional d2t expansion.
//! \param d2t Optional [draftVocabSize], on GPU.
//! \return [batchSize, numDraftProbRows, targetVocabSize] float32 probabilities.
torch::Tensor computeDraftProbsForDynamicTreeRejection(torch::Tensor const& draftLogits,
    torch::Tensor const& temperatures, runtime::SizeType32 numDraftProbRows, torch::optional<torch::Tensor> const& topK,
    torch::optional<torch::Tensor> const& topP, runtime::SizeType32 targetVocabSize, bool skipTemperature,
    torch::optional<torch::Tensor> const& d2t, runtime::SizeType32 kMax = 0, bool skipAllSamplingParams = false);

//! \brief Compute target probabilities for dynamic-tree rejection sampling from logits.
//! \param targetLogits [batchSize * numDraftTokens, targetVocabSize], on GPU.
//! \param temperatures [batchSize], on GPU.
//! \param numDraftTokens runtime::SizeType32. Total tree nodes per request (including root).
//! \param topK Optional [batchSize], on GPU.
//! \param topP Optional [batchSize], on GPU.
//! \param kMax runtime::SizeType32. Max top-K value across the batch; enables the fast topk
//!        path when > 0. Must be computed on CPU (e.g. topK.max().item()). Default 0 = fallback
//!        to full sort.
//! \return Tuple of:
//!   1. [batchSize, numDraftTokens, targetVocabSize] float32 probabilities.
//!   2. [batchSize, numDraftTokens, maxTargetSupportSize] int32 compact token ids that survive
//!      top-k/top-p filtering for each row, padded with -1. Empty when no filtering is active.
//!   3. [batchSize, numDraftTokens] int32 support lengths. Empty when no filtering is active.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> computeTargetProbsForDynamicTreeRejection(
    torch::Tensor const& targetLogits, torch::Tensor const& temperatures, runtime::SizeType32 numDraftTokens,
    torch::optional<torch::Tensor> const& topK, torch::optional<torch::Tensor> const& topP, bool skipTemperature,
    runtime::SizeType32 kMax = 0, bool skipAllSamplingParams = false);

} // namespace kernels::speculative_decoding

TRTLLM_NAMESPACE_END
