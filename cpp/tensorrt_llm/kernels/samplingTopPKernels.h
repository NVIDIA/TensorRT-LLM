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

#include "tensorrt_llm/kernels/decodingCommon.h"
#include <curand_kernel.h>

namespace tensorrt_llm
{
namespace kernels
{

//! \brief Initialize buffers for topP inference
//!
//! \param topPIdValBuf output buffer [batchSize, vocabSize]. Value at {bi,vi} position contains vi token id.
//! \param topPOffsetBuf output buffer [batchSize+1].
//! \param beginTopPOffsetBuf output buffer [batchSize+1].
//! \param batchSize number of requests in batch
//! \param vocabSize size of the inner dimension
//! \param stream stream
void invokeTopPInitialize(int* topPIdValBuf, int* topPOffsetBuf, int* beginTopPOffsetBuf, const size_t batchSize,
    const int vocabSize, cudaStream_t stream);

/**
//! \brief Given logProbs, performs top P sampling. Fills sampled tokens to outputIds.
//! Computes sequenceLength, finished state, cumLogProbs inplace.
//! Sampling per request can be controlled using skipDecode and topPs parameters.
//! Function sets workspaceSize and cubTempStorageSize and exits early if workspace is nullptr.
//!
//! \param workspace pointer to the workspace. Has to be pre-allocated by caller. Function does not take ownership of
the
//! buffer.
//! \param workspaceSize size of the workspace in bytes.
//! \param cubTempStorageSize workspace size for cub radix sort.
//! \param outputIds output buffer [batchSize][maxSeqLen]. Contains pointers to rows with output tokens per request.
//! \param sequenceLength input/output buffer [batchSize]. Current sequence length of the request up to, but excluding
endId token.
//! \param finishedInput input buffer [batchSize]. Exit early if true.
//! \param finishedOutput output buffer [batchSize]. Set flag if sequence has finished (if finished || outputId ==
endId).
//! \param cumLogProbs input/output buffer [batchSize]. Cumulative log probability of selected tokens. Ignored if
nullptr.
//! \param outputLogProbs output buffer [batchSize]. Log probs is the probability
//! induced by the top-k sampling. We normalize the probability 'expLogit' of the selected token by the probability
//! 's_sum' of a set of top-k tokens, meaning the logProb is the probability of the selected token, conditioned on the
//! event that it is selected, i.e., log_prob = log P(i | i is in top-k) = log(expLogit / s_sum). Ignored if nullptr.
//! \param logProbs input buffer [batchSize x vocabSizePadded]. Log probabilities of each token in the vocab.
//! If cumLogProbs or outputLogProbs are specified, logProbs must contain **just** probabilities instead of log
//! probabilities.
//! \param idVals input buffer [batchSize, vocabSize]. Value at {bi,vi} position contains vi token id.
//! Initialized using invokeTopPInitialize.
//! \param offsetBuf input buffer [batchSize+1]. Array of offsets initialized using invokeTopPInitialize.
//! \param beginOffsetBuf input buffer [batchSize+1]. Array of offsets initialized using invokeTopPInitialize.
//! \param curandstate input buffer [batchSize]. Curand states properly initialized using
//! invokeCurandInitialize per request.
//! \param batchSize batch size
//! \param vocabSizePadded size of padded vocab
//! \param endIds input buffer [batchSize]. EOS token ids per request
//! \param maxTopP maximum among all topPs P for topP sampling
//! \param topPs input buffer [batchSize]. P for topP sampling per request. Supported P is in range (0.0; 1.0].
//! If nullptr maxTopP is used for all requests.
//! \param stream cuda stream
//! \param cudaDeviceProp
//! \param skipDecode input buffer [batchSize]. Flags whether to skip decoding per request
 */
template <typename T>
void invokeBatchTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize, int** outputIds,
    int* sequenceLength, const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, const T* logProbs, const int* idVals, int* offsetBuf, int* beginOffsetBuf,
    curandState_t* curandstate, const int batchSize, const size_t vocabSizePadded, const int* endIds,
    const float maxTopP, const float* topPs, cudaStream_t stream, const bool* skipDecode);

//! \brief Specialization of invokeBatchTopPSampling with topPs=nullptr
template <typename T>
void invokeTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize, int** outputIds,
    int* sequenceLength, const FinishedState* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, const T* logProbs, const int* idVals, int* offsetBuf, int* beginOffsetBuf,
    curandState_t* curandstate, const int batchSize, const size_t vocabSizePadded, const int* endIds, const float topPp,
    cudaStream_t stream, const bool* skipDecode);

//! \brief Compute the topp decay by https://arxiv.org/pdf/2206.04624.pdf
//!        In short, the formula is
//!          runtimeTopP = max(runtimeTopP * topPDecay, topPMin)
//!        If generating the topPResetIds, then reset the runtimeTopP.
//!
//! \param runtimeTopP
//! \param runtimeInitialTopP
//! \param outputIds
//! \param topPDecay
//! \param topPMin
//! \param topPResetIds
//! \param localBatchSize
void invokeComputeToppDecay(float* runtimeTopP, const float* runtimeInitialTopP, const int** outputIds,
    const float* topPDecay, const float* topPMin, const int32_t* topPResetIds, const int* sequenceLengths,
    const int localBatchSize, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
