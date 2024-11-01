/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/runtime/common.h"
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{
//! \brief Sets finished state to FinishedState::FINISHED_STOP_WORDS if any of the stopWords is met.
//!
//! \param outputIds input buffer [maxBatchSize][maxSeqLen]. Contains pointers to rows with output tokens per request
//! \param parentIds input buffer [maxBatchSize][maxSeqLen]. Contains pointers to rows with parent ids. Applicable when
//! beamWidth > 1
//! \param stopWords input buffer [maxBatchSize][2, stopWordsLen]. For each instance in batch the first row
//! is the token ids of the stop words. The second row is the exclusive prefix sum of the word lengths.
//! In case all the words are made of a single token,
//! the inner-most dimension of the tensor must be increased by 1.
//! \param finished input/output buffer [maxBatchSize, beamWidth].
//! Finished states. Set to FinishedState::FINISHED_STOP_WORDS if any
//! sequence of the stop words is met
//! \param sequenceLengths input/output buffer [maxBatchSize, beamWidth]. Current sequence
//! lengths of the request tokens.
//! When numNewTokens is not nullptr, it is updated to the first entrance of the found stop words.
//! \param batchSlots input buffer[batchSize], optional. Indices of rows of data in memory pool
//! \param stopWordsLen input buffer [maxBatchSize], cumulative length of all stop words per request
//! \param numNewTokens input/output buffer [maxBatchSize], optional, number of tokens predicted per step.
//! If nullptr, 1 is used.
//! \param maxStopWordsLen maximum stopWordsLen over all requests in the batch
//! \param batchSize batch size
//! \param beamWidth beam width
//! \param maxSeqLen maximum length of the sequence
//! \param stream stream
void invokeStopWordsCriterion(runtime::TokenIdType const** outputIds, runtime::SizeType32 const** parentIds,
    runtime::TokenIdType const* const* stopWords, FinishedState* finished, runtime::SizeType32* sequenceLengths,
    runtime::SizeType32 const* batchSlots, runtime::SizeType32 const* stopWordsLen, runtime::SizeType32* numNewTokens,
    runtime::SizeType32 maxStopWordsLen, runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth,
    runtime::SizeType32 maxSeqLen, cudaStream_t stream);

//! \brief Sets finished states based on the sequenceLimitLength and computes number of finished sequences in the batch.
//!
//! \param finished input/output buffer [maxBatchSize, beamWidth]. Finished states. Set to
//! FinishedState::FINISHED_MAX_LENGTH if sequenceLengths >= sequenceLimitLength
//! \param finishedSum output buffer [1].
//! Total sum of finished requests
//! \param sequenceLimitLength input buffer [maxBatchSize]. Maximum sequence length.
//! \param sequenceLengths input/output buffer [maxBatchSize, beamWidth].
//! Current sequence lengths of the request tokens.
//! \param numNewTokens output buffer [maxBatchSize], optional. Number of tokens per step for each request.
//! It is assumed that all requests have maxTokensPerStep tokens per step if nullptr.
//! \param batchSlots input buffer[batchSize], optional. Indices of rows of data in memory pool
//! \param batchSize batch size
//! \param beamWidth beam width
//! \param stream stream
void invokeLengthCriterion(FinishedState* finished, runtime::SizeType32* finishedSum,
    runtime::SizeType32 const* sequenceLimitLength, runtime::SizeType32* sequenceLengths,
    runtime::SizeType32* numNewTokens, runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize,
    runtime::SizeType32 beamWidth, cudaStream_t stream);

//! \brief Sets finished states based on the endIds and adjusts sequence length to length before the first EOS token.
//! Does not support beamWidth > 1 for now.
//!
//! \param outputIds input buffer [maxBatchSize][beamWidth, maxSeqLen].
//! Contains pointers to rows with output tokens per request.
//! \param endIds input buffer [maxBatchSize]. EOS token ids per request
//! \param finished input/output buffer
//! [maxBatchSize, beamWidth]. Finished states. Set to FinishedState::FINISHED_EOS if any new tokens contain EOS.
//! \param sequenceLengths input/output buffer [maxBatchSize, beamWidth].
//! Current sequence lengths of the request tokens.
//! \param numNewTokens input/output buffer [maxBatchSize], optional. Number of tokens per step for each request.
//! It is assumed that all requests have maxTokensPerStep tokens per step if nullptr.
//! \param batchSlots input buffer[batchSize], optional. Indices of rows of data in memory pool
//! \param batchSize batch size
//! \param beamWidth beam width. beamWidth > 1 is not supported for now.
//! \param maxTokensPerStep maximum number of tokens decoded per step
//! \param stream stream
void invokeExplicitEOSCriterion(runtime::TokenIdType const** outputIds, runtime::TokenIdType const* endIds,
    FinishedState* finished, runtime::SizeType32* sequenceLengths, runtime::SizeType32* numNewTokens,
    runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth,
    runtime::SizeType32 maxTokensPerStep, cudaStream_t stream);
} // namespace kernels
} // namespace tensorrt_llm
