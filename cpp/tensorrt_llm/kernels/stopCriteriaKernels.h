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
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{
//! \brief Sets finished state to FinishedState::FINISHED_STOP_WORDS if any of the stopWords is met.
//!
//! \param outputIds input buffer [batchSize][maxSeqLen]. Contains pointers to rows with output tokens per request
//! \param parentIds input buffer [batchSize][maxSeqLen]. Contains pointers to rows with parent ids. Applicable when
//! beamWidth > 1
//! \param stopWords input buffer [batchSize, 2, stopWordsLen]. For each instance in batch the first row
//! is the token ids of the stop words. The second row is the exclusive prefix sum of the word lengths.
//! In case all the words are made of a single token,
//! the inner-most dimension of the tensor must be increased by 1.
//! \param finished input/output buffer [batchSize, beamWidth].
//! Finished states. Set to FinishedState::FINISHED_STOP_WORDS if any
//! sequence of the stop words is met
//! \param sequenceLengths input buffer [batchSize, beamWidth]. Current sequence
//! lengths of the request tokens.
//! \param stopWordsLen cumulative length of all stop words
//! \param batchSize batch size
//! \param beamWidth beam width
//! \param maxSeqLen maximum length of the sequence
//! \param stream stream
void invokeStopWordsCriterion(const int** outputIds, const int** parentIds, const int* stopWords,
    FinishedState* finished, const int* sequenceLengths, size_t stopWordsLen, int batchSize, int beamWidth,
    int maxSeqLen, cudaStream_t stream);

//! \brief Sets finished states based on the sequenceLimitLength and computes number of finished sequences in the batch.
//!
//! \param finished input/output buffer [batchSize, beamWidth]. Finished states. Set to
//! FinishedState::FINISHED_MAX_LENGTH if sequenceLengths >= sequenceLimitLength
//! \param finishedSum output buffer [1].
//! Total sum of finished requests
//! \param sequenceLimitLength input buffer [batchSize]. Maximum sequence length.
//! \param sequenceLengths input buffer [batchSize, beamWidth].
//! Current sequence lengths of the request tokens.
//! \param batchSize batch size
//! \param beamWidth beam width
//! \param stream stream
void invokeLengthCriterion(FinishedState* finished, int* finishedSum, const uint32_t* sequenceLimitLength,
    const int* sequenceLengths, int batchSize, int beamWidth, cudaStream_t stream);
} // namespace kernels
} // namespace tensorrt_llm
