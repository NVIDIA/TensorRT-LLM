/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/speculativeDecoding/common.h"
#include "tensorrt_llm/runtime/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tensorrt_llm::kernels::speculative_decoding
{

template <typename T>
struct ExtractExplicitDraftTokensParams
{
    //! [maxBatchSize, maxSeqLen]
    runtime::TokenIdType* outputIds{nullptr};
    //! [maxBatchSize]
    runtime::SizeType32* outputPositionIdsBase{nullptr};
    //! [maxBatchSize, maxDecodingTokens]
    runtime::SizeType32* outputPositionIds{nullptr};
    //! [maxBatchSize, maxDecodingDraftTokens]
    runtime::TokenIdType* outputNextDraftTokens{nullptr};
    //! [maxBatchSize, maxNumPaths, maxPathLength]
    runtime::TokenIdType* unpackedNextDraftTokens{nullptr};
    //! [maxBatchSize, maxNumPaths, maxPathLength]
    runtime::SizeType32* unpackedNextDraftIndices{nullptr};
    //! [maxBatchSize]
    runtime::SizeType32* acceptedLengths{nullptr};
    //! [maxBatchSize]
    runtime::SizeType32* nextDraftLengths{nullptr};
    //! [maxBatchSize]
    runtime::SizeType32* sequenceLengths{nullptr};
    //! [maxBatchSize]
    T* randDataSample{nullptr};
    //! [maxBatchSize, maxNumPaths, maxPathDraftLength]
    T* randDataVerification{nullptr};
    //! [maxBatchSize, maxNumPaths, maxPathDraftLength, maxVocabSize]
    T* outputDraftProbs{nullptr};
    //! [maxBatchSize]
    T* outputTemperatures{nullptr};
    //! [forwardBatchSize]
    runtime::SizeType32 const* batchSlots{nullptr};
    //! [forwardBatchSize, maxNumPaths, maxPathLength]
    runtime::TokenIdType const* nextDraftTokens{nullptr};
    //! [forwardBatchSize, maxNumPaths, maxPathLength]
    runtime::TokenIdType const* lastDraftTokens{nullptr};
    //! [forwardBatchSize, maxNumPaths, maxPathLength]
    runtime::SizeType32 const* inputUnpackedNextDraftIndices{nullptr};
    //! [forwardBatchSize]
    runtime::SizeType32 const* bestPathLengths{nullptr};
    //! [forwardBatchSize]
    runtime::SizeType32 const* bestPathIndices{nullptr};
    //! [forwardBatchSize]
    runtime::SizeType32 const* inputPositionIdsBase{nullptr};
    //! [forwardBatchSize * maxDecodingTokens]
    runtime::SizeType32 const* packedPositionIds{nullptr};
    //! [forwardBatchSize * maxDecodingTokens]
    runtime::TokenIdType const* nextFlatTokens{nullptr};
    //! [forwardBatchSize]
    runtime::SizeType32 const* generationLengthInclusiveSum{nullptr};
    //! [forwardBatchSize, maxNumPaths, maxPathDraftLength, maxVocabSize]
    T const* nextDraftProbs{nullptr};
    //! [maxBatchSize]
    float const* inputTemperatures{nullptr};
    //! [maxBatchSize]
    curandState_t* curandState{nullptr};
    runtime::SizeType32 batchSize;
    runtime::SizeType32 numPaths;
    runtime::SizeType32 maxPathLength;
    runtime::SizeType32 maxSeqLen;
    runtime::SizeType32 vocabSize;
};

//! @brief Modifies `outputIds` and `sequenceLengths` according to the accepted tokens
//! derived from `nextDraftTokens`, `lastDraftTokens`, `inputUnpackedNextDraftIndices`, `bestPathIndices` and
//! `bestPathLengths`. Sets new draft tokens `outputNextDraftTokens` and their lengths `nextDraftLengths`. Splits input
//! tensors mapped lienarly from ExplicitDraftTokens network into respective outputs at batch slots. `nextDraftTokens`
//! -> `unpackedNextDraftTokens` `inputUnpackedNextDraftIndices` -> `unpackedNextDraftIndices` `packedPositionIds` ->
//! `outputPositionIds` Generates random data for `randDataSample` and `randDataVerification`.
template <typename T>
void invokeExtractExplicitDraftTokens(ExtractExplicitDraftTokensParams<T> const& params, cudaStream_t stream);

//! @brief Copies linear draft probs from linear batch index at `nextDraftProbs` to `outputDraftProbs` at `batchSlot`
//! batch indices.
template <typename T>
void invokeCopyProbs(ExtractExplicitDraftTokensParams<T> const& params, cudaStream_t stream);

template <typename T>
struct PackExplicitDraftTokensParams
{
    //! [forwardBatchSize]
    runtime::SizeType32 const* batchSlots{nullptr};
    //! [forwardBatchSize]
    runtime::SizeType32 const* cumSumGenerationLengths{nullptr};
    //! [1]
    runtime::SizeType32 const* maxGenerationLength{nullptr};

    //! [forwardBatchSize]
    runtime::SizeType32* outputPositionIdsBase{nullptr};
    //! [maxBatchSize]
    runtime::SizeType32 const* inputPositionIdsBase{nullptr};

    //! [forwardBatchSize]
    runtime::SizeType32* outputGenerationLengths{nullptr};
    //! [maxBatchSize]
    runtime::SizeType32 const* inputGenerationLengths{nullptr};

    //! [forwardBatchSize]
    T* outputRandomDataSample{nullptr};
    //! [maxBatchSize]
    T const* inputRandomDataSample{nullptr};

    //! [forwardBatchSize, maxNumPaths, maxPathDraftLength]
    T* outputRandomDataValidation{nullptr};
    //! [maxBatchSize, maxNumPaths, maxPathDraftLength]
    T const* inputRandomDataValidation{nullptr};

    //! [forwardBatchSize, maxNumPaths, maxPathLength]
    runtime::TokenIdType* outputNextDraftTokens{nullptr};
    //! [maxBatchSize, maxNumPaths, maxPathLength]
    runtime::TokenIdType const* inputNextDraftTokens{nullptr};

    //! [forwardBatchSize, maxNumPaths, maxPathLength]
    runtime::SizeType32* outputNextDraftIndices{nullptr};
    //! [maxBatchSize, maxNumPaths, maxPathLength]
    runtime::SizeType32 const* inputNextDraftIndices{nullptr};

    //! [maxBatchSize, maxGenerationLength, divUp(maxGenerationLength, 32)]
    int32_t* outputPackedMask{nullptr};
    //! [forwardBatchSize, maxGenerationLength, divUp(maxGenerationLength, 32)]
    int32_t const* inputPackedMask{nullptr};

    //! [forwardBatchSize, maxGenerationLength]
    runtime::SizeType32* outputPositionOffsets{nullptr};
    //! [maxBatchSize, maxGenerationLength]
    runtime::SizeType32 const* inputPositionOffsets{nullptr};

    //! [forwardBatchSize, maxNumPaths, maxPathDraftLength, maxVocabSize]
    T* outputDraftProbs{nullptr};
    //! [maxBatchSize, maxNumPaths, maxPathDraftLength, maxVocabSize]
    T const* inputDraftProbs{nullptr};

    //! [forwardBatchSize]
    T* outputTemperatures{nullptr};
    //! [maxBatchSize]
    T const* inputTemperatures{nullptr};

    runtime::SizeType32 batchSize;
    runtime::SizeType32 numPaths;
    runtime::SizeType32 maxPathLength;
    runtime::SizeType32 vocabSize;
};

//! @brief Copy all rows at `batchSlots[batchIdx]` from `input*` tensors to `batchIdx` rows at `output*` tensor.
template <typename T>
void invokePackExplicitDraftTokens(PackExplicitDraftTokensParams<T> const& params, cudaStream_t stream);

//! @brief Copies draft probs from `batchSlot` rows to linear batch index. From `inputDraftProbs` to `outputDraftProbs`.
template <typename T>
void invokeCopyProbs(PackExplicitDraftTokensParams<T> const& params, cudaStream_t stream);

size_t invokeScanSpecDecodingGenerationLengths(void* __restrict__ reduceMaxTempStorage, size_t reduceTempStorageBytes,
    runtime::SizeType32 const* __restrict__ specDecodingGenerationLengths,
    runtime::SizeType32* __restrict__ scannedSpecDecodingGenerationLengths, runtime::SizeType32 batchSize,
    cudaStream_t stream);
size_t invokeReduceMaxSpecDecodingGenerationLengths(void* __restrict__ reduceMaxTempStorage,
    size_t reduceTempStorageBytes, runtime::SizeType32 const* __restrict__ specDecodingGenerationLengths,
    runtime::SizeType32* __restrict__ scannedSpecDecodingGenerationLengths, runtime::SizeType32 batchSize,
    cudaStream_t stream);

// inclusive prefix sum specDecodingGenerationLengths
void invokeScanReduceSpecDecodingGenerationLengths(runtime::SizeType32 batchSize,
    runtime::SizeType32 const* __restrict__ specDecodingGenerationLengths, void* __restrict__ scanTempStorage,
    size_t scanTempStorageBytes, runtime::SizeType32* __restrict__ scanedSpecDecodingGenerationLengths,
    void* __restrict__ reduceMaxTempStorage, size_t reduceMaxTempStorageBytes,
    runtime::SizeType32* maxSpecDecodingGenerationLengths, cudaStream_t stream);

void invokeConvertSpecDecodingMaskToPackedMask(runtime::SizeType32 batchSize,
    runtime::SizeType32 const* __restrict__ specDecodingCumGenerationLengths,
    runtime::SizeType32 const* __restrict__ specDecodingMaxGenerationLengths, bool const* __restrict__ specDecodingMask,
    runtime::SizeType32 const* __restrict__ batchSlots, runtime::SizeType32 maxDraftTokens,
    runtime::SizeType32 maxGenerationLength, runtime::SizeType32* __restrict__ specDecodingPackedMask,
    cudaStream_t stream);

} // namespace tensorrt_llm::kernels::speculative_decoding
