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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/types.h"
#include <cstdint>
#include <curand_kernel.h>

namespace tensorrt_llm::kernels
{

class FinishedState
{
public:
    static auto constexpr empty()
    {
        return FinishedState{0};
    }

    static auto constexpr finished()
    {
        return FinishedState{kFinished};
    }

    static auto constexpr skipDecoding()
    {
        return FinishedState{kSkipDecoding};
    }

    static auto constexpr finishedEOS()
    {
        return FinishedState{kFinishedEos};
    }

    static auto constexpr finishedMaxLength()
    {
        return FinishedState{kFinishedMaxLength};
    }

    static auto constexpr finishedStopWords()
    {
        return FinishedState{kFinishedStopWords};
    }

    __host__ __device__ void constexpr setFinishedEOS()
    {
        mState |= kFinishedEos;
    }

    __host__ __device__ bool constexpr isFinishedEOS() const
    {
        return anyBitSet(kFinishedEos);
    }

    __host__ __device__ void constexpr setFinishedStopWords()
    {
        mState |= kFinishedStopWords;
    }

    __host__ __device__ bool constexpr isFinishedStopWords() const
    {
        return anyBitSet(kFinishedStopWords);
    }

    __host__ __device__ void constexpr setFinishedMaxLength()
    {
        mState |= kFinishedMaxLength;
    }

    __host__ __device__ bool constexpr isFinishedMaxLength() const
    {
        return anyBitSet(kFinishedMaxLength);
    }

    __host__ __device__ void constexpr setFinished()
    {
        mState |= kFinished;
    }

    __host__ __device__ bool constexpr isFinished() const
    {
        return anyBitSet(kFinished);
    }

    __host__ __device__ void constexpr setSkipDecoding()
    {
        mState |= kSkipDecoding;
    }

    __host__ __device__ bool constexpr isSkipDecoding() const
    {
        return anyBitSet(kSkipDecoding);
    }

    [[nodiscard]] constexpr executor::FinishReason toFinishReason() const
    {
        if (isFinishedEOS())
        {
            return executor::FinishReason::kEND_ID;
        }
        if (isFinishedStopWords())
        {
            return executor::FinishReason::kSTOP_WORDS;
        }
        if (isFinishedMaxLength())
        {
            return executor::FinishReason::kLENGTH;
        }
        return executor::FinishReason::kNOT_FINISHED;
    }

    using UnderlyingType = uint8_t;

    [[nodiscard]] constexpr UnderlyingType toUnderlying() const noexcept
    {
        return mState;
    }

private:
    // The default state is interpreted as not finished.
    __host__ __device__ constexpr FinishedState(UnderlyingType state)
        : mState(state)
    {
    }

    // Request has finished based on the generation of EOS token
    static UnderlyingType constexpr kFinishedEos{1u << 0};
    // Request has finished based on the generation of stop words
    static UnderlyingType constexpr kFinishedStopWords{1u << 1};
    // Request has finished based on reaching max sequence length
    static UnderlyingType constexpr kFinishedMaxLength{1u << 2};
    // Finished by any condition
    static UnderlyingType constexpr kFinished{kFinishedEos | kFinishedStopWords | kFinishedMaxLength};
    // Skip decoding. E.g. used for not accepted tokens in speculative decoding
    static UnderlyingType constexpr kSkipDecoding{1u << 3};

    __host__ __device__ bool constexpr anyBitSet(UnderlyingType bits) const
    {
        return (mState & bits) != 0;
    }

    UnderlyingType mState{};
};

static_assert(!FinishedState::empty().isFinished());
static_assert(!FinishedState::empty().isSkipDecoding());
static_assert(FinishedState::finished().isFinished());
static_assert(FinishedState::skipDecoding().isSkipDecoding());
static_assert(FinishedState::finishedEOS().isFinishedEOS());
static_assert(FinishedState::finishedStopWords().isFinishedStopWords());
static_assert(FinishedState::finishedMaxLength().isFinishedMaxLength());

template <typename T>
struct ScatterDecodingParamEntry
{
    // Contiguous values to scatter
    T const* mVector;
    // Value used to scatter if mVector is nullptr
    T mScalar;
    // Target base address to scatter
    T* mTarget;

    ScatterDecodingParamEntry() = default;

    ScatterDecodingParamEntry(T const* vector, T scalar, T* target)
        : mVector(vector)
        , mScalar(scalar)
        , mTarget(target)
    {
    }

    ScatterDecodingParamEntry(void const* vector, T scalar, T* target)
        : ScatterDecodingParamEntry(static_cast<T const*>(vector), scalar, target)
    {
    }
};

//! \brief Initialize batchSize curand states with given seed.
//!
//! \param state output buffer [maxBatchSize]. Curand states to be initialized
//! \param batchSlots input buffer[batchSize], optional. Indices of rows of data in memory pool
//! \param batchSize number of states to initialize
//! \param randomSeed seed to initialize states
//! \param stream stream
void invokeCurandInitialize(
    curandState_t* state, int const* batchSlots, size_t const batchSize, uint64_t randomSeed, cudaStream_t stream);

//! \brief Initialize batchSize curand states with given seed per request.
//!
//! \param state output buffer [maxBatchSize] of curand states to be initialized
//! \param batchSlots input buffer[batchSize], optional. Indices of rows of data in memory pool
//! \param batchSize number of states to initialize
//! \param randomSeeds input buffer [maxBatchSize] with seeds
//! \param stream stream
void invokeCurandBatchInitialize(curandState_t* states, int const* batchSlots, size_t const batchSize,
    uint64_t const* randomSeeds, cudaStream_t stream);

template <typename T>
struct BiasSoftmaxParams
{
    //! input/output buffer [maxBatchSize, vocabSize]. Logits to be modified by mask and bias.
    //! If nullptr, logitsPtrs has to be provided.
    T* logits{nullptr};
    //! input/output buffer [maxBatchSize][maxBeamWidth, vocabSize] or
    //! [maxBatchSize, maxBeamWidth][vocabSize] if ptrsForBeams is true.
    //! Vector of pointers to the logits.
    //! If nullptr, logits has to be provided.
    T** logitsPtrs{nullptr};
    //! output buffer [maxBatchSize, vocabSize]. Probabilities of logits compute by softmax.
    //! Can be the same pointer as logits
    T* probs{nullptr};
    //! output buffer [maxBatchSize], optional. Entropy of the computed probs distribution.
    //! When specified, skipSoftMax must be false and probs must be specified.
    float* outputEntropy{nullptr};
    //! input buffer [vocabSize], optional. Bias to logit per token. Ignored if nullptr.
    T const* bias{nullptr};
    //! input buffer [batchSize], optional. Temperature per logit. Ignored if nullptr.
    float const* temperatures{nullptr};
    //! input buffer [maxBatchSize], optional. EOS token ids per request
    int32_t const* endIds{nullptr};
    //! input buffer [maxBatchSize], optional.
    //! Flag is set to true if request has finished the generation
    FinishedState const* finished{nullptr};
    //! input buffer [maxBatchSize], optional. Actual width of the beam per request.
    int32_t const* beamWidths{nullptr};
    //! input buffer[batchSize], optional. Indices of rows of data in memory pool
    int32_t const* batchSlots{nullptr};
    //! input buffer[batchSize], optional. min_p values per request
    float* minPs{nullptr};
    //! current batch size
    int32_t batchSize{0};
    //! max batch size
    int32_t maxBatchSize{0};
    //! max beam width
    int32_t maxBeamWidth{0};
    //! unpadded vocab size
    int32_t vocabSize{0};
    //! padded vocab size
    int32_t vocabSizePadded{0};
    //! flag to skip softmax computation
    bool skipSoftMax{false};
    //! flag to use batchSlot as index for logits and probs
    bool batchSlotsLogits{false};
    //! flag to indicate the layout of logitsPtrs
    bool ptrsForBeams{false};
    //! input buffer [maxBatchSize]. Flags whether to skip decoding per request
    bool const* skipDecode{nullptr};

    void checkParams()
    {
        TLLM_CHECK(logits || logitsPtrs);
        TLLM_CHECK(((outputEntropy != nullptr) && (probs != nullptr)) || (outputEntropy == nullptr));
        TLLM_CHECK(((outputEntropy != nullptr) && !skipSoftMax) || (outputEntropy == nullptr));

        if (batchSlotsLogits)
        {
            TLLM_CHECK(batchSlots);
        }

        if (ptrsForBeams)
        {
            TLLM_CHECK(logitsPtrs);
        }

        TLLM_CHECK(batchSize > 0);
        TLLM_CHECK(maxBatchSize > 0);
        TLLM_CHECK(batchSize <= maxBatchSize);
        TLLM_CHECK(maxBeamWidth > 0);
        TLLM_CHECK(vocabSize > 0);
        TLLM_CHECK(vocabSizePadded > 0);
        TLLM_CHECK(vocabSize <= vocabSizePadded);
    }
};

//! \brief Applies mask, applies temperature, adds bias to logits and computes softmax values.
//! Sets -MAX_FLT value for tokens in range [vocabSize; vocabSizePadded) to prevent them from being chosen.
//! If request finished the generation, sets MAX_FLT to endId token and -MAX_FLT to all other tokens forcing to choose
//! endId token. Otherwise, adds bias per token if bias pointer is not nullptr.
//! Computes entropy if outputEntropy is not nullptr.
//! \param stream stream
template <typename T>
void invokeAddBiasSoftMax(BiasSoftmaxParams<T> const params, cudaStream_t stream);

//! \brief Distributes values located in src to dst according to the indieces from batchSlots
//!
//! \param src input buffer [batchSize], optional.
//! \param scalar value used if src is nullptr.
//! \param dst output buffer [maxBatchSize].
//! \param batchSlots input buffer [batchSize]. Indices of rows of data in memory pool
//! \param batchSize batch size
//! \param stream stream
template <typename T>
void invokeScatterDecodingParams(
    T const* src, T scalar, T* dst, int const* batchSlots, int batchSize, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
