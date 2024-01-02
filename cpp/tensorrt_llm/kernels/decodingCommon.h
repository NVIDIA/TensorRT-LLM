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

#include <cstdint>
#include <curand_kernel.h>

namespace tensorrt_llm
{
namespace kernels
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

    __host__ __device__ bool constexpr isFinishedEOS()
    {
        return anyBitSet(kFinishedEos);
    }

    __host__ __device__ void constexpr setFinishedStopWords()
    {
        mState |= kFinishedStopWords;
    }

    __host__ __device__ bool constexpr isFinishedStopWords()
    {
        return anyBitSet(kFinishedStopWords);
    }

    __host__ __device__ void constexpr setFinishedMaxLength()
    {
        mState |= kFinishedMaxLength;
    }

    __host__ __device__ bool constexpr isFinishedMaxLength()
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
        mState = kSkipDecoding;
    }

    __host__ __device__ bool constexpr isSkipDecoding() const
    {
        return anyBitSet(kSkipDecoding);
    }

    using UnderlyingType = uint8_t;

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

//! \brief Initialize batchSize curand states with given seed.
//!
//! \param state output buffer [batchSize]. Curand states to be initialized
//! \param batchSize number of states to initialize
//! \param randomSeed seed to initialize states
//! \param stream stream
void invokeCurandInitialize(curandState_t* state, const size_t batchSize, uint64_t randomSeed, cudaStream_t stream);

//! \brief Initialize batchSize curand states with given seed per request.
//!
//! \param state output buffer [batchSize] of curand states to be initialized
//! \param batchSize number of states to initialize
//! \param randomSeeds input buffer [batchSize] with seeds
//! \param stream stream
void invokeCurandBatchInitialize(
    curandState_t* states, const size_t batchSize, const uint64_t* randomSeeds, cudaStream_t stream);

//! \brief Applies mask, adds bias to logits and computes softmax values.
//! Sets -MAX_FLT value for tokens in range [vocabSize; vocabSizePadded) to prevent them from being chosen.
//! If request finished the generation, sets MAX_FLT to endId token and -MAX_FLT to all other tokens forcing to choose
//! endId token. Otherwise, adds bias per token if bias pointer is not nullptr.
//!
//! \param logits input/output buffer [batchSize, vocabSize]. Logits to be modified by mask and bias.
//! \param probs output buffer [batchSize, vocabSize]. Probabilities of logits compute by softmax.
//! Can be the same pointer as logits
//! \param bias input buffer [vocabSize]. Bias to logit per token. Ignored if nullptr
//! \param endIds input buffer [batchSize]. EOS token ids per request
//! \param finished input buffer [batchSize] with flags set to true if request has finished the generation
//! \param batchSize batch size
//! \param vocabSize unpadded vocab size
//! \param vocabSizePadded padded vocab size
//! \param stream stream
template <typename T>
void invokeAddBiasSoftMax(T* logits, T* probs, const T* bias, const int* endIds, const FinishedState* finished,
    const int batchSize, const int vocabSize, const int vocabSizePadded, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
