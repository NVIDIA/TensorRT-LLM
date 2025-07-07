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

#include "lookaheadPoolManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <cstddef>
#include <memory>
#include <tuple>

namespace tensorrt_llm::layers
{

//! @brief An CPU implementation of Lookahead with ITensor.
class LookaheadAlgorithm
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorConstPtr = runtime::ITensor::SharedConstPtr;
    using SizeType32 = runtime::SizeType32;
    using TokenIdType = runtime::TokenIdType;

    //! @brief Currently the resource management is to be aligned with batch manager.
    //! @param w, n, g is the Jacobi window, n-gram level and guess set size respectively.
    //! wili: argument `id` is useless?
    LookaheadAlgorithm(
        runtime::SizeType32 maxW, runtime::SizeType32 maxN, runtime::SizeType32 maxG, runtime::SizeType32 id = 0);

    //! @brief setup per request, fill internal states from @param prompt.
    void setup(TensorConstPtr const& prompt, runtime::SizeType32 w, runtime::SizeType32 n, runtime::SizeType32 g,
        uint64_t seed);

    //! @brief accept the new generated tokens.
    //! LookaheadDecodingLayer need call once for the first token in generation phase.
    void accept(TensorConstPtr const& generatedTokens);

    //! @brief combine lookahead and guess to prepare the tensors.
    //! input @param lastPositionIdPtr is position id of the last golden token, in a TensorPtr.
    //! input @param lastTokenPtr the last golden token for searching in the pool, in a TensorPtr.
    //! output @param draftTokens, positionIds includes the lookahead and the verification branch information.
    //! output @param draftLengthPtr holds the draft tokens length.
    //! output @param attentionMask holds the draft tokens dependency mask, and attentionMaskOffset is the index offset
    //! in attentionMask.
    void prepare(TensorPtr const& draftTokens, TensorPtr const& positionIds, TensorPtr const& draftLengthPtr,
        TensorPtr const& attentionMask, SizeType32 const offset, TokenIdType const lastToken);

    //! @brief update the internal states and generate accepted tokens from @param outputTokens.
    //! input @param sampledTokens is the all the tokens from the language model.
    //! input @param endToken is the end token for `verify` early quit.
    //! output @param acceptedTokens, acceptedOffsets in @param acceptedLength.
    void update(TensorPtr const& acceptedTokens, TensorPtr const& acceptedOffsets, TensorPtr const& acceptedLength,
        TensorConstPtr const& sampledTokens, TensorConstPtr const& endToken);

    //! generate attention @param mask from @param posIds.
    static void posIdsToMask(TensorPtr const& mask, TensorConstPtr const& posIds);

    //! inplace encode the @param tokens and @param posIds according to attention @param masks, and record the offsets
    //! in @param encodeMap.
    static runtime::SizeType32 treeEncode(
        TensorPtr const& tokens, TensorPtr const& posIds, TensorPtr const& masks, TensorPtr const& encodeMap);

    //! @brief print content of the LookaheadAlgorithm
    void print(char const* functionName, int const lineNumber) const noexcept;

private:
    //! @brief generate lookahead branch information.
    //! input @param startPosId is the first position id of the draftTokens.
    //! output @param draftTokens, positionIds of the lookahead branch.
    //! @return the actual filled lookahead length.
    runtime::SizeType32 lookahead(
        TensorPtr const& draftTokens, TensorPtr const& positionIds, runtime::SizeType32 startPosId);

    //! @brief generate verification branch information. Also save the guessed tokens for future verification.
    //! input @param startPosId the first position id.
    //! input @param lastToken the last golden token for searching in the pool.
    //! output @param guessTokens, guessIds of the verification branch.
    //! @return the actual filled guess length.
    runtime::SizeType32 guess(TensorPtr const& guessTokens, TensorPtr const& guessIds, runtime::SizeType32 startPosId,
        runtime::TokenIdType lastToken);

    //! @brief verify the guessed tokens results and generate the longest accepted tokens.
    //! input @param newLastToken is the new-generated last golden token.
    //! input @param sampledTokens is the generated token results from the language model.
    //! input @param endToken is the end token for early quit detection.
    //! output @param accepted in @param acceptedLength, including the first golden one.
    //! output @param acceptedOffsets is the offsets of draft tokens, excluding the first golden one.
    void verify(TensorPtr const& accepted, TensorPtr const& acceptedOffsets, TensorPtr const& acceptedLength,
        runtime::TokenIdType newLastToken, TensorConstPtr const& sampledTokens, TensorConstPtr const& endToken);

private:
    LookaheadPoolManager mPoolManager;
    //! Random prefill tokens, shapes are [mMaxN-2] and [mN-2] respectively
    TensorPtr mPrefillsMax;
    TensorPtr mPrefills;
    //! Look ahead branch window, [mMaxW * (mMaxN-1)] and [mW, (mN-1)]
    TensorPtr mPastTokensMax;
    TensorPtr mPastTokens;
    //! Shifted mPastTokens as key tokens, [mMaxW] and [mW]
    TensorPtr mKeyTokensMax;
    TensorPtr mKeyTokens;
    //! all the moving tail golden tokens, [mMaxN*2-1] and [mN*2-1]
    TensorPtr mGoldenTokensMax;
    TensorPtr mGoldenTokens;
    //! Tokens provided by `guess` and used in `verify`, [mMaxG*(mMaxN-1)] and [mG*(mN-1)]
    TensorPtr mGuessTokensMax;
    TensorPtr mGuessTokens;
    //! All draft tokens (mPastTokens + mGuessTokens), [maxDraftLen]
    TensorPtr mDraftTokensMax;
    TensorPtr mDraftTokens;
    //! [maxDraftLen, maxDraftLen]
    TensorPtr mAttentionMask;
    //! [maxDraftLen]
    TensorPtr mEncodeMapMax;
    TensorPtr mEncodeMap;
    //! [maxGeneratedLen]
    TensorPtr mSampledTokensMax;
    TensorPtr mSampledTokens;

    //! look ahead algorithm parameters, Window size, Level and Guess set size.
    //! max for reserving resources and current for current request.
    runtime::SizeType32 const mMaxW{0};
    runtime::SizeType32 const mMaxN{0};
    runtime::SizeType32 const mMaxG{0};
    runtime::SizeType32 mW{0};
    runtime::SizeType32 mN{0};
    runtime::SizeType32 mG{0};
    runtime::SizeType32 mRuntimeMaxDraftLen{0};
    runtime::SizeType32 mRuntimeMaxDraftPathLen{0};
    //! in prefilling mode when mFilling < mN-1.
    runtime::SizeType32 mFilling;
};

} // namespace tensorrt_llm::layers
