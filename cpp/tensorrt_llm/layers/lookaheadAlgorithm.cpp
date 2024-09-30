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

#include "tensorrt_llm/layers/lookaheadAlgorithm.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"
#include "tensorrt_llm/runtime/lookaheadModule.h"
#include <tuple>

namespace tensorrt_llm::layers
{

using namespace tensorrt_llm::runtime;

void LookaheadAlgorithm::setup(TensorConstPtr const& prompt, SizeType32 w, SizeType32 n, SizeType32 g)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(w <= mMaxW, "lookahead requires setup w (%d) <= max_w (%d)", w, mMaxW);
    TLLM_CHECK_WITH_INFO(n <= mMaxN, "lookahead requires setup n (%d) <= max_n (%d)", n, mMaxN);
    TLLM_CHECK_WITH_INFO(g <= mMaxG, "lookahead requires setup g (%d) <= max_g (%d)", g, mMaxG);
    mW = w;
    mN = n;
    mG = g;
    std::tie(std::ignore, std::ignore, mRuntimeMaxDraftLen, std::ignore)
        = executor::LookaheadDecodingConfig(mW, mN, mG).calculateSpeculativeResource();

    mPoolManager.setup(mG);
    mPoolManager.accept(prompt, mN);
    mGoldenTokens = ITensor::slice(mGoldenTokensMax, 0, mN * 2 - 1);
    mPrefills = ITensor::slice(mPrefillsMax, 0, mN <= 1 ? 0 : mN - 2);
    mKeyTokens = ITensor::slice(mKeyTokensMax, 0, mW);
    mPastTokens = ITensor::slice(mPastTokensMax, 0, mW * (mN - 1));
    mPastTokens->reshape(ITensor::makeShape({mW, mN - 1}));

    BufferRange<TokenIdType const> promptRange(*prompt);
    BufferRange<TokenIdType> prefillRange(*mPrefills);
    BufferRange<TokenIdType> pastRange(*mPastTokens);
    BufferRange<TokenIdType> goldRange(*mGoldenTokens);
    auto randToken = [&promptRange](auto& item) { item = promptRange[rand() % promptRange.size()]; };
    std::for_each(prefillRange.begin(), prefillRange.end(), randToken);
    std::for_each(pastRange.begin(), pastRange.end(), [](auto& a) { a = -1; });
    for (SizeType32 i = 0; i < mW; i++)
    {
        if (mN - 1 > 0)
        {
            randToken(pastRange[i * (mN - 1)]);
        }
    }
    std::copy(std::prev(promptRange.end(), mN - 1), promptRange.end(), goldRange.begin());
    mGuessTokens = ITensor::slice(mGuessTokensMax, 0, 0);
    mFilling = (mN - 1) > 0 ? 1 : 0;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadAlgorithm::accept(TensorConstPtr const& generatedTokens)
{
    TLLM_CHECK(ITensor::volume(generatedTokens->getShape()) <= mN);
    BufferRange<TokenIdType const> generatedRange(*generatedTokens);
    BufferRange<TokenIdType> goldRange(*mGoldenTokens);
    auto genLen = generatedTokens->getShape().d[0];
    TLLM_CHECK(genLen <= mN);
    std::copy(generatedRange.begin(), generatedRange.end(), goldRange.begin() + mN - 1);
    TensorPtr newGold = ITensor::slice(mGoldenTokens, 0, mN - 1 + genLen);
    mPoolManager.accept(newGold, mN);
    std::copy(goldRange.begin() + genLen, goldRange.begin() + genLen + mN - 1, goldRange.begin());
}

//! lookahead has two phase, prefill the past tokens matrix and maintain past tokens matrix.
runtime::SizeType32 LookaheadAlgorithm::lookahead(TensorPtr const& draftTokens, TensorPtr const& positionIds,
    TensorPtr const& samplingMask, runtime::SizeType32 offset)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    SizeType32 prefill = mN - 2 - mFilling;
    SizeType32 len = prefill + mFilling * mW;
    TLLM_CHECK(len <= ITensor::volume(draftTokens->getShape()));
    TLLM_CHECK(len <= ITensor::volume(positionIds->getShape()));
    TLLM_CHECK(len <= ITensor::volume(samplingMask->getShape()));
    BufferRange<TokenIdType> prefillRange(*mPrefills);
    BufferRange<TokenIdType> pastRange(*mPastTokens);
    BufferRange<TokenIdType> draftRange(*draftTokens);
    PRINT_TOKENS(mPrefills);

    if (mFilling < mN - 1)
    { // prefilling
        std::copy(prefillRange.begin() + mFilling, prefillRange.end(), draftRange.begin());
        for (SizeType32 i = 0; i < mW; i++)
        {
            auto start = pastRange.begin() + i * (mN - 1);
            auto end = pastRange.begin() + i * (mN - 1) + mFilling;
            std::copy(start, end, draftRange.begin() + prefill + i * mFilling);
        }
    }
    else
    { // shift up
        std::copy(pastRange.begin() + 1, pastRange.begin() + mFilling * mW, draftRange.begin());
    }

    BufferRange<TokenIdType> positionIdsRange(*positionIds);
    BufferRange<bool> samplingMaskRange(*samplingMask);
    for (auto& v : samplingMaskRange)
    {
        v = 0;
    }
    SizeType32 idx = 0, wj = 0;
    auto fillPosition = [&positionIdsRange, &idx](SizeType32 start, SizeType32 len)
    {
        for (SizeType32 i = start; i < start + len; i++)
        {
            positionIdsRange[idx++] = i;
        }
    };
    if (prefill >= 0)
    {
        fillPosition(offset, prefill);
        for (wj = 0; wj < mW; wj++)
        {
            fillPosition(offset + prefill + wj, mFilling);
            samplingMaskRange[prefill + wj * mFilling + mFilling - 1] = true;
        }
    }
    else
    {
        fillPosition(offset, mFilling - 1);
        for (wj = 1; wj < mW; wj++)
        {
            fillPosition(offset - 1 + wj, mFilling);
            samplingMaskRange[wj * mFilling + mFilling - 1 - 1] = true;
        }
    }
    PRINT_VALUES(positionIds);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return len;
}

runtime::SizeType32 LookaheadAlgorithm::guess(TensorPtr const& guessTokens, TensorPtr const& guessIds,
    TensorPtr const& samplingMask, runtime::SizeType32 offset, runtime::TokenIdType lastToken)
{
    auto guesses = mPoolManager.guess(lastToken, mW);

    SizeType32 len = 0;
    std::for_each(guesses.begin(), guesses.end(), [&len](auto& a) { len += ITensor::volume(a->getShape()); });
    TLLM_CHECK(len <= ITensor::volume(guessTokens->getShape()));
    TLLM_CHECK(len <= ITensor::volume(guessIds->getShape()));
    TLLM_CHECK(len <= ITensor::volume(samplingMask->getShape()));
    BufferRange<TokenIdType> guessTokensRange(*guessTokens);
    BufferRange<SizeType32> guessIdsRange(*guessIds);
    BufferRange<bool> samplingMaskRange(*samplingMask);

    SizeType32 cur = 0;
    for (auto guess : guesses)
    {
        BufferRange<TokenIdType const> guessRange(*guess);
        std::copy(guessRange.begin(), guessRange.end(), guessTokensRange.begin() + cur);
        SizeType32 tmp = offset;
        std::for_each(
            guessIdsRange.begin() + cur, guessIdsRange.begin() + cur + mN - 1, [&tmp](auto& v) { v = tmp++; });
        cur += ITensor::volume(guess->getShape());
    }

    std::for_each(samplingMaskRange.begin(), samplingMaskRange.begin() + len, [](auto& a) { a = true; });

    return len;
}

void LookaheadAlgorithm::prepare(TensorPtr const& draftTokens, TensorPtr const& positionIds,
    TensorPtr const& samplingMask, TensorPtr const& length, TensorConstPtr const& offsetPtr,
    TensorConstPtr const& lastTokenPtr)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mRuntimeMaxDraftLen == 0)
    {
        (BufferRange<SizeType32>(*length))[0] = 0;
        return;
    }

    auto lastToken = BufferRange<TokenIdType const>(*lastTokenPtr)[0];
    auto offset = BufferRange<SizeType32 const>(*offsetPtr)[0];

    SizeType32 inputLen = ITensor::volume(draftTokens->getShape());
    TLLM_CHECK(inputLen >= mRuntimeMaxDraftLen);

    BufferRange<TokenIdType> draftRange(*draftTokens);
    BufferRange<TokenIdType> positionRange(*positionIds);
    BufferRange<bool> samplingRange(*samplingMask);

    SizeType32 filledLen = 0;

    filledLen += lookahead(ITensor::slice(draftTokens, filledLen, mRuntimeMaxDraftLen - filledLen),
        ITensor::slice(positionIds, filledLen, mRuntimeMaxDraftLen - filledLen),
        ITensor::slice(samplingMask, filledLen, mRuntimeMaxDraftLen - filledLen), offset);

    auto guessStart = filledLen;
    filledLen += guess(ITensor::slice(draftTokens, filledLen, mRuntimeMaxDraftLen - filledLen),
        ITensor::slice(positionIds, filledLen, mRuntimeMaxDraftLen - filledLen),
        ITensor::slice(samplingMask, filledLen, mRuntimeMaxDraftLen - filledLen), offset, lastToken);
    auto guessEnd = filledLen;

    mGuessTokens = ITensor::slice(mGuessTokensMax, 0, guessEnd - guessStart);

    std::copy(draftRange.begin() + guessStart, draftRange.begin() + guessEnd,
        BufferRange<TokenIdType>(*mGuessTokens).begin());

    (BufferRange<SizeType32>(*length))[0] = filledLen;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadAlgorithm::verify(TensorPtr const& accepted, TensorPtr const& acceptedOffsets,
    TensorPtr const& acceptedLength, TokenIdType newLastToken, TensorConstPtr const& goldenTokens,
    TensorConstPtr const& endToken)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(ITensor::volume(goldenTokens->getShape()) == ITensor::volume(mGuessTokens->getShape()));
    BufferRange<TokenIdType const> goldRange(*goldenTokens);
    BufferRange<TokenIdType> guessTokensRange(*mGuessTokens);
    auto guessSize = ITensor::volume(mGuessTokens->getShape());

    SizeType32 guesses = (mN - 1 > 0) ? (guessSize / (mN - 1)) : 0;
    SizeType32 hit = 0, maxHit = 0, hitIdx = 0;
    for (SizeType32 i = 0; i < guesses; i++)
    {
        SizeType32 hit = 0;
        for (SizeType32 j = 0; j < mN - 1; j++)
        {
            auto idx = i * (mN - 1) + j;
            bool ok
                = (j == 0) ? (newLastToken == guessTokensRange[idx]) : (goldRange[idx - 1] == guessTokensRange[idx]);
            bool finish = guessTokensRange[idx] == *BufferRange<TokenIdType const>(*endToken).begin();
            if (ok && !finish)
            {
                hit++;
            }
            else
            {
                break;
            }
        }
        if (hit > maxHit)
        {
            maxHit = hit;
            hitIdx = i;
        }
    }

    BufferRange<TokenIdType> acceptedRange(*accepted);
    acceptedRange[0] = newLastToken;
    std::copy(goldRange.begin() + hitIdx * (mN - 1), goldRange.begin() + hitIdx * (mN - 1) + maxHit,
        acceptedRange.begin() + 1);

    BufferRange<SizeType32> acceptedOffsetsRange(*acceptedOffsets);
    auto lookSize = 1 + mN - 2 - mFilling + mFilling * mW;
    // acceptedOffsetsRange[0] = 0;
    for (SizeType32 i = 0; i < maxHit; i++)
    {
        acceptedOffsetsRange[i] = lookSize + hitIdx * (mN - 1) + i - 1;
    }

    *BufferRange<SizeType32>(*acceptedLength).begin() = maxHit + 1;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

//! lookahead Jacobi matrix has prefilling phase and maintenance phase.
//! W=5, N=5.
//! *prefilling phase*
//! mFilling = 1->2, Tokens initialized from prompt. To fill the second line.
//! 0>1 2 3 *
//!       4 *
//!       5 *
//!       6 *
//!       7 *
//! mFilling = 2->3.
//! 0 1>2 3 4 *
//!       4 5 *
//!       5 6 *
//!       6 7 *
//!       7 8 *
//! mFilling = 3->4.
//! 0 1 2>3 4 5 *
//!       4 5 6 *
//!       5 6 7 *
//!       6 7 9 *
//!       7 8 a *
//! *maintenance phase*
//! mFilling = 4->4. shift up and generate five n-grams.
//! 0 1 2 3>4 5 6 *
//!       4 5 6 7 *
//!       5 6 7 8 *
//!       6 7 8 9 *
//!       7 8 9 a *
//! mFilling = 4.
//! 0 1 2 3 4>5 6 7 *
//!         5 6 7 8 *
//!         6 7 8 9 *
//!         7 8 9 a *
//!         8 9 a b *
//! mFilling = 4.
//! 0 1 2 3 4 5>6 7 8 *
//!           6 7 8 9 *
//!           7 8 9 a *
//!           8 9 a b *
//!           9 a b c *
void LookaheadAlgorithm::update(TensorPtr const& acceptedTokens, TensorPtr const& acceptedOffsets,
    TensorPtr const& acceptedLength, TensorConstPtr const& sampledTokens, TensorConstPtr const& endToken)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(ITensor::volume(acceptedTokens->getShape()) >= mN);
    BufferRange<TokenIdType const> sampledRange(*sampledTokens);
    BufferRange<TokenIdType> keyRange(*mKeyTokens);
    BufferRange<TokenIdType> pastRange(*mPastTokens);

    auto newLastToken = sampledRange[0];
    SizeType32 prefill = mN - 2 - mFilling;
    for (SizeType32 i = 0; i < mW; i++)
    {
        keyRange[i] = sampledRange[prefill + i * mFilling + mFilling];
    }

    if (mFilling < mN - 1)
    {
        for (SizeType32 i = 0; i < mW; i++)
        {
            pastRange[i * (mN - 1) + mFilling] = keyRange[i];
        }
    }
    else if (mN > 1)
    {
        for (SizeType32 i = 0; i < mW; i++)
        {
            auto begin = pastRange.begin() + i * (mN - 1);
            auto end = pastRange.begin() + i * (mN - 1) + mN - 1;
            auto key = *begin;
            std::copy(begin + 1, end, begin);
            *(std::prev(end, 1)) = keyRange[i];
            keyRange[i] = key;
        }
        keyRange[0] = newLastToken;
        mPoolManager.update(mKeyTokens, mPastTokens);
    }

    auto guessSize = ITensor::volume(mGuessTokens->getShape());
    auto outputSize = ITensor::volume(sampledTokens->getShape());
    auto lookSize = 1 + (mN > 1 ? mN - 2 : 0) - mFilling + mFilling * mW;
    TLLM_CHECK(guessSize + lookSize == outputSize);

    TensorConstPtr goldenTokens = ITensor::slice(sampledTokens, lookSize, guessSize);

    verify(acceptedTokens, acceptedOffsets, acceptedLength, newLastToken, goldenTokens, endToken);

    accept(ITensor::slice(acceptedTokens, 0, *BufferRange<SizeType32>(*acceptedLength).begin()));

    if (mFilling < mN - 1)
    {
        mFilling++;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::layers
