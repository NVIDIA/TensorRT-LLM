
#include "tensorrt_llm/layers/lookaheadAlgorithm.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"

namespace tensorrt_llm::layers
{

using namespace tensorrt_llm::runtime;

void LookaheadAlgorithm::setup(TensorPtr prompt)
{
    mPoolManager.clear();
    mPoolManager.fillWithPrompt(prompt, mN);
    auto promptRange = BufferRange<TokenIdType>(*prompt);
    auto prefillRange = BufferRange<TokenIdType>(*mPrefills);
    auto pastRange = BufferRange<TokenIdType>(*mPastTokens);
    auto goldRange = BufferRange<TokenIdType>(*mGoldenTokens);
    auto randToken = [&promptRange](auto& item) { item = promptRange[rand() % promptRange.size()]; };
    std::for_each(prefillRange.begin(), prefillRange.end(), randToken);
    std::for_each(pastRange.begin(), pastRange.end(), [](auto& a) { a = -1; });
    for (SizeType32 i = 0; i < mW; i++)
    {
        randToken(pastRange[i * (mN - 1)]);
    }
    std::copy(std::prev(promptRange.end(), mN - 1), promptRange.end(), goldRange.begin());
    mFilling = 1;
    PRINT_TOKENS(prompt);
    PRINT_TOKENS(mPrefills);
}

void LookaheadAlgorithm::accept(TensorPtr generatedTokens)
{
    TLLM_CHECK(ITensor::volume(generatedTokens->getShape()) <= mN);
    auto generatedRange = BufferRange<TokenIdType>(*generatedTokens);
    auto goldRange = BufferRange<TokenIdType>(*mGoldenTokens);
    auto genLen = generatedTokens->getShape().d[0];
    TLLM_CHECK(genLen <= mN);
    std::copy(generatedRange.begin(), generatedRange.end(), goldRange.begin() + mN - 1);
    TensorPtr newGold = ITensor::slice(mGoldenTokens, 0, mN - 1 + genLen);
    mPoolManager.fillWithPrompt(newGold, mN);
    std::copy(goldRange.begin() + genLen, goldRange.begin() + genLen + mN - 1, goldRange.begin());
}

//! lookahead has two phase, prefill the past tokens matrix and maintain past tokens matrix.
runtime::SizeType32 LookaheadAlgorithm::lookahead(TensorPtr draftTokens, TensorPtr positionIds,
    TensorPtr samplingMask, // outputs
    runtime::SizeType32 offset)
{
    SizeType32 prefill = mN - 2 - mFilling;
    SizeType32 len = prefill + mFilling * mW;
    TLLM_CHECK(len <= ITensor::volume(draftTokens->getShape()));
    TLLM_CHECK(len <= ITensor::volume(positionIds->getShape()));
    TLLM_CHECK(len <= ITensor::volume(samplingMask->getShape()));
    auto prefillRange = BufferRange<TokenIdType>(*mPrefills);
    auto pastRange = BufferRange<TokenIdType>(*mPastTokens);
    auto draftRange = BufferRange<TokenIdType>(*draftTokens);
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

    auto positionIdsRange = BufferRange<TokenIdType>(*positionIds);
    auto samplingMaskRange = BufferRange<bool>(*samplingMask);
    mBufferManager->setZero(*samplingMask);
    SizeType32 idx = 0, i = 0, j = 0;
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
        for (j = 0; j < mW; j++)
        {
            fillPosition(offset + prefill + j, mFilling);
            samplingMaskRange[prefill + j * mFilling + mFilling - 1] = true;
        }
    }
    else
    {
        fillPosition(offset, mFilling - 1);
        samplingMaskRange[mFilling - 1 - 1] = true;
        for (j = 1; j < mW; j++)
        {
            fillPosition(offset + j, mFilling);
            samplingMaskRange[j * mFilling + mFilling - 1 - 1] = true;
        }
    }
    return len;
}

runtime::SizeType32 LookaheadAlgorithm::guess(TensorPtr guessTokens, TensorPtr guessIds,
    TensorPtr samplingMask, // outputs
    runtime::SizeType32 offset, runtime::TokenIdType lastToken)
{
    auto guesses = mPoolManager.guess(lastToken, mW);

    SizeType32 len = 0;
    std::for_each(guesses.begin(), guesses.end(), [&len](auto& a) { len += ITensor::volume(a->getShape()); });
    TLLM_CHECK(len <= ITensor::volume(guessTokens->getShape()));
    TLLM_CHECK(len <= ITensor::volume(guessIds->getShape()));
    TLLM_CHECK(len <= ITensor::volume(samplingMask->getShape()));
    auto guessTokensRange = BufferRange<TokenIdType>(*guessTokens);
    auto guessIdsRange = BufferRange<SizeType32>(*guessIds);
    auto samplingMaskRange = BufferRange<bool>(*samplingMask);

    SizeType32 cur = 0;
    for (auto guess : guesses)
    {
        auto guessRange = BufferRange<TokenIdType>(*guess);
        std::copy(guessRange.begin(), guessRange.end(), guessTokensRange.begin() + cur);
        SizeType32 tmp = offset;
        std::for_each(
            guessIdsRange.begin() + cur, guessIdsRange.begin() + cur + mN - 1, [&tmp](auto& v) { v = tmp++; });
        cur += ITensor::volume(guess->getShape());
    }

    std::for_each(samplingMaskRange.begin(), samplingMaskRange.begin() + len, [](auto& a) { a = true; });

    return len;
}

void LookaheadAlgorithm::prepare(TensorPtr draftTokens, TensorPtr positionIds, TensorPtr samplingMask,
    TensorPtr length, // outputs
    TensorPtr offsetPtr, TensorPtr lastTokenPtr)
{
    TokenIdType const lastToken = BufferRange<TokenIdType const>(*lastTokenPtr)[0];
    SizeType32 const offset = BufferRange<SizeType32 const>(*offsetPtr)[0];

    SizeType32 inputLen = ITensor::volume(draftTokens->getShape());

    // mCurrentToken = lastToken;

    auto draftRange = BufferRange<TokenIdType>(*draftTokens);
    auto positionRange = BufferRange<TokenIdType>(*positionIds);
    TLLM_LOG_DEBUG("CAST BOOL");
    auto samplingRange = BufferRange<bool>(*samplingMask);

    TLLM_LOG_DEBUG("CAST BOOL");

    // draftRange[0] = lastToken;
    // positionRange[0] = offset;
    // samplingRange[0] = true;
    // SizeType32 filledLen = 1;
    SizeType32 filledLen = 0;

    filledLen += lookahead(ITensor::slice(draftTokens, filledLen, inputLen - filledLen),
        ITensor::slice(positionIds, filledLen, inputLen - filledLen),
        ITensor::slice(samplingMask, filledLen, inputLen - filledLen), offset);

    auto guessStart = filledLen;
    filledLen += guess(ITensor::slice(draftTokens, filledLen, inputLen - filledLen),
        ITensor::slice(positionIds, filledLen, inputLen - filledLen),
        ITensor::slice(samplingMask, filledLen, inputLen - filledLen), offset, lastToken);
    auto guessEnd = filledLen;

    mGuessTokens = ITensor::slice(mGuessTokensMax, 0, guessEnd - guessStart);
    std::copy(draftRange.begin() + guessStart, draftRange.begin() + guessEnd,
        BufferRange<TokenIdType>(*mGuessTokens).begin());

    (BufferRange<SizeType32>(*length))[0] = filledLen;
}

void LookaheadAlgorithm::verify(TensorPtr accepted, TensorPtr acceptedOffsets, TensorPtr acceptedLength, // outputs
    TokenIdType newLastToken, TensorPtr goldenTokens, TensorPtr endToken)
{
    TLLM_CHECK(ITensor::volume(goldenTokens->getShape()) == ITensor::volume(mGuessTokens->getShape()));
    auto goldRange = BufferRange<TokenIdType>(*goldenTokens);
    auto guessTokensRange = BufferRange<TokenIdType>(*mGuessTokens);
    auto guessSize = ITensor::volume(mGuessTokens->getShape());

    SizeType32 guesses = guessSize / (mN - 1), hit = 0, maxHit = 0, hitIdx = 0;
    for (SizeType32 i = 0; i < guesses; i++)
    {
        SizeType32 hit = 0;
        for (SizeType32 j = 0; j < mN - 1; j++)
        {
            auto idx = i * (mN - 1) + j;
            bool ok
                = (j == 0) ? (newLastToken == guessTokensRange[idx]) : (goldRange[idx - 1] == guessTokensRange[idx]);
            bool finish = guessTokensRange[idx] == *BufferRange<TokenIdType>(*endToken).begin();
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

    auto acceptedRange = BufferRange<TokenIdType>(*accepted);
    acceptedRange[0] = newLastToken;
    std::copy(goldRange.begin() + hitIdx * (mN - 1), goldRange.begin() + hitIdx * (mN - 1) + maxHit,
        acceptedRange.begin() + 1);

    auto acceptedOffsetsRange = BufferRange<SizeType32>(*acceptedOffsets);
    auto lookSize = 1 + mN - 2 - mFilling + mFilling * mW;
    acceptedOffsetsRange[0] = 0;
    for (SizeType32 i = 0; i < maxHit; i++)
    {
        acceptedOffsetsRange[1 + i] = lookSize + hitIdx * (mN - 1) + i;
    }

    *BufferRange<SizeType32>(*acceptedLength).begin() = maxHit + 1;
}

//! lookahead Jacobi matrix has prefilling phase and maintenance phase.
//! W=5, N=5.
//! *prefilling phase*
//! mFilling = 1->2, Tokens initialized from prompt. To fill the second line.
//! 0>1 2 3 4 5 6 7
//!       * * * * *
//! mFilling = 2->3.
//! 0 1>2 3 4 5 6 7
//!       4 5 6 7 8
//!       * * * * *
//! mFilling = 3->4.
//! 0 1 2>3 4 5 6 7
//!       4 5 6 7 8
//!       5 6 7 8 9
//!       * * * * *
//! *maintenance phase*
//! mFilling = 4->4. shift up and generate five n-grams.
//! 0 1 2 3>4 5 6 7
//!       4 5 6 7 8
//!       5 6 7 8 9
//!       6 7 8 9 a
//!       * * * * *
//! mFilling = 4.
//! 0 1 2 3 4>5 6 7 8
//!         5 6 7 8 9
//!         6 7 8 9 a
//!         7 8 9 a b
//!         * * * * *
//! mFilling = 4.
//! 0 1 2 3 4 5>6 7 8 9
//!           6 7 8 9 a
//!           7 8 9 a b
//!           8 9 a b c
//!           * * * * *
void LookaheadAlgorithm::update(TensorPtr acceptedTokens, TensorPtr acceptedOffsets,
    TensorPtr acceptedLength, // outputs
    TensorPtr sampledTokens, TensorPtr endToken)
{
    TLLM_CHECK(ITensor::volume(acceptedTokens->getShape()) >= mN);
    auto sampledRange = BufferRange<TokenIdType>(*sampledTokens);
    auto keyRange = BufferRange<TokenIdType>(*mKeyTokens);
    auto pastRange = BufferRange<TokenIdType>(*mPastTokens);

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
    else
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
    auto lookSize = 1 + mN - 2 - mFilling + mFilling * mW;
    TLLM_CHECK(guessSize + lookSize == outputSize);
    TensorPtr goldenTokens = ITensor::slice(sampledTokens, lookSize, guessSize);

    verify(acceptedTokens, acceptedOffsets, acceptedLength, newLastToken, goldenTokens, endToken);

    accept(ITensor::slice(acceptedTokens, 0, *BufferRange<SizeType32>(*acceptedLength).begin()));

    if (mFilling < mN - 1)
    {
        mFilling++;
    }
}

} // namespace tensorrt_llm::layers
