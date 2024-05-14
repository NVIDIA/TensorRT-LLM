
#include "tensorrt_llm/layers/lookaheadAlgorithm.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"

namespace tensorrt_llm::layers
{

using namespace tensorrt_llm::runtime;

void LookaheadAlgorithm::setup(TensorPtr prompt)
{
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
    PRINT_TOKENS2D(mPastTokens);
}

//! lookahead has two phase, prefill the past tokens matrix and maintain past tokens matrix.
std::tuple<LookaheadAlgorithm::TensorPtr, LookaheadAlgorithm::TensorPtr, LookaheadAlgorithm::TensorPtr>
LookaheadAlgorithm::lookahead(SizeType32 offset)
{
    SizeType32 prefill = mN - 2 - mFilling;
    SizeType32 len = prefill + mFilling * mW;
    TensorPtr inputTokens = BufferManager::cpu(ITensor::makeShape({len}), nvinfer1::DataType::kINT32);
    auto prefillRange = BufferRange<TokenIdType>(*mPrefills);
    auto pastRange = BufferRange<TokenIdType>(*mPastTokens);
    auto inputRange = BufferRange<TokenIdType>(*inputTokens);

    if (mFilling < mN - 1)
    { // prefilling
        std::copy(prefillRange.begin() + mFilling, prefillRange.end(), inputRange.begin());
        for (SizeType32 i = 0; i < mW; i++)
        {
            auto start = pastRange.begin() + i * (mN - 1) + prefill;
            auto end = pastRange.begin() + i * (mN - 1) + prefill + mFilling;
            std::copy(start, end, inputRange.begin() + i * mFilling);
        }
    }
    else
    { // shift up
        std::copy(pastRange.begin() + 1, pastRange.begin() + mFilling * mW, inputRange.begin());
    }

    TensorPtr positionIds = mBufferManager->cpu(ITensor::makeShape({len}), nvinfer1::DataType::kINT32);
    TensorPtr samplingMask = mBufferManager->cpu(ITensor::makeShape({len}), nvinfer1::DataType::kINT32);
    auto positionIdsRange = BufferRange<TokenIdType>(*positionIds);
    auto samplingMaskRange = BufferRange<TokenIdType>(*samplingMask);
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
        fillPosition(offset + 1, prefill);
        for (j = 0; j < mW; j++)
        {
            fillPosition(offset + 1 + prefill + j, mFilling);
            samplingMaskRange[prefill + j * mFilling + mFilling - 1] = true;
        }
    }
    else
    {
        fillPosition(offset + 1, mFilling - 1);
        samplingMaskRange[mFilling - 1 - 1] = true;
        for (j = 1; j < mW; j++)
        {
            fillPosition(offset + 1 + j, mFilling);
            samplingMaskRange[j * mFilling + mFilling - 1 - 1] = true;
        }
    }

    return std::make_tuple(inputTokens, positionIds, samplingMask);
}

std::tuple<LookaheadAlgorithm::TensorPtr, LookaheadAlgorithm::TensorPtr> LookaheadAlgorithm::guess(
    SizeType32 offset, TokenIdType lastToken)
{
    auto guesses = mPoolManager.guess(lastToken, mW);

    SizeType32 len = 0;
    std::for_each(guesses.begin(), guesses.end(), [&len](auto& a) { len += ITensor::volume(a->getShape()); });
    TensorPtr guessTokens = mBufferManager->cpu(ITensor::makeShape({len}), nvinfer1::DataType::kINT32);
    auto guessTokensRange = BufferRange<TokenIdType>(*guessTokens);
    TensorPtr guessIds = mBufferManager->cpu(ITensor::makeShape({len}), nvinfer1::DataType::kINT32);
    auto guessIdsRange = BufferRange<TokenIdType>(*guessIds);

    SizeType32 cur = 0;
    for (auto guess : guesses)
    {
        auto guessRange = BufferRange<TokenIdType>(*guess);
        std::copy(guessRange.begin(), guessRange.end(), guessTokensRange.begin() + cur);
        SizeType32 tmp = offset + 1;
        std::for_each(
            guessIdsRange.begin() + cur, guessIdsRange.begin() + cur + mN - 1, [&tmp](auto& v) { v = tmp++; });
        cur += ITensor::volume(guess->getShape());
    }

    return std::make_tuple(guessTokens, guessIds);
}

std::tuple<LookaheadAlgorithm::TensorPtr, LookaheadAlgorithm::TensorPtr, LookaheadAlgorithm::TensorPtr>
LookaheadAlgorithm::prepare(SizeType32 offset, TokenIdType lastToken)
{
    mCurrentToken = lastToken;
    auto [lookTokens, lookIds, lookSamplingMask] = lookahead(offset);
    auto [guessTokens, guessIds] = guess(offset, lastToken);
    mGuessTokens = guessTokens;

    PRINT_TOKENS(lookTokens);
    PRINT_TENSOR(lookIds);
    PRINT_TOKENS(guessTokens);
    PRINT_TENSOR(guessIds);
    auto lookLen = ITensor::volume(lookTokens->getShape());
    auto guessLen = ITensor::volume(guessTokens->getShape());
    auto len = (SizeType32) (1 + lookLen + guessLen);
    TensorPtr input = mBufferManager->cpu(ITensor::makeShape({len}), nvinfer1::DataType::kINT32);
    TensorPtr position = mBufferManager->cpu(ITensor::makeShape({len}), nvinfer1::DataType::kINT32);
    TensorPtr sampling = mBufferManager->cpu(ITensor::makeShape({len}), nvinfer1::DataType::kINT32);

    auto lookTokensRange = BufferRange<TokenIdType>(*lookTokens);
    auto lookIdsRange = BufferRange<TokenIdType>(*lookIds);
    auto lookSamplingMaskRange = BufferRange<TokenIdType>(*lookSamplingMask);
    auto guessTokensRange = BufferRange<TokenIdType>(*guessTokens);
    auto guessIdsRange = BufferRange<TokenIdType>(*guessIds);
    auto inputRange = BufferRange<TokenIdType>(*input);
    auto positionRange = BufferRange<TokenIdType>(*position);
    auto samplingRange = BufferRange<TokenIdType>(*sampling);

    inputRange[0] = lastToken;
    positionRange[0] = offset;
    std::copy(lookTokensRange.begin(), lookTokensRange.end(), inputRange.begin() + 1);
    std::copy(guessTokensRange.begin(), guessTokensRange.end(), inputRange.begin() + 1 + lookLen);

    std::copy(lookIdsRange.begin(), lookIdsRange.end(), positionRange.begin() + 1);
    std::copy(guessIdsRange.begin(), guessIdsRange.end(), positionRange.begin() + 1 + lookLen);

    for (SizeType32 i = 0; i < len; i++)
    {
        samplingRange[i] = true;
    }
    std::copy(lookSamplingMaskRange.begin(), lookSamplingMaskRange.end(), samplingRange.begin() + 1);

    return std::make_tuple(input, position, sampling);
}

LookaheadAlgorithm::TensorPtr LookaheadAlgorithm::verify(TokenIdType newLastToken, TensorPtr goldenTokens)
{
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
            bool finish = guessTokensRange[idx] == mEndToken;
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

    SizeType32 hitLen = maxHit > 1 ? maxHit : 1;
    TensorPtr accepted = mBufferManager->cpu(ITensor::makeShape({hitLen}), nvinfer1::DataType::kINT32);
    auto acceptedRange = BufferRange<TokenIdType>(*accepted);
    if (maxHit > 1)
    {
        std::copy(guessTokensRange.begin() + hitIdx * (mN - 1), guessTokensRange.begin() + hitIdx * (mN - 1) + maxHit,
            acceptedRange.begin());
    }
    else
    {
        acceptedRange[0] = newLastToken;
    }
    return accepted;
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
LookaheadAlgorithm::TensorPtr LookaheadAlgorithm::update(TensorPtr outputTokens)
{
    TensorPtr newTokens = mBufferManager->cpu(ITensor::makeShape({mW}), nvinfer1::DataType::kINT32);
    auto outputRange = BufferRange<TokenIdType>(*outputTokens);
    auto newRange = BufferRange<TokenIdType>(*newTokens);
    auto pastRange = BufferRange<TokenIdType>(*mPastTokens);

    auto newLastToken = outputRange[0];
    SizeType32 prefill = mN - 2 - mFilling;
    for (SizeType32 i = 0; i < mW; i++)
    {
        newRange[i] = outputRange[prefill + i * mFilling + mFilling];
    }

    if (mFilling < mN - 1)
    {
        for (SizeType32 i = 0; i < mW; i++)
        {
            pastRange[i * (mN - 1) + mFilling] = newRange[i];
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
            *(std::prev(end, 1)) = newRange[i];
            newRange[i] = key;
        }
        newRange[0] = newLastToken;
        mPoolManager.update(newTokens, mPastTokens);
    }

    PRINT_TOKENS2D(mPastTokens);

    auto guessSize = ITensor::volume(mGuessTokens->getShape());
    auto outputSize = ITensor::volume(outputTokens->getShape());
    TLLM_CHECK(guessSize + 1 + mN - 2 - mFilling + mFilling * mW == outputSize);
    TensorPtr goldenTokens = ITensor::slice(outputTokens, 1 + mN - 2 - mFilling + mFilling * mW, guessSize);

    auto generatedTokens = verify(newLastToken, goldenTokens);

    auto generatedRange = BufferRange<TokenIdType>(*generatedTokens);
    auto goldRange = BufferRange<TokenIdType>(*mGoldenTokens);
    auto genLen = generatedTokens->getShape().d[0];
    TLLM_CHECK(genLen < mN);
    std::copy(generatedRange.begin(), generatedRange.end(), goldRange.begin() + mN - 1);
    TensorPtr newGold = ITensor::slice(mGoldenTokens, 0, mN - 1 + genLen);
    mPoolManager.fillWithPrompt(newGold, mN);
    std::copy(goldRange.begin() + genLen, goldRange.begin() + genLen + mN - 1, goldRange.begin());

    if (mFilling < mN - 1)
    {
        mFilling++;
    }

    return generatedTokens;
}

} // namespace tensorrt_llm::layers
