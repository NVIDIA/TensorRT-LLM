/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::executor
{

SamplingConfig::SamplingConfig(SizeType32 beamWidth, std::optional<SizeType32> const& topK,
    std::optional<FloatType> const& topP, std::optional<FloatType> const& topPMin,
    std::optional<TokenIdType> const& topPResetIds, std::optional<FloatType> const& topPDecay,
    std::optional<RandomSeedType> const& seed, std::optional<FloatType> const& temperature,
    std::optional<SizeType32> const& minTokens, std::optional<FloatType> const& beamSearchDiversityRate,
    std::optional<FloatType> const& repetitionPenalty, std::optional<FloatType> const& presencePenalty,
    std::optional<FloatType> const& frequencyPenalty, std::optional<FloatType> const& lengthPenalty,
    std::optional<SizeType32> const& earlyStopping, std::optional<SizeType32> const& noRepeatNgramSize,
    std::optional<SizeType32> const& numReturnSequences, std::optional<FloatType> const& minP)
    : mBeamWidth(checkBeamWidth(beamWidth))
    , mTopK(checkTopK(topK))
    , mTopP(checkTopP(topP))
    , mTopPMin(checkTopPMin(topPMin))
    , mTopPResetIds(checkTopPResetIds(topPResetIds))
    , mTopPDecay(checkTopPDecay(topPDecay))
    , mSeed(seed)
    , mTemperature(checkTemperature(temperature))
    , mMinTokens(checkMinTokens(minTokens))
    , mBeamSearchDiversityRate(checkBeamSearchDiversityRate(beamSearchDiversityRate))
    , mRepetitionPenalty(checkRepetitionPenalty(repetitionPenalty))
    , mPresencePenalty(presencePenalty)
    , mFrequencyPenalty(frequencyPenalty)
    , mLengthPenalty(lengthPenalty)
    , mEarlyStopping(earlyStopping)
    , mNoRepeatNgramSize(checkNoRepeatNgramSize(noRepeatNgramSize))
    , mNumReturnSequences(checkNumReturnSequences(numReturnSequences, beamWidth))
    , mMinP(checkMinP(minP))
{
    updateNumReturnBeams();
}

bool SamplingConfig::operator==(SamplingConfig const& other) const
{
    return mBeamWidth == other.mBeamWidth && mTopK == other.mTopK && mTopP == other.mTopP && mTopPMin == other.mTopPMin
        && mTopPResetIds == other.mTopPResetIds && mTopPDecay == other.mTopPDecay && mSeed == other.mSeed
        && mTemperature == other.mTemperature && mMinTokens == other.mMinTokens
        && mBeamSearchDiversityRate == other.mBeamSearchDiversityRate && mRepetitionPenalty == other.mRepetitionPenalty
        && mPresencePenalty == other.mPresencePenalty && mFrequencyPenalty == other.mFrequencyPenalty
        && mLengthPenalty == other.mLengthPenalty && mEarlyStopping == other.mEarlyStopping
        && mNoRepeatNgramSize == other.mNoRepeatNgramSize && mNumReturnSequences == other.mNumReturnSequences
        && mMinP == other.mMinP;
}

SizeType32 SamplingConfig::getBeamWidth() const
{
    return mBeamWidth;
}

SizeType32 SamplingConfig::getNumReturnBeams() const
{
    return mNumReturnBeams;
}

std::optional<SizeType32> SamplingConfig::getTopK() const
{
    return mTopK;
}

std::optional<FloatType> SamplingConfig::getTopP() const
{
    return mTopP;
}

std::optional<FloatType> SamplingConfig::getTopPMin() const
{
    return mTopPMin;
}

std::optional<SizeType32> SamplingConfig::getTopPResetIds() const
{
    return mTopPResetIds;
}

std::optional<FloatType> SamplingConfig::getTopPDecay() const
{
    return mTopPDecay;
}

std::optional<RandomSeedType> SamplingConfig::getSeed() const
{
    return mSeed;
}

std::optional<RandomSeedType> SamplingConfig::getRandomSeed() const
{
    TLLM_LOG_WARNING("getRandomSeed is being deprecated; please use getSeed instead.");
    return mSeed;
}

std::optional<FloatType> SamplingConfig::getTemperature() const
{
    return mTemperature;
}

std::optional<SizeType32> SamplingConfig::getMinTokens() const
{
    return mMinTokens;
}

std::optional<SizeType32> SamplingConfig::getMinLength() const
{
    TLLM_LOG_WARNING("getMinLength is being deprecated; please use getMinTokens instead.");
    return mMinTokens;
}

std::optional<FloatType> SamplingConfig::getBeamSearchDiversityRate() const
{
    return mBeamSearchDiversityRate;
}

std::optional<FloatType> SamplingConfig::getRepetitionPenalty() const
{
    return mRepetitionPenalty;
}

std::optional<FloatType> SamplingConfig::getPresencePenalty() const
{
    return mPresencePenalty;
}

std::optional<FloatType> SamplingConfig::getFrequencyPenalty() const
{
    return mFrequencyPenalty;
}

std::optional<FloatType> SamplingConfig::getLengthPenalty() const
{
    return mLengthPenalty;
}

std::optional<SizeType32> SamplingConfig::getEarlyStopping() const
{
    return mEarlyStopping;
}

std::optional<SizeType32> SamplingConfig::getNoRepeatNgramSize() const
{
    return mNoRepeatNgramSize;
}

std::optional<SizeType32> SamplingConfig::getNumReturnSequences() const
{
    return mNumReturnSequences;
}

std::optional<FloatType> SamplingConfig::getMinP() const
{
    return mMinP;
}

// the setters

void SamplingConfig::setBeamWidth(SizeType32 beamWidth)
{
    mBeamWidth = checkBeamWidth(beamWidth);
    updateNumReturnBeams();
}

void SamplingConfig::setTopK(std::optional<SizeType32> const& topK)
{
    mTopK = checkTopK(topK);
}

void SamplingConfig::setTopP(std::optional<FloatType> const& topP)
{
    mTopP = checkTopP(topP);
}

void SamplingConfig::setTopPMin(std::optional<FloatType> const& topPMin)
{
    mTopPMin = checkTopPMin(topPMin);
}

void SamplingConfig::setTopPResetIds(std::optional<TokenIdType> const& topPResetIds)
{
    mTopPResetIds = checkTopPResetIds(topPResetIds);
}

void SamplingConfig::setTopPDecay(std::optional<FloatType> const& topPDecay)
{
    mTopPDecay = checkTopPDecay(topPDecay);
}

void SamplingConfig::setSeed(std::optional<RandomSeedType> const& seed)
{
    mSeed = seed;
}

void SamplingConfig::setRandomSeed(std::optional<RandomSeedType> const& randomSeed)
{
    TLLM_LOG_WARNING("setRandomSeed is being deprecated; please use setSeed instead.");
    mSeed = randomSeed;
}

void SamplingConfig::setTemperature(std::optional<FloatType> const& temperature)
{
    mTemperature = checkTemperature(temperature);
}

void SamplingConfig::setMinTokens(std::optional<SizeType32> const& minTokens)
{
    mMinTokens = checkMinTokens(minTokens);
}

void SamplingConfig::setMinLength(std::optional<SizeType32> const& minLength)
{
    TLLM_LOG_WARNING("setMinLength is being deprecated; please use setMinTokens instead.");
    mMinTokens = checkMinTokens(minLength);
}

void SamplingConfig::setBeamSearchDiversityRate(std::optional<FloatType> const& beamSearchDiversityRate)
{
    mBeamSearchDiversityRate = checkBeamSearchDiversityRate(beamSearchDiversityRate);
}

void SamplingConfig::setRepetitionPenalty(std::optional<FloatType> const& repetitionPenalty)
{
    mRepetitionPenalty = checkRepetitionPenalty(repetitionPenalty);
}

void SamplingConfig::setPresencePenalty(std::optional<FloatType> const& presencePenalty)
{
    mPresencePenalty = presencePenalty;
}

void SamplingConfig::setFrequencyPenalty(std::optional<FloatType> const& frequencyPenalty)
{
    mFrequencyPenalty = frequencyPenalty;
}

void SamplingConfig::setLengthPenalty(std::optional<FloatType> const& lengthPenalty)
{
    mLengthPenalty = lengthPenalty;
}

void SamplingConfig::setEarlyStopping(std::optional<SizeType32> const& earlyStopping)
{
    mEarlyStopping = earlyStopping;
}

void SamplingConfig::setNoRepeatNgramSize(std::optional<SizeType32> const& noRepeatNgramSize)
{
    mNoRepeatNgramSize = checkNoRepeatNgramSize(noRepeatNgramSize);
}

void SamplingConfig::setNumReturnSequences(std::optional<SizeType32> const& numReturnSequences)
{
    mNumReturnSequences = checkNumReturnSequences(numReturnSequences, mBeamWidth);
    updateNumReturnBeams();
}

void SamplingConfig::setMinP(std::optional<FloatType> const& minP)
{
    mMinP = checkMinP(minP);
}

SizeType32 SamplingConfig::checkBeamWidth(SizeType32 beamWidth)
{
    TLLM_CHECK(beamWidth > 0);
    return beamWidth;
}

std::optional<FloatType> const& SamplingConfig::checkTopK(std::optional<FloatType> const& topK)
{
    if (topK.has_value())
    {
        TLLM_CHECK(topK.value() >= 0);
    }
    return topK;
}

std::optional<FloatType> const& SamplingConfig::checkTopP(std::optional<FloatType> const& topP)
{
    if (topP.has_value())
    {
        TLLM_CHECK(topP.value() > 0.f);
        TLLM_CHECK(topP.value() <= 1.f);
    }
    return topP;
}

std::optional<FloatType> const& SamplingConfig::checkTopPMin(std::optional<FloatType> const& topPMin)
{
    if (topPMin.has_value())
    {
        TLLM_CHECK(topPMin.value() >= 0.f);
        TLLM_CHECK(topPMin.value() <= 1.f);
    }
    return topPMin;
}

std::optional<TokenIdType> const& SamplingConfig::checkTopPResetIds(std::optional<TokenIdType> const& topPResetIds)
{
    if (topPResetIds.has_value())
    {
        TLLM_CHECK(topPResetIds.value() >= 0);
    }
    return topPResetIds;
}

std::optional<FloatType> const& SamplingConfig::checkTopPDecay(std::optional<FloatType> const& topPDecay)
{
    if (topPDecay.has_value())
    {
        TLLM_CHECK(topPDecay.value() > 0.f);
        TLLM_CHECK(topPDecay.value() <= 1.f);
    }
    return topPDecay;
}

std::optional<FloatType> const& SamplingConfig::checkTemperature(std::optional<FloatType> const& temperature)
{
    if (temperature.has_value())
    {
        TLLM_CHECK(temperature.value() >= 0.f);
    }
    return temperature;
}

std::optional<SizeType32> const& SamplingConfig::checkMinTokens(std::optional<SizeType32> const& minTokens)
{
    if (minTokens.has_value())
    {
        TLLM_CHECK(minTokens.value() >= 0);
    }
    return minTokens;
}

std::optional<FloatType> const& SamplingConfig::checkRepetitionPenalty(std::optional<FloatType> const& penalty)
{
    if (penalty.has_value())
    {
        TLLM_CHECK_WITH_INFO(penalty.value() > 0.F,
            "Repetition penalty should be strictly greater than zero. Provided value was %f", penalty.value());
    }
    return penalty;
}

std::optional<SizeType32> const& SamplingConfig::checkNoRepeatNgramSize(
    std::optional<SizeType32> const& noRepeatNgramSize)
{
    if (noRepeatNgramSize.has_value())
    {
        TLLM_CHECK(noRepeatNgramSize.value() > 0);
    }
    return noRepeatNgramSize;
}

std::optional<FloatType> const& SamplingConfig::checkBeamSearchDiversityRate(
    std::optional<FloatType> const& beamSearchDiversityRate)
{
    if (beamSearchDiversityRate.has_value())
    {
        TLLM_CHECK(beamSearchDiversityRate.value() >= 0.f);
    }
    return beamSearchDiversityRate;
}

std::optional<SizeType32> const& SamplingConfig::checkNumReturnSequences(
    std::optional<SizeType32> const& numReturnSequences, SizeType32 beamWidth)
{
    if (numReturnSequences.has_value())
    {
        TLLM_CHECK(numReturnSequences.value() > 0);
        TLLM_CHECK(beamWidth == 1 || numReturnSequences.value() <= beamWidth);
    }
    return numReturnSequences;
}

std::optional<FloatType> const& SamplingConfig::checkMinP(std::optional<FloatType> const& minP)
{
    if (minP.has_value())
    {
        TLLM_CHECK(minP.value() >= 0.f && minP.value() <= 1.0f);
    }
    return minP;
}

void SamplingConfig::updateNumReturnBeams()
{
    mNumReturnBeams
        = (mNumReturnSequences && mBeamWidth > 1) ? std::min(mNumReturnSequences.value(), mBeamWidth) : mBeamWidth;
}

} // namespace tensorrt_llm::executor
