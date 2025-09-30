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
#include "tensorrt_llm/kernels/beamSearchKernels.h"

namespace tensorrt_llm::executor
{

template <typename T>
using OptVec = std::optional<std::vector<T>>;

using OptSize32 = std::optional<SizeType32>;
using OptFloat = std::optional<FloatType>;

SamplingConfig::SamplingConfig(SizeType32 beamWidth, OptSize32 const& topK, OptFloat const& topP,
    OptFloat const& topPMin, std::optional<TokenIdType> const& topPResetIds, OptFloat const& topPDecay,
    std::optional<RandomSeedType> const& seed, OptFloat const& temperature, OptSize32 const& minTokens,
    OptFloat const& beamSearchDiversityRate, OptFloat const& repetitionPenalty, OptFloat const& presencePenalty,
    OptFloat const& frequencyPenalty, OptFloat const& lengthPenalty, OptSize32 const& earlyStopping,
    OptSize32 const& noRepeatNgramSize, OptSize32 const& numReturnSequences, OptFloat const& minP,
    OptVec<SizeType32> const& beamWidthArray)
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
    , mLengthPenalty(checkLengthPenalty(lengthPenalty))
    , mEarlyStopping(checkEarlyStopping(earlyStopping))
    , mNoRepeatNgramSize(checkNoRepeatNgramSize(noRepeatNgramSize))
    , mNumReturnSequences(checkNumReturnSequences(numReturnSequences, beamWidth))
    , mMinP(checkMinP(minP))
{
    updateNumReturnBeams();
    std::tie(mBeamWidthArray, mBeamWidth) = checkBeamWidthArray(beamWidthArray, mBeamWidth);
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
        && mMinP == other.mMinP && mBeamWidthArray == other.mBeamWidthArray;
}

// Getters
SizeType32 SamplingConfig::getBeamWidth() const
{
    return mBeamWidth;
}

SizeType32 SamplingConfig::getNumReturnBeams() const
{
    return mNumReturnBeams;
}

OptSize32 SamplingConfig::getTopK() const
{
    return mTopK;
}

OptFloat SamplingConfig::getTopP() const
{
    return mTopP;
}

OptFloat SamplingConfig::getTopPMin() const
{
    return mTopPMin;
}

OptSize32 SamplingConfig::getTopPResetIds() const
{
    return mTopPResetIds;
}

OptFloat SamplingConfig::getTopPDecay() const
{
    return mTopPDecay;
}

std::optional<RandomSeedType> SamplingConfig::getSeed() const
{
    return mSeed;
}

OptFloat SamplingConfig::getTemperature() const
{
    return mTemperature;
}

OptSize32 SamplingConfig::getMinTokens() const
{
    return mMinTokens;
}

OptFloat SamplingConfig::getBeamSearchDiversityRate() const
{
    return mBeamSearchDiversityRate;
}

OptFloat SamplingConfig::getRepetitionPenalty() const
{
    return mRepetitionPenalty;
}

OptFloat SamplingConfig::getPresencePenalty() const
{
    return mPresencePenalty;
}

OptFloat SamplingConfig::getFrequencyPenalty() const
{
    return mFrequencyPenalty;
}

OptFloat SamplingConfig::getLengthPenalty() const
{
    return mLengthPenalty;
}

OptSize32 SamplingConfig::getEarlyStopping() const
{
    return mEarlyStopping;
}

OptSize32 SamplingConfig::getNoRepeatNgramSize() const
{
    return mNoRepeatNgramSize;
}

OptSize32 SamplingConfig::getNumReturnSequences() const
{
    return mNumReturnSequences;
}

std::optional<FloatType> SamplingConfig::getMinP() const
{
    return mMinP;
}

OptVec<SizeType32> SamplingConfig::getBeamWidthArray() const
{
    return mBeamWidthArray;
}

// Setters
void SamplingConfig::setBeamWidth(SizeType32 beamWidth)
{
    mBeamWidth = checkBeamWidth(beamWidth);
    updateNumReturnBeams();
}

void SamplingConfig::setTopK(OptSize32 const& topK)
{
    mTopK = checkTopK(topK);
}

void SamplingConfig::setTopP(OptFloat const& topP)
{
    mTopP = checkTopP(topP);
}

void SamplingConfig::setTopPMin(OptFloat const& topPMin)
{
    mTopPMin = checkTopPMin(topPMin);
}

void SamplingConfig::setTopPResetIds(std::optional<TokenIdType> const& topPResetIds)
{
    mTopPResetIds = checkTopPResetIds(topPResetIds);
}

void SamplingConfig::setTopPDecay(OptFloat const& topPDecay)
{
    mTopPDecay = checkTopPDecay(topPDecay);
}

void SamplingConfig::setSeed(std::optional<RandomSeedType> const& seed)
{
    mSeed = seed;
}

void SamplingConfig::setTemperature(OptFloat const& temperature)
{
    mTemperature = checkTemperature(temperature);
}

void SamplingConfig::setMinTokens(OptSize32 const& minTokens)
{
    mMinTokens = checkMinTokens(minTokens);
}

void SamplingConfig::setBeamSearchDiversityRate(OptFloat const& beamSearchDiversityRate)
{
    mBeamSearchDiversityRate = checkBeamSearchDiversityRate(beamSearchDiversityRate);
}

void SamplingConfig::setRepetitionPenalty(OptFloat const& repetitionPenalty)
{
    mRepetitionPenalty = checkRepetitionPenalty(repetitionPenalty);
}

void SamplingConfig::setPresencePenalty(OptFloat const& presencePenalty)
{
    mPresencePenalty = presencePenalty;
}

void SamplingConfig::setFrequencyPenalty(OptFloat const& frequencyPenalty)
{
    mFrequencyPenalty = frequencyPenalty;
}

void SamplingConfig::setLengthPenalty(OptFloat const& lengthPenalty)
{
    mLengthPenalty = lengthPenalty; // TODO: re-enable `checkLengthPenalty` later
}

void SamplingConfig::setEarlyStopping(OptSize32 const& earlyStopping)
{
    mEarlyStopping = earlyStopping; // TODO: re-enable `checkEarlyStopping` later
}

void SamplingConfig::setNoRepeatNgramSize(OptSize32 const& noRepeatNgramSize)
{
    mNoRepeatNgramSize = checkNoRepeatNgramSize(noRepeatNgramSize);
}

void SamplingConfig::setNumReturnSequences(OptSize32 const& numReturnSequences)
{
    mNumReturnSequences = checkNumReturnSequences(numReturnSequences, mBeamWidth);
    updateNumReturnBeams();
}

void SamplingConfig::setMinP(std::optional<FloatType> const& minP)
{
    mMinP = checkMinP(minP);
}

void SamplingConfig::setBeamWidthArray(OptVec<SizeType32> const& beamWidthArray)
{
    std::tie(mBeamWidthArray, mBeamWidth) = checkBeamWidthArray(beamWidthArray, mBeamWidth);
}

// Checkers
SizeType32 SamplingConfig::checkBeamWidth(SizeType32 beamWidth)
{
    TLLM_CHECK(beamWidth > 0 && beamWidth <= static_cast<SizeType32 const>(tensorrt_llm::kernels::kMaxBeamWidth));
    return beamWidth;
}

OptFloat const& SamplingConfig::checkTopK(OptFloat const& topK)
{
    if (topK.has_value())
    {
        TLLM_CHECK(topK.value() >= 0);
    }
    return topK;
}

OptFloat const& SamplingConfig::checkTopP(OptFloat const& topP)
{
    if (topP.has_value())
    {
        TLLM_CHECK(topP.value() > 0.f);
        TLLM_CHECK(topP.value() <= 1.f);
    }
    return topP;
}

OptFloat const& SamplingConfig::checkTopPMin(OptFloat const& topPMin)
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

OptFloat const& SamplingConfig::checkTopPDecay(OptFloat const& topPDecay)
{
    if (topPDecay.has_value())
    {
        TLLM_CHECK(topPDecay.value() > 0.f);
        TLLM_CHECK(topPDecay.value() <= 1.f);
    }
    return topPDecay;
}

OptFloat const& SamplingConfig::checkTemperature(OptFloat const& temperature)
{
    if (temperature.has_value())
    {
        TLLM_CHECK(temperature.value() >= 0.f);
    }
    return temperature;
}

OptSize32 const& SamplingConfig::checkMinTokens(OptSize32 const& minTokens)
{
    if (minTokens.has_value())
    {
        TLLM_CHECK(minTokens.value() >= 0);
    }
    return minTokens;
}

OptFloat const& SamplingConfig::checkBeamSearchDiversityRate(OptFloat const& beamSearchDiversityRate)
{
    if (beamSearchDiversityRate.has_value())
    {
        TLLM_CHECK(beamSearchDiversityRate.value() >= 0.f);
    }
    return beamSearchDiversityRate;
}

OptFloat const& SamplingConfig::checkRepetitionPenalty(OptFloat const& repetitionpenalty)
{
    if (repetitionpenalty.has_value())
    {
        TLLM_CHECK(repetitionpenalty.value() > 0.f);
    }
    return repetitionpenalty;
}

OptFloat const& SamplingConfig::checkLengthPenalty(OptFloat const& lengthPenalty)
{
    if (lengthPenalty.has_value())
    {
        TLLM_CHECK(lengthPenalty.value() >= 0.f);
    }
    return lengthPenalty;
}

OptSize32 const& SamplingConfig::checkEarlyStopping(OptSize32 const& earlyStopping)
{
    if (earlyStopping.has_value())
    {
        TLLM_CHECK(earlyStopping.value() >= 0);
    }
    return earlyStopping;
}

OptSize32 const& SamplingConfig::checkNoRepeatNgramSize(OptSize32 const& noRepeatNgramSize)
{
    if (noRepeatNgramSize.has_value())
    {
        TLLM_CHECK(noRepeatNgramSize.value() >= 0);
    }
    return noRepeatNgramSize;
}

OptSize32 const& SamplingConfig::checkNumReturnSequences(OptSize32 const& numReturnSequences, SizeType32 beamWidth)
{
    if (numReturnSequences.has_value())
    {
        TLLM_CHECK(numReturnSequences.value() > 0);
        TLLM_CHECK(beamWidth == 1 || numReturnSequences.value() <= beamWidth);
    }
    return numReturnSequences;
}

OptFloat const& SamplingConfig::checkMinP(OptFloat const& minP)
{
    if (minP.has_value())
    {
        TLLM_CHECK(minP.value() >= 0.f && minP.value() <= 1.0f);
    }
    return minP;
}

std::pair<OptVec<SizeType32> const&, SizeType32 const> const SamplingConfig::checkBeamWidthArray(
    OptVec<SizeType32> const& beamWidthArray, SizeType32 const beamWidth)
{
    SizeType32 maxBeamWidth = beamWidth;
    if (beamWidthArray.has_value())
    {
        auto array = beamWidthArray.value();
        TLLM_CHECK(array.size() <= static_cast<SizeType32 const>(tensorrt_llm::kernels::kMaxBeamWidthArrayLength));
        for (auto const& bm : array)
        {
            TLLM_CHECK(bm > 0 && bm < static_cast<SizeType32 const>(tensorrt_llm::kernels::kMaxBeamWidth));
            maxBeamWidth = std::max(maxBeamWidth, bm);
        }
    }
    return {beamWidthArray, maxBeamWidth};
}

void SamplingConfig::updateNumReturnBeams()
{
    mNumReturnBeams
        = (mNumReturnSequences && mBeamWidth > 1) ? std::min(mNumReturnSequences.value(), mBeamWidth) : mBeamWidth;
}

} // namespace tensorrt_llm::executor
