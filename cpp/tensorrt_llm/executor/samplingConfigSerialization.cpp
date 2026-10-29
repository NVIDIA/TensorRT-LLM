/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/executor/serializeUtils.h"

namespace su = tensorrt_llm::executor::serialize_utils;

namespace tensorrt_llm::executor
{

// SamplingConfig
SamplingConfig Serialization::deserializeSamplingConfig(std::istream& is)
{
    auto beamWidth = su::deserialize<SizeType32>(is);
    auto topK = su::deserialize<std::optional<SizeType32>>(is);
    auto topP = su::deserialize<std::optional<FloatType>>(is);
    auto topPMin = su::deserialize<std::optional<FloatType>>(is);
    auto topPResetIds = su::deserialize<std::optional<TokenIdType>>(is);
    auto topPDecay = su::deserialize<std::optional<FloatType>>(is);
    auto randomSeed = su::deserialize<std::optional<RandomSeedType>>(is);
    auto temperature = su::deserialize<std::optional<FloatType>>(is);
    auto minLength = su::deserialize<std::optional<SizeType32>>(is);
    auto beamSearchDiversityRate = su::deserialize<std::optional<FloatType>>(is);
    auto repetitionPenalty = su::deserialize<std::optional<FloatType>>(is);
    auto presencePenalty = su::deserialize<std::optional<FloatType>>(is);
    auto frequencyPenalty = su::deserialize<std::optional<FloatType>>(is);
    auto promptIgnoreLength = su::deserialize<std::optional<SizeType32>>(is);
    auto lengthPenalty = su::deserialize<std::optional<FloatType>>(is);
    auto earlyStopping = su::deserialize<std::optional<SizeType32>>(is);
    auto noRepeatNgramSize = su::deserialize<std::optional<SizeType32>>(is);
    auto numReturnSequences = su::deserialize<std::optional<SizeType32>>(is);
    auto minP = su::deserialize<std::optional<FloatType>>(is);
    auto beamWidthArray = su::deserialize<std::optional<std::vector<SizeType32>>>(is);

    return SamplingConfig{beamWidth, topK, topP, topPMin, topPResetIds, topPDecay, randomSeed, temperature, minLength,
        beamSearchDiversityRate, repetitionPenalty, presencePenalty, frequencyPenalty, promptIgnoreLength,
        lengthPenalty, earlyStopping, noRepeatNgramSize, numReturnSequences, minP, beamWidthArray};
}

void Serialization::serialize(SamplingConfig const& config, std::ostream& os)
{
    su::serialize(config.mBeamWidth, os);
    su::serialize(config.mTopK, os);
    su::serialize(config.mTopP, os);
    su::serialize(config.mTopPMin, os);
    su::serialize(config.mTopPResetIds, os);
    su::serialize(config.mTopPDecay, os);
    su::serialize(config.mSeed, os);
    su::serialize(config.mTemperature, os);
    su::serialize(config.mMinTokens, os);
    su::serialize(config.mBeamSearchDiversityRate, os);
    su::serialize(config.mRepetitionPenalty, os);
    su::serialize(config.mPresencePenalty, os);
    su::serialize(config.mFrequencyPenalty, os);
    su::serialize(config.mPromptIgnoreLength, os);
    su::serialize(config.mLengthPenalty, os);
    su::serialize(config.mEarlyStopping, os);
    su::serialize(config.mNoRepeatNgramSize, os);
    su::serialize(config.mNumReturnSequences, os);
    su::serialize(config.mMinP, os);
    su::serialize(config.mBeamWidthArray, os);
}

size_t Serialization::serializedSize(SamplingConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mBeamWidth);
    totalSize += su::serializedSize(config.mTopK);
    totalSize += su::serializedSize(config.mTopP);
    totalSize += su::serializedSize(config.mTopPMin);
    totalSize += su::serializedSize(config.mTopPResetIds);
    totalSize += su::serializedSize(config.mTopPDecay);
    totalSize += su::serializedSize(config.mSeed);
    totalSize += su::serializedSize(config.mTemperature);
    totalSize += su::serializedSize(config.mMinTokens);
    totalSize += su::serializedSize(config.mBeamSearchDiversityRate);
    totalSize += su::serializedSize(config.mRepetitionPenalty);
    totalSize += su::serializedSize(config.mPresencePenalty);
    totalSize += su::serializedSize(config.mFrequencyPenalty);
    totalSize += su::serializedSize(config.mPromptIgnoreLength);
    totalSize += su::serializedSize(config.mLengthPenalty);
    totalSize += su::serializedSize(config.mEarlyStopping);
    totalSize += su::serializedSize(config.mNoRepeatNgramSize);
    totalSize += su::serializedSize(config.mNumReturnSequences);
    totalSize += su::serializedSize(config.mMinP);
    totalSize += su::serializedSize(config.mBeamWidthArray);
    return totalSize;
}

} // namespace tensorrt_llm::executor
