/* * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.  * * Licensed under the Apache License,
 * Version 2.0 (the "License"); * you may not use this file except in compliance with the License.  * You may obtain a
 * copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Invoke;

namespace tr = tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;
namespace texec = tensorrt_llm::executor;

TEST(samplingConfigTest, validInputs)
{
    {
        texec::SamplingConfig execSamplingCfg(1);
        tr::SamplingConfig samplingCfg(execSamplingCfg, std::nullopt);
        EXPECT_EQ(samplingCfg.beamWidth, execSamplingCfg.getBeamWidth());
        EXPECT_EQ(samplingCfg.draftAcceptanceThreshold, std::nullopt);
    }
    {
        texec::SamplingConfig execSamplingCfg(1);
        texec::ExternalDraftTokensConfig specCfg({1}, std::nullopt, 0.5);
        tr::SamplingConfig samplingCfg(execSamplingCfg, specCfg);
        EXPECT_EQ(samplingCfg.beamWidth, execSamplingCfg.getBeamWidth());
        EXPECT_TRUE(samplingCfg.draftAcceptanceThreshold.has_value());
        EXPECT_THAT(samplingCfg.draftAcceptanceThreshold.value(), testing::ElementsAre(0.5f));
    }
    {
        texec::SizeType32 topK = 1;
        texec::FloatType topP = 0.5;
        texec::FloatType topPMin = 0.1;
        texec::SizeType32 topPResetIds = 1;
        texec::FloatType topPDecay = 0.6;
        uint64_t randomSeed = 7777;
        texec::FloatType temperature = 0.245;
        texec::SizeType32 minLength = 1234;
        texec::FloatType beamSearchDiversityRate = 0.9999;
        texec::FloatType repetitionPenalty = 0.11;
        texec::FloatType presencePenalty = 0.22;
        texec::FloatType frequencyPenalty = 0.33;
        texec::FloatType lengthPenalty = 0.44;
        texec::SizeType32 earlyStopping = 1;

        texec::SamplingConfig execSamplingCfg(1, topK, topP, topPMin, topPResetIds, topPDecay, randomSeed, temperature,
            minLength, beamSearchDiversityRate, repetitionPenalty, presencePenalty, frequencyPenalty, lengthPenalty,
            earlyStopping);
        texec::ExternalDraftTokensConfig specCfg({1}, std::nullopt, 0.5);
        tr::SamplingConfig samplingCfg(execSamplingCfg, specCfg);
        EXPECT_EQ(samplingCfg.beamWidth, execSamplingCfg.getBeamWidth());
        EXPECT_THAT(samplingCfg.draftAcceptanceThreshold.value(), testing::ElementsAre(0.5f));
        EXPECT_THAT(samplingCfg.temperature.value(), testing::ElementsAre(temperature));
        EXPECT_THAT(samplingCfg.minLength.value(), testing::ElementsAre(minLength));
        EXPECT_THAT(samplingCfg.repetitionPenalty.value(), testing::ElementsAre(repetitionPenalty));
        EXPECT_THAT(samplingCfg.presencePenalty.value(), testing::ElementsAre(presencePenalty));
        EXPECT_THAT(samplingCfg.frequencyPenalty.value(), testing::ElementsAre(frequencyPenalty));
        EXPECT_THAT(samplingCfg.topK.value(), testing::ElementsAre(topK));
        EXPECT_THAT(samplingCfg.topP.value(), testing::ElementsAre(topP));
        EXPECT_THAT(samplingCfg.randomSeed.value(), testing::ElementsAre(randomSeed));
        EXPECT_THAT(samplingCfg.topPMin.value(), testing::ElementsAre(topPMin));
        EXPECT_THAT(samplingCfg.topPResetIds.value(), testing::ElementsAre(topPResetIds));
        EXPECT_THAT(samplingCfg.beamSearchDiversityRate.value(), testing::ElementsAre(beamSearchDiversityRate));
        EXPECT_THAT(samplingCfg.lengthPenalty.value(), testing::ElementsAre(lengthPenalty));
        EXPECT_THAT(samplingCfg.earlyStopping.value(), testing::ElementsAre(earlyStopping));
    }
}
