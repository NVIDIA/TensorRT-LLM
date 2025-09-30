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

#include "tensorrt_llm/executor/types.h"
#include "tests/unit_tests/layers/baseSamplingLayerTest.h"

namespace
{

namespace tle = tensorrt_llm::executor;

using namespace tensorrt_llm::tests::layers::sampling;
using namespace tensorrt_llm::runtime;

template <typename T>
class BeamSearchLayerTest : public BaseSamplingLayerTest<T>
{
    void SetUp() override
    {
        this->mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        this->mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(this->mStream);
    }

    void initLayer(TestSamplingParams const& params) override
    {
        auto decodingMode = tle::DecodingMode::BeamSearch();
        auto const decodingDomain = tensorrt_llm::layers::DecoderDomain(
            this->maxBatchSize(), params.beamWidth, this->mVocabSize, this->mVocabSizePadded);
        this->mSamplingLayer = std::make_shared<tensorrt_llm::layers::BeamSearchLayer<T>>(
            decodingMode, decodingDomain, this->mBufferManager);
    }
};

TYPED_TEST_SUITE(BeamSearchLayerTest, FloatAndHalfTypes);

TYPED_TEST(BeamSearchLayerTest, BeamWidth2)
{
    TestSamplingParams params;

    params.batchSize = 3;
    params.beamWidth = 2;

    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, // step 3
        {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, // step 3
    };
    this->runTest(expectedOutputIds, params);
}

} // namespace
