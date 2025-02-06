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
class SamplingLayerTest : public BaseSamplingLayerTest<T>
{
    void SetUp() override
    {
        this->mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        this->mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(this->mStream);
    }

    void initLayer(TestSamplingParams const& params) override
    {
        auto decodingMode = tle::DecodingMode::Auto();
        if (params.topKs.size() && params.topPs.size())
        {
            decodingMode = tle::DecodingMode::TopKTopP();
        }
        else if (params.topKs.size())
        {
            decodingMode = tle::DecodingMode::TopK();
        }
        else if (params.topPs.size())
        {
            decodingMode = tle::DecodingMode::TopP();
        }

        auto const decodingDomain
            = tensorrt_llm::layers::DecoderDomain(this->maxBatchSize(), 1, this->mVocabSize, this->mVocabSizePadded);
        this->mSamplingLayer = std::make_shared<tensorrt_llm::layers::SamplingLayer<T>>(
            decodingMode, decodingDomain, this->mBufferManager);
    }
};

TYPED_TEST_SUITE(SamplingLayerTest, FloatAndHalfTypes);

TYPED_TEST(SamplingLayerTest, TopKToPPSkipDecode)
{
    SizeType32 topK = 2;
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, // step 0
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, // step 1
        {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, // step 2
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, TopKSkipDecodeTopP)
{
    SizeType32 topK = 0;
    float topP = 0.5f;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, // step 0
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, // step 1
        {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, // step 2
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, BatchTopKTopP)
{
    std::vector<SizeType32> topKs = {0, 2, 1, 0, 1, 0};
    std::vector<float> topPs = {0.3f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4, 5}, {4}, {4, 5}, {4}, {4, 5}, // step 0
        {0}, {0, 1}, {0}, {0, 1}, {0}, {0, 1}, // step 1
        {2}, {2, 3}, {2}, {2, 3}, {2}, {2, 3}, // step 2
        {0}, {0, 1}, {0}, {0, 1}, {0}, {0, 1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, TopPDecay)
{
    TestSamplingParams params;
    params.topPs = {0.8f, 0.5f, 0.3f, 0.2f, 0.5f, 1.0f};
    params.decay = {0.3f, 0.3f, 0.3f, 0.9f, 0.3f, 0.8f};
    params.topPResetIds = {2, -1, 2, -1, 2, -1};
    params.minTopP = {0.5f, 0.1f, 0.3f, 0.1f, 0.1f, 0.1f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5, 6}, {4, 5}, {4}, {4}, {4, 5}, {4, 5, 6, 7}, // step 0
        {0, 1}, {0}, {0}, {0}, {0}, {0, 1, 2},             // step 1
        {2, 3}, {2}, {2}, {2}, {2}, {2, 3},                // step 2
        {0, 1, 2}, {0}, {0}, {0}, {0, 1}, {0, 1}           // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, TopK)
{
    SizeType32 topK = 2;
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, // step 0
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, // step 1
        {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, // step 2
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, TopK1TopP0)
{
    SizeType32 topK = 1;
    float topP = 0.0f;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, BatchTopK)
{
    std::vector<SizeType32> topKs = {2, 1, 1, 2, 1, 1};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4}, {4}, {4, 5}, {4}, {4}, // step 0
        {0, 1}, {0}, {0}, {0, 1}, {0}, {0}, // step 1
        {2, 3}, {2}, {2}, {2, 3}, {2}, {2}, // step 2
        {0, 1}, {0}, {0}, {0, 1}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, TopKTopP)
{
    SizeType32 topK = 2;
    float topP = 0.3;
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = {topP};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, BatchTopKTopP1)
{
    std::vector<SizeType32> topKs = {2, 2, 1, 2, 2, 1};
    float topP = 0.3;
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = {topP};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, BatchTopKBatchTopP)
{
    std::vector<SizeType32> topKs = {2, 2, 0, 2, 2, 1};
    std::vector<float> topPs = {0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4}, {4, 5}, {4, 5}, {4}, {4}, // step 0
        {0, 1}, {0}, {0, 1}, {0, 1}, {0}, {0}, // step 1
        {2, 3}, {2}, {2, 3}, {2, 3}, {2}, {2}, // step 2
        {0, 1}, {0}, {0, 1}, {0, 1}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, InvalidArgsZeroTopK)
{
    SizeType32 topK = 0;
    TestSamplingParams params;
    params.topKs = {topK};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, InvalidArgsZeroTopP)
{
    float topP = 0;
    SizeType32 topK = 0;
    TestSamplingParams params;
    params.topPs = {topP};
    params.topKs = {topK};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, InvalidArgsZeroTopKTopP)
{
    SizeType32 topK = 0;
    float topP = 0;
    TestSamplingParams params;
    params.topPs = {topP};
    params.topKs = {topK};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, InvalidArgsZeroBatchTopKTopP)
{
    std::vector<SizeType32> topKs = {0, 0, 0, 0, 0, 0};
    float topP = 0;
    TestSamplingParams params;
    params.topPs = {topP};
    params.topKs = topKs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, InvalidArgsZeroTopKBatchTopP)
{
    SizeType32 topK = 0;
    std::vector<float> topPs = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    TestSamplingParams params;
    params.topPs = topPs;
    params.topKs = {topK};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, InvalidArgsBatchTopKContainZero)
{
    std::vector<SizeType32> topKs = {2, 1, 0, 0, 2, 1};
    TestSamplingParams params;
    params.topKs = topKs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4}, {4}, {4}, {4, 5}, {4}, // step 0
        {0, 1}, {0}, {0}, {0}, {0, 1}, {0}, // step 1
        {2, 3}, {2}, {2}, {2}, {2, 3}, {2}, // step 2
        {0, 1}, {0}, {0}, {0}, {0, 1}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, InvalidArgsBatchTopKTopPContainZero)
{
    std::vector<SizeType32> topKs = {2, 2, 1, 0, 2, 0};
    float topP = 0.0;
    TestSamplingParams params;
    params.topPs = {topP};
    params.topKs = topKs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4, 5}, {4}, {4}, {4, 5}, {4}, // step 0
        {0, 1}, {0, 1}, {0}, {0}, {0, 1}, {0}, // step 1
        {2, 3}, {2, 3}, {2}, {2}, {2, 3}, {2}, // step 2
        {0, 1}, {0, 1}, {0}, {0}, {0, 1}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, OnlyTopK)
{
    std::vector<SizeType32> topKs = {2, 2, 1, 0, 2, 0};
    TestSamplingParams params;
    params.topKs = topKs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4, 5}, {4}, {4}, {4, 5}, {4}, // step 0
        {0, 1}, {0, 1}, {0}, {0}, {0, 1}, {0}, // step 1
        {2, 3}, {2, 3}, {2}, {2}, {2, 3}, {2}, // step 2
        {0, 1}, {0, 1}, {0}, {0}, {0, 1}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(SamplingLayerTest, OnlyTopP)
{
    std::vector<float> topPs = {0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f};
    TestSamplingParams params;
    params.topPs = topPs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

} // namespace
