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

#include "tests/unit_tests/layers/baseSamplingLayerTest.h"

namespace
{

using namespace tensorrt_llm::tests::layers::sampling;
using namespace tensorrt_llm::runtime;

template <typename T>
class TopKSamplingLayerTest : public BaseSamplingLayerTest<T>
{
    void SetUp() override
    {
        this->mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        this->mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(this->mStream);
    }

    void initLayer(TestSamplingParams const& params) override
    {
        auto const decodingDomain
            = tensorrt_llm::layers::DecoderDomain(this->maxBatchSize(), 1, this->mVocabSize, this->mVocabSizePadded);
        this->mSamplingLayer
            = std::make_shared<tensorrt_llm::layers::TopKSamplingLayer<T>>(decodingDomain, this->mBufferManager);
    }
};

TYPED_TEST_SUITE(TopKSamplingLayerTest, FloatAndHalfTypes);

TYPED_TEST(TopKSamplingLayerTest, TopK)
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

TYPED_TEST(TopKSamplingLayerTest, TopK1TopP0)
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

TYPED_TEST(TopKSamplingLayerTest, BatchTopK)
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

TYPED_TEST(TopKSamplingLayerTest, SkipDecode)
{
    // Skip topK decode
    float topP = 0.3;
    TestSamplingParams params;
    params.topPs = {topP};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {0}, {0}, {0}, {0}, {0}, {0}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {0}, {0}, {0}, {0}, {0}, {0}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopKSamplingLayerTest, TopKTopP)
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

TYPED_TEST(TopKSamplingLayerTest, BatchTopKTopP)
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

TYPED_TEST(TopKSamplingLayerTest, TopKBatchTopP)
{
    SizeType32 topK = 2;
    std::vector<float> topPs = {0.5, 0.3, 0.5, 0.5, 0.3, 0.5};
    TestSamplingParams params;
    params.topKs = {topK};
    params.topPs = topPs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4}, {4, 5}, {4, 5}, {4}, {4, 5}, // step 0
        {0, 1}, {0}, {0, 1}, {0, 1}, {0}, {0, 1}, // step 1
        {2, 3}, {2}, {2, 3}, {2, 3}, {2}, {2, 3}, // step 2
        {0, 1}, {0}, {0, 1}, {0, 1}, {0}, {0, 1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopKSamplingLayerTest, BatchTopKBatchTopP)
{
    std::vector<SizeType32> topKs = {2, 2, 1, 2, 2, 1};
    std::vector<float> topPs = {0.0, 0.3, 0.5, 0.0, 0.3, 0.5};
    TestSamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4}, {4}, {4, 5}, {4}, {4}, // step 0
        {0, 1}, {0}, {0}, {0, 1}, {0}, {0}, // step 1
        {2, 3}, {2}, {2}, {2, 3}, {2}, {2}, // step 2
        {0, 1}, {0}, {0}, {0, 1}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopKSamplingLayerTest, InvalidArgsZeroTopK)
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

TYPED_TEST(TopKSamplingLayerTest, InvalidArgsZeroTopP)
{
    float topP = 0;
    TestSamplingParams params;
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

TYPED_TEST(TopKSamplingLayerTest, InvalidArgsZeroTopKTopP)
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

TYPED_TEST(TopKSamplingLayerTest, InvalidArgsZeroBatchTopKTopP)
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

TYPED_TEST(TopKSamplingLayerTest, InvalidArgsZeroTopKBatchTopP)
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

TYPED_TEST(TopKSamplingLayerTest, InvalidArgsBatchTopKContainZero)
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

TYPED_TEST(TopKSamplingLayerTest, InvalidArgsBatchTopKTopPContainZero)
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

TYPED_TEST(TopKSamplingLayerTest, InvalidArgsBatchTopKBatchTopPContainZero)
{
    std::vector<SizeType32> topKs = {0, 2, 1, 2, 2, 0};
    std::vector<float> topPs = {0.0, 0.3, 0.9, 0.0, 0.3, 0.5};
    TestSamplingParams params;
    params.topPs = topPs;
    params.topKs = topKs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4, 5}, {4}, {0}, // step 0
        {0}, {0}, {0}, {0, 1}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2, 3}, {2}, {0}, // step 2
        {0}, {0}, {0}, {0, 1}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

} // namespace
