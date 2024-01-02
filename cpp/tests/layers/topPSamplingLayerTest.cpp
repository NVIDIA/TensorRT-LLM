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

#include "tests/layers/samplingLayerTest.h"

namespace
{

using namespace tensorrt_llm::tests::layers::sampling;

template <typename T>
class TopPSamplingLayerTest : public SamplingLayerTest<T>
{
    void SetUp() override
    {
        this->mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        this->mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(this->mStream);

        this->mAllocator = std::make_shared<tensorrt_llm::common::CudaAllocator>(*this->mBufferManager);

        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&mDeviceProp, device);

        this->mSamplingLayer = std::make_shared<tensorrt_llm::layers::TopPSamplingLayer<T>>(
            this->mVocabSize, this->mVocabSizePadded, this->mStream->get(), this->mAllocator, false, &mDeviceProp);
    }

    struct cudaDeviceProp mDeviceProp;
};

TYPED_TEST_SUITE(TopPSamplingLayerTest, FloatAndHalfTypes);

TYPED_TEST(TopPSamplingLayerTest, TopKSkipDecode)
{
    uint32_t topK = 2;
    float topP = 0.0f;
    SamplingParams params;
    params.topKs = {topK};
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

TYPED_TEST(TopPSamplingLayerTest, TopKTopPSkipDecode)
{
    uint32_t topK = 2;
    float topP = 1.0f;
    SamplingParams params;
    params.topKs = {topK};
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

TYPED_TEST(TopPSamplingLayerTest, BatchTopKTopP)
{
    std::vector<uint32_t> topKs = {0, 1, 1, 0, 1, 0};
    std::vector<float> topPs = {0.3f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    SamplingParams params;
    params.topKs = topKs;
    params.topPs = topPs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {0}, {0}, {4, 5}, {0}, {4, 5}, // step 0
        {0}, {0}, {0}, {0, 1}, {0}, {0, 1}, // step 1
        {2}, {0}, {0}, {2, 3}, {0}, {2, 3}, // step 2
        {0}, {0}, {0}, {0, 1}, {0}, {0, 1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopP)
{
    uint32_t topK = 0;
    float topP = 0.3f;
    SamplingParams params;
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

TYPED_TEST(TopPSamplingLayerTest, BatchTopP)
{
    std::vector<float> topPs = {0.3f, 0.3f, 0.5f, 0.8f, 0.5f, 0.8f};
    SamplingParams params;
    params.topPs = topPs;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4, 5}, {4, 5, 6}, {4, 5}, {4, 5, 6}, // step 0
        {0}, {0}, {0, 1}, {0, 1, 2}, {0, 1}, {0, 1, 2}, // step 1
        {2}, {2}, {2, 3}, {2, 3, 4}, {2, 3}, {2, 3, 4}, // step 2
        {0}, {0}, {0, 1}, {0, 1, 2}, {0, 1}, {0, 1, 2}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopKBatchTopP)
{
    std::vector<float> topPs = {0.5f, 0.3f, 0.5f, 0.5f, 0.3f, 0.5f};
    SamplingParams params;
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

TYPED_TEST(TopPSamplingLayerTest, TopPTemperature)
{
    float temperature = 0.05f;
    SamplingParams params;
    params.temperatures = {temperature};
    params.topPs = {1.0f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPTemperatureBatch)
{
    std::vector<float> temperatures = {0.05f, 1e3f, 1.0f, 1.0f, 0.05f, 1.0f};
    SamplingParams params;
    params.temperatures = temperatures;
    params.topPs = {0.5f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        {4}, {4, 5, 6, 7}, {4, 5}, {4, 5}, {4}, {4, 5}, // step 0
        {0}, {0, 1, 2, 3}, {0, 1}, {0, 1}, {0}, {0, 1}, // step 1
        {2}, {2, 3, 4, 5}, {2, 3}, {2, 3}, {2}, {2, 3}, // step 2
        {0}, {0, 1, 2, 3}, {0, 1}, {0, 1}, {0}, {0, 1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPRepetitionPenalty)
{
    uint32_t topK = 1;
    float repetitionPenalty = 1e9f;
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPRepetitionPenaltiesBatch)
{
    std::vector<float> repetitionPenalties = {1e9f, 1e9f, 1.0f, 1.0f, 1.0f, 1e9f};
    SamplingParams params;
    params.repetitionPenalties = repetitionPenalties;
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPPresencePenalty)
{
    float presencePenalty = 1e9f;
    SamplingParams params;
    params.presencePenalties = {presencePenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPPresencePenaltiesBatch)
{
    std::vector<float> presencePenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.presencePenalties = presencePenalties;
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPFrequencyPenalty)
{
    float frequencyPenalty = 1e9f;
    SamplingParams params;
    params.frequencyPenalties = {frequencyPenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPFrequencyPenaltiesBatch)
{
    std::vector<float> frequencyPenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.frequencyPenalties = frequencyPenalties;
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPRepetitionPresencePenalty)
{
    float repetitionPenalty = 1e9f;
    float presencePenalty = 1e9f;
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalty};
    params.presencePenalties = {presencePenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPRepetitionPresencePenaltiesBatch)
{
    std::vector<float> repetitionPenalties = {1e9f, 1e9f, 1.0f, 1.0f, 1.0f, 1e9f};
    std::vector<float> presencePenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalties};
    params.presencePenalties = {presencePenalties};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPRepetitionFrequencyPenalty)
{
    float repetitionPenalty = 1e9f;
    float frequencyPenalty = 1e9f;
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalty};
    params.frequencyPenalties = {frequencyPenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPRepetitionFrequencyPenaltiesBatch)
{
    std::vector<float> repetitionPenalties = {1e9f, 1e9f, 1.0f, 1.0f, 1.0f, 1e9f};
    std::vector<float> frequencyPenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalties};
    params.frequencyPenalties = {frequencyPenalties};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPPresenceFrequencyPenalty)
{
    float presencePenalty = 1e9f;
    float frequencyPenalty = 1e9f;
    SamplingParams params;
    params.presencePenalties = {presencePenalty};
    params.frequencyPenalties = {frequencyPenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPPresenceFrequencyPenaltiesBatch)
{
    std::vector<float> presencePenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    std::vector<float> frequencyPenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.presencePenalties = {presencePenalties};
    params.frequencyPenalties = {frequencyPenalties};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPFullPenalty)
{
    float repetitionPenalty = 1e9f;
    float presencePenalty = 1e9f;
    float frequencyPenalty = 1e9f;
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalty};
    params.presencePenalties = {presencePenalty};
    params.frequencyPenalties = {frequencyPenalty};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {1}, {1}, {1}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPFullPenaltiesBatch)
{
    std::vector<float> repetitionPenalties = {1e9f, 1e9f, 1.0f, 1.0f, 1.0f, 1e9f};
    std::vector<float> presencePenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    std::vector<float> frequencyPenalties = {1e9f, 1e9f, 0.0f, 0.0f, 0.0f, 1e9f};
    SamplingParams params;
    params.repetitionPenalties = {repetitionPenalties};
    params.presencePenalties = {presencePenalties};
    params.frequencyPenalties = {frequencyPenalties};
    params.topPs = {0.3f};
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {0}, {0}, {0}, {0}, {0}, {0}, // step 1
        {2}, {2}, {2}, {2}, {2}, {2}, // step 2
        {1}, {1}, {0}, {0}, {0}, {1}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPMinLengthBatch)
{
    std::vector<int32_t> minLengths = {3, 1, 1, 3, 0, 3};
    SamplingParams params;
    params.minLengths = minLengths;
    params.topPs = {0.3f};
    int32_t const endId = 0;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4}, {4}, {4}, {4}, {4}, {4}, // step 0
        {1}, {0}, {0}, {1}, {0}, {1}, // step 1
        {2}, {0}, {0}, {2}, {0}, {2}, // step 2
        {0}, {0}, {0}, {0}, {0}, {0}  // step 3
    };
    this->runTest(expectedOutputIds, params, endId);
}

TYPED_TEST(TopPSamplingLayerTest, TopPBias)
{
    SamplingParams params;
    params.topPs = {0.5f};
    params.useBias = true;
    std::vector<std::set<int32_t>> expectedOutputIds{
        // batch
        {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, // step 0
        {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, // step 1
        {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, // step 2
        {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}  // step 3
    };
    this->runTest(expectedOutputIds, params);
}

TYPED_TEST(TopPSamplingLayerTest, TopPDecay)
{
    SamplingParams params;
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

} // namespace
