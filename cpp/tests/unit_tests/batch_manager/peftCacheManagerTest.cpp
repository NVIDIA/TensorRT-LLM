/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/peftCacheManagerConfig.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/loraCache.h"
#include "tensorrt_llm/runtime/loraModule.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace
{

auto const TEST_RESOURCE_PATH = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources/data";
auto const TEST_SOURCE_LORA_TP1 = TEST_RESOURCE_PATH / "lora-test-weights-tp1/source.npy";
auto const TEST_DEST_LORA_TP1 = TEST_RESOURCE_PATH / "lora-test-weights-tp1/target.npy";
auto const TEST_KEYS_LORA_TP1 = TEST_RESOURCE_PATH / "lora-test-weights-tp1/config.npy";
auto const TEST_KEYS_LORA_TP1_PAGES_RANK0 = TEST_RESOURCE_PATH / "lora-test-weights-tp1/cache_pages_rank0.npy";
auto const TEST_SOURCE_LORA_TP2 = TEST_RESOURCE_PATH / "lora-test-weights-tp2/source.npy";
auto const TEST_DEST_LORA_TP2 = TEST_RESOURCE_PATH / "lora-test-weights-tp2/target.npy";
auto const TEST_KEYS_LORA_TP2 = TEST_RESOURCE_PATH / "lora-test-weights-tp2/config.npy";
auto const TEST_KEYS_LORA_TP2_PAGES_RANK0 = TEST_RESOURCE_PATH / "lora-test-weights-tp2/cache_pages_rank0.npy";
auto const TEST_KEYS_LORA_TP2_PAGES_RANK1 = TEST_RESOURCE_PATH / "lora-test-weights-tp2/cache_pages_rank1.npy";
auto const TEST_PREFETCH = TEST_RESOURCE_PATH / "lora_prefetch";
} // namespace

namespace tensorrt_llm::batch_manager
{

using namespace tensorrt_llm::runtime;

using TensorPtr = ITensor::SharedPtr;

class PeftCacheManagerTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    PeftCacheManagerTest() {}

    void SetUp() override
    {
        mModelConfig = std::make_unique<ModelConfig>(0, 2, 2, 0, 1, 16, nvinfer1::DataType::kFLOAT);
        mModelConfig->setMlpHiddenSize(32);
        mWorldConfig = std::make_unique<WorldConfig>(2, 1, 1, 0);
        std::vector<LoraModule> modules{
            LoraModule(LoraModule::ModuleType::kATTN_QKV, 16, 3 * 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kATTN_Q, 16, 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kATTN_K, 16, 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kATTN_V, 16, 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kATTN_DENSE, 16, 16, false, true, 1, -1),
            LoraModule(LoraModule::ModuleType::kMLP_H_TO_4H, 16, 32, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kMLP_4H_TO_H, 32, 16, false, true, 1, -1),
            LoraModule(LoraModule::ModuleType::kMLP_GATE, 16, 32, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kCROSS_ATTN_QKV, 16, 3 * 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kCROSS_ATTN_Q, 16, 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kCROSS_ATTN_K, 16, 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kCROSS_ATTN_V, 16, 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kCROSS_ATTN_DENSE, 16, 16, false, true, 1, -1),
        };
        mModelConfig->setLoraModules(modules);
        mModelConfig->setMaxLoraRank(64);
        mStream = std::make_shared<CudaStream>();
        mManager = std::make_unique<BufferManager>(mStream);

        PeftCacheManagerConfig config(2 * 8 * 128, 2 * 8 * 92, 8, 64, 1, 1);

        mPeftManager = std::make_unique<PeftCacheManager>(config, *mModelConfig, *mWorldConfig, *mManager);

        loraWeightsTp2 = utils::loadNpy(*mManager, TEST_SOURCE_LORA_TP2.string(), MemoryType::kPINNEDPOOL);
        loraWeightsTp2->unsqueeze(0);
        loraConfigTp2 = utils::loadNpy(*mManager, TEST_KEYS_LORA_TP2.string(), MemoryType::kPINNEDPOOL);
        loraConfigTp2->unsqueeze(0);
    }

    std::shared_ptr<BufferManager> mManager;
    BufferManager::CudaStreamPtr mStream;
    std::unique_ptr<ModelConfig> mModelConfig;
    std::unique_ptr<WorldConfig> mWorldConfig;
    std::unique_ptr<PeftCacheManager> mPeftManager;

    TensorPtr loraWeightsTp2;
    TensorPtr loraConfigTp2;
};

TEST_F(PeftCacheManagerTest, addRequestPeftMissingTask)
{
    SamplingConfig sampleConfig;
    // inSamplingConfig.temperature = std::vector{2.0f};
    uint64_t requestId = 0;
    auto maxNewTokens = 4;
    auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{1, 2, 3, 4});
    auto llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, tokens, sampleConfig, false);

    mPeftManager->addRequestPeft(llmRequest);

    llmRequest->setLoraTaskId(1234);
    EXPECT_THAT([&]() { mPeftManager->addRequestPeft(llmRequest); },
        testing::Throws<PeftTaskNotCachedException>(
            testing::Property(&std::runtime_error::what, testing::HasSubstr("Please send LoRA weights with request"))));

    llmRequest->setLoraWeights(loraWeightsTp2);
    llmRequest->setLoraConfig(loraConfigTp2);

    mPeftManager->addRequestPeft(llmRequest);
}

TEST_F(PeftCacheManagerTest, putGet)
{
    SamplingConfig sampleConfig;
    // inSamplingConfig.temperature = std::vector{2.0f};
    uint64_t requestId = 0;
    auto maxNewTokens = 4;
    auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{1, 2, 3, 4});
    auto llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, tokens, sampleConfig, false);
    llmRequest->setLoraTaskId(1234);
    llmRequest->setLoraWeights(loraWeightsTp2);
    llmRequest->setLoraConfig(loraConfigTp2);

    std::vector<std::shared_ptr<LlmRequest>> reqVector{llmRequest};

    mPeftManager->addRequestPeft(llmRequest);
    auto const peftTable = mPeftManager->ensureBatch(reqVector, {});

    std::vector<LoraCache::TaskLayerModuleConfig> expectedValues{{0, 0, 128, 192, 0, 0, 8, 14},
        {0, 14, 128, 192, 0, 1, 8, 14}, {0, 28, 64, 32, 1, 0, 4, 4}, {0, 32, 64, 32, 1, 1, 4, 4},
        {0, 36, 64, 32, 2, 0, 4, 4}, {0, 40, 64, 32, 2, 1, 4, 4}, {0, 44, 64, 32, 3, 0, 4, 4},
        {0, 48, 64, 32, 3, 1, 4, 4}, {0, 52, 64, 128, 4, 0, 8, 8}, {0, 60, 64, 128, 4, 1, 8, 8},
        {0, 68, 128, 128, 5, 0, 8, 11}, {0, 79, 128, 128, 5, 1, 8, 11}, {0, 90, 128, 128, 6, 0, 8, 11},
        {0, 101, 128, 128, 6, 1, 8, 11}, {0, 112, 128, 128, 7, 0, 8, 11}, {0, 123, 128, 128, 7, 1, 8, 11},
        {0, 134, 128, 192, 8, 0, 8, 14}, {0, 148, 128, 192, 8, 1, 8, 14}, {0, 162, 64, 32, 9, 0, 4, 4},
        {0, 166, 64, 32, 9, 1, 4, 4}, {0, 170, 64, 32, 10, 0, 4, 4}, {0, 174, 64, 32, 10, 1, 4, 4},
        {0, 178, 64, 32, 11, 0, 4, 4}, {0, 182, 64, 32, 11, 1, 4, 4}, {0, 186, 64, 128, 12, 0, 8, 8},
        {0, 194, 64, 128, 12, 1, 8, 8}};

    for (auto [requestId, valuesWeekPtr] : peftTable)
    {
        auto values = valuesWeekPtr;
        for (size_t i = 0; i < values.size(); ++i)
        {
            TLLM_LOG_DEBUG("actual:   " + to_string(values.at(i)));
            TLLM_LOG_DEBUG("expected: " + to_string(expectedValues.at(i)));
            EXPECT_EQ(expectedValues.at(i), values.at(i));
            auto v = values.at(i);
            cudaPointerAttributes attrs;
            cudaError_t err = cudaPointerGetAttributes(&attrs, reinterpret_cast<void*>(v.weightsInPointer));
            ASSERT_EQ(err, 0);
            EXPECT_EQ(attrs.type, cudaMemoryTypeDevice);
            err = cudaPointerGetAttributes(&attrs, reinterpret_cast<void*>(v.weightsOutPointer));
            ASSERT_EQ(err, 0);
            EXPECT_EQ(attrs.type, cudaMemoryTypeDevice);
        }
    }
}

TEST_F(PeftCacheManagerTest, putToCapacity)
{
    SamplingConfig sampleConfig;
    uint64_t requestId = 0;
    auto maxNewTokens = 4;
    auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{1, 2, 3, 4});
    std::map<uint64_t, std::shared_ptr<LlmRequest>> reqTable{};

    auto availablePages = mPeftManager->getMaxHostPages();

    auto constexpr taskStart = 0;
    uint64_t taskId = taskStart;
    bool last = false;
    while (true)
    {
        auto llmRequest = std::make_shared<LlmRequest>(requestId++, maxNewTokens, tokens, sampleConfig, false);
        llmRequest->setLoraTaskId(taskId++);
        llmRequest->setLoraWeights(loraWeightsTp2);
        llmRequest->setLoraConfig(loraConfigTp2);
        auto const neededPages = mPeftManager->determineNumPages(llmRequest);
        if (neededPages <= availablePages || last)
        {
            if (last)
            {
                EXPECT_THAT([&]() { mPeftManager->addRequestPeft(llmRequest); },
                    testing::Throws<std::runtime_error>(testing::Property(
                        &std::runtime_error::what, testing::HasSubstr("There are no done tasks to evict"))));
                break;
            }
            reqTable.insert_or_assign(llmRequest->mRequestId, llmRequest);
            mPeftManager->addRequestPeft(llmRequest, false);
            availablePages -= neededPages;
        }
        else
        {
            last = true;
        }
    }

    for (auto const& [reqId, req] : reqTable)
    {
        std::map<uint64_t, std::string> reqIdToEx{};
        std::vector<std::shared_ptr<LlmRequest>> batchRequests{req};

        PeftCacheManager::PeftTable peftTable;
        try
        {
            peftTable = mPeftManager->ensureBatch(batchRequests, {});
#ifndef NDEBUG
            for (auto const& [requestId, valuesWeakPtr] : peftTable)
            {
                auto const& values = valuesWeakPtr;
                std::cout << requestId << std::endl;
                for (auto const& value : values)
                {
                    std::cout << "\t" << value << std::endl;
                }
            }
#endif
            reqIdToEx.insert_or_assign(reqId, "");
        }
        catch (std::runtime_error& e)
        {
            reqIdToEx.insert_or_assign(reqId, std::string(e.what()));
        }

        for (auto const& [reqId, e] : reqIdToEx)
        {
            std::cout << reqId << " : " << e << std::endl;
            if (reqId < 11)
            {
                EXPECT_EQ("", e);
            }
            else
            {
                EXPECT_TRUE(e.find_first_of("Caught exception during copyTask ensure batch") != std::string::npos);
            }
        }
    }
}

TEST_F(PeftCacheManagerTest, gptManagerSim)
{
    SamplingConfig sampleConfig;
    auto maxNewTokens = 4;
    auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{1, 2, 3, 4});

    auto tpSize = mWorldConfig->getTensorParallelism();
    auto tpRank = mWorldConfig->getTensorParallelRank();

    PeftCacheManagerConfig config(2 * 8 * 128, 2 * 8 * 92, 8, 64, 4, 2);
    auto peftManager = std::make_unique<PeftCacheManager>(config, *mModelConfig, *mWorldConfig, *mManager);

    auto pageConfig = LoraCachePageManagerConfig(
        runtime::MemoryType::kCPU, nvinfer1::DataType::kFLOAT, 128, 128, 2 * 8 * 64, 4 * 16, 1);
    auto loraCache = std::make_unique<LoraCache>(pageConfig, *mModelConfig, *mWorldConfig, *mManager);

    std::map<uint64_t, std::pair<TensorPtr, TensorPtr>> loras;

    SizeType32 constexpr numTasks = 128;
    auto const multiLoraPath = TEST_RESOURCE_PATH / "multi_lora";
    for (SizeType32 taskId = 0; taskId < numTasks; ++taskId)
    {
        auto const weightsFn = (multiLoraPath / std::to_string(taskId) / "source.npy").string();
        auto const configFn = (multiLoraPath / std::to_string(taskId) / "config.npy").string();
        TensorPtr weights = utils::loadNpy(*mManager, weightsFn, runtime::MemoryType::kCPU);
        weights->unsqueeze(0);
        TensorPtr config = utils::loadNpy(*mManager, configFn, runtime::MemoryType::kCPU);
        config->unsqueeze(0);

        loras.insert_or_assign(taskId, std::make_pair(weights, config));
        loraCache->put(taskId, weights, config, true);
    }

    std::vector<std::shared_ptr<LlmRequest>> reqList;
    std::set<SizeType32> seenTasks{};
    SizeType32 constexpr numReqs = 500;
    SizeType32 constexpr batchSize = 4;
    std::map<uint64_t, std::shared_ptr<LlmRequest>> localTable{};
    SizeType32 numReqsWithLora = 0;
    for (SizeType32 reqId = 0; reqId < numReqs; ++reqId)
    {
        auto taskId = std::rand() % (numTasks + 32);
        auto llmRequest = std::make_shared<LlmRequest>(reqId, maxNewTokens, tokens, sampleConfig, false);
        if (taskId < numTasks)
        {
            llmRequest->setLoraTaskId(taskId);
            if (!seenTasks.count(taskId))
            {
                llmRequest->setLoraWeights(std::get<0>(loras.at(taskId)));
                llmRequest->setLoraConfig(std::get<1>(loras.at(taskId)));
            }
            seenTasks.insert(taskId);
            ++numReqsWithLora;
        }
        reqList.push_back(llmRequest);
        if (llmRequest->getLoraTaskId() && !peftManager->isTaskCached(llmRequest->getLoraTaskId().value()))
        {
            llmRequest->setLoraWeights(std::get<0>(loras.at(taskId)));
            llmRequest->setLoraConfig(std::get<1>(loras.at(taskId)));
        }
        peftManager->addRequestPeft(llmRequest);

        localTable.try_emplace(reqId, llmRequest);
        if (localTable.size() == batchSize)
        {
            std::cout << "===============" << localTable.size() << std::endl;
            peftManager->resetDeviceCache();
            std::vector<std::shared_ptr<LlmRequest>> batchRequests{};
            batchRequests.reserve(localTable.size());
            std::transform(localTable.begin(), localTable.end(), std::back_inserter(batchRequests),
                [](auto const& llmReq) { return llmReq.second; });

            auto peftTable = peftManager->ensureBatch(batchRequests, {});
            ASSERT_EQ(numReqsWithLora, peftTable.size());
            for (auto const [id, valuesWeakPtr] : peftTable)
            {
                auto values = valuesWeakPtr;
                EXPECT_TRUE(localTable.find(id) != localTable.end());
                auto hasLora = localTable.at(id)->getLoraTaskId().has_value();
                if (!hasLora)
                {
                    EXPECT_TRUE(values.empty());
                }
                else
                {

                    auto taskId = localTable.at(id)->getLoraTaskId().value();
                    EXPECT_EQ(loras.at(taskId).second->getShape().d[1], values.size());
                    // get target weights from extra cache
                    auto targetValues = loraCache->get(taskId);
                    EXPECT_EQ(targetValues.size(), values.size());

                    auto numVals = targetValues.size();

                    for (size_t valIdx = 0; valIdx < numVals; ++valIdx)
                    {
                        auto v = values.at(valIdx);
                        std::cout << taskId << v << std::endl;
                        auto targetValue = targetValues.at(valIdx);
                        float* weightsInPtr = reinterpret_cast<float*>(v.weightsInPointer);
                        float* weightsOutPtr = reinterpret_cast<float*>(v.weightsOutPointer);

                        TensorPtr hostWeightsIn = mManager->copyFrom(
                            *ITensor::wrap(weightsInPtr, ITensor::makeShape({v.inSize})), runtime::MemoryType::kCPU);
                        TensorPtr hostWeightsOut = mManager->copyFrom(
                            *ITensor::wrap(weightsOutPtr, ITensor::makeShape({v.outSize})), runtime::MemoryType::kCPU);
                        float* hostWeightsInPtr = bufferCast<float>(*hostWeightsIn);
                        float* hostWeightsOutPtr = bufferCast<float>(*hostWeightsOut);

                        float* targetWeightsInPtr = reinterpret_cast<float*>(targetValue.weightsInPointer);
                        float* targetWeightsOutPtr = reinterpret_cast<float*>(targetValue.weightsOutPointer);

                        EXPECT_EQ(targetValue.inSize, v.inSize);
                        EXPECT_EQ(targetValue.outSize, v.outSize);

                        for (size_t i = 0; i < targetValue.inSize; ++i)
                        {
                            ASSERT_FLOAT_EQ(targetWeightsInPtr[i], hostWeightsInPtr[i]);
                        }
                        for (size_t i = 0; i < targetValue.outSize; ++i)
                        {
                            ASSERT_FLOAT_EQ(targetWeightsOutPtr[i], hostWeightsOutPtr[i]);
                        }
                    }
                }
            }
            for (auto const [id, r] : localTable)
            {
                peftManager->markRequestDone(*r);
            }
            localTable.clear();
            numReqsWithLora = 0;
        }
    }
}

TEST_F(PeftCacheManagerTest, updateTaskState)
{
    ASSERT_TRUE(mPeftManager->getActiveTasks().empty());
    ASSERT_TRUE(mPeftManager->getPausedTasks().empty());

    mPeftManager->updateTaskState(0, 1, true, false);
    ASSERT_TRUE(mPeftManager->getActiveTasks().empty());
    ASSERT_TRUE(mPeftManager->getPausedTasks().empty());

    mPeftManager->updateTaskState(0, 1, true, true);
    ASSERT_TRUE(mPeftManager->getActiveTasks().empty());
    ASSERT_EQ(1, mPeftManager->getPausedTasks().size());
    EXPECT_TRUE(mPeftManager->getPausedTasks().at(0).count(1));

    mPeftManager->updateTaskState(0, 1);
    ASSERT_EQ(1, mPeftManager->getActiveTasks().size());
    EXPECT_TRUE(mPeftManager->getActiveTasks().at(0).count(1));
    ASSERT_TRUE(mPeftManager->getPausedTasks().empty());

    mPeftManager->updateTaskState(0, 1);
    ASSERT_EQ(1, mPeftManager->getActiveTasks().size());
    EXPECT_TRUE(mPeftManager->getActiveTasks().at(0).count(1));
    ASSERT_TRUE(mPeftManager->getPausedTasks().empty());

    SamplingConfig sampleConfig;
    auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{1, 2, 3, 4});
    auto llmRequest = std::make_shared<LlmRequest>(12345, 4, tokens, sampleConfig, false);
    auto llmRequest2 = std::make_shared<LlmRequest>(54321, 4, tokens, sampleConfig, false);
    llmRequest->setLoraTaskId(1234);
    llmRequest->setLoraWeights(loraWeightsTp2);
    llmRequest->setLoraConfig(loraConfigTp2);

    llmRequest2->setLoraTaskId(1234);

    std::vector<std::shared_ptr<LlmRequest>> reqVector{llmRequest};

    mPeftManager->addRequestPeft(llmRequest);
    mPeftManager->addRequestPeft(llmRequest2);
    ASSERT_EQ(2, mPeftManager->getActiveTasks().size());
    ASSERT_EQ(2, mPeftManager->getActiveTasks().at(1234).size());
    EXPECT_TRUE(mPeftManager->getActiveTasks().at(0).count(1));
    EXPECT_TRUE(mPeftManager->getActiveTasks().at(1234).count(12345));
    EXPECT_TRUE(mPeftManager->getActiveTasks().at(1234).count(54321));
    ASSERT_TRUE(mPeftManager->getPausedTasks().empty());

    // sync with copy threads and populate device cache
    auto peftTable = mPeftManager->ensureBatch(reqVector, {});

    mPeftManager->markRequestDone(*llmRequest2, true);
    ASSERT_EQ(2, mPeftManager->getActiveTasks().size());
    ASSERT_EQ(1, mPeftManager->getActiveTasks().at(1234).size());
    EXPECT_TRUE(mPeftManager->getActiveTasks().at(0).count(1));
    EXPECT_TRUE(mPeftManager->getActiveTasks().at(1234).count(12345));
    ASSERT_EQ(1, mPeftManager->getPausedTasks().size());
    ASSERT_TRUE(mPeftManager->getPausedTasks().at(1234).count(54321));

    EXPECT_TRUE(mPeftManager->isTaskCached(1234));
    EXPECT_FALSE(mPeftManager->isTaskDone(1234));
    EXPECT_FALSE(mPeftManager->isTaskDoneDevice(1234));

    mPeftManager->markRequestDone(*llmRequest, true);
    ASSERT_EQ(1, mPeftManager->getActiveTasks().size());
    EXPECT_TRUE(mPeftManager->getActiveTasks().at(0).count(1));
    ASSERT_EQ(1, mPeftManager->getPausedTasks().size());
    ASSERT_TRUE(mPeftManager->getPausedTasks().at(1234).count(54321));
    ASSERT_TRUE(mPeftManager->getPausedTasks().at(1234).count(12345));

    EXPECT_TRUE(mPeftManager->isTaskCached(1234));
    EXPECT_FALSE(mPeftManager->isTaskDone(1234));
    EXPECT_TRUE(mPeftManager->isTaskDoneDevice(1234));

    mPeftManager->markRequestDone(*llmRequest, false);
    ASSERT_EQ(1, mPeftManager->getActiveTasks().size());
    EXPECT_TRUE(mPeftManager->getActiveTasks().at(0).count(1));
    ASSERT_EQ(1, mPeftManager->getPausedTasks().size());
    ASSERT_TRUE(mPeftManager->getPausedTasks().at(1234).count(54321));

    EXPECT_TRUE(mPeftManager->isTaskCached(1234));
    EXPECT_FALSE(mPeftManager->isTaskDone(1234));
    EXPECT_TRUE(mPeftManager->isTaskDoneDevice(1234));

    mPeftManager->markRequestDone(*llmRequest2, false);
    ASSERT_EQ(1, mPeftManager->getActiveTasks().size());
    EXPECT_TRUE(mPeftManager->getActiveTasks().at(0).count(1));
    ASSERT_EQ(0, mPeftManager->getPausedTasks().size());

    EXPECT_TRUE(mPeftManager->isTaskCached(1234));
    EXPECT_TRUE(mPeftManager->isTaskDone(1234));
    EXPECT_TRUE(mPeftManager->isTaskDoneDevice(1234));
}

TEST_F(PeftCacheManagerTest, getMaxNumSlots)
{
    PeftCacheManagerConfig config;
    config.numHostModuleLayer = 8192 * 8;
    config.numDeviceModuleLayer = 8292 * 2;
    auto [hostSlots, deviceSlots]
        = PeftCacheManager::getMaxNumSlots(config, nvinfer1::DataType::kHALF, 256, 4 * 256, *mManager);
    EXPECT_EQ(262144, hostSlots);
    EXPECT_EQ(66336, deviceSlots);

    config.hostCacheSize = 100000;
    config.numHostModuleLayer = 0;
    config.maxAdapterSize = 8;
    config.maxPagesPerBlockHost = 4;
    config.maxPagesPerBlockDevice = 8;

    std::tie(hostSlots, deviceSlots)
        = PeftCacheManager::getMaxNumSlots(config, nvinfer1::DataType::kHALF, 256, 4 * 256, *mManager);

    EXPECT_EQ(195, hostSlots);
    EXPECT_EQ(66336, deviceSlots);

    std::tie(hostSlots, deviceSlots)
        = PeftCacheManager::getMaxNumSlots(config, nvinfer1::DataType::kFLOAT, 384, 4 * 1024, *mManager);

    config.hostCacheSize = 100000000;
    config.numHostModuleLayer = 8291 * 2;

    EXPECT_EQ(65, hostSlots);
    EXPECT_EQ(182424, deviceSlots);
}

TEST_F(PeftCacheManagerTest, getPageManagerConfig)
{
    PeftCacheManagerConfig config;
    config.numHostModuleLayer = 8192 * 8;
    config.numDeviceModuleLayer = 8292 * 2;
    auto [hostCfg, deviceCfg] = PeftCacheManager::getPageManagerConfig(config, *mModelConfig, *mWorldConfig, *mManager);

    EXPECT_EQ(runtime::MemoryType::kCPU, hostCfg.getMemoryType());
    EXPECT_EQ(nvinfer1::DataType::kFLOAT, hostCfg.getDataType());
    EXPECT_EQ(456, hostCfg.getTotalNumPages());
    EXPECT_EQ(24, hostCfg.getMaxPagesPerBlock());
    EXPECT_EQ(288, hostCfg.getSlotsPerPage());
    EXPECT_EQ(24, hostCfg.getPageWidth());
    EXPECT_FALSE(hostCfg.getInitToZero());

    EXPECT_EQ(runtime::MemoryType::kGPU, deviceCfg.getMemoryType());
    EXPECT_EQ(nvinfer1::DataType::kFLOAT, deviceCfg.getDataType());
    EXPECT_EQ(116, deviceCfg.getTotalNumPages());
    EXPECT_EQ(8, deviceCfg.getMaxPagesPerBlock());
    EXPECT_EQ(288, deviceCfg.getSlotsPerPage());
    EXPECT_EQ(24, deviceCfg.getPageWidth());
    EXPECT_FALSE(deviceCfg.getInitToZero());

    config.hostCacheSize = 100000000;
    config.numHostModuleLayer = 0;
    config.maxAdapterSize = 8;
    config.maxPagesPerBlockHost = 4;
    config.maxPagesPerBlockDevice = 8;
    std::tie(hostCfg, deviceCfg)
        = PeftCacheManager::getPageManagerConfig(config, *mModelConfig, *mWorldConfig, *mManager);

    EXPECT_EQ(runtime::MemoryType::kCPU, hostCfg.getMemoryType());
    EXPECT_EQ(nvinfer1::DataType::kFLOAT, hostCfg.getDataType());
    EXPECT_EQ(3617, hostCfg.getTotalNumPages());
    EXPECT_EQ(4, hostCfg.getMaxPagesPerBlock());
    EXPECT_EQ(288, hostCfg.getSlotsPerPage());
    EXPECT_EQ(24, hostCfg.getPageWidth());
    EXPECT_FALSE(hostCfg.getInitToZero());

    EXPECT_EQ(runtime::MemoryType::kGPU, deviceCfg.getMemoryType());
    EXPECT_EQ(nvinfer1::DataType::kFLOAT, deviceCfg.getDataType());
    EXPECT_EQ(116, deviceCfg.getTotalNumPages());
    EXPECT_EQ(8, deviceCfg.getMaxPagesPerBlock());
    EXPECT_EQ(288, deviceCfg.getSlotsPerPage());
    EXPECT_EQ(24, deviceCfg.getPageWidth());
    EXPECT_FALSE(deviceCfg.getInitToZero());
}

class PeftCacheManagerPrefetchTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    PeftCacheManagerPrefetchTest() {}

    void SetUp() override
    {
        mModelConfig = std::make_unique<ModelConfig>(0, 2, 2, 0, 1, 16, nvinfer1::DataType::kFLOAT);
        mModelConfig->setMlpHiddenSize(32);
        mWorldConfig = std::make_unique<WorldConfig>(2, 1, 1, 0);
        std::vector<LoraModule> modules{
            LoraModule(LoraModule::ModuleType::kATTN_QKV, 16, 3 * 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kATTN_Q, 16, 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kATTN_K, 16, 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kATTN_V, 16, 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kATTN_DENSE, 16, 16, false, true, 1, -1),
            LoraModule(LoraModule::ModuleType::kMLP_H_TO_4H, 16, 32, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kMLP_4H_TO_H, 32, 16, false, true, 1, -1),
            LoraModule(LoraModule::ModuleType::kMLP_GATE, 16, 32, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kCROSS_ATTN_QKV, 16, 3 * 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kCROSS_ATTN_Q, 16, 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kCROSS_ATTN_K, 16, 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kCROSS_ATTN_V, 16, 16, false, true, -1, 0),
            LoraModule(LoraModule::ModuleType::kCROSS_ATTN_DENSE, 16, 16, false, true, 1, -1),
        };
        mModelConfig->setLoraModules(modules);
        mModelConfig->setMaxLoraRank(64);
        mStream = std::make_shared<CudaStream>();
        mManager = std::make_unique<BufferManager>(mStream);

        PeftCacheManagerConfig config(
            2 * 8 * 128, 2 * 8 * 92, 8, 64, 1, 1, 1, 24, 8, std::nullopt, std::nullopt, TEST_PREFETCH.string());

        mPeftManager = std::make_unique<PeftCacheManager>(config, *mModelConfig, *mWorldConfig, *mManager);
    }

    std::shared_ptr<BufferManager> mManager;
    BufferManager::CudaStreamPtr mStream;
    std::unique_ptr<ModelConfig> mModelConfig;
    std::unique_ptr<WorldConfig> mWorldConfig;
    std::unique_ptr<PeftCacheManager> mPeftManager;
};

TEST_F(PeftCacheManagerPrefetchTest, prefetch)
{
    EXPECT_TRUE(mPeftManager->isTaskCached(3));
    EXPECT_TRUE(mPeftManager->isTaskCached(5));
}

} // namespace tensorrt_llm::batch_manager
