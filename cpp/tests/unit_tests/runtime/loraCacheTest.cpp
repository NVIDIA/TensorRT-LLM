/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include "tensorrt_llm/runtime/loraCache.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/loraCachePageManagerConfig.h"
#include "tensorrt_llm/runtime/loraModule.h"
#include "tensorrt_llm/runtime/loraUtils.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <NvInferRuntime.h>

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

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

auto const TEST_SOURCE_DORA_TP1 = TEST_RESOURCE_PATH / "lora-test-weights-tp1/source_dora.npy";
auto const TEST_DEST_DORA_TP1 = TEST_RESOURCE_PATH / "lora-test-weights-tp1/target_dora.npy";
auto const TEST_KEYS_DORA_TP1 = TEST_RESOURCE_PATH / "lora-test-weights-tp1/config_dora.npy";
auto const TEST_KEYS_DORA_TP1_PAGES_RANK0 = TEST_RESOURCE_PATH / "lora-test-weights-tp1/cache_pages_rank0_dora.npy";
auto const TEST_SOURCE_DORA_TP2 = TEST_RESOURCE_PATH / "lora-test-weights-tp2/source_dora.npy";
auto const TEST_DEST_DORA_TP2 = TEST_RESOURCE_PATH / "lora-test-weights-tp2/target_dora.npy";
auto const TEST_KEYS_DORA_TP2 = TEST_RESOURCE_PATH / "lora-test-weights-tp2/config_dora.npy";
auto const TEST_KEYS_DORA_TP2_PAGES_RANK0 = TEST_RESOURCE_PATH / "lora-test-weights-tp2/cache_pages_rank0_dora.npy";
auto const TEST_KEYS_DORA_TP2_PAGES_RANK1 = TEST_RESOURCE_PATH / "lora-test-weights-tp2/cache_pages_rank1_dora.npy";
} // namespace

namespace tensorrt_llm::runtime
{

using TensorPtr = ITensor::SharedPtr;
using ParamType = bool;

class LoraCacheTest : public ::testing::Test,
                      public ::testing::WithParamInterface<ParamType> // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    LoraCacheTest() {}

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
        mStream = std::make_shared<CudaStream>();
        mManager = std::make_unique<BufferManager>(mStream);

        auto pageConfig = LoraCachePageManagerConfig(
            runtime::MemoryType::kCPU, nvinfer1::DataType::kFLOAT, 2 * 8, 6, 64, 4 * 16, 1);
        pageConfig.setInitToZero(true);
        auto pageConfig2 = pageConfig;
        pageConfig2.setInitToZero(true);
        pageConfig2.setMemoryType(runtime::MemoryType::kGPU);
        mLoraCache = std::make_unique<LoraCache>(pageConfig, *mModelConfig, *mWorldConfig, *mManager);
        mLoraCache2 = std::make_unique<LoraCache>(pageConfig2, *mModelConfig, *mWorldConfig, *mManager);
    }

    std::shared_ptr<BufferManager> mManager;
    BufferManager::CudaStreamPtr mStream;
    std::unique_ptr<ModelConfig> mModelConfig;
    std::unique_ptr<WorldConfig> mWorldConfig;
    std::unique_ptr<LoraCache> mLoraCache;
    std::unique_ptr<LoraCache> mLoraCache2;
};

TEST_F(LoraCacheTest, LoraCachePageManagerTest)
{
    SizeType32 constexpr maxAdapterSize = 4;
    SizeType32 constexpr maxAdapterWeights = 8;
    auto pageShape = ITensor::makeShape({maxAdapterSize, maxAdapterWeights});

    LoraCachePageManagerConfig config(
        runtime::MemoryType::kCPU, nvinfer1::DataType::kFLOAT, 8, 6, maxAdapterSize, maxAdapterWeights, 1);
    LoraCachePageManager manager(config, *mManager);

    auto block0 = manager.blockPtr(0);
    auto block1 = manager.blockPtr(1);

    auto expectedBlockShape0 = ITensor::makeShape({6, maxAdapterSize, maxAdapterWeights});
    auto expectedBlockShape1 = ITensor::makeShape({2, maxAdapterSize, maxAdapterWeights});
    EXPECT_TRUE(ITensor::shapeEquals(block0->getShape(), expectedBlockShape0));
    EXPECT_TRUE(ITensor::shapeEquals(block1->getShape(), expectedBlockShape1));

    std::vector<ITensor::SharedConstPtr> expectedPages;
    for (SizeType32 blockIdx = 0; blockIdx < 2; ++blockIdx)
    {
        auto block = blockIdx == 0 ? block0 : block1;
        for (SizeType32 i = 0; i < (blockIdx == 0 ? 6 : 2); ++i)
        {
            TensorPtr page = ITensor::slice(
                ITensor::wrap(const_cast<void*>(block->data()), block->getDataType(), block->getShape()), i, 1);
            page->squeeze(0);
            expectedPages.push_back(page);
        }
    }

    // auto [claimed, singlePageId] = manager.claimPages(1);
    auto singlePageId = manager.claimPages(1);
    ASSERT_TRUE(singlePageId.has_value());
    ASSERT_EQ(singlePageId.value().size(), 1);
    auto singlePage = manager.pagePtr(singlePageId.value().at(0));
    EXPECT_EQ(singlePage->data(), expectedPages.at(0)->data());

    // auto [claimed2, pages] = manager.claimPages(7);
    auto pages = manager.claimPages(7);
    ASSERT_TRUE(pages.has_value());
    EXPECT_EQ(pages.value().size(), 7);
    for (std::size_t i = 1; i < 8; ++i)
    {
        EXPECT_EQ(manager.pagePtr(pages.value().at(i - 1))->data(), expectedPages.at(i)->data());
    }

    // auto [claimed3, empty1] = manager.claimPages(1);
    auto empty1 = manager.claimPages(1);
    ASSERT_FALSE(empty1.has_value());
    // EXPECT_EQ(empty1.size(), 0);

    manager.releasePages(std::move(singlePageId.value()));
    // auto [claimed4, singlePageId2] = manager.claimPages(1);
    auto singlePageId2 = manager.claimPages(1);

    // check that page slots are freed when it's released
    ASSERT_TRUE(singlePageId2.has_value());
    EXPECT_EQ(singlePageId2.value().size(), 1);
    EXPECT_EQ(manager.pagePtr(singlePageId2.value().at(0))->data(), expectedPages.at(0)->data());
}

TEST_F(LoraCacheTest, determineNumPages)
{
    ModelConfig modelConfig(0, 2, 2, 0, 1, 4, nvinfer1::DataType::kFLOAT);
    modelConfig.setLoraModules(LoraModule::createLoraModules({"attn_dense", "attn_qkv"}, 4, 4, 1, 1, 2, 2, 0));
    WorldConfig worldConfig(1, 1, 1, 0);

    LoraCachePageManagerConfig pageConfig(MemoryType::kCPU, nvinfer1::DataType::kFLOAT, 12393, 40, 80, 16, 1);

    LoraCache cache(pageConfig, modelConfig, worldConfig, *mManager);

    std::vector<int32_t> loraConfigVector{
        0,
        0,
        64,
        0,
        1,
        64,
    };
    TensorPtr loraConfig = ITensor::wrap(
        loraConfigVector, ITensor::makeShape({static_cast<SizeType32>(loraConfigVector.size()) / 3, 3}));
    auto numPages = cache.determineNumPages(loraConfig);
    EXPECT_EQ(numPages, 2);

    loraConfigVector = std::vector<int32_t>{
        0,
        0,
        32,
        0,
        0,
        32,
        0,
        0,
        64,
    };

    loraConfig = ITensor::wrap(
        loraConfigVector, ITensor::makeShape({static_cast<SizeType32>(loraConfigVector.size()) / 3, 3}));
    numPages = cache.determineNumPages(loraConfig);
    EXPECT_EQ(numPages, 2);

    loraConfigVector = std::vector<int32_t>{
        0,
        0,
        32,
        0,
        0,
        32,
        0,
        0,
        64,
        0,
        0,
        24,
    };

    loraConfig = ITensor::wrap(
        loraConfigVector, ITensor::makeShape({static_cast<SizeType32>(loraConfigVector.size()) / 3, 3}));
    numPages = cache.determineNumPages(loraConfig);
    EXPECT_EQ(numPages, 3);

    loraConfigVector = std::vector<int32_t>{
        0,
        0,
        60,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
    };

    loraConfig = ITensor::wrap(
        loraConfigVector, ITensor::makeShape({static_cast<SizeType32>(loraConfigVector.size()) / 3, 3}));
    numPages = cache.determineNumPages(loraConfig);
    EXPECT_EQ(numPages, 2);

    loraConfigVector = std::vector<int32_t>{
        0,
        0,
        60,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        64,
        0,
        0,
        1,
    };

    loraConfig = ITensor::wrap(
        loraConfigVector, ITensor::makeShape({static_cast<SizeType32>(loraConfigVector.size()) / 3, 3}));
    numPages = cache.determineNumPages(loraConfig);
    EXPECT_EQ(numPages, 4);
}

TEST_F(LoraCacheTest, basicPutGet)
{
    TensorPtr loraReqWeights = utils::loadNpy(*mManager, TEST_SOURCE_LORA_TP2.string(), MemoryType::kCPU);
    TensorPtr loraReqKeys = utils::loadNpy(*mManager, TEST_KEYS_LORA_TP2.string(), MemoryType::kCPU);
    TensorPtr loraDestWeights = utils::loadNpy(*mManager, TEST_DEST_LORA_TP2.string(), MemoryType::kCPU);

    EXPECT_FALSE(mLoraCache->has(1234));
    mLoraCache->put(1234, loraReqWeights, loraReqKeys);
    EXPECT_TRUE(mLoraCache->has(1234));
    EXPECT_TRUE(mLoraCache->isLoaded(1234));
    auto const& values = mLoraCache->get(1234);

    std::vector<LoraCache::TaskLayerModuleConfig> expectedValues{{0, 0, 128, 192, 0, 0, 8, 5},
        {0, 5, 128, 192, 0, 1, 8, 5}, {0, 10, 64, 32, 1, 0, 4, 2}, {0, 12, 64, 32, 1, 1, 4, 2},
        {0, 14, 64, 32, 2, 0, 4, 2}, {0, 16, 64, 32, 2, 1, 4, 2}, {0, 18, 64, 32, 3, 0, 4, 2},
        {0, 20, 64, 32, 3, 1, 4, 2}, {0, 22, 64, 128, 4, 0, 8, 3}, {0, 25, 64, 128, 4, 1, 8, 3},
        {0, 28, 128, 128, 5, 0, 8, 4}, {0, 32, 128, 128, 5, 1, 8, 4}, {0, 36, 128, 128, 6, 0, 8, 4},
        {0, 40, 128, 128, 6, 1, 8, 4}, {0, 44, 128, 128, 7, 0, 8, 4}, {0, 48, 128, 128, 7, 1, 8, 4},
        {0, 52, 128, 192, 8, 0, 8, 5}, {0, 57, 128, 192, 8, 1, 8, 5}, {0, 62, 64, 32, 9, 0, 4, 2},
        {1, 0, 64, 32, 9, 1, 4, 2}, {1, 2, 64, 32, 10, 0, 4, 2}, {1, 4, 64, 32, 10, 1, 4, 2},
        {1, 6, 64, 32, 11, 0, 4, 2}, {1, 8, 64, 32, 11, 1, 4, 2}, {1, 10, 64, 128, 12, 0, 8, 3},
        {1, 13, 64, 128, 12, 1, 8, 3}};

    ASSERT_EQ(expectedValues.size(), values.size());
    for (std::size_t i = 0; i < expectedValues.size(); ++i)
    {
        EXPECT_EQ(expectedValues.at(i), values.at(i));
    }

    ASSERT_EQ(values.size(), loraDestWeights->getShape().d[0]);
    auto const tpSize = mWorldConfig->getTensorParallelism();
    for (size_t i = 0; i < values.size(); ++i)
    {
        auto const configRowPtr = bufferCast<int32_t>(*ITensor::slice(loraReqKeys, i, 1));
        auto const modId = configRowPtr[lora::kLORA_CONFIG_MODULE_OFF];
        auto const adapterSize = configRowPtr[lora::kLORA_CONFIG_ADAPTER_SIZE_OFF];
        auto modIt = std::find_if(mModelConfig->getLoraModules().begin(), mModelConfig->getLoraModules().end(),
            [modId = modId](LoraModule const& m) { return m.value() == modId; });
        auto const inSize = modIt->inSize(adapterSize);
        auto const outSize = modIt->outSize(adapterSize);

        float const* weightsInPtr = reinterpret_cast<float*>(values[i].weightsInPointer);
        float const* weightsOutPtr = reinterpret_cast<float*>(values[i].weightsOutPointer);

        TensorPtr row = ITensor::slice(loraDestWeights, i, 1);
        auto const rowSize = static_cast<SizeType32>(ITensor::volume(row->getShape()));
        TensorPtr rowFlatView = ITensor::view(row, ITensor::makeShape({rowSize}));
        TensorPtr expectedIn = ITensor::slice(rowFlatView, 0, inSize);
        TensorPtr expectedOut = ITensor::slice(rowFlatView, inSize, outSize);
        auto const expectedInPtr = bufferCast<float>(*expectedIn);
        auto const expectedOutPtr = bufferCast<float>(*expectedOut);
        for (size_t j = 0; j < values.at(i).inSize; ++j)
        {
            EXPECT_FLOAT_EQ(weightsInPtr[j], expectedInPtr[j]);
        }
        for (size_t j = 0; j < values.at(i).outSize; ++j)
        {
            EXPECT_FLOAT_EQ(weightsOutPtr[j], expectedOutPtr[j]);
        }
    }

    mLoraCache->copyTask(1234, *mLoraCache2);

    auto const& values2 = mLoraCache2->get(1234);
    ASSERT_EQ(values.size(), values2.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        EXPECT_EQ(values.at(i), values2.at(i));
        auto page1 = mLoraCache->getPagePtr(values.at(i).pageId);
        auto page2 = mLoraCache2->getPagePtr(values2.at(i).pageId);
        auto hostPage2 = mManager->copyFrom(*page2, runtime::MemoryType::kCPU);
        ASSERT_TRUE(ITensor::shapeEquals(page1->getShape(), page2->getShape()));
        auto const pageSize = page1->getSize();
        auto const p1 = bufferCast<float>(*page1);
        auto const p2 = bufferCast<float>(*hostPage2);
        for (size_t i = 0; i < static_cast<size_t>(pageSize); ++i)
        {
            ASSERT_FLOAT_EQ(p1[i], p2[i]);
        }
    }
}

TEST_F(LoraCacheTest, splitTransposeCpu)
{
    auto modelConfig = ModelConfig(0, 2, 2, 0, 1, 16, nvinfer1::DataType::kFLOAT);
    auto worldConfig = WorldConfig(2, 1, 1, 0);

    SizeType32 const split{2};
    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};
    std::vector<std::int32_t> const outputRank0{28524, 5093, 23316, 11, 263, 355};
    std::vector<std::int32_t> const outputRank1{287, 12, 4881, 30022, 8776, 257};
    std::vector<std::int32_t> const output2Rank0{28524, 287, 23316, 4881, 263, 8776};
    std::vector<std::int32_t> const output2Rank1{5093, 12, 11, 30022, 355, 257};

    {
        SizeType32 const batchSize{6};
        auto const inputLength = static_cast<SizeType32>(input.size() / batchSize);
        auto const inputShape = ITensor::makeShape({batchSize, inputLength});
        auto const outputShape = ITensor::makeShape({batchSize, inputLength / split});

        auto inputTensor = mManager->copyFrom(input, inputShape, MemoryType::kCPU);
        auto outputTensorRank0 = mManager->cpu(outputShape, nvinfer1::DataType::kINT32);
        auto outputTensorRank1 = mManager->cpu(outputShape, nvinfer1::DataType::kINT32);
        mManager->setZero(*outputTensorRank0);
        mManager->setZero(*outputTensorRank1);

        auto outputPtrRank0 = bufferCast<SizeType32>(*outputTensorRank0);
        auto outputPtrRank1 = bufferCast<SizeType32>(*outputTensorRank1);

        LoraCache::splitTransposeCpu(*outputTensorRank0, *inputTensor, split, 0);
        LoraCache::splitTransposeCpu(*outputTensorRank1, *inputTensor, split, 1);

        for (SizeType32 i = 0; i < static_cast<SizeType32>(outputRank0.size()); ++i)
        {
            EXPECT_EQ(outputPtrRank0[i], outputRank0[i]);
            EXPECT_EQ(outputPtrRank1[i], outputRank1[i]);
        }
    }

    {
        SizeType32 const batchSize{3};
        auto const inputLength = static_cast<SizeType32>(input.size() / batchSize);
        auto const inputShape = ITensor::makeShape({batchSize, inputLength});
        auto const outputShape = ITensor::makeShape({batchSize, inputLength / split});

        auto inputTensor = mManager->copyFrom(input, inputShape, MemoryType::kCPU);
        auto outputTensorRank0 = mManager->cpu(outputShape, nvinfer1::DataType::kINT32);
        auto outputTensorRank1 = mManager->cpu(outputShape, nvinfer1::DataType::kINT32);
        mManager->setZero(*outputTensorRank0);
        mManager->setZero(*outputTensorRank1);

        LoraCache::splitTransposeCpu(*outputTensorRank0, *inputTensor, split, 0);
        LoraCache::splitTransposeCpu(*outputTensorRank1, *inputTensor, split, 1);

        auto outputPtrRank0 = bufferCast<SizeType32>(*outputTensorRank0);
        auto outputPtrRank1 = bufferCast<SizeType32>(*outputTensorRank1);

        for (SizeType32 i = 0; i < static_cast<SizeType32>(outputRank0.size()); ++i)
        {
            EXPECT_EQ(outputPtrRank0[i], output2Rank0[i]);
            EXPECT_EQ(outputPtrRank1[i], output2Rank1[i]);
        }
    }
}

TEST_P(LoraCacheTest, copyToPages_tp1)
{
    bool const isDora = GetParam();
    auto modelConfig = ModelConfig(0, 2, 2, 0, 1, 16, nvinfer1::DataType::kFLOAT);
    modelConfig.setMlpHiddenSize(32);
    auto worldConfig = WorldConfig(1, 1, 1, 0);
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
    modelConfig.setLoraModules(modules);
    std::unordered_map<SizeType32, LoraModule> moduleIdToModule;
    for (auto const& m : modelConfig.getLoraModules())
    {
        moduleIdToModule[m.value()] = m;
    }

    TensorPtr loraReqWeights = utils::loadNpy(
        *mManager, isDora ? TEST_SOURCE_DORA_TP1.string() : TEST_SOURCE_LORA_TP1.string(), MemoryType::kCPU);
    loraReqWeights->unsqueeze(0);
    TensorPtr loraReqKeys = utils::loadNpy(
        *mManager, isDora ? TEST_KEYS_DORA_TP1.string() : TEST_KEYS_LORA_TP1.string(), MemoryType::kCPU);
    loraReqKeys->unsqueeze(0);
    TensorPtr loraTargetTensors = utils::loadNpy(
        *mManager, isDora ? TEST_DEST_DORA_TP1.string() : TEST_DEST_LORA_TP1.string(), MemoryType::kCPU);

    TensorPtr targetPageBlock = utils::loadNpy(*mManager,
        isDora ? TEST_KEYS_DORA_TP1_PAGES_RANK0.string() : TEST_KEYS_LORA_TP1_PAGES_RANK0.string(), MemoryType::kCPU);
    TensorPtr pageBlock = mManager->cpu(targetPageBlock->getShape(), targetPageBlock->getDataType());
    mManager->setZero(*pageBlock);
    std::vector<TensorPtr> pages;
    for (SizeType32 p = 0; p < pageBlock->getShape().d[0]; ++p)
    {
        pages.push_back(ITensor::view(ITensor::slice(pageBlock, p, 1),
            ITensor::makeShape({pageBlock->getShape().d[1], pageBlock->getShape().d[2]})));
    }
    std::vector<std::size_t> pageIds{};
    pageIds.resize(pages.size());
    std::iota(pageIds.begin(), pageIds.end(), 0);

    auto locations = LoraCache::copyToPages(
        loraReqWeights, loraReqKeys, modelConfig, worldConfig, moduleIdToModule, *mManager, pages, pageIds);

    auto pagePtr = bufferCast<float>(*pageBlock);
    auto targetPtr = bufferCast<float>(*targetPageBlock);

    for (SizeType32 i = 0; i < pageBlock->getSize(); ++i)
    {
        EXPECT_FLOAT_EQ(pagePtr[i], targetPtr[i]);
    }
}

TEST_P(LoraCacheTest, copyToPages_tp2_rank0)
{
    bool const isDora = GetParam();
    auto modelConfig = ModelConfig(0, 2, 2, 0, 1, 16, nvinfer1::DataType::kFLOAT);
    modelConfig.setMlpHiddenSize(32);
    auto worldConfig = WorldConfig(2, 1, 1, 0);
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
    modelConfig.setLoraModules(modules);
    std::unordered_map<SizeType32, LoraModule> moduleIdToModule;
    for (auto const& m : modelConfig.getLoraModules())
    {
        moduleIdToModule[m.value()] = m;
    }

    TensorPtr loraReqWeights = utils::loadNpy(
        *mManager, isDora ? TEST_SOURCE_DORA_TP2.string() : TEST_SOURCE_LORA_TP2.string(), MemoryType::kCPU);
    loraReqWeights->unsqueeze(0);
    TensorPtr loraReqKeys = utils::loadNpy(
        *mManager, isDora ? TEST_KEYS_DORA_TP2.string() : TEST_KEYS_LORA_TP2.string(), MemoryType::kCPU);
    loraReqKeys->unsqueeze(0);

    TensorPtr targetPageBlock = utils::loadNpy(*mManager,
        isDora ? TEST_KEYS_DORA_TP2_PAGES_RANK0.string() : TEST_KEYS_LORA_TP2_PAGES_RANK0.string(), MemoryType::kCPU);
    TensorPtr pageBlock = mManager->cpu(targetPageBlock->getShape(), targetPageBlock->getDataType());
    mManager->setZero(*pageBlock);
    std::vector<TensorPtr> pages;
    for (SizeType32 p = 0; p < pageBlock->getShape().d[0]; ++p)
    {
        pages.push_back(ITensor::view(ITensor::slice(pageBlock, p, 1),
            ITensor::makeShape({pageBlock->getShape().d[1], pageBlock->getShape().d[2]})));
    }
    std::vector<std::size_t> pageIds{};
    pageIds.resize(pages.size());
    std::iota(pageIds.begin(), pageIds.end(), 0);

    auto locations = LoraCache::copyToPages(
        loraReqWeights, loraReqKeys, modelConfig, worldConfig, moduleIdToModule, *mManager, pages, pageIds);

    auto pagePtr = bufferCast<float>(*pageBlock);
    auto targetPtr = bufferCast<float>(*targetPageBlock);

    for (SizeType32 i = 0; i < pageBlock->getSize(); ++i)
    {
        EXPECT_FLOAT_EQ(pagePtr[i], targetPtr[i]);
    }
}

TEST_P(LoraCacheTest, copyToPages_tp2_rank1)
{
    bool const isDora = GetParam();
    auto modelConfig = ModelConfig(0, 2, 2, 0, 1, 16, nvinfer1::DataType::kFLOAT);
    modelConfig.setMlpHiddenSize(32);
    auto worldConfig = WorldConfig(2, 1, 1, 1);
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
    modelConfig.setLoraModules(modules);
    std::unordered_map<SizeType32, LoraModule> moduleIdToModule;
    for (auto const& m : modelConfig.getLoraModules())
    {
        moduleIdToModule[m.value()] = m;
    }

    TensorPtr loraReqWeights = utils::loadNpy(
        *mManager, isDora ? TEST_SOURCE_DORA_TP2.string() : TEST_SOURCE_LORA_TP2.string(), MemoryType::kCPU);
    loraReqWeights->unsqueeze(0);
    TensorPtr loraReqKeys = utils::loadNpy(
        *mManager, isDora ? TEST_KEYS_DORA_TP2.string() : TEST_KEYS_LORA_TP2.string(), MemoryType::kCPU);
    loraReqKeys->unsqueeze(0);

    TensorPtr targetPageBlock = utils::loadNpy(*mManager,
        isDora ? TEST_KEYS_DORA_TP2_PAGES_RANK1.string() : TEST_KEYS_LORA_TP2_PAGES_RANK1.string(), MemoryType::kCPU);
    TensorPtr pageBlock = mManager->cpu(targetPageBlock->getShape(), targetPageBlock->getDataType());
    mManager->setZero(*pageBlock);
    std::vector<TensorPtr> pages;
    for (SizeType32 p = 0; p < pageBlock->getShape().d[0]; ++p)
    {
        pages.push_back(ITensor::view(ITensor::slice(pageBlock, p, 1),
            ITensor::makeShape({pageBlock->getShape().d[1], pageBlock->getShape().d[2]})));
    }
    std::vector<std::size_t> pageIds{};
    pageIds.resize(pages.size());
    std::iota(pageIds.begin(), pageIds.end(), 0);

    auto locations = LoraCache::copyToPages(
        loraReqWeights, loraReqKeys, modelConfig, worldConfig, moduleIdToModule, *mManager, pages, pageIds);

    auto pagePtr = bufferCast<float>(*pageBlock);
    auto targetPtr = bufferCast<float>(*targetPageBlock);

    for (SizeType32 i = 0; i < pageBlock->getSize(); ++i)
    {
        EXPECT_FLOAT_EQ(pagePtr[i], targetPtr[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(LoraCacheTest, LoraCacheTest, testing::Values(false, true));

} // namespace tensorrt_llm::runtime
