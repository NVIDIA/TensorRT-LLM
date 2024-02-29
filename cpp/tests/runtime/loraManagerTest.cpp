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

#include <algorithm>
#include <gtest/gtest.h>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/loraManager.h"
#include "tensorrt_llm/runtime/loraModule.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include "tensorrt_llm/runtime/utils/numpyUtils.h"

#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
auto const TEST_RESOURCE_PATH = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources/data";
auto const TEST_SOURCE_LORA_TP1 = TEST_RESOURCE_PATH / "lora-test-weights-tp1/source.npy";
auto const TEST_DEST_LORA_TP1 = TEST_RESOURCE_PATH / "lora-test-weights-tp1/target.npy";
auto const TEST_KEYS_LORA_TP1 = TEST_RESOURCE_PATH / "lora-test-weights-tp1/config.npy";
auto const TEST_SOURCE_LORA_TP2 = TEST_RESOURCE_PATH / "lora-test-weights-tp2/source.npy";
auto const TEST_DEST_LORA_TP2 = TEST_RESOURCE_PATH / "lora-test-weights-tp2/target.npy";
auto const TEST_KEYS_LORA_TP2 = TEST_RESOURCE_PATH / "lora-test-weights-tp2/config.npy";
auto const TEST_MODEL_CONFIG = TEST_RESOURCE_PATH / "test_model_lora_config.json";
} // namespace

namespace tensorrt_llm::runtime
{
using TensorPtr = ITensor::SharedPtr;

class LoraManagerTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    LoraManagerTest()
        : mModelConfig(1, 2, 1, 4, nvinfer1::DataType::kFLOAT)
    {
    }

    void SetUp() override
    {
        mStream = std::make_unique<CudaStream>();
        mManager = std::make_unique<BufferManager>(mStream);

        mWorldConfig = WorldConfig(2);

        mModelConfig.setLoraModules(LoraModule::createLoraModules({"attn_dense", "attn_qkv"}, 4, 4, 1, 1, 2, 2));
    }

    std::unique_ptr<BufferManager> mManager;
    BufferManager::CudaStreamPtr mStream;
    GptModelConfig mModelConfig;
    WorldConfig mWorldConfig;
};

TEST_F(LoraManagerTest, moduleParsing)
{
    auto jsonConfig = GptJsonConfig::parse(TEST_MODEL_CONFIG);
    auto loraModules = jsonConfig.getModelConfig().getLoraModules();

    std::vector<LoraModule> expectedModules{
        LoraModule(LoraModule::ModuleType::kATTN_QKV, 2048, 6144, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_Q, 2048, 2048, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_K, 2048, 2048, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_V, 2048, 2048, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_DENSE, 2048, 2048, false, true, 1, -1),
        LoraModule(LoraModule::ModuleType::kMLP_GATE, 2048, 4096, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kMLP_H_TO_4H, 2048, 4096, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kMLP_4H_TO_H, 4096, 2048, false, true, 1, -1),
    };
    ASSERT_EQ(expectedModules.size(), loraModules.size());
    for (size_t i = 0; i < expectedModules.size(); ++i)
    {
        EXPECT_EQ(expectedModules[i].value(), loraModules[i].value());
        EXPECT_EQ(expectedModules[i].name(), loraModules[i].name());
        EXPECT_EQ(expectedModules[i].inDim(), loraModules[i].inDim());
        EXPECT_EQ(expectedModules[i].outDim(), loraModules[i].outDim());
        EXPECT_EQ(expectedModules[i].inTpSplitDim(), loraModules[i].inTpSplitDim());
        EXPECT_EQ(expectedModules[i].outTpSplitDim(), loraModules[i].outTpSplitDim());
    }
}

TEST_F(LoraManagerTest, formatTensors_tp1)
{
    LoraManager loraManager;
    auto modelConfig = GptModelConfig(0, 2, 1, 16, nvinfer1::DataType::kFLOAT);
    modelConfig.setMlpHiddenSize(32);
    auto worldConfig = WorldConfig(1, 1, 0);
    std::vector<LoraModule> modules{
        LoraModule(LoraModule::ModuleType::kATTN_QKV, 16, 3 * 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_Q, 16, 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_K, 16, 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_V, 16, 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_DENSE, 16, 16, false, true, 1, -1),
        LoraModule(LoraModule::ModuleType::kMLP_H_TO_4H, 16, 32, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kMLP_4H_TO_H, 32, 16, false, true, 1, -1),
        LoraModule(LoraModule::ModuleType::kMLP_GATE, 16, 32, false, true, -1, 0),
    };
    modelConfig.setLoraModules(modules);
    loraManager.create(modelConfig, worldConfig, *mManager);

    TensorPtr loraReqWeights = utils::loadNpy(*mManager, TEST_SOURCE_LORA_TP1, MemoryType::kGPU);
    loraReqWeights->unsqueeze(0);
    TensorPtr loraReqKeys = utils::loadNpy(*mManager, TEST_KEYS_LORA_TP1, MemoryType::kCPU);
    loraReqKeys->unsqueeze(0);
    TensorPtr loraTargetTensors = utils::loadNpy(*mManager, TEST_DEST_LORA_TP1, MemoryType::kCPU);

    loraManager.formatTaskTensors(loraReqWeights, loraReqKeys, modelConfig, worldConfig, *mManager);
    TensorPtr hostWeights = mManager->copyFrom(*loraReqWeights, MemoryType::kCPU);
    mManager->getStream().synchronize();

    auto srcPtr = bufferCast<float>(*hostWeights);
    auto destPtr = bufferCast<float>(*loraTargetTensors);

    for (SizeType i = 0; i < loraReqWeights->getSize(); ++i)
    {
        EXPECT_FLOAT_EQ(srcPtr[i], destPtr[i]);
    }
}

TEST_F(LoraManagerTest, formatTensors_tp2)
{
    LoraManager loraManager;
    auto modelConfig = GptModelConfig(0, 2, 1, 16, nvinfer1::DataType::kFLOAT);
    modelConfig.setMlpHiddenSize(32);
    auto worldConfig = WorldConfig(2, 1, 0);
    std::vector<LoraModule> modules{
        LoraModule(LoraModule::ModuleType::kATTN_QKV, 16, 3 * 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_Q, 16, 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_K, 16, 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_V, 16, 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_DENSE, 16, 16, false, true, 1, -1),
        LoraModule(LoraModule::ModuleType::kMLP_H_TO_4H, 16, 32, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kMLP_4H_TO_H, 32, 16, false, true, 1, -1),
        LoraModule(LoraModule::ModuleType::kMLP_GATE, 16, 32, false, true, -1, 0),
    };
    modelConfig.setLoraModules(modules);
    loraManager.create(modelConfig, worldConfig, *mManager);

    TensorPtr loraReqWeights = utils::loadNpy(*mManager, TEST_SOURCE_LORA_TP2, MemoryType::kGPU);
    loraReqWeights->unsqueeze(0);
    TensorPtr loraReqKeys = utils::loadNpy(*mManager, TEST_KEYS_LORA_TP2, MemoryType::kCPU);
    loraReqKeys->unsqueeze(0);
    TensorPtr loraTargetTensors = utils::loadNpy(*mManager, TEST_DEST_LORA_TP2, MemoryType::kCPU);

    loraManager.formatTaskTensors(loraReqWeights, loraReqKeys, modelConfig, worldConfig, *mManager);
    TensorPtr hostWeights = mManager->copyFrom(*loraReqWeights, MemoryType::kCPU);
    mManager->getStream().synchronize();

    auto srcPtr = bufferCast<float>(*hostWeights);
    auto destPtr = bufferCast<float>(*loraTargetTensors);

    for (SizeType i = 0; i < loraReqWeights->getSize(); ++i)
    {
        EXPECT_FLOAT_EQ(srcPtr[i], destPtr[i]);
    }
}

TEST_F(LoraManagerTest, LoraManager_addTask)
{
    LoraManager manager;
    manager.create(mModelConfig, mWorldConfig, *mManager);

    std::vector<int32_t> taskNLayers{4, 6};
    std::vector<int32_t> taskMod{0, 1};
    std::vector<int32_t> taskSizes{16, 8};

    for (SizeType taskNum = 0; taskNum < static_cast<SizeType>(taskSizes.size()); ++taskNum)
    {

        auto mod = taskMod[taskNum];
        auto nLayers = taskNLayers[taskNum];
        auto taskSize = taskSizes[taskNum];
        auto taskName = taskNum;
        // bs=1
        // nbModules=1
        // nbLayers=4
        // adapterSize=16
        // Hi=128
        // Ho=3*128
        auto weightsShape = ITensor::makeShape({1, 1 * nLayers, taskSize * 128 + taskSize * 3 * 128});
        auto weights = mManager->cpu(weightsShape, nvinfer1::DataType::kFLOAT);
        auto weightsPtr = bufferCast<float>(*weights);
        std::fill_n(weightsPtr, weights->getSize(), 1.f * taskNum);

        auto keysShape = ITensor::makeShape({1, 1 * nLayers, 3});

        auto keys = mManager->cpu(keysShape, nvinfer1::DataType::kINT32);
        auto keysPtr = bufferCast<int32_t>(*keys);
        SizeType off = 0;
        for (SizeType i = 0; i < nLayers; ++i)
        {
            keysPtr[off++] = mod;
            keysPtr[off++] = i;
            keysPtr[off++] = taskSize;
        }

        weights->squeeze(0);
        keys->squeeze(0);

        manager.addTask(taskName, std::move(weights), std::move(keys));
    }

    for (SizeType taskNum = 0; taskNum < static_cast<SizeType>(taskSizes.size()); ++taskNum)
    {
        auto mod = taskMod[taskNum];
        auto nLayers = taskNLayers[taskNum];
        auto taskSize = taskSizes[taskNum];
        auto taskName = taskNum;
        auto modName = taskNum == 0 ? "attn_qkv" : "attn_q";

        auto [taskWeights, taskKeys] = manager.getTask(taskName);
        auto taskKeysPtr = bufferCast<int32_t>(*taskKeys);

        auto numWeights = static_cast<SizeType>(taskWeights->getSize());
        auto hostWeightsPtr = bufferCast<float>(*taskWeights);

        for (SizeType i = 0; i < numWeights; ++i)
        {
            EXPECT_FLOAT_EQ(1.f * taskNum, hostWeightsPtr[i]);
        }

        SizeType off = 0;
        for (SizeType i = 0; i < taskNLayers[taskNum]; ++i)
        {
            EXPECT_EQ(taskKeysPtr[off++], taskMod[taskNum]);
            EXPECT_EQ(taskKeysPtr[off++], i);
            EXPECT_EQ(taskKeysPtr[off++], taskSizes[taskNum]);
        }
    }
}

static void checkLoraTensors(LoraManager const& loraManager, std::vector<int64_t> const& targetPtrs,
    TensorPtr weightsPtrs, std::vector<int32_t> const& targetAdapterSizes, TensorPtr adapterSizes,
    GptModelConfig const& modelConfig, WorldConfig const& worldConfig, std::vector<LoraModule> const& modules,
    SizeType numModules, SizeType numLayers, SizeType numSeqs)
{
    auto adapterSizesPtr = bufferCast<SizeType>(*adapterSizes);
    auto weightsPtrsPtr = bufferCast<int64_t>(*weightsPtrs);
    ASSERT_EQ(targetPtrs.size(), weightsPtrs->getSize());
    ASSERT_EQ(targetAdapterSizes.size(), adapterSizes->getSize());
    auto firstLayerId
        = modelConfig.getNbLayers(worldConfig.getPipelineParallelism()) * worldConfig.getPipelineParallelRank();
    LoraManager::TensorMap expectedTensors;

    for (SizeType m = 0; m < numModules; ++m)
    {
        TensorPtr modSlice = ITensor::slice(weightsPtrs, m, 1);
        TensorPtr modAdapterSlice = ITensor::slice(adapterSizes, m, 1);
        modSlice->squeeze(0);
        modAdapterSlice->squeeze(0);
        for (SizeType l = 0; l < numLayers; ++l)
        {
            TensorPtr layerSlice = ITensor::slice(modSlice, l, 1);
            TensorPtr layerAdapterSlice = ITensor::slice(modAdapterSlice, l, 1);
            layerSlice->squeeze(0);
            layerAdapterSlice->squeeze(0);
            auto weightsFieldName
                = std::string(modules.at(m).name()) + "_lora_weights_pointers_" + std::to_string(l + firstLayerId);
            expectedTensors.insert_or_assign(weightsFieldName, layerSlice);
            auto adapterSizeFieldName
                = std::string(modules.at(m).name()) + "_lora_ranks_" + std::to_string(l + firstLayerId);
            expectedTensors.insert_or_assign(adapterSizeFieldName, layerAdapterSlice);
            for (SizeType i = 0; i < numSeqs; ++i)
            {
                SizeType adapterSizeOff = common::flat_index3(m, l, i, numLayers, numSeqs);
                EXPECT_EQ(targetAdapterSizes[adapterSizeOff], adapterSizesPtr[adapterSizeOff]);
                SizeType inPtrIdx = common::flat_index4(m, l, i, 0, numLayers, numSeqs, 2);
                SizeType outPtrIdx = common::flat_index4(m, l, i, 1, numLayers, numSeqs, 2);
                EXPECT_EQ(targetPtrs[inPtrIdx], weightsPtrsPtr[inPtrIdx]);
                EXPECT_EQ(targetPtrs[outPtrIdx], weightsPtrsPtr[outPtrIdx]);
            }
        }
    }

    LoraManager::TensorMap inputTensors;
    loraManager.insertInputTensors(inputTensors, weightsPtrs, adapterSizes, modelConfig, worldConfig);

    ASSERT_EQ(expectedTensors.size(), inputTensors.size());
    for (auto& [fieldName, tensor] : expectedTensors)
    {
        ASSERT_NE(inputTensors.find(fieldName), inputTensors.end());
        auto expectedTensor = expectedTensors.find(fieldName)->second;
        auto actualTensor = inputTensors.find(fieldName)->second;
        ITensor::shapeEquals(expectedTensor->getShape(), actualTensor->getShape());
        if (expectedTensor->getDataType() == nvinfer1::DataType::kINT64)
        {
            auto expT = bufferCast<int64_t>(*expectedTensor);
            auto actT = bufferCast<int64_t>(*actualTensor);
            for (size_t i = 0; i < expectedTensor->getSize(); ++i)
            {
                EXPECT_EQ(expT[i], actT[i]);
            }
        }
        else
        {
            auto expT = bufferCast<int32_t>(*expectedTensor);
            auto actT = bufferCast<int32_t>(*actualTensor);
            for (size_t i = 0; i < expectedTensor->getSize(); ++i)
            {
                EXPECT_EQ(expT[i], actT[i]);
            }
        }
    }
}

TEST_F(LoraManagerTest, fillInputTensors_tp1_pp1)
{
    LoraManager loraManager;
    auto modelConfig = GptModelConfig(0, 2, 1, 16, nvinfer1::DataType::kFLOAT);
    modelConfig.setMlpHiddenSize(32);
    auto worldConfig = WorldConfig(1, 1, 0);
    std::vector<LoraModule> modules{
        LoraModule(LoraModule::ModuleType::kATTN_QKV, 16, 3 * 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_Q, 16, 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_K, 16, 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_V, 16, 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_DENSE, 16, 16, false, true, 1, -1),
        LoraModule(LoraModule::ModuleType::kMLP_H_TO_4H, 16, 32, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kMLP_GATE, 16, 32, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kMLP_4H_TO_H, 32, 16, false, true, 1, -1),
    };
    modelConfig.setLoraModules(modules);
    loraManager.create(modelConfig, worldConfig, *mManager);
    auto numModules = static_cast<SizeType>(modelConfig.getLoraModules().size());
    auto numLayers = static_cast<SizeType>(modelConfig.getNbLayers());
    SizeType numSeqs = 4;
    TensorPtr weightsPtrs
        = mManager->cpu(ITensor::makeShape({numModules, numLayers, numSeqs, 2}), nvinfer1::DataType::kINT64);
    TensorPtr adapterSizes
        = mManager->cpu(ITensor::makeShape({numModules, numLayers, numSeqs}), nvinfer1::DataType::kINT32);

    mManager->setZero(*weightsPtrs);
    mManager->setZero(*adapterSizes);

    SizeType numContextRequests = 1;
    std::vector<uint64_t> reqIds{1, 2, 3};
    std::vector<SizeType> reqBeamWidth{1, 2, 1};
    std::vector<bool> loraEnabled{true, true, false};

    TensorPtr loraReqKeys = utils::loadNpy(*mManager, TEST_KEYS_LORA_TP1, MemoryType::kCPU);
    TensorPtr loraWeights = utils::loadNpy(*mManager, TEST_DEST_LORA_TP1, MemoryType::kGPU);

    loraManager.addTask(1, loraWeights, loraReqKeys);
    loraManager.addTask(2, loraWeights, loraReqKeys);

    loraManager.fillInputTensors(
        weightsPtrs, adapterSizes, reqIds, reqBeamWidth, loraEnabled, numContextRequests, modelConfig, worldConfig);

    // set in order litest in modelConfig
    SizeType attnQkvOff = 1;
    SizeType attnDense = 0;

    auto inputWeightsPtrs = bufferCast<float>(*loraWeights);

    auto adapterSizesPtr = bufferCast<SizeType>(*adapterSizes);
    auto weightsPtrsPtr = bufferCast<int64_t>(*weightsPtrs);

    auto weightsRowSize = loraWeights->getShape().d[1];

    std::vector<int32_t> targetAdapterSizes{
        8, 8, 8, 0, // attn_qkv layer 0
        8, 8, 8, 0, // attn_qkv layer 1
        4, 4, 4, 0, // attn_q layer 0
        4, 4, 4, 0, // attn_q layer 1
        4, 4, 4, 0, // attn_k layer 0
        4, 4, 4, 0, // attn_k layer 1
        4, 4, 4, 0, // attn_v layer 0
        4, 4, 4, 0, // attn_v layer 1
        8, 8, 8, 0, // attn_dense layer 0
        8, 8, 8, 0, // attn_dense layer 1
        8, 8, 8, 0, // mlp_h_to_4h layer 0
        8, 8, 8, 0, // mlp_h_to_4h layer 1
        8, 8, 8, 0, // mlp_gate layer 0
        8, 8, 8, 0, // mlp_gate layer 1
        8, 8, 8, 0, // mlp_4h_to_h layer 0
        8, 8, 8, 0, // mlp_4h_to_h layer 1
    };

    std::vector<int64_t> targetPtrs{
        // attn_qkv layer 0
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(0, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(0, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(0, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(0, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(0, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(0, 8 * 16, weightsRowSize)),
        0,
        0,

        // attn_qkv layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(1, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(1, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(1, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(1, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(1, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(1, 8 * 16, weightsRowSize)),
        0,
        0,

        // attn_q layer 0
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(2, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(2, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(2, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(2, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(2, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(2, 4 * 16, weightsRowSize)),
        0,
        0,

        // attn_q layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(3, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(3, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(3, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(3, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(3, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(3, 4 * 16, weightsRowSize)),
        0,
        0,

        // attn_k layer 0
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(4, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(4, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(4, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(4, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(4, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(4, 4 * 16, weightsRowSize)),
        0,
        0,

        // attn_k layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(5, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(5, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(5, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(5, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(5, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(5, 4 * 16, weightsRowSize)),
        0,
        0,

        // attn_v layer 0
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(6, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(6, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(6, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(6, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(6, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(6, 4 * 16, weightsRowSize)),
        0,
        0,

        // attn_v layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(7, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(7, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(7, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(7, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(7, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(7, 4 * 16, weightsRowSize)),
        0,
        0,

        // attn_dense layer 0
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(8, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(8, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(8, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(8, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(8, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(8, 8 * 16, weightsRowSize)),
        0,
        0,

        // attn_dense layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(9, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(9, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(9, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(9, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(9, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(9, 8 * 16, weightsRowSize)),
        0,
        0,

        // mlp_h_to_4h layer 0
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(10, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(10, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(10, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(10, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(10, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(10, 8 * 16, weightsRowSize)),
        0,
        0,

        // mlp_h_to_4h layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(11, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(11, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(11, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(11, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(11, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(11, 8 * 16, weightsRowSize)),
        0,
        0,

        // mlp_gate layer 0
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(14, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(14, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(14, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(14, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(14, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(14, 8 * 16, weightsRowSize)),
        0,
        0,

        // mlp_gate layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(15, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(15, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(15, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(15, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(15, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(15, 8 * 16, weightsRowSize)),
        0,
        0,

        // mlp_4h_to_h layer 0
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(12, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(12, 8 * 32, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(12, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(12, 8 * 32, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(12, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(12, 8 * 32, weightsRowSize)),
        0,
        0,

        // mlp_4h_to_h layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(13, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(13, 8 * 32, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(13, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(13, 8 * 32, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(13, 0, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(13, 8 * 32, weightsRowSize)),
        0,
        0,
    };

    checkLoraTensors(loraManager, targetPtrs, weightsPtrs, targetAdapterSizes, adapterSizes, modelConfig, worldConfig,
        modules, numModules, numLayers, numSeqs);
}

TEST_F(LoraManagerTest, fillInputTensors_tp2_pp2)
{
    LoraManager loraManager;
    auto modelConfig = GptModelConfig(0, 2, 1, 16, nvinfer1::DataType::kFLOAT);
    modelConfig.setMlpHiddenSize(32);
    auto worldConfig = WorldConfig(2, 2, 3); // tpRank = 1, ppRank = 1
    std::vector<LoraModule> modules{
        LoraModule(LoraModule::ModuleType::kATTN_QKV, 16, 3 * 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_Q, 16, 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_K, 16, 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_V, 16, 16, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kATTN_DENSE, 16, 16, false, true, 1, -1),
        LoraModule(LoraModule::ModuleType::kMLP_H_TO_4H, 16, 32, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kMLP_GATE, 16, 32, false, true, -1, 0),
        LoraModule(LoraModule::ModuleType::kMLP_4H_TO_H, 32, 16, false, true, 1, -1),
    };
    modelConfig.setLoraModules(modules);
    loraManager.create(modelConfig, worldConfig, *mManager);
    auto numModules = static_cast<SizeType>(modelConfig.getLoraModules().size());
    auto numLayers = static_cast<SizeType>(modelConfig.getNbLayers(2));
    SizeType numSeqs = 4;
    TensorPtr weightsPtrs
        = mManager->cpu(ITensor::makeShape({numModules, numLayers, numSeqs, 2}), nvinfer1::DataType::kINT64);
    TensorPtr adapterSizes
        = mManager->cpu(ITensor::makeShape({numModules, numLayers, numSeqs}), nvinfer1::DataType::kINT32);

    mManager->setZero(*weightsPtrs);
    mManager->setZero(*adapterSizes);

    SizeType numContextRequests = 1;
    std::vector<uint64_t> reqIds{1, 2, 3};
    std::vector<SizeType> reqBeamWidth{1, 2, 1};
    std::vector<bool> loraEnabled{true, true, false};

    TensorPtr loraReqKeys = utils::loadNpy(*mManager, TEST_KEYS_LORA_TP2, MemoryType::kCPU);
    TensorPtr loraWeights = utils::loadNpy(*mManager, TEST_DEST_LORA_TP2, MemoryType::kGPU);

    loraManager.addTask(1, loraWeights, loraReqKeys);
    loraManager.addTask(2, loraWeights, loraReqKeys);

    loraManager.fillInputTensors(
        weightsPtrs, adapterSizes, reqIds, reqBeamWidth, loraEnabled, numContextRequests, modelConfig, worldConfig);

    // set in order litest in modelConfig
    SizeType attnQkvOff = 1;
    SizeType attnDense = 0;

    auto inputWeightsPtrs = bufferCast<float>(*loraWeights);

    auto adapterSizesPtr = bufferCast<SizeType>(*adapterSizes);
    auto weightsPtrsPtr = bufferCast<int64_t>(*weightsPtrs);

    auto weightsRowSize = loraWeights->getShape().d[1];

    std::vector<int32_t> targetAdapterSizes{
        8, 8, 8, 0, // attn_qkv layer 1
        4, 4, 4, 0, // attn_q layer 1
        4, 4, 4, 0, // attn_k layer 1
        4, 4, 4, 0, // attn_v layer 1
        8, 8, 8, 0, // attn_dense layer 1
        8, 8, 8, 0, // mlp_h_to_4h layer 1
        8, 8, 8, 0, // mlp_gate layer 1
        8, 8, 8, 0, // mlp_4h_to_h layer 1
    };

    SizeType attnQkvInRank1Off = 0;
    SizeType attnQkvOutRank1Off = (8 * 16) + (4 * (3 * 16));

    SizeType attnQInRank1Off = 0;
    SizeType attnQOutRank1Off = (4 * 16) + (2 * 16);

    SizeType mlphto4hInRank1Off = 0;
    SizeType mlphto4hOutRank1Off = (8 * 16) + (4 * 32);

    SizeType mlp4htohInRank1Off = (4 * 32);
    SizeType mlp4htohOutRank1Off = (8 * 32);

    std::vector<int64_t> targetPtrs{
        // attn_qkv layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(1, attnQkvInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(1, attnQkvOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(1, attnQkvInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(1, attnQkvOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(1, attnQkvInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(1, attnQkvOutRank1Off, weightsRowSize)),
        0,
        0,

        // attn_q layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(3, attnQInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(3, attnQOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(3, attnQInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(3, attnQOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(3, attnQInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(3, attnQOutRank1Off, weightsRowSize)),
        0,
        0,

        // attn_k layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(5, attnQInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(5, attnQOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(5, attnQInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(5, attnQOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(5, attnQInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(5, attnQOutRank1Off, weightsRowSize)),
        0,
        0,

        // attn_v layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(7, attnQInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(7, attnQOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(7, attnQInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(7, attnQOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(7, attnQInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(7, attnQOutRank1Off, weightsRowSize)),
        0,
        0,

        // attn_dense layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(9, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(9, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(9, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(9, 8 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(9, 4 * 16, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(9, 8 * 16, weightsRowSize)),
        0,
        0,

        // mlp_h_to_4h layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(11, mlphto4hInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(11, mlphto4hOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(11, mlphto4hInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(11, mlphto4hOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(11, mlphto4hInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(11, mlphto4hOutRank1Off, weightsRowSize)),
        0,
        0,

        // mlp_gate layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(15, mlphto4hInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(15, mlphto4hOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(15, mlphto4hInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(15, mlphto4hOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(15, mlphto4hInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(15, mlphto4hOutRank1Off, weightsRowSize)),
        0,
        0,

        // mlp_4h_to_h layer 1
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(13, mlp4htohInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(13, mlp4htohOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(13, mlp4htohInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(13, mlp4htohOutRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(13, mlp4htohInRank1Off, weightsRowSize)),
        reinterpret_cast<int64_t>(inputWeightsPtrs + common::flat_index2(13, mlp4htohOutRank1Off, weightsRowSize)),
        0,
        0,
    };

    checkLoraTensors(loraManager, targetPtrs, weightsPtrs, targetAdapterSizes, adapterSizes, modelConfig, worldConfig,
        modules, numModules, numLayers, numSeqs);
}
} // namespace tensorrt_llm::runtime
