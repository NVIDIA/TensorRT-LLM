
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

#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include "tensorrt_llm/batch_manager/trtEncoderModel.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tensorrt_llm/runtime/utils/runtimeUtils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

using namespace tensorrt_llm::runtime;
namespace fs = std::filesystem;

using TensorPtr = ITensor::SharedPtr;

namespace
{
auto const TEST_RESOURCE_PATH = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources";
auto const ENC_DEC_BASE = TEST_RESOURCE_PATH / "models/enc_dec/trt_engines";
auto const ENC_DEC_ENGINE_BASE = TEST_RESOURCE_PATH / "models/enc_dec/trt_engines";
auto const BART_TP1_PP1_ENCODER_RMPAD_DIR = "bart-large-cnn/1-gpu/float16/tp1/encoder";
auto const BART_TP2_PP1_ENCODER_RMPAD_DIR = "bart-large-cnn/2-gpu/float16/tp2/encoder";
auto const BART_TP2_PP2_ENCODER_RMPAD_DIR = "bart-large-cnn/4-gpu/float16/tp2/encoder";
auto const T5_TP1_PP1_ENCODER_RMPAD_DIR = "t5-small/1-gpu/float16/tp1/encoder";
auto const ENC_DEC_DATA_BASE = TEST_RESOURCE_PATH / "data/enc_dec";
} // namespace

namespace tensorrt_llm::batch_manager
{

class EncoderModelTestSingleGPU : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    EncoderModelTestSingleGPU(std::filesystem::path const& modelPath)
        : mModelConfig(1, 2, 1, 1, 1, 1, nvinfer1::DataType::kFLOAT)
        , mModelPath(modelPath)
    {
    }

    EncoderModelTestSingleGPU()
        : EncoderModelTestSingleGPU(ENC_DEC_ENGINE_BASE / T5_TP1_PP1_ENCODER_RMPAD_DIR)
    {
    }

    void SetUp() override
    {
        std::filesystem::path trtEnginePath = mModelPath;

        mBeamWidth = 1;

        mLogger = std::make_shared<TllmLogger>();

        initTrtLlmPlugins(mLogger.get());

        auto const json = GptJsonConfig::parse(trtEnginePath / "config.json");
        mModelConfig = json.getModelConfig();
        mWorldConfig = WorldConfig::mpi(json.getGpusPerNode(), json.getTensorParallelism(),
            json.getPipelineParallelism(), json.getContextParallelism());
        mVocabSizePadded = mModelConfig.getVocabSizePadded(mWorldConfig.getSize());

        auto const enginePath = trtEnginePath / json.engineFilename(mWorldConfig);
        auto const dtype = mModelConfig.getDataType();

        ASSERT_TRUE(fs::exists(enginePath));
        mEngineBuffer = utils::loadEngine(enginePath.string());

        mStream = std::make_unique<CudaStream>();
        mManager = std::make_unique<BufferManager>(mStream);
    }

    void TearDown() override {}

    int32_t mMaxNumRequests;
    int32_t mMaxSeqLen;
    int32_t mBeamWidth;
    int32_t mVocabSizePadded;
    // SamplingConfig mSamplingConfig;
    std::string mDataPath;
    std::shared_ptr<nvinfer1::ILogger> mLogger;
    ModelConfig mModelConfig;
    WorldConfig mWorldConfig;
    std::vector<std::uint8_t> mEngineBuffer;
    std::unique_ptr<BufferManager> mManager;
    BufferManager::CudaStreamPtr mStream;
    std::filesystem::path mModelPath;
};

// test for TP2PP2
class TrtEncoderModelTestMultiGPU : public EncoderModelTestSingleGPU
{
protected:
    TrtEncoderModelTestMultiGPU()
        : EncoderModelTestSingleGPU(ENC_DEC_ENGINE_BASE / BART_TP2_PP2_ENCODER_RMPAD_DIR)
    {
    }
};

namespace
{

void runEncoderTest(std::unique_ptr<BufferManager>& bufferManager, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, std::vector<std::uint8_t> const& engineBuffer,
    std::shared_ptr<nvinfer1::ILogger>& logger)
{
    using VecTokens = LlmRequest::VecTokens;
    using TokenIdType = LlmRequest::TokenIdType;

    auto inputsIdsHost
        = utils::loadNpy(*bufferManager, (ENC_DEC_DATA_BASE / "input_ids.npy").string(), MemoryType::kCPU);
    auto inputsIdsPtr = bufferCast<SizeType32>(*inputsIdsHost);
    auto inputLengthsHost
        = utils::loadNpy(*bufferManager, (ENC_DEC_DATA_BASE / "input_lengths.npy").string(), MemoryType::kCPU);
    auto inputLengthsPtr = bufferCast<SizeType32>(*inputLengthsHost);
    auto encoderOutput
        = utils::loadNpy(*bufferManager, (ENC_DEC_DATA_BASE / "encoder_output.npy").string(), MemoryType::kCPU);
    auto encoderOutputPrt = bufferCast<half>(*encoderOutput);

    SizeType32 const nbRequests = inputLengthsHost->getShape().d[0];
    SizeType32 const stride = inputsIdsHost->getShape().d[1];
    SizeType32 const hiddenSize = encoderOutput->getShape().d[1];
    ASSERT_EQ(nbRequests, inputsIdsHost->getShape().d[0]);

    // std::vector<std::shared_ptr<VecTokens>> inputIds(nbRequests);
    RequestVector requestList;
    for (SizeType32 i = 0; i < nbRequests; i++)
    {
        SizeType32 length = inputLengthsPtr[i];
        auto currentInputId = std::make_shared<VecTokens>(0);
        currentInputId->insert(currentInputId->end(), inputsIdsPtr, inputsIdsPtr + length);
        executor::Request req(*currentInputId, 1);
        req.setEncoderInputTokenIds(*currentInputId);
        auto request = std::make_shared<LlmRequest>(i, req);
        inputsIdsPtr += stride;
        requestList.push_back(request);
    }

    tensorrt_llm::executor::ExecutorConfig executorConfig{};
    auto trtEncoderModel = std::make_shared<TrtEncoderModel>(
        modelConfig, worldConfig, runtime::RawEngine(engineBuffer.data(), engineBuffer.size()), logger, executorConfig);

    trtEncoderModel->forward(requestList);

    if (worldConfig.isLastPipelineParallelRank() && worldConfig.getTensorParallelRank() == 0)
    {
        auto arrayEqual = [](auto it0, auto it1, SizeType32 length)
        {
            SizeType32 nbNotEqual = 0;
            for (SizeType32 i = 0; i < length; i++)
            {
                auto v0 = static_cast<float>(*it0);
                auto v1 = static_cast<float>(*it1);
                if (std::abs(v0 - v1) > 1e-3)
                {
                    nbNotEqual++;
                }
                it0++;
                it1++;
            }
            return static_cast<double>(nbNotEqual) / length;
        };
        ASSERT_EQ(requestList.size(), inputLengthsHost->getShape().d[0]);
        {
            auto curLengthPtr = inputLengthsPtr;
            auto curOutPtr = encoderOutputPrt;
            for (auto const& req : requestList)
            {
                ASSERT_TRUE(req->getEncoderOutputHost()) << "Encoder output is empty!";
                EXPECT_EQ(req->getState(), LlmRequestState::kCONTEXT_INIT);
                auto actualOut = bufferCast<half>(*(req->getEncoderOutputHost()));
                auto unequalFraction = arrayEqual(curOutPtr, actualOut, *curLengthPtr);
                EXPECT_TRUE(unequalFraction == 0)
                    << "Req " << req->mRequestId << ": " << unequalFraction << " of outputs are different";
                curOutPtr += *curLengthPtr * hiddenSize;
                curLengthPtr++;
            }
        }
    }
}

} // Anonymous namespace

TEST_F(EncoderModelTestSingleGPU, Forward)
{
    runEncoderTest(mManager, mModelConfig, mWorldConfig, mEngineBuffer, mLogger);
}

TEST_F(TrtEncoderModelTestMultiGPU, Forward)
{

    runEncoderTest(mManager, mModelConfig, mWorldConfig, mEngineBuffer, mLogger);
}

} // namespace tensorrt_llm::batch_manager
