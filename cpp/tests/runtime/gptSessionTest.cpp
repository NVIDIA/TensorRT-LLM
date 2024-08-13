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

#include <gtest/gtest.h>

#include "modelSpec.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"

#include <algorithm>
#include <filesystem>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace fs = std::filesystem;
using tensorrt_llm::testing::ModelSpec;
using tensorrt_llm::testing::KVCacheType;
using tensorrt_llm::testing::QuantMethod;
using tensorrt_llm::testing::OutputContentType;

namespace
{
auto const TEST_RESOURCE_PATH = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources";
auto const ENGINE_PATH = TEST_RESOURCE_PATH / "models/rt_engine";
auto const DATA_PATH = TEST_RESOURCE_PATH / "data";

auto const GPT_MODEL_DIR = "gpt2";
auto const GPTJ_MODEL_DIR = "gpt-j-6b";
auto const LLAMA_MODEL_DIR = "llama-7b-hf";
auto const CHATGLM_MODEL_DIR = "chatglm-6b";
auto const CHATGLM2_MODEL_DIR = "chatglm2-6b";
auto const CHATGLM3_MODEL_DIR = "chatglm3-6b";
auto const MAMBA_MODEL_DIR = "mamba-2.8b-hf";
auto const INPUT_FILE = "input_tokens.npy";
auto const CHATGLM_INPUT_FILE = "input_tokens_chatglm-6b.npy";
auto const CHATGLM2_INPUT_FILE = "input_tokens_chatglm2-6b.npy";
auto const CHATGLM3_INPUT_FILE = "input_tokens_chatglm3-6b.npy";

// Engines need to be generated using cpp/tests/resources/scripts/build_*_engines.py.
auto const FP32_GPT_DIR = "fp32-default";
auto const FP32_GPT_ATTENTION_DIR = "fp32-plugin";
auto const FP16_GPT_DIR = "fp16-default";
auto const FP16_GPT_ATTENTION_DIR = "fp16-plugin";
auto const FP16_GPT_ATTENTION_PACKED_DIR = FP16_GPT_ATTENTION_DIR + std::string("-packed");
auto const FP16_GPT_ATTENTION_PACKED_PAGED_DIR = FP16_GPT_ATTENTION_PACKED_DIR + std::string("-paged");

// Expected outputs need to be generated using cpp/tests/resources/scripts/generate_expected_*_output.py.
auto const FP32_RESULT_FILE = "output_tokens_fp32_tp1_pp1.npy";
auto const FP32_PLUGIN_RESULT_FILE = "output_tokens_fp32_plugin_tp1_pp1.npy";
auto const FP16_RESULT_FILE = "output_tokens_fp16_tp1_pp1.npy";
auto const FP16_PLUGIN_RESULT_FILE = "output_tokens_fp16_plugin_tp1_pp1.npy";
auto const FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP4_FILE = "output_tokens_fp16_plugin_packed_paged_tp1_pp4.npy";
auto const FP16_PLUGIN_PACKED_PAGED_RESULT_TP4_PP1_FILE = "output_tokens_fp16_plugin_packed_paged_tp4_pp1.npy";
auto const FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE = "output_tokens_fp16_plugin_packed_paged_tp2_pp2.npy";
auto const FP16_PLUGIN_PACKED_RESULT_FILE = "output_tokens_fp16_plugin_packed_tp1_pp1.npy";
auto const FP16_PLUGIN_PACKED_PAGED_RESULT_FILE = "output_tokens_fp16_plugin_packed_paged_tp1_pp1.npy";

struct ModelIds
{
    int endId;
    int padId;
};

struct ModelParams
{
    char const* baseDir;
    ModelIds ids;
};

struct MicroBatchSizes
{
    std::optional<SizeType32> ctxMicroBatchSize{std::nullopt};
    std::optional<SizeType32> genMicroBatchSize{std::nullopt};
};
} // namespace

class SessionTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();

        if (mDeviceCount == 0)
            GTEST_SKIP() << "No GPUs found";

        mLogger = std::make_shared<TllmLogger>();

        initTrtLlmPlugins(mLogger.get());
    }

    void TearDown() override {}

    int mDeviceCount;
    std::shared_ptr<nvinfer1::ILogger> mLogger{};
};

namespace
{
void verifyModelConfig(ModelConfig const& modelConfig, ModelSpec const& modelSpec)
{
    ASSERT_EQ(modelSpec.mUseGptAttentionPlugin, modelConfig.useGptAttentionPlugin());
    ASSERT_EQ(modelSpec.mUsePackedInput, modelConfig.usePackedInput());
    ASSERT_EQ(modelSpec.mKVCacheType == KVCacheType::kPAGED, modelConfig.isPagedKVCache());
    ASSERT_EQ(modelSpec.mDataType, modelConfig.getDataType());
}

void testGptSession(fs::path const& modelPath, ModelSpec const& modelSpec, ModelIds const modelIds,
    SizeType32 beamWidth, std::initializer_list<int> const& batchSizes, fs::path const& resultsFile,
    std::shared_ptr<nvinfer1::ILogger> const& logger, bool cudaGraphMode, MicroBatchSizes microBatchSizes,
    bool const isChatGlmTest = false)
{
    auto manager = BufferManager(std::make_shared<CudaStream>());

    ASSERT_TRUE(fs::exists(DATA_PATH));
    std::string modelName{isChatGlmTest ? resultsFile.parent_path().parent_path().filename().string() : ""};

    fs::path inputPath = DATA_PATH / modelSpec.mInputFile;

    auto const& givenInput = utils::loadNpy(manager, inputPath.string(), MemoryType::kCPU);
    auto const& inputShape = givenInput->getShape();
    ASSERT_EQ(inputShape.nbDims, 2);
    auto const nbGivenInputs = static_cast<SizeType32>(inputShape.d[0]);
    ASSERT_GT(nbGivenInputs, 0);

    std::string outputPath = resultsFile.string();
    auto expectedOutput = utils::loadNpy(manager, outputPath, MemoryType::kCPU);
    auto const& outputShape = expectedOutput->getShape();
    ASSERT_EQ(outputShape.nbDims, 2);
    ASSERT_EQ(inputShape.d[0] * beamWidth, outputShape.d[0]);

    auto const givenInputData = bufferCast<TokenIdType const>(*givenInput);
    auto expectedOutputData = bufferCast<TokenIdType>(*expectedOutput);

    ASSERT_TRUE(fs::exists(modelPath));
    auto const json = GptJsonConfig::parse(modelPath / "config.json");
    auto const modelConfig = json.getModelConfig();
    verifyModelConfig(modelConfig, modelSpec);

    int const worldSize = modelSpec.mTPSize * modelSpec.mPPSize;
    auto const worldConfig = WorldConfig::mpi(worldSize, modelSpec.mTPSize, modelSpec.mPPSize);
    auto enginePath = modelPath / json.engineFilename(worldConfig);
    ASSERT_TRUE(fs::exists(enginePath));

    auto const maxInputLength = static_cast<SizeType32>(inputShape.d[1]);
    SizeType32 const maxSeqLength = static_cast<SizeType32>(outputShape.d[1]);
    ASSERT_LT(maxInputLength, maxSeqLength);
    SizeType32 const maxNewTokens = maxSeqLength - maxInputLength;

    SamplingConfig samplingConfig{beamWidth};
    samplingConfig.temperature = std::vector{1.0f};
    SizeType32 const minLength = 1;
    samplingConfig.minLength = std::vector{minLength};
    samplingConfig.randomSeed = std::vector{static_cast<uint64_t>(42ull)};
    samplingConfig.topK = std::vector{0};
    samplingConfig.topP = std::vector{0.0f};
    samplingConfig.lengthPenalty = std::vector{1.0f};
    samplingConfig.earlyStopping = std::vector{1};
    samplingConfig.noRepeatNgramSize = std::vector{1 << 30};

    auto const padId = modelIds.padId;
    auto endId = modelIds.endId;

    std::vector<SizeType32> givenInputLengths(nbGivenInputs);
    for (SizeType32 i = 0; i < nbGivenInputs; ++i)
    {
        auto const seqBegin = givenInputData + i * maxInputLength;
        auto const it = std::find(seqBegin, seqBegin + maxInputLength, padId);
        givenInputLengths[i] = std::distance(seqBegin, it);
    }

    std::srand(42);
    if (modelSpec.mRandomEndId)
    {
        auto const endIdRow = std::rand() % nbGivenInputs;
        auto const endIdBeam = std::rand() % beamWidth;
        auto const endIdCol = givenInputLengths[endIdRow] + minLength + std::rand() % (maxNewTokens - minLength);
        auto const endIdIndex = tc::flat_index2((endIdRow * beamWidth + endIdBeam), endIdCol, maxSeqLength);
        endId = expectedOutputData[endIdIndex];
    }

    std::vector<SizeType32> expectedLengths(nbGivenInputs * beamWidth, 0);
    for (SizeType32 bi = 0; bi < nbGivenInputs; ++bi)
    {
        for (SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            auto const seqBegin = expectedOutputData + bi * maxSeqLength * beamWidth + beam * maxSeqLength;
            auto const padIt = std::find(seqBegin, seqBegin + maxSeqLength, padId);
            auto const endIt = std::find(seqBegin, seqBegin + maxSeqLength, endId);
            SizeType32 const outputLength
                = std::min(std::distance(seqBegin, padIt), std::distance(seqBegin, endIt)) - givenInputLengths[bi];
            SizeType32 expectedLen = givenInputLengths[bi] + std::min(outputLength, maxNewTokens);
            if (modelSpec.mRandomEndId)
            {
                for (SizeType32 si = givenInputLengths[bi]; si < maxSeqLength; ++si)
                {
                    auto const expectIndex = tc::flat_index2((bi * beamWidth + beam), si, maxSeqLength);
                    if (expectedOutputData[expectIndex] == endId)
                    {
                        expectedLen = si;
                        break;
                    }
                }
                // Fill new EOS token to the expected data
                for (SizeType32 si = expectedLen; si < maxSeqLength; ++si)
                {
                    auto const expectIndex = tc::flat_index2((bi * beamWidth + beam), si, maxSeqLength);
                    expectedOutputData[expectIndex] = endId;
                }
            }
            expectedLengths[bi * beamWidth + beam] = expectedLen;
        }
    }

    auto const maxBatchSize = *std::max_element(batchSizes.begin(), batchSizes.end());
    GptSession::Config sessionConfig{maxBatchSize, beamWidth, maxSeqLength};
    sessionConfig.decoderPerRequest = modelSpec.mDecoderPerRequest;
    sessionConfig.ctxMicroBatchSize = microBatchSizes.ctxMicroBatchSize;
    sessionConfig.genMicroBatchSize = microBatchSizes.genMicroBatchSize;
    sessionConfig.cudaGraphMode = cudaGraphMode;
    sessionConfig.kvCacheConfig.useUvm = false;

    GptSession session{sessionConfig, modelConfig, worldConfig, enginePath.string(), logger};
    EXPECT_EQ(session.getDevice(), worldConfig.getDevice());
    // Use bufferManager for copying data to and from the GPU
    auto& bufferManager = session.getBufferManager();

    for (auto const batchSize : batchSizes)
    {
        std::cout << "=== batchSize:" << batchSize << " ===\n";

        std::vector<SizeType32> inputLengthsHost(batchSize);
        for (SizeType32 i = 0; i < batchSize; ++i)
        {
            int const inputIdx = i % nbGivenInputs;
            inputLengthsHost[i] = givenInputLengths[inputIdx];
        }
        auto inputLengths = bufferManager.copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

        // copy inputs and wrap into shared_ptr
        GenerationInput::TensorPtr inputIds;
        if (modelConfig.usePackedInput())
        {
            std::vector<SizeType32> inputOffsetsHost(batchSize + 1);
            tc::stl_utils::inclusiveScan(
                inputLengthsHost.begin(), inputLengthsHost.end(), inputOffsetsHost.begin() + 1);
            auto const totalInputSize = inputOffsetsHost.back();

            std::vector<std::int32_t> inputsHost(totalInputSize);
            for (SizeType32 i = 0; i < batchSize; ++i)
            {
                auto const seqBegin = givenInputData + (i % nbGivenInputs) * maxInputLength;
                std::copy(seqBegin, seqBegin + inputLengthsHost[i], inputsHost.begin() + inputOffsetsHost[i]);
            }
            inputIds = bufferManager.copyFrom(inputsHost, ITensor::makeShape({1, totalInputSize}), MemoryType::kGPU);
        }
        else
        {
            std::vector<std::int32_t> inputsHost(batchSize * maxInputLength, padId);
            for (SizeType32 i = 0; i < batchSize; ++i)
            {
                auto const seqBegin = givenInputData + (i % nbGivenInputs) * maxInputLength;
                std::copy(seqBegin, seqBegin + inputLengthsHost[i], inputsHost.begin() + i * maxInputLength);
            }
            inputIds
                = bufferManager.copyFrom(inputsHost, ITensor::makeShape({batchSize, maxInputLength}), MemoryType::kGPU);
        }

        GenerationInput generationInput{
            endId, padId, std::move(inputIds), std::move(inputLengths), modelConfig.usePackedInput()};
        generationInput.maxNewTokens = maxNewTokens;

        // runtime will allocate memory for output if this tensor is empty
        GenerationOutput generationOutput{bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32),
            bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)};

        // repeat the same inputs multiple times for testing idempotency of `generate()`
        auto constexpr repetitions = 10;
        for (auto r = 0; r < repetitions; ++r)
        {
            SizeType32 numSteps = 0;
            generationOutput.onTokenGenerated
                = [&numSteps, &modelSpec, maxNewTokens](
                      [[maybe_unused]] GenerationOutput::TensorPtr const& outputIds, SizeType32 step, bool finished)
            {
                // check that we execute the callback in each step
                EXPECT_EQ(step, numSteps);
                ++numSteps;
                if (!modelSpec.mRandomEndId)
                {
                    // check that we only finish after producing `maxNewTokens` tokens
                    EXPECT_TRUE(!finished || numSteps == maxNewTokens);
                }
                // check that `finished` is set to true after producing `maxNewTokens` tokens
                EXPECT_TRUE(numSteps != maxNewTokens || finished);
            };

            session.generate(generationOutput, generationInput, samplingConfig);

            // compare outputs
            if (worldConfig.isFirstPipelineParallelRank())
            {
                if (!modelSpec.mRandomEndId)
                {
                    EXPECT_EQ(numSteps, maxNewTokens);
                }
                auto const& outputLengths = generationOutput.lengths;
                auto const& outputLengthsDims = outputLengths->getShape();
                EXPECT_EQ(outputLengthsDims.nbDims, 2);
                EXPECT_EQ(outputLengthsDims.d[0], batchSize) << "r: " << r;
                EXPECT_EQ(outputLengthsDims.d[1], beamWidth) << "r: " << r;
                auto outputLengthsHost = bufferManager.copyFrom(*outputLengths, MemoryType::kCPU);
                auto lengths = bufferCast<std::int32_t>(*outputLengthsHost);
                bufferManager.getStream().synchronize();
                bool anyMismatch = false;
                for (auto b = 0; b < batchSize; ++b)
                {
                    for (auto beam = 0; beam < beamWidth; ++beam)
                    {
                        auto const lengthsIndex = tc::flat_index2(b, beam, beamWidth);
                        auto const expectedLength = expectedLengths[b % nbGivenInputs * beamWidth + beam];
                        EXPECT_EQ(lengths[lengthsIndex], expectedLength) << " b: " << b << " beam: " << beam;
                        anyMismatch |= (lengths[lengthsIndex] != expectedLength);
                    }
                }
                ASSERT_FALSE(anyMismatch) << "wrong output lengths";
            }

            if (worldConfig.isFirstPipelineParallelRank())
            {
                auto const& outputIds = generationOutput.ids;
                auto const& outputDims = outputIds->getShape();
                EXPECT_EQ(outputDims.nbDims, 3);
                EXPECT_EQ(outputDims.d[0], batchSize) << "r: " << r;
                EXPECT_EQ(outputDims.d[1], beamWidth) << "r: " << r;
                EXPECT_EQ(outputDims.d[2], maxSeqLength) << "r: " << r;
                auto outputHost = bufferManager.copyFrom(*outputIds, MemoryType::kCPU);
                auto output = bufferCast<std::int32_t>(*outputHost);
                bufferManager.getStream().synchronize();
                for (auto b = 0; b < batchSize; ++b)
                {
                    for (auto beam = 0; beam < beamWidth; ++beam)
                    {
                        bool anyMismatch = false;
                        for (auto i = 0; i < maxSeqLength; ++i)
                        {
                            int const outputIndex = tc::flat_index3(b, beam, i, beamWidth, maxSeqLength);
                            int const expectIndex
                                = tc::flat_index2((b % nbGivenInputs * beamWidth + beam), i, maxSeqLength);
                            if (expectedOutputData[expectIndex] == endId)
                            {
                                break;
                            }
                            EXPECT_EQ(output[outputIndex], expectedOutputData[expectIndex])
                                << " b: " << b << " beam: " << beam << " i: " << i;
                            anyMismatch |= (output[outputIndex] != expectedOutputData[expectIndex]);
                        }
                        ASSERT_FALSE(anyMismatch) << "batchSize: " << batchSize << ", r: " << r << ", b: " << b;
                    }
                }

                // make sure to recreate the outputs in the next repetition
                outputIds->release();
            }
        }
    }
}

auto constexpr kBatchSizes = {1, 8};

using ParamType = std::tuple<ModelParams, ModelSpec, SizeType32, bool, MicroBatchSizes, bool>;

std::string generateTestName(testing::TestParamInfo<ParamType> const& info)
{
    auto const modelSpec = std::get<1>(info.param);
    std::string name{modelSpec.mDataType == nvinfer1::DataType::kFLOAT ? "Float" : "Half"};
    auto const beamWidth = std::get<2>(info.param);
    name.append(beamWidth == 1 ? "Sampling" : "BeamWidth" + std::to_string(beamWidth));
    if (modelSpec.mUseGptAttentionPlugin)
        name.append("AttentionPlugin");
    if (modelSpec.mUsePackedInput)
        name.append("Packed");
    if (modelSpec.mKVCacheType == KVCacheType::kPAGED)
        name.append("PagedKvCache");
    if (modelSpec.mDecoderPerRequest)
        name.append("DecoderBatch");
    if (std::get<3>(info.param))
        name.append("CudaGraph");
    auto const microBatcheSizes = std::get<4>(info.param);
    if (microBatcheSizes.ctxMicroBatchSize)
        name.append("CBS" + std::to_string(microBatcheSizes.ctxMicroBatchSize.value()));
    if (microBatcheSizes.genMicroBatchSize)
        name.append("GBS" + std::to_string(microBatcheSizes.genMicroBatchSize.value()));
    if (modelSpec.mPPSize > 1)
        name.append("PP" + std::to_string(modelSpec.mPPSize));
    if (modelSpec.mTPSize > 1)
        name.append("TP" + std::to_string(modelSpec.mTPSize));
    if (modelSpec.mRandomEndId)
        name.append("EndId");
    return name;
}
} // namespace

class ParamTest : public SessionTest, public ::testing::WithParamInterface<ParamType>
{
};

TEST_P(ParamTest, Test)
{
    auto const modelParams = std::get<0>(GetParam());
    auto const modelDir = modelParams.baseDir;
    auto const modelIds = modelParams.ids;
    auto const modelSpec = std::get<1>(GetParam());
    SizeType32 const beamWidth{std::get<2>(GetParam())};
    auto const cudaGraphMode = std::get<3>(GetParam());
    auto const microBatchSizes = std::get<4>(GetParam());
    auto const isChatGlmTest = std::get<5>(GetParam());

    if (!modelSpec.mUsePackedInput && modelSpec.mRandomEndId)
    {
        GTEST_SKIP() << "Test does not support endId test with padded inputs";
    }

    if (modelSpec.mRandomEndId && beamWidth > 1)
    {
        GTEST_SKIP() << "Test does not support endId test with beam search";
    }

    std::ostringstream gpuSizePath;
    gpuSizePath << "tp" << modelSpec.mTPSize << "-pp" << modelSpec.mPPSize << "-gpu";
    auto const modelPath{ENGINE_PATH / modelDir / modelSpec.getModelPath() / gpuSizePath.str()};
    auto const resultsPath
        = DATA_PATH / modelDir / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
    fs::path const resultsFile{resultsPath / modelSpec.getResultsFile()};

    // Warning: This should be the last check before running the test.
    // It will initialize MPI which can take significant time.
    if (modelSpec.mTPSize * modelSpec.mPPSize != COMM_SESSION.getSize())
    {
        GTEST_SKIP() << "Model's world size " << modelSpec.mPPSize * modelSpec.mTPSize
                     << " is not equal to the system world size";
    }

    testGptSession(modelPath, modelSpec, modelIds, beamWidth, kBatchSizes, resultsFile, mLogger, cudaGraphMode,
        microBatchSizes, isChatGlmTest);
}

INSTANTIATE_TEST_SUITE_P(GptSessionOtbTest, ParamTest,
    testing::Combine(testing::Values(ModelParams{GPT_MODEL_DIR, {50256, 50256}}),
        testing::Values(
            // single decoder
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kFLOAT}, ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF},
            // decoderBatch
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kFLOAT}.useDecoderPerRequest(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}.useDecoderPerRequest()

                ),
        testing::Values(1),           // beamWidth
        testing::Values(false, true), // cudaGraphMode
        testing::Values(MicroBatchSizes(), MicroBatchSizes{3, 3}, MicroBatchSizes{3, 6}),
        testing::Values(false)        // isChatGlmTest
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptSessionTest, ParamTest,
    testing::Combine(testing::Values(ModelParams{GPT_MODEL_DIR, {50256, 50256}}),
        testing::Values(
            // single decoder
            // Disabled because of flakey beam search test
            // ModelSpec{FP32_GPT_ATTENTION_DIR, FP32_PLUGIN_RESULT_FILE, nvinfer1::DataType::kFLOAT}
            //     .useGptAttentionPlugin(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}.useGptAttentionPlugin(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}.useGptAttentionPlugin().usePackedInput(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}.useGptAttentionPlugin().usePackedInput().setKVCacheType(
                KVCacheType::kPAGED),

            // decoderBatch
            // Disabled because of flakey beam search test
            // ModelSpec{FP32_GPT_ATTENTION_DIR, FP32_PLUGIN_RESULT_FILE, nvinfer1::DataType::kFLOAT}
            //     .useGptAttentionPlugin()
            //     .useDecoderPerRequest(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}.useGptAttentionPlugin().useDecoderPerRequest(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .useDecoderPerRequest(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useDecoderPerRequest(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useDecoderPerRequest()
                .useRandomEndId()

                ),
        testing::Values(1, 2),        // beamWidth
        testing::Values(false, true), // cudaGraphMode
        testing::Values(MicroBatchSizes(), MicroBatchSizes{3, 3}, MicroBatchSizes{3, 6}),
        testing::Values(false)        // isChatGlmTest
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptjSessionTest, ParamTest,
    testing::Combine(testing::Values(ModelParams{GPTJ_MODEL_DIR, {50256, 50256}}),
        testing::Values(
            // single decoder
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}.useGptAttentionPlugin(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}.useGptAttentionPlugin().usePackedInput(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}.useGptAttentionPlugin().usePackedInput().setKVCacheType(
                KVCacheType::kPAGED),
            // decoderBatch
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}.useGptAttentionPlugin().useDecoderPerRequest(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .useDecoderPerRequest(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useDecoderPerRequest()

                ),
        testing::Values(1, 2),  // beamWidth
        testing::Values(false), // cudaGraphMode
        testing::Values(MicroBatchSizes()),
        testing::Values(false)  // isChatGlmTest
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(MambaSessionOOTBTest, ParamTest,
    testing::Combine(testing::Values(ModelParams{MAMBA_MODEL_DIR, {0, 1}}),
        testing::Values(
            // single decoder
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}),
        testing::Values(1),     // beamWidth
        testing::Values(false), // cudaGraphMode
        testing::Values(MicroBatchSizes()),
        testing::Values(false)  // isChatGlmTest
        ),
    generateTestName);
INSTANTIATE_TEST_SUITE_P(MambaSessionPluginTest, ParamTest,
    testing::Combine(testing::Values(ModelParams{MAMBA_MODEL_DIR, {0, 1}}),
        testing::Values(ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}.useMambaPlugin()),
        testing::Values(1),     // beamWidth
        testing::Values(false), // cudaGraphMode
        testing::Values(MicroBatchSizes()),
        testing::Values(false)  // isChatGlmTest
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(LlamaSessionTest, ParamTest,
    testing::Combine(testing::Values(ModelParams{LLAMA_MODEL_DIR, {2, 2}}),
        testing::Values(
            // single decoder
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}.useGptAttentionPlugin().usePackedInput().setKVCacheType(
                KVCacheType::kPAGED),
            // decoderBatch
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useDecoderPerRequest(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useDecoderPerRequest()
                .usePipelineParallelism(4),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useDecoderPerRequest()
                .useTensorParallelism(4),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useDecoderPerRequest()
                .usePipelineParallelism(2)
                .useTensorParallelism(2)

                ),
        testing::Values(1, 2),  // beamWidth
        testing::Values(false), // cudaGraphMode
        testing::Values(MicroBatchSizes()),
        testing::Values(false)  // isChatGlmTest
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(ChatGlmSessionTest, ParamTest,
    testing::Combine(testing::Values(ModelParams{CHATGLM_MODEL_DIR, {130005, 3}}), // end_id, pad_id
        testing::Values(ModelSpec{CHATGLM_INPUT_FILE, nvinfer1::DataType::kHALF}.useGptAttentionPlugin()

                ),
        testing::Values(1, 2),  // beamWidth
        testing::Values(false), // cudaGraphMode
        testing::Values(MicroBatchSizes()),
        testing::Values(true)   // isChatGlmTest
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(ChatGlm2SessionTest, ParamTest,
    testing::Combine(testing::Values(ModelParams{CHATGLM2_MODEL_DIR, {2, 0}}), // end_id, pad_id
        testing::Values(ModelSpec{CHATGLM2_INPUT_FILE, nvinfer1::DataType::kHALF}.useGptAttentionPlugin()

                ),
        testing::Values(1, 2),  // beamWidth
        testing::Values(false), // cudaGraphMode
        testing::Values(MicroBatchSizes()),
        testing::Values(true)   // isChatGlmTest
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(ChatGlm3SessionTest, ParamTest,
    testing::Combine(testing::Values(ModelParams{CHATGLM3_MODEL_DIR, {2, 0}}), // end_id, pad_id
        testing::Values(ModelSpec{CHATGLM3_INPUT_FILE, nvinfer1::DataType::kHALF}.useGptAttentionPlugin()

                ),
        testing::Values(1, 2),  // beamWidth
        testing::Values(false), // cudaGraphMode
        testing::Values(MicroBatchSizes()),
        testing::Values(true)   // isChatGlmTest
        ),
    generateTestName);

class LlamaSessionOnDemandTest : public SessionTest
{
};

TEST_F(LlamaSessionOnDemandTest, SamplingFP16WithAttentionPlugin)
{
    GTEST_SKIP() << "Run only on demand";
    auto const modelDir = "llama_7bf";
    auto const engineDir = "llama_7bf_outputs_tp1";
    auto const modelPath{ENGINE_PATH / modelDir / engineDir};
    SizeType32 constexpr beamWidth{1};
    auto const batchSizes = {8};

    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto otherModelSpecPtr = std::make_shared<ModelSpec>(INPUT_FILE, dtype);
    auto const modelSpec = ModelSpec{INPUT_FILE, dtype}.useGptAttentionPlugin();
    fs::path resultsFile{DATA_PATH / modelDir / modelSpec.getResultsFile()};
    auto const modeIds = ModelIds{2, 2};

    testGptSession(
        modelPath, modelSpec, modeIds, beamWidth, batchSizes, resultsFile, mLogger, false, MicroBatchSizes());
}

TEST_F(LlamaSessionOnDemandTest, SamplingFP16AttentionPluginDecoderBatch)
{
    GTEST_SKIP() << "Run only on demand";
    auto const modelDir = "llamav2";
    auto const modelPath{ENGINE_PATH / modelDir};
    SizeType32 constexpr beamWidth{1};
    auto const batchSizes = {8};

    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto otherModelSpecPtr = std::make_shared<ModelSpec>(INPUT_FILE, dtype);
    auto const modelSpec = ModelSpec{INPUT_FILE, dtype, otherModelSpecPtr}
                               .useGptAttentionPlugin()
                               .usePackedInput()
                               .useDecoderPerRequest();
    fs::path resultsFile{DATA_PATH / modelDir / modelSpec.getResultsFile()};
    auto const modeIds = ModelIds{2, 2};

    testGptSession(
        modelPath, modelSpec, modeIds, beamWidth, batchSizes, resultsFile, mLogger, false, MicroBatchSizes());
}
