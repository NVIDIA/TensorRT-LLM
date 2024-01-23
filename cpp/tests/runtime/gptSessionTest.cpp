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

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"

#include <algorithm>
#include <filesystem>
#include <mpi.h>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace fs = std::filesystem;

namespace
{
auto const TEST_RESOURCE_PATH = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources";
auto const ENGINE_PATH = TEST_RESOURCE_PATH / "models/rt_engine";
auto const DATA_PATH = TEST_RESOURCE_PATH / "data";

auto const GPT_MODEL_DIR = "gpt2";
auto const GPTJ_MODEL_DIR = "gpt-j-6b";
auto const LLAMA_MODEL_DIR = "llama-7b-hf";

// Engines need to be generated using cpp/tests/resources/scripts/build_gpt_engines.py.
auto const FP32_GPT_DIR = "fp32-default";
auto const FP32_GPT_ATTENTION_DIR = "fp32-plugin";
auto const FP16_GPT_DIR = "fp16-default";
auto const FP16_GPT_ATTENTION_DIR = "fp16-plugin";
auto const FP16_GPT_ATTENTION_PACKED_DIR = FP16_GPT_ATTENTION_DIR + std::string("-packed");
auto const FP16_GPT_ATTENTION_PACKED_PAGED_DIR = FP16_GPT_ATTENTION_PACKED_DIR + std::string("-paged");

// Expected outputs need to be generated using cpp/tests/resources/scripts/generate_expected_gpt_output.py.
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

class ModelSpec
{
public:
    ModelSpec(fs::path modelPath, fs::path resultsFile, nvinfer1::DataType dtype)
        : mModelPath{std::move(modelPath)}
        , mResultsFile{std::move(resultsFile)}
        , mDataType{dtype}
        , mUseGptAttentionPlugin{false}
        , mUsePackedInput{false}
        , mUsePagedKvCache{false}
        , mDecoderPerRequest{false}
        , mPPSize(1)
        , mTPSize(1)
        , mRandomEndId(false)
    {
    }

    ModelSpec& useGptAttentionPlugin()
    {
        mUseGptAttentionPlugin = true;
        return *this;
    }

    ModelSpec& usePackedInput()
    {
        mUsePackedInput = true;
        return *this;
    }

    ModelSpec& usePagedKvCache()
    {
        mUsePagedKvCache = true;
        return *this;
    }

    ModelSpec& useDecoderPerRequest()
    {
        mDecoderPerRequest = true;
        return *this;
    }

    ModelSpec& usePipelineParallelism(int ppSize)
    {
        mPPSize = ppSize;
        return *this;
    }

    ModelSpec& useTensorParallelism(int tpSize)
    {
        mTPSize = tpSize;
        return *this;
    }

    ModelSpec& useRandomEndId()
    {
        mRandomEndId = true;
        return *this;
    }

    fs::path mModelPath;
    fs::path mResultsFile;
    nvinfer1::DataType mDataType;
    bool mUseGptAttentionPlugin;
    bool mUsePackedInput;
    bool mUsePagedKvCache;
    bool mDecoderPerRequest;
    int mPPSize;
    int mTPSize;
    bool mRandomEndId;
};

struct MicroBatchSizes
{
    std::optional<SizeType> ctxMicroBatchSize{std::nullopt};
    std::optional<SizeType> genMicroBatchSize{std::nullopt};
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
void verifyModelConfig(GptModelConfig const& modelConfig, ModelSpec const& modelSpec)
{
    ASSERT_EQ(modelSpec.mUseGptAttentionPlugin, modelConfig.useGptAttentionPlugin());
    ASSERT_EQ(modelSpec.mUsePackedInput, modelConfig.usePackedInput());
    ASSERT_EQ(modelSpec.mUsePagedKvCache, modelConfig.usePagedKvCache());
    ASSERT_EQ(modelSpec.mDataType, modelConfig.getDataType());
}

void testGptSession(fs::path const& modelPath, ModelSpec const& modelSpec, ModelIds const modelIds, SizeType beamWidth,
    std::initializer_list<int> const& batchSizes, fs::path const& resultsFile,
    std::shared_ptr<nvinfer1::ILogger> const& logger, bool cudaGraphMode, MicroBatchSizes microBatchSizes,
    bool const isChatGlmTest = false, std::string const& modelName = "")
{
    auto manager = BufferManager(std::make_shared<CudaStream>());

    ASSERT_TRUE(fs::exists(DATA_PATH));
    fs::path inputPath = DATA_PATH / "input_tokens.npy";
    std::string fileNameSuffix;
    if (isChatGlmTest)
    {
        ASSERT_TRUE(fs::exists(DATA_PATH / modelName));
        const int batchSize = *batchSizes.begin();
        fileNameSuffix
            = std::string("-BS") + std::to_string(batchSize) + "-BM" + std::to_string(beamWidth) + std::string(".npy");
        inputPath = DATA_PATH / modelName / (std::string("inputId") + fileNameSuffix);
    }

    auto const& givenInput = utils::loadNpy(manager, inputPath.string(), MemoryType::kCPU);
    auto const& inputShape = givenInput->getShape();
    ASSERT_EQ(inputShape.nbDims, 2);
    ASSERT_GT(inputShape.d[0], 0);

    auto const nbGivenInputs = static_cast<SizeType>(inputShape.d[0]);
    std::string outputPath = resultsFile.string();
    if (isChatGlmTest)
    {
        fs::path expectedOutputPath = DATA_PATH / modelName / (std::string("outputId") + fileNameSuffix);
        outputPath = expectedOutputPath.string();
    }

    auto expectedOutput = utils::loadNpy(manager, outputPath, MemoryType::kCPU);
    auto const& outputShape = expectedOutput->getShape();
    if (isChatGlmTest)
    {
        ASSERT_EQ(outputShape.nbDims, 3);
        ASSERT_EQ(inputShape.d[0], outputShape.d[0]);
    }
    else
    {
        ASSERT_EQ(outputShape.nbDims, 2);
        ASSERT_EQ(inputShape.d[0] * beamWidth, outputShape.d[0]);
    }

    auto const givenInputData = bufferCast<TokenIdType const>(*givenInput);
    auto expectedOutputData = bufferCast<TokenIdType>(*expectedOutput);

    ASSERT_TRUE(fs::exists(modelPath));
    auto const json = GptJsonConfig::parse(modelPath / "config.json");
    auto const modelConfig = json.getModelConfig();
    verifyModelConfig(modelConfig, modelSpec);

    const int worldSize = modelSpec.mTPSize * modelSpec.mPPSize;
    auto const worldConfig = WorldConfig::mpi(worldSize, modelSpec.mTPSize, modelSpec.mPPSize);

    auto enginePath = modelPath / json.engineFilename(worldConfig);
    ASSERT_TRUE(fs::exists(enginePath));

    auto const maxInputLength = static_cast<SizeType>(inputShape.d[1]);

    auto const maxSeqLengthGroundTruth = isChatGlmTest ? static_cast<SizeType>(outputShape.d[2]) : 0;
    SizeType maxNewTokens, maxSeqLength;
    if (isChatGlmTest)
    {
        maxNewTokens = 512;
        maxSeqLength = maxInputLength + maxNewTokens;
    }
    else
    {
        maxSeqLength = static_cast<SizeType>(outputShape.d[1]);
        ASSERT_LT(maxInputLength, maxSeqLength);
        maxNewTokens = maxSeqLength - maxInputLength;
    }

    SamplingConfig samplingConfig{beamWidth};
    samplingConfig.temperature = std::vector{1.0f};
    SizeType const minLength = 1;
    samplingConfig.minLength = std::vector{minLength};
    if (isChatGlmTest)
    {
        samplingConfig.randomSeed = std::vector{static_cast<uint64_t>(1ull)};
        samplingConfig.topK = std::vector{1};
        samplingConfig.topP = std::vector{1.0f};
        samplingConfig.lengthPenalty = std::vector{1.0f};
    }
    else
    {
        samplingConfig.randomSeed = std::vector{static_cast<uint64_t>(42ull)};
        samplingConfig.topK = std::vector{0};
        samplingConfig.topP = std::vector{0.0f};
    }

    auto const padId = modelIds.padId;
    auto endId = modelIds.endId;

    std::vector<SizeType> givenInputLengths(nbGivenInputs);
    for (SizeType i = 0; i < nbGivenInputs; ++i)
    {
        auto const seqBegin = givenInputData + i * maxInputLength;
        auto const it = std::find(seqBegin, seqBegin + maxInputLength, padId);
        givenInputLengths[i] = std::distance(seqBegin, it);
    }

    std::vector<SizeType> expectedLengths;
    if (!isChatGlmTest)
    {
        std::srand(42);
        if (modelSpec.mRandomEndId)
        {
            const auto endIdRow = std::rand() % nbGivenInputs;
            const auto endIdBeam = std::rand() % beamWidth;
            const auto endIdCol = givenInputLengths[endIdRow] + minLength + std::rand() % (maxNewTokens - minLength);
            auto const endIdIndex = tc::flat_index2((endIdRow * beamWidth + endIdBeam), endIdCol, maxSeqLength);
            endId = expectedOutputData[endIdIndex];
        }

        expectedLengths.resize(nbGivenInputs * beamWidth);
        for (SizeType bi = 0; bi < nbGivenInputs; ++bi)
        {
            for (SizeType beam = 0; beam < beamWidth; ++beam)
            {
                SizeType expectedLen = givenInputLengths[bi] + maxNewTokens;
                if (modelSpec.mRandomEndId)
                {
                    for (SizeType si = givenInputLengths[bi]; si < maxSeqLength; ++si)
                    {
                        auto const expectIndex = tc::flat_index2((bi * beamWidth + beam), si, maxSeqLength);
                        if (expectedOutputData[expectIndex] == endId)
                        {
                            expectedLen = si;
                            break;
                        }
                    }
                    // Fill new EOS token to the expected data
                    for (SizeType si = expectedLen; si < maxSeqLength; ++si)
                    {
                        auto const expectIndex = tc::flat_index2((bi * beamWidth + beam), si, maxSeqLength);
                        expectedOutputData[expectIndex] = endId;
                    }
                }
                expectedLengths[bi * beamWidth + beam] = expectedLen;
            }
        }
    }

    auto const maxBatchSize = *std::max_element(batchSizes.begin(), batchSizes.end());
    GptSession::Config sessionConfig{maxBatchSize, beamWidth, maxSeqLength};
    sessionConfig.decoderPerRequest = modelSpec.mDecoderPerRequest;
    sessionConfig.ctxMicroBatchSize = microBatchSizes.ctxMicroBatchSize;
    sessionConfig.genMicroBatchSize = microBatchSizes.genMicroBatchSize;
    sessionConfig.cudaGraphMode = cudaGraphMode;

    GptSession session{sessionConfig, modelConfig, worldConfig, enginePath.string(), logger};
    EXPECT_EQ(session.getDevice(), worldConfig.getDevice());
    // Use bufferManager for copying data to and from the GPU
    auto& bufferManager = session.getBufferManager();

    for (auto const batchSize : batchSizes)
    {
        if (!isChatGlmTest)
        {
            std::cout << "=== batchSize:" << batchSize << " ===\n";
        }

        // use 5 to 12 tokens from input
        std::vector<SizeType> inputLengthsHost(batchSize);
        for (SizeType i = 0; i < batchSize; ++i)
        {
            const int inputIdx = i % nbGivenInputs;
            inputLengthsHost[i] = givenInputLengths[inputIdx];
        }
        auto inputLengths = bufferManager.copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

        // copy inputs and wrap into shared_ptr
        GenerationInput::TensorPtr inputIds;
        if (modelConfig.usePackedInput())
        {
            std::vector<SizeType> inputOffsetsHost(batchSize + 1);
            tc::stl_utils::inclusiveScan(
                inputLengthsHost.begin(), inputLengthsHost.end(), inputOffsetsHost.begin() + 1);
            auto const totalInputSize = inputOffsetsHost.back();

            std::vector<std::int32_t> inputsHost(totalInputSize);
            for (SizeType i = 0; i < batchSize; ++i)
            {
                auto const seqBegin = givenInputData + (i % nbGivenInputs) * maxInputLength;
                std::copy(seqBegin, seqBegin + inputLengthsHost[i], inputsHost.begin() + inputOffsetsHost[i]);
            }
            inputIds = bufferManager.copyFrom(inputsHost, ITensor::makeShape({1, totalInputSize}), MemoryType::kGPU);
        }
        else
        {
            std::vector<std::int32_t> inputsHost(batchSize * maxInputLength, padId);
            for (SizeType i = 0; i < batchSize; ++i)
            {
                auto const seqBegin = givenInputData + (i % nbGivenInputs) * maxInputLength;
                std::copy(seqBegin, seqBegin + inputLengthsHost[i], inputsHost.begin() + i * maxInputLength);
            }
            inputIds
                = bufferManager.copyFrom(inputsHost, ITensor::makeShape({batchSize, maxInputLength}), MemoryType::kGPU);
        }

        GenerationInput generationInput{
            endId, padId, std::move(inputIds), std::move(inputLengths), modelConfig.usePackedInput()};
        if (!isChatGlmTest)
        {
            generationInput.maxNewTokens = maxNewTokens;
        }

        // runtime will allocate memory for output if this tensor is empty
        GenerationOutput generationOutput{bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32),
            bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)};

        // repeat the same inputs multiple times for testing idempotency of `generate()`
        auto constexpr repetitions = 10;
        for (auto r = 0; r < repetitions; ++r)
        {
            SizeType numSteps = 0;

            if (!isChatGlmTest)
            {
                generationOutput.onTokenGenerated
                    = [&numSteps, &modelSpec, maxNewTokens](
                          [[maybe_unused]] GenerationOutput::TensorPtr const& outputIds, SizeType step, bool finished)
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
            }

            session.generate(generationOutput, generationInput, samplingConfig);

            // compare outputs
            if (!isChatGlmTest && worldConfig.isFirstPipelineParallelRank())
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

            if (isChatGlmTest || worldConfig.isFirstPipelineParallelRank())
            {
                auto const& outputIds = generationOutput.ids;
                auto const& outputDims = outputIds->getShape();
                EXPECT_EQ(outputDims.nbDims, 3);
                EXPECT_EQ(outputDims.d[0], batchSize) << "r: " << r;
                EXPECT_EQ(outputDims.d[1], beamWidth) << "r: " << r;
                if (!isChatGlmTest)
                {
                    EXPECT_EQ(outputDims.d[2], maxSeqLength) << "r: " << r;
                }
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
                            int expectIndex;
                            int outputIndex = tc::flat_index3(b, beam, i, beamWidth, maxSeqLength);
                            if (isChatGlmTest)
                            {
                                expectIndex = tc::flat_index3(b, beam, i, beamWidth, maxSeqLengthGroundTruth);
                            }
                            else
                            {
                                expectIndex = tc::flat_index2((b % nbGivenInputs * beamWidth + beam), i, maxSeqLength);
                            }

                            if (!isChatGlmTest && expectedOutputData[expectIndex] == endId)
                            {
                                break;
                            }
                            EXPECT_EQ(output[outputIndex], expectedOutputData[expectIndex])
                                << " b: " << b << " beam: " << beam << " i: " << i;
                            anyMismatch |= (output[outputIndex] != expectedOutputData[expectIndex]);

                            if (isChatGlmTest && output[outputIndex] == endId)
                            {
                                break;
                            }
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

using ParamType = std::tuple<ModelParams, ModelSpec, SizeType, bool, MicroBatchSizes>;

std::string generateTestName(const testing::TestParamInfo<ParamType>& info)
{
    auto const modelSpec = std::get<1>(info.param);
    std::string name{modelSpec.mDataType == nvinfer1::DataType::kFLOAT ? "Float" : "Half"};
    auto const beamWidth = std::get<2>(info.param);
    name.append(beamWidth == 1 ? "Sampling" : "BeamWidth" + std::to_string(beamWidth));
    if (modelSpec.mUseGptAttentionPlugin)
        name.append("AttentionPlugin");
    if (modelSpec.mUsePackedInput)
        name.append("Packed");
    if (modelSpec.mUsePagedKvCache)
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
    SizeType const beamWidth{std::get<2>(GetParam())};
    auto const cudaGraphMode = std::get<3>(GetParam());
    auto const microBatchSizes = std::get<4>(GetParam());

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
    auto const modelPath{ENGINE_PATH / modelDir / modelSpec.mModelPath / gpuSizePath.str()};
    auto const resultsPath
        = DATA_PATH / modelDir / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
    fs::path const resultsFile{resultsPath / modelSpec.mResultsFile};

    // Warning: This should be the last check before running the test.
    // It will initialize MPI which can take significant time.
    if (!WorldConfig::validConfig(modelSpec.mTPSize, modelSpec.mPPSize))
    {
        GTEST_SKIP() << "Model's world size " << modelSpec.mPPSize * modelSpec.mTPSize
                     << " is not equal to the system world size";
    }

    testGptSession(
        modelPath, modelSpec, modelIds, beamWidth, kBatchSizes, resultsFile, mLogger, cudaGraphMode, microBatchSizes);
}

INSTANTIATE_TEST_SUITE_P(GptSessionOtbTest, ParamTest,
    testing::Combine(testing::Values(ModelParams{GPT_MODEL_DIR, {50256, 50256}}),
        testing::Values(
            // single decoder
            ModelSpec{FP32_GPT_DIR, FP32_RESULT_FILE, nvinfer1::DataType::kFLOAT},
            ModelSpec{FP16_GPT_DIR, FP16_RESULT_FILE, nvinfer1::DataType::kHALF},
            // decoderBatch
            ModelSpec{FP32_GPT_DIR, FP32_RESULT_FILE, nvinfer1::DataType::kFLOAT}.useDecoderPerRequest(),
            ModelSpec{FP16_GPT_DIR, FP16_RESULT_FILE, nvinfer1::DataType::kHALF}.useDecoderPerRequest()

                ),
        testing::Values(1 /*, 2*/),   // beamWidth, DISABLED beam search
        testing::Values(false, true), // cudaGraphMode
        testing::Values(MicroBatchSizes(), MicroBatchSizes{3, 3}, MicroBatchSizes{3, 6})),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptSessionTest, ParamTest,
    testing::Combine(testing::Values(ModelParams{GPT_MODEL_DIR, {50256, 50256}}),
        testing::Values(
            // single decoder
            ModelSpec{FP32_GPT_ATTENTION_DIR, FP32_PLUGIN_RESULT_FILE, nvinfer1::DataType::kFLOAT}
                .useGptAttentionPlugin(),
            ModelSpec{FP16_GPT_ATTENTION_DIR, FP16_PLUGIN_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin(),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_DIR, FP16_PLUGIN_PACKED_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput(),
            ModelSpec{
                FP16_GPT_ATTENTION_PACKED_PAGED_DIR, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .usePagedKvCache(),
            // decoderBatch
            ModelSpec{FP32_GPT_ATTENTION_DIR, FP32_PLUGIN_RESULT_FILE, nvinfer1::DataType::kFLOAT}
                .useGptAttentionPlugin()
                .useDecoderPerRequest(),
            ModelSpec{FP16_GPT_ATTENTION_DIR, FP16_PLUGIN_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .useDecoderPerRequest(),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_DIR, FP16_PLUGIN_PACKED_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .useDecoderPerRequest(),
            ModelSpec{
                FP16_GPT_ATTENTION_PACKED_PAGED_DIR, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .usePagedKvCache()
                .useDecoderPerRequest(),
            ModelSpec{
                FP16_GPT_ATTENTION_PACKED_PAGED_DIR, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .usePagedKvCache()
                .useDecoderPerRequest()
                .useRandomEndId()

                ),
        testing::Values(1, 2),        // beamWidth
        testing::Values(false, true), // cudaGraphMode
        testing::Values(MicroBatchSizes(), MicroBatchSizes{3, 3}, MicroBatchSizes{3, 6})),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptjSessionTest, ParamTest,
    testing::Combine(testing::Values(ModelParams{GPTJ_MODEL_DIR, {50256, 50256}}),
        testing::Values(
            // single decoder
            ModelSpec{FP16_GPT_ATTENTION_DIR, FP16_PLUGIN_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin(),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_DIR, FP16_PLUGIN_PACKED_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput(),
            ModelSpec{
                FP16_GPT_ATTENTION_PACKED_PAGED_DIR, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .usePagedKvCache(),
            // decoderBatch
            ModelSpec{FP16_GPT_ATTENTION_DIR, FP16_PLUGIN_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .useDecoderPerRequest(),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_DIR, FP16_PLUGIN_PACKED_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .useDecoderPerRequest(),
            ModelSpec{
                FP16_GPT_ATTENTION_PACKED_PAGED_DIR, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .usePagedKvCache()
                .useDecoderPerRequest()

                ),
        testing::Values(1, 2),  // beamWidth
        testing::Values(false), // cudaGraphMode
        testing::Values(MicroBatchSizes())),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(LlamaSessionTest, ParamTest,
    testing::Combine(testing::Values(ModelParams{LLAMA_MODEL_DIR, {2, 2}}),
        testing::Values(
            // single decoder
            ModelSpec{
                FP16_GPT_ATTENTION_PACKED_PAGED_DIR, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .usePagedKvCache(),
            // decoderBatch
            ModelSpec{
                FP16_GPT_ATTENTION_PACKED_PAGED_DIR, FP16_PLUGIN_PACKED_PAGED_RESULT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .usePagedKvCache()
                .useDecoderPerRequest(),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP4_FILE,
                nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .usePagedKvCache()
                .useDecoderPerRequest()
                .usePipelineParallelism(4),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, FP16_PLUGIN_PACKED_PAGED_RESULT_TP4_PP1_FILE,
                nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .usePagedKvCache()
                .useDecoderPerRequest()
                .useTensorParallelism(4),
            ModelSpec{FP16_GPT_ATTENTION_PACKED_PAGED_DIR, FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE,
                nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .usePagedKvCache()
                .useDecoderPerRequest()
                .usePipelineParallelism(2)
                .useTensorParallelism(2)

                ),
        testing::Values(1, 2),  // beamWidth
        testing::Values(false), // cudaGraphMode
        testing::Values(MicroBatchSizes())),
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
    SizeType constexpr beamWidth{1};
    fs::path resultsFile{DATA_PATH / modelDir / FP16_RESULT_FILE};
    auto const batchSizes = {8};

    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const modelSpec = ModelSpec{"", "", dtype}.useGptAttentionPlugin();
    auto const modeIds = ModelIds{2, 2};

    testGptSession(
        modelPath, modelSpec, modeIds, beamWidth, batchSizes, resultsFile, mLogger, false, MicroBatchSizes());
}

TEST_F(LlamaSessionOnDemandTest, SamplingFP16AttentionPluginDecoderBatch)
{
    GTEST_SKIP() << "Run only on demand";
    auto const modelDir = "llamav2";
    auto const modelPath{ENGINE_PATH / modelDir};
    SizeType constexpr beamWidth{1};
    fs::path resultsFile{DATA_PATH / modelDir / FP16_RESULT_FILE};
    auto const batchSizes = {8};

    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const modelSpec = ModelSpec{"", "", dtype}.useGptAttentionPlugin().usePackedInput().useDecoderPerRequest();
    auto const modeIds = ModelIds{2, 2};

    testGptSession(
        modelPath, modelSpec, modeIds, beamWidth, batchSizes, resultsFile, mLogger, false, MicroBatchSizes());
}

class ChatGlmSessionTest : public SessionTest // for ChatGLM-6B
{
};

class ChatGlm2SessionTest : public SessionTest // for ChatGLM2-6B
{
};

class ChatGlm3SessionTest : public SessionTest // for ChatGLM3-6B
{
};

TEST_F(ChatGlmSessionTest, HalfSamplingAttentionPluginBS1BM1)
{
    auto const modelName{"chatglm_6b"};
    auto const modelPath{ENGINE_PATH / modelName};
    auto const batchSizes = {1};
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const modelSpec = ModelSpec{"", "", dtype}.useGptAttentionPlugin();
    auto const modeIds = ModelIds{130005, 130005};

    testGptSession(
        modelPath, modelSpec, modeIds, 1, batchSizes, "", mLogger, false, MicroBatchSizes(), true, modelName);
}

TEST_F(ChatGlm2SessionTest, HalfSamplingAttentionPluginBS1BM1)
{
    auto const modelName{"chatglm2_6b"};
    auto const modelPath{ENGINE_PATH / modelName};
    auto const batchSizes = {1};
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const modelSpec = ModelSpec{"", "", dtype}.useGptAttentionPlugin();
    auto const modeIds = ModelIds{2, 2};

    testGptSession(
        modelPath, modelSpec, modeIds, 1, batchSizes, "", mLogger, false, MicroBatchSizes(), true, modelName);
}

TEST_F(ChatGlm2SessionTest, HalfSamplingAttentionPluginBS2BM1)
{
    auto const modelName{"chatglm2_6b"};
    auto const modelPath{ENGINE_PATH / modelName};
    auto const batchSizes = {2};
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const modelSpec = ModelSpec{"", "", dtype}.useGptAttentionPlugin();
    auto const modeIds = ModelIds{2, 2};

    testGptSession(
        modelPath, modelSpec, modeIds, 1, batchSizes, "", mLogger, false, MicroBatchSizes(), true, modelName);
}

TEST_F(ChatGlm2SessionTest, HalfSamplingAttentionPluginBS1BM2)
{
    auto const modelName{"chatglm2_6b"};
    auto const modelPath{ENGINE_PATH / modelName};
    auto const batchSizes = {1};
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const modelSpec = ModelSpec{"", "", dtype}.useGptAttentionPlugin();
    auto const modeIds = ModelIds{2, 2};

    testGptSession(
        modelPath, modelSpec, modeIds, 2, batchSizes, "", mLogger, false, MicroBatchSizes(), true, modelName);
}

TEST_F(ChatGlm3SessionTest, HalfSamplingAttentionPluginBS1BM1)
{
    auto const modelName{"chatglm3_6b"};
    auto const modelPath{ENGINE_PATH / modelName};
    auto const batchSizes = {1};
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const modelSpec = ModelSpec{"", "", dtype}.useGptAttentionPlugin();
    auto const modeIds = ModelIds{2, 2};

    testGptSession(
        modelPath, modelSpec, modeIds, 1, batchSizes, "", mLogger, false, MicroBatchSizes(), true, modelName);
}
