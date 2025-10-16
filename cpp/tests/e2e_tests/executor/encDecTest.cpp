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

#include "executorTest.h"

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tensorrt_llm/testing/modelSpec.h"
#include "tests/utils/common.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace tr = tensorrt_llm::runtime;

using namespace tensorrt_llm::testing;
using namespace tensorrt_llm::executor;
using namespace std::chrono_literals;
using tensorrt_llm::testing::KVCacheType;

namespace
{

std::string getEncDecEnginePath(std::string const& modelName, SizeType32 tp, SizeType32 pp, SizeType32 cp)
{
    return modelName + '/' + std::to_string(tp * pp * cp) + "-gpu/float16";
}

TokenIdType getDecTokenFromJsonConfig(std::filesystem::path decEnginePath, std::string const& token_name)
{
    TokenIdType tokenId = 0;
    try
    {
        std::ifstream decoderJsonConfigPath(decEnginePath / "config.json");
        auto const decoderPretrainedConfig
            = nlohmann::json::parse(decoderJsonConfigPath, nullptr, true, true).at("pretrained_config");
        tokenId = decoderPretrainedConfig.at(token_name).template get<int32_t>();
    }
    catch (nlohmann::json::out_of_range& e)
    {
        TLLM_LOG_ERROR(
            "Parameter %s cannot be found from decoder config.json in pretrained_config. Using default id 0.",
            token_name.c_str());
    }
    catch (nlohmann::json::type_error const& e)
    {
        TLLM_LOG_ERROR(
            "Parameter %s has a different type from decoder config.json in pretrained_config. Using default id 0.",
            token_name.c_str());
    }
    return tokenId;
}

} // namespace

using EncDecParamsType = std::tuple<std::string, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32,
    std::vector<SizeType32>>;

std::string generateTestNameEncDec(testing::TestParamInfo<EncDecParamsType> const& info)
{
    auto modelName = std::get<0>(info.param);
    auto const beamWidth = std::get<1>(info.param);
    auto const maxNewTokens = std::get<2>(info.param);
    auto const tp = std::get<3>(info.param);
    auto const pp = std::get<4>(info.param);

    // GTEST does not allow '-' in its test name
    for (auto& c : modelName)
    {
        if (c == '-')
        {
            c = '_';
        }
    }

    std::string name = "EncDecTest";
    name.append("_" + modelName);
    name.append("_BeamWidth" + std::to_string(beamWidth));
    name.append("_MaxNewTokens" + std::to_string(maxNewTokens));
    name.append("_TP" + std::to_string(tp));
    name.append("_PP" + std::to_string(pp));
    return name;
}

bool isLanguageAdapterName(std::string const& modelName)
{
    return modelName == LANGUAGE_ADAPTER_NAME;
}

class EncDecParamsTest : public GptExecutorTest, public ::testing::WithParamInterface<EncDecParamsType>
{
};

TEST_P(EncDecParamsTest, validEncDecCtor)
{
    auto const modelName = std::get<0>(GetParam());
    SizeType32 const beamWidth = std::get<1>(GetParam());
    SizeType32 const maxNewTokens = std::get<2>(GetParam());
    SizeType32 const tp = std::get<3>(GetParam());
    SizeType32 const pp = std::get<4>(GetParam());
    SizeType32 const cp = std::get<5>(GetParam());

    auto const enginePathName = getEncDecEnginePath(modelName, tp, pp, cp);
    std::filesystem::path encEnginePath = ENC_DEC_ENGINE_BASE / enginePathName / "encoder";
    std::filesystem::path decEnginePath = ENC_DEC_ENGINE_BASE / enginePathName / "decoder";
    ExecutorConfig executorConfig{};
    FloatType freeGpuMemoryFraction = 0.4f;
    FloatType crossKvCacheFraction = 0.4f;
    KvCacheConfig kvCacheConfig{false, std::nullopt, std::nullopt, std::nullopt, freeGpuMemoryFraction};
    kvCacheConfig.setCrossKvCacheFraction(crossKvCacheFraction);
    executorConfig.setKvCacheConfig(kvCacheConfig);
    auto executor = Executor(encEnginePath, decEnginePath, ModelType::kENCODER_DECODER, executorConfig);
}

TEST_P(EncDecParamsTest, Forward)
{
    bool constexpr VERBOSE = false;
    auto const modelName = std::get<0>(GetParam());
    SizeType32 const beamWidth = std::get<1>(GetParam());
    SizeType32 const maxNewTokens = std::get<2>(GetParam());
    SizeType32 const tp = std::get<3>(GetParam());
    SizeType32 const pp = std::get<4>(GetParam());
    SizeType32 const cp = std::get<5>(GetParam());

    // Parameters for language adapter test
    SizeType32 const numLanguages = std::get<6>(GetParam());
    std::vector<SizeType32> languageAdapterUids = std::get<7>(GetParam());

    bool const streaming = false;

    auto const enginePathName = getEncDecEnginePath(modelName, tp, pp, cp);
    std::filesystem::path encEnginePath = ENC_DEC_ENGINE_BASE / enginePathName / "encoder";
    std::filesystem::path decEnginePath = ENC_DEC_ENGINE_BASE / enginePathName / "decoder";

    // load ground truth input & output data
    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    auto inputsIdsHost
        = tr::utils::loadNpy(manager, (ENC_DEC_DATA_BASE / "input_ids.npy").string(), tr::MemoryType::kCPU);
    auto inputsIdsPtr = tr::bufferCast<TokenIdType>(*inputsIdsHost);
    auto inputLengthsHost
        = tr::utils::loadNpy(manager, (ENC_DEC_DATA_BASE / "input_lengths.npy").string(), tr::MemoryType::kCPU);
    auto inputLengthsPtr = tr::bufferCast<SizeType32>(*inputLengthsHost);
    auto encoderOutputHost
        = tr::utils::loadNpy(manager, (ENC_DEC_DATA_BASE / "encoder_output.npy").string(), tr::MemoryType::kCPU);
    auto encoderOutputPtr = tr::bufferCast<half>(*encoderOutputHost);
    auto decoderOutputHost = tr::utils::loadNpy(manager,
        (ENC_DEC_DATA_BASE / "output_ids_beam").string() + std::to_string(beamWidth) + ".npy", tr::MemoryType::kCPU);
    auto decoderOutputPtr = tr::bufferCast<TokenIdType>(*decoderOutputHost);

    // Rank and size info
    auto& comm = tensorrt_llm::mpi::MpiComm::world();
    auto const worldRank = comm.getRank();
    auto const worldSize = comm.getSize();

    // create executor
    BatchingType const batchingType = BatchingType::kINFLIGHT;
    FloatType freeGpuMemoryFraction = 0.5f;
    FloatType crossKvCacheFraction = 0.5f;
    KvCacheConfig kvCacheConfig{false, std::nullopt, std::nullopt, std::nullopt, freeGpuMemoryFraction};
    kvCacheConfig.setCrossKvCacheFraction(crossKvCacheFraction);

    ExecutorConfig executorConfig{beamWidth};
    executorConfig.setBatchingType(batchingType);
    executorConfig.setKvCacheConfig(kvCacheConfig);
    executorConfig.setNormalizeLogProbs(false);

    // TODO: OrchestratorMode test does not pass
    bool const useOrchestratorMode = (tp * pp) > worldSize;
    std::optional<OrchestratorConfig> orchestratorConfig = std::nullopt;
    if (useOrchestratorMode)
    {
        orchestratorConfig = OrchestratorConfig(true, PathUtil::EXECUTOR_WORKER_PATH());
    }
    auto parallelConfig = ParallelConfig(CommunicationType::kMPI,
        useOrchestratorMode ? CommunicationMode::kORCHESTRATOR : CommunicationMode::kLEADER, std::nullopt, std::nullopt,
        orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    auto executor = Executor(encEnginePath, decEnginePath, ModelType::kENCODER_DECODER, executorConfig);

    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = false;
    outConfig.returnLogProbs = false;
    outConfig.returnGenerationLogits = false;
    outConfig.returnContextLogits = false;
    outConfig.returnEncoderOutput = false;

    TokenIdType bosId = getDecTokenFromJsonConfig(decEnginePath, "bos_token_id");
    TokenIdType padId = getDecTokenFromJsonConfig(decEnginePath, "pad_token_id");
    TokenIdType eosId = getDecTokenFromJsonConfig(decEnginePath, "eos_token_id");
    TokenIdType decoderStartTokenId = getDecTokenFromJsonConfig(decEnginePath, "decoder_start_token_id");

    bool const isLanguageAdapterTest = isLanguageAdapterName(modelName);
    // create requests
    SizeType32 const nbRequests = inputLengthsHost->getShape().d[0];
    std::vector<Request> requests;
    for (int i = 0, cumInputLen = 0; i < nbRequests; i++)
    {
        auto encoderInput = VecTokens(&inputsIdsPtr[cumInputLen],
            &inputsIdsPtr[cumInputLen] + inputLengthsPtr[i]); // assume inputIds is flattened / no-padding
        cumInputLen += inputLengthsPtr[i];
        auto decoderInput = VecTokens{decoderStartTokenId};
        Request req(decoderInput, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig,
            eosId, padId);
        req.setEncoderInputTokenIds(encoderInput);
        if (isLanguageAdapterTest)
        {
            req.setLanguageAdapterUid(languageAdapterUids[i]);
        }
        requests.emplace_back(req);
    }

    using namespace std::chrono;

    // enqueue requests
    if (worldRank == 0)
    {
        auto tik = high_resolution_clock::now();
        std::vector<IdType> reqIds = executor.enqueueRequests(std::move(requests));

        // get responses
        milliseconds waitTime(5000);
        auto responsesAll = executor.awaitResponses(reqIds, waitTime);
        auto tok = high_resolution_clock::now();
        TLLM_LOG_DEBUG("TRT-LLM C++ E2E time %d ms", duration_cast<milliseconds>(tok - tik).count());
        TLLM_LOG_DEBUG("Number of responses: %d", responsesAll.size());

        int32_t numFinished = 0;
        int iter = 0;
        SizeType32 numResponses = 0;
        std::unordered_map<IdType, std::vector<VecTokens>> outputTokens;
        for_each(reqIds.begin(), reqIds.end(),
            [&outputTokens, &beamWidth](auto const& id)
            {
                TLLM_LOG_DEBUG("Request IDs: %d", id);
                outputTokens[id] = {};
                for (int i = 0; i < beamWidth; i++)
                {
                    outputTokens[id].emplace_back(VecTokens{});
                }
            });
        for (int i = 0; i < reqIds.size(); i++)
        {
            auto& responses = responsesAll[i];
            for (auto& response : responses)
            {
                numResponses++;
                if (!response.hasError())
                {
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                    for (int beam = 0; beam < beamWidth; beam++)
                    {
                        auto& resTokens = result.outputTokenIds.at(beam);
                        auto& outTokens = outputTokens.at(response.getRequestId()).at(beam);
                        outTokens.insert(outTokens.end(), std::make_move_iterator(resTokens.begin()),
                            std::make_move_iterator(resTokens.end()));
                    }
                }
                else
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
            }
        }

        // print output & check correctness with ground truth
        for (auto const& [reqId, tokens] : outputTokens)
        {
            SizeType32 gtMaxLength = decoderOutputHost->getShape().d[1];
            auto gtOutput = decoderOutputPtr + (reqId - 1) * gtMaxLength;

            if constexpr (VERBOSE)
            {
                std::cout << ">>> Request ID: " << reqId << std::endl;
                for (int beam = 0; beam < beamWidth; beam++)
                {
                    std::cout << "output tokens, beam " << beam << ", output length " << tokens[beam].size() << ": "
                              << std::endl;
                    for_each(tokens[beam].begin(), tokens[beam].end(),
                        [](auto const& token) { std::cout << token << ", "; });
                    std::cout << std::endl;
                }
                std::cout << "ground truth tokens: " << std::endl;

                SizeType32 gtLength = 0;
                for (int i = 0; i < gtMaxLength; i++)
                {
                    if (gtOutput[i] != eosId)
                    {
                        std::cout << gtOutput[i] << ", ";
                        gtLength++;
                    }
                }
                std::cout << std::endl;
                std::cout << "ground truth length: " << gtLength << std::endl;
            }

            // check token-by-token match between beam 0 & ground truth
            ASSERT_TRUE(tokens.size() <= gtMaxLength)
                << "Request ID " << reqId << "'s generated length is longer than ground truth length " << gtMaxLength;
            for (int i = 0; i < gtMaxLength; i++)
            {
                if (outConfig.excludeInputFromOutput)
                {
                    // if results exclude decoder start token, skip it in ground truth too
                    continue;
                }
                if (i < tokens[0].size())
                {
                    ASSERT_EQ(tokens[0][i], gtOutput[i])
                        << "Generated token id: " << tokens[0][i] << " v.s. ground truth: " << gtOutput[i];
                }
                else
                {
                    ASSERT_EQ(gtOutput[i], eosId) << "Request ID " << reqId << "'s generated length " << tokens.size()
                                                  << " is shorter than ground truth length " << gtMaxLength;
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(T5BasicTest, EncDecParamsTest,
    testing::Combine(testing::Values(T5_NAME), testing::Values(1), testing::Values(64), testing::Values(1),
        testing::Values(1), testing::Values(1), testing::Values(0), testing::Values(std::vector<SizeType32>{})),
    generateTestNameEncDec);

INSTANTIATE_TEST_SUITE_P(T5Beam2Test, EncDecParamsTest,
    testing::Combine(testing::Values(T5_NAME), testing::Values(2), testing::Values(64), testing::Values(1),
        testing::Values(1), testing::Values(1), testing::Values(0), testing::Values(std::vector<SizeType32>{})),
    generateTestNameEncDec);

INSTANTIATE_TEST_SUITE_P(T5MultiGPUTest, EncDecParamsTest,
    testing::Combine(testing::Values(T5_NAME), testing::Values(1), testing::Values(64), testing::Values(4),
        testing::Values(1), testing::Values(1), testing::Values(0), testing::Values(std::vector<SizeType32>{})),
    generateTestNameEncDec);

INSTANTIATE_TEST_SUITE_P(BartBasicTest, EncDecParamsTest,
    testing::Combine(testing::Values(BART_NAME), testing::Values(1), testing::Values(64), testing::Values(1),
        testing::Values(1), testing::Values(1), testing::Values(0), testing::Values(std::vector<SizeType32>{})),
    generateTestNameEncDec);

INSTANTIATE_TEST_SUITE_P(BartBeam2Test, EncDecParamsTest,
    testing::Combine(testing::Values(BART_NAME), testing::Values(2), testing::Values(64), testing::Values(1),
        testing::Values(1), testing::Values(1), testing::Values(0), testing::Values(std::vector<SizeType32>{})),
    generateTestNameEncDec);

INSTANTIATE_TEST_SUITE_P(BartMultiGPUTest, EncDecParamsTest,
    testing::Combine(testing::Values(BART_NAME), testing::Values(1), testing::Values(64), testing::Values(4),
        testing::Values(1), testing::Values(1), testing::Values(0), testing::Values(std::vector<SizeType32>{})),
    generateTestNameEncDec);

INSTANTIATE_TEST_SUITE_P(LanguageAdapterBasicTest, EncDecParamsTest,
    testing::Combine(testing::Values(LANGUAGE_ADAPTER_NAME), testing::Values(1), testing::Values(64),
        testing::Values(1), testing::Values(1), testing::Values(1), testing::Values(4),
        testing::Values(std::vector<SizeType32>{2, 3})),
    generateTestNameEncDec);
