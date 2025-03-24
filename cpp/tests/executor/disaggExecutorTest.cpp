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

#include "disaggExecutor.h"
#include "executorTest.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tests/utils/common.h"

namespace tr = tensorrt_llm::runtime;

using namespace tensorrt_llm::testing;

using DisaggParamsType = std::tuple<int, std::vector<std::string>, std::vector<std::vector<int>>,
    std::vector<std::vector<int>>, std::vector<int>, int>;

enum InstanceRole : int
{
    CONTEXT = 1,
    GENERATION = 0,
    MIXED = 2
};

std::string generateTestNameDisaggParams(testing::TestParamInfo<DisaggParamsType> const& info)
{
    auto const processNum = std::get<0>(info.param);
    auto const modelNames = std::get<1>(info.param);
    auto const participantIdsEachInstance = std::get<2>(info.param);       // std::vector<std::vector<int>>
    auto const participantDeviceIdsEachInstance = std::get<3>(info.param); // std::vector<std::vector<int>>;
    auto const instanceRoles = std::get<4>(info.param); // std::vector<int> ; //1 is context , 0 is generation
    auto const controllerRank = std::get<5>(info.param);

    auto convertToString = [](std::vector<std::vector<int>> const& vec)
    {
        std::ostringstream oss;
        oss << "XX";

        for (size_t i = 0; i < vec.size(); ++i)
        {
            for (size_t j = 0; j < vec[i].size(); ++j)
            {
                oss << vec[i][j];
                if (j < vec[i].size() - 1)
                {
                    oss << "_";
                }
            }
            if (i < vec.size() - 1)
            {
                oss << "X_X";
            }
        }

        oss << "XX";
        return oss.str();
    };
    std::string name = "DisaggExecutorTest_";

    name.append("ProcessNum_" + std::to_string(processNum));
    // name.append("_contextModel_" + contextModel + "_genModel_" + genModel);
    name.append("_modelNames_");
    for (auto&& modelName : modelNames)
    {
        name.append(modelName).append("_");
    }

    name.append("_controllerRank_" + std::to_string(controllerRank));

    name.append("_ranks_").append(convertToString(participantIdsEachInstance));
    name.append("_devices_").append(convertToString(participantDeviceIdsEachInstance));
    name.append("_roles_").append(convertToString({instanceRoles}));
    name.append("_controllerRank_" + std::to_string(controllerRank));

    return name;
}

class DisaggParamsTest : public GptExecutorTest, public ::testing::WithParamInterface<DisaggParamsType>
{
};

class DisaggOrchestratorParamsTest : public GptExecutorTest, public ::testing::WithParamInterface<DisaggParamsType>
{
};

namespace
{
void verifyGenerateDistStats(std::deque<RequestStatsPerIteration> const& iterationStats)
{
    for (auto const& iteration : iterationStats)
    {
        for (auto const& requestStats : iteration.requestStats)
        {
            // exclude context only requests for mixed server
            if (requestStats.stage == RequestStage::kGENERATION_COMPLETE && requestStats.numGeneratedTokens > 1)
            {
                EXPECT_TRUE(requestStats.disServingStats.has_value());
                EXPECT_GT(requestStats.disServingStats.value().kvCacheTransferMS, 0.0);
            }
            if (requestStats.stage != RequestStage::kQUEUED)
            {
                EXPECT_TRUE(requestStats.disServingStats.has_value());
            }
            else
            {
                EXPECT_FALSE(requestStats.disServingStats.has_value());
            }
        }
    }
}
} // namespace

void runDisaggTest(tensorrt_llm::testing::disaggexecutor::DisaggExecutorLeader& executor,
    tensorrt_llm::runtime::BufferManager& manager, ITensor const& givenInput, ModelIds const& modelIds,
    FlakyTestInfo const& flakyTestInfo, bool streaming, SizeType32 const vocabSizePadded, BeamResult const& beamResult,
    OutputConfig const& outConfig, bool isSpeculativeDecoding, int maxWaitMs, BatchingType batchingType,
    bool returnAllGeneratedTokens)
{

    auto& comm = tensorrt_llm::mpi::MpiComm::world();
    auto const worldRank = comm.getRank();
    auto const worldSize = comm.getSize();
    auto const beamWidth = beamResult.beamWidth;

    std::unordered_map<IdType, SizeType32> reqIdToBatchId;
    std::unordered_map<SizeType32, std::vector<BeamTokens>> tokens;
    auto [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(givenInput, modelIds.padId);
    auto const* const givenInputData = tr::bufferCast<TokenIdType const>(givenInput);

    auto const& inputShape = givenInput.getShape();
    ASSERT_EQ(inputShape.nbDims, 2);
    ASSERT_GT(inputShape.d[0], 0);

    // Load expected outputs for each beam width value
    auto testData = TestData::loadTestData(beamResult, givenInput, beamWidth, manager, outConfig, modelIds);
    auto const maxSeqLen = testData.maxSeqLen;

    // Load expected outputs and inputs
    SizeType32 numRequests = static_cast<SizeType32>(givenInputLengths.size());
    SizeType32 maxRequests = numRequests;
    std::vector<Request> requests;
    std::vector<SizeType32> reqMaxNewTokens;
    SizeType32 const numReturnSequences = 1;

    for (SizeType32 req = 0; req < maxRequests; ++req)
    {
        SizeType32 inputLen = givenInputLengths.at(req);
        auto maxNewTokens = maxSeqLen - maxInputLength;
        reqMaxNewTokens.push_back(maxNewTokens);
        SizeType32 endId = -1;
        auto const* const seqBegin = givenInputData + req * maxInputLength;
        VecTokens tokens(seqBegin, seqBegin + inputLen);
        auto samplingConfig = tensorrt_llm::executor::SamplingConfig(beamWidth);
        samplingConfig.setNumReturnSequences(numReturnSequences);
        auto request = Request(
            VecTokens(seqBegin, seqBegin + inputLen), maxNewTokens, streaming, samplingConfig, outConfig, endId);
        request.setReturnAllGeneratedTokens(returnAllGeneratedTokens);
        request.setRequestType(RequestType::REQUEST_TYPE_CONTEXT_ONLY);
        requests.emplace_back(std::move(request));
    }

    if (executor.isControllerRank())
    {
        std::vector<IdType> reqIds;

        for (int i = 0; i < requests.size(); ++i)
        {
            std::vector<BeamTokens> resultTokens;
            resultTokens.reserve(numReturnSequences);
            for (SizeType32 seqIdx = 0; seqIdx < numReturnSequences; ++seqIdx)
            {
                resultTokens.emplace_back(beamWidth);
            }
            auto retReqId = executor.enqueueRequests({requests[i]});
            reqIds.push_back(retReqId.front());
            tokens[i] = std::move(resultTokens);
            reqIdToBatchId[retReqId.front()] = i;
        }

        // Get the new tokens for each requests
        int32_t numFinished = 0;
        int iter = 0;
        SizeType32 numResponses = 0;
        while (numFinished < maxRequests && iter < maxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(waitTime);
            for (auto& response : responses)
            {
                numResponses++;
                if (!response.hasError())
                {
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                    auto batchId = reqIdToBatchId.at(response.getRequestId());
                    auto seqIdx = result.sequenceIndex;

                    auto& contextLogits = result.contextLogits;
                    auto& genLogits = result.generationLogits;
                    auto& outputTokenIds = result.outputTokenIds;

                    EXPECT_EQ(result.finishReasons.size(), beamWidth);
                    for (SizeType32 beam = 0; beam < beamWidth; ++beam)
                    {
                        auto& newTokens = outputTokenIds.at(beam);
                        auto& reqTokens = tokens.at(batchId).at(seqIdx).at(beam);

                        reqTokens.insert(reqTokens.end(), newTokens.begin(), newTokens.end());
                        // FinishReason is only supported for bw=1 and inflight batching.
                        if (beamWidth == 1 && batchingType == BatchingType::kINFLIGHT)
                        {
                            EXPECT_EQ(result.finishReasons.at(beam),
                                result.isFinal ? FinishReason::kLENGTH : FinishReason::kNOT_FINISHED);
                        }
                    }

                    auto& cumLogProbs = result.cumLogProbs;
                    auto& logProbs = result.logProbs;
                    auto& beamTokens = tokens.at(batchId).at(seqIdx);
                    testData.verifyLogProbs(outConfig.returnLogProbs, streaming, outConfig.excludeInputFromOutput,
                        givenInputLengths.at(batchId), beamWidth, beamTokens, cumLogProbs, logProbs, batchId,
                        flakyTestInfo);

                    testData.validateContextLogits(outConfig.returnContextLogits, givenInputLengths.at(batchId),
                        beamWidth, contextLogits, vocabSizePadded, batchId, batchingType);
                    testData.validateGenerationLogits(outConfig.returnGenerationLogits, result.isFinal, streaming,
                        outConfig.excludeInputFromOutput, givenInputLengths.at(batchId), reqMaxNewTokens.at(batchId),
                        beamWidth, beamTokens, genLogits, vocabSizePadded, batchId, batchingType,
                        returnAllGeneratedTokens);
                }
                else
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
            }
            ++iter;
        }
        EXPECT_LT(iter, maxWaitMs);
        testData.verifyOutput(tokens, givenInputLengths, nbGivenInputs, streaming, outConfig.excludeInputFromOutput,
            flakyTestInfo, isSpeculativeDecoding, returnAllGeneratedTokens, beamWidth, numReturnSequences, false);
    }
    comm.barrier();
    if (executor.isGenerationRank())
    {
        verifyGenerateDistStats(executor.getLatestRequestStats());
    }
}

void runDisaggTest(DisaggExecutorOrchestrator& executor, tensorrt_llm::runtime::BufferManager& manager,
    ITensor const& givenInput, ModelIds const& modelIds, FlakyTestInfo const& flakyTestInfo, bool streaming,
    SizeType32 const vocabSizePadded, BeamResult const& beamResult, OutputConfig const& outConfig,
    bool isSpeculativeDecoding, int maxWaitMs, BatchingType batchingType, bool returnAllGeneratedTokens)
{

    auto& comm = tensorrt_llm::mpi::MpiComm::world();
    auto const worldRank = comm.getRank();
    auto const worldSize = comm.getSize();
    auto const beamWidth = beamResult.beamWidth;

    std::unordered_map<IdType, SizeType32> reqIdToBatchId;
    std::unordered_map<SizeType32, std::vector<BeamTokens>> tokens;
    // std::unordered_map<IdType, IdType> gGenIdIdTogContextId;
    auto [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(givenInput, modelIds.padId);
    auto const* const givenInputData = tr::bufferCast<TokenIdType const>(givenInput);

    auto const& inputShape = givenInput.getShape();
    ASSERT_EQ(inputShape.nbDims, 2);
    ASSERT_GT(inputShape.d[0], 0);

    // Load expected outputs for each beam width value
    auto testData = TestData::loadTestData(beamResult, givenInput, beamWidth, manager, outConfig, modelIds);
    auto const maxSeqLen = testData.maxSeqLen;

    // Load expected outputs and inputs
    SizeType32 numRequests = static_cast<SizeType32>(givenInputLengths.size());
    SizeType32 maxRequests = numRequests;
    std::vector<Request> requests;
    std::vector<SizeType32> reqMaxNewTokens;
    SizeType32 const numReturnSequences = 1;

    for (SizeType32 req = 0; req < maxRequests; ++req)
    {
        SizeType32 inputLen = givenInputLengths.at(req);
        auto maxNewTokens = maxSeqLen - maxInputLength;
        reqMaxNewTokens.push_back(maxNewTokens);
        SizeType32 endId = -1;
        auto const* const seqBegin = givenInputData + req * maxInputLength;
        VecTokens tokens(seqBegin, seqBegin + inputLen);
        auto samplingConfig = tensorrt_llm::executor::SamplingConfig(beamWidth);
        samplingConfig.setNumReturnSequences(numReturnSequences);
        auto request = Request(
            VecTokens(seqBegin, seqBegin + inputLen), maxNewTokens, streaming, samplingConfig, outConfig, endId);
        request.setReturnAllGeneratedTokens(returnAllGeneratedTokens);
        request.setRequestType(RequestType::REQUEST_TYPE_CONTEXT_ONLY);
        requests.emplace_back(std::move(request));
    }

    if (worldRank == 0)
    {
        std::vector<IdType> reqIds;

        for (int i = 0; i < requests.size(); ++i)
        {
            std::vector<BeamTokens> resultTokens;
            resultTokens.reserve(numReturnSequences);
            for (SizeType32 seqIdx = 0; seqIdx < numReturnSequences; ++seqIdx)
            {
                resultTokens.emplace_back(beamWidth);
            }
            auto retReqId = executor.enqueueContext({requests[i]}, std::nullopt);
            reqIds.push_back(retReqId.front());
            tokens[i] = std::move(resultTokens);
            reqIdToBatchId[retReqId.front()] = i;
        }

        int32_t numContextFinished = 0;
        int contextIter = 0;
        while (numContextFinished < maxRequests && contextIter < maxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);

            auto contextResponses = executor.awaitContextResponses(waitTime);
            contextIter++;
            numContextFinished += contextResponses.size();

            for (auto&& responseWithId : contextResponses)
            {
                auto contextGid = responseWithId.gid;
                int batchId = reqIdToBatchId[contextGid];
                auto&& request = requests[batchId];
                request.setRequestType(RequestType::REQUEST_TYPE_GENERATION_ONLY);
                request.setContextPhaseParams(responseWithId.response.getResult().contextPhaseParams.value());
                executor.enqueueGeneration({request}, {responseWithId.gid}, std::nullopt);
            }
        }
        // Get the new tokens for each requests
        int32_t numFinished = 0;
        int iter = 0;
        SizeType32 numResponses = 0;
        while (numFinished < maxRequests && iter < maxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitGenerationResponses(waitTime);
            for (auto& responseWithId : responses)
            {
                numResponses++;
                if (!responseWithId.response.hasError())
                {
                    auto result = responseWithId.response.getResult();
                    numFinished += result.isFinal;
                    auto batchId = reqIdToBatchId.at(responseWithId.gid);
                    auto seqIdx = result.sequenceIndex;

                    auto& contextLogits = result.contextLogits;
                    auto& genLogits = result.generationLogits;
                    auto& outputTokenIds = result.outputTokenIds;

                    EXPECT_EQ(result.finishReasons.size(), beamWidth);
                    for (SizeType32 beam = 0; beam < beamWidth; ++beam)
                    {
                        auto& newTokens = outputTokenIds.at(beam);
                        auto& reqTokens = tokens.at(batchId).at(seqIdx).at(beam);

                        reqTokens.insert(reqTokens.end(), newTokens.begin(), newTokens.end());
                        // FinishReason is only supported for bw=1 and inflight batching.
                        if (beamWidth == 1 && batchingType == BatchingType::kINFLIGHT)
                        {
                            EXPECT_EQ(result.finishReasons.at(beam),
                                result.isFinal ? FinishReason::kLENGTH : FinishReason::kNOT_FINISHED);
                        }
                    }

                    auto& cumLogProbs = result.cumLogProbs;
                    auto& logProbs = result.logProbs;
                    auto& beamTokens = tokens.at(batchId).at(seqIdx);
                    testData.verifyLogProbs(outConfig.returnLogProbs, streaming, outConfig.excludeInputFromOutput,
                        givenInputLengths.at(batchId), beamWidth, beamTokens, cumLogProbs, logProbs, batchId,
                        flakyTestInfo);

                    testData.validateContextLogits(outConfig.returnContextLogits, givenInputLengths.at(batchId),
                        beamWidth, contextLogits, vocabSizePadded, batchId, batchingType);
                    testData.validateGenerationLogits(outConfig.returnGenerationLogits, result.isFinal, streaming,
                        outConfig.excludeInputFromOutput, givenInputLengths.at(batchId), reqMaxNewTokens.at(batchId),
                        beamWidth, beamTokens, genLogits, vocabSizePadded, batchId, batchingType,
                        returnAllGeneratedTokens);
                }
                else
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(responseWithId.gid)
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(responseWithId.response.getErrorMsg(), err);
                }
            }
            ++iter;
        }
        EXPECT_LT(iter, maxWaitMs);
        testData.verifyOutput(tokens, givenInputLengths, nbGivenInputs, streaming, outConfig.excludeInputFromOutput,
            flakyTestInfo, isSpeculativeDecoding, returnAllGeneratedTokens, beamWidth, numReturnSequences, false);
    }
    comm.barrier();
}

TEST_P(DisaggParamsTest, DisaggTokenComparison)
{

#if ENABLE_MULTI_DEVICE

    if (!(tensorrt_llm::common::getEnvUseUCXKvCache()))
    {
        setenv("UCX_TLS", "^cuda_ipc", 1); // disable cuda_ipc for testing for mpi
    }
    auto const processNum = std::get<0>(GetParam());
    auto const modelNames = std::get<1>(GetParam());
    auto const participantIdsEachInstance = std::get<2>(GetParam());       // std::vector<std::vector<int>>
    auto const participantDeviceIdsEachInstance = std::get<3>(GetParam()); // std::vector<std::vector<int>>;
    auto const instanceRoles
        = std::get<4>(GetParam()); // std::vector<int> ; //1 is context , 0 is generation, 2 is mixed
    auto const controllerRank = std::get<5>(GetParam());

    // params_check
    auto const& world_comm = tensorrt_llm::mpi::MpiComm::world();
    int const commRank = world_comm.getRank();
    int const commSize = world_comm.getSize();
    if (commSize != processNum)
    {
        GTEST_SKIP() << " need " << processNum << " processes but got " << commSize << " mpi processes, skip test.";
    }
    ASSERT_EQ(participantIdsEachInstance.size(), participantDeviceIdsEachInstance.size());
    SizeType32 instanceNum = participantIdsEachInstance.size();
    ASSERT_EQ(instanceNum, instanceRoles.size());
    ASSERT_EQ(instanceNum, modelNames.size());
    ASSERT_GE(controllerRank, 0);
    ASSERT_LT(controllerRank, commSize);
    int ranksNum = 0;
    std::unordered_map<SizeType32, SizeType32> rankCounter;
    std::unordered_map<SizeType32, SizeType32> deviceCounter;
    SizeType32 deviceRuseNum = 1;
    bool isContext = false;
    bool isGeneration = false;
    std::vector<int> participatntIds;
    std::vector<int> deviceIds;
    std::string modelName;
    bool isController = (commRank == controllerRank);
    for (SizeType32 i = 0; i < instanceNum; i++)
    {
        auto const& ranksThisInstance = participantIdsEachInstance[i];
        auto const& devicesThisInstance = participantDeviceIdsEachInstance[i];

        ASSERT_EQ(ranksThisInstance.size(), devicesThisInstance.size());
        SizeType32 rankNumThisInstance = ranksThisInstance.size();
        ASSERT_GT(rankNumThisInstance, 0);
        ranksNum += rankNumThisInstance;
        for (SizeType32 j = 0; j < rankNumThisInstance; j++)
        {
            rankCounter[ranksThisInstance[j]]++;
            deviceCounter[devicesThisInstance[j]]++;
            ASSERT_GE(rankCounter[ranksThisInstance[j]], 1);
            deviceRuseNum = std::max(deviceCounter[devicesThisInstance[j]], deviceRuseNum);
            ASSERT_GE(ranksThisInstance[j], 0);
            ASSERT_LT(ranksThisInstance[j], commSize);

            if (commRank == ranksThisInstance[j])
            {
                participatntIds = ranksThisInstance;
                deviceIds = devicesThisInstance;
                isContext = instanceRoles[i] == InstanceRole::CONTEXT || instanceRoles[i] == InstanceRole::MIXED;
                isGeneration = instanceRoles[i] == InstanceRole::GENERATION || instanceRoles[i] == InstanceRole::MIXED;
                // modelName = isContext ? contextModel : genModel;
                modelName = modelNames[i];
            }
        }
    }
    ASSERT_GE(ranksNum, commSize);

    OutputConfig outConfig;
    int const beamWidth = 1;
    BeamResult beamResult{beamWidth};

    bool streaming = false;
    int const maxBeamWidth = 1;
    ASSERT_TRUE(fs::exists(DATA_PATH));

    fs::path modelPath;
    // set defaults and adjust if needed by different models
    fs::path inputPath = DATA_PATH / "input_tokens.npy";
    ModelIds modelIds{50256, 50256};
    bool isSpeculativeDecoding{false};

    // NOTE: This can be used to disable checks for certain prompt batch entries
    FlakyTestInfo flakyTestInfo;

    if (modelName == "gpt")
    {
        auto const resultsPath
            = GPT_DATA_PATH / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
        if (outConfig.returnContextLogits || outConfig.returnGenerationLogits)
        {
            modelPath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR() / "tp1-pp1-cp1-gpu";
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE();
            beamResult.contextLogitsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE();
            beamResult.genLogitsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE();
            if (outConfig.returnLogProbs)
            {
                beamResult.cumLogProbsFile
                    = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_CUM_LOG_PROBS_FILE();
                beamResult.logProbsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_LOG_PROBS_FILE();
            }
        }
        else
        {
            modelPath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_FILE();
            if (outConfig.returnLogProbs)
            {
                beamResult.cumLogProbsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_CUM_LOG_PROBS_FILE();
                beamResult.logProbsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_LOG_PROBS_FILE();
            }
        }
    }
    else if (modelName == "llama_tp4_pp1_cp1" || modelName == "llama_tp1_pp4_cp1" || modelName == "llama_tp2_pp2_cp1"
        || modelName == "llama_tp1_pp2_cp1" || modelName == "llama_tp2_pp1_cp1" || modelName == "llama_tp1_pp1_cp1")
    {
        auto const resultsPath
            = LLAMA_DATA_PATH / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
        modelIds.padId = 2;
        modelIds.endId = 2;
        if (modelName == "llama_tp4_pp1_cp1")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP4_PP1_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp4-pp1-cp1-gpu";
        }
        else if (modelName == "llama_tp1_pp4_cp1")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP4_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp4-cp1-gpu";
        }
        else if (modelName == "llama_tp1_pp2_cp1")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP2_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp2-cp1-gpu";
        }
        else if (modelName == "llama_tp2_pp1_cp1")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP1_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp2-pp1-cp1-gpu";
        }
        else if (modelName == "llama_tp2_pp2_cp1")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp2-pp2-cp1-gpu";
        }
        else if (modelName == "llama_tp1_pp1_cp1")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
        }
    }
    else
    {
        TLLM_THROW("Unrecognized modelName");
    }

    // Warning: This should be the last check before running the test.
    // It will initialize MPI which can take significant time.
    if (modelName == "llama_tp4_pp1_cp1" || modelName == "llama_tp1_pp4_cp1" || modelName == "llama_tp2_pp2_cp1"
        || modelName == "llama_tp1_pp2_cp1" || modelName == "llama_tp2_pp1_cp1")
    {
        // For llama model, only run for multiple GPUs
        // This is detected by setting an env variable when running the test
        char const* val = getenv("RUN_LLAMA_MULTI_GPU");
        if (val == NULL)
        {
            GTEST_SKIP() << "Skipping Llama test";
        }
        else
        {
            if (outConfig.returnLogProbs || outConfig.returnContextLogits || outConfig.returnGenerationLogits)
            {
                GTEST_SKIP() << "Skipping logits and log probs tests for mpi runs";
            }
        }
    }

    SizeType32 constexpr vocabSizePadded{50257}; // gpt vocabSizePadded

    // Returning logits will bring higher latency
    if (streaming && (outConfig.returnContextLogits || outConfig.returnGenerationLogits))
    {
        mMaxWaitMs = 20000;
    }

    auto executorConfig = ExecutorConfig(maxBeamWidth);
    FloatType freeGpuMemoryFraction = 0.9f / (deviceRuseNum); // context and gen instance run on same device
    KvCacheConfig kvCacheConfig{true, std::nullopt, std::nullopt, std::nullopt, freeGpuMemoryFraction};
    executorConfig.setKvCacheConfig(kvCacheConfig);
    executorConfig.setRequestStatsMaxIterations(1000);
    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    auto const& givenInput = tr::utils::loadNpy(manager, inputPath.string(), tr::MemoryType::kCPU);
    auto [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(*givenInput, modelIds.padId);
    world_comm.barrier();
    auto disaggExecutor = tensorrt_llm::testing::disaggexecutor::DisaggExecutorLeader(modelPath,
        ModelType::kDECODER_ONLY, executorConfig, isController, isContext, isGeneration, givenInputLengths.size(),
        participatntIds, deviceIds, commRank);

    runDisaggTest(disaggExecutor, manager, *givenInput, modelIds, flakyTestInfo, streaming, vocabSizePadded, beamResult,
        outConfig, isSpeculativeDecoding, mMaxWaitMs, executorConfig.getBatchingType(), false);

#else

    GTEST_SKIP() << "Skipping DisaggExecutor Test";

#endif
}

TEST_P(DisaggOrchestratorParamsTest, DisaggTokenComparison)
{

#if ENABLE_MULTI_DEVICE

    if (!(tensorrt_llm::common::getEnvUseUCXKvCache()))
    {
        setenv("UCX_TLS", "^cuda_ipc", 1); // disable cuda_ipc for testing for mpi
    }
    auto const processNum = std::get<0>(GetParam());
    auto const modelNames = std::get<1>(GetParam());
    auto const participantIdsEachInstance = std::get<2>(GetParam());       // std::vector<std::vector<int>>
    auto const participantDeviceIdsEachInstance = std::get<3>(GetParam()); // std::vector<std::vector<int>>;
    auto const instanceRoles = std::get<4>(GetParam()); // std::vector<int> ; //1 is context , 0 is generation
    auto const controllerRank = std::get<5>(GetParam());

    // params_check
    auto const& world_comm = tensorrt_llm::mpi::MpiComm::world();
    int const commRank = world_comm.getRank();
    int const commSize = world_comm.getSize();
    if (commSize != processNum)
    {
        GTEST_SKIP() << " need " << processNum << " processes but got " << commSize << " mpi processes, skip test.";
    }
    ASSERT_EQ(participantIdsEachInstance.size(), participantDeviceIdsEachInstance.size());
    SizeType32 instanceNum = participantIdsEachInstance.size();
    ASSERT_EQ(instanceNum, instanceRoles.size());
    ASSERT_EQ(instanceNum, modelNames.size());
    ASSERT_GE(controllerRank, 0);
    ASSERT_LT(controllerRank, commSize);
    std::string modelName = modelNames[0];
    bool isController = (commRank == controllerRank);
    std::vector<fs::path> contextModels;
    std::vector<fs::path> genModels;

    auto getModelPath = [=](std::string modelNN)
    {
        fs::path retPath;
        if (modelNN == "llama_tp4_pp1")
        {
            retPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp4-pp1-cp1-gpu";
        }
        else if (modelNN == "llama_tp1_pp4")
        {
            retPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp4-cp1-gpu";
        }
        else if (modelNN == "llama_tp1_pp2")
        {
            retPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp2-cp1-gpu";
        }
        else if (modelNN == "llama_tp2_pp1")
        {
            retPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp2-pp1-cp1-gpu";
        }
        else if (modelNN == "llama_tp2_pp2")
        {
            retPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp2-pp2-cp1-gpu";
        }
        else if (modelNN == "llama_tp1_pp1")
        {
            retPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
        }
        return retPath;
    };
    for (SizeType32 i = 0; i < instanceNum; i++)
    {
        if (instanceRoles[i] == 1)
        {
            contextModels.push_back(getModelPath(modelNames[i]));
        }
        else
        {
            genModels.push_back(getModelPath(modelNames[i]));
        }
    }

    OutputConfig outConfig;
    int const beamWidth = 1;
    BeamResult beamResult{beamWidth};

    bool streaming = false;
    int const maxBeamWidth = 1;
    ASSERT_TRUE(fs::exists(DATA_PATH));

    fs::path modelPath;
    // set defaults and adjust if needed by different models
    fs::path inputPath = DATA_PATH / "input_tokens.npy";
    ModelIds modelIds{50256, 50256};
    bool isSpeculativeDecoding{false};

    // NOTE: This can be used to disable checks for certain prompt batch entries
    FlakyTestInfo flakyTestInfo;
    if (modelName == "llama_tp4_pp1" || modelName == "llama_tp1_pp4" || modelName == "llama_tp2_pp2"
        || modelName == "llama_tp1_pp2" || modelName == "llama_tp2_pp1" || modelName == "llama_tp1_pp1")
    {
        auto const resultsPath
            = LLAMA_DATA_PATH / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
        modelIds.padId = 2;
        modelIds.endId = 2;
        if (modelName == "llama_tp4_pp1")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP4_PP1_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp4-pp1-cp1-gpu";
        }
        else if (modelName == "llama_tp1_pp4")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP4_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp4-cp1-gpu";
        }
        else if (modelName == "llama_tp1_pp2")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP2_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp2-cp1-gpu";
        }
        else if (modelName == "llama_tp2_pp1")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP1_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp2-pp1-cp1-gpu";
        }
        else if (modelName == "llama_tp2_pp2")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp2-pp2-cp1-gpu";
        }
        else if (modelName == "llama_tp1_pp1")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
        }
    }

    else
    {
        TLLM_THROW("Unrecognized modelName");
    }

    // Warning: This should be the last check before running the test.
    // It will initialize MPI which can take significant time.
    if (modelName == "llama_tp4_pp1" || modelName == "llama_tp1_pp4" || modelName == "llama_tp2_pp2"
        || modelName == "llama_tp1_pp2" || modelName == "llama_tp2_pp1")
    {
        // For llama model, only run for multiple GPUs
        // This is detected by setting an env variable when running the test
        char const* val = getenv("RUN_LLAMA_MULTI_GPU");
        if (val == NULL)
        {
            GTEST_SKIP() << "Skipping Llama test";
        }
        else
        {
            if (outConfig.returnLogProbs || outConfig.returnContextLogits || outConfig.returnGenerationLogits)
            {
                GTEST_SKIP() << "Skipping logits and log probs tests for mpi runs";
            }
        }
    }

    SizeType32 constexpr vocabSizePadded{50257}; // gpt vocabSizePadded

    // Returning logits will bring higher latency
    if (streaming && (outConfig.returnContextLogits || outConfig.returnGenerationLogits))
    {
        mMaxWaitMs = 20000;
    }

    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    auto const& givenInput = tr::utils::loadNpy(manager, inputPath.string(), tr::MemoryType::kCPU);
    auto [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(*givenInput, modelIds.padId);
    world_comm.barrier();
    auto contextNum = contextModels.size();
    auto genNum = genModels.size();
    //     int deviceCount = -1;
    // TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    bool isOrchestrator = commRank == 0;
    std::vector<ExecutorConfig> ctxExecutorConfigs;
    std::vector<ExecutorConfig> genExecutorConfigs;
    for (int in = 0; in < instanceNum; in++)
    {
        tensorrt_llm::executor::SchedulerConfig schedulerConfig(CapacitySchedulerPolicy::kMAX_UTILIZATION);
        KvCacheConfig kvCacheConfig{true, std::nullopt, std::nullopt, std::nullopt, 0.2};

        tensorrt_llm::executor::ExecutorConfig executorConfig(maxBeamWidth, schedulerConfig, kvCacheConfig);
        tensorrt_llm::executor::OrchestratorConfig orchestratorConfig{isOrchestrator, "", nullptr, false};
        tensorrt_llm::executor::ParallelConfig parallelConfig{tensorrt_llm::executor::CommunicationType::kMPI,
            tensorrt_llm::executor::CommunicationMode::kORCHESTRATOR, participantDeviceIdsEachInstance.at(in),
            participantIdsEachInstance.at(in), orchestratorConfig};
        executorConfig.setParallelConfig(parallelConfig);
        if (in < contextNum)
        {
            ctxExecutorConfigs.push_back(executorConfig);
        }
        else
        {
            genExecutorConfigs.push_back(executorConfig);
        }
    }
    auto disaggExecutor
        = DisaggExecutorOrchestrator(contextModels, genModels, ctxExecutorConfigs, genExecutorConfigs, true, true);

    runDisaggTest(disaggExecutor, manager, *givenInput, modelIds, flakyTestInfo, streaming, vocabSizePadded, beamResult,
        outConfig, isSpeculativeDecoding, mMaxWaitMs, BatchingType::kINFLIGHT, false);

#else

    GTEST_SKIP() << "Skipping DisaggExecutor Test";

#endif
}

INSTANTIATE_TEST_SUITE_P(GptDisaggSymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(2), testing::Values(std::vector<std::string>{"gpt", "gpt"}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}}), testing::Values(std::vector<int>{1, 0}),
        testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(GptDisaggSymmetricExecutorTest2, DisaggParamsTest,
    testing::Combine(testing::Values(2), testing::Values(std::vector<std::string>{"gpt", "gpt"}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}}), testing::Values(std::vector<int>{1, 0}),
        testing::Values(1)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(GptDisaggSymmetricExecutorMixedTest, DisaggParamsTest,
    testing::Combine(testing::Values(2), testing::Values(std::vector<std::string>{"gpt", "gpt"}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}}), testing::Values(std::vector<int>{2, 2}),
        testing::Values(1)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(GptSingleDeviceDisaggSymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(2), testing::Values(std::vector<std::string>{"gpt", "gpt"}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}}),
        testing::Values(std::vector<std::vector<int>>{{0}, {0}}), testing::Values(std::vector<int>{1, 0}),
        testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(GptSingleDeviceDisaggSymmetricExecutorMixedTest, DisaggParamsTest,
    testing::Combine(testing::Values(2), testing::Values(std::vector<std::string>{"gpt", "gpt"}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}}),
        testing::Values(std::vector<std::vector<int>>{{0}, {0}}), testing::Values(std::vector<int>{2, 2}),
        testing::Values(1)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaTP2DisaggSymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(4),
        testing::Values(std::vector<std::string>{"llama_tp2_pp1_cp1", "llama_tp2_pp1_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0, 1}, {2, 3}}),
        testing::Values(std::vector<std::vector<int>>{{0, 1}, {2, 3}}), testing::Values(std::vector<int>{1, 0}),
        testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaPP2DisaggSymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(4),
        testing::Values(std::vector<std::string>{"llama_tp1_pp2_cp1", "llama_tp1_pp2_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0, 1}, {2, 3}}),
        testing::Values(std::vector<std::vector<int>>{{1, 0}, {3, 2}}), testing::Values(std::vector<int>{1, 0}),
        testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaTP2DisaggSymmetricExecutorMixedTest, DisaggParamsTest,
    testing::Combine(testing::Values(2), testing::Values(std::vector<std::string>{"llama_tp2_pp1_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0, 1}}), testing::Values(std::vector<std::vector<int>>{{0, 1}}),
        testing::Values(std::vector<int>{2}), testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaPP2DisaggSymmetricExecutorMixedTest, DisaggParamsTest,
    testing::Combine(testing::Values(2), testing::Values(std::vector<std::string>{"llama_tp1_pp2_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0, 1}}), testing::Values(std::vector<std::vector<int>>{{0, 1}}),
        testing::Values(std::vector<int>{2}), testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaTP2PP2DisaggSymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(8),
        testing::Values(std::vector<std::string>{"llama_tp2_pp2_cp1", "llama_tp2_pp2_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0, 1, 2, 3}, {4, 5, 6, 7}}),
        testing::Values(std::vector<std::vector<int>>{{2, 3, 0, 1}, {2, 3, 0, 1}}),
        testing::Values(std::vector<int>{1, 0}), testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaConPP2GenTP2DisaggAsymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(4),
        testing::Values(std::vector<std::string>{"llama_tp1_pp2_cp1", "llama_tp2_pp1_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0, 1}, {2, 3}}), // (1,0) (2,3)
        testing::Values(std::vector<std::vector<int>>{{1, 0}, {2, 3}}), testing::Values(std::vector<int>{1, 0}),
        testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaConTP2GenPP2DisaggAsymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(4),
        testing::Values(std::vector<std::string>{"llama_tp2_pp1_cp1", "llama_tp1_pp2_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0, 1}, {2, 3}}), // (0,1), (3,2)
        testing::Values(std::vector<std::vector<int>>{{0, 1}, {3, 2}}), testing::Values(std::vector<int>{1, 0}),
        testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaConTP2PP2GenPP2DisaggAsymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(6),
        testing::Values(std::vector<std::string>{"llama_tp2_pp2_cp1", "llama_tp1_pp2_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0, 1, 2, 3}, {4, 5}}), // (2,3,0,1) , (5,4)
        testing::Values(std::vector<std::vector<int>>{{2, 3, 0, 1}, {1, 0}}), testing::Values(std::vector<int>{1, 0}),
        testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaConTP2PP2GenTP2DisaggAsymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(6),
        testing::Values(std::vector<std::string>{"llama_tp2_pp2_cp1", "llama_tp2_pp1_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0, 1, 2, 3}, {4, 5}}), // (2,3,0,1), (4,5)
        testing::Values(std::vector<std::vector<int>>{{2, 3, 0, 1}, {0, 1}}), testing::Values(std::vector<int>{1, 0}),
        testing::Values(0)),
    generateTestNameDisaggParams);
INSTANTIATE_TEST_SUITE_P(LlamaConTP2PP1GenTP2PP2DisaggAsymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(6),
        testing::Values(std::vector<std::string>{"llama_tp2_pp1_cp1", "llama_tp2_pp2_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0, 1}, {2, 3, 4, 5}}), // (0,1) , (4,5,2,3)%4
        testing::Values(std::vector<std::vector<int>>{{0, 1}, {0, 1, 2, 3}}), testing::Values(std::vector<int>{1, 0}),
        testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaConTP2GenPP4DisaggAsymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(6),
        testing::Values(std::vector<std::string>{"llama_tp2_pp1_cp1", "llama_tp1_pp4_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{4, 5}, {0, 1, 2, 3}}), // (4,5) ,(3,2,1,0)
        testing::Values(std::vector<std::vector<int>>{{0, 1}, {3, 2, 1, 0}}), testing::Values(std::vector<int>{1, 0}),
        testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaCon4TP1Gen1TP4DisaggAsymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(8),
        testing::Values(std::vector<std::string>{
            "llama_tp1_pp1_cp1", "llama_tp1_pp1_cp1", "llama_tp1_pp1_cp1", "llama_tp1_pp1_cp1", "llama_tp4_pp1_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}, {2}, {3}, {4, 5, 6, 7}}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}, {2}, {3}, {0, 1, 2, 3}}),
        testing::Values(std::vector<int>{1, 1, 1, 1, 0}), testing::Values(4)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaCon2TP1Gen2TP2AndPP2DisaggAsymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(6),
        testing::Values(std::vector<std::string>{
            "llama_tp1_pp1_cp1", "llama_tp1_pp1_cp1", "llama_tp2_pp1_cp1", "llama_tp1_pp2_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}, {2, 3}, {4, 5}}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}, {2, 3}, {1, 0}}),
        testing::Values(std::vector<int>{1, 1, 0, 0}), testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaCon2TP1Gen2PP2DisaggAsymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(6),
        testing::Values(std::vector<std::string>{
            "llama_tp1_pp1_cp1", "llama_tp1_pp1_cp1", "llama_tp1_pp2_cp1", "llama_tp1_pp2_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}, {2, 3}, {4, 5}}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}, {3, 2}, {1, 0}}),
        testing::Values(std::vector<int>{1, 1, 0, 0}), testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaCon4TP1Gen1TP2PP2DisaggAsymmetricExecutorTest, DisaggParamsTest,
    testing::Combine(testing::Values(8),
        testing::Values(std::vector<std::string>{
            "llama_tp1_pp1_cp1", "llama_tp1_pp1_cp1", "llama_tp1_pp1_cp1", "llama_tp1_pp1_cp1", "llama_tp2_pp2_cp1"}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}, {2}, {3}, {4, 5, 6, 7}}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}, {2}, {3}, {2, 3, 0, 1}}),
        testing::Values(std::vector<int>{1, 1, 1, 1, 0}), testing::Values(4)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaCon2TP1Gen2TP2DisaaggOrchestrator, DisaggOrchestratorParamsTest,
    testing::Combine(testing::Values(7),
        testing::Values(std::vector<std::string>{"llama_tp1_pp1", "llama_tp1_pp1", "llama_tp2_pp1", "llama_tp2_pp1"}),
        testing::Values(std::vector<std::vector<int>>{{1}, {2}, {3, 4}, {5, 6}}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}, {2, 3}, {0, 1}}),
        testing::Values(std::vector<int>{1, 1, 0, 0}), testing::Values(0)),
    generateTestNameDisaggParams);
// for disaggOrchestrator 1->0, 2->1, 3->2, 4->3, 5->0, 6->1

INSTANTIATE_TEST_SUITE_P(LlamaCon2TP2Gen2TP1DisaaggOrchestrator, DisaggOrchestratorParamsTest,
    testing::Combine(testing::Values(7),
        testing::Values(std::vector<std::string>{"llama_tp2_pp1", "llama_tp2_pp1", "llama_tp1_pp1", "llama_tp1_pp1"}),
        testing::Values(std::vector<std::vector<int>>{{1, 2}, {3, 4}, {5}, {6}}),
        testing::Values(std::vector<std::vector<int>>{{0, 1}, {2, 3}, {0}, {1}}),
        testing::Values(std::vector<int>{1, 1, 0, 0}), testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaCon2TP1Gen2PP2DisaaggOrchestrator, DisaggOrchestratorParamsTest,
    testing::Combine(testing::Values(7),
        testing::Values(std::vector<std::string>{"llama_tp1_pp1", "llama_tp1_pp1", "llama_tp1_pp2", "llama_tp1_pp2"}),
        testing::Values(std::vector<std::vector<int>>{{1}, {2}, {3, 4}, {5, 6}}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}, {3, 2}, {1, 0}}),
        testing::Values(std::vector<int>{1, 1, 0, 0}), testing::Values(0)),
    generateTestNameDisaggParams);

INSTANTIATE_TEST_SUITE_P(LlamaCon2TP1Gen1TP2PP2DisaaggOrchestrator, DisaggOrchestratorParamsTest,
    testing::Combine(testing::Values(7),
        testing::Values(std::vector<std::string>{"llama_tp1_pp1", "llama_tp1_pp1", "llama_tp2_pp2"}),
        testing::Values(std::vector<std::vector<int>>{{1}, {2}, {3, 4, 5, 6}}),
        testing::Values(std::vector<std::vector<int>>{{0}, {1}, {0, 1, 2, 3}}),
        testing::Values(std::vector<int>{1, 1, 0}), testing::Values(0)),
    generateTestNameDisaggParams);
