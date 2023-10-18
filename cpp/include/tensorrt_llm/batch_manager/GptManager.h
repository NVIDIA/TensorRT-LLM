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

#pragma once

#include "tensorrt_llm/batch_manager/BatchManager.h"
#include "tensorrt_llm/batch_manager/batchScheduler.h"
#include "tensorrt_llm/batch_manager/callbacks.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/trtGptModelOptionalParams.h"
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>

namespace nvinfer1
{
class ILogger;
}

namespace tensorrt_llm::batch_manager
{

class InferenceRequest;
class TrtGptModel;

/* Responsible for shepherding requests through to completion
   using TRT Backend. */
class GptManager
{
public:
    using SizeType = tensorrt_llm::runtime::SizeType;
    using RequestList = std::list<std::shared_ptr<LlmRequest>>;

    GptManager(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType, int32_t maxBeamWidth,
        batch_scheduler::SchedulerPolicy schedulerPolicy, GetInferenceRequestsCallback getInferenceRequestsCb,
        SendResponseCallback sendResponseCb, PollStopSignalCallback pollStopSignalCb = nullptr,
        ReturnBatchManagerStatsCallback returnBatchManagerStatsCb = nullptr,
        const TrtGptModelOptionalParams& optionalParams = TrtGptModelOptionalParams(),
        std::optional<uint64_t> terminateReqId = std::nullopt);

    /* Wraps the user-provided callback for requests.
       Adds requests to request table.
       Invoked every generation loop iteration. */
    BatchManagerErrorCode_t fetchNewRequests();

    /* Returns completed requests.
       Deletes entry from activeRequests */
    BatchManagerErrorCode_t returnCompletedRequests();

    BatchManagerErrorCode_t pollStopSignals();

    BatchManagerErrorCode_t returnBatchManagerStats();

    BatchManagerErrorCode_t waitUntilTerminate();

    virtual ~GptManager();

protected:
    /* Invokes one step of backend
       Updates state of all requests */
    virtual BatchManagerErrorCode_t step(RequestList& activeRequests, std::set<uint64_t>& activeRequestsIds);

private:
    void validateLlmRequest(LlmRequest& newReq) const;
    static std::shared_ptr<LlmRequest> fillLlmRequest(std::shared_ptr<InferenceRequest> newReq);
    static std::shared_ptr<std::vector<int32_t>> getReqInputTokens(std::shared_ptr<InferenceRequest> new_req);
    static int32_t getMaxNewTokens(std::shared_ptr<InferenceRequest> newReq);

    std::shared_ptr<TrtGptModel> mTrtGptModel;
    SizeType mMaxInputLen;
    SizeType mMaxOutputLen;
    SizeType mMaxNumSequences;
    std::optional<uint64_t> mTerminateReqId;

    // Iteration counter - incremented every iteration of the generation loop
    int64_t mIterationCounter;
    // List of live requests
    RequestList mActiveRequests;
    // IDs of live requests
    std::set<uint64_t> mActiveRequestsIds;

    GetInferenceRequestsCallback mGetInferenceRequestsCb;
    SendResponseCallback mSendResponseCb;
    PollStopSignalCallback mPollStopSignalCb;
    ReturnBatchManagerStatsCallback mReturnBatchManagerStatsCb;

    std::atomic<bool> destructor_called_;
    void decoupled_execution_loop();
    std::shared_ptr<std::thread> worker_thread_;
    inline static const std::string kInputIdsTensorName_ = "input_ids";
    inline static const std::string kMaxNewTokensTensorName_ = "request_output_len";
    inline static const std::string kBeamWidthTensorName_ = "beam_width";
    inline static const std::string kEndIdTensorName_ = "end_id";
    inline static const std::string kPadIdTensorName_ = "pad_id";
    inline static const std::string kTemperatureTensorName_ = "temperature";
    inline static const std::string kRuntimeTopKTensorName_ = "runtime_top_k";
    inline static const std::string kRuntimeTopPTensorName_ = "runtime_top_p";
    inline static const std::string kLengthPenaltyTensorName_ = "len_penalty";
    inline static const std::string kRepetitionPenaltyTensorName_ = "repetition_penalty";
    inline static const std::string kMinLengthTensorName_ = "min_length";
    inline static const std::string kPresencePenaltyTensorName_ = "presence_penalty";
    inline static const std::string kRandomSeedTensorName_ = "random_seed";
    inline static const std::string kOutputIdsTensorName_ = "output_ids";

    std::shared_ptr<nvinfer1::ILogger> mLogger{};
};

} // namespace tensorrt_llm::batch_manager
