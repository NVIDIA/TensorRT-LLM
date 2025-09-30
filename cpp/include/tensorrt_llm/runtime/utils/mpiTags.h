/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

namespace tensorrt_llm::mpi
{

enum class MpiTag : int
{
    kDefault = 0,

    // DecoderStepAsyncSend
    kDecoderStepNewOutputTokensHost = 0,
    kDecoderStepFinishedSumHost = 1,
    kDecoderStepSequenceLengthsHost = 2,
    kDecoderStepCumLogProbsHost = 3,
    kDecoderStepLogProbsHost = 4,
    kDecoderStepCacheIndirectionOutput = 5,
    kDecoderStepAcceptedLengthsCumSumDevice = 6,
    kDecoderStepAcceptedPackedPathsDevice = 7,
    kDecoderStepFinishReasonsHost = 8,

    // DecoderSlotAsyncSend
    kDecoderSlotOutputIds = 9,
    kDecoderSlotSequenceLengths = 10,
    kDecoderSlotCumLogProbs = 11,
    kDecoderSlotLogProbs = 12,

    // CancelledRequestsAsyncSend
    kCancelledRequestsNumReq = 13,
    kCancelledRequestsIds = 14,

    // RequestWithIdAsyncSend
    kRequestWithIdNumReq = 15,
    kRequestWithIdVecSize = 16,
    kRequestWithIdPacked = 17,

    // Executor
    kExecutorNumActiveRequests = 18,
    kExecutorLowestPriorityActiveHasValue = 19,
    kExecutorLowestPriorityActive = 20,
    kExecutorShouldExit = 21,

    // TrtGptModelInflightBatching
    kTrtGptModelInflightBatchingContextLogits = 22,
    kTrtGptModelInflightBatchingGenerationLogits = 23,

    // Orchestrator
    kOrchestratorId = 127,
    kOrchestratorData = 1023,
    kOrchestratorStatsId = 128,
    kOrchestratorStatsData = 1024,

    // LogitsThread
    kSpecDecLogitsId = 129,
    kSpecDecLogitsData = 1025,

    // KvCacheEventManager
    kKvCacheEventSize = 1026,
    kKvCacheEvent = 1027
};

} // namespace tensorrt_llm::mpi
