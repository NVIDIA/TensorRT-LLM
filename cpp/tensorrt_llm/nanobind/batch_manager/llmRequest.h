/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/llmRequest.h"

#include <ATen/ATen.h>
#include <ATen/ops/tensor.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <optional>

namespace nb = nanobind;

namespace tensorrt_llm::nanobind::batch_manager
{

namespace tb = tensorrt_llm::batch_manager;

/* Unfortunately, torch's default nanobind bindings don't know about c10::cuda::CUDAStream,
 * so we have to pass the more generic c10::Stream, and convert it back to a full-fledged
 * torch.cuda.Stream in python. See example in test/bindings/test_gpt_manager.py
 */
class LlmRequest : public tb::GenericLlmRequest<at::Tensor, c10::Stream>
{
public:
    using Base = GenericLlmRequest<at::Tensor, c10::Stream>;
    using TensorPtr = Base::TensorPtr;
    using SizeType32 = Base::SizeType32;
    using TokenIdType = Base::TokenIdType;
    using RequestIdType = Base::RequestIdType;
    using LoraTaskIdType = Base::LoraTaskIdType;
    using VecLogProbs = Base::VecLogProbs;
    using BeamTokens = Base::BeamTokens;
    using VecTokens = Base::VecTokens;
    using VecTokenExtraIds = Base::VecTokenExtraIds;
    using LogitsPostProcessor = Base::LogitsPostProcessor;

    LlmRequest(RequestIdType requestId, SizeType32 maxNewTokens, std::vector<TokenIdType> inputTokens,
        runtime::SamplingConfig samplingConfig, bool isStreaming, std::optional<SizeType32> endId = std::nullopt,
        std::optional<SizeType32> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
        std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
        std::optional<std::vector<SizeType32>> positionIds = std::nullopt,
        std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
        std::optional<SizeType32> promptVocabSize = std::nullopt,
        std::optional<std::vector<std::vector<SizeType32>>> multimodalHashes = std::nullopt,
        std::optional<std::vector<SizeType32>> multimodalPositions = std::nullopt,
        std::optional<std::vector<SizeType32>> multimodalLengths = std::nullopt,
        std::optional<TensorPtr> multimodalEmbedding = std::nullopt,
        std::optional<TensorPtr> mropeRotaryCosSin = std::nullopt,
        std::optional<SizeType32> mropePositionDeltas = std::nullopt,
        std::optional<LoraTaskIdType> loraTaskId = std::nullopt, std::optional<TensorPtr> loraWeights = std::nullopt,
        std::optional<TensorPtr> loraConfig = std::nullopt,
        std::optional<executor::LookaheadDecodingConfig> lookaheadConfig = std::nullopt,
        std::optional<executor::KvCacheRetentionConfig> kvCacheRetentionConfig = std::nullopt,
        bool returnLogProbs = false, bool returnContextLogits = false, bool returnGenerationLogits = false,
        std::optional<VecTokens> draftTokens = std::nullopt, std::optional<TensorPtr> draftLogits = std::nullopt,
        bool excludeInputFromOutput = false, std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt,
        bool applyLogitsPostProcessorBatched = false, std::optional<VecTokens> encoderInputTokens = std::nullopt,
        bool returnEncoderOutput = false, std::optional<RequestIdType> clientId = std::nullopt,
        executor::PriorityType priority = executor::Request::kDefaultPriority,
        std::optional<TensorPtr> encoderInputFeatures = std::nullopt,
        std::optional<SizeType32> encoderOutputLength = std::nullopt,
        std::optional<TensorPtr> crossAttentionMask = std::nullopt,
        tb::LlmRequestType llmRequestType = tb::LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        std::optional<VecTokenExtraIds> inputTokenExtraIds = std::nullopt, SizeType32 numReturnSequences = 1,
        std::optional<executor::EagleConfig> eagleConfig = std::nullopt,
        std::optional<TensorPtr> skipCrossAttnBlocks = std::nullopt, bool returnPerfMetrics = false,
        std::optional<executor::GuidedDecodingParams> guidedDecodingParams = std::nullopt,
        std::optional<SizeType32> languageAdapterUid = std::nullopt,
        std::optional<MillisecondsType> allottedTimeMs = std::nullopt,
        std::optional<executor::ContextPhaseParams> const& contextPhaseParams = std::nullopt,
        std::optional<CacheSaltIDType> cacheSaltID = std::nullopt,
        std::optional<TimePoint> arrivalTime = std::nullopt)
        : Base(requestId,                                                                                       //
            maxNewTokens,                                                                                       //
            std::make_shared<std::vector<TokenIdType>>(std::move(inputTokens)),                                 //
            samplingConfig,                                                                                     //
            isStreaming,                                                                                        //
            endId,                                                                                              //
            padId,                                                                                              //
            embeddingBias,                                                                                      //
            badWordsList,                                                                                       //
            stopWordsList,                                                                                      //
            positionIds.has_value() ? std::make_shared<std::vector<SizeType32>>(std::move(positionIds.value())) //
                                    : std::optional<std::shared_ptr<std::vector<SizeType32>>>(std::nullopt),    //
            promptEmbeddingTable,                                                                               //
            promptVocabSize,                                                                                    //
            multimodalHashes.has_value()
                ? std::make_optional(
                    std::make_shared<std::vector<std::vector<SizeType32>>>(std::move(multimodalHashes.value()))) //
                : std::optional<std::shared_ptr<std::vector<std::vector<SizeType32>>>>(std::nullopt),            //
            multimodalPositions.has_value()
                ? std::make_shared<std::vector<SizeType32>>(std::move(multimodalPositions.value()))              //
                : std::optional<std::shared_ptr<std::vector<SizeType32>>>(std::nullopt),                         //
            multimodalLengths.has_value()
                ? std::make_shared<std::vector<SizeType32>>(std::move(multimodalLengths.value()))                //
                : std::optional<std::shared_ptr<std::vector<SizeType32>>>(std::nullopt),                         //
            multimodalEmbedding,                                                                                 //
            mropeRotaryCosSin,                                                                                   //
            mropePositionDeltas,                                                                                 //
            loraTaskId,                                                                                          //
            loraWeights,                                                                                         //
            loraConfig,                                                                                          //
            lookaheadConfig,                                                                                     //
            kvCacheRetentionConfig,                                                                              //
            returnLogProbs,                                                                                      //
            returnContextLogits,                                                                                 //
            returnGenerationLogits,                                                                              //
            draftTokens.has_value() ? std::make_shared<VecTokens>(std::move(draftTokens.value()))                //
                                    : std::make_shared<VecTokens>(),                                             //
            draftLogits,                                                                                         //
            excludeInputFromOutput,                                                                              //
            logitsPostProcessor,                                                                                 //
            applyLogitsPostProcessorBatched,                                                                     //
            encoderInputTokens ? std::make_optional(std::make_shared<VecTokens>(std::move(*encoderInputTokens))) //
                               : std::optional<std::shared_ptr<VecTokens>>(std::nullopt),                        //
            returnEncoderOutput,                                                                                 //
            clientId,                                                                                            //
            priority,                                                                                            //
            encoderInputFeatures,                                                                                //
            encoderOutputLength,                                                                                 //
            crossAttentionMask,                                                                                  //
            llmRequestType,                                                                                      //
            inputTokenExtraIds                                                                                   //
                ? std::make_optional(std::make_shared<VecTokenExtraIds>(std::move(*inputTokenExtraIds)))         //
                : std::optional<std::shared_ptr<VecTokenExtraIds>>(std::nullopt),                                //
            numReturnSequences,                                                                                  //
            eagleConfig,                                                                                         //
            skipCrossAttnBlocks,                                                                                 //
            returnPerfMetrics,                                                                                   //
            guidedDecodingParams,                                                                                //
            languageAdapterUid,                                                                                  //
            allottedTimeMs,                                                                                      //
            contextPhaseParams,                                                                                  //
            cacheSaltID,                                                                                         //
            arrivalTime                                                                                          //
        )
    {
    }

    static std::optional<tb::LlmRequest::LogitsPostProcessor> callbackAdapter(
        std::optional<LlmRequest::LogitsPostProcessor> callback);

    [[nodiscard]] std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest> toTrtLlm() const;
};

} // namespace tensorrt_llm::nanobind::batch_manager
