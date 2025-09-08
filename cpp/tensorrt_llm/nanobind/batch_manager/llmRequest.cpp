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
#include "llmRequest.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/nanobind/common/bindTypes.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <memory>

namespace tb = tensorrt_llm::batch_manager;
namespace tr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;

using namespace tensorrt_llm::nanobind::batch_manager;

using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
using RequestList = std::list<LlmRequestPtr>;

namespace
{

std::optional<tb::LlmRequest::TensorPtr> from_torch(std::optional<LlmRequest::TensorPtr> torchPtr)
{
    if (torchPtr)
    {
        return tr::TorchView::of(torchPtr.value());
    }
    return std::nullopt;
}

} // namespace

std::optional<tb::LlmRequest::LogitsPostProcessor> LlmRequest::callbackAdapter(
    std::optional<LlmRequest::LogitsPostProcessor> callback)
{
    if (!callback)
    {
        return std::nullopt;
    }

    return [callback](RequestIdType reqId, tr::ITensor::SharedPtr& tensor, tb::LlmRequest::BeamTokens const& tokens,
               tr::BufferManager::CudaStreamPtr stream, std::optional<RequestIdType> clientId)
    {
        at::Tensor atTensor = tr::Torch::tensor(tensor);
        callback.value()(reqId, atTensor, tokens, runtime::TorchUtils::stream(*stream).unwrap(), clientId);
    };
}

std::shared_ptr<tb::LlmRequest> LlmRequest::toTrtLlm() const
{

    auto const draftTokens = std::make_shared<std::vector<TokenIdType>>(*mDraftTokens.get());
    auto const optDraftTokens = std::optional<std::shared_ptr<std::vector<TokenIdType>>>(draftTokens);
    auto const encoderInputTokens = mEncoderTokens.has_value()
        ? std::make_shared<std::vector<TokenIdType>>(*mEncoderTokens.value().get())
        : nullptr;
    auto const optEncoderInputTokens = std::optional<std::shared_ptr<std::vector<TokenIdType>>>(encoderInputTokens);
    return std::make_shared<tb::LlmRequest>(                       //
        mRequestId,                                                //
        mMaxNewTokens,                                             //
        std::make_shared<std::vector<TokenIdType>>(mTokens.at(0)), //
        mSamplingConfig,                                           //
        mIsStreaming,                                              //
        mEndId,                                                    //
        mPadId,                                                    //
        from_torch(mEmbeddingBias),                                //
        from_torch(mBadWordsList),                                 //
        from_torch(mStopWordsList),                                //
        mPositionIds,                                              //
        from_torch(mPromptEmbeddingTable),                         //
        mPromptVocabSize,                                          //
        mMultimodalHashes,                                         //
        mMultimodalPositions,                                      //
        mMultimodalLengths,                                        //
        from_torch(mMultimodalEmbedding),                          //
        from_torch(mMropeRotaryCosSin),                            //
        mMropePositionDeltas,                                      //
        mLoraTaskId,                                               //
        from_torch(mLoraWeights),                                  //
        from_torch(mLoraConfig),                                   //
        mLookaheadConfig,                                          //
        mKvCacheRetentionConfig,                                   //
        mReturnLogProbs,                                           //
        mReturnContextLogits,                                      //
        mReturnGenerationLogits,                                   //
        optDraftTokens,                                            //
        from_torch(mDraftLogits),                                  //
        mExcludeInputFromOutput,                                   //
        callbackAdapter(mLogitsPostProcessor),                     //
        mApplyLogitsPostProcessorBatched,                          //
        optEncoderInputTokens,                                     //
        mReturnEncoderOutput,                                      //
        mClientId,                                                 //
        mPriority,                                                 //
        from_torch(mEncoderInputFeatures),                         //
        mEncoderOutputLength,                                      //
        from_torch(mCrossAttentionMask),                           //
        getLlmRequestType(),                                       //
        std::nullopt,                                              // inputTokenExtraIds
        mNumReturnSequences,                                       //
        mEagleConfig,                                              //
        from_torch(mSkipCrossAttnBlocks),                          //
        false,                                                     // returnPerfMetrics
        mGuidedDecodingParams,                                     //
        mLanguageAdapterUid,                                       //
        mAllottedTimeMs,                                           //
        mContextPhaseParams,                                       //
        mCacheSaltID,                                              //
        mPerfMetrics.timingMetrics.arrivalTime                     //
    );
}
