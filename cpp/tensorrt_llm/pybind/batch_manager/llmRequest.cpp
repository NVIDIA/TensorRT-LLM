/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/pybind/common/bindTypes.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <memory>

namespace tb = tensorrt_llm::batch_manager;
namespace tr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;

using namespace tensorrt_llm::pybind::batch_manager;

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

    return [callback](RequestIdType reqId, tensorrt_llm::runtime::ITensor::SharedPtr& tensor,
               tensorrt_llm::batch_manager::LlmRequest::BeamTokens const& tokens,
               tensorrt_llm::runtime::BufferManager::CudaStreamPtr stream, std::optional<RequestIdType> clientId)
    {
        at::Tensor atTensor = tr::Torch::tensor(tensor);
        callback.value()(reqId, atTensor, tokens, runtime::TorchUtils::stream(*stream).unwrap(), clientId);
    };
}

std::shared_ptr<tb::LlmRequest> LlmRequest::toTrtLlm() const
{
    auto embeddingBias = from_torch(mEmbeddingBias);
    auto badWordsList = from_torch(mBadWordsList);
    auto stopWordsList = from_torch(mStopWordsList);
    auto promptEmbeddingTable = from_torch(mPromptEmbeddingTable);
    auto mropeRotarySinCos = from_torch(mMropeRotarySinCos);

    auto loraWeights = from_torch(mLoraWeights);
    auto loraConfig = from_torch(mLoraConfig);
    auto draftLogits = from_torch(mDraftLogits);
    auto encoderInputFeatures = from_torch(mEncoderInputFeatures);
    auto crossAttentionMask = from_torch(mCrossAttentionMask);
    auto skipCrossAttnBlocks = from_torch(mSkipCrossAttnBlocks);

    return std::make_shared<tb::LlmRequest>(mRequestId, mMaxNewTokens,
        std::make_shared<std::vector<TokenIdType>>(mTokens.at(0)), mSamplingConfig, mIsStreaming, mEndId, mPadId,
        embeddingBias, badWordsList, stopWordsList, mPositionIds, promptEmbeddingTable, mPromptVocabSize,
        mropeRotarySinCos, mMropePositionDeltas, mLoraTaskId, loraWeights, loraConfig, mLookaheadConfig,
        mKvCacheRetentionConfig, returnLogProbs(), mReturnContextLogits, mReturnGenerationLogits, mDraftTokens,
        draftLogits, mExcludeInputFromOutput, callbackAdapter(mLogitsPostProcessor), mApplyLogitsPostProcessorBatched,
        mEncoderTokens, mReturnEncoderOutput, mClientId, mPriority, encoderInputFeatures, mEncoderOutputLength,
        crossAttentionMask, tb::LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, mInputTokenExtraIds,
        mNumReturnSequences, std::nullopt, skipCrossAttnBlocks);
}
