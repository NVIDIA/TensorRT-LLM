/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/torchView.h"
#include <memory>

namespace tb = tensorrt_llm::batch_manager;
namespace tr = tensorrt_llm::runtime;

using namespace tensorrt_llm::pybind::batch_manager;

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

std::shared_ptr<tb::LlmRequest> LlmRequest::toTrtLlm() const
{
    auto embeddingBias = from_torch(mEmbeddingBias);
    auto badWordsList = from_torch(mBadWordsList);
    auto stopWordsList = from_torch(mStopWordsList);
    auto promptEmbeddingTable = from_torch(mPromptEmbeddingTable);
    auto draftLogits = from_torch(mDraftLogits);

    return std::make_shared<tb::LlmRequest>(mRequestId, mMaxNewTokens,
        std::make_shared<std::vector<TokenIdType>>(mTokens.at(0)), mSamplingConfig, mIsStreaming, mEndId, mPadId,
        embeddingBias, badWordsList, stopWordsList, promptEmbeddingTable, mPromptVocabSize, mReturnLogProbs,
        mDraftTokens, draftLogits);
}
