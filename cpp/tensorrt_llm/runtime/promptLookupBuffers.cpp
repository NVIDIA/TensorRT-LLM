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

#include "tensorrt_llm/runtime/promptLookupBuffers.h"

namespace tensorrt_llm::runtime
{

PromptLookupBuffers::PromptLookupBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, BufferManager const& manager,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig, executor::DecodingConfig const& decodingConfig,
    TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(maxBeamWidth == 1, "Beam search is not supported in Prompt-Lookup speculative decoding.");

    mPool.clear();
    mStartIndices.clear();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void PromptLookupBuffers::printPool() const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_TRACE("PLD Pool size=%zu", mPool.size());

    using std::string;

    for (auto const& [id, promptMap] : mPool)
    {
        TLLM_LOG_TRACE("Request %zu, size=%zu", id, promptMap.size());
        for (auto const& [key, value] : promptMap)
        {
            std::string info{"    "};
            info += common::vec2str(key) + "->";
            for (long unsigned int j = 0; j < value.size(); ++j)
            {
                info += common::vec2str(value[j]) + ",";
            }
            TLLM_LOG_TRACE(info.c_str());
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void PromptLookupBuffers::updateDraftTokens(std::shared_ptr<LlmRequest> request)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const optPromptLookupConfig = request->getPromptLookupConfig();
    TLLM_CHECK_WITH_INFO(optPromptLookupConfig.has_value(), "Config of Prompt-Lookup must has value.");
    auto const promptLookupConfig = optPromptLookupConfig.value();
    SizeType32 const maxSeqLens = request->getOrigPromptLen() + request->mMaxNewTokens;
    SizeType32 const promptLookupNumTokens = promptLookupConfig.getPromptLookupNumTokens();
    SizeType32 const maxMatchingNgramSize = promptLookupConfig.getMaxMatchingNgramSize();
    SizeType32 const candidateSetSize = promptLookupConfig.getCandidateSetSize();

    auto const& prompt = request->getTokens(0);
    int const promptLen = (int) prompt.size(); // `request->mPromptLen` includes draft token
    auto const& id = request->mRequestId;
    auto const& startIndex = mStartIndices[id];
    auto& map = mPool[id];
    std::vector<TokenIdType> chosenIds{};
    auto const draftTokens = request->getDraftTokens();

    if (promptLen < maxSeqLens - 1)
    {
        // Update pool
        auto const& sequence = prompt.begin() + startIndex;
        auto const sequenceLen = promptLen - startIndex;
        for (int size = std::min(maxMatchingNgramSize, (int) promptLen - 1); size > 0; --size)
        {
            for (int l = 0; l < (int) sequenceLen - size; ++l)
            {
                int const r = std::min(l + size + promptLookupNumTokens, (int) sequenceLen);
                auto const key = std::vector<TokenIdType>(sequence + l, sequence + l + size);
                auto const value = std::vector<TokenIdType>(sequence + l + size, sequence + r);

                // TODO: need performance comparison to decide whether to keep this constrain
                if (static_cast<SizeType32>(value.size()) < promptLookupNumTokens)
                {
                    continue;
                }

                if (map.find(key) == map.end() || candidateSetSize == 1
                    || static_cast<SizeType32>(map[key][0].size()) < promptLookupNumTokens)
                {
                    // Replace the value if
                    // 1. the key does not exist
                    // 2. we only keep the newest one value for each key (MRU)
                    // 3. the length of the value saved before is less than `prompt_lookup_num_tokens`
                    map[key] = std::vector<std::vector<TokenIdType>>{value};
                }
                else if (static_cast<SizeType32>(map[key].size()) < candidateSetSize)
                {
                    map[key].push_back(value);
                }
            }
        }

        // Find match
        for (int size = std::min(maxMatchingNgramSize, (int) promptLen - 1); size > 0; --size)
        {
            auto const pattern = std::vector<TokenIdType>(prompt.end() - size, prompt.end());
            if (map.find(pattern) == map.end())
            {
                continue;
            }
            chosenIds = map[pattern][0]; // Always choose the first match (aligned with HF)
            break;
        }
        mStartIndices[id] = std::max(0, (int) promptLen - (promptLookupNumTokens + maxMatchingNgramSize - 1));
    }

    request->setDraftTokens(std::make_shared<std::vector<TokenIdType>>(chosenIds));

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void PromptLookupBuffers::removePool(RequestIdType requestId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mPool.find(requestId) != mPool.end())
    {
        mPool.erase(requestId);
        mStartIndices.erase(requestId);
    }
    else
    {
        TLLM_LOG_DEBUG("requestId (%ld) not in Prompt-Lookup pool", requestId);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::runtime
