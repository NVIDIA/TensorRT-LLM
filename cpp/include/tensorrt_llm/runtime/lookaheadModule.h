/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/speculativeDecodingModule.h"

namespace tensorrt_llm::runtime
{

class LookaheadModule : public SpeculativeDecodingModule
{
public:
    explicit LookaheadModule(SizeType32 maxDraftPathLen, SizeType32 maxDecodingDraftTokens) noexcept
        : SpeculativeDecodingModule(maxDraftPathLen, maxDecodingDraftTokens, maxDecodingDraftTokens)
    {
    }

    explicit LookaheadModule() noexcept
        : LookaheadModule(0, 0)
    {
    }

    void setExecutionConfig(executor::LookaheadDecodingConfig const& config)
    {
        mExecutionConfig = config;
    }

    [[nodiscard]] executor::LookaheadDecodingConfig const& getExecutionConfig() const
    {
        return mExecutionConfig;
    }

private:
    executor::LookaheadDecodingConfig mExecutionConfig;
};

} // namespace tensorrt_llm::runtime
