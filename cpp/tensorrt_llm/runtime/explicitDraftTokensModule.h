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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/speculativeDecodingModule.h"

namespace tensorrt_llm::runtime
{

class ExplicitDraftTokensModule : public SpeculativeDecodingModule
{
public:
    explicit ExplicitDraftTokensModule(
        SizeType32 maxDraftPathLen, SizeType32 maxDecodingDraftTokens, SizeType32 maxNumPaths) noexcept
        : SpeculativeDecodingModule(maxDraftPathLen, maxDecodingDraftTokens, maxNumPaths)
    {
        TLLM_CHECK(maxNumPaths * maxDraftPathLen == maxDecodingDraftTokens);
    }

    explicit ExplicitDraftTokensModule() noexcept
        : ExplicitDraftTokensModule(0, 0, 0)
    {
    }
};
} // namespace tensorrt_llm::runtime
