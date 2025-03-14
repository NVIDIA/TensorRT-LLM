/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::executor
{
PromptTuningConfig::PromptTuningConfig(Tensor embeddingTable, std::optional<VecTokenExtraIds> inputTokenExtraIds)
    : mEmbeddingTable(std::move(embeddingTable))
    , mInputTokenExtraIds(std::move(inputTokenExtraIds))
{
    TLLM_CHECK_WITH_INFO(mEmbeddingTable.getShape().size() == 2,
        "Expected prompt embedding table to have shape [vocabSize, hiddenSize]");
}

Tensor PromptTuningConfig::getEmbeddingTable() const
{
    return mEmbeddingTable;
}

std::optional<VecTokenExtraIds> PromptTuningConfig::getInputTokenExtraIds() const
{
    return mInputTokenExtraIds;
}

} // namespace tensorrt_llm::executor
