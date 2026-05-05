/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <limits>

namespace
{
using SizeType32 = tensorrt_llm::executor::SizeType32;
using MultimodalItemRuns = tensorrt_llm::executor::MultimodalItemRuns;

struct OccupiedRun
{
    SizeType32 start;
    SizeType32 end;
    size_t itemIdx;
    size_t runIdx;
};

void validateMultimodalInputLayout(std::vector<std::vector<SizeType32>> const& multimodalHashes,
    MultimodalItemRuns const& multimodalItemRuns,
    std::optional<std::vector<std::optional<std::string>>> const& multimodalUuids, std::optional<SizeType32> promptLen)
{
    TLLM_CHECK_WITH_INFO(multimodalHashes.size() == multimodalItemRuns.size(),
        "multimodal_item_runs length (%zu) must match multimodal_hashes length (%zu)", multimodalItemRuns.size(),
        multimodalHashes.size());
    if (multimodalUuids && !multimodalUuids->empty())
    {
        TLLM_CHECK_WITH_INFO(multimodalUuids->size() == multimodalHashes.size(),
            "multimodal_uuids length (%zu) must match multimodal_hashes length (%zu)", multimodalUuids->size(),
            multimodalHashes.size());
    }
    if (promptLen)
    {
        TLLM_CHECK_WITH_INFO(*promptLen >= 0, "promptLen must be non-negative");
    }

    std::vector<OccupiedRun> occupiedRuns;
    for (size_t itemIdx = 0; itemIdx < multimodalHashes.size(); ++itemIdx)
    {
        TLLM_CHECK_WITH_INFO(multimodalHashes[itemIdx].size() == 8,
            "multimodal_hashes[%zu] must contain exactly 8 int32 values", itemIdx);

        auto const& itemRuns = multimodalItemRuns[itemIdx];
        TLLM_CHECK_WITH_INFO(!itemRuns.empty(), "multimodal_item_runs[%zu] must not be empty", itemIdx);

        std::optional<SizeType32> previousRunEnd = std::nullopt;
        for (size_t runIdx = 0; runIdx < itemRuns.size(); ++runIdx)
        {
            auto const& run = itemRuns[runIdx];
            TLLM_CHECK_WITH_INFO(run.promptStart >= 0,
                "multimodal_item_runs[%zu][%zu] must not contain negative positions", itemIdx, runIdx);
            TLLM_CHECK_WITH_INFO(
                run.runLength > 0, "multimodal_item_runs[%zu][%zu] must have positive length", itemIdx, runIdx);
            TLLM_CHECK_WITH_INFO(run.promptStart <= std::numeric_limits<SizeType32>::max() - run.runLength,
                "multimodal_item_runs[%zu][%zu] end position overflows SizeType32", itemIdx, runIdx);

            auto const runEnd = run.promptStart + run.runLength;
            if (promptLen)
            {
                TLLM_CHECK_WITH_INFO(runEnd <= *promptLen,
                    "multimodal_item_runs[%zu][%zu] ends at %d, exceeding input sequence length %d", itemIdx, runIdx,
                    runEnd, *promptLen);
            }
            if (previousRunEnd)
            {
                TLLM_CHECK_WITH_INFO(run.promptStart >= *previousRunEnd,
                    "multimodal_item_runs[%zu] must be ordered and non-overlapping", itemIdx);
            }

            std::optional<SizeType32> previousOffset = std::nullopt;
            for (auto const offset : run.nonEmbedOffsets)
            {
                TLLM_CHECK_WITH_INFO(offset >= 0 && offset < run.runLength,
                    "multimodal_item_runs[%zu][%zu] non-embed offsets must be within the run", itemIdx, runIdx);
                TLLM_CHECK_WITH_INFO(!previousOffset || offset > *previousOffset,
                    "multimodal_item_runs[%zu][%zu] non-embed offsets must be ordered and unique", itemIdx, runIdx);
                previousOffset = offset;
            }

            occupiedRuns.push_back(OccupiedRun{run.promptStart, runEnd, itemIdx, runIdx});
            previousRunEnd = runEnd;
        }
    }

    std::sort(occupiedRuns.begin(), occupiedRuns.end(),
        [](OccupiedRun const& lhs, OccupiedRun const& rhs) { return lhs.start < rhs.start; });
    for (size_t i = 1; i < occupiedRuns.size(); ++i)
    {
        auto const& previousRun = occupiedRuns[i - 1];
        auto const& currentRun = occupiedRuns[i];
        TLLM_CHECK_WITH_INFO(currentRun.start >= previousRun.end,
            "multimodal_item_runs must be globally non-overlapping but [%zu][%zu] overlaps [%zu][%zu]",
            currentRun.itemIdx, currentRun.runIdx, previousRun.itemIdx, previousRun.runIdx);
    }
}
} // namespace

namespace tensorrt_llm::executor
{
MultimodalInput::MultimodalInput(std::vector<std::vector<SizeType32>> multimodalHashes,
    MultimodalItemRuns multimodalItemRuns, std::optional<std::vector<std::optional<std::string>>> multimodalUuids,
    std::optional<SizeType32> promptLen)
    : mMultimodalHashes(std::move(multimodalHashes))
    , mMultimodalUuids(std::move(multimodalUuids))
    , mMultimodalItemRuns(std::move(multimodalItemRuns))
{
    validateMultimodalInputLayout(mMultimodalHashes, mMultimodalItemRuns, mMultimodalUuids, promptLen);
}

std::vector<std::vector<SizeType32>> MultimodalInput::getMultimodalHashes() const
{
    return mMultimodalHashes;
}

std::optional<std::vector<std::optional<std::string>>> const& MultimodalInput::getMultimodalUuids() const
{
    return mMultimodalUuids;
}

MultimodalItemRuns const& MultimodalInput::getMultimodalItemRuns() const
{
    return mMultimodalItemRuns;
}

} // namespace tensorrt_llm::executor
