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

namespace tensorrt_llm::executor
{
MultimodalInput::MultimodalInput(std::vector<std::vector<SizeType32>> multimodalHashes,
    std::vector<SizeType32> multimodalPositions, std::vector<SizeType32> multimodalLengths,
    std::optional<std::vector<std::optional<std::string>>> multimodalUuids,
    std::optional<std::vector<SizeType32>> multimodalItemRunCuOffsets,
    std::optional<std::vector<SizeType32>> multimodalRunPositions,
    std::optional<std::vector<SizeType32>> multimodalRunLengths)
    : mMultimodalHashes(std::move(multimodalHashes))
    , mMultimodalPositions(std::move(multimodalPositions))
    , mMultimodalLengths(std::move(multimodalLengths))
    , mMultimodalUuids(std::move(multimodalUuids))
    , mMultimodalItemRunCuOffsets(std::move(multimodalItemRunCuOffsets))
    , mMultimodalRunPositions(std::move(multimodalRunPositions))
    , mMultimodalRunLengths(std::move(multimodalRunLengths))
{
}

std::vector<std::vector<SizeType32>> MultimodalInput::getMultimodalHashes() const
{
    return mMultimodalHashes;
}

std::vector<SizeType32> MultimodalInput::getMultimodalPositions() const
{
    return mMultimodalPositions;
}

std::vector<SizeType32> MultimodalInput::getMultimodalLengths() const
{
    return mMultimodalLengths;
}

std::optional<std::vector<std::optional<std::string>>> const& MultimodalInput::getMultimodalUuids() const
{
    return mMultimodalUuids;
}

std::optional<std::vector<SizeType32>> const& MultimodalInput::getMultimodalItemRunCuOffsets() const
{
    return mMultimodalItemRunCuOffsets;
}

std::optional<std::vector<SizeType32>> const& MultimodalInput::getMultimodalRunPositions() const
{
    return mMultimodalRunPositions;
}

std::optional<std::vector<SizeType32>> const& MultimodalInput::getMultimodalRunLengths() const
{
    return mMultimodalRunLengths;
}

} // namespace tensorrt_llm::executor
