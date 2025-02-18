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

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/speculativeDecodingModule.h"

namespace tensorrt_llm::runtime
{

class EagleModule : public SpeculativeDecodingModule
{
public:
    // Number of paths is maxDecodingTokens = maxDecodingDraftTokens + 1 to account for very flat trees with
    // depth 1.
    explicit EagleModule(SizeType32 maxDraftPathLen, SizeType32 maxDecodingDraftTokens, SizeType32 numTransformersLayer,
        SizeType32 maxNonLeafNodesPerLayer) noexcept
        : SpeculativeDecodingModule(maxDraftPathLen, maxDecodingDraftTokens, maxDecodingDraftTokens + 1)
        , mNumTransformersLayer(numTransformersLayer)
        , mMaxNonLeafNodesPerLayer(maxNonLeafNodesPerLayer)
    {
    }

    explicit EagleModule() noexcept
        : EagleModule(0, 0, 0, 0)
    {
    }

    [[nodiscard]] executor::EagleChoices const& getDefaultEagleChoices() const noexcept
    {
        return mDefaultEagleChoices;
    }

    [[nodiscard]] SizeType32 getNumTransformerLayers() const noexcept
    {
        return mNumTransformersLayer;
    }

    [[nodiscard]] SizeType32 getMaxNonLeafNodesPerLayer() const noexcept
    {
        return mMaxNonLeafNodesPerLayer;
    }

private:
    SizeType32 mNumTransformersLayer;
    SizeType32 mMaxNonLeafNodesPerLayer;

    // We use mc_sim_7b_63 from official Medusa implementation, i.e. one of the best trees with 63 nodes found for 7B
    // Vicuna model. We use it as default, if no other are trees are specified per request or on the server level.
    executor::EagleChoices mDefaultEagleChoices = {{0}, {0, 0}, {1}, {0, 1}, {2}, {0, 0, 0}, {1, 0}, {0, 2}, {3},
        {0, 3}, {4}, {0, 4}, {2, 0}, {0, 5}, {0, 0, 1}, {5}, {0, 6}, {6}, {0, 7}, {0, 1, 0}, {1, 1}, {7}, {0, 8},
        {0, 0, 2}, {3, 0}, {0, 9}, {8}, {9}, {1, 0, 0}, {0, 2, 0}, {1, 2}, {0, 0, 3}, {4, 0}, {2, 1}, {0, 0, 4},
        {0, 0, 5}, {0, 0, 0, 0}, {0, 1, 1}, {0, 0, 6}, {0, 3, 0}, {5, 0}, {1, 3}, {0, 0, 7}, {0, 0, 8}, {0, 0, 9},
        {6, 0}, {0, 4, 0}, {1, 4}, {7, 0}, {0, 1, 2}, {2, 0, 0}, {3, 1}, {2, 2}, {8, 0}, {0, 5, 0}, {1, 5}, {1, 0, 1},
        {0, 2, 1}, {9, 0}, {0, 6, 0}, {0, 0, 0, 1}, {1, 6}, {0, 7, 0}};
};
} // namespace tensorrt_llm::runtime
