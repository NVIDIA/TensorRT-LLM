/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/layers/banWordsLayer.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingLayer.h"
#include "tensorrt_llm/layers/penaltyLayer.h"
#include "tensorrt_llm/layers/stopCriteriaLayer.h"
#include <memory>
#include <vector>

namespace tensorrt_llm::layers
{
enum DecodingLayers_t
{
    PENALTY_LAYER,
    BAN_WORDS_LAYER,
    DECODING_LAYER,
    STOP_CRITERIA_LAYER
};

static std::vector<DecodingLayers_t> createDecodingLayerTypes(executor::DecodingMode const& mode)
{
    std::vector<DecodingLayers_t> types = {};
    if (mode.isUsePenalty())
    {
        types.push_back(DecodingLayers_t::PENALTY_LAYER);
    }
    if (mode.isUseBanWords())
    {
        types.push_back(DecodingLayers_t::BAN_WORDS_LAYER);
    }
    types.push_back(DecodingLayers_t::DECODING_LAYER);
    if (mode.isUseStopCriteria())
    {
        types.push_back(DecodingLayers_t::STOP_CRITERIA_LAYER);
    }
    return types;
}

template <typename T>
static std::vector<std::unique_ptr<BaseLayer>> createLayers(executor::DecodingMode const& mode,
    DecoderDomain const& decodingDomain, std::shared_ptr<runtime::BufferManager> const& bufferManager)
{
    std::vector<std::unique_ptr<BaseLayer>> layers;
    auto layerTypes = createDecodingLayerTypes(mode);
    // Only when draft tokens and predicted and decoded by the engine, we can skip penalty layer.
    if (!mode.isExplicitDraftTokens() && !mode.isEagle())
    {
        TLLM_CHECK_WITH_INFO(layerTypes.size() && layerTypes[0] == DecodingLayers_t::PENALTY_LAYER,
            "Penalty layer is required to be the first layer for any decoder configuration");
    }
    for (auto&& type : layerTypes)
    {
        std::unique_ptr<BaseLayer> layer;
        switch (type)
        {
        case DecodingLayers_t::PENALTY_LAYER:
            layer = std::make_unique<PenaltyLayer<T>>(mode, decodingDomain, bufferManager);
            break;

        case DecodingLayers_t::BAN_WORDS_LAYER:
            layer = std::make_unique<BanWordsLayer<T>>(mode, decodingDomain, bufferManager);
            break;

        case DecodingLayers_t::DECODING_LAYER:
            layer = std::make_unique<DecodingLayer<T>>(mode, decodingDomain, bufferManager);
            break;

        case DecodingLayers_t::STOP_CRITERIA_LAYER:
            layer = std::make_unique<StopCriteriaLayer<T>>(mode, decodingDomain, bufferManager);
            break;

        default: TLLM_CHECK_WITH_INFO(false, "Unknown DecodingLayers_t"); break;
        }
        layers.push_back(std::move(layer));
    }
    return layers;
}
} // namespace tensorrt_llm::layers
