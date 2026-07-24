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

#include "tensorrt_llm/runtime/common.h"
#include <vector>

#include <NvInfer.h>

namespace tensorrt_llm::runtime
{
struct LayerProfile
{
    std::string name;
    std::vector<float> timeMs;
};

class LayerProfiler : public nvinfer1::IProfiler
{

public:
    void reportLayerTime(char const* layerName, float timeMs) noexcept override;

    std::string getLayerProfile() noexcept;

private:
    [[nodiscard]] float getTotalTime() const noexcept;

    std::vector<LayerProfile> mLayers;
    std::vector<LayerProfile>::iterator mIterator{mLayers.begin()};
    int32_t mUpdatesCount{0};
};
} // namespace tensorrt_llm::runtime
