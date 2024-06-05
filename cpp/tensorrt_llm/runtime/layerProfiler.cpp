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

#include "tensorrt_llm/runtime/layerProfiler.h"
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

using namespace tensorrt_llm::runtime;

void LayerProfiler::reportLayerTime(char const* layerName, float timeMs) noexcept
{
    if (mIterator == mLayers.end())
    {
        bool const first = !mLayers.empty() && mLayers.begin()->name == layerName;
        mUpdatesCount += mLayers.empty() || first;
        if (first)
        {
            mIterator = mLayers.begin();
        }
        else
        {
            mLayers.emplace_back();
            mLayers.back().name = layerName;
            mIterator = mLayers.end() - 1;
        }
    }

    mIterator->timeMs.push_back(timeMs);
    ++mIterator;
}

float LayerProfiler::getTotalTime() const noexcept
{
    auto const plusLayerTime = [](float accumulator, LayerProfile const& lp)
    { return accumulator + std::accumulate(lp.timeMs.begin(), lp.timeMs.end(), 0.F, std::plus<float>()); };
    return std::accumulate(mLayers.begin(), mLayers.end(), 0.0F, plusLayerTime);
}

std::string LayerProfiler::getLayerProfile() noexcept
{
    std::string const nameHdr("   Layer");
    std::string const timeHdr("   Time(ms)");

    float const totalTimeMs = getTotalTime();

    auto const timeLength = timeHdr.size();

    std::unordered_map<std::string, float> layer2times;
    std::vector<std::string> layer_order;
    for (auto const& p : mLayers)
    {
        if (!layer2times.count(p.name))
        {
            layer2times[p.name] = 0;
            layer_order.push_back(p.name);
        }
        for (auto const& t : p.timeMs)
        {
            layer2times[p.name] += t;
        }
    }

    std::stringstream ss;
    ss << "\n=== Per-layer Profile ===\n" << timeHdr << nameHdr << "\n";

    for (auto const& name : layer_order)
    {
        if (layer2times[name] == 0.0f)
        {
            continue;
        }
        ss << std::setw(timeLength) << std::fixed << std::setprecision(2) << layer2times[name] << "   " << name << "\n";
    }

    ss << std::setw(timeLength) << std::fixed << std::setprecision(2) << totalTimeMs << "   Total\n";
    ss << "\n";

    // clear data
    mLayers.clear();

    return ss.str();
}
