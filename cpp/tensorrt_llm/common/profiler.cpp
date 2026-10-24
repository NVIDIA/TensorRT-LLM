/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/common/profiler.h"

#include <atomic>
#include <cstdlib>
#include <iterator>
#include <stdexcept>

TRTLLM_NAMESPACE_BEGIN

namespace common
{
namespace
{

constexpr std::string_view kRegionNames[]{"workflow", "mpi", "nccl", "nanobind", "kv", "moe"};
constexpr std::uint8_t kDefaultRegions{0x0FU};

std::atomic<int>& initializedRegions() noexcept
{
    static std::atomic<int> regions{-1};
    return regions;
}

std::string_view trim(std::string_view region)
{
    auto const first = region.find_first_not_of(" \t\r\n");
    return first == std::string_view::npos ? std::string_view{}
                                           : region.substr(first, region.find_last_not_of(" \t\r\n") - first + 1);
}

std::uint8_t parseRegions()
{
    char const* value = std::getenv("TLLM_PROFILE_CUSTOM_REGIONS");
    if (value == nullptr)
    {
        return kDefaultRegions;
    }
    std::string_view remaining{value};
    if (trim(remaining) == "none")
    {
        return 0;
    }
    std::uint8_t regions{0};
    while (true)
    {
        auto const separator = remaining.find(',');
        auto const token = trim(remaining.substr(0, separator));
        std::uint8_t mask{0};
        for (std::uint8_t index = 0; index < std::size(kRegionNames); ++index)
        {
            if (token == kRegionNames[index])
            {
                mask = static_cast<std::uint8_t>(1U << index);
                break;
            }
        }
        if (mask == 0 || (regions & mask) != 0)
        {
            throw std::invalid_argument("Invalid TLLM_PROFILE_CUSTOM_REGIONS region: " + std::string{token});
        }
        regions |= mask;
        if (separator == std::string_view::npos)
        {
            return regions;
        }
        remaining.remove_prefix(separator + 1);
    }
}

struct Selection
{
    std::uint8_t regions{0};
    std::string error;
};

} // namespace

void initializeProfiler()
{
    static Selection const selection = []
    {
        Selection result;
        try
        {
            result.regions = parseRegions();
        }
        catch (std::invalid_argument const& error)
        {
            // Cache invalid configuration too, so an environment mutation cannot retry parsing.
            result.error = error.what();
        }
        return result;
    }();
    if (!selection.error.empty())
    {
        throw std::invalid_argument(selection.error);
    }
    initializedRegions().store(selection.regions, std::memory_order_release);
}

bool customRegionEnabled(ProfilerRegion region) noexcept
{
    auto const index = static_cast<std::uint8_t>(region);
    auto const enabled = initializedRegions().load(std::memory_order_acquire);
    return enabled >= 0 && index < std::size(kRegionNames) && (enabled & (1U << index)) != 0;
}

ProfilerMetadata::ProfilerMetadata(std::string_view region)
    : mEncoded{"TRTLLM1"}
{
    mEncoded += std::to_string(region.size());
    mEncoded += ':';
    mEncoded += region;
    mEncoded += '{';
}

void ProfilerMetadata::add(std::string_view name, std::string_view value)
{
    appendName(name);
    appendQuoted(value);
}

void ProfilerMetadata::add(std::string_view name, std::int64_t value)
{
    appendName(name);
    mEncoded += std::to_string(value);
}

void ProfilerMetadata::add(std::string_view name, std::vector<std::int64_t> const& value)
{
    appendName(name);
    mEncoded += '[';
    bool first = true;
    for (auto const item : value)
    {
        if (!first)
        {
            mEncoded += ',';
        }
        first = false;
        mEncoded += std::to_string(item);
    }
    mEncoded += ']';
}

std::string ProfilerMetadata::take() &&
{
    mEncoded += '}';
    return std::move(mEncoded);
}

void ProfilerMetadata::appendName(std::string_view name)
{
    if (!mFirst)
    {
        mEncoded += ',';
    }
    mFirst = false;
    appendQuoted(name);
    mEncoded += ':';
}

void ProfilerMetadata::appendQuoted(std::string_view value)
{
    constexpr char kHexDigits[] = "0123456789abcdef";
    mEncoded += '\'';
    for (unsigned char const character : value)
    {
        switch (character)
        {
        case '\\': mEncoded += "\\\\"; break;
        case '\'': mEncoded += "\\\'"; break;
        case '\n': mEncoded += "\\n"; break;
        case '\r': mEncoded += "\\r"; break;
        case '\t': mEncoded += "\\t"; break;
        default:
            if (character < 0x20U)
            {
                mEncoded += "\\x";
                mEncoded += kHexDigits[character >> 4U];
                mEncoded += kHexDigits[character & 0x0FU];
            }
            else
            {
                mEncoded += static_cast<char>(character);
            }
        }
    }
    mEncoded += '\'';
}

RecordFunctionWithMetadata::RecordFunctionWithMetadata(ProfilerRegion region) noexcept
{
    if (!customRegionEnabled(region))
    {
        return;
    }
    mRegion = kRegionNames[static_cast<std::uint8_t>(region)].data();
    try
    {
        mRecord.emplace(at::RecordScope::USER_SCOPE);
        if (!mRecord->isActive())
        {
            mRecord.reset();
        }
    }
    catch (...)
    {
        // Constructing an observational probe must not prevent the operation from running.
        mRecord.reset();
    }
}

RecordFunctionWithMetadata::~RecordFunctionWithMetadata() noexcept
{
    try
    {
        if (mStarted)
        {
            mRecord->end();
        }
    }
    catch (...)
    {
        // Cleanup must not replace an exception from the instrumented operation.
    }
}

} // namespace common

TRTLLM_NAMESPACE_END
