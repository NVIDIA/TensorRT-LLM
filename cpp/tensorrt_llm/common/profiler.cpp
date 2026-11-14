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
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <stdexcept>

TRTLLM_NAMESPACE_BEGIN

namespace common
{
namespace
{

constexpr std::string_view kRegionNames[]{"workflow", "mpi", "nccl", "nanobind", "kv", "moe"};
static_assert(std::size(kRegionNames) == static_cast<std::uint8_t>(ProfilerRegion::kCount));

constexpr std::uint8_t regionMask(ProfilerRegion region)
{
    return static_cast<std::uint8_t>(1U << static_cast<std::uint8_t>(region));
}

constexpr std::uint8_t kDefaultRegions = regionMask(ProfilerRegion::kWorkflow) | regionMask(ProfilerRegion::kMpi)
    | regionMask(ProfilerRegion::kNccl) | regionMask(ProfilerRegion::kNanobind);

std::atomic<int>& initializedRegions() noexcept
{
    static std::atomic<int> regions{-1};
    return regions;
}

//! Return a selector-token view without surrounding spaces, tabs, or line endings.
std::string_view trim(std::string_view region)
{
    auto const first = region.find_first_not_of(" \t\r\n");
    return first == std::string_view::npos ? std::string_view{}
                                           : region.substr(first, region.find_last_not_of(" \t\r\n") - first + 1);
}

//! Parse startup regions, rejecting unknown names, duplicate names, and invalid none mixtures.
std::uint8_t parseRegions()
{
    char const* value = std::getenv("TLLM_PROFILE_CUSTOM_REGIONS");
    if (value == nullptr)
    {
        return kDefaultRegions;
    }
    std::string_view remaining{value};
    if (trim(remaining).empty())
    {
        return kDefaultRegions;
    }
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

//! Cache startup selection once, retaining a diagnostic and default regions after malformed input.
std::optional<std::string> initializeProfilerRegions()
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
            result.regions = kDefaultRegions;
            result.error = std::string{error.what()} + "; using default regions";
        }
        return result;
    }();
    initializedRegions().store(selection.regions, std::memory_order_release);
    return selection.error.empty() ? std::nullopt : std::make_optional(selection.error);
}

//! Read a valid region's cached selection without parsing or consulting the environment.
bool customRegionEnabled(ProfilerRegion region) noexcept
{
    auto const index = static_cast<std::uint8_t>(region);
    auto const enabled = initializedRegions().load(std::memory_order_acquire);
    return enabled >= 0 && index < std::size(kRegionNames) && (enabled & (1U << index)) != 0;
}

//! Emit at most one process-wide observation warning without constructing the native Logger.
void reportProfilerFailure(char const* diagnostic) noexcept
{
    static std::atomic_flag warned = ATOMIC_FLAG_INIT;
    if (!warned.test_and_set(std::memory_order_relaxed))
    {
        std::fprintf(stderr, "TensorRT-LLM profiler warning: observation omitted: %s\n", diagnostic);
    }
}

ProfilerMetadata::ProfilerMetadata(std::string_view region)
    : mPrefix{"TRTLLM1:" + std::to_string(region.size()) + ':' + std::string{region}}
{
}

void ProfilerMetadata::add(std::string_view name, std::string_view value)
{
    mFields[std::string{name}] = std::string{value};
}

void ProfilerMetadata::add(std::string_view name, std::int64_t value)
{
    mFields[std::string{name}] = value;
}

//! Append a signed integer list, preserving order and allowing an empty vector.
void ProfilerMetadata::add(std::string_view name, std::vector<std::int64_t> const& value)
{
    mFields[std::string{name}] = value;
}

std::string ProfilerMetadata::take() &&
{
    return std::move(mPrefix) + mFields.dump();
}

//! Prepare a same-thread user scope only when the selected region has an active callback.
RecordFunctionWithMetadata::RecordFunctionWithMetadata(ProfilerRegion region) noexcept
{
    if (!customRegionEnabled(region) || !at::hasCallbacks())
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
    catch (std::exception const& error)
    {
        reportProfilerFailure(error.what());
        mRecord.reset();
    }
    catch (...)
    {
        reportProfilerFailure("non-standard exception constructing a scope");
        mRecord.reset();
    }
}

//! End a started scope once, reporting cleanup failures without replacing an operation exception.
RecordFunctionWithMetadata::~RecordFunctionWithMetadata() noexcept
{
    try
    {
        if (mStarted)
        {
            mRecord->end();
        }
    }
    catch (std::exception const& error)
    {
        reportProfilerFailure(error.what());
    }
    catch (...)
    {
        reportProfilerFailure("non-standard exception ending a scope");
    }
}

} // namespace common

TRTLLM_NAMESPACE_END
