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

#pragma once

#include "tensorrt_llm/common/config.h"

#include <ATen/record_function.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

//! Categories shared by the Python and native process-start selectors.
enum class ProfilerRegion : std::uint8_t
{
    kWorkflow,
    kMpi,
    kNccl,
    kNanobind,
    kKv,
    kMoe,
};

//! Validate and cache TLLM_PROFILE_CUSTOM_REGIONS once, at application startup.
//! Throws std::invalid_argument for malformed configuration, including on repeated calls.
//! Call from an application exception boundary, before installing instrumented consumers;
//! never from a collective, profiler callback, ABI entry point, or static loader initializer.
//! Thread safe. Environment changes after initialization have no effect.
void initializeProfiler();

//! Return cached runtime selection, or false before successful explicit initialization.
//! Runtime selection alone does not enable instrumentation omitted at build time.
[[nodiscard]] bool customRegionEnabled(ProfilerRegion region) noexcept;

//! A TRTLLM1 literal dictionary supporting UTF-8 strings, int64 values and int64 vectors.
//! The builder owns its encoded bytes; take() transfers them to the caller.
class ProfilerMetadata
{
public:
    //! Start a payload with a UTF-8 byte-length-prefixed region name.
    explicit ProfilerMetadata(std::string_view region);

    //! Append a string field using Python literal escaping.
    void add(std::string_view name, std::string_view value);
    //! Append a signed int64 field.
    void add(std::string_view name, std::int64_t value);
    //! Append an int64 list, including an empty list.
    void add(std::string_view name, std::vector<std::int64_t> const& value);

    //! Complete the dictionary and transfer its storage; consume each builder once.
    [[nodiscard]] std::string take() &&;

private:
    void appendName(std::string_view name);
    void appendQuoted(std::string_view value);

    std::string mEncoded;
    bool mFirst{true};
};

//! An observational, same-thread PyTorch USER_SCOPE owning its name and string input.
//! Probes remain inactive until initializeProfiler() succeeds. Active callbacks determine
//! whether metadata is needed. Factories, profiler start and cleanup never change the body.
//! This owner cannot be copied or moved; its storage outlives the RecordFunction callbacks.
class RecordFunctionWithMetadata
{
public:
    //! Create an inactive or active probe without evaluating factories or parsing configuration.
    explicit RecordFunctionWithMetadata(ProfilerRegion region) noexcept;
    //! End the scope once on its creating thread, without replacing an operation exception.
    ~RecordFunctionWithMetadata() noexcept;

    RecordFunctionWithMetadata(RecordFunctionWithMetadata const&) = delete;
    RecordFunctionWithMetadata& operator=(RecordFunctionWithMetadata const&) = delete;
    RecordFunctionWithMetadata(RecordFunctionWithMetadata&&) = delete;
    RecordFunctionWithMetadata& operator=(RecordFunctionWithMetadata&&) = delete;

    //! Start at most once; metadataFactory receives the builder only if inputs are needed.
    template <typename MetadataFactory>
    void start(std::string name, MetadataFactory&& metadataFactory) noexcept
    {
        startWithLazyName(
            [&name]() -> std::string { return std::move(name); }, std::forward<MetadataFactory>(metadataFactory));
    }

    //! Evaluate nameFactory only for an active probe, and metadataFactory only for requested inputs.
    template <typename NameFactory, typename MetadataFactory>
    void startWithLazyName(NameFactory&& nameFactory, MetadataFactory&& metadataFactory) noexcept
    {
        if (!mRecord || mStarted)
        {
            return;
        }
        try
        {
            mName = std::forward<NameFactory>(nameFactory)();
            if (mRecord->needsInputs())
            {
                ProfilerMetadata metadata{mRegion};
                std::forward<MetadataFactory>(metadataFactory)(metadata);
                mInputs.emplace_back(std::move(metadata).take());
            }
            mRecord->before(mName, &mInputs);
            mStarted = true;
        }
        catch (...)
        {
            // Only observational setup is caught; the caller's operation is outside this boundary.
            mRecord.reset();
        }
    }

private:
    char const* mRegion{nullptr};
    std::string mName;
    std::vector<c10::IValue> mInputs;
    std::optional<at::RecordFunction> mRecord;
    bool mStarted{false};
};

} // namespace common

TRTLLM_NAMESPACE_END
