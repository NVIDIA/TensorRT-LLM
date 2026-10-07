/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <ATen/core/dispatch/Dispatcher.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <span>

struct TorchRegistration
{
    char const* name;
    c10::DispatchKey dispatch;
    bool hasSchema;
};

inline void expectTorchRegistrations(std::span<TorchRegistration const> registrations)
{
    auto& dispatcher = c10::Dispatcher::singleton();
    for (auto const& registration : registrations)
    {
        auto const op = dispatcher.findOp({registration.name, ""});
        ASSERT_TRUE(op.has_value()) << registration.name;
        EXPECT_EQ(dispatcher.findSchema({registration.name, ""}).has_value(), registration.hasSchema)
            << registration.name;
        EXPECT_TRUE(op->hasKernelForDispatchKey(registration.dispatch)) << registration.name;
    }
}

inline void expectOnlyTorchFamily(std::span<TorchRegistration const> registrations)
{
    auto const names = c10::Dispatcher::singleton().getAllOpNames();
    auto const count = std::count_if(names.begin(), names.end(),
        [](auto const& name)
        { return (name.name.starts_with("trtllm::") || name.name.starts_with("tensorrt_llm::")); });
    EXPECT_EQ(count, registrations.size());
    for (auto const& name : names)
    {
        if ((name.name.starts_with("trtllm::") || name.name.starts_with("tensorrt_llm::")))
        {
            EXPECT_TRUE(std::any_of(registrations.begin(), registrations.end(),
                [&](auto const& registration) { return name.name == registration.name; }))
                << name.name;
        }
    }
}
