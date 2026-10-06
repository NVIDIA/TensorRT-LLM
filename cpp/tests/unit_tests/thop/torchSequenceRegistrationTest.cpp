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

#include <ATen/core/dispatch/Dispatcher.h>
#include <gtest/gtest.h>

#include <algorithm>

TEST(TorchSequenceRegistration, SchemasAndCudaKernels)
{
    auto& dispatcher = c10::Dispatcher::singleton();
    for (auto const* name : {"trtllm::causal_conv1d_fwd", "trtllm::causal_conv1d_update", "trtllm::mamba_conv1d",
             "trtllm::selective_scan", "trtllm::mamba2_mtp_ssm_cache_update"})
    {
        auto const op = dispatcher.findSchema({name, ""});
        ASSERT_TRUE(op.has_value()) << name;
        EXPECT_TRUE(op->hasKernelForDispatchKey(c10::DispatchKey::CUDA)) << name;
    }
}

TEST(TorchSequenceRegistration, ExcludesUnrelatedOperators)
{
    auto& dispatcher = c10::Dispatcher::singleton();
    auto const names = dispatcher.getAllOpNames();
    auto const familyCount
        = std::count_if(names.begin(), names.end(), [](auto const& name) { return name.name.starts_with("trtllm::"); });
    constexpr int kExpectedCount = 5;
    EXPECT_EQ(familyCount, kExpectedCount);
    EXPECT_FALSE(dispatcher.findSchema({"trtllm::fused_dit_split_norm", ""}).has_value());
    EXPECT_FALSE(dispatcher.findSchema({"trtllm::allreduce", ""}).has_value());
}
