/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/poison.h"

#include <gtest/gtest.h>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2::test
{

// Fails any test that leaves the KVCM2 poison latch set.
//
// A broken invariant is recorded rather than aborting, so nothing else in a gtest binary would
// notice one: the detecting code is usually a destructor, which is attached to no assertion. This
// listener is what makes such a failure visible, and it clears the latch afterwards so a single
// bad test does not condemn every test that follows it.
//
// Including this header registers the listener; there is nothing to call.
class PoisonCheckListener : public ::testing::EmptyTestEventListener
{
public:
    void OnTestStart(::testing::TestInfo const& /*testInfo*/) override
    {
        // Only reachable if the previous test could not clear the latch, which means a manager
        // outlived it. Say so here rather than blaming this test for the earlier one's damage.
        if (auto const reason = Poison::reason())
        {
            ADD_FAILURE() << "KVCM2 was already poisoned before this test started, by an earlier "
                             "test whose manager is still alive: "
                          << *reason;
        }
    }

    void OnTestEnd(::testing::TestInfo const& /*testInfo*/) override
    {
        auto const reason = takePoison();
        if (!reason)
        {
            return;
        }
        // Still set means takePoison() refused to clear, which it only does while a manager is
        // alive -- so every test after this one will trip the OnTestStart check too.
        bool const stillPoisoned = Poison::reason().has_value();
        ADD_FAILURE() << "KVCM2 was poisoned during this test: " << *reason
                      << (stillPoisoned ? " A manager is still alive, so the latch could not be"
                                          " cleared and the tests after this one will also fail."
                                        : "");
    }
};

// Registered at static initialization, before gtest_main runs the suite. `inline` so a binary
// built from several translation units still appends exactly one listener.
inline bool const kPoisonCheckRegistered = []
{
    ::testing::UnitTest::GetInstance()->listeners().Append(new PoisonCheckListener());
    return true;
}();

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2::test
