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

// Fails any KVCM2 test that leaves the poison latch set.
//
// A broken invariant is recorded rather than aborting, so nothing else in a gtest binary would
// notice one: the detecting code is usually a destructor, which is attached to no assertion. This
// listener is what makes such a failure visible, and it clears the latch afterwards so a single
// bad test does not condemn every test that follows it.
//
// Linked into each KVCM2 test target by CMakeLists.txt rather than included, so registration
// cannot be lost to an unused-include cleanup and does not depend on inline-variable semantics.

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/poison.h"

#include <gtest/gtest.h>

namespace
{

namespace kv = tensorrt_llm::batch_manager::kv_cache_manager_v2;

class PoisonCheckListener : public ::testing::EmptyTestEventListener
{
public:
    void OnTestStart(::testing::TestInfo const& /*testInfo*/) override
    {
        // Only reachable if the previous test could not clear the latch, which means a manager
        // outlived it. Say so here rather than blaming this test for the earlier one's damage.
        if (auto const reason = kv::Poison::reason())
        {
            ADD_FAILURE() << "KVCM2 was already poisoned before this test started, by an earlier "
                             "test whose manager is still alive: "
                          << *reason;
        }
    }

    void OnTestEnd(::testing::TestInfo const& /*testInfo*/) override
    {
        auto const reason = kv::takePoison();
        if (!reason)
        {
            return;
        }
        // Still set means takePoison() refused to clear, which it only does while a manager is
        // alive -- so every test after this one will trip the OnTestStart check too.
        bool const stillPoisoned = kv::Poison::reason().has_value();
        ADD_FAILURE() << "KVCM2 was poisoned during this test: " << *reason
                      << (stillPoisoned ? " A manager is still alive, so the latch could not be"
                                          " cleared and the tests after this one will also fail."
                                        : "");
    }
};

// Runs before gtest_main enters the suite. This translation unit is linked straight into the test
// executable, so its initializer always runs.
struct PoisonCheckRegistrar
{
    PoisonCheckRegistrar()
    {
        ::testing::UnitTest::GetInstance()->listeners().Append(new PoisonCheckListener());
    }
};

PoisonCheckRegistrar const kPoisonCheckRegistrar;

} // namespace
