/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"

#include <gtest/gtest.h>

#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using ::tensorrt_llm::executor::kv_cache::genUniqueAgentName;

namespace
{

// Helper: split an agent name into its underscore-delimited components.
// Expected format: {hostname}_{pid}_{randomSuffix}_{counter}
std::vector<std::string> splitAgentName(std::string const& name)
{
    std::vector<std::string> parts;
    std::istringstream stream(name);
    std::string token;
    while (std::getline(stream, token, '_'))
    {
        parts.push_back(token);
    }
    return parts;
}

TEST(GenUniqueAgentNameTest, ReturnsNonEmptyString)
{
    std::string name = genUniqueAgentName();
    EXPECT_FALSE(name.empty());
}

TEST(GenUniqueAgentNameTest, HasFourComponents)
{
    std::string name = genUniqueAgentName();
    std::vector<std::string> parts = splitAgentName(name);
    // hostname_pid_randomSuffix_counter
    ASSERT_EQ(parts.size(), 4) << "Expected 4 underscore-delimited components, got: " << name;
    EXPECT_FALSE(parts[0].empty()) << "hostname should be non-empty";
    EXPECT_FALSE(parts[1].empty()) << "pid should be non-empty";
    EXPECT_FALSE(parts[2].empty()) << "randomSuffix should be non-empty";
    EXPECT_FALSE(parts[3].empty()) << "counter should be non-empty";
}

TEST(GenUniqueAgentNameTest, RandomSuffixIsStableWithinProcess)
{
    std::string name1 = genUniqueAgentName();
    std::string name2 = genUniqueAgentName();
    std::string name3 = genUniqueAgentName();

    std::string suffix1 = splitAgentName(name1)[2];
    std::string suffix2 = splitAgentName(name2)[2];
    std::string suffix3 = splitAgentName(name3)[2];

    EXPECT_EQ(suffix1, suffix2);
    EXPECT_EQ(suffix2, suffix3);
}

TEST(GenUniqueAgentNameTest, CounterIncrements)
{
    std::string name1 = genUniqueAgentName();
    std::string name2 = genUniqueAgentName();

    uint64_t counter1 = std::stoull(splitAgentName(name1)[3]);
    uint64_t counter2 = std::stoull(splitAgentName(name2)[3]);

    EXPECT_EQ(counter2, counter1 + 1);
}

TEST(GenUniqueAgentNameTest, AllNamesAreUnique)
{
    constexpr int kNumCalls = 1000;
    std::set<std::string> names;
    for (int i = 0; i < kNumCalls; ++i)
    {
        names.insert(genUniqueAgentName());
    }
    EXPECT_EQ(names.size(), kNumCalls);
}

TEST(GenUniqueAgentNameTest, ThreadSafety)
{
    constexpr int kNumThreads = 8;
    constexpr int kCallsPerThread = 500;

    std::mutex mtx;
    std::set<std::string> allNames;

    std::vector<std::thread> threads;
    for (int t = 0; t < kNumThreads; ++t)
    {
        threads.emplace_back(
            [&]()
            {
                std::vector<std::string> localNames;
                localNames.reserve(kCallsPerThread);
                for (int i = 0; i < kCallsPerThread; ++i)
                {
                    localNames.push_back(genUniqueAgentName());
                }

                std::lock_guard<std::mutex> lock(mtx);
                for (std::string& n : localNames)
                {
                    allNames.insert(std::move(n));
                }
            });
    }

    for (std::thread& t : threads)
    {
        t.join();
    }

    EXPECT_EQ(allNames.size(), kNumThreads * kCallsPerThread)
        << "All names from concurrent threads must be unique";
}

TEST(GenUniqueAgentNameTest, HostnameAndPidAreConsistent)
{
    std::string name1 = genUniqueAgentName();
    std::string name2 = genUniqueAgentName();

    std::vector<std::string> parts1 = splitAgentName(name1);
    std::vector<std::string> parts2 = splitAgentName(name2);

    EXPECT_EQ(parts1[0], parts2[0]) << "hostname should be the same across calls";
    EXPECT_EQ(parts1[1], parts2[1]) << "pid should be the same across calls";
}

} // namespace
