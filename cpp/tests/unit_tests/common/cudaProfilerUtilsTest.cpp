/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <gtest/gtest.h>
#include <stdlib.h>

#include "tensorrt_llm/common/cudaProfilerUtils.h"

#ifdef _WIN32
int setenv(char const* name, char const* value, int overwrite)
{
    int errcode = 0;
    if (!overwrite)
    {
        size_t envsize = 0;
        errcode = getenv_s(&envsize, NULL, 0, name);
        if (errcode || envsize)
            return errcode;
    }
    return _putenv_s(name, value);
}
#endif

std::string kEnvVarName{"TLLM_PROFILE_START_STOP"};
std::string kLegacyEnvVarName{"TLLM_GPTM_PROFILE_START_STOP"};

struct TestCase
{
    std::optional<std::string> legacyEnvVarVal;
    std::string envVarVal;
    std::pair<std::unordered_set<int32_t>, std::unordered_set<int32_t>> result;
};

TEST(CudaProfilerUtils, populateIterationIndexes)
{
    std::vector<TestCase> testCases;
    testCases.emplace_back(TestCase{std::nullopt, "", {{}, {}}});
    testCases.emplace_back(TestCase{std::nullopt, "1", {{1}, {1}}});
    testCases.emplace_back(TestCase{std::nullopt, "1,2,3", {{1, 2, 3}, {1, 2, 3}}});
    testCases.emplace_back(TestCase{std::nullopt, "1-4,7-8", {{1, 7}, {4, 8}}});
    testCases.emplace_back(TestCase{std::nullopt, "1,2,10-15", {{1, 2, 10}, {1, 2, 15}}});
    testCases.emplace_back(TestCase{std::nullopt, "1,,10-15", {{1, 10}, {1, 15}}});

    // Only legacy env var set
    testCases.emplace_back(TestCase{"1-4,7-8", "", {{1, 7}, {4, 8}}});

    // Both set, non-legacy has priority
    testCases.emplace_back(TestCase{"1-4,7-8", "2-10,88-99", {{2, 88}, {10, 99}}});

    for (auto const& testCase : testCases)
    {
        auto ret = setenv(kEnvVarName.c_str(), testCase.envVarVal.c_str(), 1); // does overwrite
        EXPECT_EQ(ret, 0);
        ret = setenv(
            kLegacyEnvVarName.c_str(), testCase.legacyEnvVarVal.value_or(std::string()).c_str(), 1); // does overwrite
        auto const [profileIterIdxs, stopIterIdxs]
            = tensorrt_llm::common::populateIterationIndexes(kEnvVarName, kLegacyEnvVarName);
        EXPECT_EQ(profileIterIdxs, testCase.result.first)
            << testCase.envVarVal << " " << testCase.legacyEnvVarVal.value_or("");
        EXPECT_EQ(stopIterIdxs, testCase.result.second)
            << testCase.envVarVal << " " << testCase.legacyEnvVarVal.value_or("");
    }
}
