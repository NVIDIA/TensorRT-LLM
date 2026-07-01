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

#include "tensorrt_llm/common/envUtils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace
{

template <typename TestBody>
void runInFork(TestBody&& testBody)
{
    pid_t const pid = fork();
    ASSERT_NE(pid, -1) << "fork failed";
    if (pid == 0)
    {
        testBody();
        std::exit(testing::Test::HasFailure() ? 1 : 0);
    }

    int status = 0;
    ASSERT_NE(waitpid(pid, &status, 0), -1) << "waitpid failed";
    ASSERT_TRUE(WIFEXITED(status)) << "child process terminated abnormally";
    EXPECT_EQ(WEXITSTATUS(status), 0) << "expectation failed in child process";
}

void setChunkSizeEnv(std::optional<std::string> const& value)
{
    constexpr char const* envName = "TRTLLM_KVCACHE_TRANSFER_CHUNK_SIZE_BLOCKS";
    if (value.has_value())
    {
        ASSERT_EQ(setenv(envName, value->c_str(), 1), 0);
    }
    else
    {
        ASSERT_EQ(unsetenv(envName), 0);
    }
}

} // namespace

TEST(EnvUtilsTest, KVCacheTransferChunkSizeBlocksUnset)
{
    runInFork(
        []
        {
            setChunkSizeEnv(std::nullopt);
            EXPECT_EQ(tensorrt_llm::common::getEnvKVCacheTransferChunkSizeBlocks(), std::nullopt);
        });
}

TEST(EnvUtilsTest, KVCacheTransferChunkSizeBlocksZero)
{
    runInFork(
        []
        {
            setChunkSizeEnv(std::string{"0"});
            EXPECT_EQ(tensorrt_llm::common::getEnvKVCacheTransferChunkSizeBlocks(), std::nullopt);
        });
}

TEST(EnvUtilsTest, KVCacheTransferChunkSizeBlocksPositive)
{
    runInFork(
        []
        {
            setChunkSizeEnv(std::string{"17"});
            auto const chunkSize = tensorrt_llm::common::getEnvKVCacheTransferChunkSizeBlocks();
            ASSERT_TRUE(chunkSize.has_value());
            EXPECT_EQ(chunkSize.value(), 17);
        });
}

TEST(EnvUtilsTest, KVCacheTransferChunkSizeBlocksInvalid)
{
    for (auto const& value : {std::string{""}, std::string{"17blocks"}, std::string{"+1"}, std::string{"1_0"},
             std::string{"1 "}, std::string{" 1"}})
    {
        runInFork(
            [&value]
            {
                setChunkSizeEnv(value);
                EXPECT_THROW(tensorrt_llm::common::getEnvKVCacheTransferChunkSizeBlocks(), std::exception);
            });
    }
}

TEST(EnvUtilsTest, KVCacheTransferChunkSizeBlocksNegative)
{
    runInFork(
        []
        {
            setChunkSizeEnv(std::string{"-1"});
            EXPECT_THROW(tensorrt_llm::common::getEnvKVCacheTransferChunkSizeBlocks(), std::exception);
        });
}

TEST(EnvUtilsTest, KVCacheTransferChunkSizeBlocksOverflow)
{
    runInFork(
        []
        {
            setChunkSizeEnv(std::to_string(static_cast<long long>(std::numeric_limits<int32_t>::max()) + 1));
            EXPECT_THROW(tensorrt_llm::common::getEnvKVCacheTransferChunkSizeBlocks(), std::exception);
        });
}

TEST(EnvUtilsTest, KVCacheTransferEarlyRelease)
{
    runInFork(
        []
        {
            ASSERT_EQ(setenv("TRTLLM_KVCACHE_TRANSFER_EARLY_RELEASE", "1", 1), 0);
            EXPECT_TRUE(tensorrt_llm::common::getEnvKVCacheTransferEarlyRelease());
        });
}
