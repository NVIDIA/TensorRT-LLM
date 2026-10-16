/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/logger.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

using namespace tensorrt_llm::common;

TEST(LoggerModuleTest, FormatModuleNoTrailingSpaces)
{
    for (auto const* raw : {"batch_manager", "common", "cutlass_extensions", "deep_ep", "deep_gemm", "executor",
             "flash_mla", "kernels", "layers", "nanobind", "runtime", "testing", "thop"})
    {
        auto const fmt = formatModule(raw);
        EXPECT_FALSE(fmt.empty());
        EXPECT_NE(fmt.back(), ' ') << "trailing space for \"" << raw << "\": \"" << fmt << "\"";
    }
}

TEST(LoggerModuleTest, ExtractModuleMatchesEnvKey)
{
    EXPECT_EQ(Logger::extractModule("/code/tensorrt_llm/runtime/foo.cpp"), "runtime");
    EXPECT_EQ(Logger::extractModule("/code/tensorrt_llm/batch_manager/s.cpp"), "batchmgr");
    EXPECT_EQ(Logger::extractModule("/code/tensorrt_llm/common/logger.h"), "common");
    EXPECT_EQ(Logger::extractModule("/code/tensorrt_llm/thop/alloc.cpp"), "thop");
    EXPECT_EQ(Logger::extractModule("/random/path/foo.cpp"), "others");
}

TEST(LoggerModuleTest, PerModuleFilterGainControl)
{
    // Set per-module override: runtime gets DEBUG while global stays WARNING.
    // Use a new thread so its thread_local Logger picks up the env var.
    setenv("TLLM_LOG_LEVEL_BY_MODULE", "debug:runtime", 1);

    std::thread t(
        []
        {
            auto* logger = Logger::getLogger();
            logger->setLevel(Logger::WARNING);

            constexpr auto runtimeMod = Logger::extractModule("/code/tensorrt_llm/runtime/foo.cpp");
            constexpr auto commonMod = Logger::extractModule("/code/tensorrt_llm/common/bar.cpp");

            // runtime has per-module DEBUG override — verbose messages enabled
            EXPECT_TRUE(logger->isEnabled(Logger::DEBUG, runtimeMod));
            // common has no override — falls back to global WARNING
            EXPECT_FALSE(logger->isEnabled(Logger::DEBUG, commonMod));
            // WARNING enabled for both
            EXPECT_TRUE(logger->isEnabled(Logger::WARNING, runtimeMod));
            EXPECT_TRUE(logger->isEnabled(Logger::WARNING, commonMod));
        });
    t.join();

    unsetenv("TLLM_LOG_LEVEL_BY_MODULE");
}

TEST(LoggerFormatTest, FormatsValuesAndLiteralBraces)
{
    std::thread t(
        []
        {
            auto* logger = Logger::getLogger();
            logger->setLevel(Logger::INFO);
            logger->setRank(-1);

            testing::internal::CaptureStdout();
            logger->log(Logger::INFO, Logger::Module{"runtime"}, 2,
                "name={}, count={}, hex={:x}, ratio={:.2f}%, shape={{{}, {}}}", "model", 4294967296ULL, 255, 12.345, 2,
                3);
            EXPECT_EQ(testing::internal::GetCapturedStdout(),
                "[TRT-LLM] [I] [runtime][RANK 2] name=model, count=4294967296, hex=ff, ratio=12.35%, shape={2, 3}\n");
        });
    t.join();
}

TEST(LoggerFormatTest, PreservesRuntimeMessageText)
{
    std::thread t(
        []
        {
            auto* logger = Logger::getLogger();
            logger->setLevel(Logger::INFO);
            logger->setRank(-1);
            std::string const message = "literal {braces}, %s, %d, 100%";

            testing::internal::CaptureStdout();
            logger->log(Logger::INFO, "{}", message);
            EXPECT_EQ(testing::internal::GetCapturedStdout(), "[TRT-LLM] [I] " + message + "\n");

            testing::internal::CaptureStderr();
            logger->log(std::runtime_error{message});
            EXPECT_EQ(testing::internal::GetCapturedStderr(), "[TRT-LLM] [E] std::runtime_error: " + message + "\n");
        });
    t.join();
}

TEST(LoggerFormatTest, DisabledLevelSkipsArgumentEvaluation)
{
    std::thread t(
        []
        {
            Logger::getLogger()->setLevel(Logger::WARNING);
            int evaluations = 0;
            auto const count = [&evaluations] { return ++evaluations; };

            TLLM_LOG_DEBUG("evaluated {}", count());
            EXPECT_EQ(evaluations, 0);

            testing::internal::CaptureStderr();
            TLLM_LOG_WARNING("evaluated {}", count());
            EXPECT_EQ(evaluations, 1);
            EXPECT_NE(testing::internal::GetCapturedStderr().find("evaluated 1"), std::string::npos);
        });
    t.join();
}
