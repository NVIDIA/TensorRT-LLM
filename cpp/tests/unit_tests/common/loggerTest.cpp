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
#include <thread>

using namespace tensorrt_llm::common;

TEST(LoggerModuleTest, FormatModuleNoTrailingSpaces)
{
    for (auto const* raw : {"batch_manager", "common", "cutlass_extensions", "deep_ep", "deep_gemm", "executor",
             "executor_worker", "flash_mla", "kernels", "layers", "nanobind", "plugins", "runtime", "testing", "thop"})
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
    EXPECT_EQ(Logger::extractModule("/random/path/foo.cpp"), "  others");
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
