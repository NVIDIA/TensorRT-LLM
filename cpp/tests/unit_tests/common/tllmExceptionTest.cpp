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

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"

#include <string>

using ::testing::HasSubstr;

TEST(TllmException, StackTrace)
{
    // clang-format off
    auto ex = NEW_TLLM_EXCEPTION("TestException %d", 1);
    std::string const what{ex.what()};
    EXPECT_THAT(what, HasSubstr(std::to_string(__LINE__ - 2)));
    // clang-format on
    EXPECT_THAT(what, HasSubstr("TestException 1"));
    EXPECT_THAT(what, HasSubstr(__FILE__));
#if !defined(_MSC_VER)
    EXPECT_THAT(what, ::testing::Not(HasSubstr("tensorrt_llm::common::TllmException::TllmException")));
    EXPECT_THAT(what, HasSubstr("unit_tests/common/tllmExceptionTest"));
    EXPECT_THAT(what, HasSubstr("main"));
#endif
}

TEST(TllmException, Logger)
{
    try
    {
        // clang-format off
        TLLM_THROW("TestException %d", 1);
    }
    catch (const std::exception& e)
    {
        testing::internal::CaptureStdout();
        TLLM_LOG_EXCEPTION(e);
        auto const out = testing::internal::GetCapturedStdout();
        EXPECT_THAT(out, HasSubstr(std::to_string(__LINE__ - 7)));
        // clang-format on
        EXPECT_THAT(out, HasSubstr("TestException 1"));
        EXPECT_THAT(out, HasSubstr(__FILE__));
#if !defined(_MSC_VER)
        EXPECT_THAT(out, ::testing::Not(HasSubstr("tensorrt_llm::common::TllmException::TllmException")));
        EXPECT_THAT(out, HasSubstr("unit_tests/common/tllmExceptionTest"));
        EXPECT_THAT(out, HasSubstr("main"));
#endif
    }
}
