/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Invoke;

using namespace tensorrt_llm::executor;
using namespace tensorrt_llm::common;

TEST(ResponseTest, ctorValidInputs)
{
    {
        auto response = Response(1, "This is an error message");
    }
    {
        auto result = Result{true, {{1}}};
        auto response = Response(1, result);
    }
}

TEST(ResponseTest, ctorInvalidErrorMsg)
{
    try
    {
        auto response = Response(1, "");
        FAIL() << "Expected TllmException";
    }
    catch (TllmException& e)
    {
        EXPECT_THAT(e.what(), testing::HasSubstr("Error message should not be empty"));
    }
    catch (std::exception const& e)
    {
        FAIL() << "Expected TllmException";
    }
}

TEST(ResponseTest, responseWithResult)
{
    IdType requestId = 1;
    auto cumLogProbs = VecLogProbs{1.};
    auto logProbs = std::vector<VecLogProbs>{{1.}};
    auto finishReasons = std::vector<FinishReason>{FinishReason::kEND_ID};
    auto result = Result{
        true, {{1}}, cumLogProbs, logProbs, std::nullopt, std::nullopt, std::nullopt, std::nullopt, finishReasons};
    auto response = Response(requestId, result);
    EXPECT_EQ(response.getRequestId(), requestId);
    auto resp_result = response.getResult();
    EXPECT_EQ(resp_result.isFinal, result.isFinal);
    EXPECT_EQ(resp_result.outputTokenIds, result.outputTokenIds);
    EXPECT_EQ(resp_result.logProbs, result.logProbs);
    EXPECT_EQ(resp_result.cumLogProbs, result.cumLogProbs);
    EXPECT_EQ(resp_result.finishReasons, result.finishReasons);
    EXPECT_EQ(response.hasError(), false);

    try
    {
        auto err = response.getErrorMsg();
        FAIL() << "Expected TllmException";
    }
    catch (TllmException& e)
    {
        EXPECT_THAT(e.what(), testing::HasSubstr("Cannot get the error message"));
    }
}

TEST(ResponseTest, responseWithError)
{
    IdType requestId = 1;
    auto result = Result{true, {{1}}};
    std::string errMsg = "my_error_message";
    auto response = Response(requestId, errMsg);
    EXPECT_EQ(response.getRequestId(), requestId);
    EXPECT_EQ(response.hasError(), true);
    EXPECT_EQ(response.getErrorMsg(), errMsg);

    try
    {
        auto err = response.getResult();
        FAIL() << "Expected TllmException";
    }
    catch (TllmException& e)
    {
        EXPECT_THAT(e.what(), testing::HasSubstr("Cannot get the result"));
        EXPECT_THAT(e.what(), testing::HasSubstr(errMsg));
    }
}
