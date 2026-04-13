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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"

#include <sstream>

namespace tkc = tensorrt_llm::executor::kv_cache;

// -------------------------------------------------------------------
//  Test 1: ErrorSignalInfo serialization round-trip
// -------------------------------------------------------------------

TEST(ErrorPropagation, ErrorSignalSerializationRoundTrip)
{
    tkc::ErrorSignalInfo original;
    original.mAgentName = "test-agent-001";
    original.mContext = tkc::DataContext{42};
    original.mRequestId = 12345;
    original.mErrorMessage = "Couldn't find the requested block in the reuse tree";

    // Serialize
    std::stringstream ss;
    tkc::ErrorSignalInfo::serialize(original, ss);

    // Deserialize
    auto deserialized = tkc::ErrorSignalInfo::deserialize(ss);

    EXPECT_EQ(deserialized.mAgentName, original.mAgentName);
    EXPECT_EQ(deserialized.mContext.getTag(), original.mContext.getTag());
    EXPECT_EQ(deserialized.mRequestId, original.mRequestId);
    EXPECT_EQ(deserialized.mErrorMessage, original.mErrorMessage);
}

// -------------------------------------------------------------------
//  Test 2: ErrorSignalInfo serializedSize matches actual output
// -------------------------------------------------------------------

TEST(ErrorPropagation, ErrorSignalSerializedSizeIsAccurate)
{
    tkc::ErrorSignalInfo info;
    info.mAgentName = "sender-agent";
    info.mContext = tkc::DataContext{99};
    info.mRequestId = 67890;
    info.mErrorMessage = "sendSync failed: block evicted";

    std::stringstream ss;
    tkc::ErrorSignalInfo::serialize(info, ss);
    auto actualSize = ss.str().size();
    auto reportedSize = tkc::ErrorSignalInfo::serializedSize(info);

    EXPECT_EQ(actualSize, reportedSize);
}

// -------------------------------------------------------------------
//  Test 3: NotificationInfo with ErrorSignalInfo variant round-trip
// -------------------------------------------------------------------

TEST(ErrorPropagation, NotificationInfoErrorSignalRoundTrip)
{
    tkc::ErrorSignalInfo errorInfo;
    errorInfo.mAgentName = "prefill-worker";
    errorInfo.mContext = tkc::DataContext{7};
    errorInfo.mRequestId = 99999;
    errorInfo.mErrorMessage = "Transfer failed: network error";

    tkc::NotificationInfo notification{errorInfo};

    // Verify the variant holds ErrorSignalInfo
    ASSERT_TRUE(std::holds_alternative<tkc::ErrorSignalInfo>(notification.mInfo));

    // Serialize the full NotificationInfo
    std::stringstream ss;
    tkc::NotificationInfo::serialize(notification, ss);

    // Deserialize
    auto deserialized = tkc::NotificationInfo::deserialize(ss);

    // Verify variant type is preserved
    ASSERT_TRUE(std::holds_alternative<tkc::ErrorSignalInfo>(deserialized.mInfo));

    auto const& deserializedError = std::get<tkc::ErrorSignalInfo>(deserialized.mInfo);
    EXPECT_EQ(deserializedError.mAgentName, errorInfo.mAgentName);
    EXPECT_EQ(deserializedError.mRequestId, errorInfo.mRequestId);
    EXPECT_EQ(deserializedError.mErrorMessage, errorInfo.mErrorMessage);
}

// -------------------------------------------------------------------
//  Test 4: NotificationInfo serializedSize with ErrorSignalInfo
// -------------------------------------------------------------------

TEST(ErrorPropagation, NotificationInfoErrorSignalSerializedSize)
{
    tkc::ErrorSignalInfo errorInfo;
    errorInfo.mAgentName = "agent-x";
    errorInfo.mContext = tkc::DataContext{1};
    errorInfo.mRequestId = 42;
    errorInfo.mErrorMessage = "test error";

    tkc::NotificationInfo notification{errorInfo};

    std::stringstream ss;
    tkc::NotificationInfo::serialize(notification, ss);
    auto actualSize = ss.str().size();
    auto reportedSize = tkc::NotificationInfo::serializedSize(notification);

    EXPECT_EQ(actualSize, reportedSize);
}

// -------------------------------------------------------------------
//  Test 5: Existing ReadySignalInfo still works (regression)
// -------------------------------------------------------------------

TEST(ErrorPropagation, ReadySignalInfoStillWorks)
{
    tkc::ReadySignalInfo readyInfo;
    readyInfo.mAgentName = "test-agent";
    readyInfo.mContext = tkc::DataContext{10};
    readyInfo.mIsReady = true;

    tkc::NotificationInfo notification{readyInfo};

    std::stringstream ss;
    tkc::NotificationInfo::serialize(notification, ss);
    auto deserialized = tkc::NotificationInfo::deserialize(ss);

    ASSERT_TRUE(std::holds_alternative<tkc::ReadySignalInfo>(deserialized.mInfo));
    auto const& deserializedReady = std::get<tkc::ReadySignalInfo>(deserialized.mInfo);
    EXPECT_EQ(deserializedReady.mAgentName, readyInfo.mAgentName);
    EXPECT_EQ(deserializedReady.mIsReady, true);
}

// -------------------------------------------------------------------
//  Test 6: Empty error message serialization
// -------------------------------------------------------------------

TEST(ErrorPropagation, EmptyErrorMessageRoundTrip)
{
    tkc::ErrorSignalInfo info;
    info.mAgentName = "";
    info.mContext = tkc::DataContext{0};
    info.mRequestId = 0;
    info.mErrorMessage = "";

    std::stringstream ss;
    tkc::ErrorSignalInfo::serialize(info, ss);
    auto deserialized = tkc::ErrorSignalInfo::deserialize(ss);

    EXPECT_EQ(deserialized.mAgentName, "");
    EXPECT_EQ(deserialized.mRequestId, 0u);
    EXPECT_EQ(deserialized.mErrorMessage, "");
}
