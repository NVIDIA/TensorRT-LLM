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

#include "tensorrt_llm/common/timestampUtils.h"

#include <chrono>
#include <sstream>
#include <thread>

using namespace tensorrt_llm::common;

TEST(TimestampUtils, getCurrentTimestamp)
{
    int32_t sleepMs = 100;
    int32_t sleepUs = sleepMs * 1000;
    ;
    int32_t tolUs = 5000;
    auto timestamp = getCurrentTimestamp();
    std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    auto timestamp2 = getCurrentTimestamp();
    auto microseconds = std::stoi(timestamp.erase(0, timestamp.find('.') + 1));
    std::cout << microseconds << std::endl;
    auto microseconds2 = std::stoi(timestamp2.erase(0, timestamp2.find('.') + 1));

    int32_t delta = (microseconds2 - microseconds);
    if (delta < 0)
    {
        delta += 1000000;
    }
    EXPECT_NEAR(delta, sleepUs, tolUs) << "delta: " << delta << " expected " << sleepUs << std::endl;
}
