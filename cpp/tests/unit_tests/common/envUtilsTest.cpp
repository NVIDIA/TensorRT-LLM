/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/envUtils.h"

namespace tc = tensorrt_llm::common;

TEST(EnvUtils, ResolveNixlPortLockPathDefault)
{
    std::string const kDefaultNixlPortLockPath = "/tmp/trtllm_nixl_port.lock";
    // Unset (nullptr) and set-but-empty both fall back to the default path.
    EXPECT_EQ(tc::resolveNixlPortLockPath(nullptr), kDefaultNixlPortLockPath);
    EXPECT_EQ(tc::resolveNixlPortLockPath(""), kDefaultNixlPortLockPath);
}

TEST(EnvUtils, ResolveNixlPortLockPathOverride)
{
    std::string const kOverrideNixlPortLockPath = "/run/deployA/nixl_port.lock";
    // A non-empty value is used verbatim.
    EXPECT_EQ(tc::resolveNixlPortLockPath(kOverrideNixlPortLockPath.c_str()), kOverrideNixlPortLockPath);
}
