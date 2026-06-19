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
    // Unset (nullptr) and set-but-empty both fall back to the default path.
    EXPECT_EQ(tc::resolveNixlPortLockPath(nullptr), "/tmp/trtllm_nixl_port.lock");
    EXPECT_EQ(tc::resolveNixlPortLockPath(""), "/tmp/trtllm_nixl_port.lock");
}

TEST(EnvUtils, ResolveNixlPortLockPathOverride)
{
    // A non-empty value is used verbatim.
    EXPECT_EQ(tc::resolveNixlPortLockPath("/run/deployA/nixl_port.lock"), "/run/deployA/nixl_port.lock");
}
