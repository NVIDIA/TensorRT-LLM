/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/ncclUtils.h"

#include <gtest/gtest.h>

#if ENABLE_MULTI_DEVICE
namespace nccl_util = tensorrt_llm::common::nccl_util;

TEST(NCCLWindowSupportTest, RuntimeVersionAndGB10Gate)
{
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    EXPECT_FALSE(nccl_util::isNcclWindowSupportedForPlatform(121, true, NCCL_VERSION(2, 27, 9)));
    EXPECT_FALSE(nccl_util::isNcclWindowSupportedForPlatform(121, true, NCCL_VERSION(2, 29, 2)));
    EXPECT_FALSE(nccl_util::isNcclWindowSupportedForPlatform(121, true, NCCL_VERSION(2, 30, 3)));

    EXPECT_TRUE(nccl_util::isNcclWindowSupportedForPlatform(121, true, NCCL_VERSION(2, 30, 4)));
    EXPECT_TRUE(nccl_util::isNcclWindowSupportedForPlatform(121, false, NCCL_VERSION(2, 29, 2)));
    EXPECT_TRUE(nccl_util::isNcclWindowSupportedForPlatform(120, true, NCCL_VERSION(2, 29, 2)));
    EXPECT_TRUE(nccl_util::isNcclWindowSupportedForPlatform(100, false, NCCL_VERSION(2, 29, 2)));
#else
    GTEST_SKIP() << "NCCL window buffers are not compiled in";
#endif
}

#endif // ENABLE_MULTI_DEVICE
