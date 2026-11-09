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

#include "kv_cache_manager_v2/utils/cudaContextScope.h"

#include <gtest/gtest.h>

#include <cuda.h>

#include <thread>

namespace
{

using tensorrt_llm::batch_manager::kv_cache_manager_v2::CudaContextScope;

CUcontext currentContext()
{
    CUcontext context = nullptr;
    EXPECT_EQ(cuCtxGetCurrent(&context), CUDA_SUCCESS);
    return context;
}

//! The device's primary context, current on the test thread: the one a manager would record.
class KvCacheManagerV2CudaContextScopeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
        int deviceCount = 0;
        ASSERT_EQ(cuDeviceGetCount(&deviceCount), CUDA_SUCCESS);
        if (deviceCount == 0)
        {
            GTEST_SKIP() << "no CUDA device";
        }
        ASSERT_EQ(cuDeviceGet(&mDevice, 0), CUDA_SUCCESS);
        ASSERT_EQ(cuDevicePrimaryCtxRetain(&mOwner, mDevice), CUDA_SUCCESS);
        ASSERT_EQ(cuCtxSetCurrent(mOwner), CUDA_SUCCESS);
    }

    void TearDown() override
    {
        if (mOwner != nullptr)
        {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(mDevice);
        }
    }

    //! Runs `body` on a fresh thread, which has no current CUDA context, and joins it.
    template <typename F>
    static void onFreshThread(F&& body)
    {
        std::thread thread(std::forward<F>(body));
        thread.join();
    }

    CUdevice mDevice{};
    CUcontext mOwner = nullptr;
};

TEST_F(KvCacheManagerV2CudaContextScopeTest, PushesOnAThreadWithNoContextAndLeavesItWithNone)
{
    onFreshThread(
        [this]
        {
            ASSERT_EQ(currentContext(), nullptr);
            {
                CudaContextScope const scope(mOwner);
                EXPECT_TRUE(scope.usable());
                EXPECT_TRUE(scope.pushed());
                EXPECT_EQ(scope.error(), CUDA_SUCCESS);
                EXPECT_EQ(currentContext(), mOwner);
            }
            EXPECT_EQ(currentContext(), nullptr);
        });
}

TEST_F(KvCacheManagerV2CudaContextScopeTest, IsANoOpWhenTheContextIsAlreadyCurrent)
{
    CudaContextScope const scope(mOwner);
    EXPECT_TRUE(scope.usable());
    EXPECT_FALSE(scope.pushed());
    EXPECT_EQ(currentContext(), mOwner);
}

TEST_F(KvCacheManagerV2CudaContextScopeTest, RestoresAForeignContextOnExit)
{
    onFreshThread(
        [this]
        {
            // A context of the thread's own, which the owner's must not replace for good.
            CUcontext foreign = nullptr;
#if CUDA_VERSION >= 13000
            ASSERT_EQ(cuCtxCreate(&foreign, nullptr, 0, mDevice), CUDA_SUCCESS);
#else
            ASSERT_EQ(cuCtxCreate(&foreign, 0, mDevice), CUDA_SUCCESS);
#endif
            ASSERT_EQ(currentContext(), foreign);
            {
                CudaContextScope const scope(mOwner);
                EXPECT_TRUE(scope.usable());
                EXPECT_TRUE(scope.pushed());
                EXPECT_EQ(currentContext(), mOwner);
            }
            EXPECT_EQ(currentContext(), foreign);
            EXPECT_EQ(cuCtxDestroy(foreign), CUDA_SUCCESS);
        });
}

} // namespace
