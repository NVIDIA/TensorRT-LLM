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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/opUtils.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <thread>
#include <utility>

namespace
{

size_t getFreeDeviceMemory()
{
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    TLLM_CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    return freeBytes;
}

template <typename Func>
void runInNewThread(Func&& func)
{
    std::thread{std::forward<Func>(func)}.join();
}

} // namespace

// Only cuBLAS handles are checked: cublasLtCreate returns the same handle on every call (cuBLAS 13), so how cuBLASLt
// handles are handed out is not observable.

// A thread started late (such as the serving thread that replaces the profiling thread) may find the PyTorch caching
// allocator holding nearly all device memory, so it must take over the handle of an exited thread instead of allocating
// a new one. The exited thread's stream and workspace may be gone with it, so the taken-over handle must be back on the
// default stream (cuBLAS has no getter for the workspace; cublasSetStream resets it together with the stream).
TEST(CublasHandlePoolTest, NewThreadTakesOverCublasHandleOfExitedThreadOnDefaultStreamWithoutAllocating)
{
    constexpr size_t kWorkspaceSize = 1 << 20;
    cudaStream_t streamOfExitedThread{};
    void* workspaceOfExitedThread{};
    TLLM_CUDA_CHECK(cudaStreamCreate(&streamOfExitedThread));
    TLLM_CUDA_CHECK(cudaMalloc(&workspaceOfExitedThread, kWorkspaceSize));
    cublasHandle_t handleOfExitedThread{};
    runInNewThread(
        [&]
        {
            auto const handle = tensorrt_llm::getCublasHandle();
            handleOfExitedThread = *handle;
            // As CublasMMWrapper does.
            TLLM_CUDA_CHECK(cublasSetStream(*handle, streamOfExitedThread));
            TLLM_CUDA_CHECK(cublasSetWorkspace(*handle, workspaceOfExitedThread, kWorkspaceSize));
        });
    TLLM_CUDA_CHECK(cudaFree(workspaceOfExitedThread));
    TLLM_CUDA_CHECK(cudaStreamDestroy(streamOfExitedThread));

    cublasHandle_t handleOfNewThread{};
    cudaStream_t streamOfNewThread{};
    size_t freeMemoryBefore = 0;
    size_t freeMemoryAfter = 0;
    runInNewThread(
        [&]
        {
            freeMemoryBefore = getFreeDeviceMemory();
            auto const handle = tensorrt_llm::getCublasHandle();
            freeMemoryAfter = getFreeDeviceMemory();
            handleOfNewThread = *handle;
            TLLM_CUDA_CHECK(cublasGetStream(*handle, &streamOfNewThread));
        });

    ASSERT_EQ(handleOfNewThread, handleOfExitedThread);
    EXPECT_EQ(streamOfNewThread, nullptr);
    EXPECT_EQ(freeMemoryAfter, freeMemoryBefore);
}

TEST(CublasHandlePoolTest, LiveThreadsGetDistinctCublasHandlesAndEachThreadKeepsItsOwn)
{
    // Leave one idle handle: the main thread takes it, so the other thread must get a different one.
    runInNewThread([] { tensorrt_llm::getCublasHandle(); });
    auto const handle = tensorrt_llm::getCublasHandle();
    EXPECT_EQ(*tensorrt_llm::getCublasHandle(), *handle);

    cublasHandle_t handleOfOtherThread{};
    runInNewThread([&] { handleOfOtherThread = *tensorrt_llm::getCublasHandle(); });

    EXPECT_NE(handleOfOtherThread, *handle);
}
