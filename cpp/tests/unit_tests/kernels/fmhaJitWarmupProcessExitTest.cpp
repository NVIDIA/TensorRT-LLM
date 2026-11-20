/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// Process-exit test for the asynchronous TRTLLM-Gen FMHA JIT warmup worker, run
// in a child process (gtest death test). The worker is created first, through
// the pre-capture barrier with nothing queued, before the install-path string
// or any other sweep dependency has been touched; work is queued afterwards
// and is still pending when the process exits. The child exits 0 only if that
// pending work ran to completion during exit and observed the same install path
// the main thread saw, i.e. the worker was shut down with its dependencies
// intact regardless of which path created it.

#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

namespace
{

using tensorrt_llm::kernels::asyncJITWarmupWorkerCreated;
using tensorrt_llm::kernels::getAsyncJITWarmupWorker;
using tensorrt_llm::kernels::TllmGenFmhaKernel;

std::atomic<bool> gTaskFinished{false};
std::atomic<bool> gTaskSawSameExecPath{false};
std::string gMainThreadExecPath;

[[noreturn]] void runChild()
{
    setenv("TRTLLM_GEN_FMHA_ASYNC_WARMUP", "1", 1);

    // Registered first, so it runs last, after the worker's own atexit shutdown.
    std::atexit(
        []()
        {
            bool const ok = gTaskFinished.load() && gTaskSawSameExecPath.load();
            std::_Exit(ok ? 0 : 42);
        });

    // "Empty barrier": nothing has been queued and the install path has never
    // been computed. This must not create a worker whose shutdown ordering
    // depends on static objects that do not exist yet.
    EXPECT_EQ(TllmGenFmhaKernel::drainAndVerifyAllJITWarmups(), 0);
    EXPECT_FALSE(asyncJITWarmupWorkerCreated().load());

    // Create the worker explicitly before the install path exists, which is the
    // worst-case creation order, then compute the path on this thread.
    auto& worker = getAsyncJITWarmupWorker();
    EXPECT_TRUE(asyncJITWarmupWorkerCreated().load());
    gMainThreadExecPath = TllmGenFmhaKernel::getExecPath();

    // Queue work that is still pending when exit() starts.
    bool const queued = worker.enqueue(
        []()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            gTaskSawSameExecPath = (TllmGenFmhaKernel::getExecPath() == gMainThreadExecPath);
            gTaskFinished = true;
        });
    EXPECT_TRUE(queued);
    EXPECT_EQ(worker.numPendingTasks(), 1);

    // Normal process exit: atexit handlers and static destruction run.
    std::exit(0);
}

TEST(TllmGenFmhaAsyncJITWarmupProcessExit, PendingWorkFinishesWithDependenciesAliveAtExit)
{
    EXPECT_EXIT(runChild(), ::testing::ExitedWithCode(0), "");
}

} // namespace
