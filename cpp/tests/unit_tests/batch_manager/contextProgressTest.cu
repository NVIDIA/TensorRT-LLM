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

#include "tensorrt_llm/batch_manager/contextProgress.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thread>

using namespace tensorrt_llm::batch_manager;
using namespace std::chrono;

__global__ void fakeAttention(int* cache, int layerIdx, unsigned computeTimeNs)
{
    // maximum sleep duration: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#nanosleep-function
    constexpr unsigned maxDuration = 1'000'000;
    while (computeTimeNs > maxDuration)
    {
        __nanosleep(maxDuration);
        computeTimeNs -= maxDuration;
    }
    __nanosleep(computeTimeNs);
    cache[layerIdx] = layerIdx;
}

class ContextProgressTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        cudaStreamCreate(&mComputeStream);
        cudaStreamCreate(&mExportStream);
    }

    void TearDown() override
    {
        cudaStreamDestroy(mComputeStream);
        cudaStreamDestroy(mExportStream);
    }

public:
    void runFakePlugin()
    {
        int numLayers = mProgress->getNumLayers();
        for (int i = 0; i < numLayers; i++)
        {
            std::this_thread::sleep_for(mPluginTime);
            fakeAttention<<<1, 1, 0, mComputeStream>>>(mDeviceCache, i, mComputeTime.count());
            EXPECT_EQ(cudaGetLastError(), cudaSuccess);
            mProgress->recordEvent(i, mComputeStream);
        }
    }

    void receiver()
    {
        int numLayers = mProgress->getNumLayers();
        for (int i = 0; i < numLayers; i++)
        {
            mProgress->wait(i);
            int cache;
            TLLM_CUDA_CHECK(
                cudaMemcpyAsync(&cache, mDeviceCache + i, sizeof(int), cudaMemcpyDeviceToHost, mExportStream));
            TLLM_CUDA_CHECK(cudaStreamSynchronize(mExportStream));
            EXPECT_EQ(cache, i);
            std::this_thread::sleep_for(mTransmissionTime);
        }
    }

    void runFakeModel(int numLayers)
    {
        TLLM_CUDA_CHECK(cudaMallocAsync((void**) &mDeviceCache, numLayers * sizeof(int), mComputeStream));
        TLLM_CUDA_CHECK(cudaMemsetAsync(mDeviceCache, -1, numLayers * sizeof(int), mComputeStream));
        mProgress = std::make_unique<ContextProgress>(numLayers);

        std::thread receiverThread(&ContextProgressTest::receiver, this);

        runFakePlugin();

        receiverThread.join();
        TLLM_CUDA_CHECK(cudaFree(mDeviceCache));
    }

    nanoseconds mPluginTime;
    nanoseconds mComputeTime;
    nanoseconds mTransmissionTime;
    std::unique_ptr<ContextProgress> mProgress;
    cudaStream_t mComputeStream;
    cudaStream_t mExportStream;
    int* mDeviceCache = nullptr;
};

TEST_F(ContextProgressTest, ContextProgress)
{
    mPluginTime = milliseconds(0);
    mComputeTime = milliseconds(0);
    mTransmissionTime = milliseconds(0);
    runFakeModel(10);
}

TEST_F(ContextProgressTest, SlowPlugin)
{
    mPluginTime = milliseconds(10);
    mComputeTime = milliseconds(1);
    mTransmissionTime = milliseconds(0);
    runFakeModel(10);
}

TEST_F(ContextProgressTest, SlowCompute)
{
    mPluginTime = milliseconds(1);
    mComputeTime = milliseconds(10);
    mTransmissionTime = milliseconds(0);
    runFakeModel(10);
}

TEST_F(ContextProgressTest, SlowTransmission)
{
    mPluginTime = milliseconds(0);
    mComputeTime = milliseconds(1);
    mTransmissionTime = milliseconds(10);
    runFakeModel(10);
}
