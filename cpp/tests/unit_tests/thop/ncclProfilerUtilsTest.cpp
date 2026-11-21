/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/thop/ncclProfilerUtils.h"

#include <gtest/gtest.h>

namespace
{

class NcclProfilerUtilsTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
    }

    void TearDown() override
    {
        // Only test cleanup consumes errors; the observation helper must preserve them.
        static_cast<void>(cudaGetLastError());
        if (mCapturing)
        {
            cudaGraph_t graph{};
            EXPECT_EQ(cudaStreamEndCapture(mStream, &graph), cudaSuccess);
            if (graph)
            {
                EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
            }
        }
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    cudaStream_t mStream{};
    bool mCapturing{false};
};

TEST_F(NcclProfilerUtilsTest, EagerStream)
{
    EXPECT_STREQ(tensorrt_llm::torch_ext::detail::getNcclCudaGraphState(mStream), "eager");
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
}

TEST_F(NcclProfilerUtilsTest, CapturingStream)
{
    ASSERT_EQ(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal), cudaSuccess);
    mCapturing = true;
    EXPECT_STREQ(tensorrt_llm::torch_ext::detail::getNcclCudaGraphState(mStream), "capture");
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
}

TEST_F(NcclProfilerUtilsTest, PreservesPendingWorkloadError)
{
    ASSERT_EQ(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal), cudaSuccess);
    mCapturing = true;
    ASSERT_EQ(cudaEventRecord(nullptr, mStream), cudaErrorInvalidResourceHandle);
    ASSERT_EQ(cudaPeekAtLastError(), cudaErrorInvalidResourceHandle);
    EXPECT_STREQ(tensorrt_llm::torch_ext::detail::getNcclCudaGraphState(cudaStreamLegacy), "unknown");
    EXPECT_EQ(cudaPeekAtLastError(), cudaErrorInvalidResourceHandle);
    EXPECT_EQ(cudaGetLastError(), cudaErrorInvalidResourceHandle);
}

TEST_F(NcclProfilerUtilsTest, FailedCaptureQueryDoesNotPolluteRuntimeError)
{
    // The legacy stream implicitly depends on this blocking stream's global capture.
    ASSERT_EQ(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal), cudaSuccess);
    mCapturing = true;
    CUstreamCaptureStatus captureStatus{};
    ASSERT_EQ(cuStreamIsCapturing(CU_STREAM_LEGACY, &captureStatus), CUDA_ERROR_STREAM_CAPTURE_IMPLICIT);
    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
    EXPECT_STREQ(tensorrt_llm::torch_ext::detail::getNcclCudaGraphState(cudaStreamLegacy), "unknown");
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

} // namespace
