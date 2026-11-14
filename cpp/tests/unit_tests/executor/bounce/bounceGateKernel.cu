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

#include "bounceGateKernel.h"

namespace bounce_test
{
namespace
{
__global__ void gateKernel(int const volatile* flag)
{
    if (threadIdx.x == 0)
    {
        while (*flag == 0)
        {
            __nanosleep(1000);
        }
    }
    __syncthreads();
}
} // namespace

GateKernel::GateKernel()
{
    void* host = nullptr;
    if (cudaHostAlloc(&host, sizeof(int), cudaHostAllocMapped) == cudaSuccess)
    {
        mFlag = static_cast<int volatile*>(host);
        *mFlag = 0;
        void* dev = nullptr;
        (void) cudaHostGetDevicePointer(&dev, host, 0);
        mFlagDev = static_cast<int*>(dev);
    }
    (void) cudaEventCreateWithFlags(&mDone, cudaEventDisableTiming);
}

GateKernel::~GateKernel()
{
    release();
    if (mStream != nullptr && mDone != nullptr)
    {
        (void) cudaEventSynchronize(mDone);
    }
    if (mDone != nullptr)
    {
        (void) cudaEventDestroy(mDone);
    }
    if (mFlag != nullptr)
    {
        (void) cudaFreeHost(const_cast<int*>(mFlag));
    }
}

cudaError_t GateKernel::launch(cudaStream_t stream)
{
    if (mFlagDev == nullptr || mDone == nullptr)
    {
        return cudaErrorMemoryAllocation;
    }
    int dev = 0;
    int sms = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err == cudaSuccess)
    {
        err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    if (err != cudaSuccess)
    {
        return err;
    }
    gateKernel<<<sms, 64, 0, stream>>>(mFlagDev);
    err = cudaGetLastError();
    if (err == cudaSuccess)
    {
        err = cudaEventRecord(mDone, stream);
        mStream = stream;
    }
    return err;
}

void GateKernel::release()
{
    if (mFlag != nullptr)
    {
        *mFlag = 1;
        __sync_synchronize();
    }
}

bool GateKernel::running() const
{
    return mStream != nullptr && cudaEventQuery(mDone) == cudaErrorNotReady;
}

} // namespace bounce_test
