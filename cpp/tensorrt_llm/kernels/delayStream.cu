/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/delayStream.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{
__global__ void delayStreamKernel(long long delay_micro_secs)
{
    for (int i = 0; i < delay_micro_secs; ++i)
    {
        // The largest delay __nanosleep can do is 1 millisecond, thus we use for loop to achieve longer delay.
        __nanosleep(1000);
    }
}

void invokeDelayStreamKernel(long long delay_micro_secs, cudaStream_t stream)
{
    delayStreamKernel<<<1, 1, 0, stream>>>(delay_micro_secs);
    check_cuda_error(cudaGetLastError());
}
} // namespace tensorrt_llm::kernels
