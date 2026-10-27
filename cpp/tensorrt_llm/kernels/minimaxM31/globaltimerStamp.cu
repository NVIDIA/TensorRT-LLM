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

#include "tensorrt_llm/kernels/minimaxM31/globaltimerStamp.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{

__global__ void globaltimerStampKernel(int64_t* out)
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *out = static_cast<int64_t>(t);
}

} // namespace

cudaError_t invokeGlobaltimerStamp(int64_t* out, cudaStream_t stream)
{
    globaltimerStampKernel<<<1, 1, 0, stream>>>(out);
    return cudaGetLastError();
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
