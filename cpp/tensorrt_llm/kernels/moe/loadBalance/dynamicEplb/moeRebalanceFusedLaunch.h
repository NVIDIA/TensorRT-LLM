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

// Cold launch preparation shared by the fused scheduler + copy kernels of per-layer dynamic EPLB: the HALO-Q kernel
// with the in-switch copy (moeRebalanceTma.cu) and the HALO-R kernel with the relay (moeRebalanceHaloR.cu). Each
// caller sizes its own shared memory and reports failures its own way.

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime.h>

#include <cstddef>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! Loads `kernel` and reads its attributes, then raises the device stack limit to its local memory. Doing either at
//! the first launch waits for the device, which deadlocks against kernels that spin on work enqueued after that launch.
inline cudaError_t loadMoeRebalanceFusedKernel(void const* kernel, cudaFuncAttributes& attributes)
{
    cudaError_t error = cudaFuncGetAttributes(&attributes, kernel);
    if (error != cudaSuccess)
    {
        return error;
    }
    std::size_t stack = 0;
    error = cudaDeviceGetLimit(&stack, cudaLimitStackSize);
    if (error == cudaSuccess && stack < attributes.localSizeBytes)
    {
        error = cudaDeviceSetLimit(cudaLimitStackSize, attributes.localSizeBytes);
    }
    return error;
}

//! Gives `kernel` `dynamicBytes` of dynamic shared memory and the largest shared-memory carveout, and sets `active` to
//! the number of its CTAs of `threads` threads that fit one SM.
inline cudaError_t sizeMoeRebalanceFusedKernel(void const* kernel, int threads, int dynamicBytes, int& active)
{
    cudaError_t error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamicBytes);
    if (error == cudaSuccess)
    {
        error = cudaFuncSetAttribute(
            kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    }
    if (error == cudaSuccess)
    {
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads, dynamicBytes);
    }
    return error;
}

//! Fills the launch attributes that run a fused kernel in the remote memory-sync domain, so that fences and completions
//! of kernels in the default domain do not wait for its NVLink traffic. Returns the attribute count: 2, or 0 when the
//! device has a single domain.
inline unsigned setMoeRebalanceRemoteSyncDomain(int device, cudaLaunchAttribute (&attributes)[2])
{
    int domains = 0;
    if (cudaDeviceGetAttribute(&domains, cudaDevAttrMemSyncDomainCount, device) != cudaSuccess || domains < 2)
    {
        return 0;
    }
    attributes[0].id = cudaLaunchAttributeMemSyncDomain;
    attributes[0].val.memSyncDomain = cudaLaunchMemSyncDomainRemote;
    attributes[1].id = cudaLaunchAttributeMemSyncDomainMap;
    attributes[1].val.memSyncDomainMap.default_ = 0;
    attributes[1].val.memSyncDomainMap.remote = 1;
    return 2;
}

} // namespace kernels

TRTLLM_NAMESPACE_END
