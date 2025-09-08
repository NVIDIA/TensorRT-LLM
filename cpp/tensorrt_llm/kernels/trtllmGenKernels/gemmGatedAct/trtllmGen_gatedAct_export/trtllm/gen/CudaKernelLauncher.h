/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#pragma once

#ifdef TLLM_ENABLE_CUDA
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif
namespace gemmGatedAct
{

namespace trtllm
{
namespace gen
{

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef TLLM_ENABLE_CUDA
inline CUresult launchKernel(void* kernelParams, void* cudaStream, int32_t smemSize, CUfunction kernel, dim3 block3,
    dim3 grid3, dim3 cluster3, bool enablesPdl)
{
    // Make sure we can launch with that much shared memory.
    if (smemSize > 48 * 1024)
    {
        CUresult result = cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smemSize);
        if (result != CUDA_SUCCESS)
        {
            return result;
        }
    }

    auto clusterDim = cluster3.x * cluster3.y * cluster3.z;

    CUlaunchConfig launchConfig;
    launchConfig.blockDimX = block3.x;
    launchConfig.blockDimY = block3.y;
    launchConfig.blockDimZ = block3.z;
    launchConfig.gridDimX = grid3.x;
    launchConfig.gridDimY = grid3.y;
    launchConfig.gridDimZ = grid3.z;
    launchConfig.hStream = reinterpret_cast<CUstream>(cudaStream);
    launchConfig.sharedMemBytes = smemSize;

    CUlaunchAttribute launchAttrs[3];
    launchAttrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    launchAttrs[0].value.clusterDim.x = cluster3.x;
    launchAttrs[0].value.clusterDim.y = cluster3.y;
    launchAttrs[0].value.clusterDim.z = cluster3.z;
    launchAttrs[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    launchAttrs[1].value.clusterSchedulingPolicyPreference
        = (clusterDim > 1) ? CU_CLUSTER_SCHEDULING_POLICY_SPREAD : CU_CLUSTER_SCHEDULING_POLICY_DEFAULT;
    launchAttrs[2].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    launchAttrs[2].value.programmaticStreamSerializationAllowed = enablesPdl;
    launchConfig.attrs = launchAttrs;
    launchConfig.numAttrs = 3;

    // Add setting for non-portable cluster size.
    if (clusterDim > 8)
    {
        CUresult result = cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
            1 // Enable non-portable cluster sizes
        );
        if (result != CUDA_SUCCESS)
        {
            return result;
        }
    }

    // Launch the kernel.
    return cuLaunchKernelEx(&launchConfig, kernel, &kernelParams, nullptr);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gen
} // namespace trtllm

} // namespace gemmGatedAct
