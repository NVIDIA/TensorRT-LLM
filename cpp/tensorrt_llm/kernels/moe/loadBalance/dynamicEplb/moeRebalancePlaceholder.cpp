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

#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloQ.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceTma.h"

#include "tensorrt_llm/common/assert.h"

#include <cuda_runtime_api.h>

#ifdef TRTLLM_DYNAMIC_EPLB_PLACEHOLDER_HALO_Q
TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeMoeRebalanceHaloQ(MoeRebalanceHaloQParams const&, cudaStream_t)
{
    TLLM_THROW("Dynamic EPLB HALO-Q requires a requested SM100, SM103, or SM107 target");
}

} // namespace kernels

TRTLLM_NAMESPACE_END
#endif

#ifdef TRTLLM_DYNAMIC_EPLB_PLACEHOLDER_TMA
namespace
{

int unsupported()
{
    return static_cast<int>(cudaErrorNotSupported);
}

} // namespace

extern "C" int megamoe_tma_copy_create(uint64_t, int, int, MegamoeTmaCopyState** out)
{
    if (out != nullptr)
    {
        *out = nullptr;
    }
    return unsupported();
}

extern "C" int megamoe_tma_copy_configure_gpu_plan(MegamoeTmaCopyState*, MegamoeTmaGpuPlanConfig const*)
{
    return unsupported();
}

extern "C" int megamoe_tma_copy_bind_gpu_direct(
    MegamoeTmaCopyState*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, int)
{
    return unsupported();
}

extern "C" int megamoe_tma_copy_submit_gpu_direct(MegamoeTmaCopyState*, uint64_t, uint64_t, void*)
{
    return unsupported();
}

extern "C" int megamoe_tma_copy_gpu_plan_result(MegamoeTmaCopyState*, MegamoeTmaGpuPlanResult*)
{
    return unsupported();
}

extern "C" int megamoe_tma_copy_destroy(MegamoeTmaCopyState** state)
{
    return state == nullptr || *state == nullptr ? static_cast<int>(cudaSuccess) : unsupported();
}

extern "C" int megamoe_tma_copy_config_info(MegamoeTmaCopyState const*, MegamoeTmaCopyConfig*)
{
    return unsupported();
}

extern "C" char const* megamoe_tma_copy_error_string(int error)
{
    if (error == static_cast<int>(cudaErrorNotSupported))
    {
        return "Dynamic EPLB TMA requires CUDA 13.1+ and a requested SM100, SM103, or SM107 target";
    }
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
#endif
