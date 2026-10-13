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

#pragma once

#include <cuda.h>

#include <array>

namespace tensorrt_llm::ugpu::detail
{

constexpr int kUgpuCount = 2;

#if CUDA_VERSION >= 13040

using SmResourceGroupParams = std::array<CU_DEV_SM_RESOURCE_GROUP_PARAMS, kUgpuCount>;

constexpr bool isBalancedSmCountValid(unsigned int totalSmCount)
{
    constexpr unsigned int kSmCountAlignment = 2;
    constexpr unsigned int kMinimumTotalSmCount = kSmCountAlignment * static_cast<unsigned int>(kUgpuCount);
    return totalSmCount >= kMinimumTotalSmCount && (totalSmCount % kMinimumTotalSmCount) == 0;
}

constexpr bool isStrictSplitCountValid(
    unsigned int totalSmCount, unsigned int localityDomainSmCount, unsigned int remainderSmCount)
{
    return localityDomainSmCount > 0 && localityDomainSmCount <= totalSmCount / static_cast<unsigned int>(kUgpuCount)
        && remainderSmCount == totalSmCount - localityDomainSmCount * static_cast<unsigned int>(kUgpuCount);
}

inline SmResourceGroupParams makeStrictSmResourceGroupParams()
{
    SmResourceGroupParams groupParams{};
    for (int ugpuId = 0; ugpuId < kUgpuCount; ++ugpuId)
    {
        groupParams[ugpuId].flags = CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID;
        groupParams[ugpuId].localityDomainId = static_cast<unsigned int>(ugpuId);
    }
    return groupParams;
}

inline SmResourceGroupParams makeBalancedSmResourceGroupParams(unsigned int totalSmCount)
{
    SmResourceGroupParams groupParams = makeStrictSmResourceGroupParams();
    unsigned int const smCountPerUgpu = totalSmCount / static_cast<unsigned int>(kUgpuCount);
    for (auto& params : groupParams)
    {
        params.smCount = smCountPerUgpu;
        params.flags |= CU_DEV_SM_RESOURCE_GROUP_BACKFILL;
    }
    return groupParams;
}

#endif // CUDA_VERSION >= 13040

} // namespace tensorrt_llm::ugpu::detail
