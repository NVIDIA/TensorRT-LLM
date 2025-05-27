/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/moeCommKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <vector>

#include "tensorrt_llm/kernels/moeLoadBalance/moeLoadBalanceKernels.h"
#include "tensorrt_llm/runtime/moeLoadBalancer.h"

namespace torch_ext
{

torch::Tensor moeLoadBalanceWaitGpuStage(int64_t singleLayerLoadBalancerPtr)
{
    TORCH_CHECK(singleLayerLoadBalancerPtr != 0, "singleLayerLoadBalancerPtr must be non-null");
    auto* loadBalancer
        = reinterpret_cast<tensorrt_llm::runtime::SingleLayerMoeLoadBalancer*>(singleLayerLoadBalancerPtr);
    auto enabled = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto signal = loadBalancer->getSignal();
    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::moeWaitSignalForGpuStageDevice(signal, enabled.data_ptr<int>(), stream);

    return enabled;
}

void moeLoadBalanceSetCpuStage(int64_t singleLayerLoadBalancerPtr)
{
    TORCH_CHECK(singleLayerLoadBalancerPtr != 0, "singleLayerLoadBalancerPtr must be non-null");
    auto* loadBalancer
        = reinterpret_cast<tensorrt_llm::runtime::SingleLayerMoeLoadBalancer*>(singleLayerLoadBalancerPtr);
    auto signal = loadBalancer->getSignal();
    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::moeSetSignalForCpuStageDevice(signal, stream);
}

void moeLoadBalanceStatistic(torch::Tensor gatheredRawExpertIds, torch::Tensor enabled,
    int64_t singleLayerLoadBalancerPtr, int64_t isFirstStage, int64_t isLastStage)
{
    CHECK_INPUT(gatheredRawExpertIds, torch::kInt32);
    CHECK_INPUT(enabled, torch::kInt32);
    TORCH_CHECK(gatheredRawExpertIds.dim() == 2, "gatheredRawExpertIds must be a 2D tensor");
    int topK = gatheredRawExpertIds.size(1);
    TORCH_CHECK(enabled.dim() == 1, "enabled must be a 1D tensor");
    TORCH_CHECK(enabled.size(0) == 1, "enabled must have 1 element");
    TORCH_CHECK(isFirstStage == 0 || isFirstStage == 1, "isFirstStage must be 0 or 1");
    TORCH_CHECK(isLastStage == 0 || isLastStage == 1, "isLastStage must be 0 or 1");
    TORCH_CHECK(singleLayerLoadBalancerPtr != 0, "singleLayerLoadBalancerPtr must be non-null");

    auto* loadBalancer
        = reinterpret_cast<tensorrt_llm::runtime::SingleLayerMoeLoadBalancer*>(singleLayerLoadBalancerPtr);
    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo = loadBalancer->getMetaInfo();

    TORCH_CHECK(topK == metaInfo.topK, "topK must be equal to metaInfo.topK");

    auto statisticInfo = loadBalancer->getStatisticInfo();
    int numTotalTokens = gatheredRawExpertIds.size(0);

    tensorrt_llm::kernels::moeStatisticDevice(metaInfo, *statisticInfo, numTotalTokens, enabled.data_ptr<int>(),
        static_cast<bool>(isFirstStage), static_cast<bool>(isLastStage), gatheredRawExpertIds.data_ptr<int>(), stream);
}

torch::Tensor moeLoadBalanceRouting(torch::Tensor tokenSelectedExperts, int64_t singleLayerLoadBalancerPtr)
{
    CHECK_INPUT(tokenSelectedExperts, torch::kInt32);
    TORCH_CHECK(tokenSelectedExperts.dim() == 2, "tokenSelectedExperts must be a 2D tensor");
    int topK = tokenSelectedExperts.size(1);
    TORCH_CHECK(singleLayerLoadBalancerPtr != 0, "singleLayerLoadBalancerPtr must be non-null");
    TORCH_CHECK(singleLayerLoadBalancerPtr != 0, "singleLayerLoadBalancerPtr must be non-null");
    auto* loadBalancer
        = reinterpret_cast<tensorrt_llm::runtime::SingleLayerMoeLoadBalancer*>(singleLayerLoadBalancerPtr);
    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo = loadBalancer->getMetaInfo();

    TORCH_CHECK(topK == metaInfo.topK, "topK must be equal to metaInfo.topK");

    int tokenCount = tokenSelectedExperts.size(0);

    auto tokenRoutedSlotIds = torch::empty_like(tokenSelectedExperts);

    tensorrt_llm::kernels::moeComputeRouteDevice(metaInfo, loadBalancer->getPlacementCpuInfo()->placementInfoForGPU,
        tokenSelectedExperts.data_ptr<int>(), tokenRoutedSlotIds.data_ptr<int>(), tokenCount, stream);

    return tokenRoutedSlotIds;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("moe_load_balance_wait_gpu_stage(int single_layer_load_balancer_ptr) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("moe_load_balance_wait_gpu_stage", &torch_ext::moeLoadBalanceWaitGpuStage);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("moe_load_balance_set_cpu_stage(int single_layer_load_balancer_ptr) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("moe_load_balance_set_cpu_stage", &torch_ext::moeLoadBalanceSetCpuStage);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_load_balance_statistic(Tensor gathered_raw_expert_ids, Tensor enabled, int "
        "single_layer_load_balancer_ptr, int is_first_stage, int is_last_stage) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_load_balance_statistic", &torch_ext::moeLoadBalanceStatistic);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("moe_load_balance_routing(Tensor token_selected_experts, int single_layer_load_balancer_ptr) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_load_balance_routing", &torch_ext::moeLoadBalanceRouting);
}
