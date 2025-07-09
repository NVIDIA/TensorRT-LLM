/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/kernels/moeLoadBalance/moeLoadBalanceCommon.h"

namespace tensorrt_llm
{
namespace kernels
{

// @brief wait for the signal from cpu to gpu
//
// This function is used to launch a kernel to wait for the signal from cpu to gpu.
// The signal should be set by moeSetSignalForGpuStageHost before calling this function.
// After the signal is set, functions like moeStatisticDevice can be called.
//
// @param signal: the signal
// @param enabled: output flag on device memory to indicate if the statistic is enabled
// @param stream: the stream to wait for the signal
// @precondition: the signal is set by moeSetSignalForGpuStageHost
void moeWaitSignalForGpuStageDevice(MoeLoadBalanceSingleLayerSignal* signal, int* enabled, cudaStream_t stream);

// @brief host version of moeWaitSignalForGpuStageDevice, should only be used for tests.
//
// @param signal: the signal
// @param enabled: output flag on host memory to indicate if the statistic is enabled
void moeWaitSignalForGpuStageForTest(MoeLoadBalanceSingleLayerSignal* signal, int* enabled);

// @brief set the signal for cpu stage
//
// This function is used to launch a kernel to set the signal for cpu stage.
// Functions like moeStatisticDevice should be called before this function.
// Then host can wait for the signal by moeWaitSignalForCpuStageHost.
//
// @param signal: the signal
// @param stream: the stream to set the signal
void moeSetSignalForCpuStageDevice(MoeLoadBalanceSingleLayerSignal* signal, cudaStream_t stream);

// @brief host version of moeSetSignalForCpuStageDevice, should only be used for tests.
//
// @param signal: the signal
void moeSetSignalForCpuStageForTest(MoeLoadBalanceSingleLayerSignal* signal);

// @brief do the statistic
//
// This function is used to launch a kernel to do the statistic.
//
// @param metaInfo: the meta info
// @param statisticInfo: the statistic info
// @param numTotalTokens: the total number of tokens in gatheredRawExpertIds
// @param enabled: flag on device memory to indicate if the statistic is enabled
// @param isFirstStage: whether the current stage is the first stage (only first stage need shift window)
// @param isLastStage: whether the current stage is the last stage (only last stage need update load factor)
// @param gatheredRawExpertIds: the gathered raw expert ids, should have shape [numTotalTokens, metaInfo.topK]
void moeStatisticDevice(MoeLoadBalanceMetaInfo metaInfo, MoeLoadBalanceStatisticInfo statisticInfo, int numTotalTokens,
    int* const enabled, bool isFirstStage, bool isLastStage, int* const gatheredRawExpertIds, cudaStream_t stream);

// @brief do the statistic based on local device's data
//
// This function is used to launch a kernel to do the statistic for local tokens.
//
// @param metaInfo: the meta info
// @param numTotalTokens: the total number of tokens in localRawExpertIds
// @param localExpertTokenCount: the token count that each expert has for local tokens.
// @param enabled: flag on device memory to indicate if the statistic is enabled
// @param isFirstStage: whether the current stage is the first stage (only first stage need shift window)
// @param isLastStage: whether the current stage is the last stage (only last stage need update load factor)
// @param localRawExpertIds: the gathered raw expert ids, should have shape [numTotalTokens, metaInfo.topK]
void moeHierarchicalStatisticLocalDevice(MoeLoadBalanceMetaInfo metaInfo, int numTotalTokens,
    int* localExpertTokenCount, int* const enabled, bool isFirstStage, bool isLastStage, int* const localRawExpertIds,
    cudaStream_t stream);

// @brief update the statistic info based on global info
//
// This function is used to launch a kernel to update the statistic info per iteration.
//
// @param metaInfo: the meta info
// @param statisticInfo: the statistic info
// @param globalExpertTokenCount: the global expert token count, should have shape [metaInfo.expertCount]
// @param enabled: flag on device memory to indicate if the statistic is enabled
void moeHierarchicalStatisticUpdate(MoeLoadBalanceMetaInfo metaInfo, MoeLoadBalanceStatisticInfo statisticInfo,
    int* globalExpertTokenCount, int* const enabled, cudaStream_t stream);

// @brief compute the route
//
// This function is used to launch a kernel to compute the route based on the token selected experts and the placement
// info.
//  For all input expert < 0 or >= metaInfo.expertCount, the route is set to invalid rank (metaInfo.epSize).
//
// @param metaInfo: the meta info
// @param placementInfo: the placement info
// @param tokenSelectedExperts: the selected experts of all tokenCount tokens, has shape of [tokenCount * topK]
// @param tokenRoutedRankIds: output the routed slotIds of all tokenCount tokens, has shape of [tokenCount * topK]
// @param tokenCount: the token count to compute the route
// @param offsetByEpRank: whether to offset the round robin position by epRank
// @param stream: the CUDA stream to be used
void moeComputeRouteDevice(MoeLoadBalanceMetaInfo metaInfo, MoePlacementInfo placementInfo,
    int* const tokenSelectedExperts, int* tokenRoutedSlotIds, int tokenCount, bool offsetByEpRank, cudaStream_t stream);

// @brief wait for the signal from gpu to cpu on host
//
// This function is used to wait for the signal from gpu to cpu on host.
// The signal should be set by moeSetSignalForCpuStageDevice before calling this function.
// After this function is called, functions for weight update can be called.
//
// @param signal: the signal
void moeWaitSignalForCpuStageHost(MoeLoadBalanceSingleLayerSignal* signal);

// @brief set the signal for gpu stage on host
//
// This function is used to set the signal for gpu stage on host.
// Functions like weights update should be called before this function.
// Then host will set flag for next iteration for device by this function.
//
// @param signal: the signal
// @param iterId: the iteration id
// @param enableStatistic: whether the statistic is enabled
void moeSetSignalForGpuStageHost(MoeLoadBalanceSingleLayerSignal* signal, int64_t iterId, bool enableStatistic);

} // namespace kernels
} // namespace tensorrt_llm
