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

namespace tensorrt_llm
{
namespace kernels
{

struct MoeLoadBalanceSingleLayerSignal
{
    static constexpr unsigned long long kGPU = 0ULL;
    static constexpr unsigned long long kCPU = 1ULL;
    static constexpr unsigned long long kDevice = 1ULL;
    static constexpr unsigned long long kSkipStep = 1ULL << 1U;
    static constexpr unsigned long long kDisabled = 1ULL << 63U;
    // Bit 0 means the current owner of this layer, 0: gpu, 1: cpu, updated by cpu and gpu alternately
    // Bit 1 means whether skip statistic for current step, cpu set that at one iteration start,
    //  maybe with or without ownership, but since forward is not started, so no conflict.
    // Bits 2-62 means the current step, updated by cpu after one iteration with cpu ownership
    // Bit 63 means if step update is disabled, 0: not disabled, 1: disabled, updated by cpu
    unsigned long long int volatile stepAndOwner;
};

struct MoeLoadBalanceMetaInfo
{
    // Model Layer Info
    int expertCount;
    int topK;

    // Parallelism Info
    int epRank;
    int epSize;

    // Slot Info
    int slotCountPerRank;
};

struct MoeLoadBalanceStatisticInfo
{
    // Statistic Info
    // expertLoadFactor[i] means the load factor of expert i
    // The length of expertLoadFactor should be expertCount
    float* expertLoadFactor = nullptr;

    // expertTokenCount[i] means the number of tokens of expert i
    // The length of expertTokenCount should be rawDataWindowSize * expertCount
    int* expertTokenCount = nullptr;

    // rawDataWindowSize means the size of the raw data window.
    // e.g. how many steps of raw data are kept in the memory.
    // current we keep only the data in current iteration, previous should sum to expertLoadFactor.
    static constexpr int rawDataWindowSize = 1;

    // decayFactor means the decay factor of the raw data per step.
    // e.g. if decayFactor is 0.95, then the raw data of expert i will be decayed by 0.95 for each step.
    float decayFactor = 0.95f;
};

// The placement information for GPU
struct MoePlacementInfo
{
    // Placement Info
    // expertReplicaCount[i] means the number of replicas of expert i
    int* expertReplicaCount = nullptr;

    // expertReplicaStartOffset[i] means the start offset of expert i's replicas in globalSlotIds
    // and the values of globalSlotIds[expertReplicaStartOffset[i]] ~ globalSlotIds[expertReplicaStartOffset[i] +
    // expertReplicaCount[i] - 1] are possible globalSlotId for expert i, and can be dispatched to any one.
    int* expertReplicaStartOffset = nullptr;

    // globalSlotIds[i] means the global slot id for expert i
    // The length of globalSlotIds should be epSize * slotCountPerRank
    int* globalSlotIds = nullptr;
};

} // namespace kernels
} // namespace tensorrt_llm
