/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa::gemm2K256
{

// FC2 of the NVFP4 MoE at intermediate size 256 and tile N 8, with the inputs and output of the TRTLLM-Gen Gemm2
// cubins:
//   output[8 t + r][f] = bf16(outputScale[e] * sum_k weights[e][f][k] * act[8 t + r][k]),  e = ctaIdxXyToBatchIdx[t],
// for the routing's token tiles t < numNonExitingCtas and their valid rows r < ctaIdxXyToMnLimit[t] - 8 t.
struct Data
{
    // [numLocalExperts * hiddenSize][128 B] E2M1, the Gemm2 weight layout (rows shuffled for the 128-row epilogue
    // tile).
    void const* weights;
    // E4M3 block scales of the weights, 128x4 layout: [numLocalExperts][hiddenSize / 128][4][512 B].
    void const* weightsScale;
    // FC1 output: [rows][128 B] E2M1 and its E4M3 block scales in the 8x4 layout ([rows / 8][4][8][4]).
    void const* act;
    void const* actScale;
    float const* outputScale; // [numLocalExperts]
    int32_t const* numNonExitingCtas;
    int32_t const* ctaIdxXyToBatchIdx;
    int32_t const* ctaIdxXyToMnLimit;
    void* output; // [rows][hiddenSize] BF16
    int32_t hiddenSize;
    int32_t numLocalExperts;
    int32_t maxNumCtas; // entries of ctaIdxXyToBatchIdx and ctaIdxXyToMnLimit
    bool usePdl;
    // The launch's preferred shared-memory carveout in percent (cudaLaunchAttributePreferredSharedMemoryCarveout;
    // cudaSharedmemCarveoutMaxShared = 100), or -1: no preference, the driver chooses.
    int32_t sharedMemCarveout = -1;
};

// Whether run() covers the shape on this device: an SM100-family GPU, hiddenSize a multiple of 128 and at least
// 3,712, at most 128 token tiles, and at most 29 units (token tile x 128 output features) per SM.
bool isSupported(int32_t hiddenSize, int32_t maxNumCtas);

void run(Data const& data, cudaStream_t stream);

} // namespace kernels::glm_moe_dsa::gemm2K256

TRTLLM_NAMESPACE_END
