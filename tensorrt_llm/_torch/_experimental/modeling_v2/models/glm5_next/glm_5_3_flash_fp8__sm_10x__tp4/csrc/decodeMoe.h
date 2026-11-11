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
// The MoE branch of a decode step of at most 4 rows (GLM-5.3-Flash at moe_tp4), routing included.
#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

constexpr int kDecodeMoeMaxRows = 4;

struct DecodeMoeArgs
{
    void const* x;       // [M, 4096] bf16
    float const* logits; // [M, 288]
    float const* bias;   // [288]
    uint8_t const* w31;  // [288, 1024, 4096] e4m3: 512 up rows, then 512 gate rows
    float const* s31;    // [288, 8, 32]
    uint8_t const* w2;   // [288, 4096, 512] e4m3
    float const* s2;     // [288, 32, 4]
    uint8_t const* sw31; // the shared expert, same layout
    float const* ss31;
    uint8_t const* sw2;
    float const* ss2;
    void* out; // [M, 4096] bf16
    float scaling;
    float limit;
    int rows;
    bool pdl; // programmatic dependent launch (only the weights are read before the grid-dependency wait)
};

// One persistent launch on `stream`: routing, FC1 + SwiGLU, FC2. Its scratch (activations, per-expert completion
// counters double-buffered by call parity, per-CTA call epochs) is device-global, so calls on one device must be
// stream-ordered.
cudaError_t decodeMoe(DecodeMoeArgs const& args, cudaStream_t stream);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
