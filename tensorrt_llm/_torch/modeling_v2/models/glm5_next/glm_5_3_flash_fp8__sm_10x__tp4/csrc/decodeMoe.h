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

//! The most rows (tokens) one call takes.
constexpr int kDecodeMoeMaxRows = 4;
//! The hidden size H the kernel is compiled for.
constexpr int kDecodeMoeHidden = 4096;
//! The per-rank expert intermediate size I the kernel is compiled for (2048 at moe_tp4).
constexpr int kDecodeMoeInter = 512;
//! The number of routed experts E the kernel is compiled for.
constexpr int kDecodeMoeExperts = 288;
//! The block size of the FP8 weights' scales, along both dimensions.
constexpr int kDecodeMoeScaleBlock = 128;

//! The operands of one decodeMoe() call. Every tensor is contiguous; the weights, their scales and x start on
//! 16-byte boundaries (they are read by TMA bulk copies).
struct DecodeMoeArgs
{
    void const* x;       //!< [M, 4096] bf16
    float const* logits; //!< [M, 288] the router's outputs
    float const* bias;   //!< [288] the routing correction bias
    uint8_t const* w31;  //!< [288, 1024, 4096] e4m3: 512 up rows, then 512 gate rows
    float const* s31;    //!< [288, 8, 32]
    uint8_t const* w2;   //!< [288, 4096, 512] e4m3
    float const* s2;     //!< [288, 32, 4]
    uint8_t const* sw31; //!< [1024, 4096] e4m3: the shared expert, in a routed expert's layout
    float const* ss31;   //!< [8, 32]
    uint8_t const* sw2;  //!< [4096, 512] e4m3
    float const* ss2;    //!< [32, 4]
    void* out;           //!< [M, 4096] bf16
    float scaling;       //!< the routed scaling factor
    float limit;         //!< the SwiGLU clamp
    int rows;            //!< M, 1 to kDecodeMoeMaxRows
    bool pdl;            //!< programmatic dependent launch (only the weights are read before the grid-dependency wait)
};

//! One persistent launch on `stream`: routing, FC1 + SwiGLU, FC2. Its scratch (activations, per-expert completion
//! counters double-buffered by call parity, per-CTA call epochs) is device-global, so calls on one device must be
//! stream-ordered. The kernel needs every CTA resident at once (one per SM).
//!
//! Returns cudaErrorInvalidValue for a row count out of range, cudaErrorNotSupported on a device with fewer SMs than
//! the kernel's smallest grid, cudaErrorCooperativeLaunchTooLarge when the device cannot hold the whole grid at once
//! (the launch would hang rather than fail), and otherwise the result of the CUDA calls.
cudaError_t decodeMoe(DecodeMoeArgs const& args, cudaStream_t stream);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
