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
// The tensor-parallel all-reduce of a decode step's partial rows fused with the mHC boundary after it
// (GLM-5.3-Flash at tp4).
#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

//! The most rows (tokens) one call takes.
constexpr int kMhcArBoundaryMaxRows = 8;
//! The hidden size the kernel is compiled for.
constexpr int kMhcArBoundaryHidden = 4096;
//! The number of residual streams (hc_mult) the kernel is compiled for.
constexpr int kMhcArBoundaryMult = 4;
//! The size of the tensor-parallel group the kernel is compiled for.
constexpr int kMhcArBoundaryRanks = 4;
//! The mixer's outputs: pre and post (one per stream) and comb (one per pair of streams).
constexpr int kMhcArBoundaryMix = (2 + kMhcArBoundaryMult) * kMhcArBoundaryMult;
//! The pointers in the all-reduce workspace's table: three buffer sets per rank, the flag and the layout buffers.
constexpr int kMhcArBoundaryWorkspacePointers = 3 * kMhcArBoundaryRanks + 2;

//! The operands of one mhcArBoundary() call. Every tensor is contiguous; partial, residual, post, comb, fn and the
//! outputs written in 16-byte stores (residualOut, layerInput) start on 16-byte boundaries.
struct MhcArBoundaryArgs
{
    void** workspace;       //!< the TP group's TensorRT-LLM all-reduce workspace (Lamport buffers, flag, sizes)
    int rank;               //!< this rank in the group of 4
    void const* partial;    //!< [M, 4096] bf16 this rank's partial output of the block
    void const* residual;   //!< [M, 4, 4096] bf16 the streams before the block's post-mapping
    float const* postPrev;  //!< [M, 4]
    float const* combPrev;  //!< [M, 4, 4]
    float const* fn;        //!< [24, 4 * 4096] the next block's mixer weights
    float const* scale;     //!< [3]
    float const* base;      //!< [24]
    void const* normWeight; //!< [4096] bf16 the next block's input RMSNorm
    void* residualOut;      //!< [M, 4, 4096] bf16
    float* postOut;         //!< [M, 4]
    float* combOut;         //!< [M, 4, 4]
    void* layerInput;       //!< [M, 4096] bf16 RMS-normalized
    int rows;               //!< M, 1 to kMhcArBoundaryMaxRows
    int sinkhornIters;      //!< at least 1
    float normEps;          //!< the mixer's RMS epsilon
    float hcEps;            //!< added to pre
    float sinkhornEps;      //!< the Sinkhorn normalization's epsilon
    float postMult;         //!< post's multiplier
    float layerInputEps;    //!< the next block's RMSNorm epsilon
    bool pdl;               //!< programmatic dependent launch (only the weights are read before the wait)
};

//! One launch on `stream`: the all-reduce reads and rotates the workspace's Lamport state, so it is ordered with the
//! other users of that workspace (the library's one-shot all-reduce) by the stream.
//!
//! Returns cudaErrorInvalidValue for a row count out of range and otherwise the result of the CUDA calls.
cudaError_t mhcArBoundary(MhcArBoundaryArgs const& args, cudaStream_t stream);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
