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

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

//! IPC one-shot Lamport all-reduce of a block output (stock `allreduce_fusion_kernel_oneshot_lamport` workspace and
//! protocol) fused with a hyper-connection site's `hcOut = bf16(fma(inject, block, hc))` and grouped RMS norm
//! `normed = bf16((hcOut * rstd) * (1 + w))`; bitwise that all-reduce followed by the Triton `_hc_inject_norm_kernel`.
//! With `shared` and `gate`, the block is first `bf16(fma(sigmoid(gate), shared, routed))`.
struct HcInjectNormParams
{
    int nranks;
    int rank;
    int rows;
    int hiddenDim;               //!< 2560
    int hcCount;                 //!< 4
    int programWarps;            //!< 4 or 2: warps of the Triton program whose reduction order is reproduced
    void** workspace;            //!< the stock all-reduce fusion IPC workspace
    void const* partial;         //!< [rows, hidden] bf16, contiguous
    void const* hc;              //!< [rows, hcCount * hidden] bf16
    int64_t hcStride;            //!< row stride of hc, in elements
    void const* inject;          //!< [rows, >= hcCount] bf16
    int64_t injectStride;        //!< row stride of inject, in elements
    void const* shared;          //!< optional [rows, hidden] bf16
    int64_t sharedStride;        //!< row stride of shared, in elements
    void const* gate;            //!< optional [rows, 1] bf16
    int64_t gateStride;          //!< row stride of gate, in elements
    void const* normWeight;      //!< [hcCount * hidden] bf16
    void* hcOut;                 //!< [rows, hcCount * hidden] bf16
    int64_t hcOutStride;         //!< row stride of hcOut, in elements
    void* normed;                //!< [rows, hcCount * hidden] bf16
    int64_t normedStride;        //!< row stride of normed, in elements
    float eps;
    bool triggerCompletionAtEnd; //!< trigger the dependent launch at the end instead of after the grid dependency wait
    cudaStream_t stream;
};

//! Launches the fused all-reduce + inject + norm of `params` on `params.stream`.
void launchHcAllreduceInjectNorm(HcInjectNormParams const& params);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
