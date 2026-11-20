/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::minimaxM31Sm107::arPrepushed
{

//! OPROJ_PUSH_GEMM (TP2 decode, the o_proj -> attention boundary): the one-shot Lamport all-reduce + residual +
//! RMSNorm of allreduce_fusion_kernel_oneshot_lamport<kARResidualRMSNorm, bf16, 2 ranks, Fp32Acc> with its push loop
//! removed, because the producer (the o_proj GEMM's Lamport-push epilogue, kernels/lamport_push_gemm.py) already
//! stored its rows into this rank's section of every rank's Lamport data slot. Optionally also the MXFP8 of norm_out
//! (mx_out e4m3 [rows, hidden], mx_sf the 128x4-swizzled UE8M0 scales of pad128(rows) rows; pad rows get 0 scale
//! bytes) with the stock MXFP8 quantize's device functions. Bitwise to the stock push + all-reduce (+ mxfp8_quantize of
//! the norm).
struct Params
{
    int rank;
    int size;         //!< rows * hidden_dim
    int hidden_dim;
    void** workspace; //!< the AllReduce's Lamport workspace pointer table (2 ranks)
    void* allreduce_in;
    void* residual_in;
    void* residual_out;
    void* norm_out;
    void* rms_gamma;
    float rms_eps;
    void* mx_out; //!< nullptr: no MXFP8 output
    void* mx_sf;
};

//! The stock allreduce_fusion_kernel_launcher's one-shot shape (cluster launch); programmatic dependent launch follows
//! TRTLLM_ENABLE_PDL as the stock launcher does. size % hidden_dim == 0 and hidden_dim % 8 == 0 are the caller's to
//! check.
cudaError_t invokeArPrepushed(Params const& params, bool triggerCompletionAtEnd, cudaStream_t stream);

} // namespace kernels::minimaxM31Sm107::arPrepushed

TRTLLM_MV2_NAMESPACE_END
