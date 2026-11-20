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
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::minimaxM31
{

//! CTX layer boundary (DEP2CTX_NORMQ_V2): ctxResidualRmsNormMxfp8Sm103's residual add + RMSNorm + MXFP8 (and
//! optional FP32 / NVFP4 copies) with the same bits on every output, more rows in flight per SM (a cp.async.bulk ring
//! of rows per CTA, L6's deeper ring at one 768-thread CTA per SM on sm_107, or CBNQ_P2's two 384-thread CTAs per SM
//! with two chunks per thread; coalesced scale-tile stores). Unlike the served op the BF16 norm is optional
//! with the FP32 copy, the NVFP4 rows and scales take row strides (a MegaMoE input stage's rows), the stage's padded
//! tail (ids -1, weights 0) can be written in the same launch, and residual == null recomputes only the norm of rows
//! that already hold inter + residual (the BF16 norm a dropped-norm boundary left out).
struct CtxBoundaryNormQuantParams
{
    __nv_bfloat16* inter = nullptr;          //!< [rows, hidden]; += residual in place when residual is set
    __nv_bfloat16 const* residual = nullptr; //!< [rows, hidden], or null: inter already holds the sum (norm only)
    __nv_bfloat16 const* weight = nullptr;   //!< [hidden]
    __nv_bfloat16* normOut = nullptr;        //!< [rows, hidden] BF16 norm, optional
    uint64_t* qOut = nullptr;                //!< [rows, hidden] E4M3 (with residual: required)
    uint8_t* sfOut = nullptr;                //!< swizzled 128x4 UE8M0 scales of ceil(rows / 128) * 128 rows
    float* f32Out = nullptr;                 //!< [rows, hidden] FP32 widening of the norm, optional
    uint32_t* fp4Out = nullptr;              //!< NVFP4 rows (hidden / 2 bytes each), optional
    uint8_t* fp4SfOut = nullptr;             //!< linear E4M3 scales (hidden / 16 bytes per row), with fp4Out
    float const* fp4GlobalScale = nullptr;   //!< [>= 1] fp32, with fp4Out
    int64_t fp4StrideWords = 0;              //!< fp4Out row stride in 32-bit words
    int64_t fp4SfStride = 0;                 //!< fp4SfOut row stride in bytes (a multiple of 16)
    int32_t* tailIds = nullptr;              //!< optional stage tail: rows [rows, tailIdsRows) <- -1
    float* tailWeights = nullptr;            //!< optional stage tail: rows [rows, tailWeightsRows) <- 0
    int64_t tailIdsStride = 0;
    int64_t tailWeightsStride = 0;
    int tailIdsRows = 0;
    int tailWeightsRows = 0;
    int tailTopk = 0; //!< columns of the tail written (0: none)
    int rows = 0;
    int hidden = 0;
    float eps = 0.f;
    // Launch choice (the defaults are the served launch; each form writes the same bits):
    bool p2 = true;     //!< CBNQ_P2: try the two-chunk, two-CTAs-per-SM kernel first (sm_107); false: one chunk
    bool twoCta = true; //!< CBNQ_2CTA: false forces the three-stage one-chunk ring without a carveout attribute
    int stages = 3;     //!< requested one-chunk ring depth (3, 4, 6 or 8; other values: 3), stepped down to fit
    int p2Stages = 3;   //!< requested P2 ring depth (3 or 4; other values: 3), 3 when 4 does not fit
};

//! Whether the kernel serves this hidden size (hidden % 256 == 0, 256 <= hidden <= 6144: hidden / 8 threads within
//! the two-CTA launch bound).
bool ctxBoundaryNormQuantSupported(int hidden);

//! Launches the boundary on stream. Returns cudaErrorInvalidValue for a combination of outputs it has no form for
//! (with residual: qOut / sfOut required; without: normOut only), cudaErrorNotSupported on an unsupported hidden size
//! or device (sm_10x only), else the launch's error.
cudaError_t invokeCtxBoundaryNormQuant(CtxBoundaryNormQuantParams const& params, bool enablePdl, cudaStream_t stream);

//! (CTAs per SM, grid) the launch would use for this flavor and row count on the current device; for tests and the MB.
cudaError_t ctxBoundaryNormQuantLaunchShape(
    CtxBoundaryNormQuantParams const& params, int* ctasPerSm, int* grid, int* smemBytes);

//! The launch form as well: chunks per thread (2: the CBNQ_P2 kernel at hidden / 16 threads, two CTAs per SM; 1: the
//! hidden / 8-thread kernel) and the ring depth, with the launch shape; for tests and the MB.
cudaError_t ctxBoundaryNormQuantLaunchForm(CtxBoundaryNormQuantParams const& params, int* chunksPerThread,
    int* ctasPerSm, int* grid, int* smemBytes, int* stages);

//! DEP2CTX_TOPK_FOLD (K5 U2): the next-layer boundary of a MoE layer whose MegaMoE launch deferred its top-k reduce.
//! Per row, megamoe_topk_reduce_shared_add (moe = bf16(shared + bf16(sum_k plane[k] * scores[k]))) and then the
//! boundary's residual add + RMSNorm + MXFP8 (+ BF16 norm) of moe, in one pass: every output bit for bit those two
//! kernels' (shared ends holding moe + residual, normOut / qOut / sfOut as invokeCtxBoundaryNormQuant writes them).
struct CtxBoundaryTopkNormQuantParams
{
    __nv_bfloat16* shared = nullptr;         //!< [rows, hidden] shared-expert rows; moe + residual in place
    __nv_bfloat16 const* plane = nullptr;    //!< [rows, topk, hidden] the deferred launch's unweighted FC2 rows
    float const* scores = nullptr;           //!< [rows, topk] FP32 top-k weights
    int topk = 0;                            //!< 4 (the only K the fold serves)
    __nv_bfloat16 const* residual = nullptr; //!< [rows, hidden]
    __nv_bfloat16 const* weight = nullptr;   //!< [hidden]
    __nv_bfloat16* normOut = nullptr;        //!< [rows, hidden] BF16 norm, optional
    uint64_t* qOut = nullptr;                //!< [rows, hidden] E4M3
    uint8_t* sfOut = nullptr;                //!< swizzled 128x4 UE8M0 scales of ceil(rows / 128) * 128 rows
    int rows = 0;
    int hidden = 0;
    float eps = 0.f;
    bool dual = true; //!< CBNQ_TOPK_DUAL: two row groups per CTA where the thread limit allows (sm_107); false: one
};

//! Launches the fold on stream. Returns cudaErrorNotSupported unless topk == 4, the hidden size is supported, the
//! three-stage ring fits the device's per-block shared-memory opt-in, every pointer is 16-byte aligned and the device
//! is sm_10x; cudaErrorInvalidValue on a missing required pointer; else the launch's error.
cudaError_t invokeCtxBoundaryTopkNormQuant(
    CtxBoundaryTopkNormQuantParams const& params, bool enablePdl, cudaStream_t stream);

//! (CTAs per SM, grid, dynamic shared memory bytes) the fold's launch would use; for tests and the MB.
cudaError_t ctxBoundaryTopkNormQuantLaunchShape(
    CtxBoundaryTopkNormQuantParams const& params, int* ctasPerSm, int* grid, int* smemBytes);

//! The fold's launch form as well: dual = 1 for the CBNQ_TOPK_DUAL kernel (two hidden / 16-thread row groups per CTA,
//! sm_107), 0 for the one-group fold; for tests.
cudaError_t ctxBoundaryTopkNormQuantLaunchForm(
    CtxBoundaryTopkNormQuantParams const& params, int* dual, int* ctasPerSm, int* grid, int* smemBytes);

} // namespace kernels::minimaxM31

TRTLLM_MV2_NAMESPACE_END
