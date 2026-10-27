/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::minimaxM31Sm107::s2tail
{

constexpr int kElems = 8; // bf16 per float4 access
// The AUTO all-reduce strategy takes the one-shot Lamport kernel for up to 256 rows; this op covers the same range.
constexpr int kMaxTokens = 256;

struct Params
{
    int nranks;
    int rank;
    int size; // tokens * hidden
    int hidden_dim;
    int top_k;
    int ld_fc2;
    void** workspace;
    __nv_bfloat16 const* fc2;
    void const* scales; // [T, top_k] bf16 or fp32
    int const* perm;    // [T, top_k] expanded -> permuted row, < 0 = dropped
    __nv_bfloat16 const* shared;
    __nv_bfloat16 const* residual_in;
    __nv_bfloat16 const* rms_gamma;
    float rms_eps;
    __nv_bfloat16* residual_out;
    __nv_bfloat16* norm_out;
    uint64_t* quant_out;
    uint32_t* scale_out;
};

//! TP2W_S2_TAIL: the MoE tail of one TP2 decode boundary in one launch (routed finalize + shared add, the one-shot
//! Lamport all-reduce on the AllReduce module's workspace, residual add + RMSNorm, MXFP8 out). scalesBf16 picks the
//! expert-scale dtype (bf16 or fp32); trigger releases dependents at the end.
void invokeS2MoeTail(Params const& params, bool scalesBf16, bool triggerCompletionAtEnd, cudaStream_t stream);

} // namespace kernels::minimaxM31Sm107::s2tail

TRTLLM_NAMESPACE_END
