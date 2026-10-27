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

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp8
{
//! The split-KV merge of the MiniMax-M3 TP2 full-context decode (PrimTS split-KV decode, separate-reduction scratch)
//! straight to BF16. See fullCtxReduce.cu.
namespace fullCtxReduce
{

constexpr int kHeadDim = 128;
constexpr int kMaxSplits = 128;

struct Params
{
    void const* partialO;      // [B, Hkv, S, R, 128] normalized 16-bit partial O per split
    float const* partialStats; // [B, Hkv, S, R] log2-LSE per split
    int const* seqLensKv;      // [B]
    __nv_bfloat16* out;        // [B, SQ, Hkv * hR, 128]
    int batch;
    int numKvHeads;
    int splits;        // S, the planned split capacity (<= kMaxSplits)
    int rows;          // R = hR * SQ, rows [token, local head]
    int seqLenQ;       // SQ
    int headsPerKv;    // hR
    int qTokensPerCta; // query tokens one producer CTA owns (grouped Q)
    int tileKv;        // producer K/V tile
    int instsKv;       // producer K/V instructions per loop step
    bool causal;
    bool bf16Partial;
    uint8_t* q8 = nullptr; // optional E4M3 copy of out [B * SQ, Hkv * hR * 128] (the o_proj MXFP8 hand-off)
    uint8_t* sf = nullptr; // its UE8M0 block scales, 128x4-swizzled as trtllm::mxfp8_quantize(x, True) lays them out
    int numSfKTiles = 0;   // Hkv * hR * 128 / 128
};

//! Merges the active splits of every row in fp32 with one max rescale and writes BF16; with q8 set, also the MXFP8
//! copy of the BF16 rows (bit for bit trtllm::mxfp8_quantize of out). With pdl the launch is
//! programmatic: the grid waits on the producer (griddepcontrol.wait) before reading its scratch.
//!
//! PDL contract: griddepcontrol.wait is the kernel's first instruction, so nothing (scratch, lengths, out) is read or
//! written before every prerequisite grid has completed and flushed; a producer's early trigger
//! (griddepcontrol.launch_dependents) only lets this grid launch and wait sooner. The kernel triggers its own
//! dependents right after its wait: the next kernel's griddepcontrol.wait still waits for this whole grid, so its
//! post-wait code sees all of out, and its pre-wait prologue must not read out.
cudaError_t invokeFullCtxReduce(Params const& params, bool pdl, cudaStream_t stream);

} // namespace fullCtxReduce
} // namespace minimaxM3Sm107Tp8
} // namespace kernels

TRTLLM_NAMESPACE_END
