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
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

// The GLM-5.2 decode shape at TP16 the flash-decode sparse MLA serves: 4 query heads per rank sharing one latent KV
// head, QK head dim 576 (512 latent + 64 rope), latent V dim 512, a TopK of at most 2048 selected tokens per query
// token, and the v_b BMM (v_head_dim 256) folded into the output.
constexpr int32_t kGlmSparseMlaNumHeads = 4;
constexpr int32_t kGlmSparseMlaHeadDimQk = 576;
constexpr int32_t kGlmSparseMlaHeadDimV = 512;
constexpr int32_t kGlmSparseMlaMaxTopK = 2048;
constexpr int32_t kGlmSparseMlaVHeadDim = 256;

struct GlmSparseMlaFlashDecodeParams
{
    // [numTokensQ, 4, 576] E4M3.
    void const* q;
    // KV pool base: pool token row r holds 576 E4M3 values at byte offset r * 576.
    void const* kvPool;
    // [numTokensQ, indicesStride] selected tokens of each query token (pool rows, or positions with the remap);
    // negative entries are masked.
    int32_t const* indices;
    // [numTokensQ / seqLenQ] KV length of each sequence, this step's tokens included.
    int32_t const* seqLensKv;
    // Device scalar that maps Q.K^T to the base-2 softmax domain.
    float const* scaleSoftmaxLog2;
    // Device scalar that scales the attention output.
    float const* outputScale;
    // [4, 256, 512] BF16 v_b_proj.
    void const* vbProj;
    // out[t][h * 256 + v] = sum_d O[t][h][d] * vbProj[h][v][d] (BF16, outStride elements between query tokens), where
    // O is the attention output rounded to BF16.
    void* out;
    int64_t outStride;
    int32_t numTokensQ;
    // Query tokens per sequence: query token t of a sequence attends to the first
    // min(seqLensKv - (seqLenQ - 1 - t), topK) of its indices.
    int32_t seqLenQ;
    int32_t indicesStride;
    int32_t topK;
    // Optional in-kernel remap of the indices, with the arithmetic of convertReqIndexToGlobal: with remapBlockTable
    // set, indices hold token positions in the query token's sequence, and position p selects pool token row
    // remapBlockTable[s * remapBlockTableStride + p / remapTokensPerBlock] * remapStrideFactor
    // + p % remapTokensPerBlock + remapLayerOffset, where s = tokenQ / seqLenQ is the sequence (the row of seqLensKv
    // and of the block table). A negative position, a block at or beyond remapMaxBlocks or a negative table entry
    // masks the entry.
    int32_t const* remapBlockTable;
    int64_t remapBlockTableStride;
    int32_t remapMaxBlocks;
    int32_t remapTokensPerBlock;
    int32_t remapStrideFactor;
    int32_t remapLayerOffset;
    // With the remap, true when the indices were written by an earlier layer's top-k (a layer that shares it): the
    // kernel then reads the indices, the block table and the KV length before its grid-dependency wait and gathers the
    // rows of the positions before this step's tokens (position < seqLensKv - seqLenQ, appended by earlier steps)
    // there too; the rows of this step's tokens, written by the KV append it depends on, after the wait.
    bool gatherBeforeWait;
};

// Generation-only static-token sparse MLA with E4M3 Q and KV and the v_b BMM folded in, as flash decoding over the
// gathered keys. Every query token gets one cluster of 16 CTAs, and CTA s owns indices [128 s, 128 s + 128):
// - each warp gathers 16 token rows with one bulk copy per row onto its own mbarrier;
// - S = Q K^T and P V run as FP16 mma.sync on E4M3 values unpacked exactly to FP16, with fp32 accumulation; the MMAs
//   run 8 head columns, the 4 query heads in columns 0-3 and zero Q rows in 4-7, whose outputs are not stored;
//   P = exp2(S * scale - max) is stored as E4M3 scaled by 448, as the TRTLLM-Gen generation kernels feed their second
//   GEMM, and the query fragments are unpacked once per warp;
// - every CTA sends output slice r of its unnormalized partial output (one 16-column block at a time, during P.V), its
//   max and its sum straight into the shared memory of CTA r of the cluster (st.async), and CTA r merges the 16
//   partials of its 32 output columns;
// - CTA r also owns v in [16 r, 16 r + 16) of every head: it all-gathers the row's merged BF16 output (st.async),
//   multiplies it with its v_b_proj slice (64 KiB, bulk-copied once every token row has landed) on mma.sync BF16, and
//   writes those 64 values of out.
// Requires the SM100 family, numTokensQ * 16 CTAs of 170 KiB shared memory and clusters of 16 CTAs. PDL: the kernel
// reads global memory only after its grid-dependency wait (except the gatherBeforeWait reads above) and lets its
// dependents launch once its partial outputs are sent.
void invokeGlmSparseMlaFlashDecode(GlmSparseMlaFlashDecodeParams const& params, cudaStream_t stream);

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
