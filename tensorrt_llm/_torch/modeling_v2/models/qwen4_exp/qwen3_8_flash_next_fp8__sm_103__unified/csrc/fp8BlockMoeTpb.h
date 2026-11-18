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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::fp8_block_moe_tpb
{

//! Decode routed experts over 128-row intermediate blocks (e4m3, fp32 128x128 weight and 1x128 activation scales).
//! Block j of a rank is one (expert, intermediate block) pair, block-major:
//!   w1 [blocks, 256, 2560] e4m3 [128 up rows ; 128 gate rows], s1 [blocks, 2, 20] fp32 their block scales;
//!   w2 [blocks, 2560, 128] e4m3 the block's down_proj columns, s2 [blocks, 20] fp32 their block scales.
//! Expert e's block b lives on attention-TP rank (e + b) % 4: rank r stores block b < 4 of e at j = e and block 4
//! (when e % 4 == r) at j = 512 + e / 4, so each rank holds 640 blocks and at least one block of every expert.
constexpr int kHidden = 2560;
constexpr int kExperts = 512;
constexpr int kMaxRows = 8;
constexpr int kTpbRanks = 4;
constexpr int kTpbBlocks = 640; //!< blocks a rank holds (512 * 5 / 4)
constexpr int kShItems = 5;     //!< work items of the folded shared-expert slice
constexpr int kShItemCols = 32; //!< intermediate columns of one shared item

//! Arguments of one TPB launch (layouts above; T = numRows).
struct TpbParams
{
    void const* logits; //!< [T, 512] bf16
    uint8_t const* xq;  //!< [T, 2560] e4m3
    float const* xsf;   //!< [20, xsfStride] fp32: the 1x128 scales of xq, transposed
    int xsfStride;      //!< row stride of xsf, in elements
    uint8_t const* w1;  //!< [640, 256, 2560] e4m3
    float const* s1;    //!< [640, 2, 20] fp32
    uint8_t const* w2;  //!< [640, 2560, 128] e4m3
    float const* s2;    //!< [640, 20] fp32
    int rank;           //!< this rank of kTpbRanks
    int numRows;        //!< 1..8
    //! Optional fold of this rank's TP slice of the bf16 shared expert (shBlocks = 5 items of 32 columns; 3..8 rows):
    //! the partial also accumulates sigmoid(gate logit) * the slice's down projection in fp32 before the bf16 rounding.
    void const* shX;  //!< [T, 2560] bf16: the MoE site's rows
    void const* shGu; //!< [5, 64, 2560] bf16: item sb, 8 groups of 4 columns, each [gate rows (4) ; up rows (4)]
    void const* shGl; //!< [2560] bf16: the gate-logit row
    void const* shDn; //!< [5, 2560, 32] bf16: down_proj's columns of item sb
    int shBlocks;     //!< kShItems with the shared-expert fold, else 0
    void* out;        //!< [T, 2560] bf16: this rank's partial (the TP sum of the ranks' partials is the layer)
    void* workspace;  //!< tpbWorkspaceBytes(), zeroed once; its grid counter only grows (one per device)
};

//! Bytes of the per-device workspace launchTpb takes.
size_t tpbWorkspaceBytes();

//! One launch of 8-CTA clusters (up to 2 rows) or 4-CTA clusters (3..8 rows).
void launchTpb(TpbParams const& params, bool pdl, cudaStream_t stream);

} // namespace kernels::fp8_block_moe_tpb

TRTLLM_MV2_NAMESPACE_END
