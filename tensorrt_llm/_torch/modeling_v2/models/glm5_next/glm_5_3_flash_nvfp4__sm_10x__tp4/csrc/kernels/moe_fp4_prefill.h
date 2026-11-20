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
// Fused NVFP4 MoE experts for prefill-sized steps (see moe_fp4_prefill.cu): the grouped-GEMM path's
// gate / up GEMM, SwiGLU + NVFP4 quantization and down GEMM in one persistent launch.
#pragma once

#include "glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/csrc/common.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace glm
{
namespace k
{

constexpr int kMoePfMaxGroups = 320;
constexpr int kMoePfMaxItems = 4096;
// counters (int): [0] work queue, [1] finished CTAs, then per work item (expert, 128-row chunk) the
// bitmask of its hidden-row feature blocks stored
constexpr int kMoePfCtrSize = 2 + kMoePfMaxItems;
// hidden-row scale rows per work item (an item holds up to 208 rows)
constexpr int kMoePfHsRows = 256;

struct MoePrefillArgs
{
    // moe_permute_rows' outputs: offsets [G + 1] (rows per group), pad [G + 1] (rows padded to 128
    // per group: group g's scales at xs + pad[g] * H / 16)
    int const* offsets = nullptr;
    int const* pad = nullptr;
    // the permuted NVFP4 token rows [x_rows, H / 2] and their scales (CUTLASS layout per group)
    uint8_t const* xq = nullptr;
    uint8_t const* xs = nullptr;
    int x_rows = 0;
    // NVFP4 w13 [G, 2 Il, H / 2] (gate / up interleaved by 64, kGu64), scales [G] x nvfp4_sf_bytes(2 Il,
    // H), alpha13 [G]; w2 [G, H, Il / 2], scales, alpha2 [G]
    uint8_t const* w13 = nullptr;
    uint8_t const* sf13 = nullptr;
    float const* alpha13 = nullptr;
    uint8_t const* w2 = nullptr;
    uint8_t const* sf2 = nullptr;
    float const* alpha2 = nullptr;
    // scratch: the hidden rows [h_rows, Il / 2] (grouped row order) and their scales, per work item
    // (an expert's rows, or a 128-row chunk of a larger expert) the CUTLASS scale layout of
    // [kMoePfHsRows, Il] rows: hs [hs_items] x nvfp4_sf_bytes(kMoePfHsRows, Il)
    uint8_t* hq = nullptr;
    uint8_t* hs = nullptr;
    int h_rows = 0;
    // an upper bound of the step's grouped rows (offsets[G]; 0: h_rows): the step has at most
    // moe_fp4_prefill_items(rows, G) work items, which the counters and hs must hold
    int rows = 0;
    int hs_items = 0;
    bf16* y = nullptr;  // [rows, H] the experts' outputs per grouped row
    int* ctr = nullptr; // [kMoePfCtrSize] zero-initialized; restored to zero by the kernel
    int H = 0, Il = 0, G = 0;
    float limit = 0.f;  // SwiGLU clamp
    float gs2 = 1.f;    // global scale of the hidden rows' NVFP4 quantization
};

// the work items' upper bound for rows routed rows over G groups: ceil(rows / 128) + G
inline int moe_fp4_prefill_items(int rows, int G)
{
    return (rows + 127) / 128 + G;
}

// ctas: persistent CTAs (one per SM)
void moe_fp4_prefill(MoePrefillArgs const& a, int ctas, cudaStream_t st);

} // namespace k
} // namespace glm

TRTLLM_MV2_NAMESPACE_END
