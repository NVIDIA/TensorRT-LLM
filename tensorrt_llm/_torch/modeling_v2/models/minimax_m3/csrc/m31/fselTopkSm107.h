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
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3
{

//! Largest row (blocks per query) the FSEL top-16 select covers: every block below min(n_valid, numBlocks, 9216) is
//! selectable (the port's cap; the original page_topk_nt512r4 stops at 8192 pages).
constexpr int32_t kFselTopkMaxBlocks = 9216;

//! Rows above this many blocks run over a 4-CTA cluster under variant 7 (PORT(i8 MS4)); shorter rows are variant 6.
constexpr int32_t kFselTopkMsMinBlocks = 4096;

//! The launch shapes: 5 (256-thread CTAs), 6 (512-thread CTAs, the traced instance), 7 (variant 6, and every row of
//! more than kFselTopkMsMinBlocks valid blocks split over a cluster of 4 CTAs that merge their top-16 over DSMEM; the
//! same table bit for bit; when numBlocks <= kFselTopkMsMinBlocks no row can split and the launch is variant 6).
constexpr int32_t kFselTopkVariantMs4 = 7;

//! FSEL exact top-16 block select (reference page_topk_nt512r4 fsel_kernel<NT, 4, ST, 4>, ported), one
//! CTA per (query, KV head) row, over per-KV-head scores [numKvHeads, numBlocks, totalQ] (element strides given).
//!
//! Per row: the first initBlocks and the last localBlocks of the row's n_valid blocks (counted from the unclamped
//! count) are always selected; the others are ranked by (score desc on the fp32 bit order, block asc). The 16 ids
//! (pageTable[row's token * pageTableRowStride + block]; the M3 target passes the identity table with a zero row
//! stride) are written in ascending block order, -1 tail when the row has at most 16 valid blocks.
//! out[query * outQueryStride + head * outHeadStride + 0..15] is fully written.
//!
//! variant 5 launches 256 threads per CTA, 6 launches 512 (the traced fsel_kernel<512, 4, false, 4>); rows of at most
//! 4096 blocks run the 256-thread body either way. variant 7 (kFselTopkVariantMs4) launches 4 x 512-thread CTAs per row
//! in clusters of 4 when numBlocks > 4096 (fsel_ms_kernel): rows above 4096 valid blocks rank a quarter per CTA and
//! merge in the cluster's first CTA, shorter rows run the variant 6 path in that CTA; stamps then hold 4 x rows rows.
//! triggerPhase releases the dependent grid right after this grid's dependency wait (0, the original), after the row's
//! scores are in registers (1) or at exit (2). diag (4 ints: fast32, fast64, radix fallback, short-row copy) and stamps
//! ([rows][8] %globaltimer) are optional instrumentation.
cudaError_t invokeFselTopk(float const* scores, int64_t headStride, int64_t blockStride, int64_t queryStride,
    int32_t const* nValidBlocks, int32_t const* pageTable, int64_t pageTableRowStride, int32_t numBlocks, int32_t* out,
    int64_t outQueryStride, int64_t outHeadStride, int32_t numKvHeads, int32_t totalQ, int32_t initBlocks,
    int32_t localBlocks, int32_t variant, bool usePdl, int32_t triggerPhase, int32_t* diag, int64_t* stamps,
    cudaStream_t stream);

} // namespace minimaxM3
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
