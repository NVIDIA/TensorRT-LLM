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

namespace kernels
{

//! MiniMax-M3 MSA KV-length correction in one single-CTA launch, for all int32 operands:
//!
//!   kvTrue[b]    = min(kvLens[b], staged[b])           -> seqLensOut[b]; cuKv = exclusive prefix sum (optional)
//!   pos[t]       = kvTrue[r] - qo[r] + qIntra[t]       (r = qBatchRow[t])
//!   outLoc[t]    = table[r, min(max(pos, 0) / page, ncols - 1)] * page + max(pos, 0) % page
//!   nValid[t]    = (max(pos + 1, 0) + page - 1) / page (optional)
//!
//! cuKv and nValid may be null. table is row-major with row stride tableLd (elements) and ncols columns.
cudaError_t invokeMinimaxM3MsaKvUpdate(int32_t const* kvLens, int32_t const* staged, int32_t const* qo,
    int32_t const* qBatchRow, int32_t const* qIntra, int32_t const* table, int64_t tableLd, int32_t ncols, int32_t page,
    int32_t batch, int32_t totalQ, int32_t* seqLensOut, int32_t* cuKv, int32_t* outLoc, int32_t* nValid,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
