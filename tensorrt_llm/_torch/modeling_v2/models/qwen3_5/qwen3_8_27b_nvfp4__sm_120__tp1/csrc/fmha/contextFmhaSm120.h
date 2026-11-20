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

// The target's SM120 context FMHA: e4m3 Q/K/V, BF16 output, head size 256, causal, Q_PAGED_KV, the tiled no-loop
// flash kernel (64-row q-tiles, 128-key KV tiles) with the base-2 softmax and the register-direct O store, plus the
// last q-tile KV split. Its sources are target-private copies (fmhaSm120*.cuh, contextFmhaLastTileSplit.cuh) of the
// pieces of fmha_v2 it changes; everything else is the stock fmha_v2 (cpp/kernels/fmha_v2/src). Users of this header
// need that directory on the include path and USE_DEMO_BERT_PARAMS=1 (see the family CMakeLists.txt).

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <fused_multihead_attention.h>

#include <cstddef>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

// The largest batch the last q-tile split runs for.
constexpr int kContextFmhaSm120MaxBatch = 3;
// The most pieces the split uses in total; each piece needs 128 threads x (32 + 1) float4 of FP32 accumulators and
// row max / sum.
constexpr int kContextFmhaSm120MaxPiecesTotal = 384;
// The workspace of the split: 25 952 256 bytes.
constexpr size_t kContextFmhaSm120SplitWorkspaceBytes = size_t{kContextFmhaSm120MaxPiecesTotal} * 128 * (32 + 1) * 16;
static_assert(kContextFmhaSm120SplitWorkspaceBytes == 25952256, "");

// Runs the context FMHA for every s >= 1 (always the tiled kernel). params must describe that configuration: d = dv =
// 256, Q_PAGED_KV (q_ptr / q_stride_in_bytes, paged_kv_cache, k/v_stride_in_bytes = tokens_per_block * 256),
// cu_q_seqlens / cu_kv_seqlens, BF16 o_ptr, scale_bmm1_d (only [0] is read: the kernel multiplies by log2(e) itself)
// and scale_bmm2_d; s = the largest q length of the batch.
//
// splitWorkspace (nullable: never split) holds kContextFmhaSm120SplitWorkspaceBytes. With it, 1 to
// kContextFmhaSm120MaxBatch unpadded sequences without ALiBi / sinks / softmax stats / sage / skip-softmax / softcap,
// and a grid where dropping the last q-tile saves a whole wave, run the last q-tile as KV pieces plus a fixed-order
// merge. skipLeadingQTiles (only honored on the split path) then leaves q-tiles 0 .. Q-2 unwritten; the caller must
// read only rows of each sequence's last q-tile.
//
// Throws (TLLM_CHECK) on a launch error.
void launchContextFmhaSm120(bert::Fused_multihead_attention_params_v2 const& params, int multiProcessorCount,
    void* splitWorkspace, bool skipLeadingQTiles, cudaStream_t stream);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
