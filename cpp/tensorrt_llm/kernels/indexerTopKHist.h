/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * Adapted from:
 * https://github.com/sgl-project/sglang/blob/d03c8cee8090bdfa63f6476c6f7e150ad4244f50/python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh
 * Adapted from:
 * https://github.com/sgl-project/sglang/blob/d03c8cee8090bdfa63f6476c6f7e150ad4244f50/python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk_impl.cuh
 * SPDX-FileCopyrightText: Copyright contributors to the sglang project
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include <cuda_runtime.h>

// ============================================================================
// DSA-indexer decode top-k kernel (v1: SELECTION ONLY).
//
// A port of the fused small-batch cluster top-k launch `topk_small_batch_kernel`
// (which internally dispatches Register4 / Streaming / TopKCluster<8> per row);
// see the file header above for upstream provenance.
//
// EXACT-MATCH CONTRACT with the stock `topKPerRowDecode` split/merge path
// (indexerTopK.cu): the launcher emits int32 *local* indices into `outIndices`
// (row-major, width `topK`), padding with -1, using the SAME per-row length
//   seq_len       = seqLens[rowIdx / next_n]
//   actual_kv_len = seq_len - next_n + (rowIdx % next_n) + 1
//   rowEnd        = actual_kv_len / compressRatio      (clamped to [0, numColumns])
// so that dsa.py and the downstream convertReqIndexToGlobal are unchanged.
// It does NOT fuse the page-table gather (that is v2).
// ============================================================================

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Returns true iff invokeIndexerTopKDecodeHist supports this shape. The stock
// dispatcher uses this to decide whether the TRTLLM_DSA_TOPK_HIST env
// override may take over: cluster-capable SM (>= 90 and < 120, i.e. Hopper /
// datacenter Blackwell -- the cluster tier needs thread-block clusters, absent
// on SM120/121 workstation Blackwell), topK in {512,1024,2048}, unit inner
// stride with stride0 a multiple of 4 (16-byte vectorized loads), and
// compressRatio in {1,4}. numRows must be small enough to map one 8-block
// cluster per row (bounded by the small-batch launch geometry). Any unsupported
// shape/device falls back to the stock path.
bool indexerTopKDecodeHistSupported(int numRows, int topK, int stride0, int stride1, int compressRatio);

// Raw-pointer launcher for the selection-only decode top-k.
//
// logits        : [numRows, >=numColumns] fp32, row stride `stride0` (elements),
//                 unit inner stride assumed (16-byte vectorized loads).
// seqLens       : [numRows / next_n] int32 per-batch sequence lengths.
// outIndices    : [numRows, topK] int32, local indices, -1 padded (output).
// numRows        = batch * next_n (one logical top-k per row).
// numColumns     = allocated per-row width; the recomputed per-row length is
//                  clamped to this to bound logits reads.
// stride0        = row stride of `logits` in elements (must be a multiple of 4).
// next_n         = MTP staggering factor (>=1).
// topK           = output width == index_topk (512 / 1024 / 2048).
// compressRatio  = 1 (DSv3.2) or 4 (DSv4 overlap compressor).
// usePDL         = caller-provided (pass tensorrt_llm::common::getEnvEnablePDL()).
//
// fuseTransform is hard-disabled in v1: output is raw local indices.
void invokeIndexerTopKDecodeHist(float const* logits, int const* seqLens, int* outIndices, int numRows, int numColumns,
    int stride0, int next_n, int topK, int compressRatio, bool usePDL, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
