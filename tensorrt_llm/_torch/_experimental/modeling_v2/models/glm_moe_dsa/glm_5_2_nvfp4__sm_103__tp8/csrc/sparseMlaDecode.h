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
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace sparse_mla_decode
{

// Sparse MLA generation attention over an FP8 (e4m3) latent KV pool.
//
// Each query row attends to the pool rows listed in its `kTopK`-entry index list. A pool row
// holds the 512-dim latent followed by the 64-dim rotary part, 576 e4m3 values in total; the
// value vector is the first 512 values of the same row. Eight query heads share the row.
//
// Rules of the TRT-LLM generation FMHA for this layout, with the one difference marked below:
// * Index-list position j of query row r is used only when j < min(kTopK, kvLen - (seqLenQ - 1 - r % seqLenQ)),
//   where kvLen counts every token of the sequence including this step's rows. Repeated pool rows
//   count once per occurrence.
// * Difference: a negative entry at a used position is skipped. The native kernel does not check
//   it and reads the pool row that -1 addresses. The native producers (indexer top-k, local-to-global
//   remap) write -1 only at positions past the causal length, so the outputs agree on their inputs.
// * Scores are FP32 dot products of the e4m3 query and key times bmm1Scale[1] (log2 domain).
// * Probabilities relative to a running maximum are multiplied by 448 and rounded to e4m3 before
//   the P*V product; row sums use the unrounded FP32 probabilities.
// * A row without a usable entry produces zeros.
//
// Programmatic dependent launch (PDL), active when TRTLLM_ENABLE_PDL allows it:
// * The kernel is launched with programmatic stream serialization. Before griddepcontrol.wait it
//   only initializes its own shared-memory barriers and arrives on its cluster barrier; it accesses
//   no global memory there, so it may start while the preceding kernel still writes its inputs.
// * After the wait it reads q, indices, seqLensKv, bmm1Scale and bmm2Scale and issues the
//   asynchronous gathers of the pool rows. It then executes griddepcontrol.launch_dependents, before
//   those gathers complete and before any compute, so the next kernel of the stream can become
//   resident while this one still runs.
// * Its only global write is out, stored at the end of each CTA.
// * Obligation of a PDL dependent: execute griddepcontrol.wait (cudaGridDependencySynchronize)
//   before reading out and before writing any buffer read here (q, indices, seqLensKv, the scales,
//   the pool rows). Work before that wait may access only memory this kernel neither reads nor
//   writes, such as weights streamed into shared memory or L2; such work competes with this
//   kernel's gathers for memory bandwidth and SM resources.

int constexpr kNumHeads = 8;
int constexpr kHeadDimQk = 576;
int constexpr kHeadDimV = 512;
int constexpr kTopK = 2048;

struct SparseMlaDecodeParams
{
    // [numRows, kNumHeads, kHeadDimQk] e4m3 query, rows contiguous.
    void const* q;
    // Base of the FP8 latent pool; pool row i starts at kvPool + i * kHeadDimQk bytes.
    void const* kvPool;
    // [numRows, indicesRowStride] int32 pool rows.
    int32_t const* indices;
    int64_t indicesRowStride;
    // [2] device scales; element 1 is the dequantization scale times the softmax scale times log2(e).
    float const* bmm1Scale;
    // [1] device scale applied to the normalized P*V output.
    float const* bmm2Scale;
    // [numRows / seqLenQ] KV lengths including this step's rows.
    int32_t const* seqLensKv;
    int32_t seqLenQ;
    // [numRows, kNumHeads, kHeadDimV] BF16 output.
    __nv_bfloat16* out;
    int32_t numRows;
};

// Dynamic shared memory needed by one CTA.
size_t getSparseMlaDecodeSmemBytes();

// Whether the current device can launch the kernel (architecture and 16-CTA clusters).
bool isSparseMlaDecodeDeviceSupported(int device);

void invokeSparseMlaDecode(SparseMlaDecodeParams const& params, cudaStream_t stream);

#ifdef GLM_MOE_DSA_SPARSE_MLA_DECODE_TIMING
// Development builds only: per-CTA phase timeline of the most recent launches.
int constexpr kTimelineMaxCtas = 4096;
int constexpr kTimelineSlots = 13;
void copySparseMlaDecodeTimeline(unsigned long long* host);
#endif

} // namespace sparse_mla_decode

TRTLLM_MV2_NAMESPACE_END
