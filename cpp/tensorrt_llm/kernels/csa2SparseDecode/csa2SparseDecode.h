/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::csa2SparseDecode
{

//! Fused sparse decode attention over the packed DeepSeek-V4.1 CSA2 KV pools (SM100 family).
//!
//! For every query row (64 heads x 512 channels, BF16 or E4M3) the kernel gathers the selected SWA and main rows
//! directly from their packed pools, dequantizes them exactly to BF16, and computes softmax attention with an attention
//! sink in FP32. Invalid selections (-1, or rows outside the pool) are excluded; a row with no valid key yields zeros.
//! The output is BF16 [numTokens, 64, 512], or its MXFP8 grouped O-A projection input (outFp8).
//!
//! Usage: makeSchedule() once per token count (it depends only on shapes and the SM count, so it is static under CUDA
//! graph capture), a caller-owned workspace of getWorkspaceSize() bytes, then run() on a stream. The workspace needs no
//! initialization.
//!
//! Stream ordering: split tokens are tracked with per-device arrival counters (the module-scope device array
//! csa2SplitArrivals), which are zero at module load and which every launch leaves at zero. Every run() on one device
//! uses these counters, whatever its workspace, so all launches on a device must run in stream order (one stream, or
//! streams ordered by events), exactly like launches that share a workspace. Two launches running concurrently on one
//! device can make a combine kernel read partial results too early or wait forever.

constexpr int kNumHeads = 64;
constexpr int kHeadDim = 512;
//! Groups of the grouped O-A projection that the MXFP8 output feeds (kNumHeads / kOutputGroups heads each).
constexpr int kOutputGroups = 8;
//! SWA row: 512 x E4M3 + 16 x UE8M0 (one per 32 channels).
constexpr int kSwaRowBytes = 528;
//! Main row: 256 B of E2M1 pairs (low nibble = even channel) + 32 x E4M3 (one per 16 channels).
constexpr int kMainRowBytes = 288;
//! Required alignment of the workspace pointer.
constexpr size_t kWorkspaceAlignment = 256;

struct Csa2SparseDecodeParams
{
    int numTokens{0};                    //!< Query rows M.

    void const* q{nullptr};              //!< BF16 or E4M3 [M, 64, 512], unit channel stride, 16-B aligned.
    int64_t qTokenStride{0};             //!< In elements; 16-B multiple.
    int64_t qHeadStride{0};              //!< In elements; 16-B multiple.
    bool qFp8{false};                    //!< Q is E4M3 (converted exactly to BF16).

    void const* swaPool{nullptr};        //!< uint8 [swaRows, 528], contiguous rows, 16-B aligned.
    int64_t swaRows{0};
    void const* swaIndices{nullptr};     //!< int64 [M, swaTopk] physical rows (-1 = invalid), unit column stride.
    int64_t swaIndexStride{0};           //!< Row stride in elements; even.
    int swaTopk{0};                      //!< Even and positive.

    void const* mainPool{nullptr};       //!< uint8 [mainRows, 288], contiguous rows; unused when mainTopk == 0.
    int64_t mainRows{0};
    int32_t const* mainIndices{nullptr}; //!< int32 [M, mainTopk], unit column stride: physical rows, or logical
                                         //!< positions when mainPageTable is set. -1 = invalid.
    int64_t mainIndexStride{0};          //!< Row stride in elements; even.
    int mainTopk{0};                     //!< 0 (SWA-only layer) or a multiple of 64.

    //! Logical main selections. When set, the logical position l of token t is mapped in the kernel to the physical
    //! row (mainPageTable[r][l >> mainPageShift] << mainPageShift) | (l & (pageSize - 1)) with r = tokenRequests[t].
    //! It is valid only when 0 <= l < min(tokenVisible[t], mainMaxPositions), 0 <= r < mainPageTableRows,
    //! (l >> mainPageShift) < mainPageTableCols, the table entry is non-negative and the row lies in the pool.
    int32_t const* mainPageTable{nullptr}; //!< int32 [mainPageTableRows, mainPageTableCols], unit column stride.
    int64_t mainPageTableStride{0};        //!< Row stride in elements.
    int mainPageTableRows{0};
    int mainPageTableCols{0};
    int mainPageShift{0};                  //!< log2 of the main page size in rows.
    int64_t mainMaxPositions{0};
    int64_t const* tokenRequests{nullptr}; //!< int64 [M] page-table row of each query.
    int64_t const* tokenVisible{nullptr};  //!< int64 [M] visible logical main length of each query.

    float const* attnSinks{nullptr};       //!< FP32 [64].
    float softmaxScale{0.F};

    void* out{nullptr};        //!< BF16 [M, 64, 512], unit channel stride, 16-B aligned.
    int64_t outTokenStride{0}; //!< In elements; a multiple of 8.
    int64_t outHeadStride{0};  //!< In elements; a multiple of 8.

    //! With outFp8 the kernels write, instead of out, what projection.inverse_rope_mxfp8 makes of the BF16
    //! output for the grouped O-A projection: the last 64 channels of each head rotated back in GPT-J pairs at the
    //! token's position, then MXFP8 per 32 channels. Head h is head h % 8 of group h / 8.
    void* outFp8{nullptr};             //!< E4M3 [8, M, 8 * 512], contiguous, 32-B aligned.
    void* outScale{nullptr};           //!< UE8M0, per group [pad128(M), 128] in the CUTLASS 128x4 layout, padding
                                       //!< rows zeroed.
    int32_t const* positions{nullptr}; //!< int32 [M] RoPE positions.
    float const* ropeCosSin{nullptr};  //!< FP32 [positions, 64]: 32 cos then 32 sin per position, 16-B aligned.

    void* workspace{nullptr};          //!< getWorkspaceSize(schedule) bytes, kWorkspaceAlignment-aligned.
    bool enablePdl{false};             //!< Launch with programmatic dependent launch.
};

//! Static work split of one call: every CTA takes fullPerCta whole tokens; the KV blocks of the remaining tokens are
//! cut into remCtas contiguous chunks (makeSchedule picks remCtas as the remainder token count or a multiple of it),
//! and a token cut by a chunk boundary is merged by a combine kernel. With PDL, the combine kernel starts while the
//! attention kernel is still running and merges each split token once all its parts are out.
struct Csa2SparseDecodeSchedule
{
    int numCtas{0};
    int fullPerCta{0};
    int remCtas{0};
};

//! Work split for numTokens queries on numSms SMs. Depends only on its arguments.
Csa2SparseDecodeSchedule makeSchedule(int numTokens, int swaTopk, int mainTopk, int numSms);

//! Whether some token of the schedule is split across CTAs (run() then also launches the combine kernel).
bool scheduleHasSplit(int numTokens, Csa2SparseDecodeSchedule const& schedule);

//! Workspace bytes a call with this schedule needs.
size_t getWorkspaceSize(Csa2SparseDecodeSchedule const& schedule);

//! Workspace bytes that cover every schedule on a device with numSms SMs, for any token count.
size_t getMaxWorkspaceSize(int numSms);

//! Whether this build carries the kernel for the current device (an SM100-family GPU with a matching image).
bool isSupported();

//! Launches the attention kernel, plus the combine kernel when the schedule splits a token, on stream.
//! Allocation-free and free of host synchronization, so the launch can be captured in a CUDA graph. It must be
//! stream-ordered with every other run() on the device (see the stream-ordering note above).
void run(Csa2SparseDecodeParams const& params, Csa2SparseDecodeSchedule const& schedule, cudaStream_t stream);

} // namespace kernels::csa2SparseDecode

TRTLLM_NAMESPACE_END
