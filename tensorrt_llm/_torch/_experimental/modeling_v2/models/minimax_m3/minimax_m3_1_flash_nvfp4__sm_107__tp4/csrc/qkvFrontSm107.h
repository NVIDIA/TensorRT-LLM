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
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! The MiniMax-M3.1 sparse-layer front in one launch (SM107, decode, at most 8 tokens): the MoE boundary all-reduce
//! (prepushed 2 / 3 consumer of m3_mc_allreduce_norm's exchange) + residual + RMSNorm + MXFP8, the QKV MXFP8 GEMV and
//! the NVFP4 QKV + indexer producer epilogue (q / k / index norms, RoPE, FP8 q and index-q, NVFP4 K / V and FP8
//! index-K cache insert). See qkvFrontSm107.cu.
struct QkvFrontParams
{
    // The boundary: this rank's unicast view of the all-reduce exchange (int32 words, re-armed after reading), its
    // per-(row, 768-column slice) call-parity flags [maxt * 8], the bucket geometry and the push mode.
    int32_t* wsUc{nullptr};
    int32_t* flags{nullptr};
    int32_t rank{0};
    int32_t world{0};
    int32_t maxt{0};
    int32_t prepushed{2};                     //!< 2: the MoE pushed x into set A; 3: x in set A and x2 in set B.
    __nv_bfloat16 const* residual{nullptr};   //!< [numTokens, 6144]
    __nv_bfloat16 const* normWeight{nullptr}; //!< [6144], as m3_mc_allreduce_norm takes it
    float eps{0.F};
    __nv_bfloat16* outResidual{nullptr};      //!< [numTokens, 6144]
    __nv_bfloat16* outNormed{nullptr};        //!< optional [numTokens, 6144]
    // The front's own boundary -> GEMV hand-off (qkvFrontWorkspaceBytes): MXFP8 rows and scales in two parities
    // (unwritten bytes 0x80 / 0xFF) and one call counter per CTA.
    uint8_t* xq{nullptr};
    uint8_t* xsf{nullptr};
    int32_t* ctr{nullptr};
    // The QKV projection: E4M3 [heads * 128, 6144] (K contiguous) and its 128x4-swizzled UE8M0 scales.
    void const* weight{nullptr};
    uint8_t const* weightScale{nullptr};
    // The producer epilogue: producerPw4's operands (minimaxM3Sm107Tp4/producerPw4.h).
    __nv_fp8_e4m3* qOut{nullptr};      //!< [numTokens, headsQ, 128]
    __nv_fp8_e4m3* indexQOut{nullptr}; //!< [numTokens, 1, 128]
    uint8_t* kvData{nullptr};
    uint8_t* kvScale{nullptr};
    __nv_fp8_e4m3* indexK{nullptr};
    int32_t const* outCacheLoc{nullptr};
    float const* kvQuantScale{nullptr};
    int64_t dataPageStride{0}, dataPlaneStride{0}, dataHeadStride{0}, dataTokenStride{0};
    int64_t scalePageStride{0}, scalePlaneStride{0}, scaleHeadStride{0};
    int64_t indexPageStride{0}, indexTokenStride{0}, numPages{0};
    float peps{0.F};
    __nv_bfloat16 const* qWeight{nullptr};
    __nv_bfloat16 const* kWeight{nullptr};
    __nv_bfloat16 const* indexQWeight{nullptr};
    __nv_bfloat16 const* indexKWeight{nullptr};
    float const* rotaryCosSin{nullptr}; //!< fp32 [positions, 2, 32]
    int32_t const* positionIds{nullptr};
    int32_t numTokens{0};
    int32_t headsQ{0}; //!< query heads per rank; the projection has headsQ + 4 heads (k, v, index-q, index-k)
};

constexpr int32_t kQkvFrontHidden = 6144;
constexpr int32_t kQkvFrontMaxTokens = 8;

//! Bytes of the front's hand-off workspace: xq (fill 0x80), xsf (fill 0xFF), and int32 call counters (fill 0) for
//! up to kQkvFrontMaxCtas CTAs. One workspace per split: the counters follow the grid.
constexpr int64_t kQkvFrontXqBytes = 2LL * 16 * kQkvFrontHidden;
constexpr int64_t kQkvFrontXsfBytes = 2LL * ((kQkvFrontHidden / 128) * 256 + 512);
constexpr int32_t kQkvFrontMaxCtas = 256;

//! Whether the launcher takes `split` (the GEMV's K split = the cluster size): 8 is bitwise the served 3-kernel chain
//! (m3_mc_allreduce_norm cluster kernel -> m3_mx_gemv (8, 8) -> producerPw4); 4 is the reference kernel's geometry (K
//! split 4, four accumulators): exact-class, not bitwise.
bool isQkvFrontSplitSupported(int split);

//! How many of the front's clusters (at `split`) the current device holds at once (cudaOccupancyMaxActiveClusters),
//! or -1 when the kernel cannot run here. The launcher refuses a grid of more clusters (headsQ + 4) than that: its
//! GEMV clusters wait on the boundary clusters' hand-off, so all of them must be able to be resident together.
int qkvFrontMaxActiveClusters(int split);

//! Launch the front: (headsQ + 4) * split CTAs of 512 threads in clusters of `split`. `pdl`: programmatic stream
//! serialization; `earlyTrigger`: release the dependents before the GEMV (their reads follow their own grid wait).
cudaError_t invokeQkvFront(QkvFrontParams const& params, int split, bool pdl, bool earlyTrigger, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
