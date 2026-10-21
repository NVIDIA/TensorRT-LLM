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

//! Kimi K3 KDA multi-token speculative verify on the packed token layout, fused with the gated output RMSNorm and the
//! 1x128 MXFP8 quantization of its output: the contract of trtllm::kda_mtp_decode_fp8 with packed_token_layout = True
//! and fuse_output_norm = True. K = V = 128, conv width 4, numSpec + 1 new-token rows per request (row
//! b * (numSpec + 1) + j), slot-indexed accepted-draft counts, replay caches of numSpec rows per slot. Not
//! bit-identical to that op (different summation order, fast-math transcendentals); the state is committed with
//! round-to-nearest-even.
//!
//! The parameter block describes the op's tensors; any kernel that honors the contract can replace the one behind
//! invokeKimiK3KdaMtpVerifyKf (see isKimiK3KdaMtpVerifyKfSupported for what the current one accepts). All strides are
//! in elements; head blocks of 128 are contiguous.
struct KimiK3KdaMtpVerifyKfParams
{
    //! New-token inputs (bf16), row t = b * (numSpec + 1) + j: x_q / x_k / g [T, H * 128], x_v / onorm_g
    //! [T, H * 128] and beta [T, H], each with its own row stride.
    void const* xQ = nullptr;
    void const* xK = nullptr;
    void const* xV = nullptr;
    void const* g = nullptr;
    void const* beta = nullptr;
    void const* onormG = nullptr;
    int64_t xQRowStride = 0;
    int64_t xKRowStride = 0;
    int64_t xVRowStride = 0;
    int64_t gRowStride = 0;
    int64_t betaRowStride = 0;
    int64_t onormGRowStride = 0;

    float const* wQ = nullptr;                  //!< short-conv weights [H * 128, 4], contiguous
    float const* wK = nullptr;                  //!< [H * 128, 4]
    float const* wV = nullptr;                  //!< [H * 128, 4]
    float const* logA = nullptr;                //!< [H]
    float const* dtBias = nullptr;              //!< [H * 128]
    float const* onormWeight = nullptr;         //!< [128]

    int32_t const* ssmStateIndices = nullptr;   //!< [B] state-pool slot of each request
    int32_t const* numAcceptedTokens = nullptr; //!< [>= pool] accepted drafts of the previous round, by slot
    int64_t numAcceptedTokensStride = 1;        //!< its slot stride (the hybrid cache keeps a count block per slot)

    //! Conv windows (bf16) [pool, H * 128, >= 3 + numSpec], channel stride 1. Mutated.
    void* convQ = nullptr;
    void* convK = nullptr;
    void* convV = nullptr;
    int64_t convQSlotStride = 0;
    int64_t convQPosStride = 0;
    int64_t convKSlotStride = 0;
    int64_t convKPosStride = 0;
    int64_t convVSlotStride = 0;
    int64_t convVPosStride = 0;

    //! Recurrent state [pool, H, V, K] (V-first): statePoolSize slots of contiguous [H, V, K], stateSlotStride
    //! elements apart (the deployed pool pads the slots). Mutated.
    void* state = nullptr;
    bool stateIsBf16 = false;
    int64_t stateSlotStride = 0;
    int statePoolSize = 0;

    //! Replay caches of the drafts: k (bf16) / g (fp32) / v (bf16) [pool, numSpec, H * 128], beta (bf16)
    //! [pool, numSpec, H]; (slot, row) strides, last dimension contiguous. Mutated.
    void* kCache = nullptr;
    float* gCache = nullptr;
    void* vCache = nullptr;
    void* betaCache = nullptr;
    int64_t kCacheSlotStride = 0;
    int64_t kCacheRowStride = 0;
    int64_t gCacheSlotStride = 0;
    int64_t gCacheRowStride = 0;
    int64_t vCacheSlotStride = 0;
    int64_t vCacheRowStride = 0;
    int64_t betaCacheSlotStride = 0;
    int64_t betaCacheRowStride = 0;

    //! Output [T, H * 128], contiguous: E4M3 plus UE8M0 1x128 scales in the R128c4 layout of
    //! trtllm::kda_mtp_decode_fp8 (ceil(T / 128) * 128 * H * 4 bytes; rows >= T are not written) when outScale is set,
    //! else BF16.
    void* out = nullptr;
    uint8_t* outScale = nullptr;

    int numRequests = 0;                 //!< B
    int numHeads = 0;                    //!< H
    int numSpec = 0;                     //!< drafts per request
    float lowerBound = -5.0f;            //!< gate lower bound
    float qScale = 0.08838834764831845f; //!< 128^-0.5
    float onormEps = 1e-5f;
};

//! True when invokeKimiK3KdaMtpVerifyKf supports the parameters. The current kernel needs an SM10x GPU, a BF16 state,
//! MXFP8 output, numSpec 3 or 5, the Kimi K3 constants (lower bound -5, q scale 128^-0.5, output-norm eps 1e-5) and the
//! alignment of its vector and bulk accesses (see the .cu). Explains a failure in *reason when given.
bool isKimiK3KdaMtpVerifyKfSupported(KimiK3KdaMtpVerifyKfParams const& params, char const** reason = nullptr);

//! With enablePdl the kernel is launched with programmatic stream serialization. It waits for its predecessor before
//! its first global memory access, so any input may be written by the kernel launched right before it.
void invokeKimiK3KdaMtpVerifyKf(KimiK3KdaMtpVerifyKfParams const& params, bool enablePdl, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
