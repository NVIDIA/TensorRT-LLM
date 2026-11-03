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

#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::csa2SparseContext
{

//! FP8 sparse context attention of DeepSeek-V4.1 CSA2 over the shared KV bank (SM100 family).
//!
//! Every query token (64 heads x 512 channels) attends to the bank rows its index row lists: the first lens[t]
//! entries of indices[t], clamped to [1, maxKeys]; entries outside [0, bankRows) read row 0, which the bank keeps
//! zero. Q and the bank are E4M3. Logits are (Q K^T) * bmm1Scale[0] and the output is softmax(logits) V * bmm2Scale[0]
//! with the attention sink in the denominator; P is rounded to E4M3 (x448) for the PV product, as in the trtllm-gen
//! FP8 context kernel. The output is BF16 [numTokens, 64, 512].
//!
//! Q may also be given in BF16 together with the staging quant scale: the kernel then quantizes it on load to exactly
//! the bytes the context FP8 quantizer (quantizeCopyInputToFp8Kernel) writes, e4m3(rn_satfinite(float(q) * qScale)),
//! so no separate quantization pass is needed.
//!
//! The kernel is persistent: one CTA per SM, each holding the whole tensor memory (512 columns) and most of the
//! shared memory, so it cannot overlap another kernel on the same SMs and is launched without PDL. run() needs no
//! workspace, allocates nothing and does not synchronize, so it can be captured in a CUDA graph.

constexpr int kNumHeads = 64;
constexpr int kHeadDim = 512;

struct Csa2SparseContextParams
{
    int numTokens{0};             //!< Query tokens.

    void const* q{nullptr};       //!< E4M3 or BF16 [numTokens, 64, 512], unit channel stride, 16-B aligned.
    bool qBf16{false};            //!< Q is BF16 and is quantized with qScale in the kernel.
    int64_t qTokenStride{0};      //!< In elements; a multiple of 16 bytes.
    int64_t qHeadStride{0};       //!< In elements; a multiple of 16 bytes.
    float const* qScale{nullptr}; //!< FP32 [1] quant scale of BF16 Q (device); unused for E4M3 Q.

    void const* bank{nullptr};    //!< E4M3 [bankRows, 512], contiguous rows, 16-B aligned; row 0 is zero.
    int64_t bankRows{0};

    int32_t const* indices{nullptr}; //!< int32 [numTokens, >= maxKeys] bank rows, unit column stride, 16-B aligned.
    int64_t indexStride{0};          //!< Row stride in elements; a multiple of 4.
    int maxKeys{0};                  //!< Index columns a query may use.
    int32_t const* lens{nullptr};    //!< int32 [numTokens] keys per query.

    float const* bmm1Scale{nullptr}; //!< FP32 [2] (device): {logit scale, logit scale * log2(e)}.
    float const* bmm2Scale{nullptr}; //!< FP32 [1] (device): output scale.
    float const* attnSinks{nullptr}; //!< FP32 [64] in natural-log units, or nullptr for no sink.

    void* out{nullptr};              //!< BF16 [numTokens, 64, 512], unit channel stride, 16-B aligned.
    int64_t outTokenStride{0};       //!< In elements; a multiple of 8.
    int64_t outHeadStride{0};        //!< In elements; a multiple of 8.

    //! Instead of out, when set: the o_a projection's input, the bytes inverseRopeFp8Quant writes from out. Per head
    //! and 128-channel block an FP32 scale max(amax, 1e-12) / 448 and E4M3 values; the last 64 channels of a head are
    //! rotated back in GPT-J pairs at the token's position first. Head h is head h % headsPerGroup of group
    //! h / headsPerGroup.
    void* fp8Out{nullptr};     //!< E4M3 [groups, numTokens, headsPerGroup * 512], unit channel stride, 32-B aligned.
    int64_t fp8GroupStride{0}; //!< In bytes; a multiple of 32.
    int64_t fp8TokenStride{0}; //!< In bytes; a multiple of 32.
    float* fp8Scale{nullptr};  //!< FP32 [groups, headsPerGroup * 4, fp8ScaleRows], unit token stride.
    int64_t fp8ScaleGroupStride{0};   //!< In elements.
    int64_t fp8ScaleBlockStride{0};   //!< In elements.
    int fp8ScaleRows{0};              //!< numTokens rounded up to 4; the rows past numTokens are zeroed.
    int headsPerGroup{0};
    void const* positions{nullptr};   //!< int32 or int64 [numTokens] RoPE positions.
    bool positions64{false};
    float const* ropeCosSin{nullptr}; //!< FP32 [positions, 64]: 32 cos then 32 sin per position, 16-B aligned.
};

//! Whether this build carries the kernel for the current device (an SM100-family GPU with a matching image).
bool isSupported();

//! Launches the kernel on stream (nothing for numTokens == 0).
void run(Csa2SparseContextParams const& params, cudaStream_t stream);

} // namespace kernels::csa2SparseContext

TRTLLM_NAMESPACE_END
