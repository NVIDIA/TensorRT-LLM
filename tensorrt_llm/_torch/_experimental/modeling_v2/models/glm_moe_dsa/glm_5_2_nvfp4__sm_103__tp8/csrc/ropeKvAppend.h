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

#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace rope_kv_append
{

//! MLA absorption-mode latent width and RoPE width supported by the fused kernels.
constexpr int32_t kKvLoraRank = 512;
constexpr int32_t kRopeDim = 64;
constexpr int32_t kLatentDim = kKvLoraRank + kRopeDim;

//! Memory order of both kernels: they may start early as programmatic dependents (PDL) and read everything a
//! producer writes, including per-step metadata (cache lengths, KV block offsets, indexer slot mappings, request
//! indices, MTP accepted counts), only after griddepcontrol.wait. Only the quantization scales, which no kernel of the
//! step writes, are read before the wait.

//! Generation-phase MLA RoPE + FP8 query quantization + FP8 latent KV append, optionally fused with the DSA
//! indexer K-cache scatter. With an FP8 MLA cache the outputs are byte-identical to
//! applyMLARopeAndAssignQKVKernelGeneration (absorption mode, rope_append, no helix) followed by
//! indexerKCacheScatterUnifiedKernel.
struct RopeAppendParams
{
    // Query side.
    __nv_bfloat16 const* fusedQ{nullptr}; //!< [numTokens, numHeads, kLatentDim]; the first kKvLoraRank are read.
    __nv_bfloat16 const* qPe{nullptr};    //!< [numTokens, numHeads, kRopeDim] with strides (qPeStride, qPeLd, 1).
    int32_t qPeLd{0};
    int32_t qPeStride{0};
    uint8_t* quantQ{nullptr};          //!< FP8 E4M3 [numTokens, numHeads, kLatentDim].
    float const* quantScaleQ{nullptr}; //!< Optional; 1.0 when null.

    // KV side.
    __nv_bfloat16 const* latent{nullptr}; //!< [numTokens, kLatentDim] rows (packed).
    float2 const* cosSin{nullptr};        //!< Rotary table addressed as cosSin[kRopeDim * position + d / 2].
    int32_t cosSinRows{0};                //!< Positions in the rotary table.
    int32_t const* kvLens{nullptr};       //!< [batch] cache lengths after this step's append.
    float const* quantScaleKv{nullptr};   //!< Optional; 1.0 when null.
    kernels::KVBlockArray kvCache;

    // Scheduler buffers consumed by the FP8 MLA generation FMHA.
    int32_t* cuQSeqlens{nullptr};
    int32_t* cuKvSeqlens{nullptr};
    uint32_t* fmhaTileCounter{nullptr};
    float* bmm1Scale{nullptr};            //!< Optional [2].
    float* bmm2Scale{nullptr};            //!< Optional [1].
    float const* dequantScaleQ{nullptr};  //!< Optional; 1.0 when null.
    float const* dequantScaleKv{nullptr}; //!< Optional; 1.0 when null.
    float const* quantScaleO{nullptr};    //!< Optional; 1.0 when null.
    float hostBmm1Scale{1.F};

    int32_t numTokens{0};
    int32_t seqLen{1};
    int32_t numHeads{0};

    // Optional DSA indexer K-cache scatter (skipped when idxNumTokens == 0).
    uint8_t const* idxK{nullptr};     //!< [idxNumTokens, idxHeadBytes] bytes.
    uint8_t const* idxScale{nullptr}; //!< [idxNumTokens, 4] bytes.
    uint8_t* idxCache{nullptr};       //!< Strided 4-D byte view of one layer's indexer pool.
    int64_t const* idxSlotData{nullptr};
    int64_t const* idxSlotScale{nullptr};
    int32_t idxNumTokens{0};
    int32_t idxHeadBytes{128};
    int32_t idxDims[4]{0, 0, 0, 0};
    int64_t idxStrides[4]{0, 0, 0, 0};
};

//! Fused bridge from per-row local top-k token ids to global KV-pool indices, optionally writing the native
//! top-k reuse copy in the same pass. Output values are identical to convertReqIndexToGlobal[Grouped]Kernel.
struct TopkBridgeParams
{
    int32_t const* topk{nullptr}; //!< Source rows [srcRows, numTopk] with row stride topkStride0, unit column stride.
    int64_t topkStride0{0};
    int32_t numRows{0};           //!< Rows remapped to global indices.
    int32_t numTopk{0};

    int32_t const* reqIdx{nullptr}; //!< [numRows] request index per row, before subtracting reqIdxBias.
    int32_t reqIdxBias{0};
    int32_t const* blockTable{nullptr};
    int64_t btStride0{0};
    int64_t btStride1{1};
    int32_t maxBlocksPerReq{0};
    int32_t blockSize{0};
    int32_t strideFactor{0};

    int32_t const* layerIds{nullptr}; //!< Optional [groupSize]; when null the single layerId is used.
    int32_t layerId{0};
    int32_t groupSize{1};
    int32_t* out{nullptr}; //!< [groupSize, numRows, numTopk] with strides (outStrideG, outStride0, 1).
    int64_t outStrideG{0};
    int64_t outStride0{0};

    //! Copy rows: dst[copyRowStart + i, :numTopk] = topk[srcRow(i), :] for i in [0, copyRows).
    //! srcRow(i) = i (copyMode 0) or the MTP last-accepted row i * nextN + clamp(numAccepted[i] - 1, 0, nextN - 1)
    //! (copyMode 1; i * nextN + nextN - 1 when numAccepted is null).
    int32_t* copyDst{nullptr};
    int64_t copyStride0{0};
    int32_t copyRows{0};
    int32_t copyRowStart{0};
    int32_t copyMode{0};
    int32_t nextN{1};
    void const* numAccepted{nullptr}; //!< Already offset to the first generation request.
    bool numAcceptedIs64{false};
};

void invokeRopeAppendScatter(RopeAppendParams const& params, bool enablePdl, cudaStream_t stream);

void invokeTopkBridge(TopkBridgeParams const& params, bool enablePdl, cudaStream_t stream);

} // namespace rope_kv_append

TRTLLM_MV2_NAMESPACE_END
