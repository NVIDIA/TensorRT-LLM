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

#include <cstdint>
#include <cuda_runtime.h>

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::tri_attention_score
{

// The TriAttention trig score of one cached token t for query head h is
//     score(t, h) = sum_f  K_re(t, f) * c_re(h, f)
//                        + K_im(t, f) * c_im(h, f)
//                        + |K(t, f)|  * c_mlr(h, f)
// where the c tables fold the per-round query calibration and phase terms so
// the per-token kernel touches no trigonometry. The fold runs once per
// eviction round; the score kernel then reads paged KV directly (one thread
// per token) and writes fp32 rows in the selector's layout.

// Element type of the paged KV pools. The score kernel reads layers through
// raw per-layer base addresses (V2 exposes each layer as its own storage), so
// the caller passes the shared element type explicitly instead of a tensor.
// The quantized types (fp8/int8) are functional-only: they run the strided
// scalar kernel exclusively, and their per-layer dequantization scale is
// folded into the score coefficients (see foldScoreCoefficientsLaunch), so
// the score kernel reads raw quantized elements with zero hot-loop cost.
enum class PoolElementType : int32_t
{
    kBFloat16 = 0,
    kHalf = 1,
    kFloat32 = 2,
    kFloat8E4M3 = 3,
    kInt8 = 4,
};

// Upper bound on per-offset accumulators held by one score thread on the
// "max" aggregation path. The production "mean" path folds every offset into
// ONE coefficient plane, so this bound never constrains it; "max" needs one
// plane per offset, and the default geometric offset table exceeds 8, so a
// "max" run trips the fold op's TORCH_CHECK unless the offset budget is
// reduced. 8 keeps headroom without bloating the per-thread register budget.
inline constexpr int32_t kMaxScoreOffsets = 8;

// Threads per score CTA; one thread scores one cached token.
inline constexpr int32_t kScoreBlockThreads = 128;

// Fold the per-round score coefficients:
//     c_re  = fss * (q_re * cos - q_im * sin)
//     c_im  = fss * (q_im * cos + q_re * sin)
//     c_mlr = mlr * fss
// Mean aggregation collapses all offsets into meanCos/meanSin beforehand and
// writes one plane (numOffsets == 1). Max aggregation cannot collapse (max
// does not commute through the frequency sum), so it writes one c_re/c_im
// plane per offset with cos/sin((round_start + offset) * omega); c_mlr is
// offset independent either way. Output layout per plane is
// [numRequests, numCalibratedLayers, numQueryHeads, numFreqs] fp32, indexed
// by ABSOLUTE layer id (matching the calibration tables).
//
// kvScales (nullable) carries one fp32 dequantization scale per ABSOLUTE
// layer id for quantized (fp8/int8) KV pools: K_real = scale_l * K_quant, so
// multiplying scale_l into all three coefficient tables (every per-offset
// plane included) lets the score kernel consume raw quantized elements. The
// |K| term relies on |scale * K_q| == scale * |K_q|, which only holds for
// scale > 0 — the host wrapper validates positivity before launch.
void foldScoreCoefficientsLaunch(float const* qReal, // [L_cal * HQ * F]
    float const* qImag,                              // [L_cal * HQ * F]
    float const* mlrCoef,                            // [L_cal * HQ * F]
    float const* freqScaleSq,                        // [F]
    float const* meanCos,                            // [numRequests * F] (mean path, else nullptr)
    float const* meanSin,                            // [numRequests * F] (mean path, else nullptr)
    float const* omega,                              // [F]            (max path, else nullptr)
    float const* offsets,                            // [numOffsets]   (max path, else nullptr)
    int32_t const* roundStarts,                      // [numRequests]  (max path, else nullptr)
    float const* kvScales,                           // [L_cal] per-layer dequant scale (quantized pools, else nullptr)
    float* cRe,                                      // [numOffsets, numRequests, L_cal, HQ, F]
    float* cIm,                                      // [numOffsets, numRequests, L_cal, HQ, F]
    float* cMlr,                                     // [numRequests, L_cal, HQ, F]
    int32_t numRequests, int32_t numCalibratedLayers, int32_t numQueryHeads, int32_t numFreqs, int32_t numOffsets,
    bool useMax, cudaStream_t stream);

// Everything one folded-score launch needs. One "segment" is one
// (request, scored layer) pair; segments are request-major so
// seg % numLayers == 0 identifies each request's first segment.
struct FoldedScoreParams
{
    int64_t const* layerBaseAddrs;     // [num pools] absolute device addresses, ABSOLUTE layer id indexed
    int32_t const* blockOffsets;       // flattened native V2 page table
    int64_t const* segPageOffsets;     // [numSegments] offset of each segment's page row into blockOffsets
    int32_t const* segRequestIds;      // [numSegments]
    int32_t const* segLayerIds;        // [numSegments] ABSOLUTE layer ids
    int32_t const* requestSeqLens;     // [numRequests]
    int32_t* validWidthOut;            // [numRequests] side-store: seqLen - tokenStart, once per request
    int32_t const* requestTokenStarts; // [numRequests] pinned prompt length = decode-region origin
    float const* cRe;                  // fold output (see foldScoreCoefficientsLaunch)
    float const* cIm;
    float const* cMlr;
    float* out;        // [segment, numQueryHeads, outputWidth] fp32 decode-only scores
    int32_t outputWidth;
    int32_t numLayers; // scored layers per request (the segment period)
    int32_t numRequests;
    int32_t numCalibratedLayers;
    int32_t numQueryHeads;
    int32_t numKvHeads;
    int32_t numFreqs;
    int32_t tokensPerBlock;
    int32_t kvFactor;   // page-table entries encode role pages; entry / kvFactor = pool page
    int32_t numOffsets; // effective c_re/c_im planes (1 on the mean path)
    // Grid mapping for GQA group sizes without a dedicated template
    // instantiation: grid.z indexes single query heads instead of KV heads
    // (the KV plane is derived per block; K traffic is repeated per head).
    bool zIsQueryHead;
    // HND pool element strides (elements, not bytes) shared by all layers.
    int64_t stridePage;
    int64_t strideKvHead;
    int64_t strideSlot;
    int64_t strideDim;
};

// Launch the folded score over paged KV. useVectorized selects 16-byte
// 8-frequency chunk loads (requires numFreqs % 8 == 0, strideDim == 1,
// bf16/fp16 pools, and 16-byte aligned bases/strides — the caller audits
// alignment); otherwise a fully strided scalar path runs the same math.
// groupSize = numQueryHeads / numKvHeads must be 1, 2, 4, or 8 unless
// params.zIsQueryHead maps grid.z to single query heads.
void foldedScoreLaunch(FoldedScoreParams const& params, PoolElementType poolType, int32_t groupSize,
    int32_t numSegments, bool useVectorized, bool useMax, cudaStream_t stream);

} // namespace kernels::tri_attention_score

TRTLLM_NAMESPACE_END
