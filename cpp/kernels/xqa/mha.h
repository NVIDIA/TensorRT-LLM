/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef __CUDACC__
#include <cuda_runtime_api.h>
#endif
#include "defines.h"
#include "utils.h"
#if SPEC_DEC
#include "specDec.h"
#endif
using CacheElem = ElemType<CACHE_ELEM_ENUM>;
constexpr uint32_t validElemsPerHead = HEAD_ELEMS;
constexpr bool isMLA = IS_MLA;
static_assert((isMLA || validElemsPerHead <= 256) && (sizeof(CacheElem) * validElemsPerHead) % 16 == 0);
constexpr uint32_t headElems = validElemsPerHead <= 64    ? 64
    : validElemsPerHead <= 128                            ? 128
    : validElemsPerHead <= 256                            ? 256
    : (isMLA && validElemsPerHead <= 512)                 ? 512
    : isMLA                                               ? 576
                                                          : 256;
static_assert(headElems == 64 || headElems == 128 || headElems == 256 || headElems == 512 || headElems == 576,
    "not implemented");
constexpr uint32_t beamWidth = BEAM_WIDTH;
constexpr uint32_t headGrpSize = HEAD_GRP_SIZE;
#if SPEC_DEC
__device__ constexpr uint32_t rowsPerBlock = M_TILESIZE;
#endif

inline constexpr bool useSpecDec = SPEC_DEC;

using InputElem = INPUT_ELEM;
using InputElem2 = INPUT_ELEM2;
#if !(SPEC_DEC)
constexpr uint32_t inputSeqLen = 1; // speculative decoding if > 1
#endif

constexpr bool useKVCache = USE_KV_CACHE;

using SeqLenDataType = uint32_t;

constexpr bool usePagedKVCache = USE_PAGED_KV_CACHE;
constexpr uint32_t tokensPerPage = TOKENS_PER_PAGE;

using IOHead = Vec<InputElem, validElemsPerHead>;
using InputHead = IOHead;
using GMemCacheHead = Vec<CacheElem, validElemsPerHead>;

constexpr uint32_t validElemsPerKHead = validElemsPerHead;
constexpr bool lowPrecOutput = LOW_PREC_OUTPUT;

#if IS_MLA
// validElemsPerVHead is the "real" V-head dim used by the o_proj on the host side.
//   DSV3: 512  (kv_lora_rank=512)
//   DSV4: 448  (kv_lora_rank=448)
constexpr uint32_t validElemsPerVHead = HEAD_ELEMS_V;
static_assert(validElemsPerVHead == 512 || validElemsPerVHead == 448,
    "Only DSV3 (V=512) and DSV4 (V=448) MLA shapes are supported.");
// For DSV4 the kernel pads V internally to 512 to reuse the DSV3 tile schedule. The trailing
// (512 - validElemsPerVHead) elements are dropped at output-write time so the o_proj input
// remains validElemsPerVHead.
constexpr uint32_t paddedElemsPerVHead = 512;
static_assert(paddedElemsPerVHead >= validElemsPerVHead);
static_assert(lowPrecOutput == false);
// OutputHead is sized to paddedElemsPerVHead so the kernel's internal tile geometry
// (gemm1V, partElemsV, consumerCtaShapeX) can stay identical to the DSV3 schedule even
// when the model's real V-head dim is smaller (DSV4: 448). For DSV4 the trailing
// (paddedElemsPerVHead - validElemsPerVHead) elements per head will be either zero
// (reduce kernel) or undefined (main kernel); callers must slice to validElemsPerVHead
// before consuming the output (e.g. in the Python o_proj path).
using OutputHead = Vec<__nv_bfloat16, paddedElemsPerVHead>;
#else
constexpr uint32_t validElemsPerVHead = validElemsPerHead;
using OutputHead = mha::conditional_t<lowPrecOutput, GMemCacheHead, InputHead>;
#endif
using OutputElem = OutputHead::Elem;

using PaddedInputHead = Vec<InputElem, headElems>;
using PaddedCacheHead = Vec<CacheElem, headElems>;

// impl detail, may be moved to mha.cu/mha_sm90.cu
constexpr bool isHeadPadded = (validElemsPerHead != headElems);

constexpr bool useInputKV = USE_INPUT_KV;

using GMemKVCacheHead = mha::conditional_t<useInputKV, GMemCacheHead, GMemCacheHead const>;

using KVCachePageIndex = int32_t; // shape: KVCacheHead[nbKHeads][tokensPerPage]. Page index in the global pool of pages

constexpr bool allowSlidingWindow = SLIDING_WINDOW;

struct BeamSearchParams
{
    uint32_t const* __restrict__ indices;    // shape: [batchSize][beamWidth][capacity]
    uint32_t capacity;
    uint32_t const* __restrict__ ctxLenList; // shape: [batchSize][beamWidth]. Should be [batchSize] but we have to
                                             // match trt-llm API.
};

uint32_t computeNbSubSeqPerSeqMHA(
    cudaDeviceProp const& prop, uint32_t batchSize, uint32_t nbKHeads, uint32_t maxSeqLen);

void launchMHA(cudaDeviceProp const& prop, uint32_t const nbKHeads,
#if SLIDING_WINDOW
    uint32_t slidingWinSize,
#endif
    float qScale, OutputHead* output,
#if LOW_PREC_OUTPUT
    float const* rcpOutScale,
#endif
#if USE_INPUT_KV
    InputHead const* qkv,
#if ROPE_STYLE != 0
    Vec<float, validElemsPerHead> const* ropeCosSin,
#endif
#else
    InputHead const* q,
#endif
    float const* attentionSinks, // [headGrpSize]
#if USE_PAGED_KV_CACHE
#if PAGED_KV_CACHE_LAYOUT == 1
    GMemCacheHead* kCacheVLLM, GMemCacheHead* vCacheVLLM,
#else
    GMemCacheHead* pool, // global pool of pages
#endif
    KVCachePageIndex const*
        kvCachePageList, // device pointer. shape: KVCachePage[batchSize][beamWidth][2][maxNbPagesPerSeq]
#else
    GMemKVCacheHead* kvCacheData,
#endif
    uint32_t maxSeqLen, uint32_t const* seqLen,
#if BEAM_WIDTH > 1
    BeamSearchParams const& beamSearchParams,
#endif
    uint32_t batchSize,
    float const* __restrict__ kvCacheScale, // Device memory scalar. Same scale for K and V cache. Used only for
                                            // int8/fp8 KV cache.
#if SPEC_DEC
    SpecDecParams const& specDecParams,
#endif
#if SKIP_SOFTMAX_ATTN
    float const skipSoftmaxThresholdScaleFactor,
#if SKIP_SOFTMAX_ATTN_BLOCK_STATS
    uint32_t* __restrict__ skippedBlockCount, uint32_t* __restrict__ totalBlockCount,
#endif
#endif
    uint32_t* semaphores, void* scratch, cudaStream_t stream);

uint32_t computeNbSubSeqPerSeqHopperF8MHA(
    cudaDeviceProp const& prop, uint32_t batchSize, uint32_t nbKHeads, uint32_t maxSeqLen);

void launchHopperF8MHA(cudaDeviceProp const& prop, uint32_t nbKHeads,
#if SLIDING_WINDOW
    uint32_t slidingWinSize,
#endif
    float qScale, OutputHead* output,
#if LOW_PREC_OUTPUT
    float const* rcpOutScale,
#endif
#if USE_INPUT_KV
    InputHead const* qkv,
#if ROPE_STYLE != 0
    Vec<float, validElemsPerHead> const* ropeCosSin,
#endif
#else
    InputHead const* q,
#endif
    float const* attentionSinks, // [headGrpSize]
#if USE_PAGED_KV_CACHE
#if PAGED_KV_CACHE_LAYOUT == 1
    GMemCacheHead* kCacheVLLM, GMemCacheHead* vCacheVLLM,
#else
    GMemCacheHead* pool, // global pool of pages
#endif
    KVCachePageIndex const*
        kvCachePageList, // device pointer. shape: KVCachePageIndex[batchSize][beamWidth][2][maxNbPagesPerSeq].
#else
    GMemKVCacheHead* kvCacheData,
#endif
    uint32_t maxSeqLen, uint32_t const* seqLen,
#if BEAM_WIDTH > 1
    BeamSearchParams const& beamSearchParams,
#endif
    uint32_t batchSize,
    float const* __restrict__ kvCacheScale, // Device memory scalar. Same scale for K and V cache. Used only for
                                            // int8/fp8 KV cache.
#if SPEC_DEC
    SpecDecParams const& specDecParams,
#endif
#if SKIP_SOFTMAX_ATTN
    float const skipSoftmaxThresholdScaleFactor,
#if SKIP_SOFTMAX_ATTN_BLOCK_STATS
    uint32_t* __restrict__ skippedBlockCount, uint32_t* __restrict__ totalBlockCount,
#endif
#endif
    uint32_t* semaphores, void* scratch, cudaStream_t stream);

void launchMLA(cudaDeviceProp const& prop,
    uint32_t inputSeqLen, // uniform for all requests and causal mask is assumed
    float qScale, OutputHead* output, InputHead const* q,
#if USE_PAGED_KV_CACHE
#if PAGED_KV_CACHE_LAYOUT == 1
    GMemCacheHead* kCacheVLLM, GMemCacheHead* vCacheVLLM,
#else
    GMemCacheHead* pool, // global pool of pages
#endif
    KVCachePageIndex const*
        kvCachePageList, // device pointer. shape: KVCachePage[batchSize][beamWidth][2][maxNbPagesPerSeq] (Layout 0) or
                         // [batchSize][maxNbPagesPerSeq] (Layout 1)
#else
    GMemKVCacheHead* kvCacheData,
#endif
    uint32_t maxSeqLen, uint32_t const* seqLen, uint32_t batchSize,
    float const* __restrict__ kvCacheScale, // Device memory scalar. Same scale for K and V cache. Used only for
                                            // int8/fp8 KV cache.
    uint32_t* semaphores, void* scratch, cudaStream_t stream);

#if STATIC_NB_K_HEADS
constexpr uint32_t nbKHeads = NB_K_HEADS;

constexpr uint32_t nbVHeads = nbKHeads;
constexpr uint32_t nbQHeads = nbKHeads * headGrpSize;
constexpr uint32_t nbQKVHeads = nbQHeads + nbKHeads + nbVHeads;
#endif
constexpr uint32_t cacheElemSize = sizeof(CacheElem);
constexpr uint32_t inputElemSize = sizeof(InputElem);
constexpr uint32_t outputElemSize = sizeof(OutputElem);

constexpr uint32_t ioHeadBytes = sizeof(IOHead);
constexpr uint32_t gmemCacheHeadBytes = sizeof(GMemCacheHead);

constexpr uint32_t paddedInputHeadBytes = sizeof(PaddedInputHead);
constexpr uint32_t paddedCacheHeadBytes = sizeof(PaddedCacheHead);

constexpr bool allowMultiBlockMode = ALLOW_MULTI_BLOCK_MODE;

enum class XQAKernelType : int32_t
{
    kAMPERE_WARP_SPECIALIZED = 0,
    kHOPPER_WARP_SPECIALIZED = 1,
    kSM120_MLA = 2
};

#ifdef GENERATE_CUBIN
#define CUBIN_EXPORT extern "C"
#else
#define CUBIN_EXPORT static
#endif
