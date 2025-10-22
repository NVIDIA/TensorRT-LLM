/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
constexpr uint32_t headElems = validElemsPerHead <= 64 ? 64 : (validElemsPerHead <= 128 ? 128 : (isMLA ? 576 : 256));
static_assert(headElems == 64 || headElems == 128 || headElems == 256 || headElems == 576, "not implemented");
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
constexpr uint32_t validElemsPerVHead = 512;
static_assert(lowPrecOutput == false);
using OutputHead = Vec<__nv_bfloat16, validElemsPerVHead>;
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
    uint32_t* semaphores, void* scratch, cudaStream_t stream);

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
