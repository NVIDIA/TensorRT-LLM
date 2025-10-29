/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define EIGEN_STACK_ALLOCATION_LIMIT (1U << 20)
#include "../mha.h"
#include <Eigen/Dense>

template <bool isPaged, bool useBeamSearch>
struct CacheSeq;

template <>
struct CacheSeq<false, false>
{
    GMemCacheHead const& operator[](uint32_t i) const
    {
        return data[i];
    }

    GMemCacheHead const* data;
};

template <>
struct CacheSeq<false, true>
{
    GMemCacheHead const& operator[](uint32_t i) const
    {
        return data[2 * nbKHeads * maxSeqLen * cacheIndir[i] + i];
    }

    uint32_t nbKHeads;
    GMemCacheHead const* data;
    uint32_t const* cacheIndir;
    uint32_t maxSeqLen;
};

template <>
struct CacheSeq<true, false>
{
    GMemCacheHead const& operator[](uint32_t i) const
    {
        uint32_t const pageIdx = pageIndices[i / tokensPerPage];
#if PAGED_KV_CACHE_LAYOUT == 1 && USE_PAGED_KV_CACHE
        return pool[nbHeads * tokensPerPage * pageIdx + (i % tokensPerPage) * nbHeads + idxHead];
#else
        return pool[tokensPerPage * nbHeads * pageIdx + tokensPerPage * idxHead + i % tokensPerPage];
#endif
    }

    GMemCacheHead const* pool;
    int32_t const* pageIndices;
    uint32_t nbHeads;
    uint32_t idxHead;
};

template <>
struct CacheSeq<true, true>
{
    GMemCacheHead const& operator[](uint32_t i) const
    {
        uint32_t const pageIdx = pageIndices[cacheIndir[i] * 2 * maxNbPages + i / tokensPerPage];
        return pool[tokensPerPage * nbHeads * pageIdx + tokensPerPage * idxHead + i % tokensPerPage];
    }

    GMemCacheHead const* pool;
    int32_t const* pageIndices;
    uint32_t maxNbPages;
    uint32_t nbHeads;
    uint32_t idxHead;
    uint32_t const* cacheIndir;
};

template <typename MathElem, uint32_t tileSize, bool isPaged, bool useBeamSearch>
Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> refFlashAttention(IOHead const* q,
    CacheSeq<isPaged, useBeamSearch> const& k, CacheSeq<isPaged, useBeamSearch> const& v, uint32_t seqLen, float qScale,
    float kvScale, float xScale, uint32_t slidingWinSize, float* attentionSinks);

template <typename MathElem, bool isPaged, bool useBeamSearch>
#if SPEC_DEC
Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> refAttention(IOHead const* q,
    CacheSeq<isPaged, useBeamSearch> const& k, CacheSeq<isPaged, useBeamSearch> const& v, uint32_t seqLen, float qScale,
    float kvScale, float xScale, uint32_t slidingWinSize, bool* hostMask, const uint32_t qSeqLen, const uint32_t q_len);
#else
Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> refAttention(IOHead const* q,
    CacheSeq<isPaged, useBeamSearch> const& k, CacheSeq<isPaged, useBeamSearch> const& v, uint32_t seqLen, float qScale,
    float kvScale, float xScale, uint32_t slidingWinSize, float* attentionSinks);
#endif

template <uint32_t ropeStyle>
InputHead applyRoPE(InputHead const& head, Vec<float, validElemsPerHead> const& ropeCosSin)
{
    if constexpr (ropeStyle == 0)
    {
        return head;
    }
    constexpr uint32_t nbPairs = exactDiv(validElemsPerHead, 2);
    InputHead dst;
    constexpr bool isNeox = (ropeStyle == 1);
    for (uint32_t i = 0; i < nbPairs; i++)
    {
        float const c = ropeCosSin[i * 2];
        float const s = ropeCosSin[i * 2 + 1];
        Eigen::Matrix2f r;
        r << c, -s, s, c;
        Eigen::Vector2f v;
        uint32_t const ix = (isNeox ? i : i * 2);
        uint32_t const iy = (isNeox ? nbPairs + i : i * 2 + 1);
        v << float(head[ix]), float(head[iy]);
        auto const rv = (r * v).eval();
        dst[ix] = InputElem{rv[0]};
        dst[iy] = InputElem{rv[1]};
    }
    return dst;
}
