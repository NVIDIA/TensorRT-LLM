/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "refAttention.h"
#include <cstdint>

template <typename T>
Vec<float, validElemsPerHead> toF32Head(Vec<T, validElemsPerHead> const& src)
{
    Vec<float, validElemsPerHead> dst;
    for (uint32_t i = 0; i < validElemsPerHead; i++)
    {
        dst[i] = float(src[i]);
    }
    return dst;
}

inline float dot(Vec<float, validElemsPerHead> const& q, Vec<float, validElemsPerHead> const& k)
{
    float acc = 0;
    for (uint32_t i = 0; i < validElemsPerHead; i++)
    {
        acc += q[i] * k[i];
    }
    return acc;
};

#if EIGEN_WORLD_VERSION < 3 || (EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION < 4)
namespace Eigen
{
template <typename Type, int Size>
using Vector = Matrix<Type, Size, 1>;
}
#endif

template <typename MathElem, uint32_t tileSize, bool isPaged, bool useBeamSearch>
Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> refFlashAttention(IOHead const* q,
    CacheSeq<isPaged, useBeamSearch> const& k, CacheSeq<isPaged, useBeamSearch> const& v, uint32_t seqLen, float qScale,
    float kvScale, float xScale, uint32_t slidingWinSize, float* attentionSinks, float skipSoftmaxThresholdScaleFactor,
    uint32_t* skippedBlockCount, uint32_t* totalBlockCount, uint32_t multiBlockNum)
{
    uint32_t const nbTiles = divUp(seqLen, tileSize);
    auto gemm1Acc = Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor>::Zero().eval();
    Eigen::Vector<float, headGrpSize> rowMax, rowSum;
    rowMax.fill(-INFINITY);
    rowSum.fill(0);
    float const rcpXScale = 1.f / xScale;
    float const qkScale = qScale * kvScale / sqrtf(validElemsPerHead);
    uint32_t const seqBeg = (seqLen < slidingWinSize ? 0 : seqLen - slidingWinSize);
    uint32_t const idxTileBeg = seqBeg / tileSize;

    uint32_t const nbSubSeq = (multiBlockNum > 0 && nbTiles >= 2) ? mha::min(nbTiles, multiBlockNum) : 1;
    std::vector<Eigen::Vector<float, headGrpSize>> skipRowMaxs(nbSubSeq);
    for (uint32_t i = 0; i < nbSubSeq; i++)
    {
        skipRowMaxs[i].fill(-INFINITY);
    }
    bool const disableSkipForShortSeq = (seqLen < skipSoftmaxThresholdScaleFactor);
    float const skipSoftmaxThreshold = disableSkipForShortSeq ? 0.0f : skipSoftmaxThresholdScaleFactor / seqLen;

    for (uint32_t idxTile = idxTileBeg; idxTile < nbTiles; idxTile++)
    {
        Eigen::Matrix<float, headGrpSize, tileSize, Eigen::RowMajor> gemm0Acc;
        Vec<Vec<float, validElemsPerHead>, headGrpSize> qF32;
        for (uint32_t i = 0; i < headGrpSize; i++)
        {
            qF32[i] = toF32Head(q[i]);
        }
        uint32_t const tileTokenBeg = (idxTile == idxTileBeg ? seqBeg % tileSize : 0);
        gemm0Acc.leftCols(tileTokenBeg).fill(-INFINITY);
        for (uint32_t j = tileTokenBeg; j < tileSize; j++)
        {
            uint32_t const idxToken = tileSize * idxTile + j;
            if (idxToken < seqLen)
            {
                auto const kF32 = toF32Head(k[idxToken]);
                for (uint32_t i = 0; i < headGrpSize; i++)
                {
                    gemm0Acc(i, j) = dot(qF32[i], kF32) * qkScale;
                }
            }
            else
            {
                gemm0Acc.col(j).fill(-INFINITY);
            }
        }

        Eigen::Vector<float, headGrpSize> const localRowMax = gemm0Acc.rowwise().maxCoeff().eval();
        Eigen::Vector<float, headGrpSize> const tileRowMax = localRowMax.cwiseMax(rowMax).eval();
        auto const prevSkipRowMax = skipRowMaxs[idxTile % nbSubSeq];
        skipRowMaxs[idxTile % nbSubSeq] = localRowMax.cwiseMax(skipRowMaxs[idxTile % nbSubSeq]).eval();

        if (!disableSkipForShortSeq && skipSoftmaxThreshold > 0)
        {
            *totalBlockCount += 1;
            auto const skipSoftmaxMask = ((localRowMax - prevSkipRowMax).array() < std::log(skipSoftmaxThreshold));
            bool const skipBlock = skipSoftmaxMask.all() && ((idxTile - idxTileBeg) >= nbSubSeq);
            if (skipBlock)
            {
                *skippedBlockCount += 1;
                continue;
            }
        }

        Eigen::Matrix<float, headGrpSize, tileSize, Eigen::RowMajor> tileX
            = (gemm0Acc.colwise() - tileRowMax).array().exp().eval();
        Eigen::Vector<float, headGrpSize> const tileRowSum = tileX.rowwise().sum().eval();

        std::for_each(tileX.data(), tileX.data() + tileX.size(), [&](float& e) { e = float(MathElem(e * rcpXScale)); });

        assert((rowMax.array() <= tileRowMax.array()).eval().all());
        if ((rowMax.array() < tileRowMax.array()).any())
        {
            Eigen::Vector<float, headGrpSize> const scale = (rowMax - tileRowMax).array().exp();
            gemm1Acc.array().colwise() *= scale.array();
            rowSum.array().colwise() *= scale.array();
            rowMax = tileRowMax;
        }

        for (uint32_t j = tileTokenBeg; j < std::min(tileSize, seqLen - tileSize * idxTile); j++)
        {
            auto const vF32 = toF32Head(v[tileSize * idxTile + j]);
            for (uint32_t i = 0; i < headGrpSize; i++)
            {
                for (uint32_t k = 0; k < validElemsPerHead; k++)
                {
                    gemm1Acc(i, k) += vF32[k] * tileX(i, j);
                }
            }
        }
        rowSum += tileRowSum;
    }

    // Add the attention sinks.
    if (attentionSinks != nullptr)
    {
        for (uint32_t i = 0; i < headGrpSize; i++)
        {
            rowSum[i] += expf(attentionSinks[i] - rowMax[i]);
        }
    }

    Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> out
        = gemm1Acc.array().colwise() * (xScale * kvScale / rowSum.array());
    std::for_each(out.data(), out.data() + out.size(), [](float& e) { e = float(OutputElem(e)); });
    return out;
}

#define INSTANTIATE_refFlashAttention(prec, tileSize, isPaged, useBeamSearch)                                          \
    template Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor>                                     \
    refFlashAttention<prec, tileSize, isPaged, useBeamSearch>(IOHead const* q,                                         \
        CacheSeq<isPaged, useBeamSearch> const& k, CacheSeq<isPaged, useBeamSearch> const& v, uint32_t seqLen,         \
        float qScale, float kvScale, float xScale, uint32_t slidingWinSize, float* attentionSinks,                     \
        float skipSoftmaxThreshold, uint32_t* skippedBlockCount, uint32_t* totalBlockCount, uint32_t multiBlockNum)

INSTANTIATE_refFlashAttention(CacheElem, 64, false, false);
INSTANTIATE_refFlashAttention(CacheElem, 64, false, true);
INSTANTIATE_refFlashAttention(CacheElem, 64, true, false);
INSTANTIATE_refFlashAttention(CacheElem, 64, true, true);
INSTANTIATE_refFlashAttention(CacheElem, 128, false, false);
INSTANTIATE_refFlashAttention(CacheElem, 128, false, true);
INSTANTIATE_refFlashAttention(CacheElem, 128, true, false);
INSTANTIATE_refFlashAttention(CacheElem, 128, true, true);

template <typename MathElem, bool isPaged, bool useBeamSearch>
#if SPEC_DEC
Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> refAttention(IOHead const* q,
    CacheSeq<isPaged, useBeamSearch> const& k, CacheSeq<isPaged, useBeamSearch> const& v, uint32_t seqLen, float qScale,
    float kvScale, float xScale, uint32_t slidingWinSize, bool* hostMask, const uint32_t qSeqLen, const uint32_t q_len)
{
#else
Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> refAttention(IOHead const* q,
    CacheSeq<isPaged, useBeamSearch> const& k, CacheSeq<isPaged, useBeamSearch> const& v, uint32_t seqLen, float qScale,
    float kvScale, float xScale, uint32_t slidingWinSize, float* attentionSinks)
{
#endif
    float const rcpXScale = 1.f / xScale;
    float const qkScale = qScale * kvScale / sqrtf(validElemsPerHead);

    Eigen::Matrix<float, headGrpSize, Eigen::Dynamic, Eigen::RowMajor> gemm0Acc(headGrpSize, seqLen);
    Vec<Vec<float, validElemsPerHead>, headGrpSize> qF32;
    for (uint32_t i = 0; i < headGrpSize; i++)
    {
        qF32[i] = toF32Head(q[i]);
    }
#if SPEC_DEC && SLIDING_WINDOW
    // In Spec-dec + SLIDING WINDOW mode, only allow linear tree or !rtIsReallySliding.
    // the token starting position is seqLen - qSeqLen + 1
    assert(!IS_SPEC_DEC_TREE || seqLen - qSeqLen + 1 < slidingWinSize);
    uint32_t const tok0SeqLen = seqLen - qSeqLen + 1 + q_len;
    uint32_t const seqBeg
        = (int32_t(tok0SeqLen) < int32_t(slidingWinSize) ? 0 : int32_t(tok0SeqLen) - int32_t(slidingWinSize));
#else
    uint32_t const seqBeg = (seqLen < slidingWinSize ? 0 : seqLen - slidingWinSize);
#endif
    gemm0Acc.leftCols(seqBeg).fill(-INFINITY);
    for (uint32_t j = seqBeg; j < seqLen; j++)
    {
        auto const kF32 = toF32Head(k[j]);
        for (uint32_t i = 0; i < headGrpSize; i++)
        {
#if SPEC_DEC
            bool const validFlag = j < (seqLen - qSeqLen) || hostMask[q_len * qSeqLen + (j - seqLen + qSeqLen)];
            gemm0Acc(i, j) = validFlag ? dot(qF32[i], kF32) * qkScale : -INFINITY;
#else
            gemm0Acc(i, j) = dot(qF32[i], kF32) * qkScale;
#endif
        }
    }

    Eigen::Vector<float, headGrpSize> const rowMax = gemm0Acc.rowwise().maxCoeff().eval();

    Eigen::Matrix<float, headGrpSize, Eigen::Dynamic, Eigen::RowMajor> x
        = (gemm0Acc.colwise() - rowMax).array().exp().eval();
    Eigen::Vector<float, headGrpSize> rowSum = x.rowwise().sum().eval();

    std::for_each(x.data(), x.data() + x.size(), [&](float& e) { e = float(MathElem(e * rcpXScale)); });

    auto gemm1Acc = Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor>::Zero().eval();
    for (uint32_t j = seqBeg; j < seqLen; j++)
    {
        auto const vF32 = toF32Head(v[j]);
        for (uint32_t i = 0; i < headGrpSize; i++)
        {
            for (uint32_t k = 0; k < validElemsPerHead; k++)
            {
                gemm1Acc(i, k) += vF32[k] * x(i, j);
            }
        }
    }

    // Add the attention sinks.
#if !SPEC_DEC
    if (attentionSinks != nullptr)
    {
        for (uint32_t i = 0; i < headGrpSize; i++)
        {
            rowSum[i] += expf(attentionSinks[i] - rowMax[i]);
        }
    }
#endif

    Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> out
        = gemm1Acc.array().colwise() * (xScale * kvScale / rowSum.array());
    std::for_each(out.data(), out.data() + out.size(), [](float& e) { e = float(OutputElem(e)); });
    return out;
}

#if SPEC_DEC
#define INSTANTIATE_refAttention(prec, isPaged, useBeamSearch)                                                         \
    template Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor>                                     \
    refAttention<prec, isPaged, useBeamSearch>(IOHead const* q, CacheSeq<isPaged, useBeamSearch> const& k,             \
        CacheSeq<isPaged, useBeamSearch> const& v, uint32_t seqLen, float qScale, float kvScale, float xScale,         \
        uint32_t slidingWinSize, bool* hostMask, const uint32_t qSeqLen, const uint32_t q_len)
#else
#define INSTANTIATE_refAttention(prec, isPaged, useBeamSearch)                                                         \
    template Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor>                                     \
    refAttention<prec, isPaged, useBeamSearch>(IOHead const* q, CacheSeq<isPaged, useBeamSearch> const& k,             \
        CacheSeq<isPaged, useBeamSearch> const& v, uint32_t seqLen, float qScale, float kvScale, float xScale,         \
        uint32_t slidingWinSize, float* attentionSinks)
#endif
INSTANTIATE_refAttention(InputElem, false, false);
INSTANTIATE_refAttention(InputElem, false, true);
INSTANTIATE_refAttention(InputElem, true, false);
INSTANTIATE_refAttention(InputElem, true, true);
#if CACHE_ELEM_ENUM != 0 && !(IS_MLA)
INSTANTIATE_refAttention(CacheElem, false, false);
INSTANTIATE_refAttention(CacheElem, false, true);
INSTANTIATE_refAttention(CacheElem, true, false);
INSTANTIATE_refAttention(CacheElem, true, true);
#endif
