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
    float kvScale, float xScale, uint32_t slidingWinSize)
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

        Eigen::Vector<float, headGrpSize> const tileRowMax = gemm0Acc.rowwise().maxCoeff().cwiseMax(rowMax).eval();

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
    Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> out
        = gemm1Acc.array().colwise() * (xScale * kvScale / rowSum.array());
    std::for_each(out.data(), out.data() + out.size(), [](float& e) { e = float(OutputElem(e)); });
    return out;
}

#define INSTANTIATE_refFlashAttention(prec, tileSize, isPaged, useBeamSearch)                                          \
    template Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor>                                     \
    refFlashAttention<prec, tileSize, isPaged, useBeamSearch>(IOHead const* q,                                         \
        CacheSeq<isPaged, useBeamSearch> const& k, CacheSeq<isPaged, useBeamSearch> const& v, uint32_t seqLen,         \
        float qScale, float kvScale, float xScale, uint32_t slidingWinSize)

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
    float kvScale, float xScale, uint32_t slidingWinSize)
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
    uint32_t const seqBeg = (seqLen < slidingWinSize ? 0 : seqLen - slidingWinSize);
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
    Eigen::Vector<float, headGrpSize> const rowSum = x.rowwise().sum().eval();

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
        uint32_t slidingWinSize)
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
