#pragma once
#include "mha_components.cuh"
#include "mha_stdheaders.cuh"
#include "tma.h"
#include "utils.cuh"

template <uint32_t tileM, uint32_t totalNbRows>
__device__ inline ThrdRegRowMaxT<tileM> loadShmRowMax(
    Vec<float, totalNbRows> const& shm, uint32_t tileBaseRow, uint32_t lane = laneId())
{
    ThrdRegRowMaxT<tileM> result{};
#pragma unroll
    for (uint32_t i = 0; i < result.size; i++)
    {
        result[i] = shm[tileBaseRow + i * warp_size + lane];
    }
    return result;
}

template <uint32_t tileM, uint32_t totalNbRows>
__device__ inline void storeRowMax(
    Vec<float, totalNbRows>& shm, ThrdRegRowMaxT<tileM> const& src, uint32_t tileBaseRow, uint32_t lane = laneId())
{
#pragma unroll
    for (uint32_t i = 0; i < src.size; i++)
    {
        shm[tileBaseRow + i * warp_size + lane] = src[i];
    }
}

template <uint32_t tileM, uint32_t totalNbRows>
__device__ inline void storeRowMaxAsync(CgaBarrier& bar, Vec<float, totalNbRows>& shm, ThrdRegRowMaxT<tileM> const& src,
    uint32_t tileBaseRow, uint32_t lane = laneId())
{
#pragma unroll
    for (uint32_t i = 0; i < src.size; i++)
    {
        tma::storeAsync(&shm[tileBaseRow + i * warp_size + lane], src[i], bar);
    }
}

template <uint32_t tileM, uint32_t tileN>
__device__ inline QuadRegRowMaxT<tileM> computeRowMax(WarpAccT<tileM, tileN> const& acc)
{
    QuadRegRowMaxT<tileM> rowMaxLog2e{};
// compute per-thread row max
#pragma unroll
    for (uint32_t n = 0; n < acc.cols; n++)
    {
#pragma unroll
        for (uint32_t j = 0; j < InstAcc::cols; j++)
        {
#pragma unroll
            for (uint32_t m = 0; m < acc.rows; m++)
            {
#pragma unroll
                for (uint32_t i = 0; i < InstAcc::rows; i++)
                {
                    float& dst = rowMaxLog2e[m * InstAcc::rows + i];
                    dst = ((n == 0 && j == 0) ? acc(m, n)(i, j) : fmaxf(dst, acc(m, n)(i, j)));
                }
            }
        }
    }
// compute warp row max
#pragma unroll
    for (uint32_t xorMask = 2; xorMask != 0; xorMask /= 2)
    {
#pragma unroll
        for (uint32_t i = 0; i < rowMaxLog2e.size; i++)
        {
            rowMaxLog2e[i] = fmaxf(rowMaxLog2e[i], __shfl_xor_sync(~0U, rowMaxLog2e[i], xorMask));
        }
    }
    return rowMaxLog2e;
}

template <typename T, uint32_t n>
__device__ inline uint32_t hashRegData(Vec<T, n> const& data)
{
    static_assert(sizeof(T) == 4);
    uint32_t result = 0;
#pragma unroll
    for (uint32_t i = 0; i < n; i++)
    {
        result ^= reinterpret_cast<uint32_t const&>(data[i]);
    }
    return result;
}
