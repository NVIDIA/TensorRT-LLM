/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <fmha/hopper/tma_types.h>
#include <fmha/utils.h>

namespace fmha
{

inline __device__ uint32_t elect_one_sync();

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int DIM, cudaTmaDescType DESC_TYPE, bool USE_TMA_MULTICAST>
inline __device__ void utmaldg(cudaTmaDesc const* p_desc, // TMA desc
    uint32_t smem_ptr,                                    // desc smem address
    uint32_t smem_barrier,                                // smem_barrier
    int32_t const (&coord)[DIM],                          // coord
    uint32_t elect_one = 1);

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// UTMALDG TILED WITHOUT MULTICAST
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ void utmaldg<2, fmha::cudaTmaDescType::TILED, false>(
    cudaTmaDesc const* p_desc, uint32_t smem_ptr, uint32_t smem_barrier, int32_t const (&coord)[2], uint32_t elect_one)
{
    if (elect_one)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];\n"
            :
            : "r"(smem_ptr), "l"(reinterpret_cast<uint64_t>(p_desc)), "r"(coord[0]), "r"(coord[1]), "r"(smem_barrier)
            : "memory");
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ void utmaldg<3, fmha::cudaTmaDescType::TILED, false>(
    cudaTmaDesc const* p_desc, uint32_t smem_ptr, uint32_t smem_barrier, int32_t const (&coord)[3], uint32_t elect_one)
{
    if (elect_one)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3, %4}], [%5];\n"
            :
            : "r"(smem_ptr), "l"(reinterpret_cast<uint64_t>(p_desc)), "r"(coord[0]), "r"(coord[1]), "r"(coord[2]),
            "r"(smem_barrier)
            : "memory");
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// 4D, TILED, without Multicast
template <>
inline __device__ void utmaldg<4, fmha::cudaTmaDescType::TILED, false>(
    cudaTmaDesc const* p_desc, uint32_t smem_ptr, uint32_t smem_barrier, int32_t const (&coord)[4], uint32_t elect_one)
{
    if (elect_one)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3, %4, %5}], [%6];\n"
            :
            : "r"(smem_ptr), "l"(reinterpret_cast<uint64_t>(p_desc)), "r"(coord[0]), "r"(coord[1]), "r"(coord[2]),
            "r"(coord[3]), "r"(smem_barrier)
            : "memory");
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// UTMASTG TILED WITHOUT MULTICAST
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int DIM, cudaTmaDescType DESC_TYPE>
inline __device__ void utmastg(cudaTmaDesc const* p_desc, // TMA desc
    uint32_t smem_ptr,                                    // src smem address
    int32_t const (&coord)[DIM]);                         // coord

// 3D, TILED
template <>
inline __device__ void utmastg<3, fmha::cudaTmaDescType::TILED>(
    cudaTmaDesc const* p_desc, uint32_t smem_ptr, const int32_t (&coord)[3])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];\n" ::"l"(
                     reinterpret_cast<uint64_t>(p_desc)),
                 "r"(coord[0]), "r"(coord[1]), "r"(coord[2]), "r"(smem_ptr)
                 : "memory");
#endif
}

// 4D, TILED
template <>
inline __device__ void utmastg<4, fmha::cudaTmaDescType::TILED>(
    cudaTmaDesc const* p_desc, uint32_t smem_ptr, int32_t const (&coord)[4])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%1, %2, %3, %4}], [%5];\n" ::"l"(
                     reinterpret_cast<uint64_t>(p_desc)),
                 "r"(coord[0]), "r"(coord[1]), "r"(coord[2]), "r"(coord[3]), "r"(smem_ptr)
                 : "memory");
#endif
}

inline __device__ void tmastg_arrive()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.commit_group;");
#else
    assert(false);
#endif
}

inline __device__ void tmastg_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(0) : "memory");
#else
    assert(false);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace fmha
