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

#include "cuda_hint.cuh"
#include "utils.h"
#ifndef GENERATE_CUBIN
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include "barriers.cuh"

enum class StateSpace
{
    kCONSTANT,
    kPARAMETER,
    kGENERIC
};

#ifdef GENERATE_CUBIN
#define CU_TENSOR_MAP_NUM_QWORDS 16

typedef struct CUtensorMap_st
{
#if defined(__cplusplus) && (__cplusplus >= 201103L)
    alignas(64)
#elif __STDC_VERSION__ >= 201112L
    _Alignas(64)
#endif
        uint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
} CUtensorMap;
#endif

namespace tma
{

__device__ inline void load1DAsync(void* dst, void const* src, uint32_t nbBytes, CtaBarrier& bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 :
                 : "l"(__cvta_generic_to_shared(dst)), "l"(reinterpret_cast<uint64_t>(src)), "r"(nbBytes),
                 "l"(__cvta_generic_to_shared(&bar)));
}

template <uint32_t nbDims>
__device__ inline void loadAsync(void* dst, CUtensorMap const& tensorMap, DimsLE<nbDims> offset, CtaBarrier& bar)
{
    if constexpr (nbDims == 1)
    {
        // nbDims==1 does not need tensormap and should just use cp.async.bulk
        asm volatile(
            "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.tile [%0], [%1, {%2}], [%3];\n"
            :
            : "l"(__cvta_generic_to_shared(dst)), "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]),
            "l"(__cvta_generic_to_shared(&bar))
            : "memory");
    }
    else if constexpr (nbDims == 2)
    {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.tile [%0], [%1, {%2, %3}], "
            "[%4];\n"
            :
            : "l"(__cvta_generic_to_shared(dst)), "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]),
            "r"(offset[1]), "l"(__cvta_generic_to_shared(&bar))
            : "memory");
    }
    else if constexpr (nbDims == 3)
    {
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.tile [%0], [%1, {%2, %3, "
            "%4}], [%5];\n"
            :
            : "l"(__cvta_generic_to_shared(dst)), "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]),
            "r"(offset[1]), "r"(offset[2]), "l"(__cvta_generic_to_shared(&bar))
            : "memory");
    }
    else if constexpr (nbDims == 4)
    {
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.tile [%0], [%1, {%2, %3, %4, "
            "%5}], [%6];\n"
            :
            : "l"(__cvta_generic_to_shared(dst)), "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]),
            "r"(offset[1]), "r"(offset[2]), "r"(offset[3]), "l"(__cvta_generic_to_shared(&bar))
            : "memory");
    }
    else if constexpr (nbDims == 5)
    {
        asm volatile(
            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.tile [%0], [%1, {%2, %3, %4, "
            "%5, %6}], [%7];\n"
            :
            : "l"(__cvta_generic_to_shared(dst)), "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]),
            "r"(offset[1]), "r"(offset[2]), "r"(offset[3]), "r"(offset[4]), "l"(__cvta_generic_to_shared(&bar))
            : "memory");
    }
    else
    {
        static_assert(nbDims >= 1 && nbDims <= 5);
    }
}

template <uint32_t nbDims>
__device__ inline void loadAsync(
    void* dst, CUtensorMap const& tensorMap, DimsLE<nbDims> offset, CtaBarrier& bar, uint64_t cacheHint)
{
    if constexpr (nbDims == 1)
    {
        // nbDims==1 does not need tensormap and should just use cp.async.bulk
        asm volatile(
            "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.tile.L2::cache_hint [%0], "
            "[%1, {%2}], [%3], %4;\n"
            :
            : "l"(__cvta_generic_to_shared(dst)), "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]),
            "l"(__cvta_generic_to_shared(&bar)), "l"(cacheHint)
            : "memory");
    }
    else if constexpr (nbDims == 2)
    {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.tile.L2::cache_hint [%0], "
            "[%1, {%2, %3}], [%4], %5;\n"
            :
            : "l"(__cvta_generic_to_shared(dst)), "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]),
            "r"(offset[1]), "l"(__cvta_generic_to_shared(&bar)), "l"(cacheHint)
            : "memory");
    }
    else if constexpr (nbDims == 3)
    {
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.tile.L2::cache_hint [%0], "
            "[%1, {%2, %3, %4}], [%5], %6;\n"
            :
            : "l"(__cvta_generic_to_shared(dst)), "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]),
            "r"(offset[1]), "r"(offset[2]), "l"(__cvta_generic_to_shared(&bar)), "l"(cacheHint)
            : "memory");
    }
    else if constexpr (nbDims == 4)
    {
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.tile.L2::cache_hint [%0], "
            "[%1, {%2, %3, %4, %5}], [%6], %7;\n"
            :
            : "l"(__cvta_generic_to_shared(dst)), "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]),
            "r"(offset[1]), "r"(offset[2]), "r"(offset[3]), "l"(__cvta_generic_to_shared(&bar)), "l"(cacheHint)
            : "memory");
    }
    else if constexpr (nbDims == 5)
    {
        asm volatile(
            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.tile.L2::cache_hint [%0], "
            "[%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
            :
            : "l"(__cvta_generic_to_shared(dst)), "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]),
            "r"(offset[1]), "r"(offset[2]), "r"(offset[3]), "r"(offset[4]), "l"(__cvta_generic_to_shared(&bar)),
            "l"(cacheHint)
            : "memory");
    }
    else
    {
        static_assert(nbDims >= 1 && nbDims <= 5);
    }
}

template <uint32_t nbDims>
__device__ inline void storeAsync(CUtensorMap const& tensorMap, DimsLE<nbDims> const& offset, void* src)
{
    if constexpr (nbDims == 1)
    {
        // nbDims==1 does not need tensormap and should just use cp.async.bulk
        asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.bulk_group.tile [%0, {%1}], [%2];\n"
                     :
                     : "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]), "l"(__cvta_generic_to_shared(src))
                     : "memory");
    }
    else if constexpr (nbDims == 2)
    {
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.tile [%0, {%1, %2}], [%3];\n"
                     :
                     : "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]), "r"(offset[1]),
                     "l"(__cvta_generic_to_shared(src))
                     : "memory");
    }
    else if constexpr (nbDims == 3)
    {
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.tile [%0, {%1, %2, %3}], [%4];\n"
                     :
                     : "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]), "r"(offset[1]), "r"(offset[2]),
                     "l"(__cvta_generic_to_shared(src))
                     : "memory");
    }
    else if constexpr (nbDims == 4)
    {
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.tile [%0, {%1, %2, %3, %4}], [%5];\n"
                     :
                     : "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]), "r"(offset[1]), "r"(offset[2]),
                     "r"(offset[3]), "l"(__cvta_generic_to_shared(src))
                     : "memory");
    }
    else if constexpr (nbDims == 5)
    {
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.tile [%0, {%1, %2, %3, %4, %5}], [%6];\n"
                     :
                     : "l"(reinterpret_cast<uint64_t>(&tensorMap)), "r"(offset[0]), "r"(offset[1]), "r"(offset[2]),
                     "r"(offset[3]), "r"(offset[4]), "l"(__cvta_generic_to_shared(src))
                     : "memory");
    }
    else
    {
        static_assert(nbDims >= 1 && nbDims <= 5);
    }
}

__device__ inline void setTensorMapGlbAddr(CUtensorMap& tensorMap, void* ptr)
{
    asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;\n" ::"l"(&tensorMap), "l"(ptr)
                 : "memory");
}

__device__ inline void commitGroup()
{
    asm volatile("cp.async.bulk.commit_group;\n");
}

// wait until only targetNbInFlightGroups groups are still in-flight.
template <uint32_t targetNbInFlightGroups>
__device__ inline void waitGroup()
{
    asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(targetNbInFlightGroups));
}

__device__ inline void prefetchTensorMap(CUtensorMap const& tensorMap, StateSpace loc = StateSpace::kGENERIC)
{
    assert(reinterpret_cast<uint64_t>(&tensorMap) % alignof(CUtensorMap) == 0);
    switch (loc)
    {
    case StateSpace::kCONSTANT:
        asm volatile("prefetch.const.tensormap [%0];\n" ::"l"(__cvta_generic_to_constant(&tensorMap)) : "memory");
        break;
    case StateSpace::kPARAMETER:
        asm volatile("prefetch.param.tensormap [%0];\n" ::"l"(__cvta_generic_to_grid_constant(&tensorMap)) : "memory");
        break;
    case StateSpace::kGENERIC:
        asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(&tensorMap)) : "memory");
        break;
    default: asm volatile("trap;\n");
    }
}

} // namespace tma
