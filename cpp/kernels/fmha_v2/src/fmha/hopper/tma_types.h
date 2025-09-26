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

#include <fmha/numeric_types.h>

namespace fmha
{

// TMA desc type.
typedef enum
{
    TILED = 0,
    IM2COL
} cudaTmaDescType;

// TMA swizzle type.
typedef enum
{
    SWIZZLE_DISABLED,
    SWIZZLE_32B,
    SWIZZLE_64B,
    SWIZZLE_128B,
    SWIZZLE_MAX
} cudaTmaDescSwizzle;

typedef enum
{
    BARRIER64,
    BARRIER128
} cudaTmaDescBarrier;

// TMA interleave type.
typedef enum
{
    INTERLEAVE_DISABLED,
    INTERLEAVE_16B,
    INTERLEAVE_32B,
    INTERLEAVE_MAX
} cudaTmaDescInterleave;

// TMA L2 sector promotion.
typedef enum
{
    PROMOTION_DISABLED = 0,
    PROMOTION_64B,
    PROMOTION_128B,
    PROMOTION_256B
} cudaTmaDescPromotion;

// TMA data type.
typedef enum
{
    U8 = 0,
    U16,
    U32,
    S32,
    U64,
    S64,
    F16_RN,
    F32_RN,
    F32_FTZ_RN,
    F64_RN,
    BF16_RN,
    FORMAT_MAX
} cudaTmaDescFormat;

// TMA cache control.
typedef enum
{
    PREFETCH,      // Prefetch tma descriptor using global memory address
    INVALIDATE,    // Invalidate tma descriptor in l2 cache
    INVALIDATE_ALL // Invalidate tma descriptor and all elements in l2 cache line
} cudaTmaDescCacheCtrl;

// TMA OOB fill modes.
typedef enum
{
    TENSOR_ZFILL,
    TENSOR_CFILL
} cudaTmaDescOobFillMode;

constexpr uint64_t k_max_tensor_size = (1llu << 36);
constexpr uint64_t k_max_tensor_stride = (1llu << 36);
constexpr uint64_t k_max_block_size = 256llu;
constexpr uint64_t k_max_traversal_stride = (1llu << 3);

constexpr uint64_t k_min_tensor_size = 1llu;
constexpr uint64_t k_min_tensor_stride = 0llu;
constexpr uint64_t k_min_block_size = 1llu;
constexpr uint64_t k_min_traversal_stride = 1llu;

constexpr uint32_t k_max_cta_id = (1 << 6) - 1;

// The 512 bit of descriptor for tiled mode.
typedef struct
{
    uint64_t tensor_common0;
    uint32_t tensor_common1;

    uint32_t tensor_stride_lower[4]; //< 36b of 64b with 4B aligned
    uint32_t tensor_stride_upper;
    uint32_t tensor_size[5];         //< value -1
    uint32_t traversal_stride_box_0; //< packed 3b (-1)

    uint32_t box_size_end;
} cudaTmaDescTiled;

// The 512 bit of descritptro for im2col mode.
typedef struct
{
    uint64_t tensor_common0;
    uint32_t tensor_common1;

    uint32_t tensor_stride_lower[4];
    uint32_t tensor_stride_upper;
    uint32_t tensor_size[5];
    uint32_t traversal_stride_range_c;

    uint32_t box_corner_dhw;
    uint32_t range_ndhw;
} cudaTmaDescIm2Col;

// TMA desc size
constexpr uint32_t TMA_DESC_SIZE_IN_BYTE = 64;

// TMA desc
typedef struct alignas(64)
{
    uint64_t data[8];
} cudaTmaDesc;

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace fmha
