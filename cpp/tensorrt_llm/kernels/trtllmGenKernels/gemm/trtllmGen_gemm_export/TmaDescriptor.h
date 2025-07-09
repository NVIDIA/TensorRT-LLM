/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "trtllm/gen/DtypeDecl.h"
#include "trtllm/gen/MmaDecl.h"
#include <iostream>

#ifdef TLLM_ENABLE_CUDA
#include <cuda.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#endif

namespace gemm
{

namespace gemm
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef TLLM_ENABLE_CUDA

inline CUtensorMap buildNdTmaDescriptor(tg::Dtype dtype, tg::MmaKind mmaKind, std::vector<uint64_t> const& shapes,
    std::vector<uint64_t> const& strides, int32_t tileSizeMn, int32_t tileSizeK, void* gmemAddr, bool doSwizzle = true)
{
    // The multiplication factor of the data padding in SMEM.
    int32_t padMultiplier = 1;
    CUtensorMap desc{};
    // The data type.
    CUtensorMapDataType tmaDataFormat{CU_TENSOR_MAP_DATA_TYPE_FLOAT32};
    if (dtype == tg::Dtype::E4m3 || dtype == tg::Dtype::MxE4m3)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    }
    else if (dtype == tg::Dtype::Fp16)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    }
    else if (dtype == tg::Dtype::Bfloat16)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    }
    else if (dtype == tg::Dtype::E2m1)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    }
    else if (dtype == tg::Dtype::MxE2m1)
    {
        if (mmaKind == tg::MmaKind::MxFp8Fp6Fp4)
        {
            padMultiplier = 2;
            tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;
        }
        else if (mmaKind == tg::MmaKind::MxFp4NvFp4 || mmaKind == tg::MmaKind::Auto)
        {
            tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
        }
        else
        {
            std::cerr << "Invalid dtype / mmaKind combination " << tg::dtypeToString(dtype) << "/"
                      << tg::mmaKindToString(mmaKind) << std::endl;
            assert(false);
        }
    }
    else if (dtype == tg::Dtype::Fp32)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    }
    else
    {
        std::cerr << "buildNdTmaDescriptor: unexpected dtype " << tg::dtypeToString(dtype) << std::endl;
        assert(false);
    }

    // The swizzle type.
    CUtensorMapSwizzle swizzleType{CU_TENSOR_MAP_SWIZZLE_NONE};
    int32_t tileKSizeInBytes = (tileSizeK * tg::dtypeGetNumBits(dtype) * padMultiplier) / /* bits */ 8;
    if (doSwizzle)
    {
        if ((tileKSizeInBytes % 128) == 0)
        {
            swizzleType = CU_TENSOR_MAP_SWIZZLE_128B;
        }
        else if ((tileKSizeInBytes % 64) == 0)
        {
            swizzleType = CU_TENSOR_MAP_SWIZZLE_64B;
        }
        else if ((tileKSizeInBytes % 32) == 0)
        {
            swizzleType = CU_TENSOR_MAP_SWIZZLE_32B;
        }
        else
        {
            std::cerr << "buildNdTmaDescriptor: unexpected tileKSizeInBytes " << tileKSizeInBytes << std::endl;
            assert(false);
        }
    }

    // Check gmem address must be 16B-aligned
    assert((reinterpret_cast<uint64_t>(gmemAddr) & 0b1111) == 0); //

    // Check shape must be in range [1, 2^32]
    int32_t dim = shapes.size();
    // Expect 2 dimensions.
    assert(dim == 2 || dim == 3);
    // Check shape range.
    for (int32_t ii = 0; ii < dim; ++ii)
    {
        assert(shapes[ii] >= (uint64_t(1)));       // Size must be min 1
        assert(shapes[ii] <= (uint64_t(1) << 32)); // Size must be max 2^32
    }

    // TMA descriptor does not store the zeroth stride and assumes it is 1.
    assert(static_cast<int32_t>(strides.size()) == dim);
    assert(strides[0] == 1);

    // Build strides in bytes.
    // cuTensorMapEncodeTiled ignores the stride of the first dimension (implicitly 1).
    std::vector<uint64_t> stridesInBytes(dim - 1);
    for (int32_t ii = 0; ii < dim - 1; ++ii)
    {
        stridesInBytes[ii] = (strides[ii + 1] * tg::dtypeGetNumBits(dtype)) / /* bits */ 8;
    }

    // Set the number of elements in the packed uint32_t element.
    auto const numEltsPerUInt32 = 4 * /* bits */ 8 / (tg::dtypeGetNumBits(dtype) * padMultiplier);
    // The number of elements in 128B.
    auto const numEltsIn128B = numEltsPerUInt32 /*4B*/ * 32;
    // The number of tile K hidden size (per token) in each block of shared memory.
    auto const numEltsInClampedTileKSize = std::min(numEltsIn128B, tileSizeK);

    // Build tile shapes.
    std::vector<uint32_t> tileShapes(dim, 1);
    tileShapes[0] = numEltsInClampedTileKSize; // tileSizeK
    tileShapes[1] = tileSizeMn;                // tileSizeMn

    // Set tile strides to 1;
    std::vector<uint32_t> tileStrides(dim, 1);

    // Build the descriptor.
    CUresult result = cuTensorMapEncodeTiled(&desc, tmaDataFormat,
        /*tensorRank=*/dim, gmemAddr, shapes.data(), stridesInBytes.data(), tileShapes.data(), tileStrides.data(),
        /*interleave=*/CU_TENSOR_MAP_INTERLEAVE_NONE, swizzleType,
        /*l2Promotion=*/CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        /*oobFill=*/CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (result != CUDA_SUCCESS)
    {
        std::cerr << "Error: Failed to initialize the TMA descriptor " << result << std::endl;

        std::cerr << "tmaFormat: " << static_cast<int>(tmaDataFormat) << " dim: " << dim << " gmem: " << gmemAddr
                  << std::endl;

        std::cerr << "Shape: ";
        for (int ii = 0; ii < dim; ++ii)
        {
            std::cerr << shapes[ii] << " ";
        }
        std::cerr << std::endl;

        std::cerr << "Stride: ";
        for (int ii = 0; ii < dim - 1; ++ii)
        {
            std::cerr << stridesInBytes[ii] << " ";
        }
        std::cerr << std::endl;

        std::cerr << "tileShapes: ";
        for (int ii = 0; ii < dim; ++ii)
        {
            std::cerr << tileShapes[ii] << " ";
        }
        std::cerr << std::endl;

        std::cerr << "tileStrides: ";
        for (int ii = 0; ii < dim; ++ii)
        {
            std::cerr << tileStrides[ii] << " ";
        }
        std::cerr << std::endl;
        std::cerr << "swizzleType: " << int(swizzleType) << std::endl;
        assert(false);
    }

    return desc;
}

// TODO: make it work with the above descriptor?
inline CUtensorMap buildSfTmaDescriptor(tg::Dtype dtype, std::vector<uint64_t> const& shapes,
    std::vector<uint64_t> const& strides, std::vector<uint32_t> const& tileShapes, void* gmemAddr)
{
    CUtensorMap desc{};
    CUtensorMapDataType tmaDataFormat;
    if (dtype == tg::Dtype::E4m3 || dtype == tg::Dtype::UE8m0)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    }
    else
    {
        std::cerr << "buildSfTmaDescriptor: unexpected dtype " << tg::dtypeToString(dtype) << std::endl;
        assert(false);
    }

    // No swizzle for scaling factors.
    CUtensorMapSwizzle swizzleType = CU_TENSOR_MAP_SWIZZLE_NONE;

    // Check gmem address must be 16B-aligned
    assert((reinterpret_cast<uint64_t>(gmemAddr) & 0b1111) == 0); //

    // Check shape must be in range [1, 2^32]
    int32_t dim = shapes.size();
    // Check shape range.
    for (int32_t ii = 0; ii < dim; ++ii)
    {
        assert(shapes[ii] >= (uint64_t(1)));       // Size must be min 1
        assert(shapes[ii] <= (uint64_t(1) << 32)); // Size must be max 2^32
    }

    // TMA descriptor does not store the zeroth stride and assumes it is 1.
    assert(static_cast<int32_t>(strides.size()) == dim);
    assert(strides[0] == 1);

    // Build strides in bytes.
    // cuTensorMapEncodeTiled ignores the stride of the first dimension (implicitly 1).
    std::vector<uint64_t> stridesInBytes(dim - 1);
    for (int32_t ii = 0; ii < dim - 1; ++ii)
    {
        stridesInBytes[ii] = (strides[ii + 1] * tg::dtypeGetNumBits(dtype)) / /* bits */ 8;
    }

    // Set tile strides to 1;
    std::vector<uint32_t> tileStrides(dim, 1);

    // Build the descriptor.
    CUresult result = cuTensorMapEncodeTiled(/*tensorMap=*/&desc,
        /*tensorDataType=*/tmaDataFormat,
        /*tensorRank=*/dim,
        /*globalAddress=*/gmemAddr,
        /*globalDim=*/shapes.data(),
        /*globalStrides=*/stridesInBytes.data(),
        /*boxDim=*/tileShapes.data(),
        /*elementStrides=*/tileStrides.data(),
        /*interleave=*/CU_TENSOR_MAP_INTERLEAVE_NONE,
        /*swizzle=*/swizzleType,
        /*l2Promotion=*/CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        /*oobFill=*/CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (result != CUDA_SUCCESS)
    {
        std::cerr << "Error: Failed to initialize the TMA descriptor for SF " << result << std::endl;

        std::cerr << "tmaFormat: " << static_cast<int>(tmaDataFormat) << " dim: " << dim << " gmem: " << gmemAddr
                  << std::endl;

        std::cerr << "shape:";
        for (uint32_t shape_i : shapes)
        {
            std::cerr << " " << shape_i;
        }
        std::cerr << std::endl;

        std::cerr << "stridesInBytes:";
        for (uint32_t stride_i : stridesInBytes)
        {
            std::cerr << " " << stride_i;
        }
        std::cerr << std::endl;

        std::cerr << "tileShapes:";
        for (uint32_t tileShape_i : tileShapes)
        {
            std::cerr << " " << tileShape_i;
        }
        std::cerr << std::endl;

        std::cerr << "tileStrides:";
        for (uint32_t tileStride_i : tileStrides)
        {
            std::cerr << " " << tileStride_i;
        }
        std::cerr << std::endl;

        std::cerr << "swizzleType: " << int(swizzleType) << std::endl;
        assert(false);
    }

    return desc;
}

#endif // defined TLLM_ENABLE_CUDA

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm

} // namespace gemm
