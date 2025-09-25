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

namespace batchedGemm
{

namespace gemm
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef TLLM_ENABLE_CUDA

inline CUtensorMap buildNdTmaDescriptor(tg::Dtype dtype, tg::MmaKind mmaKind, std::vector<uint64_t> const& shapes,
    std::vector<uint64_t> const& strides, std::vector<int32_t> const& tileShapes, void* gmemAddr, bool doSwizzle = true)
{
    // The multiplication factor of the data padding in SMEM.
    int32_t padMultiplier = 1;
    CUtensorMap desc{};
    // The data type.
    CUtensorMapDataType tmaDataFormat{CU_TENSOR_MAP_DATA_TYPE_FLOAT32};
    if (dtype == tg::Dtype::E4m3 || dtype == tg::Dtype::MxE4m3 || dtype == tg::Dtype::UE8m0)
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
        else
        {
            // Note: this is used with the MMA kind MxFp4NvFp4 and also when casting to a higher-precision
            // type such as Bfloat16 before the MMA.
            tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
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
    int32_t fastestDimTileSizeBytes = (tileShapes[0] * tg::dtypeGetNumBits(dtype) * padMultiplier) / /* bits */ 8;
    if (doSwizzle)
    {
        if ((fastestDimTileSizeBytes % 128) == 0)
        {
            swizzleType = CU_TENSOR_MAP_SWIZZLE_128B;
        }
        else if ((fastestDimTileSizeBytes % 64) == 0)
        {
            swizzleType = CU_TENSOR_MAP_SWIZZLE_64B;
        }
        else if ((fastestDimTileSizeBytes % 32) == 0)
        {
            swizzleType = CU_TENSOR_MAP_SWIZZLE_32B;
            // This path is only for the scaling factors.
        }
        else if ((fastestDimTileSizeBytes % 16) == 0 && (dtype == tg::Dtype::UE8m0 || dtype == tg::Dtype::E4m3))
        {
            swizzleType = CU_TENSOR_MAP_SWIZZLE_NONE;
        }
        else
        {
            std::cerr << "buildNdTmaDescriptor: unexpected fastestDimTileSizeBytes " << fastestDimTileSizeBytes
                      << std::endl;
            assert(false);
        }
    }

    // Check gmem address must be 16B-aligned
    assert((reinterpret_cast<uint64_t>(gmemAddr) & 0b1111) == 0); //

    // Check shape must be in range [1, 2^32]
    int32_t dim = shapes.size();
    // Expect 2 dimensions for regular gemm, 3 dimensions for batched gemm or blocked layout, and 4
    // dimensions for batched gemm with blocked layout.
    assert(dim == 2 || dim == 3 || dim == 4);
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
    auto const numEltsInClampedFastestTileSize = std::min(numEltsIn128B, tileShapes[0]);

    // Build box dim array. If tileShapes is smaller than dim, just fill with 1s.
    assert(static_cast<int32_t>(tileShapes.size()) <= dim);
    std::vector<uint32_t> boxDim(dim, 1);
    boxDim[0] = numEltsInClampedFastestTileSize;
    for (size_t ii = 1; ii < tileShapes.size(); ++ii)
    {
        if (tileShapes[ii] > 256)
        {
            std::cerr << "buildNdTmaDescriptor: boxDim too large " << tileShapes[ii] << std::endl;
            assert(false);
        }
        else
        {
            boxDim[ii] = tileShapes[ii];
        }
    }

    // Set tile strides to 1;
    std::vector<uint32_t> tileStrides(dim, 1);

    // Build the descriptor.
    CUresult result = cuTensorMapEncodeTiled(&desc, tmaDataFormat,
        /*tensorRank=*/dim, gmemAddr, shapes.data(), stridesInBytes.data(), boxDim.data(), tileStrides.data(),
        /*interleave=*/CU_TENSOR_MAP_INTERLEAVE_NONE, swizzleType,
        /*l2Promotion=*/CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        /*oobFill=*/CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (result != CUDA_SUCCESS)
    {
        char const* errorString;
        cuGetErrorString(result, &errorString);
        std::stringstream ss;
        ss << "Error: Failed to initialize the TMA descriptor. " << errorString << std::endl;

        ss << "tmaFormat: " << static_cast<int>(tmaDataFormat) << " dim: " << dim << " gmem: " << gmemAddr << std::endl;

        ss << "Shape: ";
        for (int ii = 0; ii < dim; ++ii)
        {
            ss << shapes[ii] << " ";
        }
        ss << std::endl;

        ss << "Stride: ";
        for (int ii = 0; ii < dim - 1; ++ii)
        {
            ss << stridesInBytes[ii] << " ";
        }
        ss << std::endl;

        ss << "tileShapes: ";
        for (int ii = 0; ii < dim; ++ii)
        {
            ss << boxDim[ii] << " ";
        }
        ss << std::endl;

        ss << "tileStrides: ";
        for (int ii = 0; ii < dim; ++ii)
        {
            ss << tileStrides[ii] << " ";
        }
        ss << std::endl;
        ss << "swizzleType: " << int(swizzleType) << std::endl;
        ss << "(in " << __FILE__ << ":" << __LINE__ << ")" << std::endl;
        throw std::runtime_error(ss.str());
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
        char const* errorString;
        cuGetErrorString(result, &errorString);
        std::stringstream ss;
        ss << "Error: Failed to initialize the TMA descriptor for SF. " << errorString << std::endl;

        ss << "tmaFormat: " << static_cast<int>(tmaDataFormat) << " dim: " << dim << " gmem: " << gmemAddr << std::endl;

        ss << "shape:";
        for (uint32_t shape_i : shapes)
        {
            ss << " " << shape_i;
        }
        ss << std::endl;

        ss << "stridesInBytes:";
        for (uint32_t stride_i : stridesInBytes)
        {
            ss << " " << stride_i;
        }
        ss << std::endl;

        ss << "tileShapes:";
        for (uint32_t tileShape_i : tileShapes)
        {
            ss << " " << tileShape_i;
        }
        ss << std::endl;

        ss << "tileStrides:";
        for (uint32_t tileStride_i : tileStrides)
        {
            ss << " " << tileStride_i;
        }
        ss << std::endl;

        ss << "swizzleType: " << int(swizzleType) << std::endl;
        ss << "(in " << __FILE__ << ":" << __LINE__ << ")" << std::endl;
        throw std::runtime_error(ss.str());
    }

    return desc;
}

#endif // defined TLLM_ENABLE_CUDA

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm

} // namespace batchedGemm
