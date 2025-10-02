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

#include <cassert>
#include <cstdint>
#include <string>
#ifndef TLLM_GEN_EXPORT_INTERFACE
#include "trtllm/gen/CommonUtils.h"
#else  // TLLM_GEN_EXPORT_INTERFACE
#include "CommonUtils.h"
#endif // TLLM_GEN_EXPORT_INTERFACE

namespace batchedGemm
{

namespace trtllm
{
namespace gen
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// The kind of the MMA instruction
enum class MmaKind : uint32_t
{
    // For Blackwell this follows the PTX ISA description of the MMA instructions.
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-kind-shapes

    // The MMA type is auto-detected from the dtypes of the input tensors
    Auto = 0,
    // Supports dtypeA = dtypeB = Fp16 and dtypeD = [Fp16, Fp32]
    // or dtypeA = dtypeB = Bfloat16 and dtypeD = [Fp32]
    // Corresponds to the kind::f16 of tcgen05.mma.
    Fp16 = 1,
    // Supports dtypeA/B = [E4m3, E5m2, E2m3, E3m2, E2m1] and dtypeD = [Fp16, Fp32]
    // Corresponds to the kind::f8f6f4 of tcgen05.mma.
    Fp8Fp6Fp4 = 2,
    // Supports dtypeA = dtypeB = [Int8, Uint8] and dtypeD = [Int32]
    // Corresponds to the kind::i8 of tcgen05.mma.
    Int8 = 3,
    // Supports dtypeA = dtypeB = [MxE2m1, E2m1] with block scale [UM8e0, UEm4e3]
    // and dtypeD = [Fp32]
    // Corresponds to the kind::mxf4nvf4 of tcgen05.mma.
    MxFp4NvFp4 = 4,
    // Supports dtype dtypeA = dtypeB = [MxE4m3, MxE2m1] with block scale [UM8e0]
    // and dtypeD = [Fp32]
    // Corresponds to the kind::mxf8f6f4 of tcgen05.mma.
    MxFp8Fp6Fp4 = 5,
    // Supports dtypeA = dtypeB = Tf32 with dtypeD = [Fp32]
    // Corresponds to the kind::tf32 of tcgen05.mma.
    Tf32 = 6
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool mmaKindIsBlockFmt(MmaKind mmaKind)
{
    return mmaKind == MmaKind::MxFp8Fp6Fp4 || mmaKind == MmaKind::MxFp4NvFp4;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// For logging and error reporting
inline std::string mmaKindToString(MmaKind mmaKind)
{
    switch (mmaKind)
    {
    case MmaKind::Auto: return "Auto";
    case MmaKind::Fp16: return "Fp16";
    case MmaKind::Fp8Fp6Fp4: return "Fp8Fp6Fp4";
    case MmaKind::Int8: return "Int8";
    case MmaKind::MxFp4NvFp4: return "MxFp4NvFp4";
    case MmaKind::MxFp8Fp6Fp4: return "MxFp8Fp6Fp4";
    case MmaKind::Tf32: return "Tf32";
    default: assert(false); return "Unsupported type";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// function to get the TMEM column stride per group (i.e., 64 K elements)
inline int32_t getTmemColStridePerGroup(int32_t tileMn, int32_t mmaK)
{
    // Calculate the stride of TMEM column for every 64 elements in the K dimension
    int32_t div = 2 * ceilDiv(tileMn, 64);
    return mmaK == 96 ? std::max(4, div) : div;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gen
} // namespace trtllm

} // namespace batchedGemm
