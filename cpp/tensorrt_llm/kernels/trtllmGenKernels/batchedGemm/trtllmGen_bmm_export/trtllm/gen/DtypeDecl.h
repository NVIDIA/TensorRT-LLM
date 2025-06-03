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
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Be careful when modifying this file as it is included by the generated kernels. For example, do
// not add TLLM_CHECK_* constructs in this file. Thanks!
//
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace trtllm
{
namespace gen
{

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class Dtype : uint32_t
{

// We use the following encoding for the types:
//
// Byte 0: Identifier for the type (going from 0 to the number of data types - 1,
// Byte 1: Number of bits in the type,
// Byte 2: Bit 0: Is it an integer? 0x1 if true, 0x0 otherwise;
//         Bit 4: is it signed?  0x1 if true, 0x0 otherwise.
// Byte 3: Is it a block format? 0x1 if true, 0x0 otherwise.

#define TLLM_ENCODE_DTYPE(BlockFormatBit, SignedBit, IntegerBit, NumBits, Uid)                                         \
    uint32_t                                                                                                           \
    {                                                                                                                  \
        (BlockFormatBit << 24) | (SignedBit << 20) | (IntegerBit << 16) | (NumBits << 8) | (Uid)                       \
    }

    // clang-format off
  Bfloat16 = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 0u, /*bits*/  16u, /*uid*/  0u),
  Bool     = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 0u, /*int*/ 1u, /*bits*/   1u, /*uid*/  1u),
  PadType  = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 1u, /*bits*/   8u, /*uid*/  2u),
  E2m1     = TLLM_ENCODE_DTYPE(/*block*/ 1u, /*signed*/ 1u, /*int*/ 0u, /*bits*/   4u, /*uid*/  3u),
  E2m3     = TLLM_ENCODE_DTYPE(/*block*/ 1u, /*signed*/ 1u, /*int*/ 0u, /*bits*/   6u, /*uid*/  4u),
  E3m2     = TLLM_ENCODE_DTYPE(/*block*/ 1u, /*signed*/ 1u, /*int*/ 0u, /*bits*/   6u, /*uid*/  5u),
  E4m3     = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 0u, /*bits*/   8u, /*uid*/  6u),
  E5m2     = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 0u, /*bits*/   8u, /*uid*/  7u),
  Fp16     = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 0u, /*bits*/  16u, /*uid*/  8u),
  Fp32     = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 0u, /*bits*/  32u, /*uid*/  9u),
  Int8     = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 1u, /*bits*/   8u, /*uid*/ 10u),
  Int32    = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 1u, /*bits*/  32u, /*uid*/ 11u),
  Int64    = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 1u, /*bits*/  64u, /*uid*/ 12u),
  MxE2m1   = TLLM_ENCODE_DTYPE(/*block*/ 1u, /*signed*/ 1u, /*int*/ 0u, /*bits*/   4u, /*uid*/ 13u),
  MxE4m3   = TLLM_ENCODE_DTYPE(/*block*/ 1u, /*signed*/ 1u, /*int*/ 0u, /*bits*/   8u, /*uid*/ 14u),
  UE8m0    = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 0u, /*int*/ 0u, /*bits*/   8u, /*uid*/ 15u),
  UInt8    = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 0u, /*int*/ 1u, /*bits*/   8u, /*uid*/ 16u),
  UInt16   = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 0u, /*int*/ 1u, /*bits*/  16u, /*uid*/ 17u),
  UInt32   = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 0u, /*int*/ 1u, /*bits*/  32u, /*uid*/ 18u),
  UInt64   = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 0u, /*int*/ 1u, /*bits*/  64u, /*uid*/ 19u),
  UInt128  = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 0u, /*int*/ 1u, /*bits*/ 128u, /*uid*/ 20u),
  Void     = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 0u, /*bits*/   0u, /*uid*/ 21u),
// clang-format on

#undef TLLM_ENCODE_DTYPE
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// The number of bits in a data type?
inline int dtypeGetNumBits(Dtype dtype)
{
    constexpr uint32_t kMask = 0xffu << 8;
    return static_cast<int>((static_cast<uint32_t>(dtype) & kMask) >> 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Does the format use block scaling?
inline bool dtypeIsBlockFmt(Dtype dtype)
{
    constexpr uint32_t kMask = 0xffu << 24;
    return static_cast<bool>((static_cast<uint32_t>(dtype) & kMask) >> 24);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Is a given data type a floating-point type?
inline bool dtypeIsFloat(Dtype dtype)
{
    constexpr uint32_t kMask = 0x1u << 16;
    return dtype != Dtype::Void && 0 == (static_cast<uint32_t>(dtype) & kMask);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Is a given data type an 8-bit floating-point type?
inline bool dtypeIsFp8(Dtype dtype)
{
    return dtype == Dtype::E4m3 || dtype == Dtype::E5m2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Is a given data type an integer type?
inline bool dtypeIsInt(Dtype dtype)
{
    constexpr uint32_t kMask = 0x1u << 16;
    return (dtype != Dtype::Bool) && (0 != (static_cast<uint32_t>(dtype) & kMask));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Is a given data type signed?
inline bool dtypeIsSigned(Dtype dtype)
{
    constexpr uint32_t kMask = 0x1u << 20;
    return (0 != (static_cast<uint32_t>(dtype) & kMask));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// For logging and error reporting
inline std::string dtypeToString(Dtype dtype)
{
    switch (dtype)
    {
    case Dtype::Bfloat16: return "Bfloat16";
    case Dtype::Bool: return "Bool";
    case Dtype::E2m1: return "E2m1";
    case Dtype::E2m3: return "E2m3";
    case Dtype::E3m2: return "E3m2";
    case Dtype::E4m3: return "E4m3";
    case Dtype::E5m2: return "E5m2";
    case Dtype::Fp16: return "Fp16";
    case Dtype::Fp32: return "Fp32";
    case Dtype::Int8: return "Int8";
    case Dtype::Int32: return "Int32";
    case Dtype::Int64: return "Int64";
    case Dtype::MxE4m3: return "MxE4m3";
    case Dtype::UE8m0: return "UE8m0";
    case Dtype::UInt8: return "UInt8";
    case Dtype::UInt16: return "UInt16";
    case Dtype::UInt32: return "UInt32";
    case Dtype::UInt64: return "UInt64";
    case Dtype::UInt128: return "UInt128";
    case Dtype::Void: return "Void";
    default: assert(false); return "Unsupported type";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline Dtype dtypeEltType(Dtype dtype)
{
    switch (dtype)
    {
    case Dtype::MxE2m1: return Dtype::E2m1;
    case Dtype::MxE4m3: return Dtype::E4m3;
    default: return dtype;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int dtypeNumEltsPerSf(Dtype dtype)
{
    switch (dtype)
    {
    case Dtype::E2m1: return 16;
    case Dtype::MxE2m1:
    case Dtype::MxE4m3: return 32;
    default: assert(false); return -1;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Returns the dtype of scaling factors, if applicable.
inline Dtype dtypeGetBlockSfType(Dtype dtype)
{
    switch (dtype)
    {
    case Dtype::E2m1: return Dtype::E4m3;
    case Dtype::MxE2m1:
    case Dtype::MxE4m3: return Dtype::UE8m0;
    default: assert(false); return Dtype::Void;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gen
} // namespace trtllm
