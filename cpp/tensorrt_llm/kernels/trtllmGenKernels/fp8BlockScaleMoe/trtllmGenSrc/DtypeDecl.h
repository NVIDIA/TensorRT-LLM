/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
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
  E0m3     = TLLM_ENCODE_DTYPE(/*block*/ 1u, /*signed*/ 1u, /*int*/ 0u, /*bits*/   4u, /*uid*/  2u),
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
  UE8m0    = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 0u, /*int*/ 0u, /*bits*/   8u, /*uid*/ 14u),
  UInt8    = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 0u, /*int*/ 1u, /*bits*/   8u, /*uid*/ 15u),
  UInt16   = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 0u, /*int*/ 1u, /*bits*/  16u, /*uid*/ 16u),
  UInt32   = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 0u, /*int*/ 1u, /*bits*/  32u, /*uid*/ 17u),
  UInt64   = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 0u, /*int*/ 1u, /*bits*/  64u, /*uid*/ 18u),
  UInt128  = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 0u, /*int*/ 1u, /*bits*/ 128u, /*uid*/ 19u),
  Void     = TLLM_ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 0u, /*bits*/   0u, /*uid*/ 20u),
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

} // namespace gen
} // namespace trtllm
