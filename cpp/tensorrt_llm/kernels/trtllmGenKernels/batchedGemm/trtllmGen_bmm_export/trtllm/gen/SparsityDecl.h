/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
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
#include <string>

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Be careful when modifying this file as it is included by the generated kernels. For example, do
// not add TLLM_CHECK_* constructs in this file. Thanks!
//
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace batchedGemm
{

namespace trtllm
{
namespace gen
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// This enumeration defines structured sparsity modes. Please refer to the PTX ISA for more details.
enum class Sparsity
{
    // No sparsity.
    Dense,

    // For each chunk of 2 elements, 1 is non-zero. Only non-zero elements are stored.
    // A 4-bit index is used to indicate the position of the non-zero element.
    // The index may only take the value 0b1110 or 0b0100, other values are undefined behavior.
    //
    // 0b1110:                               0b0100:
    // |------ a ------|------ 0 ------|     |------ 0 ------|------ a ------|
    // |  11   |  10   |  01   |  00   |     |  11   |  10   |  01   |  00   |
    Any_1_2,

    // For each chunk of 4 elements, 2 are non-zero. Only non-zero elements are stored.
    // A 4-bit index is used to indicate the position of the non-zero elements.
    // Meaningful values are: 0b0100, 0b1000, 0b1100, 0b1001, 0b1101, 0b1110.
    // Most other values are undefined behavior.
    //
    // E.g. 0b1100 corresponds to:
    // |-- b --|-- 0 --|-- 0 --|-- a --|
    // |  11   |  10   |  01   |  00   |
    Any_2_4,

    // For each chunk of 8 elements, 4 are non-zero. Only non-zero elements are stored.
    // Further, the zero and non-zero elements are grouped in pairs.
    // A 4-bit index is used to indicate the position of the non-zero elements.
    // Meaningful values are: 0b0100, 0b1000, 0b1100, 0b1001, 0b1101, 0b1110.
    // Most other values are undefined behavior.
    //
    // E.g. 0b1100 corresponds to:
    // | d | c | 0 | 0 | 0 | 0 | b | a |
    // |  11   |  10   |  01   |  00   |
    Pairwise_4_8,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool isSparse(Sparsity sparsity)
{
    return sparsity != Sparsity::Dense;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string sparsityToString(Sparsity sparsity)
{
    switch (sparsity)
    {
    case Sparsity::Dense: return "dense";
    case Sparsity::Any_1_2: return "1:2";
    case Sparsity::Any_2_4: return "2:4";
    case Sparsity::Pairwise_4_8: return "4:8";
    default: assert(false); return "Unsupported sparsity";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Size of a sparsity chunk, for sparse modes.
inline int32_t getSparsityChunkSize(Sparsity sparsity)
{
    switch (sparsity)
    {
    case Sparsity::Any_1_2: return 2;
    case Sparsity::Any_2_4: return 4;
    case Sparsity::Pairwise_4_8: return 8;
    case Sparsity::Dense:
    default: assert(false); return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Number of bytes needed to store the sparsity information.
inline size_t getNumBytesSparsityInfo(Sparsity sparsity, size_t numElts)
{
    switch (sparsity)
    {
    case Sparsity::Dense: return 0;
    case Sparsity::Any_1_2:
    case Sparsity::Any_2_4:
    case Sparsity::Pairwise_4_8: return numElts / getSparsityChunkSize(sparsity) * 4 /*bits*/ / 8;
    default: assert(false); return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gen
} // namespace trtllm

} // namespace batchedGemm
