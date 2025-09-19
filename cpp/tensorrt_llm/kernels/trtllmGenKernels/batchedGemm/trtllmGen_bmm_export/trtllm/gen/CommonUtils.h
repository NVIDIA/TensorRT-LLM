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

namespace batchedGemm
{

namespace trtllm
{
namespace gen
{

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// TMA OOB optimization constants.
//
// CUDA Programming Guide states that "globalDim must be non-zero and less than or equal to 2^32".
// In practice, the kernel acts funny with TMA shape of 2^32 so we use 2^31.
constexpr unsigned long TmaDimMax = 1UL << 31;
// Chosen so that LargeN * XLargeN * sizeof(dtype) >= 2^64 which causes overflow and effectively
// becomes 0. As sizeof(dtype) can be as small as 0.5B, we choose LargeN = 2^30 and XLargeN = 2^35
// so overflow can happen.
constexpr unsigned long LargeN = 1UL << 30;
// Used in TMA stride. Should be less than 2^40.
constexpr unsigned long XLargeN = 1UL << 35;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline T ceilDiv(T m, T n)
{
    return (m + n - T(1)) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline T roundUp(T m, T n)
{
    return ceilDiv(m, n) * n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gen
} // namespace trtllm

} // namespace batchedGemm
