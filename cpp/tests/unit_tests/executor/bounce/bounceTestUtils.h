/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// NIXL-free helpers shared by ALL bounce test tiers (pure-GPU tests include this directly;
// the real-NIXL tier gets it through bounceTestNixlNode.h): the CUDA availability probe and
// alignment math. The byte-pattern generators intentionally stay with their users — the
// gather/scatter kernel test and the NIXL node harness use DIFFERENT formulas.

#include <cuda_runtime_api.h>

#include <cstdint>

namespace bounce_test
{

inline bool hasCuda()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

inline std::uint64_t alignUp(std::uint64_t v, std::uint64_t a)
{
    return (v + a - 1) / a * a;
}

} // namespace bounce_test
