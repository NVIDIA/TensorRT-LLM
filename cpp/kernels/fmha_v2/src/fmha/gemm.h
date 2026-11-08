/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <fmha/fragment.h>
#include <fmha/utils.h>

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Acc, typename A, typename B, int M, int N>
inline __device__ void gemm(Acc (&acc)[M][N], A const (&a)[M], B const (&b)[N])
{

#pragma unroll
    for (int mi = 0; mi < M; ++mi)
    {
#pragma unroll
        for (int ni = 0; ni < N; ++ni)
        {
            acc[mi][ni].mma(a[mi], b[ni]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
