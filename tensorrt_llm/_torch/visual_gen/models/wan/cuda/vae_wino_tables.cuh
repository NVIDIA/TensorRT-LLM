/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

// Temporal Winograd F(R,3) tables, mirrored in wino_vae_kernels.TEMPORAL_WINO.

#pragma once

namespace vaewino
{

namespace vae_wino
{

// Temporal Winograd F(R,3) tables. BT[k][i]: slab k from frame i of the group's R + 2
// frames; AT[j][k]: output frame j of the group from product k. Only the nonzero AT
// entries are read, in index order (the summation order the reference reproduces).
template <int R>
struct Wino;

template <>
struct Wino<2>
{
    static constexpr int kComps = 4;
    static constexpr int kMaxTerms = 3; // most nonzero AT entries in one row

    __host__ __device__ static constexpr float bt(int k, int i)
    {
        constexpr float t[4][4] = {{1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
        return t[k][i];
    }

    __host__ __device__ static constexpr float at(int j, int k)
    {
        constexpr float t[2][4] = {{1, 1, 1, 0}, {0, 1, -1, -1}};
        return t[j][k];
    }
};

template <>
struct Wino<4>
{
    static constexpr int kComps = 6;
    static constexpr int kMaxTerms = 5;

    __host__ __device__ static constexpr float bt(int k, int i)
    {
        constexpr float t[6][6] = {{1, 1.5f, -2, -1.5f, 1, 0}, {0, -1, -2.5f, -0.5f, 1, 0}, {0, 1, 0.5f, -2.5f, 1, 0},
            {0, -0.5f, -1, 0.5f, 1, 0}, {0, 2, -1, -2, 1, 0}, {0, 1, 1.5f, -2, -1.5f, 1}};
        return t[k][i];
    }

    __host__ __device__ static constexpr float at(int j, int k)
    {
        constexpr float t[4][6]
            = {{1, 1, 1, 1, 1, 0}, {0, 1, -1, 2, -0.5f, 0}, {0, 1, 1, 4, 0.25f, 0}, {0, 1, -1, 8, -0.125f, 1}};
        return t[j][k];
    }
};

// Index of the last frame slab k reads; the slab is emitted while that frame is normalized.
template <int R>
__host__ __device__ constexpr int lastFrameOf(int k)
{
    int last = 0;
    for (int i = 0; i < R + 2; ++i)
    {
        if (Wino<R>::bt(k, i) != 0.0f)
        {
            last = i;
        }
    }
    return last;
}

// Number of nonzero AT entries in row j, and the product index of the q-th of them.
template <int R>
__host__ __device__ constexpr int numTerms(int j)
{
    int n = 0;
    for (int k = 0; k < R + 2; ++k)
    {
        n += Wino<R>::at(j, k) != 0.0f ? 1 : 0;
    }
    return n;
}

template <int R>
__host__ __device__ constexpr int termIndex(int j, int q)
{
    int n = 0;
    for (int k = 0; k < R + 2; ++k)
    {
        if (Wino<R>::at(j, k) != 0.0f)
        {
            if (n == q)
            {
                return k;
            }
            ++n;
        }
    }
    return 0;
}

} // namespace vae_wino

} // namespace vaewino
