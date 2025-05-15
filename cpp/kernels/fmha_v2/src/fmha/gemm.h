/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
