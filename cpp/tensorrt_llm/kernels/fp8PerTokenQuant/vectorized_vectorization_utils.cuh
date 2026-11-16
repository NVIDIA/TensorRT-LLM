/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

// Adapted from vLLM (Apache-2.0):
// https://github.com/vllm-project/vllm/blob/v0.25.0/csrc/libtorch_stable/quantization/vectorization_utils.cuh

#pragma once
#include "vectorization.cuh"
#include <type_traits>

namespace vllm
{

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
struct DefaultVecOp
{
    ScaOp scalar_op;

    __device__ __forceinline__ void operator()(vec_n_t<OutT, VEC_SIZE>& dst, vec_n_t<InT, VEC_SIZE> const& src) const
    {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
        {
            scalar_op(dst.val[i], src.val[i]);
        }
    }
};

template <int VEC_SIZE, typename InT, typename OutT, typename VecOp, typename ScaOp>
__device__ inline void vectorize_with_alignment(InT const* in, OutT* out, int len, int tid, int stride,
    VecOp&& vec_op,                                    // vec_n_t<InT,16> -> vec_n_t<OutT,16>
    ScaOp&& scalar_op)
{                                                      // InT -> OutT
    static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0, "VEC_SIZE must be a positive power-of-two");
    constexpr int WIDTH = VEC_SIZE * sizeof(InT);      // eg: 16 B
    constexpr int OUT_WIDTH = VEC_SIZE * sizeof(OutT); // eg: 16 B
    uintptr_t addr = reinterpret_cast<uintptr_t>(in);
    uintptr_t out_addr = reinterpret_cast<uintptr_t>(out);

    bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((out_addr & (OUT_WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
    if (can_vec)
    {
        int num_vec = len / VEC_SIZE;

        using vin_t = vec_n_t<InT, VEC_SIZE>;
        using vout_t = vec_n_t<OutT, VEC_SIZE>;
        auto* v_in = reinterpret_cast<vin_t const*>(in);
        auto* v_out = reinterpret_cast<vout_t*>(out);

        for (int i = tid; i < num_vec; i += stride)
        {
            vout_t tmp;
            vin_t src = v_in[i];
            vec_op(tmp, src);
            v_out[i] = tmp;
        }
        return;
    }

    int misalignment_offset = addr & (WIDTH - 1);
    int alignment_bytes = WIDTH - misalignment_offset;
    int prefix_elems = alignment_bytes & (WIDTH - 1);
    prefix_elems /= sizeof(InT);
    prefix_elems = min(prefix_elems, len);

    if (((out_addr + prefix_elems * sizeof(OutT)) & (OUT_WIDTH - 1)) != 0)
    {
        for (int i = tid; i < len; i += stride)
        {
            scalar_op(out[i], in[i]);
        }
        return;
    }

    for (int i = tid; i < prefix_elems; i += stride)
    {
        scalar_op(out[i], in[i]);
    }

    in += prefix_elems;
    out += prefix_elems;
    len -= prefix_elems;

    int num_vec = len / VEC_SIZE;
    using vin_t = vec_n_t<InT, VEC_SIZE>;
    using vout_t = vec_n_t<OutT, VEC_SIZE>;
    auto* v_in = reinterpret_cast<vin_t const*>(in);
    auto* v_out = reinterpret_cast<vout_t*>(out);

    for (int i = tid; i < num_vec; i += stride)
    {
        vout_t tmp;
        vin_t src = v_in[i];
        vec_op(tmp, src);
        v_out[i] = tmp;
    }

    int tail_start = num_vec * VEC_SIZE;
    for (int i = tid + tail_start; i < len; i += stride)
    {
        scalar_op(out[i], in[i]);
    }
}

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
__device__ __forceinline__ void vectorize_with_alignment(
    InT const* in, OutT* out, int len, int tid, int stride, ScaOp&& scalar_op)
{
    using Vec = DefaultVecOp<VEC_SIZE, InT, OutT, std::decay_t<ScaOp>>;
    vectorize_with_alignment<VEC_SIZE>(in, out, len, tid, stride, Vec{scalar_op}, std::forward<ScaOp>(scalar_op));
}

template <int VEC_SIZE, typename InT, typename ScaOp>
struct DefaultReadVecOp
{
    ScaOp scalar_op;

    __device__ __forceinline__ void operator()(vec_n_t<InT, VEC_SIZE> const& src) const
    {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
        {
            scalar_op(src.val[i]);
        }
    }
};

template <int VEC_SIZE, typename InT, typename VecOp, typename ScaOp>
__device__ inline void vectorize_read_with_alignment(
    InT const* in, int len, int tid, int stride, VecOp&& vec_op, ScaOp&& scalar_op)
{
    static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0, "VEC_SIZE must be a positive power-of-two");
    constexpr int WIDTH = VEC_SIZE * sizeof(InT);
    uintptr_t addr = reinterpret_cast<uintptr_t>(in);

    bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
    if (can_vec)
    {
        int num_vec = len / VEC_SIZE;

        using vin_t = vec_n_t<InT, VEC_SIZE>;
        auto* v_in = reinterpret_cast<vin_t const*>(in);

        for (int i = tid; i < num_vec; i += stride)
        {
            vin_t tmp = v_in[i];
            vec_op(tmp);
        }
        return;
    }

    int misalignment_offset = addr & (WIDTH - 1);
    int alignment_bytes = WIDTH - misalignment_offset;
    int prefix_elems = alignment_bytes & (WIDTH - 1);
    prefix_elems /= sizeof(InT);
    prefix_elems = min(prefix_elems, len);

    for (int i = tid; i < prefix_elems; i += stride)
    {
        scalar_op(in[i]);
    }

    in += prefix_elems;
    len -= prefix_elems;

    int num_vec = len / VEC_SIZE;
    using vin_t = vec_n_t<InT, VEC_SIZE>;
    auto* v_in = reinterpret_cast<vin_t const*>(in);

    for (int i = tid; i < num_vec; i += stride)
    {
        vec_op(v_in[i]);
    }

    int tail_start = num_vec * VEC_SIZE;
    for (int i = tid + tail_start; i < len; i += stride)
    {
        scalar_op(in[i]);
    }
}

template <int VEC_SIZE, typename InT, typename ScaOp>
__device__ __forceinline__ void vectorize_read_with_alignment(
    InT const* in, int len, int tid, int stride, ScaOp&& scalar_op)
{
    using Vec = DefaultReadVecOp<VEC_SIZE, InT, std::decay_t<ScaOp>>;
    vectorize_read_with_alignment<VEC_SIZE>(in, len, tid, stride, Vec{scalar_op}, std::forward<ScaOp>(scalar_op));
}

} // namespace vllm
