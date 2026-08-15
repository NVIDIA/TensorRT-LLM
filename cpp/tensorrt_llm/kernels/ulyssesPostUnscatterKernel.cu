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

#include "tensorrt_llm/common/cudaUtils.h"
#include "ulyssesPostUnscatterKernel.h"
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// Each block copies one (p, b, psp) tile of ONE tensor (H*D bf16) into NHD storage
// [B, P*Sp, H, D]. Q/K/V may differ in (Sp, H) (cross-attn: Q=audio vs K/V=video): grid.x
// is packed over all three tensors' tiles (P*Sp_q + P*Sp_k + P*Sp_v) and each block maps
// blockIdx.x to its tensor, so no block is idle even when shapes differ. Block is sized to
// max(H)*(D/8); threads with h >= H idle only when H differs (GQA).
template <typename T>
__global__ void ulyssesPostUnscatterKernel(T const* __restrict__ q_in, T const* __restrict__ k_in,
    T const* __restrict__ v_in, T* __restrict__ q_out, T* __restrict__ k_out, T* __restrict__ v_out, int const P,
    int const B, int const D, int const vec_per_row, int const Sp_q, int const H_q, int const Sp_k, int const H_k,
    int const Sp_v, int const H_v)
{
    constexpr int VEC = 8;

    // blockIdx.x packed over Q|K|V tiles: [0, P*Sp_q) -> Q, next P*Sp_k -> K, rest -> V.
    int const nq = P * Sp_q;
    int const nk = P * Sp_k;
    int const bx = blockIdx.x;

    T const* in_ptr;
    T* out_ptr;
    int Sp, H, psp;
    if (bx < nq)
    {
        in_ptr = q_in;
        out_ptr = q_out;
        Sp = Sp_q;
        H = H_q;
        psp = bx;
    }
    else if (bx < nq + nk)
    {
        in_ptr = k_in;
        out_ptr = k_out;
        Sp = Sp_k;
        H = H_k;
        psp = bx - nq;
    }
    else
    {
        in_ptr = v_in;
        out_ptr = v_out;
        Sp = Sp_v;
        H = H_v;
        psp = bx - nq - nk;
    }

    int const h = threadIdx.x / vec_per_row;
    if (h >= H) // block sized to max(H); mask extra heads only when H differs (GQA)
        return;
    int const vec_idx = threadIdx.x - h * vec_per_row;

    int const p = psp / Sp;
    int const sp = psp - p * Sp;
    int const b = blockIdx.y;
    int const PSp = P * Sp;

    // in[p, b, sp, h, d]: ((((p*B + b)*Sp + sp)*H + h)*D + vec_idx*VEC)
    // NHD out[b, p*Sp+sp, h, d]: (((b*PSp + psp)*H + h)*D + vec_idx*VEC)
    // int64_t: P*B*Sp*H*D can exceed 2^31 at large workloads.
    int64_t const in_base = ((((static_cast<int64_t>(p) * B + b) * Sp + sp) * H + h) * D) + vec_idx * VEC;
    int64_t const out_base = (((static_cast<int64_t>(b) * PSp + psp) * H + h) * D) + vec_idx * VEC;

    uint4 const* in_v4 = reinterpret_cast<uint4 const*>(in_ptr + in_base);
    uint4* out_v4 = reinterpret_cast<uint4*>(out_ptr + out_base);
    *out_v4 = *in_v4;
}

} // namespace

void launchUlyssesPostUnscatter(void const* q_in, void const* k_in, void const* v_in, void* q_out, void* k_out,
    void* v_out, int P, int B, int D, int Sp_q, int H_q, int Sp_k, int H_k, int Sp_v, int H_v, cudaStream_t stream)
{
    constexpr int VEC = 8;
    TLLM_CHECK_WITH_INFO(D % VEC == 0, "ulyssesPostUnscatter: D must be a multiple of 8 (uint4 vec), got %d", D);
    int const vec_per_row = D / VEC;
    int const H_max = std::max(H_q, std::max(H_k, H_v));
    int const threads = H_max * vec_per_row;
    TLLM_CHECK_WITH_INFO(threads <= 1024,
        "ulyssesPostUnscatter: threads/block (maxH*D/8) must be <= 1024, got maxH=%d D=%d -> %d", H_max, D, threads);

    int const total_tiles = P * (Sp_q + Sp_k + Sp_v); // packed over Q/K/V, no idle blocks
    dim3 const grid(total_tiles, B, 1);
    dim3 const block(threads);

    auto* q_in_typed = reinterpret_cast<__nv_bfloat16 const*>(q_in);
    auto* k_in_typed = reinterpret_cast<__nv_bfloat16 const*>(k_in);
    auto* v_in_typed = reinterpret_cast<__nv_bfloat16 const*>(v_in);
    auto* q_out_typed = reinterpret_cast<__nv_bfloat16*>(q_out);
    auto* k_out_typed = reinterpret_cast<__nv_bfloat16*>(k_out);
    auto* v_out_typed = reinterpret_cast<__nv_bfloat16*>(v_out);

    ulyssesPostUnscatterKernel<__nv_bfloat16><<<grid, block, 0, stream>>>(q_in_typed, k_in_typed, v_in_typed,
        q_out_typed, k_out_typed, v_out_typed, P, B, D, vec_per_row, Sp_q, H_q, Sp_k, H_k, Sp_v, H_v);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
