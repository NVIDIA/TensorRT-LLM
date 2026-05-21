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
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// Each block handles one (p, b, sp) tile across all H heads.
// Reads H*D bf16 contiguous from input; writes the same bytes scattered across
// H rows of the output layout [B, H, P*Sp, D].
//
// threads/block = H * (D / 8); each thread copies one uint4 (8 bf16).
template <typename T>
__global__ void ulyssesPostUnscatterKernel(T const* __restrict__ q_in, T const* __restrict__ k_in,
    T const* __restrict__ v_in, T* __restrict__ q_out, T* __restrict__ k_out, T* __restrict__ v_out, int const P,
    int const B, int const Sp, int const H, int const D, int const vec_per_row)
{
    constexpr int VEC = 8;

    int const h = threadIdx.x / vec_per_row;
    int const vec_idx = threadIdx.x - h * vec_per_row;

    int const psp = blockIdx.x; // 0 .. P*Sp-1
    int const p = psp / Sp;
    int const sp = psp - p * Sp;
    int const b = blockIdx.y;
    int const PSp = P * Sp;

    T const* in_ptr;
    T* out_ptr;
    switch (blockIdx.z)
    {
    case 0:
        in_ptr = q_in;
        out_ptr = q_out;
        break;
    case 1:
        in_ptr = k_in;
        out_ptr = k_out;
        break;
    default:
        in_ptr = v_in;
        out_ptr = v_out;
        break;
    }

    // in[p, b, sp, h, d]: ((((p*B + b)*Sp + sp)*H + h)*D + vec_idx*VEC)
    int const in_base = ((((p * B + b) * Sp + sp) * H + h) * D) + vec_idx * VEC;
    // out[b, h, p*Sp+sp, d]: (((b*H + h)*PSp + psp)*D + vec_idx*VEC)
    int const out_base = (((b * H + h) * PSp) + psp) * D + vec_idx * VEC;

    uint4 const* in_v4 = reinterpret_cast<uint4 const*>(in_ptr + in_base);
    uint4* out_v4 = reinterpret_cast<uint4*>(out_ptr + out_base);
    *out_v4 = *in_v4;
}

} // namespace

void launchUlyssesPostUnscatter(void const* q_in, void const* k_in, void const* v_in, void* q_out, void* k_out,
    void* v_out, int P, int B, int Sp, int H, int D, cudaStream_t stream)
{
    constexpr int VEC = 8;
    TLLM_CHECK_WITH_INFO(D % VEC == 0, "ulyssesPostUnscatter: D must be a multiple of 8 (uint4 vec), got %d", D);
    int const vec_per_row = D / VEC;
    int const threads = H * vec_per_row;
    TLLM_CHECK_WITH_INFO(threads <= 1024,
        "ulyssesPostUnscatter: threads/block (H*D/8) must be <= 1024, got H=%d D=%d -> %d", H, D, threads);

    dim3 const grid(P * Sp, B, 3);
    dim3 const block(threads);

    ulyssesPostUnscatterKernel<__nv_bfloat16><<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16 const*>(q_in),
        reinterpret_cast<__nv_bfloat16 const*>(k_in), reinterpret_cast<__nv_bfloat16 const*>(v_in),
        reinterpret_cast<__nv_bfloat16*>(q_out), reinterpret_cast<__nv_bfloat16*>(k_out),
        reinterpret_cast<__nv_bfloat16*>(v_out), P, B, Sp, H, D, vec_per_row);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
