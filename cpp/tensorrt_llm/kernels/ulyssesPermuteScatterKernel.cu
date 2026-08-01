/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/ulyssesPermuteScatterKernel.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// Vector type: int4 = 16 bytes = 8 bf16. Issues LDG.E.128 / STG.E.128.
constexpr int VEC = 8;
constexpr int BLOCK_S = 32;            // rows per CTA
constexpr int THREADS_PER_BLOCK = 128; // 4 warps

// 1 CTA handles BLOCK_S rows × 1 head × full D.
//   For fixed h (per CTA), peer = h // H_local is constant — no warp divergence.
//   All writes within one CTA go to the same destination slot, contiguous in
//   dst space. Matches the access pattern of PyTorch inductor's permute kernel
//   epilogue stores.
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    ulyssesPermuteScatterKernel(__nv_bfloat16 const* __restrict__ input, // [B, S_local, H, D]
        __nv_bfloat16* __restrict__ send_buf,                            // [P, B, S_local, H/P, D]
        __nv_bfloat16* __restrict__ recv_buf,                            // [P, B, S_local, H/P, D]
        int const my_rank,
        int const n_rows,                                                // B * S_local
        int const H, int const D,
        int const H_local)                                               // H / P
{
    int const bs_block = blockIdx.x;
    int const h = blockIdx.y;

    // Scalar branch — same destination slot for all threads in this CTA.
    int const peer = h / H_local;
    int const h_local = h - peer * H_local;
    int const slot_idx = (peer == my_rank) ? my_rank : peer;
    __nv_bfloat16* __restrict__ dst_base = (peer == my_rank) ? recv_buf : send_buf;

    int const n_d_chunks = D / VEC;
    int const total_tasks = BLOCK_S * n_d_chunks;
    int const t = threadIdx.x;

    int const row_base = bs_block * BLOCK_S;

    int4 const* __restrict__ in_v = reinterpret_cast<int4 const*>(input);
    int4* __restrict__ dst_v = reinterpret_cast<int4*>(dst_base);

    int const row_in_stride_v = (H * D) / VEC; // = H * n_d_chunks
    int const head_in_off_v = h * n_d_chunks;
    int const slot_off_v = slot_idx * n_rows * H_local * n_d_chunks;
    int const row_dst_stride_v = H_local * n_d_chunks;
    int const head_dst_off_v = h_local * n_d_chunks;

#pragma unroll 1
    for (int idx = t; idx < total_tasks; idx += blockDim.x)
    {
        int const s_in_block = idx / n_d_chunks;
        int const d_chunk = idx - s_in_block * n_d_chunks;
        int const row = row_base + s_in_block;
        if (row >= n_rows)
            continue;

        int const src_idx = row * row_in_stride_v + head_in_off_v + d_chunk;
        int const dst_idx = slot_off_v + row * row_dst_stride_v + head_dst_off_v + d_chunk;

        dst_v[dst_idx] = in_v[src_idx];
    }
}

} // anonymous namespace

void launchUlyssesPermuteScatter(void const* input, void* send_buf, void* recv_buf, int my_rank, int B, int S_local,
    int H, int D, int P, cudaStream_t stream)
{
    int const n_rows = B * S_local;
    int const H_local = H / P;
    dim3 const grid((n_rows + BLOCK_S - 1) / BLOCK_S, H);
    dim3 const block(THREADS_PER_BLOCK);

    ulyssesPermuteScatterKernel<<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16 const*>(input),
        reinterpret_cast<__nv_bfloat16*>(send_buf), reinterpret_cast<__nv_bfloat16*>(recv_buf), my_rank, n_rows, H, D,
        H_local);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
