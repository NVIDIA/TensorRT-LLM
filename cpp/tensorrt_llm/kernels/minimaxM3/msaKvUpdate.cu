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

#include "tensorrt_llm/kernels/minimaxM3/msaKvUpdate.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{
__global__ void __launch_bounds__(256)
    msa_kv_update_kernel(int const* __restrict__ kv_lens, int const* __restrict__ staged, int const* __restrict__ qo,
        int const* __restrict__ q_batch_row, int const* __restrict__ q_intra, int const* __restrict__ table,
        long long table_ld, int ncols, int page, int batch, int total_q, int* __restrict__ seq_lens_out,
        int* __restrict__ cu_kv, int* __restrict__ out_loc, int* __restrict__ n_valid)
{
    for (int b = threadIdx.x; b < batch; b += blockDim.x)
        seq_lens_out[b] = min(kv_lens[b], staged[b]);
    if (cu_kv != nullptr && threadIdx.x == 0)
    {
        int acc = 0;
        cu_kv[0] = 0;
        for (int b = 0; b < batch; ++b)
        {
            acc += min(kv_lens[b], staged[b]);
            cu_kv[b + 1] = acc;
        }
    }
    for (int t = threadIdx.x; t < total_q; t += blockDim.x)
    {
        int const r = q_batch_row[t];
        int const pos = min(kv_lens[r], staged[r]) - qo[r] + q_intra[t];
        int const ps = max(pos, 0);
        int const col = min(ps / page, ncols - 1);
        out_loc[t] = table[r * table_ld + col] * page + ps % page;
        if (n_valid != nullptr)
            n_valid[t] = (max(pos + 1, 0) + page - 1) / page;
    }
}
} // namespace

cudaError_t invokeMinimaxM3MsaKvUpdate(int32_t const* kvLens, int32_t const* staged, int32_t const* qo,
    int32_t const* qBatchRow, int32_t const* qIntra, int32_t const* table, int64_t tableLd, int32_t ncols, int32_t page,
    int32_t batch, int32_t totalQ, int32_t* seqLensOut, int32_t* cuKv, int32_t* outLoc, int32_t* nValid,
    cudaStream_t stream)
{
    msa_kv_update_kernel<<<1, 256, 0, stream>>>(kvLens, staged, qo, qBatchRow, qIntra, table,
        static_cast<long long>(tableLd), ncols, page, batch, totalQ, seqLensOut, cuKv, outLoc, nValid);
    return cudaGetLastError();
}

} // namespace kernels

TRTLLM_NAMESPACE_END
