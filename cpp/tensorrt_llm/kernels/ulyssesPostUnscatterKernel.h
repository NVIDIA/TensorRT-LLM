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

#ifndef TRTLLM_ULYSSESPOSTUNSCATTERKERNEL_H
#define TRTLLM_ULYSSESPOSTUNSCATTERKERNEL_H

#include "tensorrt_llm/common/config.h"
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Post-Ulysses A2A unscatter for Q/K/V. Pairs with ulyssesPermuteScatter:
// PermuteScatter prepares send_buf pre-A2A; PostUnscatter consumes recv_buf
// post-A2A and produces SDPA-ready tensors in NHD layout.
//
// After the head-dim → seq-dim all-to-all, each rank holds tensors of shape
// [P, B, Sp, H, D] where P = sequence-parallel world size, Sp = local seq
// len, H = heads-per-rank, D = head dim. This kernel always writes NHD-contig
// [B, P*Sp, H, D] storage. The op wrapper returns the storage as-is for NHD
// callers (TRTLLM / FA4) or as a transpose-view for HND callers (VANILLA /
// torch SDPA) — the HND-shape return is thus HND-shape with NHD-stride,
// mirroring what the sync `_forward_unfused` path produces via
// `q.transpose(1, 2)` (without `.contiguous()`). This stride pattern lets
// cudnn SDPA preserve NHD-stride through its output, so the downstream
// `_output_a2a`'s `.transpose(1, 2).contiguous()` collapses to a no-op.
//
// Equivalent eager expression this kernel replaces:
//   t.permute(1, 0, 2, 3, 4).reshape(B, P*Sp, H, D).contiguous()
//
// Layout:
//   - Each block reads one fully contiguous (p, b, sp, :H, :D) tile of
//     H*D bf16
//   - H*(D/8) threads/block — each thread copies one uint4 (8 bf16)
//   - Grid (P*Sp, B, 3): blockIdx.z selects Q / K / V
//
// Constraints:
//   - dtype must be bf16
//   - D must be a multiple of 8 (uint4 vector load/store, 8 bf16 per thread)
//   - threads/block = H * (D / 8) must be <= 1024 (CUDA hw limit)
void launchUlyssesPostUnscatter(void const* q_in, // [P, B, Sp, H, D]
    void const* k_in, void const* v_in,
    void* q_out,                                  // [B, P*Sp, H, D] NHD-contig
    void* k_out, void* v_out, int P, int B, int Sp, int H, int D, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_ULYSSESPOSTUNSCATTERKERNEL_H
