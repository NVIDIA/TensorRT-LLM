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
// Per-forward metadata of a TensorRT-LLM batch for this target's kernels, built on the device from the
// runtime's persistent buffers (so a captured decode step rebuilds it on every replay).
#include "glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/csrc/common.h"
#include "glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/csrc/kernels/step_meta.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace glm
{
namespace k
{

// One CTA of kStepMetaThreads threads: cu [B + 1] = exclusive scan of seq_lens (the batch rows'
// new tokens, contexts first), token_seq [T] = batch row of every token, has_state [B] (contexts:
// has_initial[b]; generation rows: 1), replay [B] (generation rows: prev_accepted[slots[b] *
// prev_stride], the drafts the slot's previous verify step accepted; contexts and no prev_accepted: 0).
constexpr int kStepMetaThreads = 1024;

__global__ void __launch_bounds__(kStepMetaThreads) step_meta_kernel(int* __restrict__ cu, int* __restrict__ token_seq,
    int* __restrict__ has_state, int* __restrict__ replay, int2* __restrict__ tiles, int const* __restrict__ seq_lens,
    bool const* __restrict__ has_initial, int const* __restrict__ slots, int const* __restrict__ prev_accepted,
    int64_t prev_stride, int B, int num_contexts, int T)
{
    __shared__ int scan[kStepMetaThreads];
    int const tid = threadIdx.x;
    int const len = tid < B ? seq_lens[tid] : 0;
    scan[tid] = len;
    __syncthreads();
    for (int o = 1; o < kStepMetaThreads; o <<= 1)
    {
        int const v = tid >= o ? scan[tid - o] : 0;
        __syncthreads();
        scan[tid] += v;
        __syncthreads();
    }
    if (tid < B)
    {
        cu[tid + 1] = scan[tid];
        bool const ctx = tid < num_contexts;
        has_state[tid] = ctx ? (has_initial ? (int) has_initial[tid] : 0) : 1;
        replay[tid] = (!ctx && prev_accepted) ? prev_accepted[(int64_t) slots[tid] * prev_stride] : 0;
        if (tiles)
            tiles[tid] = make_int2(scan[tid] - len, len);
    }
    if (tid == 0)
        cu[0] = 0;
    // token -> batch row: binary search of the inclusive scan
    for (int t = tid; t < T; t += kStepMetaThreads)
    {
        int lo = 0, hi = B - 1;
        while (lo < hi)
        {
            int const mid = (lo + hi) >> 1;
            if (scan[mid] > t)
                hi = mid;
            else
                lo = mid + 1;
        }
        token_seq[t] = lo;
    }
}

void step_meta(int* cu, int* token_seq, int* has_state, int* replay, int2* tiles, int const* seq_lens,
    bool const* has_initial, int const* slots, int const* prev_accepted, int64_t prev_stride, int B, int num_contexts,
    int T, cudaStream_t st)
{
    GLM_CHECK(B >= 1 && B <= kStepMetaThreads, "step_meta: 1 .. 1024 batch rows");
    // a plain launch (no programmatic dependency): the first kernel of a forward, so every
    // later kernel of the step, early-launched or not, runs after the runtime's own preparation
    step_meta_kernel<<<1, kStepMetaThreads, 0, st>>>(cu, token_seq, has_state, replay, tiles, seq_lens, has_initial,
        slots, prev_accepted, prev_stride, B, num_contexts, T);
    CUDA_CHECK(cudaGetLastError());
}

// (Plain launches, plain loads: these read what the runtime's copies or the step's last kernels
// wrote, so they neither wait programmatically nor read through the non-coherent path.)
__global__ void spec_sample_params_kernel(
    SampleParams* out, float const* temps, int64_t const* seeds, int64_t const* offsets, int S, float greedy_below)
{
    int const r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= S)
        return;
    float const t = temps[r];
    out[r].temperature = t < greedy_below ? 0.f : t;
    out[r].seed = (uint64_t) seeds[r];
    out[r].step = (uint64_t) offsets[r];
}

void spec_sample_params(SampleParams* out, float const* temps, int64_t const* seeds, int64_t const* offsets, int S,
    float greedy_below, cudaStream_t st)
{
    if (S <= 0)
        return;
    spec_sample_params_kernel<<<(S + 255) / 256, 256, 0, st>>>(out, temps, seeds, offsets, S, greedy_below);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void spec_accept_kernel(int* new_tokens, int* lens, int* next, int const* tokens, int const* drafts,
    int const* accepted, int const* row0, int const* batch_idx, int S, int const* slots, bool const* dummy,
    int* prev_accepted, int64_t prev_stride, int B, int num_contexts, int D)
{
    int const b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;
    int const n = D + 1;
    int const r = row0 ? row0[b] : b * n;
    for (int j = 0; j < n; ++j)
        new_tokens[b * n + j] = tokens[min(r + j, S - 1)];
    int const a = accepted[b];
    lens[b] = a + 1;
    int const src = batch_idx ? batch_idx[b] : b;
    int const rs = row0 ? row0[src] : src * n;
    next[b * n] = tokens[min(rs + a, S - 1)];
    for (int j = 0; j < D; ++j)
        next[b * n + 1 + j] = drafts[b * D + j];
    if (b >= num_contexts && prev_accepted && !(dummy && dummy[b]))
        prev_accepted[(int64_t) slots[b] * prev_stride] = max(a, 0);
}

void spec_accept(int* new_tokens, int* lens, int* next, int const* tokens, int const* drafts, int const* accepted,
    int const* row0, int const* batch_idx, int S, int const* slots, bool const* dummy, int* prev_accepted,
    int64_t prev_stride, int B, int num_contexts, int D, cudaStream_t st)
{
    GLM_CHECK(B >= 1 && S >= 1 && D >= 1, "spec_accept: an empty step");
    spec_accept_kernel<<<(B + 127) / 128, 128, 0, st>>>(new_tokens, lens, next, tokens, drafts, accepted, row0,
        batch_idx, S, slots, dummy, prev_accepted, prev_stride, B, num_contexts, D);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace k
} // namespace glm

TRTLLM_MV2_NAMESPACE_END
