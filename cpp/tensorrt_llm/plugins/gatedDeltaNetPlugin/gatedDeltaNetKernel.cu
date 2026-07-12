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
 *
 * CUDA kernel for the "Gated DeltaNet" (GDN) linear-attention recurrence used by
 * the Qwen3.5/3.6 engine path. This is a naive, correctness-first port of the
 * recurrence (one CUDA thread/block walks a sequence token-by-token, fp32
 * state). Optimization (chunking, tensor cores, vectorized loads,
 * register-tiling of the state S, etc.) is explicitly OUT OF SCOPE here: the
 * goal is numerical parity with the PyTorch reference.
 *
 * Math per (batch b, v-head h), looping tokens t = 0 .. seqlens[b]-1, starting
 * from S = S_in[b, h]  (state S is [D_k, D_v]):
 *
 *   qn = l2norm(q_t) ; kn = l2norm(k_t)            # over D_k, eps = 1e-6
 *   qn *= 1 / sqrt(D_k)                            # SCALE on q only
 *   S  *= exp(g_t)                                 # elementwise decay on [D_k,D_v]
 *   kvmem[j] = sum_i S[i,j] * kn[i]                # read with (normalized) key
 *   delta[j] = (v_t[j] - kvmem[j]) * beta_t        # correction
 *   S[i,j]  += kn[i] * delta[j]                    # rank-1 outer-product update
 *   y_t[j]   = sum_i S[i,j] * qn[i]                # read with (scaled) query, on UPDATED S
 *
 * The order (decay -> read-kv -> delta -> write -> read-y on updated S) is
 * load-bearing.
 */

#include "tensorrt_llm/plugins/gatedDeltaNetPlugin/gatedDeltaNetPlugin.h"

#include <cuda_runtime.h>

namespace tensorrt_llm::plugins
{

namespace
{

// Upper bound on D_k / D_v for static per-thread local storage. The real model
// uses D_k = D_v = 128; we size the per-thread S-column to this. The kernel still
// reads the actual D_k / D_v from the launch params and only touches the valid range.
constexpr int kMaxDim = 128;

constexpr float kL2NormEps = 1e-6f;

// One CUDA block per (batch, v-head) pair. Within a block we launch D_v threads;
// thread j owns column j of the recurrent state, i.e. the length-D_k vector
// S[:, j]. The recurrence is sequential over tokens, so the block loops t inside
// while distinct (b, h) blocks run in parallel.
//
// Shared memory per block holds, for the current token:
//   - kn[D_k] : L2-normalized key
//   - qn[D_k] : L2-normalized + scaled query
// plus a small scratch slot for the block-wide reduction used to compute the
// L2 norms. The two scalars (exp(g_t), beta_t) are recomputed by every thread --
// cheap and avoids extra sync points.
__global__ void gatedDeltaRuleKernel(float const* __restrict__ q, // [B,T,H,Dk]
    float const* __restrict__ k,                                  // [B,T,H,Dk]
    float const* __restrict__ v,                                  // [B,T,H,Dv]
    float const* __restrict__ g,                                  // [B,T,H]
    float const* __restrict__ beta,                               // [B,T,H]
    float const* __restrict__ S_in,                               // [B,H,Dk,Dv]
    int const* __restrict__ seqlens,                              // [B]
    float* __restrict__ y,                                        // [B,T,H,Dv]
    float* __restrict__ S_out,                                    // [B,H,Dk,Dv]
    int B, int T, int H, int Dk, int Dv, bool useQkL2norm)
{
    int const bh = blockIdx.x; // flattened (b, h) index in [0, B*H)
    int const b = bh / H;      // batch index
    int const h = bh % H;      // v-head index
    int const j = threadIdx.x; // this thread's state column in [0, Dv)

    if (b >= B || h >= H)
    {
        return;
    }

    // Shared layout: [ kn(Dk) | qn(Dk) | scratch(blockDim.x) ].
    extern __shared__ float smem[];
    float* sh_kn = smem;            // [Dk]
    float* sh_qn = sh_kn + Dk;      // [Dk]
    float* sh_scratch = sh_qn + Dk; // [blockDim.x] reduction scratch

    bool const active = (j < Dv);

    // Per-thread state column S[:, j] for the column this thread owns, in local
    // (per-thread) memory. Only the first Dk entries are used.
    float Scol[kMaxDim];

    // Base offset of this (b, h) state block inside S_in / S_out: element [i, j]
    // lives at base + i * Dv + j.
    long const state_base = ((long) bh) * Dk * Dv;

    // Initialize the owned state column from S_in (S[i, j] for all i).
    if (active)
    {
        for (int i = 0; i < Dk; ++i)
        {
            Scol[i] = S_in[state_base + (long) i * Dv + j];
        }
    }

    int const L = seqlens[b];

    for (int t = 0; t < L; ++t)
    {
        // Per-token base offsets into the [B,T,H,*] tensors.
        long const qk_base = ((((long) b * T + t) * H) + h) * Dk; // q,k row start
        long const v_base = ((((long) b * T + t) * H) + h) * Dv;  // v / y row start
        long const gb_base = (((long) b * T + t) * H) + h;        // g,beta scalar

        // ----------------------------------------------------------------
        // Step 1 (part a): L2-norm denominators for q and k over D_k.
        // We compute sum(q^2) and sum(k^2) with a simple block reduction so the
        // (eps-stabilized) rsqrt matches the reference exactly. Thread j handles
        // index j of the length-D_k vectors during the reduction (D_k == blockDim
        // here since blockDim == D_v == D_k == 128 for this model, but we guard so
        // it works whenever blockDim >= Dk by zero-padding the tail).
        // ----------------------------------------------------------------
        float const scale = rsqrtf((float) Dk); // SCALE = 1/sqrt(D_k), applied to q only

        // -- sum of squares for q --
        {
            float partial = 0.0f;
            if (j < Dk)
            {
                float const qj = q[qk_base + j];
                partial = qj * qj;
            }
            sh_scratch[threadIdx.x] = partial;
            __syncthreads();
            // Tree reduction over the whole block.
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
            {
                if (threadIdx.x < stride)
                {
                    sh_scratch[threadIdx.x] += sh_scratch[threadIdx.x + stride];
                }
                __syncthreads();
            }
            float const q_rnorm = useQkL2norm ? rsqrtf(sh_scratch[0] + kL2NormEps) : 1.0f;
            // Step 1+2: normalize q (if enabled) then apply SCALE = 1/sqrt(D_k).
            if (j < Dk)
            {
                sh_qn[j] = q[qk_base + j] * q_rnorm * scale;
            }
            __syncthreads();
        }
        // -- sum of squares for k --
        {
            float partial = 0.0f;
            if (j < Dk)
            {
                float const kj = k[qk_base + j];
                partial = kj * kj;
            }
            sh_scratch[threadIdx.x] = partial;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
            {
                if (threadIdx.x < stride)
                {
                    sh_scratch[threadIdx.x] += sh_scratch[threadIdx.x + stride];
                }
                __syncthreads();
            }
            float const k_rnorm = useQkL2norm ? rsqrtf(sh_scratch[0] + kL2NormEps) : 1.0f;
            if (j < Dk)
            {
                sh_kn[j] = k[qk_base + j] * k_rnorm; // k is NOT scaled
            }
            __syncthreads();
        }

        // Per-token scalars (recomputed by every active thread).
        float const decay = expf(g[gb_base]); // exp(g_t)
        float const bt = beta[gb_base];       // beta_t

        if (active)
        {
            // Step 3: decay the owned state column elementwise.  S[i,j] *= decay.
            for (int i = 0; i < Dk; ++i)
            {
                Scol[i] *= decay;
            }

            // Step 4: read with normalized key.  kvmem[j] = sum_i S[i,j] * kn[i].
            float kvmem = 0.0f;
            for (int i = 0; i < Dk; ++i)
            {
                kvmem += Scol[i] * sh_kn[i];
            }

            // Step 5: delta[j] = (v_t[j] - kvmem[j]) * beta_t.
            float const delta = (v[v_base + j] - kvmem) * bt;

            // Step 6: rank-1 update.  S[i,j] += kn[i] * delta.
            for (int i = 0; i < Dk; ++i)
            {
                Scol[i] += sh_kn[i] * delta;
            }

            // Step 7: read with scaled query on the UPDATED state.
            //         y_t[j] = sum_i S[i,j] * qn[i].
            float yj = 0.0f;
            for (int i = 0; i < Dk; ++i)
            {
                yj += Scol[i] * sh_qn[i];
            }
            y[v_base + j] = yj;
        }

        // Ensure all threads finished reading sh_kn / sh_qn for this token before
        // the next iteration overwrites them.
        __syncthreads();
    }

    // Write the final state column back to S_out.
    if (active)
    {
        for (int i = 0; i < Dk; ++i)
        {
            S_out[state_base + (long) i * Dv + j] = Scol[i];
        }
    }
}

} // anonymous namespace

void invokeGatedDeltaNet(GatedDeltaNetParams const& params, cudaStream_t stream)
{
    int const B = params.batch;
    int const T = params.maxSeqLen;
    int const H = params.numVHeads;
    int const Dk = params.headKDim;
    int const Dv = params.headVDim;

    // One block per (b, h). Threads = D_v columns; we also need >= D_k lanes for
    // the L2-norm reduction. For this model D_k == D_v == 128 so D_v threads
    // suffice. The host (plugin enqueue) validates Dk/Dv <= kMaxDim and Dv >= Dk.
    int const threads = Dv;
    dim3 const grid((unsigned) (B * H));
    dim3 const block((unsigned) threads);

    // Shared: kn(Dk) + qn(Dk) + scratch(threads).
    size_t const smemBytes = (size_t) (2 * Dk + threads) * sizeof(float);

    gatedDeltaRuleKernel<<<grid, block, smemBytes, stream>>>(static_cast<float const*>(params.q),
        static_cast<float const*>(params.k), static_cast<float const*>(params.v), static_cast<float const*>(params.g),
        static_cast<float const*>(params.beta), static_cast<float const*>(params.statePtrIn), params.seqLens,
        static_cast<float*>(params.y), static_cast<float*>(params.statePtrOut), B, T, H, Dk, Dv, params.useQkL2norm);
}

} // namespace tensorrt_llm::plugins
