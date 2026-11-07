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

// Based on the fused anti-aliased activation of NVIDIA BigVGAN (https://github.com/NVIDIA/BigVGAN,
// alias_free_activation/cuda, MIT License).
//
// Fused anti-aliased SnakeBeta activation of the MiniMax-H3 BigVGAN audio decoder, with the replicate padding of
// Diffusers' MiniMaxH3AudioActivation1d at both sequence ends. For x [rows, L] (row = batch * C + channel), fp32
// throughout:
//   u[n]   = 2 * sum_{k = n + 1 mod 2} fu[k] * x[clamp((n + 15 - k) / 2 - 5, 0, L - 1)]   n in [0, 2L)
//   y[n]   = u[n] + 1 / (exp(beta[c]) + 1e-9) * sin(exp(alpha[c]) * u[n])^2
//   out[i] = sum_k fd[k] * y[clamp(2i + k - 5, 0, 2L - 1)]                                  i in [0, L)
// which is replicate pad 5 + conv_transpose1d(stride 2) * 2 cropped by 15 on each side, SnakeBeta, and replicate pad
// (5, 6) + conv1d(stride 2), each with the 12-tap Kaiser-sinc filters fu and fd.

#include "antiAliasActivation.h"

namespace tensorrt_llm::visual_gen::minimax_h3
{

namespace
{

constexpr int kBlockOut = 1024;
constexpr int kThreads = 128;
// Outputs [i0, i0 + kBlockOut) read y[2 i0 - 5, 2 i0 + 2 kBlockOut + 5), computed as kPairs pairs (y[2q], y[2q + 1])
// for q in [i0 - 3, i0 + kBlockOut + 3); a pair reads x[q - 3, q + 3].
constexpr int kPairs = kBlockOut + 6;
constexpr int kXHalo = 8;
constexpr int kXWindow = kBlockOut + 2 * kXHalo;

__global__ void __launch_bounds__(kThreads) antiAliasSnakeBetaKernel(float* __restrict__ out,
    float const* __restrict__ x, float const* __restrict__ upFilter, float const* __restrict__ downFilter,
    float const* __restrict__ logAlpha, float const* __restrict__ logBeta, int channels, int length)
{
    __shared__ float xs[kXWindow];
    __shared__ float ys[2 * kPairs];
    __shared__ float fs[2 * kAntiAliasTaps];

    int const row = blockIdx.y;
    int const i0 = blockIdx.x * kBlockOut;
    float const* xr = x + static_cast<size_t>(row) * length;
    if (threadIdx.x < 2 * kAntiAliasTaps)
    {
        fs[threadIdx.x]
            = threadIdx.x < kAntiAliasTaps ? upFilter[threadIdx.x] : downFilter[threadIdx.x - kAntiAliasTaps];
    }
    int const xBase = i0 - kXHalo;
    for (int t = threadIdx.x; t < kXWindow; t += kThreads)
    {
        xs[t] = xr[min(max(xBase + t, 0), length - 1)];
    }
    __syncthreads();

    float fu[kAntiAliasTaps];
    float fd[kAntiAliasTaps];
#pragma unroll
    for (int k = 0; k < kAntiAliasTaps; ++k)
    {
        fu[k] = fs[k];
        fd[k] = fs[kAntiAliasTaps + k];
    }
    int const channel = row % channels;
    float const alpha = expf(logAlpha[channel]);
    float const invBeta = 1.0f / (expf(logBeta[channel]) + 1e-9f);
    for (int p = threadIdx.x; p < kPairs; p += kThreads)
    {
        // y[2q] takes the odd taps on x[q + 2 - j], y[2q + 1] the even taps on x[q + 3 - j] (j = 0..5); positions
        // outside [0, 2L) take y[0] or y[2L - 1].
        int const q = i0 - 3 + p;
        int const qc = min(max(q, 0), length - 1);
        float const* xw = xs + (qc - xBase);
        float even = 0.0f;
        float odd = 0.0f;
#pragma unroll
        for (int j = 0; j < kAntiAliasTaps / 2; ++j)
        {
            even = fmaf(fu[2 * j + 1], xw[2 - j], even);
            odd = fmaf(fu[2 * j], xw[3 - j], odd);
        }
        float const upEven = 2.0f * even;
        float const upOdd = 2.0f * odd;
        float const se = sinf(alpha * upEven);
        float const so = sinf(alpha * upOdd);
        float ye = upEven + invBeta * (se * se);
        float yo = upOdd + invBeta * (so * so);
        if (q < 0)
        {
            yo = ye;
        }
        else if (q > length - 1)
        {
            ye = yo;
        }
        ys[2 * p] = ye;
        ys[2 * p + 1] = yo;
    }
    __syncthreads();

    // out[i0 + t] = sum_k fd[k] * ys[2t + k + 1]; two adjacent outputs per thread share 10 of their 12 inputs.
    float* outRow = out + static_cast<size_t>(row) * length;
    for (int t = 2 * threadIdx.x; t < kBlockOut; t += 2 * kThreads)
    {
        int const i = i0 + t;
        if (i >= length)
        {
            break;
        }
        float r[kAntiAliasTaps + 2];
#pragma unroll
        for (int k = 0; k < kAntiAliasTaps + 2; ++k)
        {
            r[k] = ys[2 * t + 1 + k];
        }
        float a0 = 0.0f;
        float a1 = 0.0f;
#pragma unroll
        for (int k = 0; k < kAntiAliasTaps; ++k)
        {
            a0 = fmaf(fd[k], r[k], a0);
            a1 = fmaf(fd[k], r[k + 2], a1);
        }
        outRow[i] = a0;
        if (i + 1 < length)
        {
            outRow[i + 1] = a1;
        }
    }
}

} // namespace

cudaError_t invokeAntiAliasSnakeBeta(float* out, float const* x, float const* upFilter, float const* downFilter,
    float const* logAlpha, float const* logBeta, int rows, int channels, int length, cudaStream_t stream)
{
    dim3 const grid(static_cast<unsigned>((length + kBlockOut - 1) / kBlockOut), static_cast<unsigned>(rows));
    antiAliasSnakeBetaKernel<<<grid, kThreads, 0, stream>>>(
        out, x, upFilter, downFilter, logAlpha, logBeta, channels, length);
    return cudaGetLastError();
}

} // namespace tensorrt_llm::visual_gen::minimax_h3
