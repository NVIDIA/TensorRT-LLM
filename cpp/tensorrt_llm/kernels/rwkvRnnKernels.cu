/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/kernels/rwkvRnnKernels.h"
#include <stdexcept>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename F, int N>
__global__ void kernel_forward(const int B, const int T, const int C, const int H, float *__restrict__ _state,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const F *__restrict__ _u,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w += h*N;
    _u += h*N;
    _state += h*N*N + i*N; // wrong if B > 1 !!!

    __shared__ float r[N], k[N], u[N], w[N];
    
    float state[N];
    #pragma unroll
    for (int j = 0; j < N; j++)
        state[j] = _state[j];
    
    __syncthreads();
    u[i] = float(_u[i]);
    w[i] = _w[i];
    __syncthreads();

    for (int t = b*T*C + h*N + i; t < (b+1)*T*C + h*N + i; t += C)
    {
        __syncthreads();
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float v = float(_v[t]);
        float y = 0;

        #pragma unroll
        for (int j = 0; j < N; j+=4)
        {
            const float4& r_ = (float4&)(r[j]);
            const float4& k_ = (float4&)(k[j]);
            const float4& w_ = (float4&)(w[j]);
            const float4& u_ = (float4&)(u[j]);
            float4& s = (float4&)(state[j]);
            float4 x;

            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;

            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
        }
        _y[t] = F(y);
    }
    #pragma unroll
    for (int j = 0; j < N; j++)
        _state[j] = state[j];
}

template <typename DType>
void invokeGeneralRwkvRnn(float* state, const DType* r, const DType* k, const DType* v, const float* w, const DType* u, DType* y,
    const int B, const int T, const int C, const int H, cudaStream_t stream)
{
    if (C % H != 0)
    {
        throw std::runtime_error(
            "C connot be divised by H, please check your arguments.");
    }
    int shmem_size = C / H * 4;
    int block_size = C / H;
    if (block_size == 64)
    {
        kernel_forward<DType, 64><<<dim3(B * H), dim3(block_size), shmem_size, stream>>>(B, T, C, H, state, r, k, v, w, u, y);
    }
    else
    {
        throw std::runtime_error(
            "Encountered an unexpected value of C / H. It may due to the updating of the rwkv models, "
            "please submit an issue for it");
    }
}

#define INSTANTIATE_GENERAL_RWKV_RNN(DType)                                                                            \
    template void invokeGeneralRwkvRnn(float* state, const DType* r, const DType* k, const DType* v, const float* w,   \
    const DType* u, DType* y, const int B, const int T, const int C, const int H, cudaStream_t stream);

INSTANTIATE_GENERAL_RWKV_RNN(float);
INSTANTIATE_GENERAL_RWKV_RNN(half);

#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_RWKV_RNN(__nv_bfloat16);
#endif

} // namespace kernels
} // namespace tensorrt_llm