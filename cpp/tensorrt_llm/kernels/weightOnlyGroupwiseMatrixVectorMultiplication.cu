#include "cutlass/cutlass.h"
#include "cutlass_extensions/interleaved_numeric_conversion.h"
#include "tensorrt_llm/kernels/weightOnlyGroupwiseMatrixVectorMultiplication.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>

namespace tensorrt_llm
{
namespace kernels
{
template <bool Zero, bool Bias, int N_PER_BLOCK, int BATCH, int BLOCK_SIZE, int GROUP_SIZE>
__global__ void groupwise_weight_only_matmul_i2f(const int32_t* qweight, const half* scales, const half* zeros,
    const half* in, const half* bias, half* out, const int n, const int k)
{
    static_assert(N_PER_BLOCK == 1 || (N_PER_BLOCK % 2 == 0));
    using Converter = cutlass::FastInterleavedAndBiasedNumericArrayConverter<cutlass::half_t, cutlass::uint4b_t, 8>;
    extern __shared__ uint8_t shmem[];
    constexpr int Interleave = 4;
    constexpr int NUM = BATCH * N_PER_BLOCK;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n_start_id = bid * N_PER_BLOCK * Interleave;
    const int interleave_n_id = (tid / 2) % Interleave;

    qweight += n_start_id * k / 8;
    scales += (n_start_id + interleave_n_id);
    if constexpr (Zero)
    {
        zeros += (n_start_id + interleave_n_id);
    }
    float(*sm)[NUM * Interleave] = reinterpret_cast<float(*)[NUM * Interleave]>(shmem);

    half reses[NUM];
    for (int i = 0; i < NUM; ++i)
    {
        reses[i] = __float2half_rn(0.f);
    }

    for (int local_k = tid * 32, real_k = tid / 8 * 64 + (tid % 2) * 32; local_k < k * Interleave;
         local_k += BLOCK_SIZE * 32, real_k += BLOCK_SIZE * 32 / Interleave)
    {
        half weights_f16[32 * N_PER_BLOCK];
        half scale[N_PER_BLOCK], zero[N_PER_BLOCK];
#pragma unroll
        for (int idx = 0; idx < N_PER_BLOCK; ++idx)
        {
            uint8_t weights_i4[16];
            *reinterpret_cast<int4*>(weights_i4)
                = *reinterpret_cast<const int4*>(qweight + idx * Interleave * k / 8 + local_k / 8);

            scale[idx] = scales[real_k / GROUP_SIZE * n + idx * Interleave];
            if constexpr (Zero)
            {
                zero[idx] = zeros[real_k / GROUP_SIZE * n + idx * Interleave];
            }
            else
            {
                zero[idx] = __float2half_rn(0.f);
            }
            half weights_vec[32];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                *reinterpret_cast<Converter::result_type*>(weights_vec + i * 8)
                    = Converter::convert(*reinterpret_cast<Converter::source_type*>(weights_i4 + i * 4));
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    half2 v = *reinterpret_cast<half2*>(weights_vec + i * 2 + j * 8);
                    v = __hfma2(v, __half2half2(scale[idx]), __half2half2(zero[idx]));
                    weights_f16[(i * 8 + j * 2 + 0) * N_PER_BLOCK + idx] = v.x;
                    weights_f16[(i * 8 + j * 2 + 1) * N_PER_BLOCK + idx] = v.y;
                }
            }
        }

#pragma unroll
        for (int b = 0; b < BATCH; ++b)
        {
            half in_v[32];
#pragma unroll
            for (int idx = 0; idx < 4; ++idx)
            {
                *reinterpret_cast<float4*>(in_v + idx * 8)
                    = *reinterpret_cast<const float4*>(in + b * k + real_k + idx * 8);
            }
            if constexpr (N_PER_BLOCK == 1)
            {
                half2 v = __float2half2_rn(0.f);
#pragma unroll
                for (int y = 0; y < 32; y += 2)
                {
                    v = __hfma2(*reinterpret_cast<half2*>(weights_f16 + y), *reinterpret_cast<half2*>(in_v + y), v);
                }
                reses[b] += __hadd(v.x, v.y);
            }
            else
            {
#pragma unroll
                for (int x = 0; x < N_PER_BLOCK / 2; ++x)
                {
#pragma unroll
                    for (int y = 0; y < 32; ++y)
                    {
                        *reinterpret_cast<half2*>(reses + b * N_PER_BLOCK + x * 2)
                            = __hfma2(*reinterpret_cast<half2*>(weights_f16 + y * N_PER_BLOCK + x * 2),
                                __half2half2(in_v[y]), *reinterpret_cast<half2*>(reses + b * N_PER_BLOCK + x * 2));
                    }
                }
            }
        }
    }
    float reses2[NUM];
#pragma unroll
    for (int i = 0; i < NUM; ++i)
    {
        reses2[i] = __half2float(reses[i]);
    }
#pragma unroll
    for (int i = 0; i < NUM; ++i)
    {
        reses2[i] += __shfl_xor_sync(~0, reses2[i], 16);
        reses2[i] += __shfl_xor_sync(~0, reses2[i], 8);
        reses2[i] += __shfl_xor_sync(~0, reses2[i], 1);
    }
    __syncthreads();
    int warp = tid / 32, lane = tid % 32;
    if (lane == 0 || lane == 2 || lane == 4 || lane == 6)
    {
#pragma unroll
        for (int i = 0; i < NUM; ++i)
        {
            sm[warp][i * Interleave + lane / 2] = reses2[i];
        }
    }
    __syncthreads();
    for (int i = tid; i < NUM * Interleave; i += BLOCK_SIZE)
    {
        int nid = i % (N_PER_BLOCK * Interleave);
        float v = 0.f;
        for (int j = 0; j < BLOCK_SIZE / 32; ++j)
        {
            v += sm[j][i];
        }
        float bias_v;
        if constexpr (Bias)
        {
            bias_v = __half2float(bias[n_start_id + nid]);
        }
        else
        {
            bias_v = 0.f;
        }
        int b = i / N_PER_BLOCK / Interleave;
        out[b * n + n_start_id + nid] = __float2half_rn(v + bias_v);
    }
}

#define RUN_groupwise_weight_only_matmul_i2f_2(Zero, Bias, N_PER_BLOCK, BATCH, BLOCKSIZE)                              \
    {                                                                                                                  \
        dim3 grid(n / N_PER_BLOCK / 4);                                                                                \
        dim3 block(BLOCKSIZE);                                                                                         \
        int size = sizeof(float) * BLOCKSIZE / 32 * BATCH * N_PER_BLOCK * 4;                                           \
        if (group_size == 64)                                                                                          \
        {                                                                                                              \
            groupwise_weight_only_matmul_i2f<Zero, Bias, N_PER_BLOCK, BATCH, BLOCKSIZE, 64>                            \
                <<<grid, block, size, *stream>>>(qweight, qscales, qzeros, in, bias, out, n, k);                       \
        }                                                                                                              \
        else if (group_size == 128)                                                                                    \
        {                                                                                                              \
            groupwise_weight_only_matmul_i2f<Zero, Bias, N_PER_BLOCK, BATCH, BLOCKSIZE, 128>                           \
                <<<grid, block, size, *stream>>>(qweight, qscales, qzeros, in, bias, out, n, k);                       \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            printf("Invalid group size. Only group size 64 and 128 supported for fine grained kernels.");              \
            std::abort();                                                                                              \
        }                                                                                                              \
        break;                                                                                                         \
    }

#define RUN_groupwise_weight_only_matmul_i2f_1(N_PER_BLOCK, BATCH, BLOCKSIZE)                                          \
    {                                                                                                                  \
        if (qzeros && bias)                                                                                            \
        {                                                                                                              \
            RUN_groupwise_weight_only_matmul_i2f_2(true, true, N_PER_BLOCK, BATCH, BLOCKSIZE);                         \
        }                                                                                                              \
        else if (qzeros && !bias)                                                                                      \
        {                                                                                                              \
            RUN_groupwise_weight_only_matmul_i2f_2(true, false, N_PER_BLOCK, BATCH, BLOCKSIZE);                        \
        }                                                                                                              \
        else if (!qzeros && bias)                                                                                      \
        {                                                                                                              \
            RUN_groupwise_weight_only_matmul_i2f_2(false, true, N_PER_BLOCK, BATCH, BLOCKSIZE);                        \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            RUN_groupwise_weight_only_matmul_i2f_2(false, false, N_PER_BLOCK, BATCH, BLOCKSIZE);                       \
        }                                                                                                              \
    }

void groupwise_weight_only_matmul_i2f_launcher(const int32_t* qweight, const half* qscales, const half* qzeros,
    const half* in, const half* bias, half* out, const int batch, const int n, const int k, const int group_size,
    cudaStream_t* stream)
{
    switch (batch)
    {
    case 1: RUN_groupwise_weight_only_matmul_i2f_1(2, 1, 256);
    case 2: RUN_groupwise_weight_only_matmul_i2f_1(2, 2, 256);
    case 3: RUN_groupwise_weight_only_matmul_i2f_1(2, 3, 128);
    case 4: RUN_groupwise_weight_only_matmul_i2f_1(2, 4, 128);
    case 5: RUN_groupwise_weight_only_matmul_i2f_1(2, 5, 128);
    case 6: RUN_groupwise_weight_only_matmul_i2f_1(2, 6, 256);
    case 7: RUN_groupwise_weight_only_matmul_i2f_1(2, 7, 128);
    case 8: RUN_groupwise_weight_only_matmul_i2f_1(2, 8, 128);
    case 9: RUN_groupwise_weight_only_matmul_i2f_1(2, 9, 128);
    case 10: RUN_groupwise_weight_only_matmul_i2f_1(4, 10, 128);
    case 11: RUN_groupwise_weight_only_matmul_i2f_1(4, 11, 128);
    case 12: RUN_groupwise_weight_only_matmul_i2f_1(2, 12, 128);
    case 13: RUN_groupwise_weight_only_matmul_i2f_1(4, 13, 128);
    case 14: RUN_groupwise_weight_only_matmul_i2f_1(4, 14, 128);
    case 15: RUN_groupwise_weight_only_matmul_i2f_1(4, 15, 128);
    case 16: RUN_groupwise_weight_only_matmul_i2f_1(4, 16, 128);
    default: printf("vecquant4matmul_nk_kernel_launcher invalid batch!! batch=%d ", batch); std::abort();
    }
}

} // namespace kernels
} // namespace tensorrt_llm
