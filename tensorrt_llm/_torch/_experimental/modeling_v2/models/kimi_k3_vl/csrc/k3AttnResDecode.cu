/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#include "kimi_k3_vl/csrc/k3AttnResDecode.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cfloat>
#include <cooperative_groups.h>
#include <cute/tensor.hpp>
#include <mutex>

namespace
{

using bf16_t = __nv_bfloat16;

__inline__ __device__ float warp_reduce_sum(float val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ bf16_t apply_output_rms_norm(bf16_t value, float rsigma, bf16_t weight)
{

    bf16_t const normalized = __float2bfloat16_rn(__bfloat162float(value) * rsigma);
    return __float2bfloat16_rn(__bfloat162float(normalized) * __bfloat162float(weight));
}

namespace sm100::fwd_prod_v2
{

CUTE_DEVICE
float2 float2_fma(float2 const& a, float2 const& b, float2 const& c)
{
    float2 d;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(d))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
    return d;
}

CUTE_DEVICE
float2 float2_add(float2 const& a, float2 const& b)
{
    float2 c;
    asm volatile("add.rn.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

template <int N, bool FUSE_OUTPUT_RMS_NORM = false, bool FUSE_LAYER_ADD = false, bool ENABLE_PDL = false>
__global__ void __launch_bounds__(256, 1) attn_res_fwd_s1_single_cta_kernel(bf16_t const* __restrict__ block_res,
    bf16_t const* __restrict__ layer_res, bf16_t const* __restrict__ layer_res_add, bf16_t const* __restrict__ res_w,
    bf16_t const* __restrict__ rms_w, bf16_t const* __restrict__ output_rms_w, bf16_t* __restrict__ updated_layer_res,
    bf16_t* __restrict__ output, float* __restrict__ rsigma_out, float* __restrict__ probs_out,
    float* __restrict__ logits_out, float rms_eps, float output_rms_eps, int num_tokens, bool early_trigger)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if constexpr (ENABLE_PDL)
    {
        cudaGridDependencySynchronize();

        if (early_trigger)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

    constexpr int H = 7168;
    constexpr int THREADS = 256;
    constexpr int WARPS = THREADS / 32;
    constexpr int ITEMS = H / THREADS;
    constexpr float LOG2_E = 1.4426950408889634f;
    static_assert(H % THREADS == 0);
    static_assert(N >= 1 && N <= 4);

    int const token = blockIdx.x;
    block_res += (size_t) token * H;
    layer_res += (size_t) token * H;
    output += (size_t) token * H;
    if constexpr (FUSE_LAYER_ADD)
    {
        layer_res_add += (size_t) token * H;
        updated_layer_res += (size_t) token * H;
    }
    if (rsigma_out != nullptr)
    {
        rsigma_out += token;
    }
    if (probs_out != nullptr)
    {
        probs_out += token;
    }
    if (logits_out != nullptr)
    {
        logits_out += token;
    }

    __shared__ float2 warp_stats[WARPS * N];
    __shared__ float weights[N];
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;

    float2 stats[N] = {};
    float output_sq_local = 0.f;
    uint32_t v_cache_bf16[ITEMS][(N + 1) / 2];
    bf16_t mixed_cache[ITEMS];
#pragma unroll
    for (int item = 0; item < ITEMS; item++)
    {
        int h = tid + item * THREADS;
        float q = __bfloat162float(res_w[h]) * __bfloat162float(rms_w[h]);
        bf16_t item_v[N];
#pragma unroll
        for (int n = 0; n < N; n++)
        {

            bf16_t const* row = n < N - 1 ? block_res + (size_t) n * num_tokens * H : layer_res;
            bf16_t packed_v = row[h];
            if constexpr (FUSE_LAYER_ADD)
            {
                if (n == N - 1)
                {
                    packed_v = __float2bfloat16_rn(__bfloat162float(packed_v) + __bfloat162float(layer_res_add[h]));
                    updated_layer_res[h] = packed_v;
                }
            }
            float v = __bfloat162float(packed_v);
            if constexpr (N == 1)
            {
                if constexpr (FUSE_OUTPUT_RMS_NORM)
                {
                    mixed_cache[item] = packed_v;
                    output_sq_local = fmaf(v, v, output_sq_local);
                }
                else
                {
                    output[h] = packed_v;
                }
            }
            else
            {
                item_v[n] = packed_v;
            }
            stats[n] = float2_fma(make_float2(v, v), make_float2(v, q), stats[n]);
        }
        if constexpr (N > 1)
        {
#pragma unroll
            for (int pair = 0; pair < N / 2; pair++)
            {
                union
                {
                    __nv_bfloat162 bf16x2;
                    uint32_t bits;
                } packed;

                packed.bf16x2 = __halves2bfloat162(item_v[2 * pair], item_v[2 * pair + 1]);
                v_cache_bf16[item][pair] = packed.bits;
            }
            if constexpr (N % 2 == 1)
            {
                union
                {
                    __nv_bfloat162 bf16x2;
                    uint32_t bits;
                } packed;

                packed.bf16x2 = __halves2bfloat162(item_v[N - 1], __float2bfloat16_rn(0.f));
                v_cache_bf16[item][N / 2] = packed.bits;
            }
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            uint64_t packed = reinterpret_cast<uint64_t&>(stats[n]);
            packed = __shfl_down_sync(0xffffffff, packed, offset);
            float2 other = reinterpret_cast<float2&>(packed);
            stats[n] = float2_add(stats[n], other);
        }
    }
    if (lane == 0)
    {
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            warp_stats[warp * N + n] = stats[n];
        }
    }
    __syncthreads();

    if (tid < N)
    {
        float2 total = {};
#pragma unroll
        for (int w = 0; w < WARPS; w++)
        {
            total = float2_add(total, warp_stats[w * N + tid]);
        }
        warp_stats[tid] = total;
    }
    __syncthreads();

    if (tid == 0)
    {
        float local_rsigma[N];
        float local_logits[N];
        float max_logit = -FLT_MAX;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            float2 total = warp_stats[n];
            local_rsigma[n] = rsqrtf(total.x / H + rms_eps);
            local_logits[n] = total.y * local_rsigma[n];
            max_logit = fmaxf(max_logit, local_logits[n]);
        }
        float denominator = 0.0f;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            weights[n] = exp2f((local_logits[n] - max_logit) * LOG2_E);
            denominator += weights[n];
        }
        float inv_denominator = 1.0f / denominator;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            weights[n] *= inv_denominator;
            if (rsigma_out)
                rsigma_out[(size_t) n * num_tokens] = local_rsigma[n];
            if (logits_out)
                logits_out[(size_t) n * num_tokens] = local_logits[n];
            if (probs_out)
                probs_out[(size_t) n * num_tokens] = weights[n];
        }
    }
    __syncthreads();

    if constexpr (N > 1)
    {
#pragma unroll
        for (int item = 0; item < ITEMS; item++)
        {
            float value = 0.0f;
#pragma unroll
            for (int pair = 0; pair < N / 2; pair++)
            {
                union
                {
                    __nv_bfloat162 bf16x2;
                    uint32_t bits;
                } packed;

                packed.bits = v_cache_bf16[item][pair];
                float2 v = __bfloat1622float2(packed.bf16x2);
                value = fmaf(weights[2 * pair], v.x, value);
                value = fmaf(weights[2 * pair + 1], v.y, value);
            }
            if constexpr (N % 2 == 1)
            {
                union
                {
                    __nv_bfloat162 bf16x2;
                    uint32_t bits;
                } packed;

                packed.bits = v_cache_bf16[item][N / 2];
                float2 v = __bfloat1622float2(packed.bf16x2);
                value = fmaf(weights[N - 1], v.x, value);
            }
            int h = tid + item * THREADS;
            bf16_t const mixed = __float2bfloat16_rn(value);
            if constexpr (FUSE_OUTPUT_RMS_NORM)
            {
                mixed_cache[item] = mixed;
                float const mixed_float = __bfloat162float(mixed);
                output_sq_local = fmaf(mixed_float, mixed_float, output_sq_local);
            }
            else
            {
                output[h] = mixed;
            }
        }
    }

    if constexpr (FUSE_OUTPUT_RMS_NORM)
    {
        output_sq_local = warp_reduce_sum(output_sq_local);
        if (lane == 0)
        {
            warp_stats[warp].x = output_sq_local;
        }
        __syncthreads();
        if (tid == 0)
        {
            float output_sq = 0.f;
#pragma unroll
            for (int w = 0; w < WARPS; w++)
            {
                output_sq += warp_stats[w].x;
            }
            weights[0] = rsqrtf(output_sq / H + output_rms_eps);
        }
        __syncthreads();

        float const output_rsigma = weights[0];
#pragma unroll
        for (int item = 0; item < ITEMS; item++)
        {
            int h = tid + item * THREADS;
            output[h] = apply_output_rms_norm(mixed_cache[item], output_rsigma, output_rms_w[h]);
        }
    }

    if constexpr (ENABLE_PDL)
    {
        if (!early_trigger)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
#else
    if (cute::thread0())
    {
        printf("attn_res_fwd_s1_single_cta_kernel requires sm_100a\n");
    }
#endif
}

struct S1Handoff
{
    bool early_trigger = false;
};

template <int N, bool FUSE_OUTPUT_RMS_NORM = false, bool FUSE_LAYER_ADD = false>
static void launch_s1_single_cta(bf16_t const* block_residual, bf16_t const* layer_residual,
    bf16_t const* layer_residual_add, bf16_t const* res_weight, bf16_t const* rms_weight,
    bf16_t const* output_rms_weight, bf16_t* updated_layer_residual, bf16_t* output, float* rsigma, float* probs,
    float* logits, float rms_eps, float output_rms_eps, int num_tokens, cudaStream_t stream, S1Handoff handoff = {})
{
    if (tensorrt_llm::common::getEnvEnablePDL())
    {
        auto kernel = &attn_res_fwd_s1_single_cta_kernel<N, FUSE_OUTPUT_RMS_NORM, FUSE_LAYER_ADD, true>;
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(num_tokens);
        config.blockDim = dim3(256);
        config.stream = stream;
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        config.attrs = &attribute;
        config.numAttrs = 1;
        cudaLaunchKernelEx(&config, kernel, block_residual, layer_residual, layer_residual_add, res_weight, rms_weight,
            output_rms_weight, updated_layer_residual, output, rsigma, probs, logits, rms_eps, output_rms_eps,
            num_tokens, handoff.early_trigger);
    }
    else
    {
        attn_res_fwd_s1_single_cta_kernel<N, FUSE_OUTPUT_RMS_NORM, FUSE_LAYER_ADD, false>
            <<<num_tokens, 256, 0, stream>>>(block_residual, layer_residual, layer_residual_add, res_weight, rms_weight,
                output_rms_weight, updated_layer_residual, output, rsigma, probs, logits, rms_eps, output_rms_eps,
                num_tokens, false);
    }
}

template <int N, int GROUPS = 8, bool FUSE_OUTPUT_RMS_NORM = false, bool FUSE_LAYER_ADD = false,
    bool ENABLE_PDL = false>
__global__ void __launch_bounds__(256, 1) attn_res_fwd_s1_splitk_kernel(bf16_t const* __restrict__ block_res,
    bf16_t const* __restrict__ layer_res, bf16_t const* __restrict__ layer_res_add, bf16_t const* __restrict__ res_w,
    bf16_t const* __restrict__ rms_w, bf16_t const* __restrict__ output_rms_w, bf16_t* __restrict__ updated_layer_res,
    bf16_t* __restrict__ output, float* __restrict__ rsigma_out, float* __restrict__ probs_out,
    float* __restrict__ logits_out, float rms_eps, float output_rms_eps, int num_tokens, bool early_trigger)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    auto const wait_for_previous_grid = [&]
    {
        if constexpr (ENABLE_PDL)
        {
            cudaGridDependencySynchronize();

            if (early_trigger)
            {
                cudaTriggerProgrammaticLaunchCompletion();
            }
        }
    };
    wait_for_previous_grid();

    namespace cg = cooperative_groups;
    constexpr int H = 7168;
    constexpr int K_PER_CTA = H / GROUPS;
    constexpr int THREADS = 256;
    constexpr int WARPS = THREADS / 32;
    constexpr int ITEMS = (K_PER_CTA + THREADS - 1) / THREADS;
    constexpr float LOG2_E = 1.4426950408889634f;
    static_assert(H % GROUPS == 0);

    extern __shared__ char smem_raw[];
    float* v_cache = reinterpret_cast<float*>(smem_raw);
    float2* warp_stats = reinterpret_cast<float2*>(smem_raw + (size_t) N * K_PER_CTA * sizeof(float));
    float* weights = reinterpret_cast<float*>(warp_stats + WARPS * N);

    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    cg::cluster_group cluster = cg::this_cluster();
    int const group = cluster.block_rank();
    int const h_begin = group * K_PER_CTA;

    int const token = blockIdx.x / GROUPS;
    block_res += (size_t) token * H;
    layer_res += (size_t) token * H;
    output += (size_t) token * H;
    if constexpr (FUSE_LAYER_ADD)
    {
        layer_res_add += (size_t) token * H;
        updated_layer_res += (size_t) token * H;
    }
    if (rsigma_out != nullptr)
    {
        rsigma_out += token;
    }
    if (probs_out != nullptr)
    {
        probs_out += token;
    }
    if (logits_out != nullptr)
    {
        logits_out += token;
    }

    float sq[N] = {};
    float dot[N] = {};

#pragma unroll
    for (int ki = tid; ki < K_PER_CTA; ki += THREADS)
    {
        int h = h_begin + ki;
        float q = __bfloat162float(res_w[h]) * __bfloat162float(rms_w[h]);
#pragma unroll
        for (int n = 0; n < N; n++)
        {

            bf16_t const* row = n < N - 1 ? block_res + (size_t) n * num_tokens * H : layer_res;
            bf16_t packed_v = row[h];
            if constexpr (FUSE_LAYER_ADD)
            {
                if (n == N - 1)
                {
                    packed_v = __float2bfloat16_rn(__bfloat162float(packed_v) + __bfloat162float(layer_res_add[h]));
                    updated_layer_res[h] = packed_v;
                }
            }
            float v = __bfloat162float(packed_v);
            v_cache[(size_t) n * K_PER_CTA + ki] = v;
            sq[n] = fmaf(v, v, sq[n]);
            dot[n] = fmaf(v, q, dot[n]);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            sq[n] += __shfl_down_sync(0xffffffff, sq[n], offset);
            dot[n] += __shfl_down_sync(0xffffffff, dot[n], offset);
        }
    }
    if (lane == 0)
    {
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            warp_stats[warp * N + n] = make_float2(sq[n], dot[n]);
        }
    }
    __syncthreads();

    if (tid < N)
    {
        float2 total = {};
#pragma unroll
        for (int w = 0; w < WARPS; w++)
        {
            total = float2_add(total, warp_stats[w * N + tid]);
        }
        warp_stats[tid] = total;
    }

    cluster.sync();

    float2 cluster_total = {};
    if (tid < N)
    {
#pragma unroll
        for (int g = 0; g < GROUPS; g++)
        {
            float2 const* remote_stats = cluster.map_shared_rank(warp_stats, g);
            cluster_total = float2_add(cluster_total, remote_stats[tid]);
        }
    }

    cluster.sync();
    if (tid < N)
    {
        warp_stats[tid] = cluster_total;
    }
    __syncthreads();

    if (tid == 0)
    {
        float local_rsigma[N];
        float local_logits[N];
        float max_logit = -FLT_MAX;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            float2 total = warp_stats[n];
            local_rsigma[n] = rsqrtf(total.x / H + rms_eps);
            local_logits[n] = total.y * local_rsigma[n];
            max_logit = fmaxf(max_logit, local_logits[n]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            weights[n] = exp2f((local_logits[n] - max_logit) * LOG2_E);
            sum += weights[n];
        }
        float inv_sum = 1.0f / sum;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            weights[n] *= inv_sum;
            if (group == 0)
            {
                if (rsigma_out)
                    rsigma_out[(size_t) n * num_tokens] = local_rsigma[n];
                if (logits_out)
                    logits_out[(size_t) n * num_tokens] = local_logits[n];
                if (probs_out)
                    probs_out[(size_t) n * num_tokens] = weights[n];
            }
        }
    }

    cluster.sync();

    float output_sq_local = 0.f;
    bf16_t mixed_cache[ITEMS];
#pragma unroll
    for (int ki = tid, item = 0; ki < K_PER_CTA; ki += THREADS, item++)
    {
        float value = 0.0f;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            value = fmaf(weights[n], v_cache[(size_t) n * K_PER_CTA + ki], value);
        }
        bf16_t const mixed = __float2bfloat16_rn(value);
        if constexpr (FUSE_OUTPUT_RMS_NORM)
        {
            float const mixed_float = __bfloat162float(mixed);
            mixed_cache[item] = mixed;
            output_sq_local = fmaf(mixed_float, mixed_float, output_sq_local);
        }
        else
        {
            output[h_begin + ki] = mixed;
        }
    }

    if constexpr (FUSE_OUTPUT_RMS_NORM)
    {
        output_sq_local = warp_reduce_sum(output_sq_local);
        if (lane == 0)
        {
            warp_stats[warp].x = output_sq_local;
        }
        __syncthreads();
        if (tid == 0)
        {
            float output_sq = 0.f;
#pragma unroll
            for (int w = 0; w < WARPS; w++)
            {
                output_sq += warp_stats[w].x;
            }
            warp_stats[0].x = output_sq;
        }

        cluster.sync();

        if (tid == 0)
        {
            float output_sq = 0.f;
#pragma unroll
            for (int g = 0; g < GROUPS; g++)
            {
                float2 const* remote_stats = cluster.map_shared_rank(warp_stats, g);
                output_sq += remote_stats[0].x;
            }
            weights[0] = rsqrtf(output_sq / H + output_rms_eps);
        }

        cluster.sync();

        float const output_rsigma = weights[0];
#pragma unroll
        for (int ki = tid, item = 0; ki < K_PER_CTA; ki += THREADS, item++)
        {
            int const h = h_begin + ki;
            output[h] = apply_output_rms_norm(mixed_cache[item], output_rsigma, output_rms_w[h]);
        }
    }

    if constexpr (ENABLE_PDL)
    {
        if (!early_trigger)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
#else
    if (cute::thread0())
    {
        printf("attn_res_fwd_s1_splitk_kernel requires sm_100a\n");
    }
#endif
}

template <int N, int GROUPS = 8, bool FUSE_OUTPUT_RMS_NORM = false, bool FUSE_LAYER_ADD = false>
static void launch_s1_splitk(bf16_t const* block_residual, bf16_t const* layer_residual,
    bf16_t const* layer_residual_add, bf16_t const* res_weight, bf16_t const* rms_weight,
    bf16_t const* output_rms_weight, bf16_t* updated_layer_residual, bf16_t* output, float* rsigma, float* probs,
    float* logits, float rms_eps, float output_rms_eps, int num_tokens, cudaStream_t stream, S1Handoff handoff = {})
{
    constexpr int K_PER_CTA = 7168 / GROUPS;
    constexpr int WARPS = 8;
    constexpr size_t smem_size
        = (size_t) N * K_PER_CTA * sizeof(float) + (size_t) WARPS * N * sizeof(float2) + (size_t) N * sizeof(float);
    bool const enable_pdl = tensorrt_llm::common::getEnvEnablePDL();
    bool early_trigger = enable_pdl && handoff.early_trigger;
    auto kernel = enable_pdl ? &attn_res_fwd_s1_splitk_kernel<N, GROUPS, FUSE_OUTPUT_RMS_NORM, FUSE_LAYER_ADD, true>
                             : &attn_res_fwd_s1_splitk_kernel<N, GROUPS, FUSE_OUTPUT_RMS_NORM, FUSE_LAYER_ADD, false>;
    static std::once_flag attrs_set[2][64];
    int dev = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&dev));
    auto const set_attr = [&]
    { TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)); };
    int const pdl_index = enable_pdl ? 1 : 0;
    if (dev >= 0 && dev < 64)
    {
        std::call_once(attrs_set[pdl_index][dev], set_attr);
    }
    else
    {
        set_attr();
    }
    void* args[] = {const_cast<bf16_t**>(&block_residual), const_cast<bf16_t**>(&layer_residual),
        const_cast<bf16_t**>(&layer_residual_add), const_cast<bf16_t**>(&res_weight), const_cast<bf16_t**>(&rms_weight),
        const_cast<bf16_t**>(&output_rms_weight), &updated_layer_residual, &output, &rsigma, &probs, &logits, &rms_eps,
        &output_rms_eps, &num_tokens, &early_trigger};
    cudaLaunchConfig_t config{};

    config.gridDim = dim3(num_tokens * GROUPS);
    config.blockDim = dim3(256);
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim.x = GROUPS;
    attributes[0].val.clusterDim.y = 1;
    attributes[0].val.clusterDim.z = 1;
    if (enable_pdl)
    {
        attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attributes[1].val.programmaticStreamSerializationAllowed = 1;
    }
    config.attrs = attributes;
    config.numAttrs = enable_pdl ? 2 : 1;
    cudaLaunchKernelExC(&config, reinterpret_cast<void const*>(kernel), args);
}

}

}

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::kimiK3DecodeAttnRes
{

template <int N, bool FUSE_LAYER_ADD>
static void launchAttnResDecodeRmsNorm(AttnResFwdParams const& params, cudaStream_t stream)
{
    using namespace sm100::fwd_prod_v2;

    auto const* layer_residual_add = FUSE_LAYER_ADD ? params.layerResidualAdd : nullptr;
    auto* updated_layer_residual = FUSE_LAYER_ADD ? params.updatedLayerResidual : nullptr;
    S1Handoff const handoff{params.earlyTrigger};

    if constexpr (N <= 4)
    {
        launch_s1_single_cta<N, true, FUSE_LAYER_ADD>(params.blockResidual, params.layerResidual, layer_residual_add,
            params.resWeight, params.rmsWeight, params.outputRmsWeight, updated_layer_residual, params.output, nullptr,
            nullptr, nullptr, params.rmsEps, params.outputRmsEps, params.seqLen, stream, handoff);
    }
    else
    {
        launch_s1_splitk<N, 8, true, FUSE_LAYER_ADD>(params.blockResidual, params.layerResidual, layer_residual_add,
            params.resWeight, params.rmsWeight, params.outputRmsWeight, updated_layer_residual, params.output, nullptr,
            nullptr, nullptr, params.rmsEps, params.outputRmsEps, params.seqLen, stream, handoff);
    }
}

template <bool FUSE_LAYER_ADD>
static void invokeAttnResDecodeRmsNorm(AttnResFwdParams const& params, cudaStream_t stream)
{
    switch (params.numCandidates)
    {
    case 1: launchAttnResDecodeRmsNorm<1, FUSE_LAYER_ADD>(params, stream); break;
    case 2: launchAttnResDecodeRmsNorm<2, FUSE_LAYER_ADD>(params, stream); break;
    case 3: launchAttnResDecodeRmsNorm<3, FUSE_LAYER_ADD>(params, stream); break;
    case 4: launchAttnResDecodeRmsNorm<4, FUSE_LAYER_ADD>(params, stream); break;
    case 5: launchAttnResDecodeRmsNorm<5, FUSE_LAYER_ADD>(params, stream); break;
    case 6: launchAttnResDecodeRmsNorm<6, FUSE_LAYER_ADD>(params, stream); break;
    case 7: launchAttnResDecodeRmsNorm<7, FUSE_LAYER_ADD>(params, stream); break;
    case 8: launchAttnResDecodeRmsNorm<8, FUSE_LAYER_ADD>(params, stream); break;
    case 9: launchAttnResDecodeRmsNorm<9, FUSE_LAYER_ADD>(params, stream); break;
    case 10: launchAttnResDecodeRmsNorm<10, FUSE_LAYER_ADD>(params, stream); break;
    case 11: launchAttnResDecodeRmsNorm<11, FUSE_LAYER_ADD>(params, stream); break;
    case 12: launchAttnResDecodeRmsNorm<12, FUSE_LAYER_ADD>(params, stream); break;

    default:
        TLLM_CHECK_WITH_INFO(false,
            "Kimi K3 attn-res fused epilogue has no instantiation for numCandidates=%d (supported: 1-12)",
            params.numCandidates);
        break;
    }
}

void invokeAttnResRmsNormFwd(AttnResFwdParams const& params, cudaStream_t stream)
{
    invokeAttnResDecodeRmsNorm<false>(params, stream);
}

void invokeAttnResAddRmsNormFwd(AttnResFwdParams const& params, cudaStream_t stream)
{
    invokeAttnResDecodeRmsNorm<true>(params, stream);
}

}

TRTLLM_MV2_NAMESPACE_END
