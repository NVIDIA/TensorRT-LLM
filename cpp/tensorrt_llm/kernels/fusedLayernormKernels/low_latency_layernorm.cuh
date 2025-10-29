/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"
#include "tensorrt_llm/common/cudaBufferUtils.cuh"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/archCondition.h"
#include "tensorrt_llm/kernels/fusedLayernormKernels/ws_layernorm.cuh"

using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{

template <uint32_t N_THREADS, typename T, size_t N>
__forceinline__ __device__ void reduceSum(
    T (&val)[N], T (&reduce_ptr)[N_THREADS / 32][N], const uint32_t thread_id, const uint32_t barrier_name)
{

    static_assert(N_THREADS % 32 == 0);
    static_assert(N == 1 || N == 2);

    auto lane = thread_id % 32;
    auto wid = thread_id / 32;

    typename PackType<T, N>::type packed;

    packed.array[0] = val[0];
    if constexpr (N == 2)
    {
        packed.array[1] = val[1];
    }

    packed = batchWarpReduceSum<T, N>(packed);

    val[0] = packed.array[0];
    if constexpr (N == 2)
    {
        val[1] = packed.array[1];
    }

    if (lane == 0)
    {
        reduce_ptr[wid][0] = val[0];
        if constexpr (N == 2)
        {
            reduce_ptr[wid][1] = val[1];
        }
        __threadfence_block();
    }

    namedBarrierSync(barrier_name, N_THREADS);

    val[0] = reduce_ptr[0][0];
    if constexpr (N == 2)
    {
        val[1] = reduce_ptr[0][1];
    }

    for (int i = 1; i < N_THREADS / 32; i++)
    {
        val[0] += reduce_ptr[i][0];
        if constexpr (N == 2)
        {
            val[1] += reduce_ptr[i][1];
        }
    }
}

template <typename _Traits>
struct LowLatencyLayerNorm
{
    using Traits = _Traits;

    static_assert(Traits::LOW_LATENCY_MODE);
    static_assert(!Traits::PERSISTENT_MODE);
    static_assert(Traits::M_BLOCK == 1);

    static constexpr auto N_THREADS = Traits::MATH_WARPGROUPS * 128;

    using Param = typename Traits::Param;
    using FusedOperator = GetFusedOperator<typename Traits::FusedOperator>;

    struct Shared
    {
        // One elem per warp per reduce. 2x reduces to reduce mean and var at the same time.
        __align__(16) typename Traits::AccumulatorType reduce[Traits::MATH_WARPGROUPS * 4][Traits::RMS_NORM ? 1 : 2];
    };

    static __device__ void compute(const Param param, Shared* shared)
    {

        auto const thread_id = threadIdx.x;

        uint32_t work_id = blockIdx.x;

        FusedOperator fused_operator(param);

        constexpr auto PACKED_PER_N_BLOCK = Traits::N_BLOCK / N_THREADS / Traits::PACKED_ELEMS_PER_COMPUTE;

        typename Traits::AccumulatorType data[PACKED_PER_N_BLOCK][Traits::PACKED_ELEMS_PER_COMPUTE];
        typename Traits::AccumulatorType mean = 0.0f, variance = 0.0f;

        typename Traits::AccumulatorType r_bias[PACKED_PER_N_BLOCK][Traits::PACKED_ELEMS_PER_COMPUTE];
        typename Traits::AccumulatorType r_input_scale[PACKED_PER_N_BLOCK][Traits::PACKED_ELEMS_PER_COMPUTE];
        typename Traits::AccumulatorType r_residual[PACKED_PER_N_BLOCK][Traits::PACKED_ELEMS_PER_COMPUTE];
        typename Traits::AccumulatorType r_gamma[PACKED_PER_N_BLOCK][Traits::PACKED_ELEMS_PER_COMPUTE];
        typename Traits::AccumulatorType r_beta[PACKED_PER_N_BLOCK][Traits::PACKED_ELEMS_PER_COMPUTE];

        auto load_to_register = [thread_id = thread_id](auto g_data, auto& r_data, uint32_t sz)
        {
            constexpr auto N_THREADS = Traits::MATH_WARPGROUPS * 128;
            constexpr auto PACKED_PER_N_BLOCK = Traits::N_BLOCK / N_THREADS / Traits::PACKED_ELEMS_PER_COMPUTE;
            using PackedType =
                typename PackType<std::decay_t<decltype(g_data[0])>, Traits::PACKED_ELEMS_PER_COMPUTE>::type;

            PackedType data[PACKED_PER_N_BLOCK];
            for (int i = 0; i < PACKED_PER_N_BLOCK; i++)
            {
                auto offset = (thread_id + i * N_THREADS) * Traits::PACKED_ELEMS_PER_COMPUTE;
                if (offset <= sz)
                {
                    data[i] = *reinterpret_cast<PackedType const*>(&g_data[offset]);
                }
            }
            for (int i = 0; i < PACKED_PER_N_BLOCK; i++)
            {
                for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
                {
                    r_data[i][j] = (typename Traits::AccumulatorType) data[i].array[j];
                }
            }
        };

        static_assert(Traits::OUTPUT_SCALE != SCALE_TYPE::VECTOR);

        if constexpr (Traits::BIAS == SCALE_TYPE::VECTOR)
        {
            load_to_register(param.bias, r_bias, param.n);
        }

        if constexpr (Traits::INPUT_SCALE == SCALE_TYPE::VECTOR)
        {
            load_to_register(param.input_deq_ptr, r_input_scale, param.n);
        }

        if constexpr (Traits::GAMMA)
        {
            load_to_register(param.gamma, r_gamma, param.n);
        }

        if constexpr (Traits::BETA)
        {
            load_to_register(param.beta, r_beta, param.n);
        }

#if (defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 12))
        if constexpr (arch::is_major_v<9> || arch::is_major_v<10>)
        {
            asm volatile("griddepcontrol.wait;\n");
            asm volatile("griddepcontrol.launch_dependents;\n");
        }
#endif
        load_to_register(&param.input[work_id * param.n], data, param.n);

        if constexpr (Traits::RESIDUAL)
        {
            load_to_register(&param.residual[work_id * param.n], r_residual, param.n);
        }

#pragma unroll PACKED_PER_N_BLOCK
        for (int i = 0; i < PACKED_PER_N_BLOCK; i++)
        {

            auto n_base = (thread_id + i * N_THREADS) * Traits::PACKED_ELEMS_PER_COMPUTE;
            auto in_bound = n_base < param.n;

            for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
            {
                if constexpr (Traits::INPUT_SCALE == SCALE_TYPE::SCALAR)
                {
                    data[i][j] *= param.deq;
                }
                else if constexpr (Traits::INPUT_SCALE == SCALE_TYPE::VECTOR)
                {
                    data[i][j] *= r_input_scale[i][j];
                }

                if constexpr (Traits::BIAS == SCALE_TYPE::SCALAR)
                {
                    data[i][j] += param.bias;
                }
                else if constexpr (Traits::BIAS == SCALE_TYPE::VECTOR)
                {
                    data[i][j] += r_bias[i][j];
                }

                if constexpr (Traits::RESIDUAL)
                {
                    data[i][j] += r_residual[i][j];
                }
            }

            if constexpr (!Traits::RMS_NORM)
            {
                for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
                {
                    mean += in_bound ? data[i][j] : 0.0f;
                }
            }

            for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
            {
                variance += in_bound ? data[i][j] * data[i][j] : 0.0f;
            }

            if constexpr (Traits::UNNORMED_OUTPUT)
            {
                typename PackType<typename Traits::InputType, Traits::PACKED_ELEMS_PER_COMPUTE>::type output;
                for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
                {
                    output.array[j] = (typename Traits::InputType) data[i][j];
                }
                if (in_bound)
                    reinterpret_cast<decltype(output)*>(&param.output[work_id * param.n + n_base])[0] = output;
            }
        }

        typename Traits::AccumulatorType var_and_mean[Traits::RMS_NORM ? 1 : 2] = {variance};

        if constexpr (!Traits::RMS_NORM)
        {
            var_and_mean[1] = mean;
        }

        reduceSum<N_THREADS>(var_and_mean, shared->reduce, thread_id, 0);

        if constexpr (!Traits::RMS_NORM)
        {
            mean = var_and_mean[1] / param.n;
            variance = rsqrtf(
                var_and_mean[0] / param.n - var_and_mean[1] * var_and_mean[1] + (Traits::AccumulatorType)(1e-5));
        }
        else
        {
            variance = rsqrtf(var_and_mean[0] / param.n + (Traits::AccumulatorType)(1e-5));
        }

        for (int i = 0; i < PACKED_PER_N_BLOCK; i++)
        {
            auto n_base = (thread_id + i * N_THREADS) * Traits::PACKED_ELEMS_PER_COMPUTE;
            auto in_bound = n_base < param.n;
            if (!in_bound)
            {
                break;
            }

            typename PackType<typename Traits::OutputType, Traits::PACKED_ELEMS_PER_COMPUTE>::type normed_output;
            typename PackType<typename Traits::AccumulatorType, Traits::PACKED_ELEMS_PER_COMPUTE>::type
                high_precision_normed_output;
            for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
            {
                typename Traits::AccumulatorType normed_out;
                if constexpr (!Traits::RMS_NORM)
                {
                    normed_out = (data[i][j] - mean) * variance;
                }
                else
                {
                    normed_out = data[i][j] * variance;
                }

                if constexpr (Traits::GAMMA)
                {
                    normed_out *= r_gamma[i][j];
                }
                if constexpr (Traits::BETA)
                {
                    normed_out += r_beta[i][j];
                }
                if constexpr (Traits::HIGH_PRECISION_NORMED_OUTPUT)
                {
                    high_precision_normed_output.array[j] = normed_out;
                }
                if constexpr (Traits::OUTPUT_SCALE == SCALE_TYPE::SCALAR)
                {
                    normed_out *= param.output_qua_ptr[0];
                }
                normed_output.array[j] = (typename Traits::OutputType) normed_out;
            }
            // TODO: Move this generic writeback into dummy fused operator.
            if constexpr (std::is_same_v<typename Traits::FusedOperator, void>)
            {
                reinterpret_cast<decltype(normed_output)*>(&param.normed_output[work_id * param.n + n_base])[0]
                    = normed_output;
            }
            else
            {
                fused_operator.template post_process<Traits::PACKED_ELEMS_PER_COMPUTE, decltype(normed_output)>(
                    work_id, n_base, normed_output);
            }
            if constexpr (Traits::HIGH_PRECISION_NORMED_OUTPUT)
            {
                reinterpret_cast<decltype(high_precision_normed_output)*>(
                    &param.high_precision_normed_output[work_id * param.n + n_base])[0]
                    = high_precision_normed_output;
            }
        }
    }

    static __device__ void run(const Param param)
    {
        __shared__ Shared shared;
        compute(param, &shared);
    }
};

} // namespace tensorrt_llm::kernels
