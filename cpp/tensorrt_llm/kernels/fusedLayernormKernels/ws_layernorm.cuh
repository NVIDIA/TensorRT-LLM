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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/archCondition.h"
#include "tensorrt_llm/kernels/fusedLayernormKernels/ws_layernorm.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{

struct DummyFusedOperator
{

    template <typename Param>
    __device__ __forceinline__ DummyFusedOperator(Param const& p)
    {
    }

    template <size_t ELEMS_PER_THREAD, typename T>
    __device__ __forceinline__ void post_process(int m, int n_base, T packed_input)
    {
    }
};

template <typename T>
using GetFusedOperator = std::conditional_t<std::is_same_v<T, void>, DummyFusedOperator, T>;

template <typename _Traits>
struct WarpSpecializedLayerNorm
{

    /**
     * @brief Warp specialized generic LayerNorm/RMSNorm kernel.
     * Requires LayerN <= BLOCK_N.
     * Largest BLOCK_N will be 128reg/thread * 128threads = 16384.
     * Target BLOCK_M * BLOCK_N = 16384.
     * FastTransformer LayerNorm supports MAX_N = 1024*8 = 8192.
     *           out = input * input_scale + residual + bias
     * normed_output = (norm(out) * gamma + beta) * output_scale
     */

    using Traits = _Traits;
    using Param = WarpSpecializedParam<typename Traits::Param>;

    struct AuxData
    {
        typename Traits::InputType gamma[Traits::GAMMA ? 1 : 0][Traits::N_BLOCK];
        typename Traits::InputType beta[Traits::BETA ? 1 : 0][Traits::N_BLOCK];
        typename Traits::InputType bias[Traits::BIAS == SCALE_TYPE::VECTOR ? 1 : 0][Traits::N_BLOCK];
        typename Traits::AccumulatorType input_scale[Traits::INPUT_SCALE == SCALE_TYPE::VECTOR ? 1 : 0]
                                                    [Traits::N_BLOCK];
    };

    struct Shared
    {

        __align__(128) typename Traits::AccumulatorType wg_reduce[Traits::MATH_WARPGROUPS][4][Traits::M_BLOCK
            * (Traits::RMS_NORM ? 1 : 2)]; // Only var is needed for RMSNorm.
        __align__(128) typename Traits::InputType
            input_vec[Traits::STAGES][Traits::RESIDUAL ? 2 : 1][Traits::M_BLOCK * Traits::N_BLOCK];
        __align__(128)
            typename Traits::OutputType output_vec[Traits::USE_BULK_STORE ? (Traits::UNNORMED_OUTPUT ? 2 : 1) : 0]
                                                  [Traits::MATH_WARPGROUPS][Traits::M_BLOCK * Traits::N_BLOCK];
        __align__(128) AuxData aux_data;
        CircularBuffer<Traits::STAGES> input_vec_fifo;
        CircularBufferWithData<1, uint32_t> sched2dma;
        CircularBufferWithData<Traits::STAGES, uint32_t> dma2compute_cmd;
        OrderedMutex math_order_mutex;

        __device__ void init(int tid0)
        {
            if constexpr (Traits::PERSISTENT_MODE)
            {
                sched2dma.init(tid0, 1, 1);
                dma2compute_cmd.init(tid0, 1, 128);
                if (Traits::MATH_WARPGROUPS == 2)
                {
                    math_order_mutex.init(tid0, 128, 128);
                }
            }
            input_vec_fifo.init(tid0, 1, 128);
        }
    };

    static __device__ uint32_t scheduler(
        const uint32_t thread_id, const uint32_t grid_sz, Param const& param, Shared* shared)
    {
        auto prefetch_a_vec = [](void const* global_ptr, const uint32_t bytes)
        {
            asm volatile(
                "cp.async.bulk.prefetch.L2.global "
                "[%0], %1;\n"
                :
                : "l"(reinterpret_cast<uint64_t>(global_ptr)), "r"(bytes)
                :);
        };
        uint32_t scheduled_tiles = 0;
        if constexpr (Traits::PERSISTENT_MODE)
        {
            if (thread_id == 0)
            {
                auto sched2dma_w = shared->sched2dma.createWriter();
                for (;;)
                {
                    auto tile_id = atomicAdd(&(param.counters->tile_ctr), Traits::M_BLOCK);
                    if (tile_id >= param.m)
                    {
                        break;
                    }
                    sched2dma_w.push(tile_id);
                    if constexpr (Traits::PREFETCH_TO_L2)
                    {
                        prefetch_a_vec(param.input + param.n * tile_id,
                            Traits::M_BLOCK * param.n * sizeof(typename Traits::InputType));
                    }
                    scheduled_tiles++;
                    // if (blockIdx.x == 0) printf("Pushed tile %d to DMA.\n", tile_id);
                }
                sched2dma_w.push(0xffffffff);
                // if (blockIdx.x == 0) printf("Pushed tile -1 to DMA.\n");
                if (atomicAdd(&(param.counters->cta_completion_ctr), 1) == grid_sz - 1)
                {
                    param.counters->tile_ctr = 0;
                    param.counters->cta_completion_ctr = 0;
                }
            }
        }
        else
        {
            scheduled_tiles = 1;
        }
        return scheduled_tiles;
    }

    static __device__ void dma(const uint32_t block_id, const uint32_t thread_id, Param const& param, Shared* shared)
    {

        auto load_a_vec
            = [](void const* global_ptr, const uint32_t smem_ptr, const uint32_t bytes, const uint32_t barrier_ptr)
        {
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                "[%0], [%1], %2, [%3];\n"
                :
                : "r"(smem_ptr), "l"(reinterpret_cast<uint64_t>(global_ptr)), "r"(bytes), "r"(barrier_ptr)
                : "memory");
        };

        if (thread_id == 0)
        {
            auto sched2dma_r = shared->sched2dma.createReader();
            auto dma2compute_cmd_w = shared->dma2compute_cmd.createWriter();
            auto input_vec_fifo_w = shared->input_vec_fifo.createWriter();

            auto load_a_tile = [&](auto first_run)
            {
                constexpr bool FIRST_RUN = decltype(first_run)::value;

                static_assert(Traits::PERSISTENT_MODE || Traits::MATH_WARPGROUPS == 1);

                uint32_t m_base = 0;
                if constexpr (Traits::PERSISTENT_MODE)
                {
                    m_base = sched2dma_r.pop();
                    dma2compute_cmd_w.push(m_base);
                    if (m_base == 0xffffffff)
                    {
                        if constexpr (Traits::MATH_WARPGROUPS == 2)
                        {
                            dma2compute_cmd_w.push(m_base);
                        }
                        return false;
                    }
                }
                else
                {
                    m_base = block_id;
                }
                // if (blockIdx.x == 0) printf("Pushed tile %d to MATH.\n", m_base);

                const auto tx
                    = (Traits::M_BLOCK * param.n * sizeof(typename Traits::InputType) * (Traits::RESIDUAL ? 2 : 1))
                    + (FIRST_RUN ? sizeof(AuxData) / Traits::N_BLOCK * param.n : 0);

                auto vec_buffer_ptr = input_vec_fifo_w.tmaReserve(tx);

                // if (blockIdx.x == 0) printf("SMEM buffer ready, start loading tile %d.\n", m_base);

                if constexpr (FIRST_RUN)
                {
                    asm volatile("griddepcontrol.wait;\n");
                }

                for (int i = 0; i < Traits::M_BLOCK; i++)
                {
                    load_a_vec(&param.input[(m_base + i) * param.n],
                        __nvvm_get_smem_pointer(&shared->input_vec[vec_buffer_ptr][0][i * Traits::N_BLOCK]),
                        param.n * sizeof(typename Traits::InputType),
                        __nvvm_get_smem_pointer(input_vec_fifo_w.barrier_ptr(vec_buffer_ptr)));
                }

                // Use templated lambdas to defer resolving the symbols like "param.residual".
                // Otherwise compiler will complain about symbols, like param.residual, doesn't exist even if
                // corresponding traits, like Traits::RESIDUAL, are false.
                if constexpr (Traits::RESIDUAL)
                {
                    [&](auto param)
                    {
                        for (int i = 0; i < Traits::M_BLOCK; i++)
                        {
                            load_a_vec(&param.residual[(m_base + i) * param.n],
                                __nvvm_get_smem_pointer(&shared->input_vec[vec_buffer_ptr][1][i * Traits::N_BLOCK]),
                                param.n * sizeof(typename Traits::InputType),
                                __nvvm_get_smem_pointer(input_vec_fifo_w.barrier_ptr(vec_buffer_ptr)));
                        }
                    }(param);
                }

                if constexpr (FIRST_RUN)
                {
                    if constexpr (Traits::GAMMA)
                    {
                        [&](auto param)
                        {
                            load_a_vec(param.gamma, __nvvm_get_smem_pointer(&shared->aux_data.gamma[0]),
                                param.n * sizeof(typename Traits::InputType),
                                __nvvm_get_smem_pointer(input_vec_fifo_w.barrier_ptr(vec_buffer_ptr)));
                        }(param);
                    }

                    if constexpr (Traits::BETA)
                    {
                        [&](auto param)
                        {
                            load_a_vec(param.beta, __nvvm_get_smem_pointer(&shared->aux_data.beta[0]),
                                param.n * sizeof(typename Traits::InputType),
                                __nvvm_get_smem_pointer(input_vec_fifo_w.barrier_ptr(vec_buffer_ptr)));
                        }(param);
                    }

                    if constexpr (Traits::BIAS == SCALE_TYPE::VECTOR)
                    {
                        [&](auto param)
                        {
                            load_a_vec(param.bias, __nvvm_get_smem_pointer(&shared->aux_data.bias[0]),
                                param.n * sizeof(typename Traits::InputType),
                                __nvvm_get_smem_pointer(input_vec_fifo_w.barrier_ptr(vec_buffer_ptr)));
                        }(param);
                    }

                    if constexpr (Traits::INPUT_SCALE == SCALE_TYPE::VECTOR)
                    {
                        [&](auto param)
                        {
                            load_a_vec(param.input_deq_ptr, __nvvm_get_smem_pointer(&shared->aux_data.input_scale[0]),
                                param.n * sizeof(typename Traits::AccumulatorType),
                                __nvvm_get_smem_pointer(input_vec_fifo_w.barrier_ptr(vec_buffer_ptr)));
                        }(param);
                    }
                }

                return true;
            };

            if (load_a_tile(ConstBool<true>{}))
            {
                if constexpr (Traits::PERSISTENT_MODE)
                {
                    while (load_a_tile(ConstBool<false>{}))
                        ;
                }
            }
        }
        return;
    }

    static __device__ void compute(
        const uint32_t block_id, const uint32_t wgid, const uint32_t thread_id, Param const& param, Shared* shared)
    {

        auto store_a_vec = [](void const* global_ptr, const uint32_t smem_ptr, const uint32_t bytes)
        {
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                         :
                         : "l"(reinterpret_cast<uint64_t>(global_ptr)), "r"(smem_ptr), "r"(bytes)
                         : "memory");
        };

        auto wait_for_store = [](auto prev)
        {
            constexpr int PREV = decltype(prev)::value;
            if constexpr (PREV == 0)
            {
                asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
            }
            else if constexpr (PREV == 1)
            {
                asm volatile("cp.async.bulk.wait_group.read 1;\n" ::: "memory");
            }
            else if constexpr (PREV == 2)
            {
                asm volatile("cp.async.bulk.wait_group.read 2;\n" ::: "memory");
            }
            else if constexpr (PREV == 4)
            {
                asm volatile("cp.async.bulk.wait_group.read 4;\n" ::: "memory");
            }
            else if constexpr (PREV == 8)
            {
                asm volatile("cp.async.bulk.wait_group.read 8;\n" ::: "memory");
            }
            else if constexpr (PREV == 16)
            {
                asm volatile("cp.async.bulk.wait_group.read 16;\n" ::: "memory");
            }
            else if constexpr (PREV == 32)
            {
                asm volatile("cp.async.bulk.wait_group.read 32;\n" ::: "memory");
            }
            else
            {
                static_assert(PREV == 0);
            }
        };

        auto warpGroupReduceSum
            = [](typename Traits::AccumulatorType(&val)[Traits::M_BLOCK * (Traits::RMS_NORM ? 1 : 2)],
                  typename Traits::AccumulatorType(&reduce_ptr)[4][Traits::M_BLOCK * (Traits::RMS_NORM ? 1 : 2)],
                  const uint32_t thread_id, const uint32_t barrier_name)
        {
            auto lane = thread_id % 32;
            auto wid = thread_id / 32;

            constexpr int PACK = std::min((size_t) Traits::M_BLOCK * (Traits::RMS_NORM ? 1 : 2),
                sizeof(uint64_t) / sizeof(typename Traits::AccumulatorType));

            static_assert(Traits::M_BLOCK * (Traits::RMS_NORM ? 1 : 2) % PACK == 0);
#pragma unroll Traits::M_BLOCK*(Traits::RMS_NORM ? 1 : 2) / PACK
            for (int i = 0; i < Traits::M_BLOCK * (Traits::RMS_NORM ? 1 : 2); i += PACK)
            {
                typename PackType<typename Traits::AccumulatorType, PACK>::type packed;
#pragma unroll PACK
                for (int j = 0; j < PACK; j++)
                {
                    packed.array[j] = val[i + j];
                }
                packed = batchWarpReduceSum<typename Traits::AccumulatorType, PACK>(packed);
#pragma unroll PACK
                for (int j = 0; j < PACK; j++)
                {
                    val[i + j] = packed.array[j];
                }
            }

            if (lane == 0)
            {
#pragma unroll Traits::M_BLOCK*(Traits::RMS_NORM ? 1 : 2)
                for (int i = 0; i < Traits::M_BLOCK * (Traits::RMS_NORM ? 1 : 2); i++)
                {
                    reduce_ptr[wid][i] = val[i];
                }
                __threadfence_block();
            }

            namedBarrierSync(barrier_name, 128);

#pragma unroll Traits::M_BLOCK*(Traits::RMS_NORM ? 1 : 2)
            for (int j = 0; j < Traits::M_BLOCK * (Traits::RMS_NORM ? 1 : 2); j++)
            {
                val[j] = reduce_ptr[0][j];
            }
            for (int i = 1; i < 4; i++)
            {
#pragma unroll Traits::M_BLOCK*(Traits::RMS_NORM ? 1 : 2)
                for (int j = 0; j < Traits::M_BLOCK * (Traits::RMS_NORM ? 1 : 2); j++)
                {
                    val[j] += reduce_ptr[i][j];
                }
            }

            namedBarrierSync(barrier_name, 128);
        };

        using InputPacked = typename PackType<typename Traits::InputType, Traits::PACKED_ELEMS_PER_COMPUTE>::type;

        auto const buffer_id = Traits::MATH_WARPGROUPS == 2 ? (wgid == 0 ? 0 : 1) : 0;
        auto dma2compute_cmd_r = shared->dma2compute_cmd.createReader();
        auto input_vec_fifo_r = shared->input_vec_fifo.createReader();

        static_assert(Traits::MATH_WARPGROUPS == 1 || Traits::MATH_WARPGROUPS == 2);

        OrderedMutexAccessor math_mutex{shared->math_order_mutex, buffer_id, {}};

        if (buffer_id == 1)
        {
            dma2compute_cmd_r.advance();
            input_vec_fifo_r.advance();
            math_mutex.arrive();
        }

        using FusedOperator = GetFusedOperator<typename Traits::FusedOperator>;

        FusedOperator fused_operator(param);

        static_assert(Traits::PERSISTENT_MODE || Traits::MATH_WARPGROUPS == 1);

        for (;;)
        {
            uint32_t m_base = 0;
            if constexpr (Traits::PERSISTENT_MODE)
            {
                m_base = dma2compute_cmd_r.pop();
                if (m_base == 0xffffffff)
                {
                    break;
                }
                if constexpr (Traits::MATH_WARPGROUPS == 2)
                {
                    dma2compute_cmd_r.advance();
                }
            }
            else
            {
                m_base = block_id;
            }
            // if (blockIdx.x == 0 && thread_id == 0) printf("MATH got tile %d.\n", m_base);

            // Peek for data ready.
            auto data_ready = input_vec_fifo_r.peek();
            auto s_input = shared->input_vec[input_vec_fifo_r.ptr()][0];
            auto s_residual = shared->input_vec[input_vec_fifo_r.ptr()][1];

            static_assert(Traits::N_BLOCK % (Traits::PACKED_ELEMS_PER_COMPUTE * 128) == 0);

            constexpr auto PACKED_PER_N_BLOCK = Traits::N_BLOCK / 128 / Traits::PACKED_ELEMS_PER_COMPUTE;

            // Accumulators. threadX holds vec[X*PACKED_ELEMS_PER_COMP...(X+1)*PACKED_ELEMS_PER_COMP),
            // vec[(128+X)*PACKED_ELEMS_PER_COMP, (129+X)*PACKED_ELEMS_PER_COMP), ... for every M.
            typename Traits::AccumulatorType data[Traits::M_BLOCK][PACKED_PER_N_BLOCK]
                                                 [Traits::PACKED_ELEMS_PER_COMPUTE];
            typename Traits::AccumulatorType
                mean_and_var[Traits::M_BLOCK * (Traits::RMS_NORM ? 1 : 2)]; // Only var is needed for RMSNorm.
            typename Traits::AccumulatorType *variance = &mean_and_var[0],
                                             *mean = Traits::RMS_NORM ? nullptr : &mean_and_var[Traits::M_BLOCK];

            for (int i = 0; i < Traits::M_BLOCK; i++)
            {
                if constexpr (!Traits::RMS_NORM)
                {
                    mean[i] = 0.0f;
                }
                variance[i] = 0.0f;
            }

            if constexpr (Traits::MATH_WARPGROUPS == 2)
            {
                math_mutex.wait();
            }

            // Wait for data ready.
            if (!data_ready)
            {
                input_vec_fifo_r.wait();
            }
// Load data.
#pragma unroll Traits::M_BLOCK
            for (int m_offset = 0; m_offset < Traits::M_BLOCK; m_offset++)
            {
#pragma unroll PACKED_PER_N_BLOCK
                for (int i = 0; i < PACKED_PER_N_BLOCK; i++)
                {
                    InputPacked input = *(InputPacked*) &s_input[m_offset * Traits::N_BLOCK
                        + (thread_id + i * 128) * Traits::PACKED_ELEMS_PER_COMPUTE];

#pragma unroll Traits::PACKED_ELEMS_PER_COMPUTE
                    for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
                    {
                        data[m_offset][i][j] = ((typename Traits::AccumulatorType) input.array[j]);
                    }
                }
            }

// Load input complete. Post-process, calculate mean and var.
#pragma unroll Traits::M_BLOCK
            for (int m_offset = 0; m_offset < Traits::M_BLOCK; m_offset++)
            {
#pragma unroll PACKED_PER_N_BLOCK
                for (int i = 0; i < PACKED_PER_N_BLOCK; i++)
                {
                    auto n_base = (thread_id + i * 128) * Traits::PACKED_ELEMS_PER_COMPUTE;
                    auto in_bound = n_base < param.n;

                    if constexpr (Traits::INPUT_SCALE != SCALE_TYPE::NONE)
                    {
                        typename Traits::AccumulatorType input_scale;
                        if constexpr (Traits::INPUT_SCALE == SCALE_TYPE::VECTOR)
                        {
                            input_scale
                                = shared->aux_data
                                      .input_scale[0][(thread_id + i * 128) * Traits::PACKED_ELEMS_PER_COMPUTE / 8];
                        }
                        else if constexpr (Traits::INPUT_SCALE == SCALE_TYPE::SCALAR)
                        {
                            input_scale = param.deq;
                        }
                        else
                        {
                            static_assert(Traits::INPUT_SCALE == SCALE_TYPE::NONE);
                        }
#pragma unroll Traits::PACKED_ELEMS_PER_COMPUTE
                        for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
                        {
                            data[m_offset][i][j] *= input_scale;
                        }
                    }

                    if constexpr (Traits::BIAS != SCALE_TYPE::NONE)
                    {
                        typename Traits::AccumulatorType bias;
                        static_assert(Traits::BIAS == SCALE_TYPE::SCALAR || Traits::BIAS == SCALE_TYPE::VECTOR);
                        if constexpr (Traits::BIAS == SCALE_TYPE::SCALAR)
                        {
                            bias = param.bias;
                        }
                        InputPacked bias_packed;
                        if constexpr (Traits::BIAS == SCALE_TYPE::VECTOR)
                        {
                            bias_packed = *(InputPacked*) &shared->aux_data
                                               .bias[0][(thread_id + i * 128) * Traits::PACKED_ELEMS_PER_COMPUTE];
                        }

#pragma unroll Traits::PACKED_ELEMS_PER_COMPUTE
                        for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
                        {
                            if constexpr (Traits::BIAS == SCALE_TYPE::VECTOR)
                            {
                                bias = bias_packed.array[j];
                            }
                            data[m_offset][i][j] += bias;
                        }
                    }

                    if constexpr (Traits::RESIDUAL)
                    {
                        InputPacked residual = *(InputPacked*) &s_residual[m_offset * Traits::N_BLOCK
                            + (thread_id + i * 128) * Traits::PACKED_ELEMS_PER_COMPUTE];

#pragma unroll Traits::PACKED_ELEMS_PER_COMPUTE
                        for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
                        {
                            data[m_offset][i][j] += ((typename Traits::AccumulatorType) residual.array[j]);
                        }
                    }

                    if constexpr (!Traits::RMS_NORM)
                    {
#pragma unroll Traits::PACKED_ELEMS_PER_COMPUTE
                        for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
                        {
                            mean[m_offset] += in_bound ? data[m_offset][i][j] : 0.0f;
                        }
                    }

#pragma unroll Traits::PACKED_ELEMS_PER_COMPUTE
                    for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
                    {
                        variance[m_offset] += in_bound ? data[m_offset][i][j] * data[m_offset][i][j] : 0.0f;
                    }
                }
            }

            if constexpr (Traits::MATH_WARPGROUPS == 2)
            {
                math_mutex.arrive();
            }
            input_vec_fifo_r.complete(input_vec_fifo_r.ptr());

#pragma unroll Traits::MATH_WARPGROUPS
            for (int i = 0; i < Traits::MATH_WARPGROUPS; i++)
            {
                input_vec_fifo_r.advance();
            }

            warpGroupReduceSum(mean_and_var, shared->wg_reduce[buffer_id], thread_id, buffer_id + 1);

#pragma unroll Traits::M_BLOCK
            for (int m_offset = 0; m_offset < Traits::M_BLOCK; m_offset++)
            {
                if constexpr (!Traits::RMS_NORM)
                {
                    mean[m_offset] /= param.n;
                    variance[m_offset] = rsqrtf(variance[m_offset] / param.n - mean[m_offset] * mean[m_offset]
                        + (Traits::AccumulatorType)(1e-5));
                }
                else
                {
                    variance[m_offset] = rsqrtf(variance[m_offset] / param.n + (Traits::AccumulatorType)(1e-5));
                }
            }

// Calculate output
#pragma unroll PACKED_PER_N_BLOCK
            for (int i = 0; i < PACKED_PER_N_BLOCK; i++)
            {
                InputPacked gamma_in, beta_in;
                typename Traits::AccumulatorType gamma[Traits::PACKED_ELEMS_PER_COMPUTE],
                    beta[Traits::PACKED_ELEMS_PER_COMPUTE];

                auto n_base = (thread_id + i * 128) * Traits::PACKED_ELEMS_PER_COMPUTE;
                auto in_bound = n_base < param.n;
                if (!in_bound)
                {
                    break;
                }

                if constexpr (Traits::GAMMA)
                {
                    gamma_in = *(InputPacked*) &shared->aux_data.gamma[0][n_base];
                }

                if constexpr (Traits::BETA)
                {
                    beta_in = *(InputPacked*) &shared->aux_data.beta[0][n_base];
                }

#pragma unroll Traits::PACKED_ELEMS_PER_COMPUTE
                for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
                {
                    if constexpr (Traits::GAMMA)
                    {
                        gamma[j] = (typename Traits::AccumulatorType) gamma_in.array[j];
                    }
                    if constexpr (Traits::BETA)
                    {
                        beta[j] = (typename Traits::AccumulatorType) beta_in.array[j];
                    }
                }

#pragma unroll Traits::M_BLOCK
                for (int m_offset = 0; m_offset < Traits::M_BLOCK; m_offset++)
                {
                    auto m = m_base + m_offset;

                    typename PackType<typename Traits::OutputType, Traits::PACKED_ELEMS_PER_COMPUTE>::type
                        normed_output;
                    typename PackType<typename Traits::InputType, Traits::PACKED_ELEMS_PER_COMPUTE>::type output;
                    typename PackType<typename Traits::AccumulatorType, Traits::PACKED_ELEMS_PER_COMPUTE>::type
                        high_precision_normed_output;

#pragma unroll Traits::PACKED_ELEMS_PER_COMPUTE
                    for (int j = 0; j < Traits::PACKED_ELEMS_PER_COMPUTE; j++)
                    {
                        auto n = n_base + j;

                        typename Traits::AccumulatorType normed_out;

                        if constexpr (!Traits::RMS_NORM)
                        {
                            normed_out = (data[m_offset][i][j] - mean[m_offset]) * variance[m_offset];
                        }
                        else
                        {
                            normed_out = data[m_offset][i][j] * variance[m_offset];
                        }

                        if constexpr (Traits::GAMMA)
                        {
                            normed_out *= gamma[j];
                        }

                        if constexpr (Traits::BETA)
                        {
                            normed_out += beta[j];
                        }

                        if constexpr (Traits::OUTPUT_SCALE != SCALE_TYPE::NONE)
                        {
                            static_assert(Traits::OUTPUT_SCALE == SCALE_TYPE::SCALAR);
                            normed_out *= param.output_qua_ptr[0];
                        }

                        if constexpr (Traits::UNNORMED_OUTPUT)
                        {
                            output.array[j] = (typename Traits::InputType) data[m_offset][i][j];
                        }

                        if constexpr (Traits::HIGH_PRECISION_NORMED_OUTPUT)
                        {
                            high_precision_normed_output.array[j] = normed_out;
                        }

                        normed_output.array[j] = (typename Traits::OutputType) normed_out;
                    }

                    if constexpr (Traits::USE_BULK_STORE)
                    {
                        if (i == 0 && m_offset == 0)
                        {
                            if (thread_id == 0)
                            {
                                wait_for_store(ConstInt<0>{});
                            }
                            namedBarrierSync(buffer_id + 1, 128);
                        }
                        reinterpret_cast<decltype(normed_output)*>(
                            &shared->output_vec[0][buffer_id][m_offset * Traits::N_BLOCK + n_base])[0]
                            = normed_output;
                        if constexpr (Traits::UNNORMED_OUTPUT)
                        {
                            reinterpret_cast<decltype(output)*>(
                                &shared->output_vec[0][buffer_id][m_offset * Traits::N_BLOCK + n_base])[0]
                                = output;
                        }
                    }
                    else
                    {
                        if constexpr (Traits::UNNORMED_OUTPUT)
                        {
                            reinterpret_cast<decltype(output)*>(&param.output[m * param.n + n_base])[0] = output;
                        }
                        // TODO: Move this generic writeback into dummy fused operator.
                        if constexpr (std::is_same_v<typename Traits::FusedOperator, void>)
                        {
                            reinterpret_cast<decltype(normed_output)*>(&param.normed_output[m * param.n + n_base])[0]
                                = normed_output;
                        }
                        else
                        {
                            fused_operator
                                .template post_process<Traits::PACKED_ELEMS_PER_COMPUTE, decltype(normed_output)>(
                                    m, n_base, normed_output);
                        }
                        if constexpr (Traits::HIGH_PRECISION_NORMED_OUTPUT)
                        {
                            reinterpret_cast<decltype(high_precision_normed_output)*>(
                                &param.high_precision_normed_output[m * param.n + n_base])[0]
                                = high_precision_normed_output;
                        }
                    }
                }
                if constexpr (Traits::USE_BULK_STORE)
                {
                    __threadfence_block();
                    namedBarrierSync(buffer_id + 1, 128);

#pragma unroll Traits::M_BLOCK
                    for (int i = 0; i < Traits::M_BLOCK; i++)
                    {
                        if (thread_id == 0)
                        {
                            store_a_vec(&param.normed_output[(m_base + i) * param.n],
                                __nvvm_get_smem_pointer(&shared->output_vec[0][buffer_id][Traits::N_BLOCK * i]),
                                param.n * sizeof(typename Traits::OutputType));
                            if constexpr (Traits::UNNORMED_OUTPUT)
                            {
                                store_a_vec(&param.output[(m_base + i) * param.n],
                                    __nvvm_get_smem_pointer(&shared->output_vec[1][buffer_id][Traits::N_BLOCK * i]),
                                    param.n * sizeof(typename Traits::InputType));
                            }
                            asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
                        }
                    }
                }
            }
            if constexpr (!Traits::PERSISTENT_MODE)
            {
                break;
            }
        }
        return;
    }

    static __device__ void run(Param const& param)
    {

        extern __shared__ char smem[];
        Shared* shared = (Shared*) smem;
        shared->init(threadIdx.x == 0);

        __syncthreads();
#if (defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 12))
#if (defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(__CUDA_ARCH_FEAT_SM100_ALL))
        if constexpr (arch::is_major_v<9> || arch::is_major_v<10>)
        {
            auto block_id = blockIdx.x;
            auto warp_id = threadIdx.x / 32;
            auto lane_id = threadIdx.x % 32;
            auto tid_in_wg = threadIdx.x % 128;

            if (warp_id < 4)
            {
                asm volatile("{setmaxnreg.dec.sync.aligned.u32 56; \n\t}");
                if (warp_id == 0)
                {
                    scheduler(lane_id, gridDim.x * gridDim.y * gridDim.z, param, shared);
                    // PRE-EXIT after all tiles have been scheduled.
                    asm volatile("griddepcontrol.launch_dependents;\n");
                }
                else if (warp_id == 1)
                {
                    dma(block_id, lane_id, param, shared);
                }
            }
            else
            {
                asm volatile("{setmaxnreg.inc.sync.aligned.u32 224; \n\t}");
                compute(block_id, threadIdx.x / 128 - 1, tid_in_wg, param, shared);
            }
        }
#endif
#endif
    }
};

template <typename T, size_t TARGET_THREADS = 384>
__global__ void __launch_bounds__(TARGET_THREADS, 1) warpSpecializedInvoker(typename T::Param param)
{
    T::run(param);
}

} // namespace tensorrt_llm::kernels
