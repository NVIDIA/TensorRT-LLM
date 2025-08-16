/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nccl.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "tensorrt_llm/kernels/communicationKernels/allReduceWorkspace.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAllReduceFusionKernels.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/kernels/rmsnormKernels.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"

namespace mpi = tensorrt_llm::mpi;
namespace tr = tensorrt_llm::runtime;
using namespace tensorrt_llm::kernels;

template <typename DType>
__global__ void residual_add_kernel(DType* data, DType* residual, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    data[idx] = data[idx] + residual[idx];
}

template <typename DType>
void residual_add(DType* data, DType* residual, int size, cudaStream_t stream)
{
    residual_add_kernel<<<size / 128, 128, 0, stream>>>(data, residual, size);
}

template <typename DType>
__global__ void cast_to_fp32_kernel(DType* in, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    out[idx] = static_cast<float>(in[idx]);
}

template <typename DType>
void cast_to_fp32(DType* in, float* out, int size, cudaStream_t stream)
{
    cast_to_fp32_kernel<<<size / 128, 128, 0, stream>>>(in, out, size);
}

template <typename T>
void print(int rank, void* _pa, int size)
{
    auto pa = reinterpret_cast<T*>(_pa);
    if (rank == 0)
    {
        printf("print: [");
        for (int n = 0; n < 20; ++n)
        {
            float v = static_cast<float>(pa[n]);
            printf("%f, ", v);
        }
        printf("...]\n");
    }
}

template <typename T>
float compare(int rank, void* _pa, void* _pb, int size, float scale, std::string const& cmp_info = "")
{
    auto pa = reinterpret_cast<T*>(_pa);
    auto pb = reinterpret_cast<T*>(_pb);
    float max_diff = 0.f, tot_diff = 0.f;
    float max_val = 0.f;
    int diff_cnt = 0;
    float threshold = 1e-7;
    static char* ar_debug = std::getenv("AR_DEBUG");
    if (ar_debug && rank == 0)
    {
        printf("TensorA: [");
        for (int n = 0; n < 20; ++n)
        {
            float v = static_cast<float>(pa[n]);
            printf("%f, ", v);
        }
        printf("...]\n");
        printf("TensorB: [");
        for (int n = 0; n < 20; ++n)
        {
            float v = static_cast<float>(pb[n]);
            printf("%f, ", v);
        }
        printf("...]\n");
    }
    int print_cnt = 0;
    for (int n = 0; n < size; ++n)
    {
        float va = static_cast<float>(pa[n]);
        float vb = static_cast<float>(pb[n]);
        max_val = std::max(max_val, vb);
        float diff = std::abs(va - vb);
        if (diff > threshold)
        {
            max_diff = std::max(max_diff, diff);
            tot_diff += diff;
            ++diff_cnt;
        }
        if (rank == 0 && print_cnt < 20 && ar_debug && diff / (std::abs(vb) + 1e-7) > 0.1)
        {
            ++print_cnt;
            printf("idx %d, va %f, vb %f\n", n, va, vb);
        }
    }
    float diff_thres = max_val * scale;
    if (rank == 0)
    {
        TLLM_LOG_INFO("[%s] rank %d, max diff %f (diff threshold %f), avg diff %f, diff cnt %d/%d", cmp_info.c_str(),
            rank, max_diff, diff_thres, tot_diff / std::max(diff_cnt, 1), diff_cnt, size);
    }
    return max_diff <= diff_thres;
}

template <typename T1, typename T2>
void random_fill(T1* data, int size, T2 minv, T2 maxv)
{
    static int rseed = 20250227;
    std::mt19937 gen(rseed++);
    std::uniform_real_distribution<float> dis(static_cast<float>(minv), static_cast<float>(maxv));
    for (int i = 0; i < size; ++i)
    {
        data[i] = static_cast<T1>(dis(gen));
    }
}

struct CudaBuffer
{
    void* m_d_data;
    void* m_h_data;
    int m_size;

    CudaBuffer(int size_in_bytes = 0)
        : m_size(size_in_bytes)
        , m_d_data(nullptr)
        , m_h_data(nullptr)
    {
        allocate(size_in_bytes);
    }

    void allocate(int size_in_bytes)
    {
        if (size_in_bytes == 0)
            return;
        TLLM_CHECK(m_d_data == nullptr && m_h_data == nullptr);
        m_size = size_in_bytes;
        TLLM_CUDA_CHECK(cudaMalloc(&m_d_data, m_size));
        TLLM_CUDA_CHECK(cudaMemset(m_d_data, 0, m_size));
        m_h_data = malloc(m_size);
    }

    template <typename T = void>
    T* device_data()
    {
        TLLM_CHECK(m_d_data != nullptr);
        return reinterpret_cast<T*>(m_d_data);
    }

    template <typename T = void>
    T* host_data()
    {
        TLLM_CHECK(m_h_data != nullptr);
        d2h();
        return reinterpret_cast<T*>(m_h_data);
    }

    template <typename DType, typename VType>
    void random(VType minv, VType maxv)
    {
        random_fill(reinterpret_cast<DType*>(m_h_data), m_size / sizeof(DType), minv, maxv);
        h2d();
    }

    void h2d()
    {
        TLLM_CUDA_CHECK(cudaMemcpy(m_d_data, m_h_data, m_size, cudaMemcpyHostToDevice));
    }

    void d2h()
    {
        TLLM_CUDA_CHECK(cudaMemcpy(m_h_data, m_d_data, m_size, cudaMemcpyDeviceToHost));
    }

    ~CudaBuffer()
    {
        if (m_d_data)
        {
            TLLM_CUDA_CHECK(cudaFree(m_d_data));
        }
        if (m_h_data)
        {
            free(m_h_data);
        }
    }
};

/////////////////////////////////////////////////////////////////
//                  * MoE Reduction Fusion *                   //
/////////////////////////////////////////////////////////////////

template <typename IOType>
union ACCESS_TYPE
{
    static constexpr int ELEM_PER_ACCESS = 16 / sizeof(IOType);

    // For LDG.128 STG.128 access
    int4 packed;
    IOType unpacked[ELEM_PER_ACCESS];
};

template <typename IOType, typename ScaleType>
__global__ void moe_reduction_kernel(IOType const* ggemm2_actexp_m_hidden_in, IOType const* fc2_m_hidden_in,
    ScaleType const* scale_actexp_m_in, int const* actexpi_to_global_expid, IOType* reduce_m_hidden_ou, int num_act_exp,
    int num_token, int hidden_size)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

    static_assert(sizeof(ScaleType) >= sizeof(IOType), "This kernel assume scale type is more precious than io type");
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();

    using ACC_TYPE = ACCESS_TYPE<IOType>;

    // Each cluster handle one token
    // Each thread handle ACC_TYPE::ELEM_PER_ACCESS element per token per expert

    int threadid_in_cluster = cluster.thread_rank();
    // Start Offset within one token's hidden_size of element
    // Current thread handle token[thread_offset_within_token : thread_offset_within_token + ACC_TYPE::ELEM_PER_ACCESS]
    int thread_offset_within_token = threadid_in_cluster * ACC_TYPE::ELEM_PER_ACCESS;

    if (thread_offset_within_token >= hidden_size)
    {
        return;
    }

    cudaGridDependencySynchronize();

    // Same as AR + Fusion kernel, use persistent kernel design
    for (int token_id = grid.cluster_rank(); token_id < num_token; token_id += grid.num_clusters())
    {

        // Offset within (num_token, hidden_size) in unit of element
        int thread_offset_across_token = token_id * hidden_size + thread_offset_within_token;

        ACC_TYPE accumulator;
#pragma unroll
        for (int i = 0; i < ACC_TYPE::ELEM_PER_ACCESS; ++i)
        {
            accumulator.unpacked[i] = static_cast<IOType>(0);
        }

        // * Iterate through all active expert
        for (int actexp_i = 0; actexp_i < num_act_exp; ++actexp_i)
        {

            // * Load active expert i's token j's partial data
            // Offset within (num_act_exp, num_token, hidden_size) in unit of element
            int thread_offset_across_actexp_token = actexp_i * (hidden_size * num_token) + thread_offset_across_token;
            ACC_TYPE actexp_i_data;
            actexp_i_data.packed = reinterpret_cast<int4 const*>(
                ggemm2_actexp_m_hidden_in)[thread_offset_across_actexp_token / ACC_TYPE::ELEM_PER_ACCESS];

            // * Load active expert i's token j's scale
            int gloabl_exp_id = actexpi_to_global_expid[actexp_i];
            int thread_offset_scale = gloabl_exp_id * num_token + token_id;
            ScaleType actexp_i_token_j_scale
                = reinterpret_cast<ScaleType const*>(scale_actexp_m_in)[thread_offset_scale];

// * acc += scale(data)
#pragma unroll
            for (int i = 0; i < ACC_TYPE::ELEM_PER_ACCESS; ++i)
            {
                // assume computation is done in ScaleType
                accumulator.unpacked[i] += static_cast<IOType>(
                    (static_cast<ScaleType>(actexp_i_data.unpacked[i]) * actexp_i_token_j_scale));
            }
        }

        // * FC2 + reduced(gGEMM2)
        ACC_TYPE fc2_data;
        fc2_data.packed
            = reinterpret_cast<int4 const*>(fc2_m_hidden_in)[thread_offset_across_token / ACC_TYPE::ELEM_PER_ACCESS];
#pragma unroll
        for (int i = 0; i < ACC_TYPE::ELEM_PER_ACCESS; ++i)
        {
            accumulator.unpacked[i] += fc2_data.unpacked[i];
        }

        // * Store
        // Only store valid section of ACC_TYPE::ELEM_PER_ACCESS
        reinterpret_cast<int4*>(reduce_m_hidden_ou)[thread_offset_across_token / ACC_TYPE::ELEM_PER_ACCESS]
            = accumulator.packed;
    }

    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename IOType, typename ScaleType>
void moe_reduction_kernel_launcher(IOType const* ggemm2_actexp_m_hidden_in, IOType const* fc2_m_hidden_in,
    ScaleType const* scale_actexp_m_in, int const* actexpi_to_global_expid, IOType* reduce_m_hidden_ou, int num_act_exp,
    int num_token, int hidden_size)
{
    // * Device Property & SM
    int device_id;
    TLLM_CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    int sm_count = device_prop.multiProcessorCount;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    using ACC_TYPE = ACCESS_TYPE<IOType>;

    // * Check for launch assumption
    if (hidden_size % ACC_TYPE::ELEM_PER_ACCESS != 0)
    {
        printf("FAILED. Unable to launch as hidden_size must be multiplier of ACC_TYPE::ELEM_PER_ACCESS\n");
        return;
    }

    // * Heuristic for launch config
    // targeting low latency inference to fully utilize as much SM as possible
    int num_thread_per_token = hidden_size / ACC_TYPE::ELEM_PER_ACCESS;
    int num_warp_per_token = (num_thread_per_token + 32 - 1) / 32;
    int cluster_dim = 8;
    while (num_warp_per_token % cluster_dim != 0)
    {
        cluster_dim /= 2;
    }
    int block_dim = num_warp_per_token / cluster_dim * 32;
    int grid_dim = min(sm_count, num_token * cluster_dim) / cluster_dim * cluster_dim;

    printf(
        "* num_act_exp %d, num_token %d, hidden_size %d, num_warp_per_token %d, heuristic pick grid %d cluster %d "
        "block %d\n",
        num_act_exp, num_token, hidden_size, num_warp_per_token, grid_dim, cluster_dim, block_dim);

    // * Launch Config
    cudaLaunchConfig_t config = {0};
    cudaLaunchAttribute attribute[2];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    attribute[1].id = cudaLaunchAttributeClusterDimension;
    attribute[1].val.clusterDim.x = cluster_dim;
    attribute[1].val.clusterDim.y = 1;
    attribute[1].val.clusterDim.z = 1;
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.stream = stream;
    config.numAttrs = 2;
    config.attrs = attribute;
    config.dynamicSmemBytes = 0;

    TLLM_CUDA_CHECK(
        cudaLaunchKernelEx(&config, moe_reduction_kernel<IOType, ScaleType>, ggemm2_actexp_m_hidden_in, fc2_m_hidden_in,
            scale_actexp_m_in, actexpi_to_global_expid, reduce_m_hidden_ou, num_act_exp, num_token, hidden_size));
    TLLM_CUDA_CHECK(cudaPeekAtLastError());
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename DType>
class MoEARFuseTestRunner
{
    static_assert(std::is_same_v<DType, half> || std::is_same_v<DType, __nv_bfloat16>);
    static constexpr ncclDataType_t kNCCLDataType = std::is_same_v<DType, half> ? ncclFloat16 : ncclBfloat16;
    static constexpr nvinfer1::DataType kTRTDataType
        = std::is_same_v<DType, half> ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kBF16;

public:
    MoEARFuseTestRunner(int max_token_num, int hidden_dim, int max_expert_num)
        : m_mpi_comm(mpi::MpiComm::world())
    {
        m_message_size = max_token_num * hidden_dim;
        m_world_size = m_mpi_comm.getSize();
        m_rank = m_mpi_comm.getRank();
        TLLM_CUDA_CHECK(cudaSetDevice(m_rank));
        ncclUniqueId id;
        if (m_rank == 0)
        {
            TLLM_NCCL_CHECK(ncclGetUniqueId(&id));
        }
        m_mpi_comm.bcast(&id, sizeof(id), mpi::MpiType::kBYTE, 0);
        TLLM_NCCL_CHECK(ncclCommInitRank(&m_nccl_comm, m_world_size, id, m_rank));

        m_allreduce_in.allocate(m_message_size * sizeof(DType));
        m_residual_in.allocate(m_message_size * sizeof(DType));
        m_residual_out.allocate(m_message_size * sizeof(DType));
        m_norm_out.allocate(m_message_size * sizeof(DType));
        m_quant_out.allocate(m_message_size * sizeof(DType));
        m_scale_out.allocate(m_message_size * sizeof(DType));
        m_rms_gamma.allocate(hidden_dim * sizeof(DType));
        m_scale_factor.allocate(sizeof(float));
        m_stream = std::make_shared<tr::CudaStream>();
        m_workspace = std::make_shared<ar_fusion::Workspace>(m_rank, m_world_size, max_token_num, hidden_dim, m_stream);

        m_params.nranks = m_world_size;
        m_params.rank = m_rank;
        m_params.dtype = kTRTDataType;
        m_params.workspace = m_workspace->get_workspace();
        m_params.allreduce_in = m_allreduce_in.device_data();
        m_params.residual_in = m_residual_in.device_data();
        m_params.residual_out = m_residual_out.device_data();
        m_params.norm_out = m_norm_out.device_data();
        m_params.quant_out = m_quant_out.device_data();
        m_params.scale_out = m_scale_out.device_data();
        m_params.rms_gamma = m_rms_gamma.device_data();
        m_params.scale_factor = m_scale_factor.device_data<float>();
        m_params.rms_eps = 1e-3;
        m_params.stream = m_stream->get();

        // * moe reduction related param
        m_max_expert_num = max_expert_num;

        // [device_num_expert, m]
        m_moe_reduction_scale_input.allocate(m_max_expert_num * max_token_num * sizeof(float));
        // [device_num_expert, m, 7168]
        m_moe_reduction_active_experts_token_input.allocate(m_max_expert_num * m_message_size * sizeof(DType));
        // [m, 7168]
        m_moe_reduction_token_input.allocate(m_message_size * sizeof(DType));
        // [1]
        m_moe_reduction_device_num_experts.allocate(sizeof(int));

        m_params.moe_reduction_scale_input = reinterpret_cast<float*>(m_moe_reduction_scale_input.device_data());
        m_params.moe_reduction_active_experts_token_input = m_moe_reduction_active_experts_token_input.device_data();
        m_params.moe_reduction_token_input = m_moe_reduction_token_input.device_data();
        m_params.moe_reduction_device_num_experts
            = reinterpret_cast<int*>(m_moe_reduction_device_num_experts.device_data());
    }

    void random_input()
    {
        m_allreduce_in.random<DType>(-100.f, 100.f);
        m_residual_in.random<DType>(-100.f, 100.f);
        m_rms_gamma.random<DType>(-1.f, 1.f);
        m_scale_factor.random<float>(5.f, 5.f);

        // * moe reduction
        m_moe_reduction_scale_input.random<float>(-100.f, 100.f);
        m_moe_reduction_active_experts_token_input.random<DType>(-100.f, 100.f);
        m_moe_reduction_token_input.random<DType>(-100.f, 100.f);
    }

    template <typename Func>
    float benchmark(Func func, int warmup, int iter, int token_num, int hidden_dim, int num_active_expert = 0)
    {
        m_params.size = token_num * hidden_dim;
        m_params.hidden_dim = hidden_dim;
        cudaMemcpy(m_params.moe_reduction_device_num_experts, &num_active_expert, sizeof(int), cudaMemcpyHostToDevice);
        cudaEvent_t begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        random_input();
        m_mpi_comm.barrier();
        for (int i = 0; i < warmup; ++i)
        {
            (this->*func)(token_num, hidden_dim, num_active_expert);
        }
        cudaEventRecord(begin, m_stream->get());
        for (int i = 0; i < iter; ++i)
        {
            (this->*func)(token_num, hidden_dim, num_active_expert);
        }
        cudaEventRecord(end, m_stream->get());
        cudaEventSynchronize(end);
        float time;
        cudaEventElapsedTime(&time, begin, end);
        time /= iter;
        m_mpi_comm.barrier();
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
        return time * 1000;
    }

    int get_sm_count() const
    {
        static int sm_count = 0;
        if (sm_count == 0)
        {
            int device_id;
            TLLM_CUDA_CHECK(cudaGetDevice(&device_id));
            cudaDeviceProp device_prop;
            cudaGetDeviceProperties(&device_prop, device_id);
            sm_count = device_prop.multiProcessorCount;
        }
        return sm_count;
    }

    void verify(int token_num, int hidden_dim, int num_active_expert)
    {
        int message_size = token_num * hidden_dim;
        CudaBuffer ref_output(message_size * sizeof(DType)), ref_scale(message_size * sizeof(DType));

        // * MoE Reduction
        moe_reduction_kernel_launcher<DType, float>(m_moe_reduction_active_experts_token_input.device_data<DType>(),
            m_moe_reduction_token_input.device_data<DType>(), m_moe_reduction_scale_input.device_data<float>(),
            ref_output.device_data<DType>(), num_active_expert, token_num, hidden_dim);

        compare<DType>(
            m_rank, m_allreduce_in.host_data(), ref_output.host_data(), message_size, 1e-3, "moe reduction out");

        // * AR
        TLLM_NCCL_CHECK(ncclAllReduce(m_allreduce_in.device_data(), ref_output.device_data(), message_size,
            kNCCLDataType, ncclSum, m_nccl_comm, 0));

        // * Add
        residual_add(ref_output.device_data<DType>(), m_residual_in.device_data<DType>(), message_size, 0);

        // * Norm
        invokeGeneralRmsNorm<DType, int8_t>(ref_output.device_data<DType>(), ref_output.device_data<DType>(),
            m_rms_gamma.device_data<DType>(), nullptr, m_params.rms_eps, token_num, hidden_dim,
            tensorrt_llm::common::QuantMode(), 0);

        compare<DType>(m_rank, m_norm_out.host_data(), ref_output.host_data(), message_size, 1e-3, "norm out");

        // * Quant
        invokeFP4Quantization(token_num, hidden_dim, m_norm_out.device_data<DType>(),
            m_scale_factor.device_data<float>(), ref_output.device_data<int64_t>(), ref_scale.device_data<int32_t>(),
            false, tensorrt_llm::QuantizationSFLayout::SWIZZLED, 128, 0);
        compare<int8_t>(m_rank, m_quant_out.host_data(), ref_output.host_data(), message_size / 2, 1e-3, "quant out");
        compare<int8_t>(m_rank, m_scale_out.host_data(), ref_scale.host_data(), message_size / 16, 1e-3, "scale out");
    }

    void run_nccl_allreduce(int token_num, int hidden_dim, int)
    {
        TLLM_NCCL_CHECK(ncclAllReduce(m_allreduce_in.device_data(), m_residual_out.device_data(),
            token_num * hidden_dim, kNCCLDataType, ncclSum, m_nccl_comm, m_stream->get()));
    }

    void run_moe_reduction(int token_num, int hidden_dim, int num_active_expert)
    {
        moe_reduction_kernel_launcher<DType, float>(m_moe_reduction_active_experts_token_input.device_data<DType>(),
            m_moe_reduction_token_input.device_data<DType>(), m_moe_reduction_scale_input.device_data<float>(),
            m_allreduce_in.device_data<DType>(), num_active_expert, token_num, hidden_dim);
    }

    void run_residual_add(int token_num, int hidden_dim, int)
    {
        residual_add(m_residual_out.device_data<DType>(), // output and input
            m_residual_in.device_data<DType>(),           // input
            token_num * hidden_dim, m_stream->get());
    }

    void run_rms_norm(int token_num, int hidden_dim, int)
    {
        invokeGeneralRmsNorm<DType, int8_t>(m_residual_out.device_data<DType>(), m_norm_out.device_data<DType>(),
            m_rms_gamma.device_data<DType>(), nullptr, m_params.rms_eps, token_num, hidden_dim,
            tensorrt_llm::common::QuantMode(), m_stream->get());
    }

    void run_fp4_quant(int token_num, int hidden_dim, int)
    {
        invokeFP4Quantization(token_num,         // m
            hidden_dim,                          // n
            m_norm_out.device_data<DType>(),     // input
            m_scale_factor.device_data<float>(), // input sf
            m_quant_out.device_data<int64_t>(),  // output
            m_scale_out.device_data<int32_t>(),  // output sf
            false, tensorrt_llm::QuantizationSFLayout::SWIZZLED, 128, m_stream->get());
    }

    void run_kernel(int token_num, int hidden_dim)
    {
        ar_fusion::moe::moereduction_allreduce_fusion_op(m_params);
    }

    ~MoEARFuseTestRunner()
    {
        TLLM_NCCL_CHECK(ncclCommDestroy(m_nccl_comm));
    }

private:
    int m_rank;
    int m_world_size;
    int m_message_size;
    mpi::MpiComm const& m_mpi_comm;
    ncclComm_t m_nccl_comm;
    CudaBuffer m_allreduce_in;
    CudaBuffer m_residual_in;
    CudaBuffer m_residual_out;
    CudaBuffer m_norm_out;
    CudaBuffer m_quant_out;
    CudaBuffer m_scale_out;
    CudaBuffer m_rms_gamma;
    CudaBuffer m_scale_factor;
    std::shared_ptr<ar_fusion::Workspace> m_workspace;
    ar_fusion::moe::MoeReductionAllReduceFusionParams m_params;
    std::shared_ptr<tr::CudaStream> m_stream;

    // * moe reduction related params
    int m_max_expert_num;
    CudaBuffer m_moe_reduction_scale_input;
    CudaBuffer m_moe_reduction_active_experts_token_input;
    CudaBuffer m_moe_reduction_token_input;
    CudaBuffer m_moe_reduction_device_num_experts;
};

TEST(Kernel, MoEReduceAddARFuse)
{
    auto& comm = mpi::MpiComm::world();
    auto world_size = comm.getSize();
    auto rank = comm.getRank();
    ASSERT_EQ(world_size % 2, 0) << "Requires even world size (got " << world_size << ")";

    int warmup = 100, iter = 100;
    int hidden_dim = 7168;
    std::vector<int> candidate_token_num{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    std::vector<int> candidate_active_expert_num{8, 12, 16};
    int max_token_num = 2048;
    int max_expert_num = 16;
    MoEARFuseTestRunner<half> runner(max_token_num, hidden_dim, max_expert_num);
    for (auto token_num : candidate_token_num)
    {
        for (auto act_exp_num : candidate_active_expert_num)
        {
            auto latency = runner.benchmark(
                &MoEARFuseTestRunner<half>::run_kernel, warmup, iter, token_num, hidden_dim, act_exp_num);
            runner.verify(token_num, hidden_dim, act_exp_num);
            if (rank == 0)
            {
                TLLM_LOG_INFO("token_num %d, hidden_dim %d, act_exp_num %d, latency %fus", token_num, hidden_dim,
                    act_exp_num, latency);
            }
            auto moe_reduce_latency = runner.benchmark(
                &MoEARFuseTestRunner<half>::run_moe_reduction, warmup, iter, token_num, hidden_dim, act_exp_num);
            if (rank == 0)
            {
                TLLM_LOG_INFO("moe reduce latency %fus", moe_reduce_latency);
            }
            auto nccl_latency
                = runner.benchmark(&MoEARFuseTestRunner<half>::run_nccl_allreduce, warmup, iter, token_num, hidden_dim);
            if (rank == 0)
            {
                TLLM_LOG_INFO("nccl allreduce latency %fus", nccl_latency);
            }
            auto residual_latency
                = runner.benchmark(&MoEARFuseTestRunner<half>::run_residual_add, warmup, iter, token_num, hidden_dim);
            if (rank == 0)
            {
                TLLM_LOG_INFO("residual add latency %fus", residual_latency);
            }
            auto rms_latency
                = runner.benchmark(&MoEARFuseTestRunner<half>::run_rms_norm, warmup, iter, token_num, hidden_dim);
            if (rank == 0)
            {
                TLLM_LOG_INFO("rms norm latency %fus", rms_latency);
            }
            auto quant_latency
                = runner.benchmark(&MoEARFuseTestRunner<half>::run_fp4_quant, warmup, iter, token_num, hidden_dim);
            if (rank == 0)
            {
                TLLM_LOG_INFO("fp4 quant latency %fus", quant_latency);
                auto tot_latency = moe_reduce_latency + nccl_latency + residual_latency + rms_latency + quant_latency;
                TLLM_LOG_INFO("fusion kernel latency %fus, moe reduce + nccl + ops latency %fus, total speedup %fx",
                    latency, tot_latency, tot_latency / latency);
            }
        }
    }
}
