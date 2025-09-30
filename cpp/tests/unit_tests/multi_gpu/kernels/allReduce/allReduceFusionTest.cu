/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdarg>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.h"
#include "tensorrt_llm/kernels/communicationKernels/allReduceWorkspace.h"
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
    residual_add_kernel<<<(size + 127) / 128, 128, 0, stream>>>(data, residual, size);
}

template <typename DType>
__global__ void quantize_to_fp8_kernel(DType* data, __nv_fp8_e4m3* data_fp8, int size, float* scale_factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    data_fp8[idx] = static_cast<__nv_fp8_e4m3>(static_cast<float>(data[idx]) * (1.f / *scale_factor));
}

template <typename DType>
void quantize_to_fp8(DType* data, __nv_fp8_e4m3* data_fp8, int size, float* scale_factor, cudaStream_t stream)
{
    quantize_to_fp8_kernel<<<(size + 127) / 128, 128, 0, stream>>>(data, data_fp8, size, scale_factor);
}

template <typename T>
bool compare(int rank, void* p_real, void* p_ref, int size, std::string const& cmp_info = "", float atol = 1e-3)
{
    auto ptr_real = reinterpret_cast<T*>(p_real);
    auto ptr_ref = reinterpret_cast<T*>(p_ref);
    float max_diff = 0.f, tot_diff = 0.f;
    int error_cnt = 0;
    float max_error_value_real = 0.f, max_error_value_ref = 0.f;
    static char* ar_debug = std::getenv("AR_DEBUG");
    if (ar_debug && rank == 0)
    {
        printf("TensorReal: [");
        for (int n = 0; n < 20; ++n)
        {
            float v = static_cast<float>(ptr_real[n]);
            printf("%f, ", v);
        }
        printf("...]\n");
        printf("TensorRef: [");
        for (int n = 0; n < 20; ++n)
        {
            float v = static_cast<float>(ptr_ref[n]);
            printf("%f, ", v);
        }
        printf("...]\n");
    }
    int print_cnt = 0;
    for (int n = 0; n < size; ++n)
    {
        float v_real = static_cast<float>(ptr_real[n]);
        float v_ref = static_cast<float>(ptr_ref[n]);
        float diff = std::abs(v_real - v_ref);

        if (diff > max_diff)
        {
            max_diff = diff;
            max_error_value_real = v_real;
            max_error_value_ref = v_ref;
        }

        bool is_error = diff > atol;
        if (diff > atol)
        {
            tot_diff += diff;
            ++error_cnt;
        }
        if (ar_debug && is_error && rank == 0 && print_cnt < 20)
        {
            ++print_cnt;
            if (rank == 0)
                printf("idx %d, v_real %f, v_ref %f\n", n, v_real, v_ref);
        }
    }
    bool pass = error_cnt == 0;
    if (!pass && rank == 0)
    {
        printf(
            "[%s] rank %d, atol %8.4f, max absolute diff %8.4f(%8.4f vs %8.4f), avg absolute diff %8.4f, absolute "
            "error count %d/%d\n",
            cmp_info.c_str(), rank, atol, max_diff, max_error_value_real, max_error_value_ref,
            tot_diff / std::max(error_cnt, 1), error_cnt, size);
    }
    return pass;
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

int get_random_int(int min_v, int max_v)
{
    static int rseed = 20250227;
    std::mt19937 gen(rseed++);
    std::uniform_int_distribution<> dis(min_v, max_v);

    return dis(gen);
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
        clear();
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

    void clear()
    {
        TLLM_CUDA_CHECK(cudaMemset(m_d_data, 0, m_size));
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

template <typename DType>
struct DTypeTraits;

template <>
struct DTypeTraits<half>
{
    static constexpr ncclDataType_t kNCCLDataType = ncclFloat16;
    static constexpr nvinfer1::DataType kTRTDataType = nvinfer1::DataType::kHALF;
};

template <>
struct DTypeTraits<__nv_bfloat16>
{
    static constexpr ncclDataType_t kNCCLDataType = ncclBfloat16;
    static constexpr nvinfer1::DataType kTRTDataType = nvinfer1::DataType::kBF16;
};

template <>
struct DTypeTraits<float>
{
    static constexpr ncclDataType_t kNCCLDataType = ncclFloat32;
    static constexpr nvinfer1::DataType kTRTDataType = nvinfer1::DataType::kFLOAT;
};

template <typename DType, ar_fusion::AllReduceFusionPattern Pattern>
class TestRunner
{
    static constexpr ncclDataType_t kNCCLDataType = DTypeTraits<DType>::kNCCLDataType;
    static constexpr nvinfer1::DataType kTRTDataType = DTypeTraits<DType>::kTRTDataType;
    static constexpr bool kFP4QuantOutSupport = !std::is_same_v<DType, float>;
    static_assert(kFP4QuantOutSupport || Pattern != ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant,
        "kARResidualRMSNormFP4Quant is not supported for float dtype");

public:
    TestRunner(int max_token_num, int hidden_dim)
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
        m_allreduce_out.allocate(m_message_size * sizeof(DType));
        m_residual_out.allocate(m_message_size * sizeof(DType));
        m_norm_out.allocate(m_message_size * sizeof(DType));
        m_quant_out.allocate(m_message_size * sizeof(DType));
        // SF layout was packed to [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
        size_t scale_out_size = ((max_token_num + 127) / 128 * 128) * ((hidden_dim + 63) / 64 * 4);
        m_scale_out.allocate(scale_out_size);
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
        m_params.allreduce_out = m_allreduce_out.device_data();
        m_params.residual_out = m_residual_out.device_data();
        m_params.norm_out = m_norm_out.device_data();
        m_params.quant_out = m_quant_out.device_data();
        m_params.scale_out = m_scale_out.device_data();
        m_params.rms_gamma = m_rms_gamma.device_data();
        m_params.scale_factor = m_scale_factor.device_data<float>();
        m_params.rms_eps = 1e-3;
        m_params.stream = m_stream->get();
        m_params.pattern = Pattern;
    }

    void reset_io()
    {
        m_allreduce_in.random<DType>(-100.f, 100.f);
        m_residual_in.random<DType>(-100.f, 100.f);
        m_rms_gamma.random<DType>(-1.f, 1.f);
        m_scale_factor.random<float>(1.f, 1.f);
        // Because scale_out internally performs layout interleaving, not all elements will be covered, so it should be
        // reset before calling the kernel to ensure correct comparison results
        if (kFP4QuantOutSupport)
        {
            m_scale_out.clear();
        }
    }

    template <typename Func>
    float benchmark(Func func, int warmup, int iter, int token_num, int hidden_dim)
    {
        m_params.size = token_num * hidden_dim;
        m_params.hidden_dim = hidden_dim;
        cudaEvent_t begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        m_mpi_comm.barrier();
        for (int i = 0; i < warmup; ++i)
        {
            (this->*func)(token_num, hidden_dim);
        }
        cudaEventRecord(begin, m_stream->get());
        for (int i = 0; i < iter; ++i)
        {
            (this->*func)(token_num, hidden_dim);
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

    template <typename Func>
    void run_once(Func func, int token_num, int hidden_dim)
    {
        benchmark(func, 0, 1, token_num, hidden_dim);
    }

    int get_sm_count()
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

    void verify(int token_num, int hidden_dim)
    {
        int message_size = token_num * hidden_dim;
        CudaBuffer ref_output(message_size * sizeof(DType));

        // We directly compare the results of AR+AddResidual here, as the accumulation order in NCCL's AR might be
        // inconsistent across different kernels. Therefore, we set atol to 1 (setting it to 0 locally also passes the
        // test).
        TLLM_NCCL_CHECK(ncclAllReduce(m_allreduce_in.device_data(), ref_output.device_data(), message_size,
            kNCCLDataType, ncclSum, m_nccl_comm, 0));
        if constexpr (ar_fusion::HasAllReduceOut<Pattern>)
        {
            TLLM_CHECK(compare<DType>(
                m_rank, m_allreduce_out.host_data(), ref_output.host_data(), message_size, "allreduce out", 1));
        }
        if constexpr (ar_fusion::HasResidual<Pattern>)
        {
            residual_add(ref_output.device_data<DType>(), m_residual_in.device_data<DType>(), message_size, 0);
            if constexpr (ar_fusion::HasResidualOut<Pattern>)
            {
                TLLM_CHECK(compare<DType>(
                    m_rank, m_residual_out.host_data(), ref_output.host_data(), message_size, "residual out", 1));
            }
        }
        if constexpr (ar_fusion::HasRMSNorm<Pattern>)
        {
            // This excludes the accumulation order errors introduced by AR and only compares the accuracy of the
            // RMSNorm. The atol is set to 1e-2 to exclude errors caused by accumulation order changes due to
            // differences in cluster/block size.
            invokeGeneralRmsNorm<DType, int8_t>(ref_output.device_data<DType>(), m_residual_out.device_data<DType>(),
                m_rms_gamma.device_data<DType>(), nullptr, m_params.rms_eps, token_num, hidden_dim,
                tensorrt_llm::common::QuantMode(), 0);
            if constexpr (ar_fusion::HasNormOut<Pattern>)
            {
                TLLM_CHECK(compare<DType>(
                    m_rank, m_norm_out.host_data(), ref_output.host_data(), message_size, "norm out", 1e-2));
            }
        }
        if constexpr (ar_fusion::GetQuantType<Pattern> == ar_fusion::QuantType::kFP4)
        {
            // We need norm out to verify the accuracy of quantization.
            static_assert(Pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant);
            // SF layout was packed to [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
            size_t scale_out_size = ((token_num + 127) / 128 * 128) * ((hidden_dim + 63) / 64 * 4);
            CudaBuffer ref_scale(scale_out_size);
            // Here, we also only compare the accuracy of quantization. Since there are no differences in
            // computation order, atol is set to 0.
            invokeFP4Quantization(1, token_num, hidden_dim, m_norm_out.device_data<DType>(),
                m_scale_factor.device_data<float>(), ref_output.device_data<int64_t>(),
                ref_scale.device_data<int32_t>(), false, tensorrt_llm::QuantizationSFLayout::SWIZZLED, 128, 0);
            TLLM_CHECK(compare<int8_t>(
                m_rank, m_quant_out.host_data(), ref_output.host_data(), message_size / 2, "fp4 quant out", 0));
            TLLM_CHECK(compare<int8_t>(
                m_rank, m_scale_out.host_data(), ref_scale.host_data(), scale_out_size, "fp4 scale out", 0));
        }
        else if constexpr (ar_fusion::GetQuantType<Pattern> == ar_fusion::QuantType::kFP8)
        {
            // We need norm out to verify the accuracy of quantization.
            static_assert(Pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant);
            CudaBuffer ref_fp8_output(message_size * sizeof(__nv_fp8_e4m3));
            quantize_to_fp8(m_norm_out.device_data<DType>(), ref_fp8_output.device_data<__nv_fp8_e4m3>(), message_size,
                m_scale_factor.device_data<float>(), m_stream->get());
            TLLM_CHECK(compare<__nv_fp8_e4m3>(
                m_rank, m_quant_out.host_data(), ref_fp8_output.host_data(), message_size, "fp8 quant out", 0));
        }
    }

    void run_nccl_allreduce(int token_num, int hidden_dim)
    {
        TLLM_NCCL_CHECK(ncclAllReduce(m_allreduce_in.device_data(), m_residual_out.device_data(),
            token_num * hidden_dim, kNCCLDataType, ncclSum, m_nccl_comm, m_stream->get()));
    }

    void run_residual_add(int token_num, int hidden_dim)
    {
        residual_add(m_residual_out.device_data<DType>(), m_residual_in.device_data<DType>(), token_num * hidden_dim,
            m_stream->get());
    }

    void run_rms_norm(int token_num, int hidden_dim)
    {
        invokeGeneralRmsNorm<DType, int8_t>(m_norm_out.device_data<DType>(), m_residual_out.device_data<DType>(),
            m_rms_gamma.device_data<DType>(), nullptr, m_params.rms_eps, token_num, hidden_dim,
            tensorrt_llm::common::QuantMode(), m_stream->get());
    }

    void run_fp4_quant(int token_num, int hidden_dim)
    {
        invokeFP4Quantization(1, token_num, hidden_dim, m_norm_out.device_data<DType>(),
            m_scale_factor.device_data<float>(), m_quant_out.device_data<int64_t>(), m_scale_out.device_data<int32_t>(),
            false, tensorrt_llm::QuantizationSFLayout::SWIZZLED, 128, m_stream->get());
    }

    void run_kernel(int token_num, int hidden_dim)
    {
        ar_fusion::allreduce_fusion_op(m_params);
    }

    ~TestRunner()
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
    CudaBuffer m_allreduce_out;
    CudaBuffer m_residual_out;
    CudaBuffer m_norm_out;
    CudaBuffer m_quant_out;
    CudaBuffer m_scale_out;
    CudaBuffer m_rms_gamma;
    CudaBuffer m_scale_factor;
    std::shared_ptr<ar_fusion::Workspace> m_workspace;
    ar_fusion::AllReduceFusionParams m_params;
    std::shared_ptr<tr::CudaStream> m_stream;
};

TEST(Kernel_AllReduceFusion, AllReduceAccuracyRandomTokenNum)
{
    using Runner = TestRunner<half, ar_fusion::AllReduceFusionPattern::kAllReduce>;
    auto& comm = mpi::MpiComm::world();
    auto world_size = comm.getSize();
    auto rank = comm.getRank();
    ASSERT_EQ(world_size % 2, 0) << "Requires even world size (got " << world_size << ")";

    int iter = 100;
    std::vector<int> candidate_hidden_dim{1024, 2048, 4096, 7168, 8192};
    int min_token_num = 1;
    int max_token_num = 2048;
    for (auto hidden_dim : candidate_hidden_dim)
    {
        Runner runner(max_token_num, hidden_dim);
        for (int i = 0; i < iter; ++i)
        {
            int token_num = get_random_int(min_token_num, max_token_num);
            if (rank == 0)
            {
                printf("[Verify] token_num %-4d, hidden_dim %-4d ...", token_num, hidden_dim);
            }
            runner.reset_io();
            runner.run_once(&Runner::run_kernel, token_num, hidden_dim);
            runner.verify(token_num, hidden_dim);
            if (rank == 0)
            {
                printf("\033[32mPass!\033[0m\n");
            }
        }
    }
}

TEST(Kernel_AllReduceFusion, AllReduceAccuracyFixedTokenNum)
{
    using Runner = TestRunner<half, ar_fusion::AllReduceFusionPattern::kAllReduce>;
    auto& comm = mpi::MpiComm::world();
    auto world_size = comm.getSize();
    auto rank = comm.getRank();
    ASSERT_EQ(world_size % 2, 0) << "Requires even world size (got " << world_size << ")";

    int iter = 10;
    std::vector<int> candidate_hidden_dim{1024, 2048, 4096, 7168, 8192};
    int min_token_num = 1;
    int max_token_num = 2048;
    for (auto hidden_dim : candidate_hidden_dim)
    {
        Runner runner(max_token_num, hidden_dim);
        for (int token_num = min_token_num; token_num <= max_token_num; token_num *= 2)
        {
            if (rank == 0)
            {
                printf("[Verify] token_num %-4d, hidden_dim %-4d ...", token_num, hidden_dim);
            }
            for (int i = 0; i < iter; ++i)
            {
                runner.reset_io();
                runner.run_once(&Runner::run_kernel, token_num, hidden_dim);
                runner.verify(token_num, hidden_dim);
            }
            if (rank == 0)
            {
                printf("\033[32mPass!\033[0m\n");
            }
        }
    }
}

TEST(Kernel_AllReduceFusion, AllReduceFusionAccuracyDifferentHiddenDim)
{
#define TEST_AR_FUSION(DType, FusionPattern)                                                                           \
    {                                                                                                                  \
        using Runner = TestRunner<DType, FusionPattern>;                                                               \
        int iter = 10;                                                                                                 \
        std::vector<int> candidate_hidden_dim{64, 128, 256, 384, 512, 640, 768, 896};                                  \
        int min_token_num = 1;                                                                                         \
        int max_token_num = 2048;                                                                                      \
        for (auto hidden_dim : candidate_hidden_dim)                                                                   \
        {                                                                                                              \
            Runner runner(max_token_num, hidden_dim);                                                                  \
            for (int token_num = min_token_num; token_num <= max_token_num; token_num *= 2)                            \
            {                                                                                                          \
                if (rank == 0)                                                                                         \
                {                                                                                                      \
                    printf("[Verify] token_num %-4d, hidden_dim %-4d ...", token_num, hidden_dim);                     \
                }                                                                                                      \
                for (int i = 0; i < iter; ++i)                                                                         \
                {                                                                                                      \
                    runner.reset_io();                                                                                 \
                    runner.run_once(&Runner::run_kernel, token_num, hidden_dim);                                       \
                    runner.verify(token_num, hidden_dim);                                                              \
                }                                                                                                      \
                if (rank == 0)                                                                                         \
                {                                                                                                      \
                    printf("\033[32mPass!\033[0m\n");                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }
    auto& comm = mpi::MpiComm::world();
    auto world_size = comm.getSize();
    auto rank = comm.getRank();
    ASSERT_EQ(world_size % 2, 0) << "Requires even world size (got " << world_size << ")";

    int const arch = tensorrt_llm::common::getSMVersion();
    if (arch >= 100)
    {
        TEST_AR_FUSION(half, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant);
    }
    else
    {
        TEST_AR_FUSION(half, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant);
    }
#undef TEST_AR_FUSION
}

TEST(Kernel_AllReduceFusion, AllReduceFusionAccuracyDifferentDType)
{
#define TEST_AR_FUSION(DType, FusionPattern)                                                                           \
    {                                                                                                                  \
        using Runner = TestRunner<DType, FusionPattern>;                                                               \
        Runner runner(max_token_num, hidden_dim);                                                                      \
        for (int token_num = min_token_num; token_num <= max_token_num; token_num *= 2)                                \
        {                                                                                                              \
            if (rank == 0)                                                                                             \
            {                                                                                                          \
                printf("[Verify] pattern %-20s, dtype %-10s, token_num %-4d, hidden_dim %-4d ...", #FusionPattern,     \
                    #DType, token_num, hidden_dim);                                                                    \
            }                                                                                                          \
            runner.reset_io();                                                                                         \
            runner.run_once(&Runner::run_kernel, token_num, hidden_dim);                                               \
            runner.verify(token_num, hidden_dim);                                                                      \
            if (rank == 0)                                                                                             \
            {                                                                                                          \
                printf("\033[32mPass!\033[0m\n");                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

    int const arch = tensorrt_llm::common::getSMVersion();
    auto& comm = mpi::MpiComm::world();
    auto world_size = comm.getSize();
    auto rank = comm.getRank();
    ASSERT_EQ(world_size % 2, 0) << "Requires even world size (got " << world_size << ")";

    std::vector<int> candidate_hidden_dim{1024, 2048, 4096, 7168, 8192};
    int min_token_num = 1;
    int max_token_num = 2048;
    for (auto hidden_dim : candidate_hidden_dim)
    {
        TEST_AR_FUSION(half, ar_fusion::AllReduceFusionPattern::kAllReduce);
        TEST_AR_FUSION(half, ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm);
        TEST_AR_FUSION(half, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant);
        if (arch >= 100)
        {
            TEST_AR_FUSION(half, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant);
        }
        TEST_AR_FUSION(float, ar_fusion::AllReduceFusionPattern::kAllReduce);
        TEST_AR_FUSION(float, ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm);
        TEST_AR_FUSION(float, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant);
#if defined(ENABLE_BF16)
        TEST_AR_FUSION(__nv_bfloat16, ar_fusion::AllReduceFusionPattern::kAllReduce);
        TEST_AR_FUSION(__nv_bfloat16, ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm);
        TEST_AR_FUSION(__nv_bfloat16, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant);
        if (arch >= 100)
        {
            TEST_AR_FUSION(__nv_bfloat16, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant);
        }
#endif
    }
#undef TEST_AR_FUSION
}

TEST(Kernel_AllReduceFusion, Perf)
{
    int const arch = tensorrt_llm::common::getSMVersion();
    if (arch < 100)
    {
        GTEST_SKIP() << "Skipping test for SM < 100";
    }

    using Runner = TestRunner<half, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant>;
    auto& comm = mpi::MpiComm::world();
    auto world_size = comm.getSize();
    auto rank = comm.getRank();
    ASSERT_EQ(world_size % 2, 0) << "Requires even world size (got " << world_size << ")";

    int warmup = 100, iter = 300;
    int hidden_dim = 7168;
    std::vector<int> candidate_token_num{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    int max_token_num = 2048;
    Runner runner(max_token_num, hidden_dim);
    for (auto token_num : candidate_token_num)
    {
        auto latency = runner.benchmark(&Runner::run_kernel, warmup, iter, token_num, hidden_dim);
        if (rank == 0)
        {
            TLLM_LOG_INFO(
                "token_num %-4d, hidden_dim %-4d, fusion kernel latency %4.4fus", token_num, hidden_dim, latency);
        }
        auto nccl_latency = runner.benchmark(&Runner::run_nccl_allreduce, warmup, iter, token_num, hidden_dim);
        if (rank == 0)
        {
            TLLM_LOG_INFO("nccl allreduce latency %4.4fus", nccl_latency);
        }
        auto residual_latency = runner.benchmark(&Runner::run_residual_add, warmup, iter, token_num, hidden_dim);
        if (rank == 0)
        {
            TLLM_LOG_INFO("residual add latency %4.4fus", residual_latency);
        }
        auto rms_latency = runner.benchmark(&Runner::run_rms_norm, warmup, iter, token_num, hidden_dim);
        if (rank == 0)
        {
            TLLM_LOG_INFO("rms norm latency %4.4fus", rms_latency);
        }
        auto quant_latency = runner.benchmark(&Runner::run_fp4_quant, warmup, iter, token_num, hidden_dim);
        if (rank == 0)
        {
            TLLM_LOG_INFO("fp4 quant latency %4.4fus", quant_latency);
            auto tot_latency = nccl_latency + residual_latency + rms_latency + quant_latency;
            TLLM_LOG_INFO("fusion kernel latency %4.4fus, nccl + ops latency %4.4fus, total speedup %2.4fx", latency,
                tot_latency, tot_latency / latency);
        }
    }
}
