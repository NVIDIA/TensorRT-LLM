#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cutlass/numeric_types.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/enabled.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelLauncher.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <type_traits>
#include <vector>

using tensorrt_llm::kernels::WeightOnlyParams;
using tensorrt_llm::kernels::WeightOnlyType;
using tensorrt_llm::kernels::WeightOnlyQuantType;
using tensorrt_llm::kernels::WeightOnlyActivationType;
using tensorrt_llm::kernels::WeightOnlyActivationFunctionType;
template <WeightOnlyActivationType T>
struct AType;

template <>
struct AType<WeightOnlyActivationType::FP16>
{
    using CudaKernelAType = half;
    using CutlassKernelAType = half;
};
#if defined(ENABLE_BF16)
template <>
struct AType<WeightOnlyActivationType::BF16>
{
    using CudaKernelAType = __nv_bfloat16;
    using CutlassKernelAType = __nv_bfloat16;
};
#endif
template <WeightOnlyQuantType T>
struct BType;

template <>
struct BType<WeightOnlyQuantType::Int4b>
{
    using CudaKernelBType = uint8_t;
    using CutlassKernelBType = cutlass::uint4b_t;
    static constexpr int elemsPerByte = 2;
};

template <>
struct BType<WeightOnlyQuantType::Int8b>
{
    using CudaKernelBType = uint8_t;
    using CutlassKernelBType = uint8_t;
    static constexpr int elemsPerByte = 1;
};
struct CutlassKernel;
struct CudaKernel;

void simple_assert(bool flag)
{
    if (!flag)
    {
        throw std::runtime_error("assert failed");
    }
}

template <typename T>
std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> get_configs(T& runner, int k)
{
    auto configs = runner.getConfigs();
    std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> rets;
    for (auto config : configs)
    {
        if (config.stages >= 5)
        {
            continue;
        }
        if (config.split_k_style != tensorrt_llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K)
        {
            int k_size = (k + config.split_k_factor - 1) / config.split_k_factor;
            if (k_size % 64)
            {
                continue;
            }
        }
        rets.push_back(config);
    }
    return rets;
}

template <typename KernelFlag, WeightOnlyActivationType AFlag, WeightOnlyQuantType BFlag>
float benchmark_perchannel(void* act, void* weight, void* scales, void* zeros, void* bias, void* out, int m, int n,
    int k, int group_size, int warmup, int iter)
{
    simple_assert(zeros == nullptr && bias == nullptr && group_size == 0);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    if constexpr (std::is_same_v<KernelFlag, CudaKernel>)
    {
        WeightOnlyParams params{reinterpret_cast<uint8_t*>(weight), scales, zeros, act, nullptr, bias, out, m, n, k,
            group_size, BFlag, WeightOnlyType::PerChannel, WeightOnlyActivationFunctionType::Identity, AFlag};
        for (int i = 0; i < warmup; ++i)
        {
            tensorrt_llm::kernels::weight_only_batched_gemv_launcher(params, s);
        }
        cudaEventRecord(begin, s);
        for (int i = 0; i < iter; ++i)
        {
            tensorrt_llm::kernels::weight_only_batched_gemv_launcher(params, s);
        }
    }
    else if (std::is_same_v<KernelFlag, CutlassKernel>)
    {
        tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<typename AType<AFlag>::CutlassKernelAType,
            typename BType<BFlag>::CutlassKernelBType, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>
            gemm;
        auto configs = get_configs(gemm, k);
        int ws_bytes = gemm.getWorkspaceSize(m, n, k);
        char* ws_ptr = nullptr;
        if (ws_bytes)
            cudaMalloc(&ws_ptr, ws_bytes);
        float fast_time = 1e8;
        auto best_config = configs[0];
        for (auto& config : configs)
        {
            for (int i = 0; i < 2; ++i)
            {
                gemm.gemm(act, weight, scales, out, m, n, k, config, ws_ptr, ws_bytes, s);
            }
            cudaEventRecord(begin, s);
            for (int i = 0; i < 5; ++i)
            {
                gemm.gemm(act, weight, scales, out, m, n, k, config, ws_ptr, ws_bytes, s);
            }
            cudaEventRecord(end, s);
            cudaEventSynchronize(end);
            float time;
            cudaEventElapsedTime(&time, begin, end);
            if (time < fast_time)
            {
                fast_time = time;
                best_config = config;
            }
        }

        for (int i = 0; i < warmup; ++i)
        {
            gemm.gemm(act, weight, scales, out, m, n, k, best_config, ws_ptr, ws_bytes, s);
        }
        cudaEventRecord(begin, s);
        for (int i = 0; i < iter; ++i)
        {
            gemm.gemm(act, weight, scales, out, m, n, k, best_config, ws_ptr, ws_bytes, s);
        }
        if (ws_ptr)
            cudaFree(ws_ptr);
    }

    cudaEventRecord(end, s);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(s);
    return time / iter;
}

template <typename KernelFlag, WeightOnlyActivationType AFlag, WeightOnlyQuantType BFlag>
float benchmark_groupwise(void* act, void* weight, void* scales, void* zeros, void* bias, void* out, int m, int n,
    int k, int group_size, int warmup, int iter)
{
    simple_assert(zeros && bias && (group_size == 64 || group_size == 128));
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    if constexpr (std::is_same_v<KernelFlag, CudaKernel>)
    {
        WeightOnlyParams params{reinterpret_cast<uint8_t*>(weight), scales, zeros, act, nullptr, bias, out, m, n, k,
            group_size, BFlag, WeightOnlyType::GroupWise, WeightOnlyActivationFunctionType::Identity, AFlag};
        for (int i = 0; i < warmup; ++i)
        {
            tensorrt_llm::kernels::weight_only_batched_gemv_launcher(params, s);
        }
        cudaEventRecord(begin, s);
        for (int i = 0; i < iter; ++i)
        {
            tensorrt_llm::kernels::weight_only_batched_gemv_launcher(params, s);
        }
    }
    else if (std::is_same_v<KernelFlag, CutlassKernel>)
    {
        tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<typename AType<AFlag>::CutlassKernelAType,
            typename BType<BFlag>::CutlassKernelBType, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>
            gemm;
        auto configs = get_configs(gemm, k);
        int ws_bytes = gemm.getWorkspaceSize(m, n, k);
        char* ws_ptr = nullptr;
        if (ws_bytes)
            cudaMalloc(&ws_ptr, ws_bytes);
        float fast_time = 1e8;
        auto best_config = configs[0];
        for (auto& config : configs)
        {
            for (int i = 0; i < 2; ++i)
            {
                gemm.gemm(act, weight, scales, zeros, bias, out, m, n, k, group_size, config, ws_ptr, ws_bytes, s);
            }
            cudaEventRecord(begin, s);
            for (int i = 0; i < 5; ++i)
            {
                gemm.gemm(act, weight, scales, zeros, bias, out, m, n, k, group_size, config, ws_ptr, ws_bytes, s);
            }
            cudaEventRecord(end, s);
            cudaEventSynchronize(end);
            float time;
            cudaEventElapsedTime(&time, begin, end);
            if (time < fast_time)
            {
                fast_time = time;
                best_config = config;
            }
        }

        for (int i = 0; i < warmup; ++i)
        {
            gemm.gemm(act, weight, scales, zeros, bias, out, m, n, k, group_size, best_config, ws_ptr, ws_bytes, s);
        }
        cudaEventRecord(begin, s);
        for (int i = 0; i < iter; ++i)
        {
            gemm.gemm(act, weight, scales, zeros, bias, out, m, n, k, group_size, best_config, ws_ptr, ws_bytes, s);
        }
        if (ws_ptr)
            cudaFree(ws_ptr);
    }

    cudaEventRecord(end, s);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(s);
    return time / iter;
}

struct CudaBuffer
{
    void* _data;
    int _size;

    CudaBuffer(int size_in_bytes)
        : _size(size_in_bytes)
    {
        cudaMalloc(&_data, _size);
    }

    template <typename T = void>
    T* data()
    {
        return reinterpret_cast<T*>(_data);
    }

    void copy_to(void* dst)
    {
        cudaMemcpy(dst, _data, _size, cudaMemcpyDeviceToHost);
    }

    void copy_from(void* src)
    {
        cudaMemcpy(_data, src, _size, cudaMemcpyHostToDevice);
    }

    ~CudaBuffer()
    {
        cudaFree(_data);
    }
};

template <typename T>
float compare(void* _pa, void* _pb, int size, float scale)
{
    auto pa = reinterpret_cast<T*>(_pa);
    auto pb = reinterpret_cast<T*>(_pb);
    float max_diff = 0.f, tot_diff = 0.f;
    float max_val = 0.f;
    int diff_cnt = 0;
    float threshold = 1e-7;
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
    }
    float diff_thres = max_val * scale;
#if defined(ENABLE_BF16)
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        // bfloat16 has fewer mantissa digits than float16(10 bits for fp16 but only 7 bits for bf16), so the cumulative
        // error will be larger.
        diff_thres *= 3.f;
    }
    else
#endif
    {
        diff_thres *= 1.5f;
    }
    printf("max diff %f (diff threshold %f), avg diff %f, diff cnt %d/%d\n", max_diff, diff_thres, tot_diff / diff_cnt,
        diff_cnt, size);
    return max_diff <= diff_thres;
}

template <typename T1, typename T2>
void random_fill(std::vector<T1>& vec, T2 minv, T2 maxv)
{
    std::mt19937 gen(20231205);
    std::uniform_real_distribution<float> dis(static_cast<float>(minv), static_cast<float>(maxv));
    for (auto& v : vec)
    {
        v = static_cast<T1>(dis(gen));
    }
}

template <WeightOnlyActivationType AFlag, WeightOnlyQuantType BFlag>
bool benchmark(int m, int n, int k, int group_size, int warmup, int iter)
{
    printf("benchmark mnk (%d, %d, %d) ", m, n, k);
    if (AFlag == WeightOnlyActivationType::FP16)
    {
        printf("FP16 Activation ");
    }
    else
    {
        printf("BF16 Activation ");
    }
    if (BFlag == WeightOnlyQuantType::Int8b)
    {
        printf("Int8b ");
    }
    else
    {
        printf("Int4b ");
    }
    if (group_size == 0)
    {
        printf("PerChannel Weight Only\n");
    }
    else
    {
        printf("GroupWise%d Weight Only\n", group_size);
    }
    using AT = typename AType<AFlag>::CudaKernelAType;
    using BT = typename BType<BFlag>::CudaKernelBType;
    constexpr int elem_per_byte = BType<BFlag>::elemsPerByte;
    CudaBuffer d_act(m * k * sizeof(AT));
    CudaBuffer d_weight(k * n * sizeof(uint8_t) / elem_per_byte);
    CudaBuffer d_scales(n * k * sizeof(AT));
    CudaBuffer d_zeros(n * k * sizeof(AT));
    CudaBuffer d_bias(n * sizeof(AT));
    CudaBuffer d_out(m * n * sizeof(AT));
    std::vector<AT> h_act(m * k);
    std::vector<uint8_t> h_weight(k * n);
    std::vector<AT> h_scales(n * k), h_zeros(n * k), h_bias(n);
    std::vector<AT> h_out1(m * n), h_out2(m * n);

    random_fill(h_act, -1.f, 1.f);
    random_fill(h_scales, -1.f, 1.f);

    for (uint8_t& v : h_weight)
    {
        v = rand() % 256;
    }

    d_act.copy_from(h_act.data());
    d_weight.copy_from(h_weight.data());
    d_scales.copy_from(h_scales.data());
    d_zeros.copy_from(h_zeros.data());
    d_bias.copy_from(h_bias.data());

    void* p_zeros = nullptr;
    void* p_bias = nullptr;
    if (group_size == 64 || group_size == 128)
    {
        p_zeros = d_zeros.data();
        p_bias = d_bias.data();
    }

    float time1, time2;
    std::function<decltype(benchmark_perchannel<CudaKernel, AFlag, BFlag>)> benchmark_func_cuda
        = benchmark_perchannel<CudaKernel, AFlag, BFlag>;
    std::function<decltype(benchmark_perchannel<CutlassKernel, AFlag, BFlag>)> benchmark_func_cutlass
        = benchmark_perchannel<CutlassKernel, AFlag, BFlag>;
    if (group_size != 0)
    {
        benchmark_func_cuda = benchmark_groupwise<CudaKernel, AFlag, BFlag>;
        benchmark_func_cutlass = benchmark_groupwise<CutlassKernel, AFlag, BFlag>;
    }
    time1 = benchmark_func_cuda(d_act.data(), d_weight.data(), d_scales.data(), p_zeros, p_bias, d_out.data(), m, n, k,
        group_size, warmup, iter);
    d_out.copy_to(h_out1.data());
    time2 = benchmark_func_cutlass(d_act.data(), d_weight.data(), d_scales.data(), p_zeros, p_bias, d_out.data(), m, n,
        k, group_size, warmup, iter);
    d_out.copy_to(h_out2.data());
    float quant_scale = 1.f / (1 << (8 / elem_per_byte - 1));
    bool pass = compare<AT>(h_out1.data(), h_out2.data(), m * n, quant_scale);
    printf(
        "cuda kernel cost time %.6f, cutlass kernel cost time %.6f, cuda speedup %.3f\n", time1, time2, time2 / time1);
    return pass;
}

TEST(Kernel, WeightOnly)
{
    bool pass;
    int warmup = 10, iter = 30;
    std::vector<int> ms{1, 2, 4};
    std::vector<int> ns{512, 1024, 2048, 4096};
    std::vector<int> ks{512, 1024, 2048, 4096};
    std::vector<int> gss{0, 64, 128};
    for (auto m : ms)
    {
        for (auto n : ns)
        {
            for (auto k : ks)
            {
                for (auto gs : gss)
                {
                    pass = benchmark<WeightOnlyActivationType::FP16, WeightOnlyQuantType::Int8b>(
                        m, n, k, gs, warmup, iter);
                    EXPECT_TRUE(pass);
                    pass = benchmark<WeightOnlyActivationType::FP16, WeightOnlyQuantType::Int4b>(
                        m, n, k, gs, warmup, iter);
                    EXPECT_TRUE(pass);
#if defined(ENABLE_BF16)
                    pass = benchmark<WeightOnlyActivationType::BF16, WeightOnlyQuantType::Int8b>(
                        m, n, k, gs, warmup, iter);
                    EXPECT_TRUE(pass);
                    pass = benchmark<WeightOnlyActivationType::BF16, WeightOnlyQuantType::Int4b>(
                        m, n, k, gs, warmup, iter);
                    EXPECT_TRUE(pass);
#endif
                }
            }
        }
    }
}
