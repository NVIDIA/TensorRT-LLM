#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cutlass/numeric_types.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/int8SQ.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
using namespace tensorrt_llm::kernels::smooth_quant;

void simple_assert(bool flag)
{
    if (!flag)
    {
        throw std::runtime_error("assert failed");
    }
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
    printf("max diff %f (diff threshold %f), avg diff %f, diff cnt %d/%d\n", max_diff, diff_thres,
        tot_diff / std::max(1, diff_cnt), diff_cnt, size);
    return max_diff <= diff_thres;
}

template <typename T1, typename T2>
void random_fill(std::vector<T1>& vec, T2 minv, T2 maxv)
{
    std::mt19937 gen(rand());
    std::uniform_real_distribution<float> dis(static_cast<float>(minv), static_cast<float>(maxv));
    for (auto& v : vec)
    {
        v = static_cast<T1>(dis(gen));
    }
}

template <typename T>
std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> get_configs(T& runner, int k)
{
    auto configs = runner.getConfigs();
    return configs;
}

template <typename T>
float run_cuda_kernel(Params& params, int warmup, int iter)
{
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (int i = 0; i < warmup; ++i)
    {
        tensorrt_llm::kernels::smooth_quant::int8_sq_launcher<T>(params, s);
    }
    cudaEventRecord(begin, s);
    for (int i = 0; i < iter; ++i)
    {
        tensorrt_llm::kernels::smooth_quant::int8_sq_launcher<T>(params, s);
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

template <typename T>
float run_cutlass_kernel(Params& params, int warmup, int iter)
{
    auto runner = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassInt8GemmRunner<T>>();
    auto& gemm = *runner;
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    auto configs = get_configs(gemm, params.k);
    int ws_bytes = gemm.getWorkspaceSize(params.m, params.n, params.k);
    char* ws_ptr = nullptr;
    if (ws_bytes)
        cudaMalloc(&ws_ptr, ws_bytes);
    float fast_time = 1e8;
    auto best_config = configs[0];
    bool found = false;
    for (auto& config : configs)
    {
        try
        {
            for (int i = 0; i < 2; ++i)
            {
                gemm.gemm(params.act, params.weight, params.quant_mode, params.scale_channels, params.scale_tokens,
                    params.output, params.m, params.n, params.k, config, ws_ptr, ws_bytes, s);
            }
            cudaEventRecord(begin, s);
            for (int i = 0; i < 5; ++i)
            {
                gemm.gemm(params.act, params.weight, params.quant_mode, params.scale_channels, params.scale_tokens,
                    params.output, params.m, params.n, params.k, config, ws_ptr, ws_bytes, s);
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
            found = true;
        }
        catch (std::exception const& e)
        {
        }
    }
    if (!found)
    {
        throw std::runtime_error("Have no valid config!");
    }

    for (int i = 0; i < warmup; ++i)
    {
        gemm.gemm(params.act, params.weight, params.quant_mode, params.scale_channels, params.scale_tokens,
            params.output, params.m, params.n, params.k, best_config, ws_ptr, ws_bytes, s);
    }
    cudaEventRecord(begin, s);
    for (int i = 0; i < iter; ++i)
    {
        gemm.gemm(params.act, params.weight, params.quant_mode, params.scale_channels, params.scale_tokens,
            params.output, params.m, params.n, params.k, best_config, ws_ptr, ws_bytes, s);
    }
    if (ws_ptr)
        cudaFree(ws_ptr);
    cudaEventRecord(end, s);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(s);
    return time / iter;
}

template <typename T>
bool benchmark_and_verify(int m, int n, int k, tensorrt_llm::common::QuantMode const& quant_mode, int warmup, int iter)
{
    std::srand(20240123);
    simple_assert(m <= 4);
    static constexpr int WSizeInBits = 8;
    bool per_token = quant_mode.hasPerTokenScaling();
    bool per_channel = quant_mode.hasPerChannelScaling();
    printf("mnk (%d, %d, %d), per token: %d, per channel: %d\n", m, n, k, per_token ? 1 : 0, per_channel ? 1 : 0);
    CudaBuffer d_act(m * k);
    CudaBuffer d_weight(k * n);
    CudaBuffer d_scale_tokens(per_token ? m * sizeof(float) : sizeof(float));
    CudaBuffer d_scale_channels(per_channel ? n * sizeof(float) : sizeof(float));
    CudaBuffer d_out(m * n * sizeof(T));
    std::vector<int8_t> h_act(m * k);
    std::vector<int8_t> h_weight(k * n);
    std::vector<float> h_scale_tokens(per_token ? m : 1), h_scale_channels(per_channel ? n : 1);
    std::vector<T> h_out1(m * n), h_out2(m * n);

    random_fill(h_scale_tokens, -1.f, 1.f);
    random_fill(h_scale_channels, -1.f, 1.f);

    for (int8_t& v : h_act)
    {
        v = (rand() % 256) - 128;
    }
    for (int8_t& v : h_weight)
    {
        v = (rand() % 256) - 128;
    }

    d_act.copy_from(h_act.data());
    d_weight.copy_from(h_weight.data());
    d_scale_tokens.copy_from(h_scale_tokens.data());
    d_scale_channels.copy_from(h_scale_channels.data());

    Params params{d_act.data<int8_t>(), d_weight.data<int8_t>(), d_scale_tokens.data<float>(),
        d_scale_channels.data<float>(), d_out.data(), m, n, k, quant_mode};

    float time1, time2;
    time1 = run_cuda_kernel<T>(params, warmup, iter);
    d_out.copy_to(h_out1.data());
    time2 = run_cutlass_kernel<T>(params, warmup, iter);
    d_out.copy_to(h_out2.data());
    float quant_scale = 1.f / (1 << (WSizeInBits - 1));
    bool pass = compare<T>(h_out1.data(), h_out2.data(), m * n, quant_scale);
    printf(
        "cuda kernel cost time %.6f, cutlass kernel cost time %.6f, cuda speedup %.3f\n", time1, time2, time2 / time1);
    return pass;
}

TEST(Kernel, WeightOnly)
{
    int const arch = tensorrt_llm::common::getSMVersion();
    bool pass;
    int warmup = 10, iter = 30;
    std::vector<int> ms{1, 2, 4};
    std::vector<int> ns{2048, 4096};
    std::vector<int> ks{2048, 4096};
    std::vector<tensorrt_llm::common::QuantMode> quant_modes(4);
    quant_modes[0] = tensorrt_llm::common::QuantMode::fromDescription(
        false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false);
    quant_modes[1] = tensorrt_llm::common::QuantMode::fromDescription(
        false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false);
    quant_modes[2] = tensorrt_llm::common::QuantMode::fromDescription(
        false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false);
    quant_modes[3] = tensorrt_llm::common::QuantMode::fromDescription(
        false, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false);
    for (auto m : ms)
    {
        for (auto n : ns)
        {
            for (auto k : ks)
            {
                for (auto quant_mode : quant_modes)
                {
                    pass = benchmark_and_verify<float>(m, n, k, quant_mode, warmup, iter);
                    EXPECT_TRUE(pass);
                    pass = benchmark_and_verify<half>(m, n, k, quant_mode, warmup, iter);
                    EXPECT_TRUE(pass);
                    pass = benchmark_and_verify<int>(m, n, k, quant_mode, warmup, iter);
                    EXPECT_TRUE(pass);
                }
            }
        }
    }
}
