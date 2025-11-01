#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include <memory>
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

using namespace tensorrt_llm;
namespace tr = tensorrt_llm::runtime;

template <typename T>
std::vector<float> toFloatCPU(tr::IBuffer const& gpu, size_t n, tr::BufferManager& mgr)
{
    auto host = mgr.copyFrom(gpu, tr::MemoryType::kCPU);
    mgr.getStream().synchronize();
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) { out[i] = static_cast<float>(reinterpret_cast<T const*>(host->data())[i]); }
    return out;
}

// Very basic tests for functionality

TEST(InvokeConversion, FP16toFP8)
{
    const int N = 1024;
    auto streamPtr = std::make_shared<tr::CudaStream>();
    tr::BufferManager mgr{streamPtr};

    std::vector<half> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = __float2half_rn(float(i % 17 - 8)); // [-8, 8] integers only

    auto d_in  = mgr.copyFrom(h_in, tr::MemoryType::kGPU);
    auto d_out = mgr.gpu(tr::ITensor::makeShape({N}), nvinfer1::DataType::kFP8);

    kernels::invokeConversion<__nv_fp8_e4m3, half>(
        reinterpret_cast<__nv_fp8_e4m3*>(d_out->data()),
        reinterpret_cast<half const*>(d_in->data()),
        N,
        nullptr,
        mgr.getStream().get());

    mgr.getStream().synchronize();

    auto h_out = toFloatCPU<__nv_fp8_e4m3>(*d_out, N, mgr);

    for (int i = 0; i < N; ++i)
    {
        float ref = __half2float(h_in[i]);
        EXPECT_NEAR(h_out[i], ref, 0.5f);
    }
}

TEST(InvokeConversion, BF16toFP8)
{
    const int N = 1024;
    auto streamPtr = std::make_shared<tr::CudaStream>();
    tr::BufferManager mgr{streamPtr};

    std::vector<__nv_bfloat16> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = __float2bfloat16(float(i % 23 - 11)); // [-11, 11] integers only

    auto d_in  = mgr.copyFrom(h_in, tr::MemoryType::kGPU);
    auto d_out = mgr.gpu(tr::ITensor::makeShape({N}), nvinfer1::DataType::kFP8);

    kernels::invokeConversion<__nv_fp8_e4m3, __nv_bfloat16>(
        reinterpret_cast<__nv_fp8_e4m3*>(d_out->data()),
        reinterpret_cast<__nv_bfloat16 const*>(d_in->data()),
        N,
        nullptr,
        mgr.getStream().get());

    mgr.getStream().synchronize();
    auto h_out = toFloatCPU<__nv_fp8_e4m3>(*d_out, N, mgr);

    for (int i = 0; i < N; ++i)
    {
        float ref = __bfloat162float(h_in[i]);
        EXPECT_NEAR(h_out[i], ref, 0.5f);
    }
}