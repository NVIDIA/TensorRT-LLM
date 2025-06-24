#include <NvInferRuntime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "cutlass/numeric_types.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemm.h"

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
using namespace tensorrt_llm::kernels::cuda_core_gemm;

void simple_assert(bool flag)
{
    if (!flag)
    {
        throw std::runtime_error("assert failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
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
bool compare(void* _pa, void* _pb, int size)
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
    float diff_thres = max_val * 2e-3;
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

template <typename T1, typename T2>
void constant_fill(std::vector<T1>& vec, T2 value)
{
    for (auto& v : vec)
    {
        v = static_cast<T1>(value);
    }
}

template <typename T1>
void linear_fill(std::vector<T1>& vec, int length)
{
    for (int i = 0; i < vec.size(); ++i)
    {
        vec[i] = static_cast<T1>((i % length) / 100.f);
    }
}

template <typename T>
void print_mat(std::vector<T> const& data, int row, int col, char const* name)
{
    assert(data.size() == row * col);
    printf("---------------%s\n", name);
    for (int n = 0; n < data.size(); ++n)
    {
        float value = static_cast<float>(data[n]);
        printf("%f, ", value);
        if (n % col == col - 1)
            printf("\n");
    }
    printf("\n");
}

template <typename InputType, typename OutputType>
void run_cpu(void* weight, void* activation, float scale, Params const& params, void* output)
{
    for (int idx_m = 0; idx_m < params.m; ++idx_m)
    {
        for (int idx_n = 0; idx_n < params.n; ++idx_n)
        {
            float acc = 0.f;
            for (int idx_k = 0; idx_k < params.k; ++idx_k)
            {
                InputType a = reinterpret_cast<InputType*>(activation)[params.k * idx_m + idx_k];
                InputType w = reinterpret_cast<InputType*>(weight)[params.k * idx_n + idx_k];
                acc += static_cast<float>(w) * static_cast<float>(a);
            }
            reinterpret_cast<OutputType*>(output)[idx_m * params.n + idx_n] = static_cast<OutputType>(acc * scale);
        }
    }
}

float run_cuda_kernel(Params& params, int warmup, int iter)
{
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (int i = 0; i < warmup; ++i)
    {
        cudaCoreGemmDispatcher(params, s);
    }
    cudaEventRecord(begin, s);
    for (int i = 0; i < iter; ++i)
    {
        cudaCoreGemmDispatcher(params, s);
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

template <typename InputType, typename OutputType>
float run_cublas_kernel(Params& params, int warmup, int iter)
{
    constexpr cudaDataType_t kOutputDatatype = std::is_same<OutputType, __nv_bfloat16>::value ? CUDA_R_16BF
        : std::is_same<OutputType, float>::value                                              ? CUDA_R_32F
                                                                                              : CUDA_R_16F;

    // use weight as A, use activation as B so that D is transposed(WIP)
    void const* A = params.weight;
    void const* B = params.act;
    void* D = params.output;

    int m = params.m, n = params.n, k = params.k;
    float h_alpha = params.alpha;
    void* workspace = nullptr;
    size_t workspaceSize = 32 * 1024 * 1024; // 32MB for Hopper
    cudaMalloc(&workspace, workspaceSize);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    cublasLtHandle_t ltHandle;
    checkCublasStatus(cublasLtCreate(&ltHandle));
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    // only support TN for FP8
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    float h_beta = 0.0; // Can be non-zero starting from 12.0

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    // table of supported type combinations can be found in the documentation:
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, k, n, k));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, k, m, k));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, kOutputDatatype, n, m, n));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, kOutputDatatype, n, m, n));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0)
    {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    for (int i = 0; i < warmup; ++i)
    {
        checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, &h_alpha, A, Adesc, B, Bdesc, &h_beta, nullptr, Cdesc,
            D, Ddesc, &heuristicResult.algo, workspace, workspaceSize, stream));
    }
    cudaEventRecord(begin, stream);
    for (int i = 0; i < iter; ++i)
    {
        checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, &h_alpha, A, Adesc, B, Bdesc, &h_beta, nullptr, Cdesc,
            D, Ddesc, &heuristicResult.algo, workspace, workspaceSize, stream));
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    if (workspace)
        cudaFree(workspace);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference)
        checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Ddesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc)
        checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));

    return time / iter;
}

template <typename InputType, typename OutputType>
bool benchmark_and_verify(int m, int n, int k, int warmup, int iter, bool debug = false, bool run_cublas = false)
{
    constexpr cudaDataType_t kInputDatatype = std::is_same<InputType, float>::value ? CUDA_R_32F
        : std::is_same<InputType, half>::value                                      ? CUDA_R_16F
        : std::is_same<InputType, __nv_bfloat16>::value                             ? CUDA_R_16BF
                                                                                    : CUDA_R_8F_E4M3;

    constexpr cudaDataType_t kOutputDatatype = std::is_same<OutputType, float>::value ? CUDA_R_32F
        : std::is_same<OutputType, half>::value                                       ? CUDA_R_16F
        : std::is_same<OutputType, __nv_bfloat16>::value                              ? CUDA_R_16BF
                                                                                      : CUDA_R_8F_E4M3;

    std::srand(20240123);
    simple_assert(m <= 4);
    printf("mnk (%d, %d, %d), output %s\n", m, n, k, typeid(OutputType).name());
    CudaBuffer d_act(m * k * sizeof(InputType));
    CudaBuffer d_weight(k * n * sizeof(InputType));
    CudaBuffer d_out(m * n * sizeof(OutputType));
    std::vector<InputType> h_act(m * k);
    std::vector<InputType> h_weight(k * n);
    std::vector<float> h_alpha(1);
    std::vector<OutputType> h_out_cuda(m * n), h_out_cublas(m * n), h_out_gt(m * n);

    random_fill(h_act, -1.f, 1.f);
    random_fill(h_weight, -1.f, 1.f);
    random_fill(h_alpha, -1.f, 1.f);

    if (debug)
    {
        print_mat(h_act, m, k, "h_act");
        print_mat(h_weight, k, n, "h_weight");
        print_mat(h_alpha, 1, 1, "h_alpha");
    }

    d_act.copy_from(h_act.data());
    d_weight.copy_from(h_weight.data());

    Params params{d_act.data(), d_weight.data(), h_alpha[0], d_out.data(), m, n, k, kInputDatatype, kOutputDatatype};

    run_cpu<InputType, OutputType>(h_weight.data(), h_act.data(), h_alpha[0], params, h_out_gt.data());

    float time1, time2;
    time1 = run_cuda_kernel(params, warmup, iter);
    d_out.copy_to(h_out_cuda.data());
    bool pass_cuda_kernel = compare<OutputType>(h_out_cuda.data(), h_out_gt.data(), m * n);

    if (debug)
    {
        print_mat<OutputType>(h_out_gt, m, n, "h_out_cpu");
        print_mat<OutputType>(h_out_cuda, m, n, "h_out_cuda");
    }

    if (run_cublas)
    {
        time2 = run_cublas_kernel<InputType, OutputType>(params, warmup, iter);
        d_out.copy_to(h_out_cublas.data());
        bool pass_cublas = compare<OutputType>(h_out_cublas.data(), h_out_gt.data(), m * n);

        if (debug)
        {
            print_mat<OutputType>(h_out_cublas, m, n, "h_out_cublas");
        }

        printf("cuda kernel cost time %.6f, cublas kernel cost time %.6f, cuda speedup %.3f\n", time1, time2,
            time2 / time1);
        return pass_cuda_kernel && pass_cublas;
    }

    printf("cuda kernel cost time %.6f\n", time1);
    return pass_cuda_kernel;
}

#ifdef ENABLE_FP8
TEST(Kernel, Fp8Gemv)
{
    int const arch = tensorrt_llm::common::getSMVersion();
    bool pass;
    int warmup = 10, iter = 30;
    std::vector<int> ms{1, 2, 3, 4};
    std::vector<int> ns{2048, 4096};
    std::vector<int> ks{2048, 4096};
    for (auto m : ms)
    {
        for (auto n : ns)
        {
            for (auto k : ks)
            {
                pass = benchmark_and_verify<__nv_fp8_e4m3, float>(m, n, k, warmup, iter);
                EXPECT_TRUE(pass);
                pass = benchmark_and_verify<__nv_fp8_e4m3, half>(m, n, k, warmup, iter);
                EXPECT_TRUE(pass);
#if defined(ENABLE_BF16)
                pass = benchmark_and_verify<__nv_fp8_e4m3, __nv_bfloat16>(m, n, k, warmup, iter);
                EXPECT_TRUE(pass);
#endif
            }
        }
    }
}
#endif

TEST(Kernel, Fp16Gemv)
{
    int const arch = tensorrt_llm::common::getSMVersion();
    bool pass;
    int warmup = 10, iter = 30;
    std::vector<int> ms{1, 2, 3, 4};
    std::vector<int> ns{2048, 4096};
    std::vector<int> ks{2048, 4096};
    for (auto m : ms)
    {
        for (auto n : ns)
        {
            for (auto k : ks)
            {
                pass = benchmark_and_verify<float, float>(m, n, k, warmup, iter);
                EXPECT_TRUE(pass);
                pass = benchmark_and_verify<half, half>(m, n, k, warmup, iter);
                EXPECT_TRUE(pass);
#if defined(ENABLE_BF16)
                pass = benchmark_and_verify<__nv_bfloat16, __nv_bfloat16>(m, n, k, warmup, iter);
                EXPECT_TRUE(pass);
#endif
            }
        }
    }
}
