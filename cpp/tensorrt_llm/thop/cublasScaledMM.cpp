/*
 * SPDX-FileCopyrightText: Copyright (out) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/userbuffers/ub_interface.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/gemmPlugin/gemmPlugin.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "userbuffersTensor.h"
#include <array>
#include <cublasLt.h>
#include <torch/extension.h>
#include <unordered_map>

using torch::Tensor;

namespace torch_ext
{

namespace
{

using tensorrt_llm::common::check;
using tensorrt_llm::common::CublasMMWrapper;

struct hash_tuple
{
    size_t operator()(std::tuple<int, int, int> const& x) const
    {
        return std::get<0>(x) ^ std::get<1>(x) ^ std::get<2>(x);
    }
};

// got from cublasTest matmultFind
// {mp2, k, n}: {algo, m_tile, m_stages, m_numsK, m_reduction, m_swizzle, m_custom, m_cga}
using AlgoListType = std::unordered_map<std::tuple<int32_t, int32_t, int32_t>, std::array<int, 8>, hash_tuple>;

// bf16*bf16->fp32->bf16
AlgoListType spark_bf16_algo_list = {
    // GPT-OSS-20b
    //-m201088 -n1 -algo21 -m_tile11 -m_stages20 -m_workmem0 -k2880
    {{8, 2880, 201088}, {21, 11, 20, 1, 0, 0, 0, 0}},
    //-m32 -n1 -algo14 -m_reduction2 -m_numsK10 -m_workmem1024 -k2880
    {{8, 2880, 32}, {14, 0, 0, 10, 2, 0, 0, 0}},
    //-m32 -n2048 -algo21 -m_tile11 -m_stages13 -m_reduction1 -m_numsK9 -m_workmem1024
    //-k2880
    {{2048, 2880, 32}, {21, 11, 13, 9, 1, 0, 0, 0}},
    //-m32 -n2175 -algo21 -m_tile11 -m_stages19 -m_reduction1 -m_numsK11
    //-m_workmem1024 -k2880
    {{4096, 2880, 32}, {21, 11, 19, 11, 1, 0, 0, 0}},
    //-m5120 -n1 -algo23 -m_tile11 -m_stages8 -m_reduction1 -m_numsK2
    //-m_workmem1024 -k2880
    {{8, 2880, 5120}, {23, 11, 8, 2, 1, 0, 0, 0}},
    //-m5120 -n2048 -algo21 -m_tile20 -m_stages15 -m_workmem1024 -k2880
    {{2048, 2880, 5120}, {21, 20, 15, 1, 0, 0, 0, 0}},
    //-m5120 -n2175 -algo21 -m_tile20 -m_stages15 -m_workmem1024 -k2880
    {{4096, 2880, 5120}, {21, 20, 15, 1, 0, 0, 0, 0}},
    //-m2880 -n1 -algo23 -m_tile11 -m_stages14 -m_reduction1 -m_numsK24 -m_workmem1024 -k4096
    {{8, 4096, 2880}, {23, 11, 14, 24, 1, 0, 0, 0}},
    //-m2880 -n2048 -ldc2880 -Poutt -ldd2880 -Ps -Pscales -algo21 -m_tile20 -m_stages15 -m_workmem1024 -k4096
    {{2048, 4096, 2880}, {21, 20, 15, 1, 0, 0, 0, 0}},
    //-m2880 -n2175 -ldc2880 -Poutt -ldd2880 -Ps -Pscales -algo21 -m_tile20 -m_stages15 -m_workmem1024 -k4096
    {{4096, 4096, 2880}, {21, 20, 15, 1, 0, 0, 0, 0}},
};

// bf16*bf16->fp32->bf16
AlgoListType bf16_algo_list = {
    // Deepseek v3/R1 router gemm
    // [-algo66 -m_tile10 -m_stages35 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom3 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 7168, 256}, {66, 10, 35, 1, 0, 0, 3, 2}},
    {{512, 7168, 256}, {66, 48, 35, 1, 0, 0, 0, 2}},
    {{1024, 7168, 256}, {66, 13, 35, 1, 0, 0, 1, 3}},
};

// fp8*fp8->fp32->fp16
AlgoListType fp8_algo_list = {
    // Llama-3.1-70B
    // [-algo66 -m_tile393 -m_stages36 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom5 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 8192, 8192}, {66, 393, 36, 1, 0, 0, 5, 2}},
    // [-algo66 -m_tile10 -m_stages36 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom1 -m_mma0 -m_cga2 -m_scheduling1]
    {{8, 8192, 57344}, {66, 10, 36, 1, 0, 0, 1, 2}},
    // Llama-3.3-70B TP4 (this is the default algo on B200. Here we aim to use the same algo on GB200.)
    // [-algo66 -m_tile393 -m_stages36 -m_numsK1 -m_reduction0 -m_swizzle0 -m_custom1 -m_mma0 -m_cga4 -m_scheduling1]
    {{8, 8192, 14336}, {66, 393, 36, 1, 0, 1, 1, 4}},
};

void set_algo_attr(cublasLtMatmulAlgo_t& algo, std::array<int, 8> const& attr_list)
{
    auto const& [algoId, tileID, stagesID, numsK, reduction, swizzle, customOption_, cga_] = attr_list;
    uint32_t customOption = customOption_;
    uint16_t cga = cga_;
    check_cuda_error(
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileID, sizeof(tileID)));
    check_cuda_error(
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesID, sizeof(stagesID)));
    check_cuda_error(
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numsK, sizeof(numsK)));
    check_cuda_error(cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction)));
    check_cuda_error(
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle)));
    check_cuda_error(cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption)));
    check_cuda_error(
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cga, sizeof(cga)));
}

bool find_special_algo(cublasLtMatmulAlgo_t& algo, std::shared_ptr<CublasMMWrapper> const& cublasWrapper, int32_t m,
    int32_t n, int32_t k, cublasComputeType_t compType, cudaDataType_t scaleType, cudaDataType_t aType,
    cudaDataType_t bType, cudaDataType_t outType)
{
    int32_t mp2 = std::max(nextPowerOfTwo(m), 8);
    AlgoListType algo_list;
    if ((aType == CUDA_R_16BF || aType == CUDA_R_16F) && (outType == aType || outType == CUDA_R_32F)
        && compType == CUBLAS_COMPUTE_32F)
    {
        // TODO: remove this after cublas fix the heuristic for Spark
        algo_list = tensorrt_llm::common::getSMVersion(/*queryRealSmArch=*/true) == 121 ? spark_bf16_algo_list
                                                                                        : bf16_algo_list;
    }
    else if (aType == CUDA_R_8F_E4M3 && compType == CUBLAS_COMPUTE_32F)
    {
        algo_list = fp8_algo_list;
    }
    else
    {
        TLLM_LOG_DEBUG(
            "No special cublasLt algo found for aType=%d, outType=%d, compType=%d\n", aType, outType, compType);
        return false;
    }
    if (auto algo_iter = algo_list.find({mp2, k, n}); algo_iter != algo_list.end())
    {
        int const algoID = algo_iter->second[0];
        check_cuda_error(cublasLtMatmulAlgoInit(
            cublasWrapper->getCublasLtHandle(), compType, scaleType, aType, bType, outType, outType, algoID, &algo));
        set_algo_attr(algo, algo_iter->second);
    }
    else
    {
        int const algoID = 66; // CUBLASLT_MATMUL_ALGO_NVJET
        check_cuda_error(cublasLtMatmulAlgoInit(
            cublasWrapper->getCublasLtHandle(), compType, scaleType, aType, bType, outType, outType, algoID, &algo));
        TLLM_LOG_DEBUG("No special cublasLt algo found for m=%d, k=%d, n=%d\n", m, k, n);
        return false;
    }
    TLLM_LOG_DEBUG("Found special cublasLt algo for m=%d, k=%d, n=%d\n", m, k, n);
    return true;
}

bool find_special_algo_deprecated(cublasLtMatmulAlgo_t& algo, std::shared_ptr<CublasMMWrapper> const& cublasWrapper,
    int32_t m, int32_t n, int32_t k, cublasComputeType_t compType, cudaDataType_t scaleType, cudaDataType_t aType,
    cudaDataType_t bType, cudaDataType_t outType)
{
    int32_t mp2 = std::max(nextPowerOfTwo(m), 8);
    if (aType != CUDA_R_8F_E4M3 || compType != CUBLAS_COMPUTE_32F)
    {
        return false;
    }
    int const algoID = 52;
    check_cuda_error(cublasLtMatmulAlgoInit(
        cublasWrapper->getCublasLtHandle(), compType, scaleType, aType, bType, outType, outType, algoID, &algo));
    int tileID = CUBLASLT_MATMUL_TILE_256x128;
    int swizzle = 0;
    uint16_t cga = CUBLASLT_CLUSTER_SHAPE_2x1x1;
    int const stagesID = CUBLASLT_MATMUL_STAGES_128xAUTO;
    int const numsK = -1;
    int const reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    if (mp2 <= 64)
    {
        tileID = CUBLASLT_MATMUL_TILE_64x64;
        swizzle = 1;
        if (n > k) // qkv & gate_up
            cga = CUBLASLT_CLUSTER_SHAPE_13x1x1;
        else       // o & down
            cga = CUBLASLT_CLUSTER_SHAPE_10x1x1;
    }
    else if (mp2 <= 256)
    {
        if (n > k) // qkv & gate_up
            tileID = CUBLASLT_MATMUL_TILE_192x128;
        else       // o & down
            tileID = CUBLASLT_MATMUL_TILE_128x128;
        swizzle = 1;
        cga = CUBLASLT_CLUSTER_SHAPE_1x2x1;
    }
    else if (mp2 <= 2048)
    {
        if (n > k) // qkv & gate_up
            tileID = CUBLASLT_MATMUL_TILE_160x128;
        else       // o & down
            tileID = CUBLASLT_MATMUL_TILE_256x128;
    }
    else
    {
        return false;
    }
    set_algo_attr(algo, {tileID, stagesID, numsK, reduction, swizzle, 0, cga});
    return true;
}

void cublas_gemm_caller(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
    std::optional<at::Tensor> const& scale_a, std::optional<at::Tensor> const& scale_b,
    std::optional<at::Tensor> const& bias, bool fast_acc = false)
{
    bool use_scale = false;
    if (scale_a.has_value() && scale_b.has_value())
    {
        use_scale = true;
    }

    int32_t m = a.sizes()[0];
    int32_t n = b.sizes()[1];
    int32_t k = a.sizes()[1];

    thread_local std::shared_ptr<CublasMMWrapper> cublasWrapper;
    if (cublasWrapper == nullptr)
    {
        auto cublasHandle = getCublasHandle();
        auto cublasLtHandle = getCublasLtHandle();
        cublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);
    }

    cudaDataType_t aType = convert_torch_dtype(a.scalar_type());
    cudaDataType_t bType = convert_torch_dtype(b.scalar_type());
    cudaDataType_t outType = convert_torch_dtype(out.scalar_type());

    // hardcode compute type for FP8
    cublasComputeType_t compType = CUBLAS_COMPUTE_32F;
    cudaDataType_t scaleType = CUDA_R_32F;
    cublasWrapper->setGemmConfig(aType, bType, outType, /*computeType=*/scaleType);

    auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
    auto workspace = torch::empty(CUBLAS_WORKSPACE_SIZE, workspace_options);

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    auto* a_ptr = static_cast<void*>(a.data_ptr());
    auto* b_ptr = static_cast<void*>(b.data_ptr());
    auto* out_ptr = static_cast<void*>(out.data_ptr());
    auto* ws_ptr = static_cast<void*>(workspace.data_ptr());
    void* a_scale = nullptr;
    void* b_scale = nullptr;
    if (use_scale)
    {
        a_scale = static_cast<void*>(scale_a.value().data_ptr());
        b_scale = static_cast<void*>(scale_b.value().data_ptr());
    }

    bool use_bias = bias.has_value();
    void* bias_ptr = nullptr;
    if (use_bias)
    {
        bias_ptr = static_cast<void*>(bias.value().data_ptr());
    }

    cublasWrapper->setStream(stream);
    cublasWrapper->setWorkspace(ws_ptr);

    // set algo according to m/n/k
    cublasLtMatmulAlgo_t algo;
#if CUDART_VERSION < 12080
    // nvjet is not supported
    bool has_algo
        = find_special_algo_deprecated(algo, cublasWrapper, m, n, k, compType, scaleType, aType, bType, outType);
#else
    bool has_algo = find_special_algo(algo, cublasWrapper, m, n, k, compType, scaleType, aType, bType, outType);
#endif

    // swap A and B. A is column major, B is row major.
    cublasWrapper->createDescriptors(
        CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, /*lda=*/k, /*ldb=*/k, /*ldc=*/n, /*fastAcc=*/fast_acc);
    if (use_scale)
        cublasWrapper->setScaleDescriptors(a_scale, b_scale);
    if (use_bias)
        cublasWrapper->setBiasDescriptor(bias_ptr);
    cublasWrapper->Gemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, /*A=*/b_ptr, /*lda=*/k, /*B=*/a_ptr, /*ldb=*/k, out_ptr,
        /*ldc=*/n, 1.0F, 0.0F, algo, has_algo, true);
    cublasWrapper->destroyDescriptors();
}

} // namespace

Tensor& cublas_scaled_mm_out(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    std::optional<at::Tensor> const& bias, Tensor& out)
{
    // Check device
    CHECK_TH_CUDA(mat_a);
    CHECK_TH_CUDA(mat_b);
    CHECK_TH_CUDA(scale_a);
    CHECK_TH_CUDA(scale_b);
    CHECK_TH_CUDA(out);

    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2 && out.dim() == 2);
    TORCH_CHECK(out.sizes()[0] == mat_a.sizes()[0] && mat_a.sizes()[1] == mat_b.sizes()[0]
        && mat_b.sizes()[1] == out.sizes()[1]);
    TORCH_CHECK(scale_a.numel() == 1 || scale_a.numel() == mat_a.sizes()[0]);
    TORCH_CHECK(scale_b.numel() == 1 || scale_b.numel() == mat_b.sizes()[1]);

    // Check for strides and alignment
    TORCH_CHECK(mat_a.strides()[1] == 1 && out.strides()[1] == 1);           // Row-major
    TORCH_CHECK(mat_b.strides()[0] == 1);                                    // Column-major
    TORCH_CHECK(out.strides()[0] % 16 == 0 && mat_b.strides()[1] % 16 == 0); // 16 Byte Alignment
    TORCH_CHECK(scale_a.is_contiguous() && scale_b.is_contiguous());

    TORCH_CHECK(mat_a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(mat_b.dtype() == torch::kFloat8_e4m3fn);

    cublas_gemm_caller(out, mat_a, mat_b, scale_a, scale_b, bias, true);
    return out;
}

Tensor cublas_scaled_mm(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    std::optional<at::Tensor> const& bias, std::optional<c10::ScalarType> out_dtype, bool to_userbuffers = false)
{
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());

    std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[1]};

    Tensor out;
    if (to_userbuffers)
    {
        out = torch_ext::create_userbuffers_tensor(output_size, out_dtype_).first;
    }
    else
    {
        out = at::empty(output_size, mat_a.options().dtype(out_dtype_));
    }

    return cublas_scaled_mm_out(mat_a, mat_b, scale_a, scale_b, bias, out);
}

Tensor& cublas_mm_out(Tensor const& mat_a, Tensor const& mat_b, std::optional<at::Tensor> const& bias, Tensor& out)
{
    // Check device
    CHECK_TH_CUDA(mat_a);
    CHECK_TH_CUDA(mat_b);
    CHECK_TH_CUDA(out);

    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2 && out.dim() == 2);
    // TODO: consider remove mat_b.to() and add extra transa & transb flag like trt's matmul
    TORCH_CHECK(out.sizes()[0] == mat_a.sizes()[0] && mat_a.sizes()[1] == mat_b.sizes()[0]
        && mat_b.sizes()[1] == out.sizes()[1]);

    // Check for strides and alignment
    TORCH_CHECK(mat_a.strides()[1] == 1 && out.strides()[1] == 1); // Row-major
    TORCH_CHECK(mat_b.strides()[0] == 1);                          // Column-major

    cublas_gemm_caller(out, mat_a, mat_b, at::nullopt, at::nullopt, bias, false);
    return out;
}

Tensor cublas_mm(Tensor const& mat_a, Tensor const& mat_b, std::optional<at::Tensor> const& bias,
    std::optional<c10::ScalarType> out_dtype)
{
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[1]};
    Tensor out = at::empty(output_size, mat_a.options().dtype(out_dtype_));
    return cublas_mm_out(mat_a, mat_b, bias, out);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "cublas_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b, Tensor? bias,"
        " ScalarType? out_dtype, bool to_userbuffers=False) -> (Tensor out)");
    m.def("cublas_mm(Tensor mat_a, Tensor mat_b, Tensor? bias, ScalarType? out_dtype) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("cublas_scaled_mm", &torch_ext::cublas_scaled_mm);
    m.impl("cublas_mm", &torch_ext::cublas_mm);
}
