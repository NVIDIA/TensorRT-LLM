/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cublasScaledMMLut.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/userbuffers/ub_interface.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/outputTensor.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "userbuffersTensor.h"
#include <cublasLt.h>
#include <map>
#include <mutex>
#include <torch/extension.h>
#include <tuple>
#include <unordered_map>

using torch::Tensor;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

using tensorrt_llm::common::check;
using tensorrt_llm::common::CublasMMWrapper;

using cublas_lut::AlgoListType;

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
    AlgoListType const* algo_list = nullptr;
    if ((aType == CUDA_R_16BF || aType == CUDA_R_16F) && (outType == aType || outType == CUDA_R_32F)
        && compType == CUBLAS_COMPUTE_32F)
    {
        // TODO: remove this after cublas fix the heuristic for Spark
        algo_list = tensorrt_llm::common::getSMVersion() == 121 ? &cublas_lut::spark_bf16_algo_list
                                                                : &cublas_lut::bf16_algo_list;
    }
    else if (aType == CUDA_R_8F_E4M3 && compType == CUBLAS_COMPUTE_32F)
    {
        algo_list = &cublas_lut::fp8_algo_list;
    }
    else
    {
        TLLM_LOG_DEBUG(
            "No special cublasLt algo found for aType=%d, outType=%d, compType=%d\n", aType, outType, compType);
        return false;
    }
    if (auto algo_iter = algo_list->find({mp2, k, n}); algo_iter != algo_list->end())
    {
        int const algoID = algo_iter->second[0];
        check_cuda_error(cublasLtMatmulAlgoInit(
            cublasWrapper->getCublasLtHandle(), compType, scaleType, aType, bType, outType, outType, algoID, &algo));
        TLLM_LOG_DEBUG("Found special cublasLt algo for m=%d, k=%d, n=%d\n", m, k, n);
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

// Helper function: Get or create a workspace tensor for the given (device, stream).
// Workspace is reused across multiple GEMM calls so the pointer captured by the
// cublasLt kernel remains valid for CUDA-graph capture/replay. Keyed by
// (device, stream) so concurrent GEMMs on different streams of the same device
// don't race on the same scratch bytes.
inline at::Tensor const& getWorkspaceTensor(c10::Device device, cudaStream_t stream)
{
    struct KeyHash
    {
        std::size_t operator()(std::pair<int, cudaStream_t> const& key) const noexcept
        {
            return std::hash<int>()(key.first) ^ (std::hash<cudaStream_t>()(key.second) << 1);
        }
    };

    thread_local std::unordered_map<std::pair<int, cudaStream_t>, at::Tensor, KeyHash> workspace_tensors;
    auto key = std::make_pair(device.index(), stream);

    if (workspace_tensors.find(key) == workspace_tensors.end())
    {
        workspace_tensors[key]
            = torch::empty(CUBLAS_WORKSPACE_SIZE, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    }

    return workspace_tensors[key];
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

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    auto const& workspace = getWorkspaceTensor(a.device(), stream.stream());

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

// cuBLASLt epilogues of cublas_scaled_mm beyond the default bias add.
enum class ScaledMmEpilogue : int64_t
{
    kDefault = 0, // bias added when given
    kGelu = 1,    // tanh GELU after the optional bias
};

struct EpilogueAlgoKey
{
    int device;
    int64_t m, n, k, lda, ldb, ldd;
    int epilogue;
    int outType;
    int biasType;

    bool operator<(EpilogueAlgoKey const& o) const
    {
        return std::tie(device, m, n, k, lda, ldb, ldd, epilogue, outType, biasType)
            < std::tie(o.device, o.m, o.n, o.k, o.lda, o.ldb, o.ldd, o.epilogue, o.outType, o.biasType);
    }
};

// FP8 x FP8 GEMM through cuBLASLt with a GELU / bias epilogue and an optional e4m3 output scaled by
// d_scale. The algorithm is the cuBLASLt heuristic's first choice, cached per problem.
void cublaslt_fp8_epilogue_gemm(Tensor& out, Tensor const& a, Tensor const& b, Tensor const& scale_a,
    Tensor const& scale_b, std::optional<at::Tensor> const& bias, ScaledMmEpilogue epilogue,
    std::optional<at::Tensor> const& d_scale)
{
    int64_t const m = a.size(0);
    int64_t const n = b.size(1);
    int64_t const k = a.size(1);
    bool const outFp8 = out.scalar_type() == torch::kFloat8_e4m3fn;
    TORCH_CHECK(outFp8 || out.scalar_type() == torch::kBFloat16, "cublas_scaled_mm epilogues output bf16 or e4m3");
    TORCH_CHECK(outFp8 == d_scale.has_value(), "e4m3 output needs d_scale, and d_scale needs e4m3 output");
    TORCH_CHECK(scale_a.numel() == 1 && scale_b.numel() == 1, "cublas_scaled_mm epilogues need per-tensor scales");

    thread_local std::shared_ptr<cublasLtHandle_t> ltHandle;
    if (ltHandle == nullptr)
    {
        ltHandle = getCublasLtHandle();
    }
    auto const handle = *ltHandle;
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    auto const& workspace = getWorkspaceTensor(a.device(), stream.stream());
    size_t const workspaceBytes = CUBLAS_WORKSPACE_SIZE;

    cublasLtMatmulDesc_t op;
    check_cuda_error(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    // Row-major out = a @ b is column-major out^T = b^T (A, transposed) x a^T (B).
    cublasOperation_t const transA = CUBLAS_OP_T;
    cublasOperation_t const transB = CUBLAS_OP_N;
    check_cuda_error(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    check_cuda_error(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    void const* scaleAPtr = scale_b.data_ptr();
    void const* scaleBPtr = scale_a.data_ptr();
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleAPtr, sizeof(scaleAPtr)));
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleBPtr, sizeof(scaleBPtr)));
    int8_t const fastAccum = 1;
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum, sizeof(fastAccum)));
    if (outFp8)
    {
        void const* dScalePtr = d_scale->data_ptr();
        check_cuda_error(
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &dScalePtr, sizeof(dScalePtr)));
    }
    bool const gelu = epilogue == ScaledMmEpilogue::kGelu;
    cublasLtEpilogue_t const epi = bias.has_value() ? (gelu ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS)
                                                    : (gelu ? CUBLASLT_EPILOGUE_GELU : CUBLASLT_EPILOGUE_DEFAULT);
    check_cuda_error(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    cudaDataType_t biasType = CUDA_R_16BF;
    if (bias.has_value())
    {
        TORCH_CHECK(bias->numel() == n && bias->is_contiguous(), "bias must be contiguous [n]");
        biasType = convert_torch_dtype(bias->scalar_type());
        void const* biasPtr = bias->data_ptr();
        check_cuda_error(
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasPtr, sizeof(biasPtr)));
        check_cuda_error(
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType)));
    }

    int64_t const lda = b.stride(1);
    int64_t const ldb = a.stride(0);
    int64_t const ldd = out.stride(0);
    cudaDataType_t const outType = convert_torch_dtype(out.scalar_type());
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
    check_cuda_error(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8F_E4M3, k, n, lda));
    check_cuda_error(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8F_E4M3, k, m, ldb));
    // An e4m3 D still takes a bf16 C descriptor; C is unused with beta = 0.
    check_cuda_error(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, n, m, ldd));
    check_cuda_error(cublasLtMatrixLayoutCreate(&layoutD, outType, n, m, ldd));

    static std::mutex algoMutex;
    static std::map<EpilogueAlgoKey, cublasLtMatmulAlgo_t> algos;
    cublasLtMatmulAlgo_t algo;
    {
        std::lock_guard<std::mutex> lock(algoMutex);
        EpilogueAlgoKey const key{a.get_device(), m, n, k, lda, ldb, ldd, static_cast<int>(epi),
            static_cast<int>(outType), static_cast<int>(biasType)};
        auto it = algos.find(key);
        if (it == algos.end())
        {
            cublasLtMatmulPreference_t pref;
            check_cuda_error(cublasLtMatmulPreferenceCreate(&pref));
            check_cuda_error(cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes)));
            cublasLtMatmulHeuristicResult_t result{};
            int found = 0;
            check_cuda_error(cublasLtMatmulAlgoGetHeuristic(
                handle, op, layoutA, layoutB, layoutC, layoutD, pref, 1, &result, &found));
            check_cuda_error(cublasLtMatmulPreferenceDestroy(pref));
            TORCH_CHECK(found > 0, "cublas_scaled_mm: no cuBLASLt algorithm for m=", m, " n=", n, " k=", k,
                " epilogue=", static_cast<int>(epi));
            it = algos.emplace(key, result.algo).first;
        }
        algo = it->second;
    }
    float const alpha = 1.f;
    float const beta = 0.f;
    void* cPtr = outFp8 ? nullptr : out.data_ptr();
    check_cuda_error(cublasLtMatmul(handle, op, &alpha, b.data_ptr(), layoutA, a.data_ptr(), layoutB, &beta, cPtr,
        layoutC, out.data_ptr(), layoutD, &algo, workspace.data_ptr(), workspaceBytes, stream));
    check_cuda_error(cublasLtMatrixLayoutDestroy(layoutA));
    check_cuda_error(cublasLtMatrixLayoutDestroy(layoutB));
    check_cuda_error(cublasLtMatrixLayoutDestroy(layoutC));
    check_cuda_error(cublasLtMatrixLayoutDestroy(layoutD));
    check_cuda_error(cublasLtMatmulDescDestroy(op));
}

} // namespace

Tensor& cublas_scaled_mm_out(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    std::optional<at::Tensor> const& bias, Tensor& out, int64_t epilogue = 0,
    std::optional<at::Tensor> const& d_scale = std::nullopt)
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

    TORCH_CHECK(epilogue == static_cast<int64_t>(ScaledMmEpilogue::kDefault)
            || epilogue == static_cast<int64_t>(ScaledMmEpilogue::kGelu),
        "cublas_scaled_mm: epilogue must be 0 (default) or 1 (GELU)");
    if (epilogue != static_cast<int64_t>(ScaledMmEpilogue::kDefault) || d_scale.has_value())
    {
        if (d_scale.has_value())
        {
            CHECK_TH_CUDA(d_scale.value());
            TORCH_CHECK(d_scale->scalar_type() == torch::kFloat32 && d_scale->numel() == 1,
                "cublas_scaled_mm: d_scale must be float32 [1]");
        }
        cublaslt_fp8_epilogue_gemm(
            out, mat_a, mat_b, scale_a, scale_b, bias, static_cast<ScaledMmEpilogue>(epilogue), d_scale);
        return out;
    }
    cublas_gemm_caller(out, mat_a, mat_b, scale_a, scale_b, bias, true);
    return out;
}

// epilogue: 0 adds the optional bias; 1 applies tanh GELU after the optional bias.
// d_scale: float32 [1] multiplier applied before an e4m3 output (1 / output dequant scale).
Tensor cublas_scaled_mm(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    std::optional<at::Tensor> const& bias, std::optional<c10::ScalarType> out_dtype, int64_t output_buffer_kind = 0,
    c10::optional<torch::List<int64_t>> group = c10::nullopt, int64_t epilogue = 0,
    std::optional<at::Tensor> const& d_scale = std::nullopt)
{
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());

    std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[1]};

    auto [out, _] = torch_ext::allocate_output(
        output_size, out_dtype_, mat_a.device(), static_cast<torch_ext::BufferKind>(output_buffer_kind), group);

    return cublas_scaled_mm_out(mat_a, mat_b, scale_a, scale_b, bias, out, epilogue, d_scale);
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
    std::optional<c10::ScalarType> out_dtype, int64_t output_buffer_kind = 0,
    c10::optional<torch::List<int64_t>> group = c10::nullopt)
{
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[1]};
    auto [out, _] = torch_ext::allocate_output(
        output_size, out_dtype_, mat_a.device(), static_cast<torch_ext::BufferKind>(output_buffer_kind), group);
    return cublas_mm_out(mat_a, mat_b, bias, out);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "cublas_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b, Tensor? bias,"
        " ScalarType? out_dtype, int output_buffer_kind=0, int[]? group=None, int epilogue=0,"
        " Tensor? d_scale=None) -> (Tensor out)");
    m.def(
        "cublas_mm(Tensor mat_a, Tensor mat_b, Tensor? bias, ScalarType? out_dtype,"
        " int output_buffer_kind=0, int[]? group=None) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("cublas_scaled_mm", &tensorrt_llm::torch_ext::cublas_scaled_mm);
    m.impl("cublas_mm", &tensorrt_llm::torch_ext::cublas_mm);
}
