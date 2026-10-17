/*
 * SPDX-FileCopyrightText: Copyright (out) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <torch/extension.h>
#include <unordered_map>

#include <algorithm>
#include <cstring>
#include <map>
#include <tuple>

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
        algo_list = tensorrt_llm::common::getSMVersion(/*queryRealSmArch=*/true) == 121
            ? &cublas_lut::spark_bf16_algo_list
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

// Candidate algorithms for one small-M BF16 GEMM on SM 121, indexed by the autotuner's tactic.
struct BF16Tactics
{
    bool hasLegacy; // index 0 is a table algorithm that passed cublasLtMatmulAlgoCheck
    std::vector<cublasLtMatmulAlgo_t> algorithms;
};

using BF16TacticKey = std::tuple<int, int32_t, int32_t, int32_t, bool>; // device, m, n, k, bias

std::map<BF16TacticKey, BF16Tactics>& getBF16TacticCache()
{
    thread_local std::map<BF16TacticKey, BF16Tactics> cache;
    return cache;
}

// Tactics are used only where they were measured: BF16 without scales, M <= 16, SM 121, packed aligned operands.
bool useBF16Tactics(torch::Tensor const& out, torch::Tensor const& a, torch::Tensor const& b, bool use_scale)
{
    auto const aligned = [](torch::Tensor const& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 256 == 0; };
    return a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16
        && !use_scale && a.size(0) <= 16 && tensorrt_llm::common::getSMVersion(/*queryRealSmArch=*/true) == 121
        && a.stride(0) == a.size(1) && b.stride(1) == b.size(0) && out.stride(0) == out.size(1) && aligned(a)
        && aligned(b) && aligned(out);
}

// Tactic 0 is today's choice, the rest are cuBLASLt's distinct heuristic answers that fit the workspace.
// Built on first use from the wrapper's descriptors for this GEMM.
BF16Tactics const& getBF16Tactics(
    CublasMMWrapper& wrapper, BF16TacticKey const& key, cublasLtMatmulAlgo_t const& algo, bool has_algo)
{
    auto& cache = getBF16TacticCache();
    if (auto it = cache.find(key); it != cache.end())
    {
        return it->second;
    }
    int32_t const m = std::get<1>(key);
    int32_t const n = std::get<2>(key);
    int32_t const k = std::get<3>(key);
    BF16Tactics tactics{has_algo && wrapper.checkTactic(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, k, k, n, algo), {algo}};
    for (auto const& heuristic : wrapper.getTactics(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, k, k, n, /*maxAlgorithms=*/64))
    {
        // An unchecked index 0 runs without an algorithm, so it cannot be a duplicate.
        bool const duplicate
            = std::any_of(tactics.algorithms.begin() + (tactics.hasLegacy ? 0 : 1), tactics.algorithms.end(),
                [&](cublasLtMatmulAlgo_t const& candidate)
                { return std::memcmp(&candidate, &heuristic.algo, sizeof(candidate)) == 0; });
        if (heuristic.state == CUBLAS_STATUS_SUCCESS && heuristic.workspaceSize <= CUBLAS_WORKSPACE_SIZE && !duplicate)
        {
            tactics.algorithms.push_back(heuristic.algo);
        }
    }
    return cache.emplace(key, std::move(tactics)).first->second;
}

// An out-of-range tactic keeps today's choice.
void selectBF16Tactic(
    CublasMMWrapper& wrapper, BF16TacticKey const& key, int64_t tactic, cublasLtMatmulAlgo_t& algo, bool& has_algo)
{
    auto const& tactics = getBF16Tactics(wrapper, key, algo, has_algo);
    if (tactic < static_cast<int64_t>(tactics.algorithms.size()))
    {
        algo = tactics.algorithms[tactic];
        has_algo = tactic != 0 || tactics.hasLegacy;
    }
}

void cublas_gemm_caller(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
    std::optional<at::Tensor> const& scale_a, std::optional<at::Tensor> const& scale_b,
    std::optional<at::Tensor> const& bias, bool fast_acc = false, int64_t tactic = -1)
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
    if (tactic >= 0 && useBF16Tactics(out, a, b, use_scale))
    {
        selectBF16Tactic(*cublasWrapper, {a.get_device(), m, n, k, use_bias}, tactic, algo, has_algo);
    }
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
    std::optional<at::Tensor> const& bias, std::optional<c10::ScalarType> out_dtype, int64_t output_buffer_kind = 0,
    c10::optional<torch::List<int64_t>> group = c10::nullopt)
{
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());

    std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[1]};

    auto [out, _] = torch_ext::allocate_output(
        output_size, out_dtype_, mat_a.device(), static_cast<torch_ext::BufferKind>(output_buffer_kind), group);

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

// cublas_mm with an autotuner tactic; -1 runs exactly what cublas_mm runs.
Tensor cublas_mm_tactic(Tensor const& mat_a, Tensor const& mat_b, std::optional<at::Tensor> const& bias,
    std::optional<c10::ScalarType> out_dtype, int64_t output_buffer_kind, c10::optional<torch::List<int64_t>> group,
    int64_t tactic)
{
    CHECK_TH_CUDA(mat_a);
    CHECK_TH_CUDA(mat_b);
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2 && mat_a.sizes()[1] == mat_b.sizes()[0]);
    TORCH_CHECK(mat_a.strides()[1] == 1 && mat_b.strides()[0] == 1);
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[1]};
    auto [out, _] = torch_ext::allocate_output(
        output_size, out_dtype_, mat_a.device(), static_cast<torch_ext::BufferKind>(output_buffer_kind), group);
    cublas_gemm_caller(out, mat_a, mat_b, at::nullopt, at::nullopt, bias, false, tactic);
    return out;
}

int64_t cublas_mm_num_tactics(Tensor const& mat_a, Tensor const& mat_b, std::optional<at::Tensor> const& bias)
{
    // Builds the candidate list through the normal call path, so counting runs one GEMM, during tuning only.
    cublas_mm_tactic(mat_a, mat_b, bias, std::nullopt, /*output_buffer_kind=*/0, std::nullopt, /*tactic=*/0);
    int32_t const m = mat_a.sizes()[0];
    int32_t const n = mat_b.sizes()[1];
    int32_t const k = mat_a.sizes()[1];
    auto const& cache = getBF16TacticCache();
    auto const it = cache.find({mat_a.get_device(), m, n, k, bias.has_value()});
    return it == cache.end() ? 0 : static_cast<int64_t>(it->second.algorithms.size());
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "cublas_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b, Tensor? bias,"
        " ScalarType? out_dtype, int output_buffer_kind=0, int[]? group=None)"
        " -> (Tensor out)");
    m.def(
        "cublas_mm(Tensor mat_a, Tensor mat_b, Tensor? bias, ScalarType? out_dtype,"
        " int output_buffer_kind=0, int[]? group=None) -> (Tensor out)");
    m.def(
        "cublas_mm_tactic(Tensor mat_a, Tensor mat_b, Tensor? bias, ScalarType? out_dtype,"
        " int output_buffer_kind, int[]? group, int tactic) -> (Tensor out)");
    m.def("cublas_mm_num_tactics(Tensor mat_a, Tensor mat_b, Tensor? bias) -> int");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("cublas_scaled_mm", &tensorrt_llm::torch_ext::cublas_scaled_mm);
    m.impl("cublas_mm", &tensorrt_llm::torch_ext::cublas_mm);
    m.impl("cublas_mm_tactic", &tensorrt_llm::torch_ext::cublas_mm_tactic);
    m.impl("cublas_mm_num_tactics", &tensorrt_llm::torch_ext::cublas_mm_num_tactics);
}
