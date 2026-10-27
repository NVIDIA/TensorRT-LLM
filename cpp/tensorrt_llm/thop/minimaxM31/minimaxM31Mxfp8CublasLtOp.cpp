/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// MXFP8 x MXFP8 GEMM on cuBLASLt with a PINNED algorithm (MiniMax-M3.1 tp2ctx, TP2CTX_MXFP8_CUBLASLT).
//
// out[M, N] (bf16) = act[M, K] (e4m3, 1x32 UE8M0 scales) x weight[N, K]^T (e4m3, 1x32 UE8M0 scales), as the
// column-major product D[N, M] = op(A)[N, K] x B[K, M] with A = weight (K x N, TRANSA = T), B = act (K x M),
// COMPUTE_32F, R_32F scale, VEC32_UE8M0 scale mode on both operands: the matmul torch._scaled_mm issues for the same
// operands. The scale tensors are the swizzled (128 x 4) UE8M0 buffers mxfp8_quantize(x, True) emits, which are
// cuBLASLt's VEC32_UE8M0 layout as they are. The algorithm is never chosen by heuristics at serving time: the caller
// passes the nine CUBLASLT_ALGO_CONFIG attributes of a configuration recorded offline
// (minimax_m31_mxfp8_cublaslt_heuristic dumps them); minimax_m31_mxfp8_cublaslt_algo_check reports whether a
// configuration serves a given (M, N, K).

#include "tensorrt_llm/common/config.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

#define M31_CUBLASLT_CHECK(call)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t const status_ = (call);                                                                         \
        TORCH_CHECK(status_ == CUBLAS_STATUS_SUCCESS, #call " failed with status ", static_cast<int>(status_));        \
    } while (0)

// The workspace the configurations are recorded with (torch's default cuBLASLt preference); a configuration's kernel
// set depends on it, so serving uses the same size.
constexpr size_t kWorkspaceBytes = size_t{1} << 20;

// The configuration attributes, in the order the algo vector carries them (CUBLASLT_ALGO_CONFIG_ID first).
constexpr std::array<cublasLtMatmulAlgoConfigAttributes_t, 9> kAlgoAttrs
    = {CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};

struct Descs
{
    cublasLtMatmulDesc_t op{nullptr};
    cublasLtMatrixLayout_t a{nullptr};
    cublasLtMatrixLayout_t b{nullptr};
    cublasLtMatrixLayout_t d{nullptr};

    Descs(int64_t m, int64_t n, int64_t k, void const* weightScale, void const* actScale)
    {
        M31_CUBLASLT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t const transA = CUBLAS_OP_T;
        cublasOperation_t const transB = CUBLAS_OP_N;
        M31_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
        M31_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
        cublasLtMatmulMatrixScale_t const scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        M31_CUBLASLT_CHECK(
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
        M31_CUBLASLT_CHECK(
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
        M31_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weightScale, sizeof(weightScale)));
        M31_CUBLASLT_CHECK(
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &actScale, sizeof(actScale)));
        M31_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, k, n, k));
        M31_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, k, m, k));
        M31_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&d, CUDA_R_16BF, n, m, n));
    }

    ~Descs()
    {
        cublasLtMatrixLayoutDestroy(d);
        cublasLtMatrixLayoutDestroy(b);
        cublasLtMatrixLayoutDestroy(a);
        cublasLtMatmulDescDestroy(op);
    }

    Descs(Descs const&) = delete;
    Descs& operator=(Descs const&) = delete;
};

cublasLtHandle_t blasLtHandle()
{
    return reinterpret_cast<cublasLtHandle_t>(at::cuda::getCurrentCUDABlasLtHandle());
}

// The storage size of a configuration attribute. cuBLASLt reads and writes an attribute only with exactly this many
// bytes (a larger buffer is CUBLAS_STATUS_INVALID_VALUE); a query with a null buffer and size 0 reports it.
size_t algoAttrSize(cublasLtMatmulAlgo_t const& algo, cublasLtMatmulAlgoConfigAttributes_t attr)
{
    size_t size = 0;
    M31_CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, nullptr, 0, &size));
    TORCH_CHECK(size > 0 && size <= sizeof(uint64_t), "minimax_m31_mxfp8_cublaslt: unexpected size ", size,
        " of algo config attribute ", static_cast<int>(attr));
    return size;
}

// The algorithm a recorded configuration names. Each attribute is written with its exact storage size (they differ:
// uint16 shape ids, 32-bit others); the value's low bytes carry it (little-endian).
cublasLtMatmulAlgo_t buildAlgo(std::vector<int64_t> const& config)
{
    TORCH_CHECK(config.size() == kAlgoAttrs.size(), "minimax_m31_mxfp8_cublaslt: the algo config holds ",
        kAlgoAttrs.size(),
        " attributes (algo_id, tile_id, stages_id, splitk_num, reduction_scheme, cta_swizzling, "
        "custom_option, inner_shape_id, cluster_shape_id), got ",
        config.size());
    cublasLtMatmulAlgo_t algo;
    M31_CUBLASLT_CHECK(cublasLtMatmulAlgoInit(blasLtHandle(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_8F_E4M3,
        CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_16BF, static_cast<int>(config[0]), &algo));
    for (size_t i = 1; i < kAlgoAttrs.size(); ++i)
    {
        uint64_t const value = static_cast<uint64_t>(config[i]);
        M31_CUBLASLT_CHECK(
            cublasLtMatmulAlgoConfigSetAttribute(&algo, kAlgoAttrs[i], &value, algoAttrSize(algo, kAlgoAttrs[i])));
    }
    return algo;
}

// One workspace per (device, stream), reused so a captured graph keeps a valid pointer.
at::Tensor const& workspace(c10::Device device, cudaStream_t stream)
{
    struct KeyHash
    {
        std::size_t operator()(std::pair<int, cudaStream_t> const& key) const noexcept
        {
            return std::hash<int>()(key.first) ^ (std::hash<cudaStream_t>()(key.second) << 1);
        }
    };

    thread_local std::unordered_map<std::pair<int, cudaStream_t>, at::Tensor, KeyHash> workspaces;
    auto const key = std::make_pair(static_cast<int>(device.index()), stream);
    auto it = workspaces.find(key);
    if (it == workspaces.end())
    {
        it = workspaces
                 .emplace(key,
                     torch::empty(static_cast<int64_t>(kWorkspaceBytes),
                         torch::TensorOptions().dtype(torch::kUInt8).device(device)))
                 .first;
    }
    return it->second;
}

} // namespace

namespace
{

// The shared launch of both entry points: D = act . weight^T into `out` ([M, N] bfloat16, row-major).
void checkOperands(at::Tensor const& act, at::Tensor const& actScale, at::Tensor const& weight,
    at::Tensor const& weightScale, char const* name)
{
    TORCH_CHECK(act.dim() == 2 && weight.dim() == 2 && act.size(1) == weight.size(1), name,
        ": act [M, K] and weight [N, K] with the same K");
    TORCH_CHECK(
        act.scalar_type() == at::ScalarType::Float8_e4m3fn && weight.scalar_type() == at::ScalarType::Float8_e4m3fn,
        name, ": act and weight must be float8_e4m3fn");
    TORCH_CHECK(
        act.is_contiguous() && weight.is_contiguous() && actScale.is_contiguous() && weightScale.is_contiguous(), name,
        ": operands must be contiguous");
    TORCH_CHECK(
        act.device() == weight.device() && act.device() == actScale.device() && act.device() == weightScale.device(),
        name, ": operands must be on one device");
    TORCH_CHECK(act.size(1) % 32 == 0, name, ": K must be a multiple of 32, got ", act.size(1));
}

void launchGemm(at::Tensor const& act, at::Tensor const& actScale, at::Tensor const& weight,
    at::Tensor const& weightScale, std::vector<int64_t> const& algo, at::Tensor& out)
{
    int64_t const m = act.size(0);
    int64_t const n = weight.size(0);
    int64_t const k = act.size(1);
    if (m == 0 || n == 0)
    {
        return;
    }
    cublasLtMatmulAlgo_t const pinned = buildAlgo(algo);
    Descs const descs(m, n, k, weightScale.data_ptr(), actScale.data_ptr());
    auto const stream = at::cuda::getCurrentCUDAStream(act.get_device()).stream();
    auto const& ws = workspace(act.device(), stream);
    float const alpha = 1.f;
    float const beta = 0.f;
    M31_CUBLASLT_CHECK(
        cublasLtMatmul(blasLtHandle(), descs.op, &alpha, weight.data_ptr(), descs.a, act.data_ptr(), descs.b, &beta,
            out.data_ptr(), descs.d, out.data_ptr(), descs.d, &pinned, ws.data_ptr(), kWorkspaceBytes, stream));
}

} // namespace

//! The pinned-algorithm MXFP8 GEMM. act [M, K] / weight [N, K] e4m3 (row-major, K contiguous), act_scale /
//! weight_scale their swizzled UE8M0 scales (mxfp8_quantize(x, True)), algo the nine recorded attributes.
//! \return out [M, N] bfloat16.
at::Tensor minimaxM31Mxfp8CublasLtGemm(at::Tensor const& act, at::Tensor const& actScale, at::Tensor const& weight,
    at::Tensor const& weightScale, std::vector<int64_t> const& algo)
{
    checkOperands(act, actScale, weight, weightScale, "minimax_m31_mxfp8_cublaslt_gemm");
    c10::cuda::CUDAGuard const deviceGuard{act.device()};
    auto out = torch::empty({act.size(0), weight.size(0)}, act.options().dtype(at::ScalarType::BFloat16));
    launchGemm(act, actScale, weight, weightScale, algo, out);
    return out;
}

//! minimaxM31Mxfp8CublasLtGemm writing into a caller's [M, N] bfloat16 buffer (TP2CTX_SP: o_proj straight into the
//! IPC partial slot). Same descriptors, algorithm and workspace, so bit for bit the allocating op's result. `out`
//! must be row-major contiguous (row stride N) and 16-byte aligned, as every cuBLASLt output.
void minimaxM31Mxfp8CublasLtGemmOut(at::Tensor const& act, at::Tensor const& actScale, at::Tensor const& weight,
    at::Tensor const& weightScale, std::vector<int64_t> const& algo, at::Tensor& out)
{
    char const* name = "minimax_m31_mxfp8_cublaslt_gemm_out";
    checkOperands(act, actScale, weight, weightScale, name);
    TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16 && out.is_contiguous() && out.dim() == 2
            && out.size(0) == act.size(0) && out.size(1) == weight.size(0) && out.device() == act.device(),
        name, ": out must be a contiguous [M, N] bfloat16 tensor on the operands' device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, name, ": out must be 16-byte aligned");
    c10::cuda::CUDAGuard const deviceGuard{act.device()};
    launchGemm(act, actScale, weight, weightScale, algo, out);
}

//! The workspace bytes a recorded configuration needs for (M, N, K), or -1 when it cannot serve that shape
//! (cublasLtMatmulAlgoCheck on the current device, the serving workspace limit included).
int64_t minimaxM31Mxfp8CublasLtAlgoCheck(int64_t m, int64_t n, int64_t k, std::vector<int64_t> const& algo)
{
    cublasLtMatmulAlgo_t const pinned = buildAlgo(algo);
    // cublasLtMatmulAlgoCheck rejects VEC32_UE8M0 operands whose scale pointers are NULL (CUBLAS_STATUS_INVALID_VALUE
    // on sm_103 and sm_107 alike), so the check sets them to valid, aligned device memory: the serving workspace. The
    // check never dereferences them.
    auto const device = c10::Device(c10::kCUDA, c10::cuda::current_device());
    auto const& ws = workspace(device, at::cuda::getCurrentCUDAStream(device.index()).stream());
    Descs const descs(m, n, k, ws.data_ptr(), ws.data_ptr());
    cublasLtMatmulHeuristicResult_t result{};
    cublasStatus_t const status
        = cublasLtMatmulAlgoCheck(blasLtHandle(), descs.op, descs.a, descs.b, descs.d, descs.d, &pinned, &result);
    if (status != CUBLAS_STATUS_SUCCESS || result.state != CUBLAS_STATUS_SUCCESS
        || result.workspaceSize > kWorkspaceBytes)
    {
        return -1;
    }
    return static_cast<int64_t>(result.workspaceSize);
}

//! The heuristic's first configuration for (M, N, K) under the serving workspace: the nine attributes, then its
//! workspace bytes. Offline only (recording configurations); serving never calls it.
std::vector<int64_t> minimaxM31Mxfp8CublasLtHeuristic(int64_t m, int64_t n, int64_t k)
{
    // Non-null scale pointers, as in minimaxM31Mxfp8CublasLtAlgoCheck (the descriptors must match the launch's).
    auto const device = c10::Device(c10::kCUDA, c10::cuda::current_device());
    auto const& ws = workspace(device, at::cuda::getCurrentCUDAStream(device.index()).stream());
    Descs const descs(m, n, k, ws.data_ptr(), ws.data_ptr());
    cublasLtMatmulPreference_t pref{nullptr};
    M31_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t const wsBytes = kWorkspaceBytes;
    M31_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof(wsBytes)));
    cublasLtMatmulHeuristicResult_t result{};
    int found = 0;
    cublasStatus_t const status = cublasLtMatmulAlgoGetHeuristic(
        blasLtHandle(), descs.op, descs.a, descs.b, descs.d, descs.d, pref, 1, &result, &found);
    cublasLtMatmulPreferenceDestroy(pref);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS && found > 0,
        "minimax_m31_mxfp8_cublaslt_heuristic: no configuration for M=", m, " N=", n, " K=", k);
    std::vector<int64_t> config;
    for (auto const attr : kAlgoAttrs)
    {
        uint64_t value = 0;
        size_t const size = algoAttrSize(result.algo, attr);
        size_t written = 0;
        M31_CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&result.algo, attr, &value, size, &written));
        config.push_back(static_cast<int64_t>(value));
    }
    config.push_back(static_cast<int64_t>(result.workspaceSize));
    return config;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m31_mxfp8_cublaslt_gemm(Tensor act, Tensor act_scale, Tensor weight, Tensor weight_scale, "
        "int[] algo) -> Tensor");
    m.def(
        "minimax_m31_mxfp8_cublaslt_gemm_out(Tensor act, Tensor act_scale, Tensor weight, Tensor weight_scale, "
        "int[] algo, Tensor(a!) out) -> ()");
    // No tensor arguments: catch-all kernels on the current device.
    m.def("minimax_m31_mxfp8_cublaslt_algo_check", &tensorrt_llm::torch_ext::minimaxM31Mxfp8CublasLtAlgoCheck);
    m.def("minimax_m31_mxfp8_cublaslt_heuristic", &tensorrt_llm::torch_ext::minimaxM31Mxfp8CublasLtHeuristic);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_mxfp8_cublaslt_gemm", &tensorrt_llm::torch_ext::minimaxM31Mxfp8CublasLtGemm);
    m.impl("minimax_m31_mxfp8_cublaslt_gemm_out", &tensorrt_llm::torch_ext::minimaxM31Mxfp8CublasLtGemmOut);
}
