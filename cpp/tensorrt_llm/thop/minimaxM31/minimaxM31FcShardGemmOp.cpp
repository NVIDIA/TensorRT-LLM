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

// FC_SHARD (MiniMax-M3.1 VR TP8): y[M, N] = x[M, K] @ w[N, K]^T in bf16 (fp32 accumulate) through cuBLASLt with a
// caller-given algo config (ALGO_ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID,
// INNER_SHAPE_ID, CLUSTER_SHAPE_ID). Pinning a column shard of the DSpark drafter's fc to the algo cuBLASLt picks for
// the full fc makes every output column the same sum of the same K-split partials as the full GEMM, so the shard is
// bit for bit the matching columns of torch.nn.functional.linear(x, w_full).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace
{

constexpr int64_t kWorkspaceBytes = 32LL << 20;

void checkLt(cublasStatus_t st, char const* what)
{
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "minimax_m31_fc_shard_gemm: ", what, " failed with cuBLAS status ",
        static_cast<int>(st));
}

struct DeviceState
{
    cublasLtHandle_t handle{nullptr};
    cublasLtMatmulDesc_t desc{nullptr};
    at::Tensor workspace;
    // (m, n, k, config) -> the initialized algo after a passing AlgoCheck
    std::map<std::tuple<int64_t, int64_t, int64_t, std::vector<int64_t>>, cublasLtMatmulAlgo_t> algos;
};

std::mutex gMutex;
std::map<int, DeviceState> gStates;

DeviceState& stateFor(at::Device const& device)
{
    auto& st = gStates[device.index()];
    if (st.handle == nullptr)
    {
        checkLt(cublasLtCreate(&st.handle), "cublasLtCreate");
        checkLt(cublasLtMatmulDescCreate(&st.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
        cublasOperation_t const ta = CUBLAS_OP_T;
        cublasOperation_t const tb = CUBLAS_OP_N;
        checkLt(cublasLtMatmulDescSetAttribute(st.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
        checkLt(cublasLtMatmulDescSetAttribute(st.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
        st.workspace = at::empty({kWorkspaceBytes}, at::TensorOptions().dtype(at::kByte).device(device));
    }
    return st;
}

struct Layouts
{
    cublasLtMatrixLayout_t a{nullptr}, b{nullptr}, c{nullptr};

    Layouts(int64_t m, int64_t n, int64_t k)
    {
        // op(A) = W (row-major [m, k] = column-major k x m, transposed); B = x (row-major [n, k] = column-major
        // k x n); D = column-major m x n = row-major [n, m].
        checkLt(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, m, k), "layout A");
        checkLt(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, n, k), "layout B");
        checkLt(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, m, n, m), "layout D");
    }

    ~Layouts()
    {
        cublasLtMatrixLayoutDestroy(a);
        cublasLtMatrixLayoutDestroy(b);
        cublasLtMatrixLayoutDestroy(c);
    }
};

template <typename T>
void setCfg(cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr, int64_t value, char const* what)
{
    T const v = static_cast<T>(value);
    checkLt(cublasLtMatmulAlgoConfigSetAttribute(&algo, attr, &v, sizeof(v)), what);
}

} // namespace

namespace torch_ext
{

//! x: bf16 [M, K] contiguous; w: bf16 [N, K] contiguous (a row slice of the full weight); cfg: the 9 config values
//! above. Returns bf16 [M, N]. Throws if cuBLASLt rejects the config for this shape (the caller keeps the stock fc).
at::Tensor minimaxM31FcShardGemm(at::Tensor const& x, at::Tensor const& w, std::vector<int64_t> const& cfg)
{
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.get_device() == w.get_device(),
        "minimax_m31_fc_shard_gemm: x and w must be CUDA tensors on one device");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "minimax_m31_fc_shard_gemm: bf16 only");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.is_contiguous() && w.is_contiguous() && x.size(1) == w.size(1),
        "minimax_m31_fc_shard_gemm: x [M, K] and w [N, K], contiguous");
    TORCH_CHECK(cfg.size() == 9, "minimax_m31_fc_shard_gemm: cfg must hold 9 values");
    c10::cuda::CUDAGuard guard(x.device());
    int64_t const n = x.size(0);
    int64_t const k = x.size(1);
    int64_t const m = w.size(0);
    auto y = at::empty({n, m}, x.options());
    if (n == 0)
    {
        return y;
    }
    std::lock_guard<std::mutex> lock(gMutex);
    auto& st = stateFor(x.device());
    Layouts lay(m, n, k);
    auto key = std::make_tuple(m, n, k, cfg);
    auto it = st.algos.find(key);
    if (it == st.algos.end())
    {
        cublasLtMatmulAlgo_t algo;
        checkLt(cublasLtMatmulAlgoInit(st.handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                    CUDA_R_16BF, static_cast<int>(cfg[0]), &algo),
            "cublasLtMatmulAlgoInit");
        setCfg<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, cfg[1], "TILE_ID");
        setCfg<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, cfg[2], "SPLITK_NUM");
        setCfg<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, cfg[3], "REDUCTION_SCHEME");
        setCfg<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, cfg[4], "CTA_SWIZZLING");
        setCfg<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, cfg[5], "CUSTOM_OPTION");
        setCfg<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, cfg[6], "STAGES_ID");
        setCfg<uint16_t>(algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, cfg[7], "INNER_SHAPE_ID");
        setCfg<uint16_t>(algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, cfg[8], "CLUSTER_SHAPE_ID");
        cublasLtMatmulHeuristicResult_t result{};
        checkLt(cublasLtMatmulAlgoCheck(st.handle, st.desc, lay.a, lay.b, lay.c, lay.c, &algo, &result),
            "cublasLtMatmulAlgoCheck");
        TORCH_CHECK(result.workspaceSize <= static_cast<size_t>(kWorkspaceBytes),
            "minimax_m31_fc_shard_gemm: the pinned algo needs more workspace than ", kWorkspaceBytes);
        it = st.algos.emplace(key, algo).first;
    }
    float const alpha = 1.0f;
    float const beta = 0.0f;
    checkLt(cublasLtMatmul(st.handle, st.desc, &alpha, w.data_ptr(), lay.a, x.data_ptr(), lay.b, &beta, y.data_ptr(),
                lay.c, y.data_ptr(), lay.c, &it->second, st.workspace.data_ptr(), kWorkspaceBytes,
                at::cuda::getCurrentCUDAStream(x.get_device()).stream()),
        "cublasLtMatmul");
    return y;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m31_fc_shard_gemm(Tensor x, Tensor w, int[] cfg) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_fc_shard_gemm", &torch_ext::minimaxM31FcShardGemm);
}
