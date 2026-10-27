// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <map>
#include <memory>
#include <mutex>
#include <torch/library.h>

namespace
{
void check(cublasStatus_t status)
{
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "M31 router cuBLASLt status ", int(status));
}

struct Plan
{
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t weight{}, input{}, output{};
    cublasLtMatmulAlgo_t algo{};
    bool supported{false};

    ~Plan()
    {
        if (output)
            cublasLtMatrixLayoutDestroy(output);
        if (input)
            cublasLtMatrixLayoutDestroy(input);
        if (weight)
            cublasLtMatrixLayoutDestroy(weight);
        if (op)
            cublasLtMatmulDescDestroy(op);
        if (handle)
            cublasLtDestroy(handle);
    }
};

std::mutex mutex;
std::map<std::pair<int64_t, int64_t>, std::unique_ptr<Plan>> plans;

int config(cublasLtMatmulAlgo_t const& algo, cublasLtMatmulAlgoConfigAttributes_t attr)
{
    int value{};
    size_t written{};
    check(cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &value, sizeof(value), &written));
    TORCH_CHECK(written == sizeof(value), "Unexpected cuBLASLt tactic attribute size");
    return value;
}

Plan& prepare(int64_t rows, int64_t device)
{
    auto key = std::make_pair(device, rows);
    auto found = plans.find(key);
    if (found != plans.end())
        return *found->second;
    c10::cuda::CUDAGuard guard{c10::Device(c10::kCUDA, device)};
    auto p = std::make_unique<Plan>();
    check(cublasLtCreate(&p->handle));
    check(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    cublasOperation_t transpose = CUBLAS_OP_T;
    check(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
    check(cublasLtMatrixLayoutCreate(&p->weight, CUDA_R_32F, 6144, 128, 6144));
    check(cublasLtMatrixLayoutCreate(&p->input, CUDA_R_32F, 6144, rows, 6144));
    check(cublasLtMatrixLayoutCreate(&p->output, CUDA_R_32F, 128, rows, 128));
    cublasLtMatmulPreference_t preference{};
    check(cublasLtMatmulPreferenceCreate(&preference));
    uint64_t workspace = 64ULL << 20;
    auto status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace));
    cublasLtMatmulHeuristicResult_t candidates[64]{};
    int count{};
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmulAlgoGetHeuristic(
            p->handle, p->op, p->weight, p->input, p->output, p->output, preference, 64, candidates, &count);
    cublasLtMatmulPreferenceDestroy(preference);
    check(status);
    for (int i = 0; i < count; ++i)
    {
        auto const& candidate = candidates[i];
        auto const& a = candidate.algo;
        if (candidate.state == CUBLAS_STATUS_SUCCESS && candidate.workspaceSize == 0
            && config(a, CUBLASLT_ALGO_CONFIG_ID) == 66
            && config(a, CUBLASLT_ALGO_CONFIG_TILE_ID) == (rows <= 4096 ? 15 : 321)
            && config(a, CUBLASLT_ALGO_CONFIG_STAGES_ID) == 34 && config(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM) == 1
            && config(a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME) == 0
            && config(a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING) == 0
            && config(a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION) == (rows <= 4096 ? 3 : 1))
        {
            p->algo = a;
            p->supported = true;
            break;
        }
    }
    return *plans.emplace(key, std::move(p)).first->second;
}

bool supported(int64_t rows, int64_t device)
{
    std::lock_guard<std::mutex> lock(mutex);
    return prepare(rows, device).supported;
}

at::Tensor router(at::Tensor const& x, at::Tensor const& w)
{
    TORCH_CHECK(x.is_cuda() && w.device() == x.device(), "router tensors must share a CUDA device");
    TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat && x.dim() == 2 && w.dim() == 2
            && x.size(1) == 6144 && w.size(0) == 128 && w.size(1) == 6144 && x.is_contiguous() && w.is_contiguous(),
        "unsupported router layout");
    c10::cuda::CUDAGuard guard(x.device());
    auto output = at::empty({x.size(0), 128}, x.options());
    std::lock_guard<std::mutex> lock(mutex);
    auto found = plans.find({x.get_device(), x.size(0)});
    TORCH_CHECK(found != plans.end() && found->second->supported, "router plan must be prepared before capture");
    auto& p = *found->second;
    float alpha = 1.0F, beta = 0.0F;
    check(
        cublasLtMatmul(p.handle, p.op, &alpha, w.data_ptr(), p.weight, x.data_ptr(), p.input, &beta, output.data_ptr(),
            p.output, output.data_ptr(), p.output, &p.algo, nullptr, 0, at::cuda::getCurrentCUDAStream()));
    return output;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("m31_router_half_pin_prepare(int rows, int device) -> bool", &supported);
    m.def("m31_router_half_pin(Tensor x, Tensor weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("m31_router_half_pin", &router);
}
