/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.h"
#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmIntraNodeReduce.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <limits>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{
namespace glm = kernels::glm_moe_dsa;
using Pair = std::tuple<at::Tensor, at::Tensor>;
using Quad = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

void bf16(at::Tensor const& t, at::Device device, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.device() == device && t.scalar_type() == at::kBFloat16 && t.is_contiguous()
            && reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0,
        name, " must be contiguous, 16-byte aligned BF16 on the input CUDA device");
}

void validate(at::Tensor const& residual, at::Tensor const& gamma, double epsilon, at::Tensor const& table,
    at::Tensor const& flags, at::Tensor const& layout, int64_t rank, int64_t nranks, int64_t bufferBytes, bool finalize)
{
    bf16(residual, residual.device(), "residual");
    TORCH_CHECK(residual.dim() == 2 && residual.size(1) == 6144 && residual.size(0) >= 1 && residual.size(0) <= 24,
        "GLM IPC requires BF16 [1..24, 6144] rows");
    bf16(gamma, residual.device(), "gamma");
    TORCH_CHECK(gamma.dim() == 1 && gamma.numel() == 6144, "gamma must have 6144 elements");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0 && epsilon <= std::numeric_limits<float>::max(),
        "invalid RMSNorm epsilon");
    TORCH_CHECK(nranks == 4 && rank >= 0 && rank < nranks, "GLM IPC requires a four-rank in-node TP group");
    TORCH_CHECK(table.device() == residual.device() && table.scalar_type() == at::kLong && table.dim() == 1
            && table.numel() == 3 * nranks + 2 && table.is_contiguous(),
        "expected the bound IPC pointer table");
    TORCH_CHECK(flags.device() == residual.device() && flags.scalar_type() == at::kInt && flags.dim() == 1
            && flags.numel() == 3 && flags.is_contiguous(),
        "expected bound IPC flags[3]");
    TORCH_CHECK(layout.device() == residual.device() && layout.scalar_type() == at::kLong && layout.dim() == 1
            && layout.numel() == 2 && layout.is_contiguous(),
        "expected bound IPC layout[2]");
    TORCH_CHECK(
        bufferBytes > 0 && bufferBytes % 16 == 0 && residual.numel() * residual.element_size() * nranks <= bufferBytes,
        "rows exceed the bound IPC Lamport buffer capacity");
    TORCH_CHECK(glm::intraNodeReduceSupported(residual.size(0), residual.size(1), nranks, finalize),
        "unsupported GLM Rubin IPC launch");
}

glm::IntraNodeReduceParams paramsFor(at::Tensor const& residual, at::Tensor const& gamma, double epsilon,
    at::Tensor const& table, at::Tensor& flags, at::Tensor& layout, int64_t rank, at::Tensor& normOut,
    at::Tensor& residualOut)
{
    glm::IntraNodeReduceParams p{};
    p.workspace = reinterpret_cast<void* const*>(table.const_data_ptr());
    // These are the actual mutated Tensor allocations, not aliases of the pointer table.
    p.flags = flags.mutable_data_ptr<int32_t>();
    p.layout = layout.mutable_data_ptr<int64_t>();
    p.residual_in = residual.const_data_ptr();
    p.rms_gamma = gamma.const_data_ptr();
    p.residual_out = residualOut.mutable_data_ptr();
    p.norm_out = normOut.mutable_data_ptr();
    p.rank = static_cast<int>(rank);
    p.nranks = 4;
    p.size = static_cast<int>(residual.numel());
    p.hidden_dim = 6144;
    p.rms_eps = static_cast<float>(epsilon);
    p.stream = at::cuda::getCurrentCUDAStream(residual.get_device());
    return p;
}
} // namespace

Pair glmIntraNodeReduce(at::Tensor const& input, at::Tensor const& residual, at::Tensor const& gamma, double epsilon,
    at::Tensor const& table, at::Tensor& flags, at::Tensor& layout, int64_t rank, int64_t nranks, int64_t bufferBytes,
    bool triggerCompletionAtEnd)
{
    c10::cuda::CUDAGuard guard(residual.device());
    validate(residual, gamma, epsilon, table, flags, layout, rank, nranks, bufferBytes, false);
    bf16(input, residual.device(), "input");
    TORCH_CHECK(input.sizes() == residual.sizes(), "input and residual shapes must match");
    auto normOut = at::empty_like(residual);
    auto residualOut = at::empty_like(residual);
    auto p = paramsFor(residual, gamma, epsilon, table, flags, layout, rank, normOut, residualOut);
    p.allreduce_in = input.const_data_ptr();
    p.trigger_completion_at_end = triggerCompletionAtEnd;
    glm::intraNodeReduce(p);
    return {normOut, residualOut};
}

Pair glmIntraNodeMoeFinalize(at::Tensor const& fc2, at::Tensor const& weights, at::Tensor const& indices,
    at::Tensor const& shared, at::Tensor const& residual, at::Tensor const& gamma, double epsilon,
    at::Tensor const& table, at::Tensor& flags, at::Tensor& layout, int64_t rank, int64_t nranks, int64_t bufferBytes)
{
    c10::cuda::CUDAGuard guard(residual.device());
    validate(residual, gamma, epsilon, table, flags, layout, rank, nranks, bufferBytes, true);
    bf16(fc2, residual.device(), "fc2");
    bf16(shared, residual.device(), "shared");
    TORCH_CHECK(fc2.dim() == 2 && fc2.size(1) == 6144 && shared.sizes() == residual.sizes(),
        "FC2 must have unpadded hidden6144; shared output must match residual");
    TORCH_CHECK(indices.device() == residual.device() && indices.scalar_type() == at::kInt && indices.is_contiguous()
            && indices.dim() == 2 && indices.size(0) == residual.size(0) && indices.size(1) >= 1
            && indices.size(1) <= 8,
        "indices must be int32 [rows, 1..8]");
    TORCH_CHECK(weights.device() == residual.device() && weights.scalar_type() == at::kBFloat16
            && weights.is_contiguous() && weights.sizes() == indices.sizes(),
        "weights must be BF16 matching indices");
    auto normOut = at::empty_like(residual);
    auto residualOut = at::empty_like(residual);
    auto p = paramsFor(residual, gamma, epsilon, table, flags, layout, rank, normOut, residualOut);
    p.allreduce_in = fc2.const_data_ptr();
    p.expert_scale_factor = weights.const_data_ptr();
    p.expanded_idx_to_permuted_idx = indices.const_data_ptr<int32_t>();
    p.shared_expert_output = shared.const_data_ptr();
    p.top_k = static_cast<int>(indices.size(1));
    glm::intraNodeMoeFinalize(p);
    return {normOut, residualOut};
}

// The table must be bound once to these real flags/layout allocations by _ipc_workspace.
// The shared FP4 kernel accesses the same state indirectly via the table's last two pointers.
Quad glmPreMoeFp4Linear(at::Tensor const& input, at::Tensor const& residual, at::Tensor const& gamma, double epsilon,
    at::Tensor const& scale, at::Tensor const& table, at::Tensor& flags, at::Tensor& layout, int64_t rank,
    int64_t nranks, int64_t bufferBytes)
{
    c10::cuda::CUDAGuard guard(residual.device());
    validate(residual, gamma, epsilon, table, flags, layout, rank, nranks, bufferBytes, false);
    bf16(input, residual.device(), "input");
    TORCH_CHECK(input.sizes() == residual.sizes(), "input and residual shapes must match");
    TORCH_CHECK(scale.device() == residual.device() && scale.scalar_type() == at::kFloat && scale.numel() == 1
            && scale.is_contiguous(),
        "scale must be one contiguous CUDA FP32 value");
    auto normOut = at::empty_like(input);
    auto residualOut = at::empty_like(residual);
    auto packed = at::empty({input.size(0), 3072}, input.options().dtype(at::kByte));
    auto scales = at::empty({input.size(0), 384}, input.options().dtype(at::kByte));
    kernels::ar_fusion::AllReduceFusionParams p{};
    p.nranks = 4;
    p.rank = static_cast<int>(rank);
    p.dtype = DataType::kBF16;
    p.size = static_cast<int>(input.numel());
    p.hidden_dim = 6144;
    p.workspace = reinterpret_cast<void**>(const_cast<void*>(table.const_data_ptr()));
    p.allreduce_in = const_cast<void*>(input.const_data_ptr());
    p.residual_in = const_cast<void*>(residual.const_data_ptr());
    p.rms_gamma = const_cast<void*>(gamma.const_data_ptr());
    p.scale_factor = const_cast<float*>(scale.const_data_ptr<float>());
    p.norm_out = normOut.mutable_data_ptr();
    p.residual_out = residualOut.mutable_data_ptr();
    p.quant_out = packed.mutable_data_ptr();
    p.scale_out = scales.mutable_data_ptr();
    p.rms_eps = static_cast<float>(epsilon);
    p.use_oneshot = true;
    p.layout = QuantizationSFLayout::LINEAR;
    p.pattern = kernels::ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant;
    p.trigger_completion_at_end = false;
    p.stream = at::cuda::getCurrentCUDAStream(input.get_device());
    kernels::ar_fusion::allreduce_fusion_op(p);
    return {normOut, residualOut, packed, scales};
}

bool glmIntraNodeReduceSupported(int64_t rows, int64_t hidden, int64_t nranks, bool finalize)
{
    return rows >= 1 && rows <= 24 && hidden == 6144 && nranks == 4
        && glm::intraNodeReduceSupported(static_cast<int>(rows), 6144, 4, finalize);
}

void rejectGlmIpcFunctionalization(c10::OperatorHandle const&, torch::jit::Stack*)
{
    TORCH_CHECK(false,
        "GLM IPC collectives support eager and ordinary CUDA graphs only: "
        "functionalization cannot clone the peer allocations referenced by the IPC pointer table");
}
} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm_intra_node_reduce(Tensor input, Tensor residual, Tensor gamma, float epsilon, Tensor table, "
        "Tensor(a!) flags, Tensor(b!) layout, int rank, int nranks, int buffer_bytes, bool trigger_completion_at_end) "
        "-> (Tensor, Tensor)");
    m.def(
        "glm_intra_node_moe_finalize(Tensor fc2, Tensor weights, Tensor indices, Tensor shared, Tensor residual, "
        "Tensor gamma, float epsilon, Tensor table, Tensor(a!) flags, Tensor(b!) layout, int rank, int nranks, "
        "int buffer_bytes) -> (Tensor, Tensor)");
    m.def(
        "glm_pre_moe_fp4_linear(Tensor input, Tensor residual, Tensor gamma, float epsilon, Tensor scale, Tensor "
        "table, "
        "Tensor(a!) flags, Tensor(b!) layout, int rank, int nranks, int buffer_bytes) -> (Tensor, Tensor, Tensor, "
        "Tensor)");
    m.def("glm_intra_node_reduce_supported", &tensorrt_llm::torch_ext::glmIntraNodeReduceSupported);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_intra_node_reduce", &tensorrt_llm::torch_ext::glmIntraNodeReduce);
    m.impl("glm_intra_node_moe_finalize", &tensorrt_llm::torch_ext::glmIntraNodeMoeFinalize);
    m.impl("glm_pre_moe_fp4_linear", &tensorrt_llm::torch_ext::glmPreMoeFp4Linear);
}

TORCH_LIBRARY_IMPL(trtllm, Functionalize, m)
{
    m.impl("glm_intra_node_reduce",
        torch::CppFunction::makeFromBoxedFunction<&tensorrt_llm::torch_ext::rejectGlmIpcFunctionalization>());
    m.impl("glm_intra_node_moe_finalize",
        torch::CppFunction::makeFromBoxedFunction<&tensorrt_llm::torch_ext::rejectGlmIpcFunctionalization>());
    m.impl("glm_pre_moe_fp4_linear",
        torch::CppFunction::makeFromBoxedFunction<&tensorrt_llm::torch_ext::rejectGlmIpcFunctionalization>());
}
