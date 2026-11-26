/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/csa2SparseContext/csa2SparseContext.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <optional>
#include <torch/library.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace csa2 = kernels::csa2SparseContext;

void checkDeviceScalars(at::Tensor const& tensor, int64_t count, at::Device device, char const* name)
{
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device && tensor.scalar_type() == at::kFloat
            && tensor.numel() >= count && tensor.is_contiguous(),
        "csa2_sparse_context: ", name, " must be a contiguous CUDA float32 tensor of at least ", count,
        " elements on the query device");
}

//! FP8 CSA2 sparse context attention over the shared KV bank; see kernels/csa2SparseContext/csa2SparseContext.h.
//! q is E4M3 [M, 64, 512], or BF16 [M, 64, 512] with q_scale (FP32 [1]) to quantize it with. Writes output
//! (BF16 [M, 64, 512]) on the current stream; allocates nothing and needs no workspace. With output_scale, output is
//! instead the o_a projection's input that fused_inv_rope_fp8_quant_vllm_port (GPT-J pairs) makes of the BF16 output:
//! E4M3 [groups, M, heads_per_group * 512] and FP32 output_scale [groups, heads_per_group * 4, M rounded up to 4],
//! rotated back at positions (int32 or int64 [M]) with rope_cos_sin (FP32, 32 cos then 32 sin per position).
void csa2SparseContext(at::Tensor const& bank, at::Tensor const& q, std::optional<at::Tensor> const& qScale,
    at::Tensor const& indices, at::Tensor const& lens, at::Tensor const& bmm1Scale, at::Tensor const& bmm2Scale,
    std::optional<at::Tensor> const& attnSinks, at::Tensor& output, std::optional<at::Tensor> const& outputScale,
    std::optional<at::Tensor> const& positions, std::optional<at::Tensor> const& ropeCosSin)
{
    bool const qBf16 = q.scalar_type() == at::kBFloat16;
    TORCH_CHECK(q.is_cuda() && (qBf16 || q.scalar_type() == at::kFloat8_e4m3fn) && q.dim() == 3
            && q.size(1) == csa2::kNumHeads && q.size(2) == csa2::kHeadDim && q.stride(2) == 1,
        "csa2_sparse_context: q must be CUDA BF16 or float8_e4m3fn [num_tokens, 64, 512] with unit channel stride");
    bool const fp8Out = outputScale.has_value();
    if (fp8Out)
    {
        TORCH_CHECK(output.is_cuda() && output.device() == q.device() && output.scalar_type() == at::kFloat8_e4m3fn
                && output.dim() == 3 && output.size(1) == q.size(0) && output.size(2) % csa2::kHeadDim == 0
                && output.size(0) * output.size(2) == csa2::kNumHeads * csa2::kHeadDim && output.stride(2) == 1,
            "csa2_sparse_context: an FP8 output must be CUDA float8_e4m3fn [groups, num_tokens, heads_per_group * "
            "512] with unit channel stride");
        TORCH_CHECK(outputScale->is_cuda() && outputScale->device() == q.device()
                && outputScale->scalar_type() == at::kFloat && outputScale->dim() == 3
                && outputScale->size(0) == output.size(0) && outputScale->size(1) == output.size(2) / 128
                && outputScale->size(2) == (q.size(0) + 3) / 4 * 4 && outputScale->stride(2) == 1,
            "csa2_sparse_context: output_scale must be CUDA float32 [groups, heads_per_group * 4, num_tokens rounded "
            "up to 4] with unit token stride");
        TORCH_CHECK(positions.has_value() && positions->is_cuda() && positions->device() == q.device()
                && (positions->scalar_type() == at::kInt || positions->scalar_type() == at::kLong)
                && positions->dim() == 1 && positions->size(0) >= q.size(0) && positions->is_contiguous(),
            "csa2_sparse_context: an FP8 output needs contiguous CUDA int32 or int64 positions [>= num_tokens]");
        TORCH_CHECK(ropeCosSin.has_value() && ropeCosSin->is_cuda() && ropeCosSin->device() == q.device()
                && ropeCosSin->scalar_type() == at::kFloat && ropeCosSin->numel() % 64 == 0
                && ropeCosSin->is_contiguous(),
            "csa2_sparse_context: an FP8 output needs a contiguous CUDA float32 rope_cos_sin of 64 values per "
            "position");
    }
    else
    {
        TORCH_CHECK(output.is_cuda() && output.device() == q.device() && output.scalar_type() == at::kBFloat16
                && output.sizes() == q.sizes() && output.stride(2) == 1,
            "csa2_sparse_context: output must be CUDA BF16 shaped like q with unit channel stride");
        TORCH_CHECK(!positions.has_value() && !ropeCosSin.has_value(),
            "csa2_sparse_context: positions and rope_cos_sin apply to an FP8 output only");
    }
    int64_t const numTokens = q.size(0);
    TORCH_CHECK(numTokens < (int64_t{1} << 31), "csa2_sparse_context: too many query tokens");
    if (numTokens == 0)
    {
        return;
    }
    c10::cuda::CUDAGuard const deviceGuard(q.device());

    TORCH_CHECK(bank.is_cuda() && bank.device() == q.device()
            && (bank.scalar_type() == at::kFloat8_e4m3fn || bank.scalar_type() == at::kByte) && bank.dim() == 2
            && bank.size(1) == csa2::kHeadDim && bank.stride(1) == 1
            && (bank.size(0) <= 1 || bank.stride(0) == csa2::kHeadDim),
        "csa2_sparse_context: bank must be a CUDA float8_e4m3fn (or uint8) [rows, 512] tensor with contiguous rows");
    TORCH_CHECK(indices.is_cuda() && indices.device() == q.device() && indices.scalar_type() == at::kInt
            && indices.dim() == 2 && indices.size(0) >= numTokens && indices.size(1) > 0
            && indices.size(1) < (int64_t{1} << 31) && (indices.size(1) == 1 || indices.stride(1) == 1),
        "csa2_sparse_context: indices must be CUDA int32 [>= num_tokens, keys] with unit column stride");
    TORCH_CHECK(lens.is_cuda() && lens.device() == q.device() && lens.scalar_type() == at::kInt && lens.dim() == 1
            && lens.size(0) >= numTokens && (lens.size(0) <= 1 || lens.stride(0) == 1),
        "csa2_sparse_context: lens must be a contiguous CUDA int32 [>= num_tokens] tensor");
    checkDeviceScalars(bmm1Scale, 2, q.device(), "bmm1_scale");
    checkDeviceScalars(bmm2Scale, 1, q.device(), "bmm2_scale");

    csa2::Csa2SparseContextParams params{};
    params.numTokens = static_cast<int>(numTokens);
    params.q = q.data_ptr();
    params.qBf16 = qBf16;
    params.qTokenStride = q.stride(0);
    params.qHeadStride = q.stride(1);
    if (qBf16)
    {
        TORCH_CHECK(qScale.has_value(), "csa2_sparse_context: BF16 q needs q_scale");
        checkDeviceScalars(*qScale, 1, q.device(), "q_scale");
        params.qScale = qScale->data_ptr<float>();
    }
    else
    {
        TORCH_CHECK(!qScale.has_value(), "csa2_sparse_context: q_scale applies to BF16 q only");
    }
    params.bank = bank.data_ptr();
    params.bankRows = bank.size(0);
    params.indices = indices.data_ptr<int32_t>();
    params.indexStride = indices.size(0) > 1 ? indices.stride(0) : indices.size(1);
    params.maxKeys = static_cast<int>(indices.size(1));
    params.lens = lens.data_ptr<int32_t>();
    params.bmm1Scale = bmm1Scale.data_ptr<float>();
    params.bmm2Scale = bmm2Scale.data_ptr<float>();
    if (attnSinks.has_value())
    {
        TORCH_CHECK(attnSinks->is_cuda() && attnSinks->device() == q.device() && attnSinks->scalar_type() == at::kFloat
                && attnSinks->numel() == csa2::kNumHeads && attnSinks->is_contiguous(),
            "csa2_sparse_context: attn_sinks must be a contiguous CUDA float32 [64] tensor");
        params.attnSinks = attnSinks->data_ptr<float>();
    }
    if (fp8Out)
    {
        params.fp8Out = output.data_ptr();
        params.fp8GroupStride = output.stride(0);
        params.fp8TokenStride = output.stride(1);
        params.headsPerGroup = static_cast<int>(output.size(2) / csa2::kHeadDim);
        params.fp8Scale = outputScale->data_ptr<float>();
        params.fp8ScaleGroupStride = outputScale->stride(0);
        params.fp8ScaleBlockStride = outputScale->stride(1);
        params.fp8ScaleRows = static_cast<int>(outputScale->size(2));
        params.positions = positions->data_ptr();
        params.positions64 = positions->scalar_type() == at::kLong;
        params.ropeCosSin = ropeCosSin->data_ptr<float>();
    }
    else
    {
        params.out = output.data_ptr();
        params.outTokenStride = output.stride(0);
        params.outHeadStride = output.stride(1);
    }

    csa2::run(params, at::cuda::getCurrentCUDAStream(q.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//! Whether csa2_sparse_context can run on the current device.
bool csa2SparseContextSupported()
{
    return csa2::isSupported();
}

} // namespace

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "csa2_sparse_context(Tensor bank, Tensor q, Tensor? q_scale, Tensor indices, Tensor lens, "
        "Tensor bmm1_scale, Tensor bmm2_scale, Tensor? attn_sinks, Tensor(a!) output, Tensor(b!)? output_scale=None, "
        "Tensor? positions=None, Tensor? rope_cos_sin=None) -> ()");
    m.def("csa2_sparse_context_supported() -> bool", &tensorrt_llm::torch_ext::csa2SparseContextSupported);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("csa2_sparse_context", &tensorrt_llm::torch_ext::csa2SparseContext);
}
