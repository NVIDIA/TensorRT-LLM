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
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/csa2SparseDecode/csa2SparseDecode.h"

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

namespace csa2 = kernels::csa2SparseDecode;

bool isUnitInnerStride(at::Tensor const& tensor)
{
    return tensor.size(-1) <= 1 || tensor.stride(-1) == 1;
}

void checkPackedPool(at::Tensor const& pool, int64_t rowBytes, char const* name)
{
    TORCH_CHECK(pool.is_cuda() && pool.scalar_type() == at::kByte && pool.dim() == 2 && pool.size(1) == rowBytes
            && pool.stride(1) == 1 && (pool.size(0) <= 1 || pool.stride(0) == rowBytes),
        "csa2_sparse_decode: ", name, " must be a CUDA uint8 [rows, ", rowBytes, "] pool with contiguous rows");
}

void checkTokenVector(at::Tensor const& tensor, int64_t numTokens, char const* name)
{
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == at::kLong && tensor.dim() == 1
            && tensor.size(0) == numTokens && (numTokens <= 1 || tensor.stride(0) == 1),
        "csa2_sparse_decode: ", name, " must be a contiguous CUDA int64 [num_tokens] tensor");
}

//! Fused CSA2 sparse decode attention; see kernels/csa2SparseDecode/csa2SparseDecode.h for the contract.
//! q is BF16 or E4M3 [M, 64, 512]. Writes output [M, 64, 512] BF16 and uses workspace as scratch; allocates nothing.
//! Launches on the current stream.
//! With output_scale, output is instead the grouped O-A projection's MXFP8 input that
//! projection.inverse_rope_mxfp8 makes of the BF16 output: E4M3 [8, M, 8 * 512] and UE8M0 output_scale
//! [8 * pad128(M) * 128] (CUTLASS 128x4 layout per group), rotated back at positions (int32 [M]) with rope_cos_sin
//! (FP32, 32 cos then 32 sin per position).
void csa2SparseDecode(at::Tensor const& q, at::Tensor const& swaPool, at::Tensor const& swaIndices,
    std::optional<at::Tensor> const& mainPool, std::optional<at::Tensor> const& mainIndices,
    std::optional<at::Tensor> const& mainPageTable, std::optional<at::Tensor> const& tokenRequests,
    std::optional<at::Tensor> const& tokenVisible, int64_t mainPageSize, int64_t mainMaxPositions,
    at::Tensor const& attnSinks, double softmaxScale, at::Tensor& output, at::Tensor& workspace,
    std::optional<at::Tensor> const& outputScale, std::optional<at::Tensor> const& positions,
    std::optional<at::Tensor> const& ropeCosSin)
{
    bool const qFp8 = q.scalar_type() == at::kFloat8_e4m3fn;
    TORCH_CHECK(q.is_cuda() && (qFp8 || q.scalar_type() == at::kBFloat16) && q.dim() == 3
            && q.size(1) == csa2::kNumHeads && q.size(2) == csa2::kHeadDim && q.stride(2) == 1,
        "csa2_sparse_decode: q must be CUDA BF16 or float8_e4m3fn [num_tokens, 64, 512] with unit channel stride");
    int64_t const numTokens = q.size(0);
    bool const fp8Out = outputScale.has_value();
    if (fp8Out)
    {
        TORCH_CHECK(output.is_cuda() && output.device() == q.device() && output.scalar_type() == at::kFloat8_e4m3fn
                && output.dim() == 3 && output.size(0) == csa2::kOutputGroups && output.size(1) == numTokens
                && output.size(2) == csa2::kNumHeads / csa2::kOutputGroups * csa2::kHeadDim && output.is_contiguous(),
            "csa2_sparse_decode: an MXFP8 output must be contiguous CUDA float8_e4m3fn [8, num_tokens, 4096]");
        TORCH_CHECK(outputScale->is_cuda() && outputScale->device() == q.device()
                && outputScale->scalar_type() == at::kByte && outputScale->is_contiguous()
                && outputScale->numel() == csa2::kOutputGroups * ((numTokens + 127) / 128 * 128) * 128,
            "csa2_sparse_decode: output_scale must be contiguous CUDA uint8 [8 * num_tokens rounded up to 128 * 128]");
        TORCH_CHECK(positions.has_value() && positions->is_cuda() && positions->device() == q.device()
                && positions->scalar_type() == at::kInt && positions->dim() == 1 && positions->size(0) == numTokens
                && positions->is_contiguous(),
            "csa2_sparse_decode: an MXFP8 output needs contiguous CUDA int32 positions [num_tokens]");
        TORCH_CHECK(ropeCosSin.has_value() && ropeCosSin->is_cuda() && ropeCosSin->device() == q.device()
                && ropeCosSin->scalar_type() == at::kFloat && ropeCosSin->numel() % 64 == 0
                && ropeCosSin->is_contiguous(),
            "csa2_sparse_decode: an MXFP8 output needs a contiguous CUDA float32 rope_cos_sin of 64 values per "
            "position");
    }
    else
    {
        TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16 && output.sizes() == q.sizes()
                && output.stride(2) == 1 && output.device() == q.device(),
            "csa2_sparse_decode: output must be CUDA BF16 [num_tokens, 64, 512] with unit channel stride");
        TORCH_CHECK(!positions.has_value() && !ropeCosSin.has_value(),
            "csa2_sparse_decode: positions and rope_cos_sin apply to an MXFP8 output only");
    }
    TORCH_CHECK(numTokens < (int64_t{1} << 30), "csa2_sparse_decode: too many query rows");
    if (numTokens == 0)
    {
        return;
    }
    c10::cuda::CUDAGuard const deviceGuard(q.device());
    TORCH_CHECK(csa2::isSupported(), "csa2_sparse_decode requires an SM100-family GPU and a build that includes it");

    checkPackedPool(swaPool, csa2::kSwaRowBytes, "swa_pool");
    TORCH_CHECK(swaPool.device() == q.device(), "csa2_sparse_decode: swa_pool must be on the query device");
    TORCH_CHECK(swaIndices.is_cuda() && swaIndices.device() == q.device() && swaIndices.dim() == 2
            && swaIndices.size(0) == numTokens && swaIndices.size(1) > 0 && isUnitInnerStride(swaIndices)
            && swaIndices.scalar_type() == at::kLong,
        "csa2_sparse_decode: swa_indices must be CUDA int64 [num_tokens, swa_topk] with unit column stride");

    csa2::Csa2SparseDecodeParams params{};
    params.numTokens = static_cast<int>(numTokens);
    params.q = q.data_ptr();
    params.qTokenStride = q.stride(0);
    params.qHeadStride = q.stride(1);
    params.qFp8 = qFp8;
    params.swaPool = swaPool.data_ptr();
    params.swaRows = swaPool.size(0);
    params.swaIndices = swaIndices.data_ptr<int64_t>();
    params.swaIndexStride = swaIndices.stride(0);
    params.swaTopk = static_cast<int>(swaIndices.size(1));

    int64_t const mainTopk = mainIndices.has_value() ? mainIndices->size(1) : 0;
    if (mainTopk > 0)
    {
        TORCH_CHECK(mainPool.has_value(), "csa2_sparse_decode: main_indices need a main_pool");
        checkPackedPool(*mainPool, csa2::kMainRowBytes, "main_pool");
        TORCH_CHECK(mainPool->device() == q.device(), "csa2_sparse_decode: main_pool must be on the query device");
        TORCH_CHECK(mainIndices->is_cuda() && mainIndices->device() == q.device() && mainIndices->dim() == 2
                && mainIndices->size(0) == numTokens && mainIndices->scalar_type() == at::kInt
                && isUnitInnerStride(*mainIndices),
            "csa2_sparse_decode: main_indices must be CUDA int32 [num_tokens, main_topk] with unit column stride");
        params.mainPool = mainPool->data_ptr();
        params.mainRows = mainPool->size(0);
        params.mainIndices = mainIndices->data_ptr<int32_t>();
        params.mainIndexStride = mainIndices->stride(0);
        params.mainTopk = static_cast<int>(mainTopk);
        if (mainPageTable.has_value())
        {
            TORCH_CHECK(tokenRequests.has_value() && tokenVisible.has_value(),
                "csa2_sparse_decode: a main page table needs token_requests and token_visible");
            TORCH_CHECK(mainPageTable->is_cuda() && mainPageTable->device() == q.device()
                    && mainPageTable->scalar_type() == at::kInt && mainPageTable->dim() == 2
                    && isUnitInnerStride(*mainPageTable),
                "csa2_sparse_decode: main_page_table must be CUDA int32 [requests, pages] with unit column stride");
            TORCH_CHECK(mainPageSize > 0 && (mainPageSize & (mainPageSize - 1)) == 0 && mainPageSize < (1 << 30),
                "csa2_sparse_decode: main_page_size must be a power of two");
            checkTokenVector(*tokenRequests, numTokens, "token_requests");
            checkTokenVector(*tokenVisible, numTokens, "token_visible");
            TORCH_CHECK(tokenRequests->device() == q.device() && tokenVisible->device() == q.device(),
                "csa2_sparse_decode: mapping tensors must be on the query device");
            TORCH_CHECK(mainPageTable->size(0) < (int64_t{1} << 31) && mainPageTable->size(1) < (int64_t{1} << 31),
                "csa2_sparse_decode: main_page_table is too large");
            params.mainPageTable = mainPageTable->data_ptr<int32_t>();
            params.mainPageTableRows = static_cast<int>(mainPageTable->size(0));
            params.mainPageTableCols = static_cast<int>(mainPageTable->size(1));
            params.mainPageTableStride = mainPageTable->size(0) > 1 ? mainPageTable->stride(0) : mainPageTable->size(1);
            int shift = 0;
            while ((int64_t{1} << shift) < mainPageSize)
            {
                ++shift;
            }
            params.mainPageShift = shift;
            params.mainMaxPositions = mainMaxPositions;
            params.tokenRequests = tokenRequests->data_ptr<int64_t>();
            params.tokenVisible = tokenVisible->data_ptr<int64_t>();
        }
    }
    TORCH_CHECK(attnSinks.is_cuda() && attnSinks.device() == q.device() && attnSinks.scalar_type() == at::kFloat
            && attnSinks.numel() == csa2::kNumHeads && attnSinks.is_contiguous(),
        "csa2_sparse_decode: attn_sinks must be a contiguous CUDA float32 [64] tensor");
    params.attnSinks = attnSinks.data_ptr<float>();
    params.softmaxScale = static_cast<float>(softmaxScale);
    if (fp8Out)
    {
        params.outFp8 = output.data_ptr();
        params.outScale = outputScale->data_ptr();
        params.positions = positions->data_ptr<int32_t>();
        params.ropeCosSin = ropeCosSin->data_ptr<float>();
    }
    else
    {
        params.out = output.data_ptr();
        params.outTokenStride = output.stride(0);
        params.outHeadStride = output.stride(1);
    }

    auto const schedule = csa2::makeSchedule(
        params.numTokens, params.swaTopk, params.mainTopk, tensorrt_llm::common::getMultiProcessorCount());
    TORCH_CHECK(workspace.is_cuda() && workspace.device() == q.device() && workspace.is_contiguous()
            && static_cast<size_t>(workspace.nbytes()) >= csa2::getWorkspaceSize(schedule),
        "csa2_sparse_decode: workspace must be a contiguous CUDA tensor of at least ", csa2::getWorkspaceSize(schedule),
        " bytes");
    params.workspace = workspace.data_ptr();
    params.enablePdl = tensorrt_llm::common::getEnvEnablePDL();

    csa2::run(params, schedule, at::cuda::getCurrentCUDAStream(q.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//! Workspace bytes that cover every csa2_sparse_decode call on the current device.
int64_t csa2SparseDecodeWorkspaceSize()
{
    return static_cast<int64_t>(csa2::getMaxWorkspaceSize(tensorrt_llm::common::getMultiProcessorCount()));
}

//! Whether csa2_sparse_decode can run on the current device.
bool csa2SparseDecodeSupported()
{
    return csa2::isSupported();
}

} // namespace

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "csa2_sparse_decode(Tensor q, Tensor swa_pool, Tensor swa_indices, Tensor? main_pool, "
        "Tensor? main_indices, Tensor? main_page_table, Tensor? token_requests, Tensor? token_visible, "
        "int main_page_size, int main_max_positions, Tensor attn_sinks, float softmax_scale, "
        "Tensor(a!) output, Tensor(b!) workspace, Tensor(c!)? output_scale=None, Tensor? positions=None, "
        "Tensor? rope_cos_sin=None) -> ()");
    m.def("csa2_sparse_decode_workspace_size() -> int", &tensorrt_llm::torch_ext::csa2SparseDecodeWorkspaceSize);
    m.def("csa2_sparse_decode_supported() -> bool", &tensorrt_llm::torch_ext::csa2SparseDecodeSupported);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("csa2_sparse_decode", &tensorrt_llm::torch_ext::csa2SparseDecode);
}
