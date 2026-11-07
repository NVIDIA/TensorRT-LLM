// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/csrc/glmMegakernel.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <tuple>
#include <vector>

namespace mega = TRTLLM_MV2_NS::kernels::glm_moe_dsa::megakernel;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{
constexpr int64_t kRows = 8;
constexpr int64_t kHidden = 6144;
constexpr int64_t kTopK = 2048;
constexpr int64_t kCtas = 212;
constexpr int64_t kMaxRoles = 16;
constexpr int64_t kWeightWords = kCtas * kMaxRoles * 3;

void checkMegakernelDevice(at::Tensor const& tensor)
{
    CHECK_TH_CUDA(tensor);
    auto const* props = at::cuda::getDeviceProperties(tensor.get_device());
    TORCH_CHECK(props->major == 10 && props->minor == 7 && props->multiProcessorCount == kCtas,
        "glm_megakernel requires SM107 with 212 SMs");
}

void checkPackedParams(at::Tensor const& params)
{
    CHECK_TH_CUDA(params);
    TORCH_CHECK(params.scalar_type() == at::kLong && params.dim() == 1 && params.is_contiguous()
            && params.numel() == mega::glm52PlainLayerNumParams() + kWeightWords / 2 + kWeightWords,
        "glm_megakernel: params must contain the int64 parameter block, role tables, and weight table");
}

void rejectMegakernelFunctionalization(c10::OperatorHandle const&, torch::jit::Stack*)
{
    TORCH_CHECK(
        false, "glm_megakernel uses packed device pointers and multicast state; functionalization is unsupported");
}
} // namespace

std::vector<int64_t> glmMegakernelInfo()
{
    return {mega::glm52PlainLayerMagic(), mega::glm52PlainLayerNumParams(), mega::glm52PlainLayerArenaBytes(),
        mega::glm52PlainLayerClearUnits(), mega::glm52PlainLayerSmemBytes(), mega::glm52PlainLayerNumCounters(),
        mega::glm52PlainLayerCandCap(), mega::glm52PlainLayerMaxKeys(), mega::glm52PlainLayerRouterLg(),
        mega::glm52PlainLayerLgBytes()};
}

at::Tensor glmMegakernelIndexerTmap(at::Tensor const& pool, int64_t rows)
{
    checkMegakernelDevice(pool);
    int64_t const availableBytes = pool.storage().nbytes() - pool.storage_offset() * pool.element_size();
    TORCH_CHECK(rows > 0 && rows <= availableBytes / 128,
        "glm_megakernel_indexer_tmap: pool storage must contain rows of 128 bytes");
    c10::cuda::CUDAGuard const deviceGuard(pool.device());
    auto out = at::empty({128}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    CUtensorMap tensorMap{};
    auto const result = mega::glm52PlainLayerIdxTmap(&tensorMap, pool.data_ptr(), rows);
    TORCH_CHECK(result == CUDA_SUCCESS, "glm_megakernel_indexer_tmap: CUDA driver error ", result);
    std::copy_n(reinterpret_cast<uint8_t const*>(&tensorMap), sizeof(tensorMap), out.data_ptr<uint8_t>());
    return out;
}

void glmMegakernelWeightTable(at::Tensor& params, int64_t offset)
{
    checkMegakernelDevice(params);
    checkPackedParams(params);
    TORCH_CHECK(offset == mega::glm52PlainLayerNumParams() + kWeightWords / 2,
        "glm_megakernel_weight_table: invalid weight table offset");
    c10::cuda::CUDAGuard const deviceGuard(params.device());
    C10_CUDA_CHECK(static_cast<cudaError_t>(mega::glm52PlainLayerInit()));
    C10_CUDA_CHECK(static_cast<cudaError_t>(mega::glm52PlainLayerWeightTable(
        params.data_ptr(), params.data_ptr<int64_t>() + offset, at::cuda::getCurrentCUDAStream(params.get_device()))));
}

std::tuple<at::Tensor, at::Tensor> glmMegakernel(at::Tensor const& params, at::Tensor const& x,
    at::Tensor const& residual, at::Tensor& topk, at::Tensor const& blockTable, at::Tensor const& kvLens,
    std::vector<at::Tensor> const& state, std::vector<at::Tensor> const& kvCache, bool pdl)
{
    checkMegakernelDevice(x);
    checkPackedParams(params);
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) == kRows && x.size(1) == kHidden
            && x.is_contiguous(),
        "glm_megakernel: x must be contiguous BF16 [8, 6144]");
    TORCH_CHECK(residual.device() == x.device() && residual.scalar_type() == x.scalar_type()
            && residual.sizes() == x.sizes() && residual.is_contiguous() && params.device() == x.device(),
        "glm_megakernel: residual and params must match the input device and residual shape");
    TORCH_CHECK(topk.device() == x.device() && topk.scalar_type() == at::kInt && topk.dim() == 2
            && topk.size(0) == kRows && topk.size(1) == kTopK && topk.is_contiguous(),
        "glm_megakernel: topk must be contiguous int32 [8, 2048]");
    TORCH_CHECK(blockTable.device() == x.device() && blockTable.scalar_type() == at::kInt && blockTable.dim() == 1
            && blockTable.numel() > 0 && blockTable.is_contiguous(),
        "glm_megakernel: block_table must be a contiguous int32 row");
    TORCH_CHECK(kvLens.device() == x.device() && kvLens.scalar_type() == at::kInt && kvLens.numel() == 1
            && kvLens.is_contiguous(),
        "glm_megakernel: kv_lens must contain one int32 request length");
    TORCH_CHECK(!state.empty() && !kvCache.empty(), "glm_megakernel: mutable state and KV pools must be supplied");
    for (auto const& tensor : state)
    {
        TORCH_CHECK(tensor.device() == x.device() || (tensor.is_cpu() && tensor.is_pinned()),
            "glm_megakernel: state must be on the input device or pinned host memory");
    }
    for (auto const& tensor : kvCache)
    {
        TORCH_CHECK(tensor.device() == x.device(), "glm_megakernel: KV pools must be on the input device");
    }
    c10::cuda::CUDAGuard const deviceGuard(x.device());
    auto xOut = at::empty_like(x);
    auto resOut = at::empty_like(residual);
    C10_CUDA_CHECK(static_cast<cudaError_t>(mega::glm52PlainLayerLaunch(params.data_ptr(), x.data_ptr(),
        residual.data_ptr(), xOut.data_ptr(), resOut.data_ptr(), topk.data_ptr(), blockTable.data_ptr(),
        kvLens.data_ptr(), at::cuda::getCurrentCUDAStream(x.get_device()), pdl)));
    return {xOut, resOut};
}
} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("glm_megakernel_info() -> int[]", &TRTLLM_MV2_NS::torch_ext::glmMegakernelInfo);
    m.def("glm_megakernel_indexer_tmap(Tensor pool, int rows) -> Tensor");
    m.def("glm_megakernel_weight_table(Tensor(a!) params, int offset) -> ()");
    m.def(
        "glm_megakernel(Tensor params, Tensor x, Tensor residual, Tensor(a!) topk, Tensor block_table, "
        "Tensor kv_lens, Tensor(b!)[] state, Tensor(c!)[] kv_cache, bool pdl) -> (Tensor, Tensor)");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("glm_megakernel_indexer_tmap", &TRTLLM_MV2_NS::torch_ext::glmMegakernelIndexerTmap);
    m.impl("glm_megakernel_weight_table", &TRTLLM_MV2_NS::torch_ext::glmMegakernelWeightTable);
    m.impl("glm_megakernel", &TRTLLM_MV2_NS::torch_ext::glmMegakernel);
}

TRTLLM_MV2_LIBRARY_IMPL(Functionalize, m)
{
    m.impl("glm_megakernel",
        torch::CppFunction::makeFromBoxedFunction<&TRTLLM_MV2_NS::torch_ext::rejectMegakernelFunctionalization>());
    m.impl("glm_megakernel_weight_table",
        torch::CppFunction::makeFromBoxedFunction<&TRTLLM_MV2_NS::torch_ext::rejectMegakernelFunctionalization>());
}
