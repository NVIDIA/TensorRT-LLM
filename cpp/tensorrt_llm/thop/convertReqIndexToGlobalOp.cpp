/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/convertReqIndexToGlobal.h"
#include "tensorrt_llm/runtime/torchUtils.h"

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

th::Tensor convertReqIndexToGlobal(th::Tensor const& reqId, th::Tensor const& blockTable,
    th::Tensor const& tokenIndices, int64_t blockSize, int64_t numTopkTokens, int64_t strideFactor, int64_t layerId)
{
    TORCH_CHECK(reqId.is_cuda() && blockTable.is_cuda() && tokenIndices.is_cuda(), "All tensors must be CUDA tensors");
    TORCH_CHECK(reqId.scalar_type() == th::kInt32, "req_id must be int32");
    TORCH_CHECK(blockTable.scalar_type() == th::kInt32, "block_table must be int32");
    TORCH_CHECK(tokenIndices.scalar_type() == th::kInt32, "token_indices must be int32");

    TORCH_CHECK(reqId.dim() == 1, "req_id must be 1D");
    TORCH_CHECK(blockTable.dim() == 2, "block_table must be 2D");
    TORCH_CHECK(tokenIndices.dim() == 2, "token_indices must be 2D");

    // Ensure contiguous
    auto reqIdC = reqId.contiguous();
    auto blockTableC = blockTable.contiguous();
    auto tokenIndicesC = tokenIndices.contiguous();

    int32_t const numTokens = static_cast<int32_t>(reqIdC.size(0));
    int32_t const maxNumBlocksPerReq = static_cast<int32_t>(blockTableC.size(1));

    // Allocate output
    auto out = th::empty_like(tokenIndicesC);

    // Extract strides
    int64_t const btStride0 = blockTableC.stride(0);
    int64_t const btStride1 = blockTableC.stride(1);
    int64_t const tiStride0 = tokenIndicesC.stride(0);
    int64_t const tiStride1 = tokenIndicesC.stride(1);
    int64_t const outStride0 = out.stride(0);
    int64_t const outStride1 = out.stride(1);

    auto stream = at::cuda::getCurrentCUDAStream(reqIdC.get_device()).stream();

    tk::invokeConvertReqIndexToGlobal(reqIdC.data_ptr<int32_t>(), blockTableC.data_ptr<int32_t>(),
        tokenIndicesC.data_ptr<int32_t>(), out.data_ptr<int32_t>(), numTokens, static_cast<int32_t>(numTopkTokens),
        maxNumBlocksPerReq, static_cast<int32_t>(blockSize), static_cast<int32_t>(strideFactor),
        static_cast<int32_t>(layerId), btStride0, btStride1, tiStride0, tiStride1, outStride0, outStride1, stream);

    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "convert_req_index_to_global(Tensor req_id, Tensor block_table, Tensor token_indices, int block_size, int "
        "num_topk_tokens, int stride_factor, int layer_id) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("convert_req_index_to_global", &tensorrt_llm::torch_ext::convertReqIndexToGlobal);
}
