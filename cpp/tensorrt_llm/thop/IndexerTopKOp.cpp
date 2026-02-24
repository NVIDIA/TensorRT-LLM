/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/runtime/torchUtils.h"

#include "tensorrt_llm/kernels/IndexerTopK.h"

// #include <NvInferRuntime.h>
// #include <c10/cuda/CUDAStream.h>
// #include <cassert>
// #include <set>
// #include <string>
// #include <torch/extension.h>
// #include <vector>

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

void indexer_topk_decode(
    th::Tensor const& logits, th::Tensor const& seq_lens, th::Tensor const& indices, int64_t next_n, int64_t index_topk)
{

    TORCH_CHECK(logits.is_cuda() && seq_lens.is_cuda() && indices.is_cuda(),
        "logits, seq_lens, and indices must be CUDA tensors");
    TORCH_CHECK(logits.get_device() == seq_lens.get_device() && logits.get_device() == indices.get_device(),
        "logits, seq_lens, and indices must be on the same device");

    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D Tensor");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be a 1D Tensor");
    TORCH_CHECK(indices.dim() == 2, "indices must be a 2D Tensor");
    auto const inputSize = logits.sizes();
    auto const numRows64 = inputSize[0];
    auto const numColumns64 = inputSize[1];
    TORCH_CHECK(
        seq_lens.size(0) * next_n == numRows64, "seq_lens length multiplied by next_n must equal logits.size(0)");
    TORCH_CHECK(indices.size(0) == numRows64, "indices first dimension must match logits.size(0)");
    TORCH_CHECK(indices.size(1) >= index_topk, "indices second dimension must be at least index_topk");
    TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

    TORCH_CHECK(next_n > 0, "next_n must be greater than 0");

    int32_t num_rows = static_cast<int32_t>(numRows64);
    int32_t num_columns = static_cast<int32_t>(numColumns64);
    int32_t logits_stride_0 = static_cast<int32_t>(logits.stride(0));
    int32_t logits_stride_1 = static_cast<int32_t>(logits.stride(1));

    TORCH_CHECK(logits_stride_0 >= 0, "logits_stride_0 must be greater than or equal to 0");
    TORCH_CHECK(logits_stride_1 >= 0, "logits_stride_1 must be greater than or equal to 0");

    int32_t splitWorkThreshold = 200 * 1000;
    th::Tensor aux_indices = th::empty({0}, th::TensorOptions().dtype(th::kInt32).device(logits.device()));
    th::Tensor aux_logits = th::empty({0}, th::TensorOptions().dtype(th::kFloat32).device(logits.device()));
    constexpr auto multipleBlocksPerRowConfig = 10;
    if (num_columns >= splitWorkThreshold)
    {
        aux_indices = th::empty({num_rows, multipleBlocksPerRowConfig, index_topk},
            th::TensorOptions().dtype(th::kInt32).device(logits.device()));
        aux_logits = th::empty({num_rows, multipleBlocksPerRowConfig, index_topk},
            th::TensorOptions().dtype(th::kFloat32).device(logits.device()));
    }
    auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());
    tk::invokeIndexerTopKDecode(logits.data_ptr<float>(), seq_lens.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
        aux_logits.data_ptr<float>(), aux_indices.data_ptr<int32_t>(), splitWorkThreshold, num_rows, num_columns,
        logits_stride_0, logits_stride_1, static_cast<int32_t>(next_n), static_cast<int32_t>(index_topk), stream);
}

void indexer_topk_prefill(th::Tensor const& logits, th::Tensor const& row_starts, th::Tensor const& row_ends,
    th::Tensor const& indices, int64_t index_topk)
{
    TORCH_CHECK(logits.is_cuda() && row_starts.is_cuda() && row_ends.is_cuda() && indices.is_cuda(),
        "logits, row_starts, row_ends, and indices must be CUDA tensors");
    TORCH_CHECK(logits.get_device() == row_starts.get_device() && logits.get_device() == row_ends.get_device()
            && logits.get_device() == indices.get_device(),
        "logits, row_starts, row_ends, and indices must be on the same device");

    TORCH_CHECK(indices.dim() == 2, "indices must be a 2D Tensor");
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D Tensor");

    auto const inputSize = logits.sizes();
    auto const numRows64 = inputSize[0];
    auto const numColumns64 = inputSize[1];
    TORCH_CHECK(row_starts.dim() == 1, "row_starts must be a 1D Tensor");
    TORCH_CHECK(row_ends.dim() == 1, "row_ends must be a 1D Tensor");
    TORCH_CHECK(row_starts.size(0) == numRows64 && row_ends.size(0) == numRows64,
        "row_starts/row_ends must have one entry per row in logits");
    TORCH_CHECK(row_starts.is_contiguous(), "row_starts must be contiguous");
    TORCH_CHECK(row_ends.is_contiguous(), "row_ends must be contiguous");

    int32_t num_rows = static_cast<int32_t>(numRows64);
    int32_t num_columns = static_cast<int32_t>(numColumns64);
    int32_t logits_stride_0 = static_cast<int32_t>(logits.stride(0));
    int32_t logits_stride_1 = static_cast<int32_t>(logits.stride(1));

    TORCH_CHECK(logits_stride_0 >= 0, "logits_stride_0 must be greater than or equal to 0");
    TORCH_CHECK(logits_stride_1 >= 0, "logits_stride_1 must be greater than or equal to 0");

    auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());
    tk::invokeIndexerTopKPrefill(logits.data_ptr<float>(), row_starts.data_ptr<int32_t>(), row_ends.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(), num_rows, num_columns, static_cast<int32_t>(logits_stride_0),
        static_cast<int32_t>(logits_stride_1), static_cast<int32_t>(index_topk), stream);
}

} // end namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "indexer_topk_decode(Tensor logits, Tensor seq_lens, Tensor indices, int next_n, int index_topk=2048) -> "
        "()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("indexer_topk_decode", &tensorrt_llm::torch_ext::indexer_topk_decode);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "indexer_topk_prefill(Tensor logits, Tensor row_starts, Tensor row_ends, Tensor indices, int "
        "index_topk=2048) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("indexer_topk_prefill", &tensorrt_llm::torch_ext::indexer_topk_prefill);
}
