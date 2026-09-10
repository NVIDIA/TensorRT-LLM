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

#include "tensorrt_llm/kernels/compressorKernels/compressorKernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace tk = tensorrt_llm::kernels::compressor;

namespace
{

void compressorPagedKvCompressIncrementalOp(torch::Tensor kvScore, torch::Tensor ape, torch::Tensor pagedKv,
    torch::Tensor pagedScore, torch::Tensor blockTableKv, torch::Tensor blockTableScore, torch::Tensor output,
    torch::Tensor kvLens, torch::Tensor cuSeqLens, torch::Tensor cuKvComp, int64_t batchSize, int64_t pageSize,
    int64_t headDim, int64_t compressRatio, int64_t nextN, torch::Tensor incrementalHcaValid)
{
    constexpr int64_t kMinNextN = 1;
    constexpr int64_t kMaxNextN = 4;
    TORCH_CHECK(nextN >= kMinNextN && nextN <= kMaxNextN, "nextN must be in [", kMinNextN, ", ", kMaxNextN,
        "] before conversion to int, got ", nextN);
    TORCH_CHECK(compressRatio == 128, "incremental HCA compression requires compressRatio == 128");

    auto stream = at::cuda::getCurrentCUDAStream();
    int const kvScoreElemBytes = static_cast<int>(kvScore.element_size());
    int const stateElemBytes = static_cast<int>(pagedKv.element_size());
    int const outputElemBytes = static_cast<int>(output.element_size());

    TORCH_CHECK(pagedScore.element_size() == pagedKv.element_size(), "pagedKv and pagedScore must use the same dtype");
    TORCH_CHECK(incrementalHcaValid.is_cuda(), "incrementalHcaValid must be a CUDA tensor");
    TORCH_CHECK(incrementalHcaValid.scalar_type() == at::kBool, "incrementalHcaValid must have bool dtype");
    TORCH_CHECK(incrementalHcaValid.is_contiguous(), "incrementalHcaValid must be contiguous");
    TORCH_CHECK(
        incrementalHcaValid.numel() >= batchSize, "incrementalHcaValid must contain at least batchSize entries");

    tk::incrementalHcaCompressLaunch(kvScore.data_ptr(), ape.data_ptr<float>(), pagedKv.data_ptr(),
        pagedScore.data_ptr(), blockTableKv.data_ptr<int32_t>(), blockTableScore.data_ptr<int32_t>(), output.data_ptr(),
        kvLens.data_ptr<int32_t>(), cuSeqLens.data_ptr<int32_t>(), cuKvComp.data_ptr<int32_t>(),
        reinterpret_cast<bool const*>(incrementalHcaValid.data_ptr()), static_cast<int>(batchSize),
        static_cast<int>(pageSize), static_cast<int>(blockTableKv.size(1)), static_cast<int>(headDim),
        static_cast<int>(nextN), kvScoreElemBytes, stateElemBytes, outputElemBytes, stream);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "compressor_paged_kv_compress_incremental("
        "Tensor kv_score, Tensor ape, "
        "Tensor(a!) paged_kv, Tensor(b!) paged_score, "
        "Tensor block_table_kv, Tensor block_table_score, "
        "Tensor(c!) output, "
        "Tensor kv_lens, "
        "Tensor cu_seq_lens, Tensor cu_kv_comp, "
        "int batch_size, int page_size, "
        "int head_dim, int compress_ratio, "
        "int next_n, Tensor incremental_hca_valid) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("compressor_paged_kv_compress_incremental", &compressorPagedKvCompressIncrementalOp);
}
