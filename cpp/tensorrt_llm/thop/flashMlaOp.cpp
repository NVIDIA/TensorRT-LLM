/*
 * MIT License
 *
 * Copyright (c) 2025 DeepSeek
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
 *
 * reference: https://github.com/deepseek-ai/FlashMLA
 */
// Adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/flash_api.cpp
// which is itself adapted from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

#include <cutlass/fast_math.h>

#include "tensorrt_llm/kernels/flashMLA/flash_mla.h"
#include "tensorrt_llm/thop/thUtils.h"

/**
 * @brief Get metadata for MLA computation
 * @param seqlens_k Tensor of shape (batch_size), dtype int32, sequence lengths of each sequence
 * @param tile_scheduler_metadata Tensor of shape (num_sm_parts, TileSchedulerMetaDataSize), dtype int32, metadata of
 * each sm part
 * @param num_splits Tensor of shape (batch_size + 1), dtype int32, split information of each sequence
 *
 * The function populates the provided tile_scheduler_metadata and num_splits tensors with
 * the computed metadata required for FlashMLA computation.
 */
void get_mla_metadata(torch::Tensor seqlens_k, torch::Tensor tile_scheduler_metadata, torch::Tensor num_splits)
{
    // This should match the logic in the MLA kernel, see cpp/tensorrt_llm/kernels/flashMLA/
    // static constexpr int block_size_m = 64;
    static constexpr int block_size_n = 64;
    static constexpr int fixed_overhead_num_blocks = 5;

    CHECK_INPUT(seqlens_k, torch::kInt32);
    CHECK_INPUT(tile_scheduler_metadata, torch::kInt32);
    CHECK_INPUT(num_splits, torch::kInt32);

    int batch_size = seqlens_k.size(0);
    int* seqlens_k_ptr = seqlens_k.data_ptr<int>();

    int num_sm_parts = tile_scheduler_metadata.size(0);
    TORCH_CHECK(num_splits.size(0) == batch_size + 1, "num_splits must be of shape (batch_size + 1)");

    int* tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    int* num_splits_ptr = num_splits.data_ptr<int>();

    at::cuda::CUDAGuard device_guard{seqlens_k.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    Mla_metadata_params params = {};
    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_sm_parts = num_sm_parts;
    get_mla_metadata_func(params, stream);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("get_mla_metadata(Tensor seqlens_k, Tensor tile_scheduler_metadata, Tensor num_splits) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("get_mla_metadata", &get_mla_metadata);
}
