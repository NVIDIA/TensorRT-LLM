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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>

namespace tensorrt_llm::torch_ext
{

// Copy src[:, :] into dest[:numTokens, dim1Start:dim1End] using cudaMemcpy2D.
// dest      : 2-D contiguous CUDA tensor, shape [destRows, destCols]
// src       : 2-D contiguous CUDA tensor, shape [numTokens, sliceWidth] where sliceWidth == dim1End - dim1Start
// dim1Start : first column index in dest to write into
// dim1End   : one-past-last column index in dest to write into
// numTokens is inferred from src.size(0)
void inplaceSliceCopy(at::Tensor& dest, at::Tensor const& src, int64_t dim1Start, int64_t dim1End)
{
    CHECK_TH_CUDA(dest);
    CHECK_TH_CUDA(src);
    TORCH_CHECK(dest.is_contiguous(), "dest must be contiguous");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dest.dim() == 2, "dest must be 2-D");
    TORCH_CHECK(src.dim() == 2, "src must be 2-D");
    TORCH_CHECK(dest.scalar_type() == src.scalar_type(), "dest and src must have the same dtype");

    int64_t const numTokens = src.size(0);
    int64_t const sliceWidth = dim1End - dim1Start;
    TORCH_CHECK(sliceWidth > 0, "dim1End must be greater than dim1Start");
    TORCH_CHECK(numTokens <= dest.size(0), "numTokens exceeds dest row count");
    TORCH_CHECK(dim1End <= dest.size(1), "dim1End exceeds dest column count");
    TORCH_CHECK(src.size(1) == sliceWidth, "src column count must equal dim1End - dim1Start");

    if (numTokens == 0 || sliceWidth == 0)
    {
        return;
    }

    int64_t const elemSize = dest.element_size();
    int64_t const destPitch = dest.size(1) * elemSize; // bytes per dest row
    int64_t const srcPitch = src.size(1) * elemSize;   // bytes per src row
    int64_t const width = sliceWidth * elemSize;       // bytes to copy per row

    char* destPtr = static_cast<char*>(dest.data_ptr()) + dim1Start * elemSize;
    char const* srcPtr = static_cast<char const*>(src.data_ptr());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dest.get_device());
    TLLM_CUDA_CHECK(cudaMemcpy2DAsync(
        destPtr, destPitch, srcPtr, srcPitch, width, static_cast<size_t>(numTokens), cudaMemcpyDeviceToDevice, stream));
}

} // namespace tensorrt_llm::torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    // dest: destination tensor (mutated in-place)
    // src:  source tensor (numTokens inferred from src.size(0))
    // dim1_start: first column index in dest
    // dim1_end:   one-past-last column index in dest
    m.def("inplace_slice_copy(Tensor(a!) dest, Tensor src, int dim1_start, int dim1_end) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("inplace_slice_copy", TORCH_FN(tensorrt_llm::torch_ext::inplaceSliceCopy));
}
