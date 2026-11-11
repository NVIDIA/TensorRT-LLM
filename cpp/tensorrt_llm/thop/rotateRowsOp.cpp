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

#include "tensorrt_llm/kernels/rotateRows.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace torch_ext
{

//! In place: every row of ``self`` becomes ``torch.roll(row, shift)``: a positive shift moves
//! elements toward higher indices, a negative one toward lower. ``self`` is 1-D or 2-D with a
//! unit-stride last dimension; rows may be strided but must not overlap; any dtype (the rotation
//! permutes bytes).
//! No scratch memory, no allocation, safe inside CUDA graph capture.
void rotate_rows_(torch::Tensor self, int64_t shift)
{
    CHECK_TH_CUDA(self);
    TORCH_CHECK(self.dim() == 1 || self.dim() == 2, "rotate_rows_: expected a 1-D or 2-D tensor");
    TORCH_CHECK(self.stride(-1) == 1, "rotate_rows_: the last dimension must be contiguous");
    int64_t const elemSize = self.element_size();
    TORCH_CHECK(elemSize == 1 || elemSize == 2 || elemSize == 4 || elemSize == 8 || elemSize == 16,
        "rotate_rows_: unsupported element size");

    int64_t const rows = self.dim() == 2 ? self.size(0) : 1;
    int64_t const cols = self.size(-1);
    int64_t const rowStride = self.dim() == 2 ? self.stride(0) : cols;
    TORCH_CHECK(rows <= 1 || rowStride >= cols, "rotate_rows_: rows overlap (row stride ", rowStride, " < ", cols,
        " columns); rotating one would corrupt another");
    if (rows == 0 || cols <= 1)
    {
        return;
    }
    at::cuda::CUDAGuard const guard(self.device());
    auto stream = at::cuda::getCurrentCUDAStream(self.get_device());
    tensorrt_llm::kernels::invokeRotateRows(
        self.data_ptr(), rows, cols, rowStride, shift, static_cast<int>(elemSize), stream);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("rotate_rows_(Tensor(a!) self, int shift) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("rotate_rows_", &torch_ext::rotate_rows_);
}
