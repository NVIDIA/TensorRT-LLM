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
#include "tensorrt_llm/kernels/rotateRows.h"

#include <algorithm>

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// Reverse [begin, end) of every row: thread (row, i) swaps element i with element end-1-i.
template <typename T>
__global__ void reverseRowsKernel(T* data, int64_t rows, int64_t rowStride, int64_t begin, int64_t end)
{
    int64_t const half = (end - begin) / 2;
    int64_t const i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= half)
    {
        return;
    }
    // Rows beyond the grid's y limit are covered by striding.
    for (int64_t row = blockIdx.y; row < rows; row += gridDim.y)
    {
        T* r = data + row * rowStride;
        T const a = r[begin + i];
        r[begin + i] = r[end - 1 - i];
        r[end - 1 - i] = a;
    }
}

template <typename T>
void reverseRows(T* data, int64_t rows, int64_t rowStride, int64_t begin, int64_t end, cudaStream_t stream)
{
    int64_t const half = (end - begin) / 2;
    if (half <= 0)
    {
        return;
    }
    constexpr int kThreads = 256;
    constexpr int64_t kMaxGridY = 65535;
    dim3 const grid(
        static_cast<unsigned>((half + kThreads - 1) / kThreads), static_cast<unsigned>(std::min(rows, kMaxGridY)));
    reverseRowsKernel<T><<<grid, kThreads, 0, stream>>>(data, rows, rowStride, begin, end);
    check_cuda_error(cudaGetLastError());
}

template <typename T>
void rotateRowsLeft(T* data, int64_t rows, int64_t cols, int64_t rowStride, int64_t shift, cudaStream_t stream)
{
    // rotate(first, middle, last) == reverse(first, middle); reverse(middle, last); reverse(first, last)
    reverseRows(data, rows, rowStride, 0, shift, stream);
    reverseRows(data, rows, rowStride, shift, cols, stream);
    reverseRows(data, rows, rowStride, 0, cols, stream);
}

} // namespace

void invokeRotateRows(
    void* data, int64_t rows, int64_t cols, int64_t rowStride, int64_t shift, int elemSize, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 1)
    {
        return;
    }
    // A right rotation by s is a left rotation by cols - s; reduce any shift to a left one in [0, cols).
    int64_t const left = (cols - shift % cols) % cols; // shift % cols first: -INT64_MIN overflows
    if (left == 0)
    {
        return;
    }
    switch (elemSize)
    {
    case 1: rotateRowsLeft(static_cast<uint8_t*>(data), rows, cols, rowStride, left, stream); break;
    case 2: rotateRowsLeft(static_cast<uint16_t*>(data), rows, cols, rowStride, left, stream); break;
    case 4: rotateRowsLeft(static_cast<uint32_t*>(data), rows, cols, rowStride, left, stream); break;
    case 8: rotateRowsLeft(static_cast<uint64_t*>(data), rows, cols, rowStride, left, stream); break;
    case 16: rotateRowsLeft(static_cast<uint4*>(data), rows, cols, rowStride, left, stream); break;
    default: TLLM_THROW("rotateRows: unsupported element size %d bytes", elemSize);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
