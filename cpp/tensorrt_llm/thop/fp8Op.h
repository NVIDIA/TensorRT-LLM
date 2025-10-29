/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace torch_ext
{
// Given the rowIdx and colIdx in the unswizzled SFMatrix, compute the 1D offset in the swizzled SFMatrix.
// colIdx and totalCloumn should be in SFMatrix, not activation Matrix, so no sfVecSize needed.
inline int computeSFIndex(int rowIdx, int colIdx, int totalRow, int totalColumn,
    tensorrt_llm::QuantizationSFLayout layout, bool useUE8M0 = false)
{
    constexpr int kColumnGroup0Size = 4;
    constexpr int kRowGroup0Size = 32;
    constexpr int kRowGroup1Size = kRowGroup0Size * 4;

    // Swizzled layout is used as default layout.
    if (layout == tensorrt_llm::QuantizationSFLayout::SWIZZLED)
    {
        // int paddedRow = PadUpFn(totalRow, 128);
        int paddedColumn = PadUpFn(totalColumn, 4);

        int columnIdxInGroup0 = colIdx % kColumnGroup0Size;
        int columnGroupIdx = colIdx / kColumnGroup0Size;
        constexpr int columnGroupStride = kColumnGroup0Size * kRowGroup1Size;

        int rowIdxInGroup0 = rowIdx % kRowGroup0Size;
        int rowIdxInGroup1 = (rowIdx % kRowGroup1Size) / kRowGroup0Size;
        int rowGroupIdx = rowIdx / kRowGroup1Size;
        constexpr int rowGroup1Stride = kColumnGroup0Size;
        constexpr int rowGroup0Stride = kColumnGroup0Size * rowGroup1Stride;
        int rowGroupStride = kRowGroup1Size * paddedColumn;

        return columnIdxInGroup0 + columnGroupIdx * columnGroupStride + rowIdxInGroup0 * rowGroup0Stride
            + rowIdxInGroup1 * rowGroup1Stride + rowGroupIdx * rowGroupStride;
    }
    // Linear layout is only used in E2M1AndUFP8SFScaleToFloatV2.
    else if (layout == tensorrt_llm::QuantizationSFLayout::LINEAR)
    {
        // no padding needed. totalColumn is multiple of kVecSize.
        return rowIdx * totalColumn + colIdx;
    }
    else
    {
        TLLM_THROW("Other layout not implemented yet.");
    }
}

std::tuple<torch::Tensor, torch::Tensor> symmetric_quantize_weight(torch::Tensor weight);
std::tuple<torch::Tensor, torch::Tensor> symmetric_quantize_activation(torch::Tensor activation);
std::tuple<torch::Tensor, torch::Tensor> symmetric_quantize_per_tensor(torch::Tensor input);
std::tuple<torch::Tensor, torch::Tensor> symmetric_static_quantize_weight(torch::Tensor weight, torch::Tensor scales);
std::tuple<torch::Tensor, torch::Tensor> symmetric_static_quantize_activation(
    torch::Tensor activation, torch::Tensor scales);
std::tuple<torch::Tensor, torch::Tensor> symmetric_static_quantize_per_tensor(
    torch::Tensor input, torch::Tensor scales);

torch::Tensor symmetric_dequantize_weight(torch::Tensor weight, torch::Tensor scales);
torch::Tensor symmetric_dequantize_activation(torch::Tensor activation, torch::Tensor scales);
torch::Tensor symmetric_dequantize_per_tensor(torch::Tensor input, torch::Tensor scales);

} // namespace torch_ext
