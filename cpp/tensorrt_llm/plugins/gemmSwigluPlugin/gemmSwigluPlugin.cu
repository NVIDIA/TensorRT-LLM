/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gemmSwigluPlugin.h"

#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass_extensions/gemm_configs.h"

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using tensorrt_llm::plugins::GemmSwigluPluginCreator;
using tensorrt_llm::plugins::GemmSwigluPlugin;
using tensorrt_llm::plugins::GemmSwigluPluginProfiler;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;

void GemmSwigluPluginProfiler::initTmpData(int m, int n, int k, char* workspace, size_t size, cudaStream_t stream)
{
    size_t bpe = getBytePerElement(mType);

    if (mType == nvinfer1::DataType::kFP8)
    {
        cutlass::reference::device::BlockFillRandomUniform(reinterpret_cast<cutlass::float_e4m3_t*>(workspace),
            m * k + n * k + 1 * n, 42, cutlass::float_e4m3_t{128}, -cutlass::float_e4m3_t{128}, -1, 0, stream);
    }
}
