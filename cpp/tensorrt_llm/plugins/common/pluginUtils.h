/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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

#pragma once

#include <NvInferRuntime.h>

namespace tensorrt_llm::plugins::utils
{
using DimType = int32_t;

inline DimType computeMDimension(bool transA, nvinfer1::Dims const& dims)
{
    DimType M{1};
    if (transA)
    {
        for (int i = dims.nbDims - 1; i > 0; --i)
        {
            M *= dims.d[i];
        }
    }
    else
    {
        for (int i = 0; i < dims.nbDims - 1; ++i)
        {
            M *= dims.d[i];
        }
    }
    return M;
}

inline DimType computeNDimension(bool transB, nvinfer1::Dims const& dims)
{
    DimType N{1};
    if (transB)
    {
        for (int32_t i = 0; i < dims.nbDims - 1; ++i)
        {
            N *= dims.d[i];
        }
    }
    else
    {
        for (int32_t i = dims.nbDims - 1; i > 0; --i)
        {
            N *= dims.d[i];
        }
    }
    return N;
}

} // namespace tensorrt_llm::plugins::utils
