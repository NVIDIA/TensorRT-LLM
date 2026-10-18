/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gtest/gtest.h>

#include "trtllm/gen/CudaArchDecl.h"

// Defined in the prebuilt libTrtLlmGen.a; not declared by the exported headers.
namespace trtllm::gen
{
int getMaxNumTmemCols(CudaArch cudaArch);
} // namespace trtllm::gen

namespace
{

using trtllm::gen::CudaArch;
using trtllm::gen::getMaxNumTmemCols;

// The CudaArch values in the exported header must mean the same architectures inside the prebuilt
// trtllm-gen archives. getMaxNumTmemCols is implemented in the archive and gives a distinct answer
// for SM107, so it checks the numbering across the boundary.
TEST(TrtllmGenCudaArchAbiTest, MaxNumTmemColsMatchesArchive)
{
    EXPECT_EQ(getMaxNumTmemCols(CudaArch::Sm100a), 512);
    EXPECT_EQ(getMaxNumTmemCols(CudaArch::Sm100f), 512);
    EXPECT_EQ(getMaxNumTmemCols(CudaArch::Sm103a), 512);
    EXPECT_EQ(getMaxNumTmemCols(CudaArch::Sm107a), 576);
}

} // namespace
