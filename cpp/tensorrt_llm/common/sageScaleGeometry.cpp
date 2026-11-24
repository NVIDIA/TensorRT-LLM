/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
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

#include "sageScaleGeometry.h"

TRTLLM_NAMESPACE_BEGIN

namespace common
{

int getSageScaleHeadStride(SageScalePartition partition, int tokenBlockSize, int sumSeqLens, int batchSize)
{
    if (tokenBlockSize <= 0 || sumSeqLens <= 0 || batchSize <= 0)
    {
        return 0;
    }
    switch (partition)
    {
    case SageScalePartition::HopperQ: return HopperQScaleGeometry::scaleHeadStride(sumSeqLens, batchSize);
    case SageScalePartition::HopperK: return HopperKScaleGeometry::scaleHeadStride(sumSeqLens, batchSize);
    case SageScalePartition::Contiguous:
        switch (tokenBlockSize)
        {
        case 1: return ContiguousScaleGeometry<1>::scaleHeadStride(sumSeqLens, batchSize);
        case 4: return ContiguousScaleGeometry<4>::scaleHeadStride(sumSeqLens, batchSize);
        case 16: return ContiguousScaleGeometry<16>::scaleHeadStride(sumSeqLens, batchSize);
        case 32: return ContiguousScaleGeometry<32>::scaleHeadStride(sumSeqLens, batchSize);
        default: break;
        }
        break;
    }
    return 0;
}

} // namespace common

TRTLLM_NAMESPACE_END
