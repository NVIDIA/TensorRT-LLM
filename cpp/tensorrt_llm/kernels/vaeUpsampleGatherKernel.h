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

#pragma once

#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Phase-shifted gather that turns the packed 2x2 convolution output of a nearest-2x
// upsample + 3x3 convolution into the final 2x-resolution frames, optionally adding
// the Wan DupUp3D shortcut while writing.
//
// temp: [B*T, 4*Cout, H+1, W+1], channels-last (physical NHWC).
//       out[b, o, t, 2i+p, 2j+q] = temp[b*T+t, (2p+q)*Cout + o, i+p, j+q]
// out:  [B, Cout, T, 2H, 2W], channels-last (physical NTHWC).
// shortcut (optional): [B, Cin, Tr, H, W], channels-last; the DupUp3D expansion
//       out_s[c, f, 2i+a, 2j+d] = shortcut[(c*factor + ((ft*2+a)*2+d)) / repeats, f/factorT, i, j]
//       with f = t + crop, ft = f % factorT, factor = factorT*4, is added to out.
struct VaeUpsampleGatherParams
{
    int batch;
    int frames;  // T (output frames)
    int height;  // H (low resolution)
    int width;   // W
    int cout;    // output channels (multiple of 8)
    int cin;     // shortcut channels (0 when no shortcut)
    int factorT; // DupUp3D temporal factor (1 or 2)
    int repeats; // DupUp3D channel repeats
    int crop;    // leading frames dropped on the first chunk (factorT - 1 or 0)
};

// True when the kernel has an instantiation for these parameters (cout % 8 == 0,
// factorT in {1, 2}, repeats a power of two <= 32).
bool vaeUpsampleGatherSupported(VaeUpsampleGatherParams const& p);

// bias (optional): [Cout], the packed convolution's bias, applied as round(temp + bias)
//       before the shortcut add.
void launchVaeUpsampleGather(void const* temp, void const* shortcut, void const* bias, void* out,
    VaeUpsampleGatherParams const& p, bool bf16, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
