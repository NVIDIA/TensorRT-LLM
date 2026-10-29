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

// Elementwise frame operations of the Wan VAE, bf16, frames as [T, P, C] rows
// (a batch-1 channels-last video), 16-byte vectors, one bf16 rounding per add in the
// eager order so results match the PyTorch chain bit for bit.

// Upsample3d time interleave with the time conv's bias: u is [T, P, 2C] (the C -> 2C
// time conv output, bias not applied), y is [2T, P, C] with
//   y[2t] = u[t, :, 0:C] + bias[0:C],  y[2t+1] = u[t, :, C:2C] + bias[C:2C].
struct VaeInterleaveParams
{
    int64_t positions;
    int channels; // C (output channels)
    int frames;   // T (input frames)
};

void launchVaeInterleaveBias(
    void const* u, void const* bias, void* y, VaeInterleaveParams const& p, cudaStream_t stream);

// Encoder down-block shortcut: y = x + AvgDown3D(r). AvgDown3D averages, for output
// channel o, the group_size members f = o*group_size + g, each mapping to source
// channel c = f / factor and sub-position (a, b, d) = unravel(f % factor, (factorT, factorS,
// factorS)) of the factorT x factorS x factorS window; source frame t*factorT + a - padT
// is zero when negative. Frames as [T', P', C] rows for x/y and [T, P, Cin] for r.
struct VaeAvgDownParams
{
    int64_t outPositions; // H' * W'
    int outHeight, outWidth;
    int inWidth;          // W (= W' * factorS)
    int inFrames;         // T (source frames, before the front padding)
    int outFrames;        // T'
    int cin, cout;
    int factorT, factorS, groupSize, padT;
};

// y = x (+ bias[c]) + AvgDown3D(r); ``bias`` (bf16 [Cout], may be null) is the downsample conv's
// bias, added here in fp32 instead of by a separate pass over its output.
void launchVaeAvgDownAdd(
    void const* x, void const* r, void const* bias, void* y, VaeAvgDownParams const& p, cudaStream_t stream);

// Decoder output assembly: chunk [1, C*patch*patch, t, H, W] (channels-last) ->
// out[:, :, frame0 : frame0 + t] of the video [1, C, T, H*patch, W*patch] (channels-last), with
// out[c, f, h*patch + j, w*patch + i] = clamp(chunk[c*patch*patch + i*patch + j, f, h, w], -1, 1),
// the mapping of wan_vae.unpatchify.
struct VaeUnpatchifyParams
{
    int frames;    // t: frames in the chunk
    int height;    // H
    int width;     // W
    int channels;  // C: output channels
    int patch;     // spatial patch side
    int outFrames; // T
    int frame0;    // first output frame of the chunk
};

void launchVaeUnpatchifyClamp(void const* chunk, void* out, VaeUnpatchifyParams const& p, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
