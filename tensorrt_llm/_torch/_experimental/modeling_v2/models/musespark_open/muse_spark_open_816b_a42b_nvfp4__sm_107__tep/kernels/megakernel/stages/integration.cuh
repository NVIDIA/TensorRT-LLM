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

// The hand-offs between the stages when one launch runs a whole decoder layer (kPIFlags bits, msoMegakernel.cu):
//
//   kIFlagFaOutArena    FaMerge -> AhOproj: the gated o_proj input as Lamport words in the arena (kArAhO replicas).
//   kIFlagOprojToMh     AhOproj -> MhH1: the o_proj partial into slot `rank` of every rank's hop1 area of the MoE hop
//                       workspace (moe_hops_layout.cuh), through the multicast address, as Lamport words; instead of
//                       the stock hop1 slot (kPAhSlot).
//   kIFlagMhRes1Global  MhH1 -> MhF1: r' through global memory (kPMhOutRes1, written by MhH1 as its check output),
//                       because MoeExperts owns the CTA's whole unit region between the two units.
//   kPIResOut != 0      MhF1 writes r'' there instead of updating io.res in place.
//   kMoeFlagMhF2        MoeExperts reads the logits and the pooled input from the F2' areas of this launch's copy
//                       (mhLogitPtr / mhPoolPtrs layouts) and publishes its weighted-sum partial into slot `rank` of
//                       every rank's finalize area (kMhFinOff) instead of a plain store to kMdOut.
//
// Included by msoMegakernel.cu inside namespace mso_mk after stages/moe_hops.cuh (device code only).

#pragma once

// kN 32-bit words at addresses a[i] (nullptr: none), all loads in flight together, then again for the ones still
// kEmpty; the whole warp calls it (the pollVecs protocol for scalar words).
template <int kN>
__device__ __forceinline__ void i1PollWords(Ctx const& ctx, uint32_t (&v)[kN], uint8_t const* const (&a)[kN], int code)
{
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        v[i] = a[i] != nullptr ? ldPoll32(a[i]) : 0u;
    }
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            ok &= v[i] != kEmpty;
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, code, kN, 1);
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (v[i] == kEmpty)
            {
                v[i] = ldPoll32(a[i]);
            }
        }
    }
}

// The byte offset (in a copy of the MoE hop workspace) of BF16 element (row, col) of this rank's hop1 slot.
__device__ __forceinline__ size_t i1Hop1Off(Ctx const& ctx, int row, int col)
{
    return kMhH1Off + ((static_cast<size_t>(mhRank(ctx)) * 8 + row) * 8192 + col) * 2;
}

// The byte offset of BF16 element (row, col) of this rank's slot of the finalize area.
__device__ __forceinline__ size_t i1FinOff(Ctx const& ctx, int row, int col)
{
    return kMhFinOff + ((static_cast<size_t>(mhRank(ctx)) * 8 + row) * kMhPoolWidth + col) * 2;
}
