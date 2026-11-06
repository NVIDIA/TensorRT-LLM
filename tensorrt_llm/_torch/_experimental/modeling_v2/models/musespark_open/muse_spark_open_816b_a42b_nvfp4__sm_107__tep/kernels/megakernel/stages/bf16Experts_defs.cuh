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

// BF16 MoE experts stage of the megakernel (layers 0 and 61): shapes, the workspace and shared-memory maps.
// Included inside namespace mso_mk after experts_defs.cuh, outside the device-code guard (the C ABI reports these
// constants). The layer descriptor is the NVFP4 stage's MoeDesc (same word indices; the NVFP4-only words are 0), so the
// shared routing (p4a::route) and weighted sum (p4a::finalUnit) run unchanged.
#pragma once

// ---- geometry ----------------------------------------------------------------------------------------------------
constexpr int kBfWindow = 2 * kMoeBlock;         // local stack [own | r - 1] = 64 experts (ring_buddy_assign)
constexpr int kBfKb = kMoeHidden / 64;           // k-blocks of 64 BF16 (128 B) per 4096-wide K: 64
constexpr int kBfFc1Tiles = 2 * kMoeInter / 128; // 64 tiles of 128 gate/up rows
constexpr int kBfFc2Tiles = kMoeHidden / 128;    // 32
#ifndef MSO_MK_BF_UKB
#define MSO_MK_BF_UKB 1
#endif
constexpr int kBfUkb = MSO_MK_BF_UKB;             // k-blocks per unit (one TMA request): 1, 2 or 4
static_assert(kBfUkb == 1 || kBfUkb == 2 || kBfUkb == 4, "k-blocks per unit");
constexpr int kBfUpt = kBfKb / kBfUkb;            // units per 128-row tile
constexpr int kBfFc1Units = kBfFc1Tiles * kBfUpt; // per slot: (tile, unit) units
constexpr int kBfFc2Units = kBfFc2Tiles * kBfUpt;
// One unit: kBfUkb BlockMajorK slabs of 128 rows x 64 BF16 (16 KiB each, contiguous per slab), one 3-D TMA box, SW128.
constexpr int kBfUnitBytes = 128 * 128 * kBfUkb;
constexpr int kBfStages = 8 / kBfUkb; // weight ring (128 KiB)
constexpr int kBfAccs = 4;            // TMEM accumulators
#ifndef MSO_MK_BF_N
#define MSO_MK_BF_N 8
#endif
constexpr int kBfN = MSO_MK_BF_N; // MMA N: 8 tokens, or 16 with rows 8-15 aliasing rows 0-7 (stride-0 B groups)
static_assert(kBfN == 8 || kBfN == 16, "MMA N");
constexpr uint32_t kBfTmemCols = kBfN == 8 ? 32 : 64;

constexpr int64_t kBfMagic = 0x4d4f454d4b5a3142LL; // "MOEMKZ1B"

// ---- the stage workspace (per copy) --------------------------------------------------------------------------------
// h (the FC2 B operand): per slot, per k-block of 64 intermediate columns, per token 128 B of BF16 (32 Lamport words).
constexpr int kBfWsH = 0;
constexpr int kBfWsHBytes = kMoeMaxSlots * kBfKb * kTok * 128; // 4 MiB
// FC2 rows of the local (token, k) pairs: BF16 [64][4096]; then the per-(token, k, FC2 tile) sums of (y (1 + w))^2.
// The two keep the NVFP4 stage's relative layout, so p4a::finalUnit reads them through (ws + kBfWsY - kMoeWsY).
constexpr int kBfWsY = kBfWsH + kBfWsHBytes;
constexpr int kBfWsSsq = kBfWsY + kMoeWsYBytes;
static_assert(kBfWsSsq - kBfWsY == kMoeWsSsq - kMoeWsY, "finalUnit's view of the rows and sums");
static_assert(kBfWsY >= kMoeWsY, "finalUnit's shifted base stays inside the copy");
// The normalized expert input, published by the CTA that owns each 64-column group (spread norm): BF16 [8][4096],
// Lamport words of two BF16.
constexpr int kBfWsX = kBfWsSsq + kMoeWsSsqBytes;
constexpr int kBfWsXBytes = kTok * kMoeHidden * 2;
constexpr int kBfWsBytes = kBfWsX + kBfWsXBytes;
static_assert(kBfWsBytes % 16 == 0, "the clear cut is in 16-byte granules");

// ---- the check-mode debug block (CTA 0): the NVFP4 stage's routing fields, then the normalized input
// -----------------
constexpr int kBfDbgX = 1280; // BF16 [8][4096] (after the Lamport mapping)
constexpr int kBfDbgBytes = kBfDbgX + kTok * kMoeHidden * 2;

// ---- shared memory of the stage (unit region)
// ------------------------------------------------------------------------
constexpr int kBfSmRing = 0;                                 // kBfStages x kBfUnitBytes
constexpr int kBfSmB = kBfSmRing + kBfStages * kBfUnitBytes; // the B operand: x during FC1, then h of the FC2 slot
constexpr int kBfSmBBytes = kBfKb * kTok * 128;              // [64 k-blocks][8 rows][128 B], SW128: 64 KiB
constexpr int kBfSmEnd = kBfSmB + kBfSmBBytes;
static_assert(kBfSmB % 1024 == 0, "SW128 atoms are 1 KiB aligned");
// The pre-expert gain (BF16 [4096]) sits in the B area until the norm has read it (the gather writes the area after the
// barrier that follows the norm); the warm-up boxes land in ring stage 0 (requested after that barrier too).
constexpr int kBfSmGain = kBfSmB;
// p4a::route's coalesced-logit staging (kMoeSmQStage, 8 KiB) must stay inside the ring, clear of ring stage 0.
static_assert(kMoeSmQStage >= kBfUnitBytes && kMoeSmQStage + kTok * kMoeExperts * 4 <= kBfSmB, "route staging");

// Reduction area: the NVFP4 stage's routing offsets (kMoeRdDesc .. kMoeRdRed, kMoeRdEpiSq) are shared; the barriers
// and the FC1 epilogue staging move into the (here unused) NVFP4 scale-staging range.
constexpr int kBfRdBars = 4096;
constexpr int kBfRdEpi = 8192; // FC1 epilogue: u32 [8 tokens][32 words]
constexpr int kBfBarFull = 0;
constexpr int kBfBarEmpty = kBfBarFull + kBfStages;
constexpr int kBfBarAccFull = kBfBarEmpty + kBfStages;
constexpr int kBfBarAccEmpty = kBfBarAccFull + kBfAccs;
constexpr int kBfBarHReady = kBfBarAccEmpty + kBfAccs;
constexpr int kBfBarHFree = kBfBarHReady + kBfKb;
constexpr int kBfBarGain = kBfBarHFree + 1;
constexpr int kBfBarFc1Done = kBfBarGain + 1; // tcgen05.commit after the CTA's last FC1 MMA: x is no longer read
constexpr int kBfNumBars = kBfBarFc1Done + 1;
static_assert(kBfRdBars + 8 * kBfNumBars <= kBfRdEpi, "barriers");
static_assert(kBfRdEpi + kTok * 32 * 4 <= kMoeRdEpiSq, "FC1 epilogue staging");

// TMEM: accumulator a at columns [a kBfN, a kBfN + kBfN).
// Watchdog wait codes (the NVFP4 stage's codes 160-166 are reused for the shared routing / weighted sum).
enum BfWaitCode : int
{
    kCodeBfPool = 170,
    kCodeBfBar = 171,
    kCodeBfPart = 172,
    kCodeBfH = 173,
    kCodeBfX = 174,
};
