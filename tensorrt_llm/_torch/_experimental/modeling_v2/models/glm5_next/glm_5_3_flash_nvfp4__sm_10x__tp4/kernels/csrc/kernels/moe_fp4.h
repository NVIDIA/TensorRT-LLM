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
// Fused MoE experts for decode-sized steps on the SM100 tensor cores (see moe_fp4.cu): NVFP4 or
// BF16 routed experts plus an optional BF16 shared expert, in one launch.
#pragma once

#include "../common.h"

namespace glm {
namespace k {

constexpr int kMoeFp4MaxItems = 4096;
constexpr int kMoeFp4MaxTokens = 128;
constexpr int kMoeFp4ShSplit = 4;  // the shared expert's gate / up units: split-K parts
// counters (int): a header ([0] the counter buffer of the next launch, [1] work queue, [2] finished
// CTAs: restored by the last CTA out) and two buffers ([0] shared-expert blocks stored (bitmask),
// [1, 1 + 32) arrived split-K parts per shared block, then per routed work item its blocks stored
// (bitmask), the tokens routed (in-kernel routing over the CTAs), the work items its launch used).
// A launch takes its buffer after the grid dependency wait, zeroes the other one (the previous
// launch's) and its last CTA hands that one to the next launch: a few plain stores and no fences
// at the kernel's end (its completion orders them before the next launch's reads)
constexpr int kMoeFp4CtrHdr = 32;
constexpr int kMoeFp4CtrItems = 1 + 32;
constexpr int kMoeFp4CtrRt = kMoeFp4CtrItems + kMoeFp4MaxItems;
constexpr int kMoeFp4CtrNused = kMoeFp4CtrRt + 1;
constexpr int kMoeFp4CtrSize = kMoeFp4CtrNused + 1;  // one buffer
// then (in-kernel routing over the CTAs) the launch epoch (bumped by every launch's last CTA) and
// the routed ids as {epoch + 1, id} 64-bit words, [kMoeFp4RtMaxT x kMoeFp4RtMaxK1]: a reader polls a
// word itself until it carries its launch's epoch (no fence and count behind the ids)
constexpr int kMoeFp4CtrTag = (kMoeFp4CtrHdr + 2 * kMoeFp4CtrSize + 1) & ~1;  // (8-byte aligned)
constexpr int kMoeFp4CtrTagWords = 64 * 16;  // kMoeFp4RtMaxT x kMoeFp4RtMaxK1
constexpr int kMoeFp4CtrAlloc = kMoeFp4CtrTag + 2 + 2 * kMoeFp4CtrTagWords;

struct MoeFp4Args {
  // this rank's experts with rows from moe_dispatch_small (one item per expert, any m-tile size
  // >= the token count): active[i] = group, count *num_active; offsets [G + 1]: grouped rows per
  // group; a_rows [rows]: token of each grouped row
  const int* active = nullptr;
  const int* num_active = nullptr;
  const int* offsets = nullptr;
  const int* a_rows = nullptr;
  bool routed_bf16 = false;  // routed experts: false NVFP4 (W4A4), true BF16
  // the step's tokens (row t: token t): NVFP4 rows [x_rows, H/2] with scales in the CUTLASS
  // layout of [x_rows, H] (nvfp4_quant swz), or BF16 rows [x_rows, H] (xs unused)
  const void* xq = nullptr;
  const uint8_t* xs = nullptr;
  int x_rows = 0;
  // routed experts: NVFP4 w13 [G, 2 Il, H/2] (gate / up rows interleaved by 64, kGu64) with scales
  // [G] x nvfp4_sf_bytes(2 Il, H) and alpha13 [G]; w2 [G, H, Il/2], scales, alpha2 [G]. BF16:
  // w13 [G, 2 Il, H] (interleaved by 64), w2 [G, H, Il] (scales / alphas unused)
  const void* w13 = nullptr;
  const uint8_t* sf13 = nullptr;
  const float* alpha13 = nullptr;
  const void* w2 = nullptr;
  const uint8_t* sf2 = nullptr;
  const float* alpha2 = nullptr;
  // scratch: the routed experts' SwiGLU output per grouped row: NVFP4 [h_rows, Il/2] with scales
  // hsa = [items] x nvfp4_sf_bytes(128, Il) (each expert's rows in their own 128-row CUTLASS scale
  // layout), or BF16 [h_rows, Il]
  void* hq = nullptr;
  uint8_t* hsa = nullptr;
  int h_rows = 0;
  bf16* y = nullptr;  // [rows, H] routed expert outputs per grouped row
  // the BF16 shared expert (sh_inter 0 or sh_T 0: none) of the sh_T tokens of sh_x (rows of H,
  // sh_x_rows allocated): gate / up [2 Is, H] interleaved by 64, down [H, Is]. Scratch: its SwiGLU
  // output sh_h [sh_h_rows >= 1, Is] and the gate / up split-K partials sh_part (fp32,
  // Is / 64 * kMoeFp4ShSplit * 128 * round16(sh_T)). Output row t to sh_y + t * sh_ldy.
  int sh_inter = 0;
  const bf16* sh_x = nullptr;
  int sh_x_rows = 0, sh_T = 0;
  const bf16* sh_gu = nullptr;
  const bf16* sh_dn = nullptr;
  bf16* sh_h = nullptr;
  int sh_h_rows = 0;
  float* sh_part = nullptr;
  bf16* sh_y = nullptr;
  int sh_ldy = 0;
  // the shared expert's weights requested into L2 by the CTAs before the grid dependency wait
  // (the dispatch before this kernel leaves HBM idle; tall kernel only): 0 none, 1 gate / up,
  // 2 gate / up and down
  int sh_pf = 0;
  // the CTAs that request them: the last sh_pf_ctas of the grid (0: every CTA). The last CTAs become resident as
  // the kernel before this one (the router) exits; the first ones while it still runs, where the requests would
  // slow it. At most the CTAs past the shared gate / up CTAs (kMoeFp4ShSplit x sh_inter / 64 first ones)
  int sh_pf_ctas = 0;
  // tall kernel: the first claim and the dispatch's tables requested together with the item count
  // (one work item per expert: <= G items)
  bool tab_early = false;
  // the dependent kernel launches only once the grid dependency wait is over (else at the start):
  // it may then read before its own wait what the kernels before this one wrote
  bool late_trigger = false;
  int* ctr = nullptr;  // [kMoeFp4CtrAlloc] zero-initialized (the header and two counter buffers)
  unsigned opts = 0;   // kernel options (kMoeFp4Opt*), set by the launchers
  int H = 0, Il = 0;
  int G = 0;           // routed expert groups (weights)
  float limit = 0.f;   // SwiGLU clamp
  float gs2 = 1.f;     // global scale (input_scale) of the NVFP4 hidden rows' quantization
  // in-kernel routing (rt_logits set; the tall kernel, rt_T <= kMoeFp4RtMaxT tokens): the tokens are
  // routed from the router logits [rt_T, rt_E] (moe_route_warp's selection and weights, bitwise) by the
  // epilogue warps of every CTA (up to kMoeFp4RtLocalT tokens) or by one epilogue warp per token over
  // the CTAs (published through a counter; a separate instantiation), while the producer / MMA warps
  // start the shared expert's first units; every CTA builds the tables of its local experts [rt_lo,
  // rt_lo + G) itself: one work item per expert with rows (up to 32 pairs in the order of the
  // experts' first pairs, published to the producer before the rows' offsets, which only its down
  // units need; else in expert order), an expert's rows in token order. active / num_active
  // / offsets / a_rows are unused (no moe_dispatch_small before the kernel). Outputs for the combine:
  // the routing ids / weights [rt_T, rt_topk + 1] (slot rt_topk: the shared expert, id rt_E, weight 1)
  // and every pair's grouped row (rt_pos; the shared expert's pair of token t: rt_T * rt_topk + t,
  // other ranks' experts -1).
  const float* rt_logits = nullptr;
  const float* rt_bias = nullptr;
  // rt_logits holds moe_router_mma_ks's {key, score} 64-bit words per (token, expert) instead of logits: the routing
  // without the scores' arithmetic (rt_bias unused); only where moe_fp4_route_ks_ok
  bool rt_ks = false;
  int rt_T = 0, rt_E = 0, rt_topk = 0, rt_lo = 0;
  float rt_scale = 1.f;
  int* rt_ids = nullptr;
  float* rt_w = nullptr;
  int* rt_pos = nullptr;
  // tall kernel: CTA 0 records the launch's local experts (in-kernel routing or the dispatch's
  // tables) for the next step's L2 prefetch of this layer's weights (k::moe_l2_prefetch): [0] the
  // count n, [1, 1 + n) the local experts (work item order), [1 + kMoeFp4HistMax, + n) their rows
  // (tokens); plain stores (read a step later)
  int* rt_hist = nullptr;
  // in-kernel routing: the routed pairs' output rows token-major (pair i = t (rt_topk + 1) + slot at y + i H instead of
  // its grouped row), so that the combine after this kernel (the MoE-side mHC boundary's kTm instances) loads a token's
  // rows without its pair positions; the shared expert's rows by sh_y / sh_ldy as ever (the host points them at slot
  // rt_topk of each token). Under MoE tensor parallelism (every pair on this rank) only.
  bool y_tm = false;
};
constexpr int kMoeFp4HistMax = 320;  // (local experts per launch: at most the kernel's kTab)
constexpr int kMoeFp4HistStride = 1 + 2 * kMoeFp4HistMax;
// REL1 (tall and wide kernels): a stored block's flag released by one acq-rel fence of the flagging thread instead
// of a full fence per epilogue thread plus one
constexpr unsigned kMoeFp4OptRel1 = 1u;
// S1_PRE (tall kernel): the shared expert's gate / up units as the static first units of the first CTAs, their
// first weight tiles issued before the grid dependency wait
constexpr unsigned kMoeFp4OptS1Pre = 4u;
// tall kernel: two units per queue claim while far from the end, with > 8 units per CTA
constexpr unsigned kMoeFp4OptPair = 8u;
// RT_WARM (tall kernel, in-kernel routing within each CTA): the MMA warp runs the routing's code dry before the PDL
// wait, so that the epilogue warps' routing after it runs from a warm instruction cache
constexpr unsigned kMoeFp4OptRtWarm = 16u;
// SH_SUM (set on launches without S2_EARLY whose gate / up units are not all entry claims): the shared expert's
// gate / up blocks summed (part order, SwiGLU, sh_h rows, flag) by the warp producers of its down units while they
// wait for them, instead of by the gate / up unit's epilogue that counted the last part (the sum chain delays that
// CTA's next gate / up flags; where its next units are no gate / up units, the epilogue's 128 threads sum sooner)
constexpr unsigned kMoeFp4OptShSum = 32u;
// RT_T0 (in-kernel routing within each CTA): a gate / up unit of token 0's experts (the work items 0 .. top_k - 1
// when all of them are on this rank) issues its first stage once token 0 is routed, before the items
constexpr unsigned kMoeFp4OptRtT0 = 64u;
// S2_EARLY (in-kernel routing within each CTA, NVFP4 routed experts): the shared expert's down units queued right
// after the gate / up units claimed at entry, for the shared gate / up CTAs (see moe_fp4_kernel)
constexpr unsigned kMoeFp4OptS2Early = 256u;
// early items (the in-CTA routing's {key, score} form): the work items built right after the tokens are routed,
// inside the routing block (warmed by the dry pass), and the routing warps of CTAs that write no routing outputs
// skip the weights' sum
constexpr unsigned kMoeFp4OptRtEi = 2048u;
// in-kernel routing limits: tokens (a token bit per local expert), routing slots per token, experts
constexpr int kMoeFp4RtMaxT = 64;
constexpr int kMoeFp4RtMaxK1 = 16;
static_assert(kMoeFp4RtMaxT * kMoeFp4RtMaxK1 == kMoeFp4CtrTagWords, "the tagged ids of every routed pair");
constexpr int kMoeFp4RtMaxE = 32 * 9;
// in-kernel routing: up to this many tokens routed by every CTA itself, more over the CTAs (under MoE tensor
// parallelism up to 8: moe_fp4.cu's kRtLocalTMoeTp)
constexpr int kMoeFp4RtLocalT = 4;

// T: the routed tokens (1 .. kMoeFp4MaxTokens); ctas: persistent CTAs (one per SM)
void moe_fp4_fused(const MoeFp4Args& a, int T, int ctas, cudaStream_t st);
// A dense NVFP4 MLP of T <= 16 tokens as one always-selected expert (the decode steps' dense layers):
// w13 = gate / up [2 Il, H/2] interleaved by 64 (kGu64) with scales sf13 (CUTLASS layout of the
// interleaved rows) and alpha13[0]; w2 = down [H, Il/2], sf2, alpha2[0]; tokens xq / xs (NVFP4,
// CUTLASS-layout scales); hidden rows hq [T, Il/2] / hsa (CUTLASS layout of [T, Il], global scale
// gs2); output y [T, H] (bf16). G = 1; no routing, no shared expert. One persistent CTA per 128-row
// block (Il / 64 gate / up, H / 128 down), each issuing its first weight stages before the grid
// dependency wait. Bitwise nvfp4_gemm -> swiglu_quant_nvfp4 (gu64) -> nvfp4_gemm.
void moe_fp4_dense(const MoeFp4Args& a, int T, cudaStream_t st);
// whether moe_fp4_fused can route in the kernel (MoeFp4Args::rt_*) for T tokens, E routed experts,
// top_k, G local experts
bool moe_fp4_route_in_ok(int T, int E, int top_k, int G);
// whether moe_fp4_fused's in-kernel routing of T tokens takes the router's {key, score} words (MoeFp4Args::rt_ks):
// NVFP4 routed experts under MoE tensor parallelism (Il 512, down units of 2 x 128 rows), the routing within each
// CTA (T <= 8)
bool moe_fp4_route_ks_ok(int T, int H, int Il, int sh_inter, bool routed_bf16);
// the wide variant (NVFP4 routed experts only): activations as the MMA's M, 256 weight rows as N;
// shared expert intermediate % 128, its split-K partials
// sh_part: Is / 128 * kMoeFp4WideSplit * round8(sh_T) * 256 floats
void moe_fp4_wide(const MoeFp4Args& a, int T, int ctas, cudaStream_t st);
constexpr int kMoeFp4WideSplit = 8;

}  // namespace k
}  // namespace glm
