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
// kda_decode_split_kernel's body (kda.cu), also compiled as kda_decode_split_co_kernel's and
// kda_decode_split_lp_kernel's: included inside each kernel with NT, NS, LATE (the late trigger, once this step's
// projections are in), G8 (the 4-CTA form's conv / gate phase: each gate warp copies its own tile rows and gate inputs
// and holds its MMA's B fragments in registers; branch-free quotient tests, the warp sums as one reduce-scatter, packed
// bf16 round trips), C2 (the 4-CTA form at NT = 4: the replays and the token pass with 2 columns per thread on 256
// threads, the token-pair dots by the other 8 warps meanwhile (DT); and the substitution's shared-memory reads ahead of
// its stores (SUB), the replay count loaded with the other per-sequence values (MD)), LP (kda_decode_split_lp_kernel:
// this step's projections from the TP in_proj's Lamport copy) in scope. The 4-CTA form (NS = 4) specializes warps: gate
// warps past the conv threads that copy their own tiles, the l2 norms in the conv warps, branch-free quotients, the
// pre-wait copies by the warps past the gate warps.
  constexpr bool kG8 = G8 && NS == 4;
  constexpr bool kSUB = C2 && NS == 4;
  constexpr bool kMD = (C2 || LP) && NS == 4;  // (LP: its pre-poll section is on the critical path)
  constexpr bool kC2 = C2 && NS == 4 && NT == 4;
  constexpr bool kDT = kC2;
  // (LP: the TP in_proj at one sequence wrote this step's projections into a Lamport copy -- proj, the sentinel 0xFFFF in
  //  every word not yet written -- so the CTA polls the words it reads instead of waiting for the in_proj grid's
  //  completion, waits for that grid only before its first global store, and resets the words it read to the sentinel)
  constexpr bool kLP = LP && kG8;
  constexpr int VC = kD / NS;             // columns of this CTA
  constexpr int TPC = kSplitThreads / VC;  // threads per column
  constexpr int RPT = kD / TPC;           // state rows per thread
  constexpr int W = 4;
  constexpr int NW32 = VC / 32;           // 32-column groups of this CTA
  static_assert(TPC <= 32 && VC % 32 == 0 && NT * VC <= kSplitThreads, "kda split: column layout");
  const int b = blockIdx.x, h = blockIdx.y, cs = blockIdx.z;  // cs: rank in the cluster
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int jj = tid / TPC, g = tid % TPC;  // local column, row group
  const int j = cs * VC + jj;              // head column
  const int c0 = cs * VC;                   // first column (and owned q / k / v channel) of this CTA
  const int HD = H * kD, C = 3 * HD;
  const int t0 = meta.cu_seqlens[b];
  const int slot = meta.slots[b];
  const bool has_state = meta.has_state[b];
  // (MD: the replay count requested with the other per-sequence values, not after has_state arrived: the pre-wait
  //  section's predicated loads -- the records, the draft conv inputs -- then wait for one round trip, not two)
  const int rc_md = kMD && meta.replay ? meta.replay[b] : 0;
  const int rc = kMD ? (has_state ? rc_md : 0) : ((meta.replay && has_state) ? meta.replay[b] : 0);
  __shared__ __align__(16) float sq[NT][kD], sk[NT][kD], sdecay[NT][kD];
  __shared__ __align__(16) float sv[NT][VC], sg2[NT][VC], so[NT][VC];
  __shared__ __align__(16) bf16 sgin[2][8][kD + 8];  // gate inputs f_a / g_a, tokens padded to 8
  __shared__ float red[NT][8];
  __shared__ __align__(16) float srec[3][kKdaRec];
  __shared__ float sbeta[NT];
  __shared__ float cl_ss[4][NT];  // the head's 4 32-column partial sums of squares (cluster-wide)
  __shared__ __align__(8) uint64_t cl_mbar;  // completes when all 4 x NT partial sums have arrived
  __shared__ __align__(8) uint64_t dec_mbar;  // completes when the other CTAs' decay rows have arrived
  // new tokens (WY form, below): the pass's per-row vectors (d_0, k_0, q_0, the drafts' decayed keys
  // / queries) in the threads' row order, the columns' 8 sums, the token-pair dots, k_0 . q_0
  __shared__ __align__(16) float sX[2 * NT + 1][kD];
  __shared__ __align__(16) float sAB[VC][8];
  __shared__ float sM[NT][NT], sN[NT][NT], sKQ;
  // ---- before the wait: the CTA's state columns, the gate weights, the replay records, the conv
  //      windows
  float* S = state + (size_t)slot * meta.state_ss + (size_t)h * kD * kD;
  // the output RMSNorm's partial sums arrive by st.async on cl_mbar (4 x NT floats, one phase); the
  // initialization is published to the cluster by the first barrier phase below
  // the decay gate is split over the cluster: CTA cs computes the decays of state rows [c0, c0 + VC) and
  // sends them to the others' sdecay by st.async on dec_mbar ((NS - 1) x VC x NT floats per CTA)
  constexpr int DW = VC / 16, GW = 2 * VC / 16;  // decay-gate warps, all gate warps
  if (tid == 0) {
    kda_mbar_init_tx(&cl_mbar, 4 * NT * sizeof(float));
    kda_mbar_init_tx(&dec_mbar, (NS - 1) * VC * NT * sizeof(float));
  }
  // first phase of the cluster barrier: the mbarriers are initialized (waited for after the PDL wait,
  // before any st.async into another CTA)
  kda_cl_arrive_relaxed();
  // gate tiles: the f_b rows of this CTA's VC state rows (their decays; the cluster's other CTAs send
  // the rest) and the g_b rows of its columns, [2 VC][128] bf16 swizzled; then the state slice [128][VC + 4]
  // fp32
  // (16-byte copies of whole row segments; threads read their rows from it after the wait and the
  // committed state goes back through it: per-thread row accesses of global memory touch a sector
  // per row, and the stores of the committed state stalled the token loop's shared-memory loads)
  extern __shared__ __align__(128) uint8_t kda_dsm[];
  constexpr int kSP = VC + 4;  // state slice pitch (floats)
  float* sS = reinterpret_cast<float*>(kda_dsm + (size_t)(2 * VC) * kD * 2);
  // (two groups in the order they are needed after the wait: the gate tiles before the gate MMA, the
  // replay records and the state slice before the token pass)
  // (4-CTA form: the gate warps -- threads [kGT0, kGT0 + 128), past the conv threads -- copy the tiles they
  //  use, 8 chunks per thread, and run the gate MMA as soon as their inputs are in, with no block barrier)
  constexpr int kGT0 = NS == 4 ? 2 * kD + VC : 0;
  if constexpr (NS == 4) {
    static_assert(kGT0 % 32 == 0 && 2 * VC / 16 == 4, "kda split: 4 gate warps past the conv threads");
    if (tid >= kGT0 && tid < kGT0 + 128) {
      if constexpr (kG8) {
        // (G8: each gate warp copies the 16 tile rows its MMA reads -- rows [16 w, 16 w + 16) of the [2 VC][128] tile,
        //  w = its gate warp index -- so that it waits for its own copies only)
        const int gw8 = (tid - kGT0) >> 5, r0 = gw8 * 16 + (lane >> 4), c = lane & 15;
        const bf16* tb = (gw8 < DW ? f_b + ((size_t)h * kD + c0 + r0) * kD : g_b + ((size_t)h * kD + c0 + r0 - VC) * kD) + c * 8;
#pragma unroll
        for (int k = 0; k < 8; ++k) kda_cp_async16(kda_dsm + kda_swz(r0 + 2 * k, c), tb + 2 * k * kD);
      } else {
        const int t = tid - kGT0, c = t & 15;
#pragma unroll
        for (int k = 0; k < VC / 8; ++k) {
          const int r = (t >> 4) + 8 * k;
          kda_cp_async16(kda_dsm + kda_swz(r, c), f_b + ((size_t)h * kD + c0 + r) * kD + c * 8);
          kda_cp_async16(kda_dsm + kda_swz(r + VC, c), g_b + ((size_t)h * kD + c0 + r) * kD + c * 8);
        }
      }
    }
  } else {
    for (int i = tid; i < (2 * VC) * 16; i += kSplitThreads) {
      const int r = i / 16, c = i % 16;
      const bf16* src = r < VC ? f_b + ((size_t)h * kD + c0 + r) * kD + c * 8
                               : g_b + ((size_t)h * kD + c0 + r - VC) * kD + c * 8;
      kda_cp_async16(kda_dsm + kda_swz(r, c), src);
    }
  }
  asm volatile("cp.async.commit_group;\n" ::);
  // gate warps gw = wid - kGT0 / 32: 0 .. DW - 1 the decay gate rows c0 + [16 gw, +16), DW .. GW - 1 this
  // CTA's output-gate columns [16 (gw - DW), +16)
  const int gq = lane >> 2, gt = lane & 3;
  const int gw = wid - kGT0 / 32;
  const bool gwarp = (kGT0 == 0 || gw >= 0) && gw < GW;
  const int gsel = gw < DW ? 0 : 1, gj0 = gw < DW ? gw * 16 : (gw - DW) * 16;
  float gdt[2] = {0.f, 0.f};
  if ((kGT0 == 0 || gw >= 0) && gw < DW) {
    gdt[0] = dt_bias[h * kD + c0 + gj0 + gq];
    gdt[1] = dt_bias[h * kD + c0 + gj0 + gq + 8];
  }
  const float* Rp = rec + (size_t)slot * meta.rec_ss + (size_t)h * kKdaRec;
  static_assert(3 * (kKdaRec / 4) <= kSplitThreads, "kda split: one record chunk per thread");
  if constexpr (NS == 4) {
    // (the replay records and the state slice -- needed only at the copies barrier, well after the wait -- by the
    //  3 warps past the gate warps, which have no work before it but the betas: the conv and gate warps issue only
    //  their own loads before the wait; compile-time trip counts)
    constexpr int kCT0 = kGT0 + 128, kCN = kSplitThreads - kCT0;
    static_assert(kCN == 96, "kda split: 3 copy warps");
    if (tid >= kCT0) {
      const int t = tid - kCT0;
#pragma unroll
      for (int k = 0; k < (3 * (kKdaRec / 4) + kCN - 1) / kCN; ++k) {
        const int q = t + k * kCN;
        if (q < rc * (kKdaRec / 4)) {  // (rc <= 3)
          const int r = q / (kKdaRec / 4), c4 = q % (kKdaRec / 4);
          kda_cp_async16(&srec[r][c4 * 4], Rp + (size_t)r * H * kKdaRec + c4 * 4);
        }
      }
      if (has_state) {
#pragma unroll
        for (int k = 0; k < (kD * (VC / 4) + kCN - 1) / kCN; ++k) {
          const int i = t + k * kCN;
          if (i < kD * (VC / 4)) {
            const int r = i / (VC / 4), c4 = i % (VC / 4);
            kda_cp_async16(sS + r * kSP + c4 * 4, S + (size_t)r * kD + c0 + c4 * 4);
          }
        }
      }
    }
  } else {
    for (int q = tid; q < rc * (kKdaRec / 4); q += kSplitThreads) {
      const int r = q / (kKdaRec / 4), c4 = q % (kKdaRec / 4);
      kda_cp_async16(&srec[r][c4 * 4], Rp + (size_t)r * H * kKdaRec + c4 * 4);
    }
    if (has_state) {
      for (int i = tid; i < kD * (VC / 4); i += kSplitThreads) {
        const int r = i / (VC / 4), c4 = i % (VC / 4);
        kda_cp_async16(sS + r * kSP + c4 * 4, S + (size_t)r * kD + c0 + c4 * 4);
      }
    }
  }
  asm volatile("cp.async.commit_group;\n" ::);
  // conv channels: q (all 128), k (all 128), v (this CTA's columns); the conv window and draft
  // inputs of channel cc are written by the CTA owning column cc (q / k / v alike)
  const int nconv = 2 * kD + VC;
  const bool conv_t = tid < nconv;
  const int sec = tid < kD ? 0 : (tid < 2 * kD ? 1 : 2);
  const int cc = sec < 2 ? tid - sec * kD : c0 + (tid - 2 * kD);
  const bool owns = cc >= c0 && cc < c0 + VC;
  const int ch = sec * HD + h * kD + cc;
  // (raw: the loads are only issued here; the values are consumed after the PDL wait, so that a CTA that
  // enters late -- when the in_proj's CTAs exit -- requests this step's projections without waiting for
  // these loads first)
  bf16 winr[W - 1], crr[3], cwr[W];
  if (conv_t) {
    const bf16* stp = conv_state + (size_t)slot * meta.conv_ss + (size_t)ch * (W - 1);
    const bf16 z = __float2bfloat16(0.f);
#pragma unroll
    for (int q = 0; q < W - 1; ++q) winr[q] = has_state ? stp[q] : z;
#pragma unroll
    for (int i = 0; i < 3; ++i) crr[i] = i < rc ? crec[kda_crec_off(meta, slot, i, sec, ch - sec * HD)] : z;
#pragma unroll
    for (int q = 0; q < W; ++q) cwr[q] = conv_w[(size_t)ch * W + q];
  }
  const float alog = a_log[h];
  // the output thread's norm weight (thread q < NT x VC: token q / VC, column c0 + q % VC)
  const bf16 onr = tid < NT * VC ? o_norm[c0 + tid % VC] : __float2bfloat16(0.f);
  // (LP: this step's projections polled in the Lamport copy -- betas, conv inputs, gate inputs (into sgin), each word by
  //  the thread that reads it, every poll an L2 load -- written ahead of the pdl_wait() text below, so that no load of this
  //  form follows a PDL wait in the source; the CTA waits for the in_proj grid before its first global store, below)
  [[maybe_unused]] bf16 lp_beta, lp_x[NT];
  if constexpr (kLP) {
    kda_cl_wait();
    if (tid >= kGT0 + 128 && tid < kGT0 + 128 + NT) lp_beta = kda_lp_ld(proj + (size_t)(t0 + tid - kGT0 - 128) * ld + 3 * HD + h);
    if (conv_t) kda_lp_ldn<NT>(lp_x, proj + (size_t)t0 * ld + ch, ld);
    if (gwarp) {
      const int q = (gw & 1) * 32 + lane, tok = q >> 4, c16 = q & 15;
      if (tok < NT)
        *reinterpret_cast<uint4*>(&sgin[gsel][tok][c16 * 8]) =
            kda_lp_ld16(proj + (size_t)(t0 + tok) * ld + 3 * HD + H + gsel * kD + c16 * 8);
      asm volatile("cp.async.commit_group;\n" ::);  // (the group count of the cp.async form)
    }
  }
  if constexpr (!kLP) pdl_wait();
  if constexpr (!kLP) kda_cl_wait();
  // ---- this step's projections (betas, conv inputs, gate inputs) requested first (the predecessor's
  //      output through L2: __ldcg loads are not hoisted above the wait)
  bf16 betar = __float2bfloat16(0.f), xr[NT], fvr[NT], gvr[NT];
  constexpr int kBT0 = NS == 4 ? kGT0 + 128 : 0;  // the betas' threads (4-CTA form: past the gate warps)
  if constexpr (kLP) {
    if (tid >= kBT0 && tid < kBT0 + NT) betar = lp_beta;
    if (conv_t) {
#pragma unroll
      for (int i = 0; i < NT; ++i) xr[i] = lp_x[i];
    }
  } else {
    if (tid >= kBT0 && tid < kBT0 + NT) betar = kda_ldbf(proj + (size_t)(t0 + tid - kBT0) * ld + 3 * HD + h);
    if (conv_t) {
#pragma unroll
      for (int i = 0; i < NT; ++i) xr[i] = kda_ldbf(proj + (size_t)(t0 + i) * ld + ch);
    }
  }
  const bool gin_t = !kG8 && tid >= nconv && tid < nconv + kD;
  // (G8: each gate warp pair -- decay: f inputs, output gate: g inputs -- copies its NT x 128 inputs into sgin[gsel] by
  //  16-byte cp.async, a chunk per lane: the 64 CTAs all read these 2 KB, so the requests stay at the staged form's 64 sectors
  //  per CTA (B fragments loaded straight by every gate warp would be 4x that: an L2 hot spot); no 4-warp barrier)
  if constexpr (kG8 && !kLP) {
    if (gwarp) {
      const int q = (gw & 1) * 32 + lane, tok = q >> 4, c16 = q & 15;
      if (tok < NT)
        kda_cp_async16(&sgin[gsel][tok][c16 * 8], proj + (size_t)(t0 + tok) * ld + 3 * HD + H + gsel * kD + c16 * 8);
      asm volatile("cp.async.commit_group;\n" ::);
    }
  }
  if (gin_t) {
    const int c2 = tid - nconv;
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const bf16* row = proj + (size_t)(t0 + i) * ld;
      fvr[i] = kda_ldbf(row + 3 * HD + H + c2);
      gvr[i] = kda_ldbf(row + 3 * HD + H + kD + c2);
    }
  }
  float win[W - 1 + NT], cw[W];
  if (conv_t) {
    float cr[3];
#pragma unroll
    for (int q = 0; q < W - 1; ++q) win[q] = __bfloat162float(winr[q]);
#pragma unroll
    for (int i = 0; i < 3; ++i) cr[i] = __bfloat162float(crr[i]);
#pragma unroll
    for (int i = 0; i < 3; ++i)  // window after the accepted drafts of the previous step
      if (i < rc) {
#pragma unroll
        for (int q = 0; q < W - 2; ++q) win[q] = win[q + 1];
        win[W - 2] = cr[i];
      }
#pragma unroll
    for (int q = 0; q < W; ++q) cw[q] = __bfloat162float(cwr[q]);
  }
  if (conv_t) {  // (the conv window's loads completed: an instruction reads their values)
    float chk = 0.f;
#pragma unroll
    for (int q = 0; q < W - 1; ++q) chk += win[q] + __bfloat162float(crr[q]);
    asm volatile("{\n.reg .f32 t;\nmov.f32 t, %0;\n}\n" ::"f"(chk));
  }
  const float amp = expf(alog);
  const float onw = __bfloat162float(onr);
  if (tid >= kBT0 && tid < kBT0 + NT) sbeta[tid - kBT0] = 1.f / (1.f + __expf(-__bfloat162float(betar)));
  if (conv_t) {
#pragma unroll
    for (int i = 0; i < NT; ++i) win[W - 1 + i] = __bfloat162float(xr[i]);
  }
  // (LATE, the co / LP instances: the late trigger once this step's projections are in, so that the dependents' CTAs
  // (the o_proj's, beside this kernel's) stage their weights after this kernel's first loads, not in front of them)
  if (LATE) pdl_trigger();
  if (gin_t) {
    const int c2 = tid - nconv;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const bf16 z = __float2bfloat16(0.f);
      sgin[0][i][c2] = i < NT ? fvr[i < NT ? i : 0] : z;
      sgin[1][i][c2] = i < NT ? gvr[i < NT ? i : 0] : z;
    }
  }
  // conv outputs; the q / k threads (warps 0-3 / 4-7, one channel each) also reduce the l2 norms'
  // sums of squares per warp (as the 512-thread form's norm pass)
  float y[NT];
  if constexpr (NS == 4) {  // (the NT quotients without branches between them: kda_div_n)
    if (conv_t) {
      float ca[NT], cb[NT];
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        float acc = 0.f;
#pragma unroll
        for (int q = 0; q < W; ++q) acc += cw[q] * win[i + q];
        ca[i] = acc;
        if constexpr (kG8)
          cb[i] = kda_exp1p(-acc);
        else
          cb[i] = 1.f + __expf(-acc);
      }
      if constexpr (kG8) {
        kda_div_nb<NT>(y, ca, cb);
#pragma unroll
        for (int i = 0; i + 1 < NT; i += 2) kda_bf16r2(y[i], y[i + 1]);
        if constexpr ((NT & 1) != 0) y[NT - 1] = bf16r(y[NT - 1]);
#pragma unroll
        for (int i = 0; i < NT; ++i)
          if (sec == 2) sv[i][cc - c0] = y[i];
      } else {
        kda_div_n<NT>(y, ca, cb);
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          y[i] = bf16r(y[i]);
          if (sec == 2) sv[i][cc - c0] = y[i];
        }
      }
    }
  } else if (conv_t) {
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      float acc = 0.f;
#pragma unroll
      for (int q = 0; q < W; ++q) acc += cw[q] * win[i + q];
      y[i] = bf16r(acc / (1.f + __expf(-acc)));
      if (sec == 2) sv[i][cc - c0] = y[i];
    }
  }
  if constexpr (NS == 4) {
    // q / k (warps 0-3 / 4-7): the l2 norms' sums of squares per warp, then (named barrier of the part's 4
    // warps) each thread normalizes its own values (same arithmetic as the norm pass of the other form)
    if (tid < 2 * kD) {
      if constexpr (kG8) {  // (one reduce-scatter: lanes 8 i hold token i's sum)
        float v4[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v4[i] = i < NT ? y[i < NT ? i : 0] * y[i < NT ? i : 0] : 0.f;
        const float ss = kda_warp_sum4(v4, lane);
        if ((lane & 7) == 0 && (lane >> 3) < NT) red[lane >> 3][wid] = ss;
      } else {
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          const float ss = warp_sum(y[i] * y[i]);
          if (lane == 0) red[i][wid] = ss;
        }
      }
      kda_bar_sync(1 + sec, kD);
      float nx[NT], nd[NT], nq[NT];
#pragma unroll
      for (int i = 0; i < NT; ++i)
        nx[i] = red[i][sec * 4] + red[i][sec * 4 + 1] + red[i][sec * 4 + 2] + red[i][sec * 4 + 3] + 1e-6f;
      kda_sqrt_n<NT>(nd, nx);
      if constexpr (kG8)
        kda_div_nb<NT>(nq, y, nd);
      else
        kda_div_n<NT>(nq, y, nd);
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        if (sec)
          sk[i][cc] = nq[i];
        else
          sq[i][cc] = nq[i] * scale;
      }
    }
    // ---- gate projections (tensor cores) by the 4 gate warps once their tiles and inputs are in
    if (gwarp) {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      const uint8_t* wt = kda_dsm + (size_t)(gsel ? VC : 0) * kD * 2;
      if constexpr (kG8) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");  // (its tile rows and gate-input chunks)
        kda_bar_sync(3 + gsel, 64);  // (the pair's gate inputs and tile rows)
        uint32_t gbf[kD / 16][2];  // (tokens >= NT: zeros, as the staged form's padding rows)
#pragma unroll
        for (int ks = 0; ks < kD / 16; ++ks) {
          gbf[ks][0] = gq < NT ? *reinterpret_cast<const uint32_t*>(&sgin[gsel][gq < NT ? gq : 0][ks * 16 + 2 * gt]) : 0u;
          gbf[ks][1] = gq < NT ? *reinterpret_cast<const uint32_t*>(&sgin[gsel][gq < NT ? gq : 0][ks * 16 + 2 * gt + 8]) : 0u;
        }
        kda_gate_mma_r(c, wt, gbf, gj0, lane);
      } else {
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");  // (the gate warps' tile copies)
        kda_bar_sync(3, 128);  // (the tiles and sgin, written by the 4 gate warps)
        kda_gate_mma(c, wt, sgin[gsel], gj0, lane);
      }
      // accumulator e of lane (gq, gt): row gj0 + gq + 8 (e >> 1), token 2 gt + (e & 1); tokens >= 4 are
      // padding (NT <= 4), so the lanes gt = 2, 3 take the row + 8 elements of lanes gt - 2: two
      // exp / divide / exp chains per lane instead of four (same values)
      const float u2 = __shfl_up_sync(0xffffffffu, c[2], 2), u3 = __shfl_up_sync(0xffffffffu, c[3], 2);
      const bool hi = gt >= 2;
      const float d0 = hi ? u2 : c[0], d1 = hi ? u3 : c[1];
      const int nb = 2 * (gt & 1), jo = gj0 + gq + (hi ? 8 : 0);
      const float dtb = hi ? gdt[1] : gdt[0];
      if (gsel == 0) {  // (both quotients without a branch between them: kda_div_n)
        float ga[2], gb[2], gr[2];
        if constexpr (kG8) {
          float gg[2] = {d0, d1};
          kda_bf16r2(gg[0], gg[1]);
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            ga[e] = lower_bound;
            gb[e] = kda_exp1p(-amp * (gg[e] + dtb));
          }
          kda_div_nb<2>(gr, ga, gb);
        } else {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float gg = bf16r(e ? d1 : d0);
            ga[e] = lower_bound;
            gb[e] = 1.f + __expf(-amp * (gg + dtb));
          }
          kda_div_n<2>(gr, ga, gb);
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int n = nb + e;
          if (n < NT) {
            const float dv = __expf(gr[e]);
            sdecay[n][c0 + jo] = dv;
#pragma unroll
            for (int r = 1; r < NS; ++r) kda_dsm_st_async(&sdecay[n][c0 + jo], &dec_mbar, (cs + r) % NS, dv);
          }
        }
      } else if constexpr (kG8) {
        float g2[2] = {d0, d1};
        kda_bf16r2(g2[0], g2[1]);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int n = nb + e;
          if (n < NT) sg2[n][jo] = g2[e];
        }
      } else {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int n = nb + e;
          if (n < NT) sg2[n][jo] = bf16r(e ? d1 : d0);
        }
      }
    }
  } else {
    if (tid < 2 * kD) {
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const float ss = warp_sum(y[i] * y[i]);
        if (lane == 0) red[i][wid] = ss;
        if (tid < kD) sq[i][cc] = y[i];  // (raw: normalized below by the warps without gate work)
        else sk[i][cc] = y[i];
      }
    }
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");  // (the gate tiles: published by the barrier)
    __syncthreads();
    // ---- gate projections (tensor cores), as kda_decode_fused4_kernel
    if (gwarp) {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      const uint8_t* wt = kda_dsm + (size_t)(gsel ? VC : 0) * kD * 2;
      kda_gate_mma(c, wt, sgin[gsel], gj0, lane);
      // accumulator e of lane (gq, gt): row gj0 + gq + 8 (e >> 1), token 2 gt + (e & 1); tokens >= 4 are
      // padding (NT <= 4), so the lanes gt = 2, 3 take the row + 8 elements of lanes gt - 2: two
      // exp / divide / exp chains per lane instead of four (same values)
      const float u2 = __shfl_up_sync(0xffffffffu, c[2], 2), u3 = __shfl_up_sync(0xffffffffu, c[3], 2);
      const bool hi = gt >= 2;
      const float d0 = hi ? u2 : c[0], d1 = hi ? u3 : c[1];
      const int nb = 2 * (gt & 1), jo = gj0 + gq + (hi ? 8 : 0);
      const float dtb = hi ? gdt[1] : gdt[0];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int n = nb + e;
        if (n < NT) {
          const float gg = bf16r(e ? d1 : d0);
          if (gsel == 0) {
            const float dv = __expf(lower_bound / (1.f + __expf(-amp * (gg + dtb))));
            sdecay[n][c0 + jo] = dv;
#pragma unroll
            for (int r = 1; r < NS; ++r) kda_dsm_st_async(&sdecay[n][c0 + jo], &dec_mbar, (cs + r) % NS, dv);
          } else {
            sg2[n][jo] = gg;
          }
        }
      }
    }
    // ---- l2 norms of q and k from the conv phase's warp sums (same arithmetic as
    //      kda_decode_fused4_kernel's norm pass), by the warps past the gate warps while those run the
    //      MMA: in place on the raw values
    {
      constexpr int kNormT0 = 32 * GW, kNormN = kSplitThreads - kNormT0;
      static_assert(kNormN >= 64, "kda split: threads for the norm pass");
      for (int q = tid - kNormT0; q >= 0 && q < 2 * kD * NT; q += kNormN) {
        const int part = q / (kD * NT), i = (q / kD) % NT, c = q % kD;
        const float n2 = red[i][part * 4] + red[i][part * 4 + 1] + red[i][part * 4 + 2] +
                         red[i][part * 4 + 3];
        if (part)
          sk[i][c] = sk[i][c] / sqrtf(n2 + 1e-6f);
        else
          sq[i][c] = sq[i][c] / sqrtf(n2 + 1e-6f) * scale;
      }
    }
}
  __syncthreads();
  kda_mbar_wait0(&dec_mbar);  // (the other CTAs' decay rows)
  // ---- the new tokens' terms that do not depend on the state (WY form, below), with
  //      Q_i = d_1 ... d_i and R_im = d_{m+1} ... d_i (per row): kQ_i = k_i Q_i, qQ_i = q_i Q_i,
  //      M_im = k_i . (R_im k_m) (m < i), N_im = q_i . (R_im k_m) (m <= i), one dot per warp; and
  //      k_0 . q_0 summed as a column's threads do (rows g + TPC r in order, then the xor tree)
  if constexpr (NS == 4) {
    // The per-row vectors of the pass below -- d_0, k_0, q_0, then kQ_i, qQ_i -- go to sX with each
    // thread's rows g + TPC (4 r4 + e), e = 0..3, in one 16-byte chunk [r4][g]; thread (part, row)
    // writes row's d_0, k_0, q_0 (part 0) or kQ_part, qQ_part (Q_part once for both; part is warp-uniform)
    static_assert(NT * kD <= kSplitThreads, "kda split: a thread per row and token");
    if (tid < NT * kD) {
      const int part = tid / kD, row = tid % kD;
      const int r = row / TPC, pos = ((r >> 2) * TPC + row % TPC) * 4 + (r & 3);
      if (part == 0) {
        sX[0][pos] = sdecay[0][row];
        sX[1][pos] = sk[0][row];
        sX[2][pos] = sq[0][row];
      } else {
        float qr = 1.f;
#pragma unroll
        for (int l = 1; l < NT; ++l)
          if (l <= part) qr *= sdecay[l][row];
        sX[2 * part + 1][pos] = sk[part][row] * qr;
        sX[2 * part + 2][pos] = sq[part][row] * qr;
      }
    }
  } else {
    // The per-row vectors of the pass below -- d_0, k_0, q_0, then kQ_i, qQ_i -- go to sX with each
    // thread's rows g + TPC (4 r4 + e), e = 0..3, in one 16-byte chunk [r4][g]; an element per
    // thread (all warps), the products unrolled with predicates
    for (int el = tid; el < (2 * NT + 1) * kD; el += kSplitThreads) {
      const int v = el / kD, row = el % kD;
      const int r = row / TPC, pos = ((r >> 2) * TPC + row % TPC) * 4 + (r & 3);
      float x;
      if (v < 3) {
        x = v == 0 ? sdecay[0][row] : v == 1 ? sk[0][row] : sq[0][row];
      } else {
        const int i = (v - 1) >> 1;  // v = 2 i + 1: kQ_i, 2 i + 2: qQ_i
        float qr = 1.f;
#pragma unroll
        for (int l = 1; l < NT; ++l)
          if (l <= i) qr *= sdecay[l][row];
        x = ((v & 1) ? sk[i][row] : sq[i][row]) * qr;
      }
      sX[v][pos] = x;
    }
  }
  if constexpr (kDT) {
    // (C2: the dots in the token pass, below)
  } else if constexpr (NT > 1) {
    constexpr int NM = NT * (NT - 1) / 2, NN = NM + NT - 1;
    static_assert(NM + NN < kSplitThreads / 32, "kda split: a warp per dot and one for k_0 . q_0");
    if (wid < NM + NN) {
      const bool isn = wid >= NM;
      int i, m;
      if constexpr (NS == 4) {
        constexpr unsigned long long kPairs = kda_dot_pairs(NT);  // (i, m) of dot warp wid
        const unsigned pr = (unsigned)(kPairs >> (4 * wid)) & 15u;
        i = pr & 3;
        m = pr >> 2;
      } else {
        int w = isn ? wid - NM : wid;  // (i, m) of dot w: M by i then m < i, N by i then m <= i
        i = 1;
        for (;;) {
          const int cnt = isn ? i + 1 : i;
          if (w < cnt) break;
          w -= cnt;
          ++i;
        }
        m = w;
      }
      float acc = 0.f;
#pragma unroll
      for (int e = 0; e < kD / 32; ++e) {
        const int row = lane + 32 * e;
        float r = 1.f;
#pragma unroll
        for (int l = 1; l < NT; ++l)
          if (l > m && l <= i) r *= sdecay[l][row];
        acc += (isn ? sq[i][row] : sk[i][row]) * r * sk[m][row];
      }
      acc = warp_sum(acc);
      if (lane == 0) (isn ? sN : sM)[i][m] = acc;
    }
  }
  if (wid == kSplitThreads / 32 - 1) {
    float kq = 0.f;
    if (lane < TPC) {
#pragma unroll
      for (int r = 0; r < RPT; ++r) kq += sk[0][lane + TPC * r] * sq[0][lane + TPC * r];
    }
#pragma unroll
    for (int o = TPC / 2; o > 0; o >>= 1) kq += __shfl_xor_sync(0xffffffffu, kq, o);
    if (lane == 0) sKQ = kq;
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");  // (records + state slice: published below)
  __syncthreads();
  // ---- the previous step's accepted drafts (the state slice and the replay records only): here, so
  //      that a CTA entering late (as the in_proj's CTAs exit) runs the conv / gate / W1 phases while its
  //      state slice is still arriving
  if constexpr (kC2) {
    // (C2: the replays and the token pass by 2 x 128 threads with 2 columns each: thread (column pair cp, row group g)
    //  holds rows g + TPC r of the local columns 2 cp, 2 cp + 1, so each row's record values, decays and WY vectors are
    //  loaded once for both columns -- half the shared-memory wavefronts of the replays and of the WY products, the MIO
    //  floor of both phases. Per column the same operations in the same order as the 1-column form (the replays element
    //  by element; the WY sums over the same rows in the same order, then the same xor tree over the 16 row groups): bitwise)
    static_assert(2 * NT <= 8 && TPC == 16 && RPT % 4 == 0 && VC == 32, "kda split: C2's layout");
    if (tid < (VC / 2) * TPC) {
      const int ca = 2 * (tid / TPC);  // (g = tid % TPC as in the 1-column form)
      float s2[2][RPT];
#pragma unroll
      for (int r = 0; r < RPT; ++r) {
        const float2 v2 = has_state ? *reinterpret_cast<const float2*>(&sS[(g + TPC * r) * kSP + ca]) : make_float2(0.f, 0.f);
        s2[0][r] = v2.x;
        s2[1][r] = v2.y;
      }
#pragma unroll
      for (int t = 0; t < 3; ++t) {
        if (t < rc) {
          const float* kv = srec[t];
          const float* dv = srec[t] + 2 * kD;
          const float2 u2 = *reinterpret_cast<const float2*>(&srec[t][kD + c0 + ca]);
#pragma unroll
          for (int r = 0; r < RPT; ++r) {
            const int row = g + TPC * r;
            const float d = dv[row], k = kv[row];
            // (explicit: the 1-column loop's s *= d; s += k u compiles to fma(s, d, k u) -- SASS: FMUL k u, FFMA s d + ku)
            s2[0][r] = __fmaf_rn(s2[0][r], d, __fmul_rn(k, u2.x));
            s2[1][r] = __fmaf_rn(s2[1][r], d, __fmul_rn(k, u2.y));
          }
        }
      }
      kda_cl_arrive_relaxed();  // (as below: this thread's reads of the records are done)
      float acc[2][8];
      float k0r[RPT];
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int v = 0; v < 8; ++v) acc[c][v] = 0.f;
#pragma unroll
      for (int r4 = 0; r4 < RPT / 4; ++r4) {
        const int off = (r4 * TPC + g) * 4;
        const float4 d4 = *reinterpret_cast<const float4*>(&sX[0][off]);
        const float4 k4 = *reinterpret_cast<const float4*>(&sX[1][off]);
        const float4 q4 = *reinterpret_cast<const float4*>(&sX[2][off]);
        const float dd[4] = {d4.x, d4.y, d4.z, d4.w}, kd[4] = {k4.x, k4.y, k4.z, k4.w},
                    qd[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int r = 4 * r4 + e;
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            s2[c][r] *= dd[e];
            acc[c][0] += s2[c][r] * kd[e];
            acc[c][1] += s2[c][r] * qd[e];
          }
          k0r[r] = kd[e];
        }
#pragma unroll
        for (int i = 1; i < NT; ++i) {
          const float4 a4 = *reinterpret_cast<const float4*>(&sX[2 * i + 1][off]);
          const float4 c4 = *reinterpret_cast<const float4*>(&sX[2 * i + 2][off]);
          const float ad[4] = {a4.x, a4.y, a4.z, a4.w}, cd[4] = {c4.x, c4.y, c4.z, c4.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
              acc[c][2 * i] += s2[c][4 * r4 + e] * ad[e];
              acc[c][2 * i + 1] += s2[c][4 * r4 + e] * cd[e];
            }
          }
        }
      }
      // per column: the 1-column form's reduce-scatter along the xor tree over the 16 row groups (lanes xor 8, 4, 2, 1)
#pragma unroll
      for (int c = 0; c < 2; ++c) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const bool hi = g & (TPC / 2);
          const float send = hi ? acc[c][k] : acc[c][k + 4], keep = hi ? acc[c][k + 4] : acc[c][k];
          acc[c][k] = keep + __shfl_xor_sync(0xffffffffu, send, TPC / 2);
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const bool hi = g & (TPC / 4);
          const float send = hi ? acc[c][k] : acc[c][k + 2], keep = hi ? acc[c][k + 2] : acc[c][k];
          acc[c][k] = keep + __shfl_xor_sync(0xffffffffu, send, TPC / 4);
        }
        {
          const bool hi = g & (TPC / 8);
          const float send = hi ? acc[c][0] : acc[c][1], keep = hi ? acc[c][1] : acc[c][0];
          acc[c][0] = keep + __shfl_xor_sync(0xffffffffu, send, TPC / 8);
        }
#pragma unroll
        for (int o = TPC / 16; o > 0; o >>= 1) acc[c][0] += __shfl_xor_sync(0xffffffffu, acc[c][0], o);
        if (g % (TPC / 8) == 0) sAB[ca + c][g / (TPC / 8)] = acc[c][0];
      }
      __syncwarp();
      float tot[2][8];
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        *reinterpret_cast<float4*>(&tot[c][0]) = *reinterpret_cast<const float4*>(&sAB[ca + c][0]);
        *reinterpret_cast<float4*>(&tot[c][4]) = *reinterpret_cast<const float4*>(&sAB[ca + c][4]);
      }
      if constexpr (kDT) kda_bar_sync(7, kSplitThreads);  // (sM / sN from warps 8-15)
      float u[2][NT], ov[2][NT];
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        u[c][0] = sbeta[0] * (sv[0][ca + c] - tot[c][0]);
        ov[c][0] = tot[c][1] + sKQ * u[c][0];
#pragma unroll
        for (int i = 1; i < NT; ++i) {
          float a = tot[c][2 * i], o = tot[c][2 * i + 1];
#pragma unroll
          for (int m = 0; m < i; ++m) a += sM[i][m] * u[c][m];
          u[c][i] = sbeta[i] * (sv[i][ca + c] - a);
#pragma unroll
          for (int m = 0; m <= i; ++m) o += sN[i][m] * u[c][m];
          ov[c][i] = o;
        }
      }
      if (g == 0) {
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
          for (int i = 0; i < NT; ++i) so[i][ca + c] = bf16r(ov[c][i]);
      }
      __syncwarp();
      if (g == 0) {
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
          for (int i = 1; i < NT; ++i) sv[i][ca + c] = u[c][i];
      }
#pragma unroll
      for (int r = 0; r < RPT; ++r)
        *reinterpret_cast<float2*>(&sS[(g + TPC * r) * kSP + ca]) =
            make_float2(s2[0][r] + k0r[r] * u[0][0], s2[1][r] + k0r[r] * u[1][0]);
    } else {
      kda_cl_arrive_relaxed();
      if constexpr (kDT) {
        // (DT: warps 8-15 -- idle in the C2 token pass -- compute the 15 token-pair dots meanwhile: warp 8 + w, w < 6, the
        //  pair M_im / N_im of (i, m) = (1,0) (2,0) (2,1) (3,0) (3,1) (3,2), which share R_im and k_m; warp 14 N_11 / N_22,
        //  warp 15 N_33. Per dot the dot warps' per-row products and order (R_im from 1.f in l order, then fmaf((x R_im), k_m,
        //  acc) over the lane's 4 rows), then one 2-value reduce-scatter along warp_sum's xor tree: each sum bitwise warp_sum's)
        kda_dt_dots<NT>(sdecay, sk, sq, sM, sN, lane, wid);
        kda_bar_arrive(7, kSplitThreads);
      }
    }
  } else {
    float s[RPT], sc[RPT];
#pragma unroll
    for (int r = 0; r < RPT; ++r) s[r] = has_state ? sS[(g + TPC * r) * kSP + jj] : 0.f;
#pragma unroll
    for (int t = 0; t < 3; ++t) {
      if (t < rc) {
        const float* kv = srec[t];
        const float* dv = srec[t] + 2 * kD;
        const float u = srec[t][kD + j];  // (the records carry u: no dot products)
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
          const int row = g + TPC * r;
          s[r] *= dv[row];
          s[r] += kv[row] * u;
        }
      }
    }
    // the second phase of the cluster barrier: this CTA's reads of the conv windows, draft inputs and
    // replay records (shared with the cluster) are done; it is waited for before they are rewritten.
    // Relaxed (no GPU-scope fence): the records were consumed by the replays, and the conv window's
    // loads were completed by an instruction that reads their values
    kda_cl_arrive_relaxed();
    // ---- the new tokens in one pass (WY form). With x = d_0 S (the state decayed for token 0),
    //      token i's decayed state is Q_i x + sum_{m<i} R_im k_m u_m, so
    //        u_i = beta_i (v_i - kQ_i . x - sum_{m<i} M_im u_m),  o_i = qQ_i . x + sum_{m<=i} N_im u_m
    //      and the committed state is x + k_0 u_0. Token 0's sums keep the serial loop's order (its
    //      output and the state are bitwise the same); the drafts' are reassociated.
    {
      static_assert(2 * NT <= 8 && TPC >= 8 && RPT % 4 == 0, "kda split: the WY pass's layout");
      float acc[8];  // kQ_i . x, qQ_i . x (i = 0: k_0, q_0) over this thread's rows; 0-padded to 8
      float k0r[RPT];
#pragma unroll
      for (int v = 0; v < 8; ++v) acc[v] = 0.f;
#pragma unroll
      for (int r4 = 0; r4 < RPT / 4; ++r4) {
        const int off = (r4 * TPC + g) * 4;
        const float4 d4 = *reinterpret_cast<const float4*>(&sX[0][off]);
        const float4 k4 = *reinterpret_cast<const float4*>(&sX[1][off]);
        const float4 q4 = *reinterpret_cast<const float4*>(&sX[2][off]);
        const float dd[4] = {d4.x, d4.y, d4.z, d4.w}, kd[4] = {k4.x, k4.y, k4.z, k4.w},
                    qd[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int r = 4 * r4 + e;
          s[r] *= dd[e];
          acc[0] += s[r] * kd[e];
          acc[1] += s[r] * qd[e];
          k0r[r] = kd[e];
        }
#pragma unroll
        for (int i = 1; i < NT; ++i) {
          const float4 a4 = *reinterpret_cast<const float4*>(&sX[2 * i + 1][off]);
          const float4 c4 = *reinterpret_cast<const float4*>(&sX[2 * i + 2][off]);
          const float ad[4] = {a4.x, a4.y, a4.z, a4.w}, cd[4] = {c4.x, c4.y, c4.z, c4.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            acc[2 * i] += s[4 * r4 + e] * ad[e];
            acc[2 * i + 1] += s[4 * r4 + e] * cd[e];
          }
        }
      }
      // sums over the column's TPC lanes as a reduce-scatter along the xor tree (each kept value
      // gets the butterfly's pairs: token 0's sums are bitwise the serial loop's), then the 8 sums
      // through shared memory to every lane of the column
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const bool hi = g & (TPC / 2);
        const float send = hi ? acc[k] : acc[k + 4], keep = hi ? acc[k + 4] : acc[k];
        acc[k] = keep + __shfl_xor_sync(0xffffffffu, send, TPC / 2);
      }
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const bool hi = g & (TPC / 4);
        const float send = hi ? acc[k] : acc[k + 2], keep = hi ? acc[k + 2] : acc[k];
        acc[k] = keep + __shfl_xor_sync(0xffffffffu, send, TPC / 4);
      }
      {
        const bool hi = g & (TPC / 8);
        const float send = hi ? acc[0] : acc[1], keep = hi ? acc[1] : acc[0];
        acc[0] = keep + __shfl_xor_sync(0xffffffffu, send, TPC / 8);
      }
#pragma unroll
      for (int o = TPC / 16; o > 0; o >>= 1) acc[0] += __shfl_xor_sync(0xffffffffu, acc[0], o);
      if (g % (TPC / 8) == 0) sAB[jj][g / (TPC / 8)] = acc[0];
      __syncwarp();
      float tot[8];
      *reinterpret_cast<float4*>(&tot[0]) = *reinterpret_cast<const float4*>(&sAB[jj][0]);
      *reinterpret_cast<float4*>(&tot[4]) = *reinterpret_cast<const float4*>(&sAB[jj][4]);
      float u[NT];
      if constexpr (kSUB) {
        // (SUB: the outputs kept in registers and stored after the chain, so that its shared-memory reads -- betas, v,
        //  the token-pair dots -- are issued together ahead of it instead of one token's after the previous one's stores;
        //  the same operations)
        float ov[NT];
        u[0] = sbeta[0] * (sv[0][jj] - tot[0]);
        ov[0] = tot[1] + sKQ * u[0];
#pragma unroll
        for (int i = 1; i < NT; ++i) {
          float a = tot[2 * i], o = tot[2 * i + 1];
#pragma unroll
          for (int m = 0; m < i; ++m) a += sM[i][m] * u[m];
          u[i] = sbeta[i] * (sv[i][jj] - a);
#pragma unroll
          for (int m = 0; m <= i; ++m) o += sN[i][m] * u[m];
          ov[i] = o;
        }
        if (g == 0) {
#pragma unroll
          for (int i = 0; i < NT; ++i) so[i][jj] = bf16r(ov[i]);
        }
      } else {
        u[0] = sbeta[0] * (sv[0][jj] - tot[0]);
        if (g == 0) so[0][jj] = bf16r(tot[1] + sKQ * u[0]);
#pragma unroll
        for (int i = 1; i < NT; ++i) {
          float a = tot[2 * i], o = tot[2 * i + 1];
#pragma unroll
          for (int m = 0; m < i; ++m) a += sM[i][m] * u[m];
          u[i] = sbeta[i] * (sv[i][jj] - a);
#pragma unroll
          for (int m = 0; m <= i; ++m) o += sN[i][m] * u[m];
          if (g == 0) so[i][jj] = bf16r(o);
        }
      }
      // the drafts' u into sv (read above by this column's lanes only, all in this warp; not read
      // again): the record loop below stores them with the records' other fields, after the token
      // pass
      __syncwarp();
      if (g == 0) {
#pragma unroll
        for (int i = 1; i < NT; ++i) sv[i][jj] = u[i];
      }
      // the state after the committed token (stored below)
#pragma unroll
      for (int r = 0; r < RPT; ++r) sc[r] = s[r] + k0r[r] * u[0];
    }
    // the committed state into the slice buffer (stored below as whole row segments)
#pragma unroll
    for (int r = 0; r < RPT; ++r) sS[(g + TPC * r) * kSP + jj] = sc[r];
  }
  __syncthreads();  // so[] and the slice buffer complete
  // ---- gated RMSNorm over the head's 128 columns: 32-column partial sums of squares to every
  //      CTA of the cluster by st.async on its mbarrier (summed in column order, as the 512-thread
  //      form's warp order); the stores below overlap the exchange
  // (one warp per token and 32-column group: the NT sums in parallel instead of one warp's serial chain)
  static_assert(NW32 * NT <= kSplitThreads / 32, "kda split: a warp per partial sum");
  if constexpr (NS == 4) {
    // (the partial sums before the cluster wait: they write only the receivers' cl_ss / cl_mbar, initialized
    //  before the first phase; the state slice by the warps past the output threads, so that those reach the
    //  output without the slice's stores ahead of it)
    if (wid < NW32 * NT) {
      const int grp = wid % NW32, i = wid / NW32;
      const float x = so[i][grp * 32 + lane];
      const float ss = warp_sum(x * x);
      if (lane < NS) kda_dsm_st_async(&cl_ss[cs * NW32 + grp][i], &cl_mbar, lane, ss);
    }
    // every CTA of the cluster has read the old conv windows / draft inputs / replay records
    kda_cl_wait();
    // (LP: the in_proj grid before the first global store -- the state slice, the conv windows, the records, the output,
    //  the copy's resets -- so that this grid's completion still implies the in_proj's for the dependents)
    if constexpr (kLP) pdl_wait();
    constexpr int kST0 = NT * VC, kSTN = kSplitThreads - kST0;
    if (tid >= kST0) {
#pragma unroll
      for (int k = 0; k < (kD * (VC / 4) + kSTN - 1) / kSTN; ++k) {
        const int i = tid - kST0 + k * kSTN;
        if (i < kD * (VC / 4)) {
          const int r = i / (VC / 4), c4 = i % (VC / 4);
          *reinterpret_cast<float4*>(S + (size_t)r * kD + c0 + c4 * 4) =
              *reinterpret_cast<const float4*>(sS + r * kSP + c4 * 4);
        }
      }
    }
  } else {
    // every CTA of the cluster has read the old conv windows / draft inputs / replay records (and has
    // started and initialized its mbarrier: its shared memory may be written)
    kda_cl_wait();
    if (wid < NW32 * NT) {
      const int grp = wid % NW32, i = wid / NW32;
      const float x = so[i][grp * 32 + lane];
      const float ss = warp_sum(x * x);
      if (lane < NS) kda_dsm_st_async(&cl_ss[cs * NW32 + grp][i], &cl_mbar, lane, ss);
    }
    for (int i = tid; i < kD * (VC / 4); i += kSplitThreads) {
      const int r = i / (VC / 4), c4 = i % (VC / 4);
      *reinterpret_cast<float4*>(S + (size_t)r * kD + c0 + c4 * 4) =
          *reinterpret_cast<const float4*>(sS + r * kSP + c4 * 4);
    }
  }
  // ---- this CTA's channels' conv window and draft inputs; the draft tokens' replay records of
  //      rows / columns [c0, c0 + VC)
  if (conv_t && owns) {
    bf16* dst = conv_state + (size_t)slot * meta.conv_ss + (size_t)ch * (W - 1);
#pragma unroll
    for (int q = 0; q < W - 1; ++q) dst[q] = __float2bfloat16(win[1 + q]);
#pragma unroll
    for (int i = 1; i < NT; ++i)
      crec[kda_crec_off(meta, slot, i - 1, sec, ch - sec * HD)] = __float2bfloat16(win[W - 1 + i]);
    if constexpr (kLP) {  // (LP: this channel's words of the copy, read by the cluster's CTAs before cluster barrier 2)
#pragma unroll
      for (int i = 0; i < NT; ++i) lpw[(size_t)(t0 + i) * ld + ch] = kda_lp_sent();
    }
  }
  if constexpr (kLP) {
    // (the head's betas, read by the cluster's 4 CTAs; the other copy's f_a / g_a columns, rows 0-3 -- read by every CTA of
    //  the previous KDA layer, which used that copy and completed before this layer's in_proj passed its PDL wait -- by CTA
    //  (0, 0, 0))
    if (cs == 0 && tid >= kBT0 && tid < kBT0 + NT) lpw[(size_t)(t0 + tid - kBT0) * ld + 3 * HD + h] = kda_lp_sent();
    if (b == 0 && h == 0 && cs == 0 && tid < 4 * 32)
      reinterpret_cast<uint4*>(lpfg + (size_t)(tid >> 5) * ld)[tid & 31] = make_uint4(~0u, ~0u, ~0u, ~0u);
  }
  if constexpr (NS == 4) {  // (by the last (NT - 1) x VC threads, past the output threads)
    const int q = tid - (kSplitThreads - (NT - 1) * VC);
    if (q >= 0) {
      const int i = 1 + q / VC, c = c0 + q % VC;
      float* R = rec + (size_t)slot * meta.rec_ss + ((size_t)(i - 1) * H + h) * kKdaRec;
      R[c] = sk[i][c];
      R[kD + c] = sv[i][c - c0];  // (u, above)
      R[2 * kD + c] = sdecay[i][c];
    }
  } else {
    for (int q = tid; q < (NT - 1) * VC; q += kSplitThreads) {
      const int i = 1 + q / VC, c = c0 + q % VC;
      float* R = rec + (size_t)slot * meta.rec_ss + ((size_t)(i - 1) * H + h) * kKdaRec;
      R[c] = sk[i][c];
      R[kD + c] = sv[i][c - c0];  // (u, above)
      R[2 * kD + c] = sdecay[i][c];
    }
  }
  if (tid < NT * VC) {  // (NT x VC <= 256 threads: one output each)
    kda_mbar_wait0(&cl_mbar);  // the cluster's partial sums (no CTA exits before it received them)
    const int i = tid / VC, c = tid % VC;
    const float n2 = cl_ss[0][i] + cl_ss[1][i] + cl_ss[2][i] + cl_ss[3][i];
    const float r = rsqrtf(n2 / kD + eps);
    const float nv = bf16r(so[i][c] * r * onw);
    out[(size_t)(t0 + i) * HD + h * kD + c0 + c] = __float2bfloat16(nv / (1.f + __expf(-sg2[i][c])));
  }
