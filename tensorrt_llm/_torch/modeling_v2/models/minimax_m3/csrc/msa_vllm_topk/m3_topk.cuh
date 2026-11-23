/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Vendored from vllm-project/MSA (https://github.com/vllm-project/MSA) at
// commit 25c144700f727b638067e98066a60507081b63ad, python/fmha_sm100/csrc/include/indexer_topk/m3_topk.cuh: the decode
// top-k of vLLM's MiniMax-M3.1 indexer (vllm-project/vllm #63232). Changes for TensorRT-LLM:
// the namespace is minimaxM31VllmTopk (so this copy cannot interpose with an MSA module loaded in the same process);
// write_row / write_row_keys emit the 16 ids in ascending order (the order this target's selects hand the sparse
// decode, whose split merge sums the 16 partials in table order). For the attention-DP decode select: a row's
// length is read at its row within its group (row % rows_per_group), so the groups (the KV heads) share one length
// per query row; a launch of one group reads it at the row, as before.

// m3_topk -- top-K select over one row of scores.
//
// Contract (kTopK = 16): a row of `n` scores yields 16 ids. Column n-1 is always
// selected and written to the last slot without being ranked; the ranking picks
// kSelK = 15 out of the remaining nvm = n-1 columns. A row of n <= kTopK emits
// its n ids in order and pads with -1, so the tail lands at slot n-1.
//
// `n` is PER ROW, always: it is `row_n[row]`, and there is no fixed-length path.
// A paged decode scores a row only over the pages before that query's own local
// page, so n varies with the sequence length; the scalar `n_max` only bounds
// them, and is the only `n` the host and the kernel entry can use. Padding the
// tail is no escape: the key quantizes over the row's own [min, max]
// (topk_select.cuh), so a -1e30 pad collapses every real score onto one key.

#pragma once

#include <cstdint>

#include "minimax_m3/csrc/msa_vllm_topk/topk_select.cuh"

namespace minimaxM31VllmTopk
{

// Rows per CTA for the warp family. A constant on purpose: it depends on neither
// the SM count nor `rows`.
constexpr int kRowsPerBlock = 4;

// Widest row the warp family ranks (64 keys per lane). Each launch runs a
// kernel instantiated for one family and the widest rung n_max needs: up to
// kWarpSmallNvm, kWarpNarrowNvm or kWarpMaxNvm candidates, or the block
// family. The register demand of wider rungs (and of the block family) would
// otherwise lower the occupancy of every launch. Rows beyond kWarpNarrowNvm
// take the warp family only in launches of at least kWarpWideMinRows rows:
// with fewer rows the block family's shorter per-row latency wins.
constexpr int kWarpMaxNvm = 2048;
constexpr int kWarpNarrowNvm = 1024;
constexpr int kWarpSmallNvm = 256;
constexpr int kWarpWideMinRows = 512;
// The warp family packs kRowsPerBlock rows into a CTA of kRowsPerBlock warps
// from this many rows on. Smaller launches take one kThreads CTA per row, the
// surplus CTAs retiring at once: measured on GB300 PDL-chained decode launches,
// packing gains up to 1.4x from 1024 rows on and loses up to 9% at 256.
constexpr int kWarpPackedMinRows = 1024;

// The contract's ceiling: the widest row the select ranks. A caller that ranks wider rows sets M3_MAX_NVM (and an
// M3_RUNGS whose last rung reaches it) before the include.
#ifndef M3_MAX_NVM
#define M3_MAX_NVM 8192
#endif
constexpr int kMaxNvm = M3_MAX_NVM;

// KPT rungs the block select is instantiated at, narrowest first. A row takes
// the first rung whose KPT*kThreads covers its own nvm, so the rung IS the size
// of that row's per-thread register array.
#ifndef M3_RUNGS
#define M3_RUNGS 2, 4, 8, 12, 16, 20, 24, 28, 32
#endif
constexpr int kRung[] = {M3_RUNGS};
constexpr int kRungs = static_cast<int>(sizeof(kRung) / sizeof(kRung[0]));
static_assert(kRung[kRungs - 1] * kThreads >= kMaxNvm, "the last rung has to reach kMaxNvm");

// The kernel's thresholds, host-callable, so a caller can label a point with the
// family and width it takes. nvm = n-1. Fed n_max, they describe the launch
// shape, not what any one row of that launch takes.
inline int m3_warp_max(int n_max, int rows);

inline char const* m3_family(int n, int rows = kWarpWideMinRows)
{
    if (n <= kTopK)
        return "trivial";
    return m3_warp_max(n, rows) > 0 ? "warp" : "block";
}

inline int m3_width(int n)
{
    int const nvm = n - 1;
    if (n <= kTopK)
        return 0;
    if (nvm <= 32)
        return 1;
    if (nvm <= 64)
        return 2;
    if (nvm <= 128)
        return 4;
    if (nvm <= 256)
        return 8;
    if (nvm <= 512)
        return 16;
    if (nvm <= 768)
        return 24;
    if (nvm <= 1024)
        return 32;
    if (nvm <= 1280)
        return 40;
    if (nvm <= 1536)
        return 48;
    if (nvm <= kWarpMaxNvm)
        return 64;
    for (int i = 0; i < kRungs; ++i)
        if (nvm <= kRung[i] * kThreads)
            return kRung[i];
    return kRung[kRungs - 1];
}

// The widest row any launch ranks. Past it the select would drop candidates.
inline int m3_max_n()
{
    return kMaxNvm + 1;
}

// ---------------------------------------------------------------------------
// What the two families do identically: read the row's own n, answer it outright
// when it is short, and lay out the 16 output slots. `slot` is the calling
// thread's index inside whatever unit owns the row -- the CTA in the block
// family, the warp in the warp family -- which is what lets one body serve both.
// Every thread of that unit reads the same `n`, so the width branch each family
// takes is uniform over that unit.
// ---------------------------------------------------------------------------

// Row lengths are clamped to the score columns, so a length past them (bad
// metadata) cannot make any family read past the row.
__device__ __forceinline__ int row_len(int const* __restrict__ row_n, int row, int n_cols, int rows_per_group)
{
    int const n = __ldg(row_n + row % rows_per_group);
    return n < n_cols ? n : n_cols;
}

// This row's nvm = n-1, the count the select ranks -- or 0 once the row is
// already written, which is every row of n <= kTopK: fewer columns than slots,
// so the answer is the n ids in order padded with -1, and nothing is ranked.
__device__ __forceinline__ int row_nvm(
    int const* __restrict__ row_n, int* __restrict__ out_ids, int row, int slot, int n_cols, int rows_per_group)
{
    int const n = row_len(row_n, row, n_cols, rows_per_group);
    if (n > kTopK)
        return n - 1; // column n-1 is forced-selected, never ranked
    if (slot < kTopK)
        out_ids[row * kTopK + slot] = (slot < n) ? slot : -1;
    return 0;
}

// Where a row's scores start. Rows lie row_stride apart within a group of
// rows_per_group rows and groups lie group_stride apart (a matrix is one group).
struct RowLayout
{
    int row_stride;
    int rows_per_group;
    int64_t group_stride;
};

__device__ __forceinline__ float const* row_scores(float const* __restrict__ scores, int row, RowLayout const& layout)
{
    int const group = row / layout.rows_per_group;
    int const group_row = row - group * layout.rows_per_group;
    return scores + static_cast<int64_t>(group) * layout.group_stride
        + static_cast<int64_t>(group_row) * layout.row_stride;
}

// With a PDL launch (decode) the predecessor writes the scores and lengths and
// may still read `out_ids`; every successor waits for this grid before reading
// `out_ids`. Both calls are no-ops for a plain launch. Every kernel below runs
// this first.
__device__ __forceinline__ void wait_for_scores()
{
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
}

// The 15 ranked ids, then the forced tail in the last slot.
// Ascending order of the 16 ids held by lanes 0..15 of the calling warp (every lane calls it), written to the row.
__device__ __forceinline__ void write_sorted_ids(int* __restrict__ out_ids, int row, int slot, int id)
{
    int const lane = slot & 31;
    unsigned v = (slot < kTopK) ? static_cast<unsigned>(id) : 0xFFFFFFFFu;
#pragma unroll
    for (int size = 2; size <= 32; size <<= 1)
    {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            unsigned const other = __shfl_xor_sync(0xFFFFFFFFu, v, stride);
            bool const ascending = (lane & size) == 0, lower = (lane & stride) == 0;
            v = (lower == ascending) ? min(v, other) : max(v, other);
        }
    }
    if (slot < kTopK)
        out_ids[row * kTopK + slot] = static_cast<int>(v);
}

// The 15 ranked ids and the forced tail, ascending. Every thread of the calling warps calls it.
__device__ __forceinline__ void write_row(
    int* __restrict__ out_ids, int row, int slot, int nvm, int const* __restrict__ sel)
{
    write_sorted_ids(out_ids, row, slot, slot < kTopK ? ((slot == kSelK) ? nvm : sel[slot]) : 0);
}

// write_row for a warp_peel_row result: sel holds keys, the id is ~key's low half.
__device__ __forceinline__ void write_row_keys(
    int* __restrict__ out_ids, int row, int slot, int nvm, int const* __restrict__ sel)
{
    write_sorted_ids(out_ids, row, slot,
        slot < kTopK ? ((slot == kSelK) ? nvm : static_cast<int>((~static_cast<unsigned>(sel[slot])) & 0xFFFFu)) : 0);
}

// Per-warp scratch: warp_peel_row's per-lane key lists (kSelK slots x 32
// lanes, the last slot a sentinel).
struct WarpSelectSmem
{
    unsigned keys[kSelK * 32];
};

// One warp ranks one row: load -> warp-reduce [min,max] -> pack_key -> sort ->
// kSelK rounds of redux.sync.max, winner pops its next key. out15 receives the
// kSelK winning KEYS (write_row_keys extracts the ids). No block barrier, and the
// result goes to a per-warp `out15` rather than shared state, so warps in one CTA
// never interact -- which is what lets each warp pick its width from its own row.
// __noinline__ keeps the width instantiations out of the caller's inline body.
#ifndef M3_PREFETCH_PEEL_MAX
#define M3_PREFETCH_PEEL_MAX 32
#endif
#ifndef M3_REG_PEEL_MAX
#define M3_REG_PEEL_MAX 2
#endif
template <int N>
__device__ __noinline__ void warp_peel_row(
    int nvm, float const* __restrict__ rsc1, int* __restrict__ out15, unsigned* __restrict__ lst)
{
    int const lane = static_cast<int>(threadIdx.x) & 31;
    constexpr int SH = N < kSelK ? N : kSelK;
    float sc[N];
    float mymax = -INFINITY, mymin = INFINITY;
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        int const idx = i * 32 + lane;
        sc[i] = (idx < nvm) ? __ldcg(rsc1 + idx) : -INFINITY;
    }
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        int const idx = i * 32 + lane;
        if (idx < nvm)
        {
            mymax = fmaxf(mymax, sc[i]);
            mymin = fminf(mymin, sc[i]);
        }
    }
    mymax = topk_sel::warp_max_f32(mymax);
    mymin = topk_sel::warp_min_f32(mymin);
    float const scale = (mymax > mymin) ? (65534.0f / (mymax - mymin)) : 0.0f;
    float const bias = fmaf(-mymin, scale, 1.5f);
    unsigned key[N];
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        int const idx = i * 32 + lane;
        key[i] = (idx < nvm) ? topk_sel::pack_key(sc[i], scale, bias, idx) : 0u;
    }
    topk_sel::sort_desc_u32<N>(key);
    if constexpr (N <= M3_REG_PEEL_MAX)
    {
        // Narrow lanes: shifting a couple of registers beats a store + load.
#pragma unroll
        for (int r = 0; r < kSelK; ++r)
        {
            unsigned M;
            asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(M) : "r"(key[0]));
            if (lane == 0)
                out15[r] = static_cast<int>(M);
            topk_sel::winner_shift_down<SH>(key, key[0] == M);
        }
        __syncwarp(); // lane 0's writes visible to the rest of this warp
        return;
    }
    // Wider lanes: the lane's head key stays in a register; key[1..SH-1] go to
    // its own column of `lst` ([slot][lane], conflict-free), closed by a 0
    // sentinel. A pop is then one predicated load, not an SH-register shift. Only
    // this lane reads its column, so no barrier is needed.
    unsigned head = key[0];
    if constexpr (SH > 1)
    {
#pragma unroll
        for (int j = 1; j < SH; ++j)
            lst[(j - 1) * 32 + lane] = key[j];
    }
    lst[(SH - 1) * 32 + lane] = 0u;
    // nx is loaded one round ahead, unconditionally: the winner takes it with a
    // select, and the load's latency hides behind the next redux. The last round
    // needs no prefetch, which also keeps off inside the column.
    int off = lane; // this lane's next unread slot
    unsigned nx = 0u;
    if constexpr (N <= M3_PREFETCH_PEEL_MAX)
        nx = lst[off];
#pragma unroll
    for (int r = 0; r < kSelK; ++r)
    {
        unsigned M;
        asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(M) : "r"(head));
        if (lane == 0)
            out15[r] = static_cast<int>(M);
        bool const w = (head == M);
        if constexpr (N <= M3_PREFETCH_PEEL_MAX)
        {
            head = w ? nx : head;
            if (r + 1 < kSelK)
            {
                off += w ? 32 : 0;
                nx = lst[off];
            }
        }
        else
        {
            // The widest rungs are register-bound: no prefetch register.
            if (w)
                head = lst[off];
            off += w ? 32 : 0;
        }
    }
    __syncwarp(); // lane 0's writes visible to the rest of this warp
}

// One row, one CTA: take the narrowest rung that covers this row's nvm.
template <int I>
__device__ __forceinline__ void block_rank_row(int nvm, float const* __restrict__ p, SelectSmem& s)
{
    constexpr int KPT = kRung[I];
    constexpr bool kLast = (I + 1 == kRungs); // covers kMaxNvm by static_assert
    if (kLast || nvm <= KPT * kThreads)
    {
        block_topk_atomicmax<KPT>(nvm, p, s);
    }
    else if constexpr (!kLast)
    { // discarded at the last rung, so it terminates
        block_rank_row<I + 1>(nvm, p, s);
    }
}

// One warp per row: take the narrowest width that covers this row's nvm, among
// the rungs up to kWarpMax (the launch guarantees nvm <= kWarpMax).
template <int kWarpMax>
__device__ __forceinline__ void warp_rank_row(
    int nvm, float const* __restrict__ p, int* __restrict__ out15, WarpSelectSmem& ws)
{
    if (nvm <= 32)
    {
        warp_peel_row<1>(nvm, p, out15, ws.keys);
    }
    else if (nvm <= 64)
    {
        warp_peel_row<2>(nvm, p, out15, ws.keys);
    }
    else if (nvm <= 128)
    {
        warp_peel_row<4>(nvm, p, out15, ws.keys);
    }
    else if (kWarpMax <= kWarpSmallNvm || nvm <= 256)
    {
        warp_peel_row<8>(nvm, p, out15, ws.keys);
    }
    else if constexpr (kWarpMax > kWarpSmallNvm)
    {
        if (nvm <= 512)
        {
            warp_peel_row<16>(nvm, p, out15, ws.keys);
        }
        else if (nvm <= 768)
        {
            warp_peel_row<24>(nvm, p, out15, ws.keys);
        }
        else if (kWarpMax <= kWarpNarrowNvm || nvm <= 1024)
        {
            warp_peel_row<32>(nvm, p, out15, ws.keys);
        }
        else if constexpr (kWarpMax > kWarpNarrowNvm)
        {
            if (nvm <= 1280)
            {
                warp_peel_row<40>(nvm, p, out15, ws.keys);
            }
            else if (nvm <= 1536)
            {
                warp_peel_row<48>(nvm, p, out15, ws.keys);
            }
            else
            {
                warp_peel_row<64>(nvm, p, out15, ws.keys);
            }
        }
    }
}

// The widest warp-family row of a launch, or 0 for the block family.
inline int m3_warp_max(int n_max, int rows)
{
    int const nvm_max = n_max - 1;
    if (nvm_max <= kWarpSmallNvm)
        return kWarpSmallNvm;
    if (nvm_max <= kWarpNarrowNvm)
        return kWarpNarrowNvm;
    if (nvm_max <= kWarpMaxNvm && rows >= kWarpWideMinRows)
        return kWarpMaxNvm;
    return 0;
}

// Rows per CTA -- what the warp family packs into one CTA, reported so a caller
// can label a launch with it. It is the kernel's own family test.
inline int m3_rows_per_block(int n_max, int rows)
{
    return m3_warp_max(n_max, rows) > 0 ? kRowsPerBlock : 1;
}

// FAMILY comes from n_max and is grid-uniform -- it fixes the thread -> row
// mapping, which the whole grid has to agree on:
//   n_max-1 <= warp max : warp family, one warp per row, kRowsPerBlock per CTA
//   n_max-1 >  warp max : block family, one row per CTA, all kThreads on it
// The launch picks the family (m3_warp_max) and instantiates the kernel for it
// alone: kWarpMax > 0 is the warp family up to kWarpMax candidates, 0 the block
// family. A warp needs no block barrier per selected id, and kRowsPerBlock rows
// share a CTA.
//
// WIDTH comes from that row's own nvm: per CTA in the block family, per warp in
// the warp family. A short row in a block-family launch takes the narrowest
// block width rather than a warp path -- correct, just wider than it needs.
template <int kWarpMax>
__global__ __launch_bounds__(kThreads) void m3_topk_kernel(float const* __restrict__ scores,
    int const* __restrict__ row_n, int* __restrict__ out_ids, int n_max, RowLayout layout, int rows)
{
    __shared__ union Smem
    {
        SelectSmem s;

        struct
        {
            int sel[kRowsPerBlock][kSelK];
            WarpSelectSmem select[kRowsPerBlock];
        } w;
    } u;

    int const tid = static_cast<int>(threadIdx.x);
    wait_for_scores();

    if constexpr (kWarpMax > 0)
    {
        // ---- warp family: one warp per row, no block barrier anywhere ----
        // This branch IS m3_rows_per_block's test, so the rows per CTA here is
        // kRowsPerBlock -- the same one the host built the grid from.
        int const warp = tid >> 5;
        int const lane = tid & 31;
        int const row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
        // A warp past kRowsPerBlock is idle and the tail CTA is bounded per warp;
        // neither may touch row_n, so the guard has to come before row_nvm rather
        // than fold into it. Both leave nvm at 0 -- the same "nothing to rank" the
        // short-row branch leaves -- so one test below covers all three cases.
        bool const owns_row = (warp < kRowsPerBlock) && (row < rows);
        int const nvm = owns_row ? row_nvm(row_n, out_ids, row, lane, n_max, layout.rows_per_group) : 0;
        if (nvm > 0)
        {
            warp_rank_row<kWarpMax>(nvm, row_scores(scores, row, layout), u.w.sel[warp], u.w.select[warp]);
            write_row_keys(out_ids, row, lane, nvm, u.w.sel[warp]);
        }
    }
    else
    {
        // ---- block family: one row per CTA, so blockIdx.x IS the row and every
        // thread of the CTA reads the same n -- the branch below is CTA-uniform,
        // which is what keeps the __syncthreads() inside block_rank_row convergent.
        int const row = static_cast<int>(blockIdx.x);
        int const nvm = row_nvm(row_n, out_ids, row, tid, n_max, layout.rows_per_group);
        if (nvm > 0)
        {
            block_rank_row<0>(nvm, row_scores(scores, row, layout), u.s);
            write_row(out_ids, row, tid, nvm, u.s.sel_ids);
        }
    }
}

inline int m3_occ()
{
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m3_topk_kernel<0>, kThreads, 0);
    return blocks;
}

// ---- narrow family: rows of n_max - 1 <= kWarpSmallNvm ----
//
// Eight threads rank a row (a 256-thread CTA holds 32 rows): each loads its
// columns as float4 (column 4*(k*8 + j) + c for float4 k of thread j), keys
// them, and keeps its own top 8 by a sorting network; kSelK rounds of an
// 8-lane max then pop the winners. Same keys and order as warp_peel_row, so
// the ids are bit-identical. A thread that loses all 8 of its keys while it
// holds more may own a 9th of the top kSelK; that row is redone by an exact
// slow path (rare: well under 0.1% of the rows of random scores).
namespace narrow
{
constexpr int kG = 8;                // threads per row
constexpr int kRows = kThreads / kG; // rows per CTA
constexpr int kMaxK = 8;             // float4 per thread: rows up to 4*kG*kMaxK columns
constexpr int kMinBlocks = 5;        // CTAs per SM (48 registers)
static_assert(4 * kG * kMaxK >= kWarpSmallNvm, "the widest narrow row must fit");

// Descending top 8 of two descending 8-lists: max(a[i], b[7-i]) is bitonic and
// holds the top 8; a 3-stage bitonic clean sorts it.
__device__ __forceinline__ void top8_merge(unsigned (&a)[8], unsigned const (&b)[8])
{
#pragma unroll
    for (int i = 0; i < 8; ++i)
        a[i] = a[i] > b[7 - i] ? a[i] : b[7 - i];
#pragma unroll
    for (int k = 4; k >= 1; k >>= 1)
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if ((i & k) == 0)
            {
                unsigned const x = a[i], y = a[i + k];
                a[i] = x > y ? x : y;
                a[i + k] = x > y ? y : x;
            }
}

// pack_key(v, scale, bias, col) in three ops: fma; cvt.rzi.sat.u16 is the
// truncation and the clamp to 65535 (and maps NaN to 0, below every real key);
// the low half ~col = 0xFFFF - col is a per-thread constant minus an offset.
// Float4 idx of a row whose columns from 4*idx on number rem, NaN past them.
// A partial float4 is read whole: usable() admits only scores whose rows reach
// past their last candidate's float4.
__device__ __forceinline__ float4 load4_upto(float4 const* __restrict__ p, int idx, int rem)
{
    float const qnan = __int_as_float(0x7fc00000);
    float4 f = make_float4(qnan, qnan, qnan, qnan);
    if (rem > 0)
        f = __ldcg(p + idx);
    if (rem > 0 && rem < 4)
    {
        if (rem < 2)
            f.y = qnan;
        if (rem < 3)
            f.z = qnan;
        f.w = qnan;
    }
    return f;
}

__device__ __forceinline__ unsigned narrow_key(float v, float scale, float bias, unsigned low)
{
    unsigned short q;
    asm("cvt.rzi.sat.u16.f32 %0, %1;" : "=h"(q) : "f"(fmaf(v, scale, bias)));
    return static_cast<unsigned>(q) * 65536u + low;
}

__device__ __forceinline__ float group_max(float v, unsigned gm)
{
#pragma unroll
    for (int d = kG / 2; d; d >>= 1)
        v = fmaxf(v, __shfl_xor_sync(gm, v, d));
    return v;
}

__device__ __forceinline__ float group_min(float v, unsigned gm)
{
#pragma unroll
    for (int d = kG / 2; d; d >>= 1)
        v = fminf(v, __shfl_xor_sync(gm, v, d));
    return v;
}

// The kSelK rounds, one copy for every K: head = this thread's largest key,
// its next 7 in lst[slot * kThreads] (a [slot][thread] column, conflict-free)
// then a 0 sentinel. Returns false if a thread that may hold a 9th key ran dry.
__device__ __noinline__ bool rounds(
    unsigned head, unsigned const* __restrict__ lst, int j, unsigned gm, int* __restrict__ o, int nvm, bool can_run_dry)
{
    constexpr int kMine = (kSelK + kG - 1) / kG;
    unsigned mine[kMine]; // thread j keeps rounds j, j + kG: a select, not a store per round
#pragma unroll
    for (int i = 0; i < kMine; ++i)
        mine[i] = 0u;
    int po = 0, pops = 0;
    unsigned nx = lst[0];
#pragma unroll
    for (int r = 0; r < kSelK; ++r)
    {
        unsigned M = head;
#pragma unroll
        for (int d = kG / 2; d; d >>= 1)
        {
            unsigned const x = __shfl_xor_sync(gm, M, d);
            M = M > x ? M : x;
        }
        mine[r / kG] = ((r & (kG - 1)) == j) ? M : mine[r / kG];
        bool const w = head == M;
        pops += w;
        head = w ? nx : head;
        if (r + 1 < kSelK)
        {
            po += w ? kThreads : 0;
            po = po < 7 * kThreads ? po : 7 * kThreads; // stay on the sentinel
            nx = lst[po];
        }
    }
    if (__any_sync(gm, can_run_dry && pops >= 8))
        return false;
#pragma unroll
    for (int i = 0; i < kMine; ++i)
        if (i * kG + j < kSelK)
            o[i * kG + j] = static_cast<int>((~mine[i]) & 0xFFFFu);
    if (j == 0)
        o[kSelK] = nvm;
    return true;
}

// Exact slow path: this thread's keys once into registers, then kSelK rounds,
// each taking the largest key below the last winner by an in-register scan.
template <int K>
__device__ __noinline__ void redo_row(
    float const* __restrict__ rs, int nvm, int j, unsigned gm, int* __restrict__ o, float scale, float bias)
{
    constexpr int E = 4 * K;
    float4 const* p = reinterpret_cast<float4 const*>(rs);
    unsigned const low = 0xFFFFu - 4u * static_cast<unsigned>(j);
    unsigned key[E];
#pragma unroll
    for (int k = 0; k < K; ++k)
    {
        int const c0 = (k * kG + j) * 4;
        const float4 f = load4_upto(p, k * kG + j, nvm - c0);
        float const v[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
        for (int c = 0; c < 4; ++c)
            key[4 * k + c]
                = c0 + c < nvm ? narrow_key(v[c], scale, bias, low - static_cast<unsigned>(k * kG * 4 + c)) : 0u;
    }
    unsigned prev = 0u;
#pragma unroll 1
    for (int r = 0; r < kSelK; ++r)
    {
        unsigned best = 0u;
#pragma unroll
        for (int e = 0; e < E; ++e)
            if ((r == 0 || key[e] < prev) && key[e] > best)
                best = key[e];
#pragma unroll
        for (int d = kG / 2; d; d >>= 1)
        {
            unsigned const x = __shfl_xor_sync(gm, best, d);
            best = best > x ? best : x;
        }
        if (j == 0)
            o[r] = static_cast<int>((~best) & 0xFFFFu);
        prev = best;
    }
    if (j == 0)
        o[kSelK] = nvm;
}

// One row of nvm <= 4*kG*K columns (K = ceil(nvm / (4*kG))).
template <int K>
__device__ __noinline__ void rank_row(
    float const* __restrict__ rs, int nvm, int j, unsigned gm, int* __restrict__ o, unsigned* __restrict__ lst)
{
    constexpr int E = 4 * K;
    float4 const* p = reinterpret_cast<float4 const*>(rs);
    // Columns past nvm hold NaN: fmaxf/fminf skip it and its key is below every
    // real one, so nothing below tests a column against nvm.
    float v[E];
#pragma unroll
    for (int k = 0; k < K; ++k)
    {
        float4 f;
        if (k + 1 < K)
        { // every float4 before the last is full
            f = __ldcg(p + k * kG + j);
        }
        else
        {
            f = load4_upto(p, k * kG + j, nvm - (k * kG + j) * 4);
        }
        v[4 * k] = f.x;
        v[4 * k + 1] = f.y;
        v[4 * k + 2] = f.z;
        v[4 * k + 3] = f.w;
    }
    float mx = -INFINITY, mn = INFINITY;
#pragma unroll
    for (int e = 0; e < E; ++e)
    {
        mx = fmaxf(mx, v[e]);
        mn = fminf(mn, v[e]);
    }
    mx = group_max(mx, gm);
    mn = group_min(mn, gm);
    // A constant row takes scale 0: its keys have q = 1, and NaN ones q = 0.
    float const scale = mx > mn ? 65534.0f / (mx - mn) : 0.0f;
    float const bias = fmaf(-mn, scale, 1.5f);
    unsigned const low = 0xFFFFu - 4u * static_cast<unsigned>(j);
    unsigned u[E];
#pragma unroll
    for (int e = 0; e < E; ++e)
        u[e] = narrow_key(v[e], scale, bias, low - static_cast<unsigned>(((e >> 2) * kG) * 4 + (e & 3)));
    // this thread's top 8, descending (fewer when E < 8: the rest are 0)
    unsigned t[8];
    if constexpr (E <= 8)
    {
        unsigned w[E];
#pragma unroll
        for (int e = 0; e < E; ++e)
            w[e] = u[e];
        topk_sel::sort_desc_u32<E>(w);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            t[i] = i < E ? w[i] : 0u;
    }
    else
    {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            t[i] = u[i];
        topk_sel::sort_desc_u32<8>(t);
#pragma unroll
        for (int c = 8; c < E; c += 8)
        {
            unsigned b[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                b[i] = c + i < E ? u[c + i] : 0u;
            topk_sel::sort_desc_u32<8>(b);
            top8_merge(t, b);
        }
    }
#pragma unroll
    for (int i = 1; i < 8; ++i)
        lst[(i - 1) * kThreads] = t[i];
    lst[7 * kThreads] = 0u;
    if (!rounds(t[0], lst, j, gm, o, nvm, E > 8))
        redo_row<K>(rs, nvm, j, gm, o, scale, bias);
}

template <int K>
__device__ __forceinline__ void rank_dispatch(float const* rs, int nvm, int j, unsigned gm, int* o, unsigned* lst)
{
    if constexpr (K >= kMaxK)
    {
        rank_row<K>(rs, nvm, j, gm, o, lst);
    }
    else
    {
        if (nvm <= 4 * kG * K)
            rank_row<K>(rs, nvm, j, gm, o, lst);
        else
            rank_dispatch<K + 1>(rs, nvm, j, gm, o, lst);
    }
}

__global__ __launch_bounds__(kThreads, kMinBlocks) void narrow_kernel(float const* __restrict__ scores,
    int const* __restrict__ row_n, int* __restrict__ out_ids, RowLayout layout, int rows, int n_cols)
{
    __shared__ unsigned lst[8 * kThreads];
    wait_for_scores();
    int const tid = static_cast<int>(threadIdx.x);
    int const lane = tid & 31, j = tid & (kG - 1);
    unsigned const gm = 0xFFu << (lane & ~(kG - 1));
    int const row = static_cast<int>(blockIdx.x) * kRows + tid / kG;
    if (row >= rows)
        return; // the row's whole group leaves together
    int const n = row_len(row_n, row, n_cols, layout.rows_per_group);
    int* o = out_ids + row * kTopK;
    if (n <= kTopK)
    { // fewer columns than slots: the n ids in order, padded with -1
        for (int t = j; t < kTopK; t += kG)
            o[t] = t < n ? t : -1;
        return;
    }
    rank_dispatch<1>(row_scores(scores, row, layout), n - 1, j, gm, o, lst + tid);
}

// Float4 loads need 16-byte rows, and read a row's last candidate float4 whole:
// up to column 4 * ceil((n_cols - 1) / 4) - 1, which stays within n_cols only
// when n_cols % 4 is 0 or 1 (a row pitch past n_cols may not be allocated).
// The float4 families trade per-row latency for throughput: a row takes more
// steps than in the warp family, so they win only once the launch has rows to
// fill the GPU. Launches below these row counts keep the warp family (GB300:
// narrow rows of K = ceil(nvm / 32) float4 layers win from about 3072 * K rows,
// at least 6144; wide rows from about 2048 rows).
inline int min_rows(int n_max)
{
    int const k = (n_max - 1 + 4 * kG - 1) / (4 * kG);
    return k * 3072 > 6144 ? k * 3072 : 6144;
}

inline bool usable(float const* scores, RowLayout const& layout, int n_cols)
{
    return (reinterpret_cast<uintptr_t>(scores) & 15u) == 0 && (layout.row_stride & 3) == 0
        && (layout.group_stride & 3) == 0 && (n_cols & 3) < 2;
}
} // namespace narrow

// ---- wide family: kWarpSmallNvm < n_max - 1 <= kWarpMaxNvm ----
//
// One warp ranks a row (a 256-thread CTA holds 8 rows), so the warp is
// converged and a round's max is one full-warp redux. Lane l loads float4 k at
// column 4*(k*32 + l); K = ceil(nvm / 128). Up to K = 9 all values stay in
// registers (one pass); beyond, a first pass takes only min/max and a second
// re-reads 8 values at a time into keys and top-8 merges, so the register
// footprint does not grow with K. Top 8 per lane, kSelK rounds, and the same
// dry-lane redo as the narrow family; ids bit-identical to warp_peel_row.
namespace wide
{
constexpr int kRows = kThreads / 32; // rows (warps) per CTA
constexpr int kOnePassK = 9;         // K up to this keeps every value in registers
constexpr int kMaxK = kWarpMaxNvm / 128;
constexpr int kMinBlocks = 5;
constexpr int kTailMaxK = 8; // rank_row_tail: up to this many full layers

__device__ __forceinline__ unsigned warp_max_u(unsigned v)
{
    unsigned r;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
    return r;
}

// Lane's float4 k of the row, NaN past nvm.
inline int min_rows(int)
{
    return 2048;
}

__device__ __forceinline__ float4 load4(float4 const* __restrict__ p, int k, int lane, int nvm)
{
    return narrow::load4_upto(p, k * 32 + lane, nvm - (k * 32 + lane) * 4);
}

__device__ __noinline__ bool rounds(
    unsigned head, unsigned const* __restrict__ lst, int lane, int* __restrict__ o, int nvm, bool can_run_dry)
{
    unsigned mine = 0u; // lane r keeps round r's winner
    int po = 0, pops = 0;
    unsigned nx = lst[0];
#pragma unroll
    for (int r = 0; r < kSelK; ++r)
    {
        unsigned const M = warp_max_u(head);
        mine = lane == r ? M : mine;
        bool const w = head == M;
        pops += w;
        head = w ? nx : head;
        if (r + 1 < kSelK)
        {
            po += w ? kThreads : 0;
            po = po < 7 * kThreads ? po : 7 * kThreads;
            nx = lst[po];
        }
    }
    if (__any_sync(0xFFFFFFFFu, can_run_dry && pops >= 8))
        return false;
    if (lane < kSelK)
        o[lane] = static_cast<int>((~mine) & 0xFFFFu);
    if (lane == 0)
        o[kSelK] = nvm;
    return true;
}

// Exact slow path: kSelK rounds, each lane re-deriving its largest key below the
// last winner from the row (rare; the row is in L2).
__device__ __noinline__ void redo_row(
    float const* __restrict__ rs, int nvm, int lane, int* __restrict__ o, float scale, float bias)
{
    float4 const* p = reinterpret_cast<float4 const*>(rs);
    unsigned const low = 0xFFFFu - 4u * static_cast<unsigned>(lane);
    unsigned prev = 0u;
    for (int r = 0; r < kSelK; ++r)
    {
        unsigned best = 0u;
        for (int k = 0; (k * 32 + lane) * 4 < nvm; ++k)
        {
            const float4 f = load4(p, k, lane, nvm);
            float const v[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                unsigned const key = narrow::narrow_key(v[c], scale, bias, low - static_cast<unsigned>(k * 128 + c));
                if ((r == 0 || key < prev) && key > best)
                    best = key;
            }
        }
        best = warp_max_u(best);
        if (lane == 0)
            o[r] = static_cast<int>((~best) & 0xFFFFu);
        prev = best;
    }
    if (lane == 0)
        o[kSelK] = nvm;
}

// Keys of float4s k0, k0+1 into w[0..7] (a missing k0+1 gives 0 keys).
template <int K>
__device__ __forceinline__ void keys8(
    float4 const* p, int k0, int lane, int nvm, float scale, float bias, unsigned low, unsigned (&w)[8])
{
#pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        int const k = k0 + h;
        if (k < K)
        {
            const float4 f = load4(p, k, lane, nvm);
            float const v[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
            for (int c = 0; c < 4; ++c)
                w[4 * h + c] = narrow::narrow_key(v[c], scale, bias, low - static_cast<unsigned>(k * 128 + c));
        }
        else
        {
#pragma unroll
            for (int c = 0; c < 4; ++c)
                w[4 * h + c] = 0u;
        }
    }
}

template <int K>
__device__ __noinline__ void rank_row(
    float const* __restrict__ rs, int nvm, int lane, int* __restrict__ o, unsigned* __restrict__ lst)
{
    constexpr int E = 4 * K;
    float4 const* p = reinterpret_cast<float4 const*>(rs);
    unsigned const low = 0xFFFFu - 4u * static_cast<unsigned>(lane);
    float mx = -INFINITY, mn = INFINITY;
    unsigned t[8];
    float scale, bias;
    if constexpr (K <= kOnePassK)
    {
        float v[E];
#pragma unroll
        for (int k = 0; k < K; ++k)
        {
            const float4 f = load4(p, k, lane, nvm);
            v[4 * k] = f.x;
            v[4 * k + 1] = f.y;
            v[4 * k + 2] = f.z;
            v[4 * k + 3] = f.w;
        }
#pragma unroll
        for (int e = 0; e < E; ++e)
        {
            mx = fmaxf(mx, v[e]);
            mn = fminf(mn, v[e]);
        }
        mx = topk_sel::warp_max_f32(mx);
        mn = topk_sel::warp_min_f32(mn);
        // A constant row takes scale 0: its keys have q = 1, and NaN ones q = 0.
        scale = mx > mn ? 65534.0f / (mx - mn) : 0.0f;
        bias = fmaf(-mn, scale, 1.5f);
        unsigned u[E];
#pragma unroll
        for (int e = 0; e < E; ++e)
            u[e] = narrow::narrow_key(v[e], scale, bias, low - static_cast<unsigned>((e >> 2) * 128 + (e & 3)));
        if constexpr (E <= 8)
        {
            unsigned w[E];
#pragma unroll
            for (int e = 0; e < E; ++e)
                w[e] = u[e];
            topk_sel::sort_desc_u32<E>(w);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                t[i] = i < E ? w[i] : 0u;
        }
        else
        {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                t[i] = u[i];
            topk_sel::sort_desc_u32<8>(t);
#pragma unroll
            for (int c = 8; c < E; c += 8)
            {
                unsigned b[8];
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    b[i] = c + i < E ? u[c + i] : 0u;
                topk_sel::sort_desc_u32<8>(b);
                narrow::top8_merge(t, b);
            }
        }
    }
    else
    {
        // pass 1: min/max only
#pragma unroll
        for (int k = 0; k < K; ++k)
        {
            const float4 f = load4(p, k, lane, nvm);
            mx = fmaxf(fmaxf(mx, f.x), fmaxf(f.y, fmaxf(f.z, f.w)));
            mn = fminf(fminf(mn, f.x), fminf(f.y, fminf(f.z, f.w)));
        }
        mx = topk_sel::warp_max_f32(mx);
        mn = topk_sel::warp_min_f32(mn);
        // A constant row takes scale 0: its keys have q = 1, and NaN ones q = 0.
        scale = mx > mn ? 65534.0f / (mx - mn) : 0.0f;
        bias = fmaf(-mn, scale, 1.5f);
        // pass 2: 8 keys at a time into the running top 8
        keys8<K>(p, 0, lane, nvm, scale, bias, low, t);
        topk_sel::sort_desc_u32<8>(t);
#pragma unroll
        for (int k = 2; k < K; k += 2)
        {
            unsigned b[8];
            keys8<K>(p, k, lane, nvm, scale, bias, low, b);
            topk_sel::sort_desc_u32<8>(b);
            narrow::top8_merge(t, b);
        }
    }
#pragma unroll
    for (int i = 1; i < 8; ++i)
        lst[(i - 1) * kThreads] = t[i];
    lst[7 * kThreads] = 0u;
    if (!rounds(t[0], lst, lane, o, nvm, E > 8))
        redo_row(rs, nvm, lane, o, scale, bias);
}

// Rows of nvm = 128*K + T with 0 < T <= 32*TS: the K full float4 layers as in
// rank_row, and the last T columns as scalars, column 128*K + 32*s + lane for
// s < TS, so a lane carries 4K + TS values instead of 4K + 4.
template <int K, int TS>
__device__ __noinline__ void rank_row_tail(
    float const* __restrict__ rs, int nvm, int lane, int* __restrict__ o, unsigned* __restrict__ lst)
{
    constexpr int E = 4 * K + TS;
    float4 const* p = reinterpret_cast<float4 const*>(rs);
    float const qnan = __int_as_float(0x7fc00000);
    float v[E];
#pragma unroll
    for (int k = 0; k < K; ++k)
    { // full layers: no bound checks
        const float4 f = __ldcg(p + k * 32 + lane);
        v[4 * k] = f.x;
        v[4 * k + 1] = f.y;
        v[4 * k + 2] = f.z;
        v[4 * k + 3] = f.w;
    }
#pragma unroll
    for (int t = 0; t < TS; ++t)
    {
        int const col = 128 * K + 32 * t + lane;
        v[4 * K + t] = col < nvm ? __ldcg(rs + col) : qnan;
    }
    float mx = -INFINITY, mn = INFINITY;
#pragma unroll
    for (int e = 0; e < E; ++e)
    {
        mx = fmaxf(mx, v[e]);
        mn = fminf(mn, v[e]);
    }
    mx = topk_sel::warp_max_f32(mx);
    mn = topk_sel::warp_min_f32(mn);
    // A constant row takes scale 0: its keys have q = 1, and NaN ones q = 0.
    float const scale = mx > mn ? 65534.0f / (mx - mn) : 0.0f;
    float const bias = fmaf(-mn, scale, 1.5f);
    unsigned const low4 = 0xFFFFu - 4u * static_cast<unsigned>(lane), low1 = 0xFFFFu - static_cast<unsigned>(lane);
    unsigned u[E];
#pragma unroll
    for (int e = 0; e < 4 * K; ++e)
        u[e] = narrow::narrow_key(v[e], scale, bias, low4 - static_cast<unsigned>((e >> 2) * 128 + (e & 3)));
#pragma unroll
    for (int t = 0; t < TS; ++t)
        u[4 * K + t] = narrow::narrow_key(v[4 * K + t], scale, bias, low1 - static_cast<unsigned>(128 * K + 32 * t));
    // top 8: full chunks of 8, then the E % 8 leftovers (one is an insertion)
    unsigned t8[8];
    constexpr int R = E % 8, C = E - R;
    if constexpr (C == 0)
    {
        unsigned w8[E];
#pragma unroll
        for (int e = 0; e < E; ++e)
            w8[e] = u[e];
        topk_sel::sort_desc_u32<E>(w8);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            t8[i] = i < E ? w8[i] : 0u;
    }
    else
    {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            t8[i] = u[i];
        topk_sel::sort_desc_u32<8>(t8);
#pragma unroll
        for (int c = 8; c < C; c += 8)
        {
            unsigned b8[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                b8[i] = u[c + i];
            topk_sel::sort_desc_u32<8>(b8);
            narrow::top8_merge(t8, b8);
        }
        if constexpr (R == 1)
        { // insert one key into the sorted 8
            unsigned x = u[C];
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                unsigned const hi = t8[i] > x ? t8[i] : x;
                x = t8[i] > x ? x : t8[i];
                t8[i] = hi;
            }
        }
        else if constexpr (R > 1)
        {
            unsigned b8[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                b8[i] = i < R ? u[C + i] : 0u;
            topk_sel::sort_desc_u32<8>(b8);
            narrow::top8_merge(t8, b8);
        }
    }
#pragma unroll
    for (int i = 1; i < 8; ++i)
        lst[(i - 1) * kThreads] = t8[i];
    lst[7 * kThreads] = 0u;
    if (!rounds(t8[0], lst, lane, o, nvm, E > 8))
        redo_row(rs, nvm, lane, o, scale, bias);
}

template <int TS, int K>
__device__ __forceinline__ void tail_dispatch(int kf, float const* rs, int nvm, int lane, int* o, unsigned* lst)
{
    if constexpr (K >= kTailMaxK)
    {
        rank_row_tail<K, TS>(rs, nvm, lane, o, lst);
    }
    else
    {
        if (kf <= K)
            rank_row_tail<K, TS>(rs, nvm, lane, o, lst);
        else
            tail_dispatch<TS, K + 1>(kf, rs, nvm, lane, o, lst);
    }
}

// K per row: exact up to kOnePassK, then every other K up to kMaxK.
template <int K>
__device__ __forceinline__ void rank_dispatch(float const* rs, int nvm, int lane, int* o, unsigned* lst)
{
    if constexpr (K >= kMaxK)
    {
        rank_row<K>(rs, nvm, lane, o, lst);
    }
    else
    {
        if (nvm <= 128 * K)
            rank_row<K>(rs, nvm, lane, o, lst);
        else
            rank_dispatch<(K < kOnePassK ? K + 1 : (K + 2 < kMaxK ? K + 2 : kMaxK))>(rs, nvm, lane, o, lst);
    }
}

// ---- two rows per warp, a half-warp (16 lanes) each, for rows of nvm <= kHalfMaxNvm ----
// Half h = lane >> 4 ranks its row; the layout (F full 64-column layers, TS tail
// scalars) comes from the longer of the two rows, so the warp stays on one path
// and every shuffle/redux is full-warp. A constant row needs no branch: its
// scale is 0, so every q is 1 (0 for NaN), as in the other families.
constexpr int kHalfMaxF = 8;
constexpr int kHalfMaxNvm = 64 * kHalfMaxF + 64; // F+1 partial layers up to 9 x 4 = 36 values
// Launches whose rows all fit this take the pair kernel: two rows per warp won
// only for short wide rows (measured: n_max 320 faster, 520 and 1088 slower).
constexpr int kPairMaxNvm = 384;
// ... and only with enough rows: halving the CTAs of a small launch leaves SMs
// idle (8192 rows of n_max 320: slower paired, 32768: faster).
constexpr int kPairMinRows = 16384;

__device__ __forceinline__ float half_max(float v)
{
#pragma unroll
    for (int d = 8; d; d >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, d));
    return v;
}

__device__ __forceinline__ float half_min(float v)
{
#pragma unroll
    for (int d = 8; d; d >>= 1)
        v = fminf(v, __shfl_xor_sync(0xFFFFFFFFu, v, d));
    return v;
}

// The kSelK rounds of both rows: per round two full-warp reduxes, each seeing
// only its half's heads. Returns a mask of halves (1, 2) that may owe a 9th key.
__device__ __noinline__ int rounds_half(
    unsigned head, unsigned const* __restrict__ lst, int lane, int* __restrict__ o, int nvm, bool can_run_dry)
{
    bool const hi = lane >= 16;
    int const hl = lane & 15;
    unsigned mine = 0u;
    int po = 0, pops = 0;
    unsigned nx = lst[0];
#pragma unroll
    for (int r = 0; r < kSelK; ++r)
    {
        unsigned const Ma = warp_max_u(hi ? 0u : head), Mb = warp_max_u(hi ? head : 0u);
        unsigned const M = hi ? Mb : Ma;
        mine = hl == r ? M : mine;
        bool const w = head == M;
        pops += w;
        head = w ? nx : head;
        if (r + 1 < kSelK)
        {
            po += w ? kThreads : 0;
            po = po < 7 * kThreads ? po : 7 * kThreads;
            nx = lst[po];
        }
    }
    unsigned const dry = __ballot_sync(0xFFFFFFFFu, can_run_dry && pops >= 8);
    int const redo = ((dry & 0xFFFFu) ? 1 : 0) | ((dry >> 16) ? 2 : 0);
    if (!(redo & (hi ? 2 : 1)))
    {
        if (hl < kSelK)
            o[hl] = static_cast<int>((~mine) & 0xFFFFu);
        if (hl == 0)
            o[kSelK] = nvm;
    }
    return redo;
}

template <int F, int TS>
__device__ __noinline__ int rank_half(float const* __restrict__ rs, int nvm, int lane, int* __restrict__ o,
    unsigned* __restrict__ lst, float& scale_out, float& bias_out)
{
    constexpr int E = 4 * F + TS;
    int const hl = lane & 15;
    float4 const* p = reinterpret_cast<float4 const*>(rs);
    float const qnan = __int_as_float(0x7fc00000);
    float v[E];
#pragma unroll
    for (int k = 0; k < F; ++k)
    { // bounds from this row's nvm (the layout is the longer row's)
        const float4 f = narrow::load4_upto(p, k * 16 + hl, nvm - (k * 16 + hl) * 4);
        v[4 * k] = f.x;
        v[4 * k + 1] = f.y;
        v[4 * k + 2] = f.z;
        v[4 * k + 3] = f.w;
    }
#pragma unroll
    for (int t = 0; t < TS; ++t)
    {
        int const col = 64 * F + 16 * t + hl;
        v[4 * F + t] = col < nvm ? __ldcg(rs + col) : qnan;
    }
    float mx = -INFINITY, mn = INFINITY;
#pragma unroll
    for (int e = 0; e < E; ++e)
    {
        mx = fmaxf(mx, v[e]);
        mn = fminf(mn, v[e]);
    }
    mx = half_max(mx);
    mn = half_min(mn);
    float const scale = mx > mn ? 65534.0f / (mx - mn) : 0.0f;
    float const bias = fmaf(-mn, scale, 1.5f);
    scale_out = scale;
    bias_out = bias;
    unsigned const low4 = 0xFFFFu - 4u * static_cast<unsigned>(hl), low1 = 0xFFFFu - static_cast<unsigned>(hl);
    unsigned u[E];
#pragma unroll
    for (int e = 0; e < 4 * F; ++e)
        u[e] = narrow::narrow_key(v[e], scale, bias, low4 - static_cast<unsigned>((e >> 2) * 64 + (e & 3)));
#pragma unroll
    for (int t = 0; t < TS; ++t)
        u[4 * F + t] = narrow::narrow_key(v[4 * F + t], scale, bias, low1 - static_cast<unsigned>(64 * F + 16 * t));
    unsigned t8[8];
    constexpr int R = E % 8, C = E - R;
    if constexpr (C == 0)
    {
        unsigned w8[E];
#pragma unroll
        for (int e = 0; e < E; ++e)
            w8[e] = u[e];
        topk_sel::sort_desc_u32<E>(w8);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            t8[i] = i < E ? w8[i] : 0u;
    }
    else
    {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            t8[i] = u[i];
        topk_sel::sort_desc_u32<8>(t8);
#pragma unroll
        for (int c = 8; c < C; c += 8)
        {
            unsigned b8[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                b8[i] = u[c + i];
            topk_sel::sort_desc_u32<8>(b8);
            narrow::top8_merge(t8, b8);
        }
        if constexpr (R == 1)
        {
            unsigned x = u[C];
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                unsigned const h = t8[i] > x ? t8[i] : x;
                x = t8[i] > x ? x : t8[i];
                t8[i] = h;
            }
        }
        else if constexpr (R > 1)
        {
            unsigned b8[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                b8[i] = i < R ? u[C + i] : 0u;
            topk_sel::sort_desc_u32<8>(b8);
            narrow::top8_merge(t8, b8);
        }
    }
#pragma unroll
    for (int i = 1; i < 8; ++i)
        lst[(i - 1) * kThreads] = t8[i];
    lst[7 * kThreads] = 0u;
    return rounds_half(t8[0], lst, lane, o, nvm, E > 8);
}

template <int TS, int F>
__device__ __forceinline__ int half_dispatch(
    int f, float const* rs, int nvm, int lane, int* o, unsigned* lst, float& sc, float& bi)
{
    if constexpr (F >= kHalfMaxF + 1)
    {
        return rank_half<F, TS>(rs, nvm, lane, o, lst, sc, bi);
    }
    else
    {
        if (f <= F)
            return rank_half<F, TS>(rs, nvm, lane, o, lst, sc, bi);
        return half_dispatch<TS, F + 1>(f, rs, nvm, lane, o, lst, sc, bi);
    }
}

// One row by the whole warp (rows the half-warp path does not take).
__device__ __forceinline__ void rank_full(float const* rs, int n, int lane, int* o, unsigned* lst)
{
    if (n <= kTopK)
    {
        if (lane < kTopK)
            o[lane] = lane < n ? lane : -1;
        return;
    }
    int const nvm = n - 1, kf = nvm >> 7, tail = nvm & 127;
    if (kf >= 1 && kf <= kTailMaxK && tail != 0 && tail <= 64)
    {
        if (tail <= 32)
            tail_dispatch<1, 1>(kf, rs, nvm, lane, o, lst);
        else
            tail_dispatch<2, 1>(kf, rs, nvm, lane, o, lst);
    }
    else
    {
        rank_dispatch<1>(rs, nvm, lane, o, lst);
    }
}

#ifndef WIDE_HALF
#define WIDE_HALF 1
#endif
// Two rows per warp: both through the half-warp path when both are ranked
// rows of nvm <= kHalfMaxNvm, else one after the other by the whole warp.
template <int kWarpMax>
__global__ __launch_bounds__(kThreads, kMinBlocks) void wide_pair_kernel(float const* __restrict__ scores,
    int const* __restrict__ row_n, int* __restrict__ out_ids, RowLayout layout, int rows, int n_cols)
{
    __shared__ unsigned lst[8 * kThreads];
    wait_for_scores();
    int const tid = static_cast<int>(threadIdx.x), lane = tid & 31;
    int const r0 = (static_cast<int>(blockIdx.x) * kRows + (tid >> 5)) * 2, r1 = r0 + 1;
    if (r0 >= rows)
        return; // warp-uniform
    bool const has1 = r1 < rows;
    int const n0 = row_len(row_n, r0, n_cols, layout.rows_per_group),
              n1 = has1 ? row_len(row_n, r1, n_cols, layout.rows_per_group) : 0;
    int const nmax = n0 > n1 ? n0 : n1;
    if (WIDE_HALF && has1 && n0 > kTopK && n1 > kTopK && nmax - 1 <= kHalfMaxNvm)
    {
        int const h = lane >> 4, row = h ? r1 : r0, nvm = (h ? n1 : n0) - 1;
        int const nvm_max = nmax - 1, f = nvm_max >> 6, tail = nvm_max & 63;
        float const* rs = row_scores(scores, row, layout);
        int* o = out_ids + row * kTopK;
        float sc, bi;
        int redo;
        if (tail == 0)
            redo = half_dispatch<0, 1>(f, rs, nvm, lane, o, lst + tid, sc, bi);
        else if (tail <= 16)
            redo = half_dispatch<1, 1>(f, rs, nvm, lane, o, lst + tid, sc, bi);
        else if (tail <= 32)
            redo = half_dispatch<2, 1>(f, rs, nvm, lane, o, lst + tid, sc, bi);
        else
            redo = half_dispatch<0, 1>(f + 1, rs, nvm, lane, o, lst + tid, sc, bi);
            // a row that may owe a 9th key: redone exactly by the whole warp
#pragma unroll 1
        for (int q = 0; q < 2; ++q)
            if (redo & (1 << q))
            {
                float const s = __shfl_sync(0xFFFFFFFFu, sc, 16 * q), b = __shfl_sync(0xFFFFFFFFu, bi, 16 * q);
                int const rq = q ? r1 : r0, nq = (q ? n1 : n0) - 1;
                redo_row(row_scores(scores, rq, layout), nq, lane, out_ids + rq * kTopK, s, b);
            }
        return;
    }
    rank_full(row_scores(scores, r0, layout), n0, lane, out_ids + r0 * kTopK, lst + tid);
    if (has1)
        rank_full(row_scores(scores, r1, layout), n1, lane, out_ids + r1 * kTopK, lst + tid);
}

// One row per warp (launches with rows beyond the pair path).
template <int kWarpMax>
__global__ __launch_bounds__(kThreads, kMinBlocks) void wide_kernel(float const* __restrict__ scores,
    int const* __restrict__ row_n, int* __restrict__ out_ids, RowLayout layout, int rows, int n_cols)
{
    __shared__ unsigned lst[8 * kThreads];
    wait_for_scores();
    int const tid = static_cast<int>(threadIdx.x), lane = tid & 31;
    int const row = static_cast<int>(blockIdx.x) * kRows + (tid >> 5);
    if (row >= rows)
        return; // warp-uniform
    int const n = row_len(row_n, row, n_cols, layout.rows_per_group);
    int* o = out_ids + row * kTopK;
    if (n <= kTopK)
    {
        if (lane < kTopK)
            o[lane] = lane < n ? lane : -1;
        return;
    }
    int const nvm = n - 1, kf = nvm >> 7, tail = nvm & 127;
    float const* rs = row_scores(scores, row, layout);
    if (kf >= 1 && kf <= kTailMaxK && tail != 0 && tail <= 64)
    { // warp-uniform
        if (tail <= 32)
            tail_dispatch<1, 1>(kf, rs, nvm, lane, o, lst + tid);
        else
            tail_dispatch<2, 1>(kf, rs, nvm, lane, o, lst + tid);
    }
    else
    {
        rank_dispatch<1>(rs, nvm, lane, o, lst + tid);
    }
}
} // namespace wide

// Launches `kernel` with programmatic stream serialization when `pdl` is set.
template <typename... Params, typename... Args>
inline void launch(void (*kernel)(Params...), int grid, int block, cudaStream_t st, bool pdl, Args... args)
{
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(grid);
    config.blockDim = dim3(block);
    config.stream = st;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = pdl;
    config.attrs = &attribute;
    config.numAttrs = 1;
    cudaLaunchKernelEx(&config, kernel, args...);
}

// Rows of 16-byte-aligned scores take the float4 families once the launch has
// the rows for them; everything else takes the warp family, packed from
// kWarpPackedMinRows rows on (idle warps would hold registers and halve the
// resident rows), or the block family at one kThreads CTA per row.
template <int kWarpMax>
inline void m3_launch_impl(float const* scores, int const* row_n, int* out, int n_max, RowLayout const& layout,
    int rows, cudaStream_t st, bool pdl)
{
    if constexpr (kWarpMax == kWarpSmallNvm)
    {
        if (narrow::usable(scores, layout, n_max) && rows >= narrow::min_rows(n_max))
        {
            launch(narrow::narrow_kernel, (rows + narrow::kRows - 1) / narrow::kRows, kThreads, st, pdl, scores, row_n,
                out, layout, rows, n_max);
            return;
        }
    }
    else if constexpr (kWarpMax > kWarpSmallNvm)
    {
        if (narrow::usable(scores, layout, n_max) && rows >= wide::min_rows(n_max))
        {
            if (n_max - 1 <= wide::kPairMaxNvm && rows >= wide::kPairMinRows)
            {
                launch(wide::wide_pair_kernel<kWarpMax>, (rows + 2 * wide::kRows - 1) / (2 * wide::kRows), kThreads, st,
                    pdl, scores, row_n, out, layout, rows, n_max);
            }
            else
            {
                launch(wide::wide_kernel<kWarpMax>, (rows + wide::kRows - 1) / wide::kRows, kThreads, st, pdl, scores,
                    row_n, out, layout, rows, n_max);
            }
            return;
        }
    }
    bool const packed = kWarpMax > 0 && rows >= kWarpPackedMinRows;
    int const block = packed ? 32 * kRowsPerBlock : kThreads;
    int const grid = packed ? (rows + kRowsPerBlock - 1) / kRowsPerBlock : rows;
    launch(m3_topk_kernel<kWarpMax>, grid, block, st, pdl, scores, row_n, out, n_max, layout, rows);
}

inline void m3_launch(float const* scores, int const* row_n, int* out, int n_max, int row_stride, int rows,
    int rows_per_group, int64_t group_stride, cudaStream_t st, bool pdl = true)
{
    const RowLayout layout{row_stride, rows_per_group, group_stride};
    switch (m3_warp_max(n_max, rows))
    {
    case kWarpSmallNvm: m3_launch_impl<kWarpSmallNvm>(scores, row_n, out, n_max, layout, rows, st, pdl); break;
    case kWarpNarrowNvm: m3_launch_impl<kWarpNarrowNvm>(scores, row_n, out, n_max, layout, rows, st, pdl); break;
    case kWarpMaxNvm: m3_launch_impl<kWarpMaxNvm>(scores, row_n, out, n_max, layout, rows, st, pdl); break;
    default: m3_launch_impl<0>(scores, row_n, out, n_max, layout, rows, st, pdl); break;
    }
}

} // namespace minimaxM31VllmTopk
