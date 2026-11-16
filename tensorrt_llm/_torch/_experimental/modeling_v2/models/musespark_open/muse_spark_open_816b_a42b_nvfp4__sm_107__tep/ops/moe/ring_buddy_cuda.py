# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""The TP8 ring-buddy placement with the prerouted tables as one NVRTC CUDA CTA (``rb_tables``).

``ring_buddy_assign``'s Triton program, the TEP8 decode case it serves on the router path (8 ranks,
32-expert blocks, balanced count rule or ``tep8_tile_placement``'s tile-cost rule, two windows,
``meta_tile`` tables), rewritten as one
1024-thread CTA, one slot per thread (up to 128 rows): the router push's successor, between it and
FC1. VR nsys (wall + rp9, rank 0, medians): the Triton program ends 3.4 / 4.1 / 3.9 / 4.3 us after
the push at bs8 / 9 / 12 / 16 and FC1 starts when it ends; its 1-GPU graph-replayed time is
3.9-4.1 us at 72..128 rows (G27): its chain of ``[SLOTS, RANKS]`` reductions, a global-scratch
histogram and global atomics, not its launch. Here every step is one or two warp instructions and
a barrier:

* census: per block ``b`` one ``redux.sync.or`` of ``1 << i`` per warp, one shared ``atomicOr``
  per warp (the same presence words);
* flow: threads 0..7 take the smallest feasible maximum ``T`` and the pointwise smallest flow
  ``x`` (``balance_ring``, the Triton program's ``need`` / ``x``), integer arithmetic;
* movers and local ids: ``popc`` of the present experts above a slot's (the same rule);
* ``tilew`` (``tep8_tile_placement``, served at 41..64 rows): tokens per expert by shared atomics,
  warp 1 + b takes block b's expert costs, the cost flow (``ring_x`` again), the tile rule's movers
  (a reverse warp scan of the costs) and the count rule's, and every thread takes the tile rule
  where its busiest rank costs less (``place_tile_reference``; the Triton ``TILEW`` block);
* tables: a slot's position within its expert is the number of earlier slots of that expert
  (``match.any`` in the warp + an exclusive prefix over warps), so an expert's rows are in token
  order (the Triton program's CTA-scope atomics give arrival order); the CTA offsets are an
  exclusive scan of the experts' tiles; ``c2b`` / ``c2m`` one thread per (expert, tile).

Bits: ``local_ids``, the BF16 weights (``cvt.rn.bf16.f32``, Triton's cast), ``stats``, ``nnc``,
``tnp``, and ``c2b`` / ``c2m`` of the ``nnc`` CTAs equal the Triton program's; ``e2p`` / ``p2t``
are the same placement (every slot of this rank in its expert's tiles, ``p2t`` back to the
token) with an expert's rows in token order: the row-independent expert GEMMs see the same rows.

The kernel's first global read follows ``griddepcontrol.wait`` (``ld_dep``: a coherent
``ld.relaxed.gpu`` in volatile asm; no ``const __restrict__`` parameter, whose read-only loads the
compiler may issue as ``ld.global.nc`` above the wait), and it is launched as a PDL
dependent (``CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION``); with the router push's
``EARLY_TRIGGER`` build (``router_push.configure(tep8_early_trigger=True)``, the same lever
param) the push triggers it right after its own dependency wait, so this CTA is resident and
waiting when the push completes. Armed by ``ring_buddy_assign.configure(cuda_tables=True)``
(the ``tep8_router_push`` lever's ``RB_CUDA``); compiled at the first eager call, never under
capture (a captured call before that keeps the Triton program).
"""

from __future__ import annotations

import ctypes
from typing import Dict, List, Optional, Tuple

EXPERTS = 256
RANKS = 8
BLOCK = 32
WINDOW = 2 * BLOCK
TOP_K = 8
#: One slot per thread: 1024 slots, 128 rows.
THREADS = 1024
MAX_SLOTS = THREADS
#: Tiles per expert the ``c2b`` / ``c2m`` step covers (one thread per (expert, tile)): 128 rows at 8-token tiles.
MAX_TILES = THREADS // WINDOW
#: The ``meta_tile`` values the kernel takes (``ring_buddy_assign.is_valid``'s).
TILES = (8, 16, 32, 64)

SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
#define NT 1024
#define RANKS 8
#define BLOCK 32
#define WIN 64
#define EXPERTS 256
#define NWARP (NT / 32)
#define MAXT (NT / WIN)
#define FULL 0xffffffffu
static_assert(NWARP * 2 == WIN && WIN * MAXT == NT && RANKS * BLOCK == EXPERTS && NWARP > RANKS, "layout");
__device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ u16 bf16_rn(float f) {
  u16 r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
  return r;
}
// A word the PDL predecessor wrote, read after griddepcontrol.wait: a coherent load the compiler keeps after the wait.
// (No const __restrict__ on the predecessor's buffers: a read-only no-alias pointer lets the compiler issue its loads as
// ld.global.nc and hoist them above the wait, which an early-triggering predecessor (the push's EARLY_TRIGGER) turns
// into reads of the buffer before the predecessor wrote it: the 8-rank gate's mismatches, 1026468.)
__device__ __forceinline__ u32 ld_dep(const void* p) {
  u32 v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// The ring flow (balance_ring) on the per-block loads ld (shared): the smallest feasible maximum T over every arc of
// n blocks ending at r, then rank me's pointwise smallest outflow max_n(arc - n T, 0).
__device__ __forceinline__ int ring_x(const int* ld, int me) {
  int load[RANKS];
#pragma unroll
  for (int r = 0; r < RANKS; ++r) load[r] = ld[r];
  int T = 0;
#pragma unroll
  for (int r = 0; r < RANKS; ++r) {
    int arc = 0;
#pragma unroll
    for (int n = 1; n <= RANKS; ++n) {
      arc += load[(r - n + 1 + RANKS) % RANKS];  // the n blocks ending at r
      T = imax(T, n == RANKS ? (arc + RANKS - 1) / RANKS : (arc + n) / (n + 1));
    }
  }
  int xr = 0;
#pragma unroll
  for (int r = 0; r < RANKS; ++r) {
    int arc = 0, xm = 0;
#pragma unroll
    for (int n = 1; n <= RANKS; ++n) {
      arc += load[(r - n + 1 + RANKS) % RANKS];
      xm = imax(xm, arc - n * T);
    }
    if (r == me) xr = xm;
  }
  return xr;
}
// tilew (tep8_tile_placement): the same ring flow on expert costs cost_a + cost_b (tiles - 1), the tile rule's movers
// (descending local index while the moved cost stays short of the outflow), taken only where its busiest rank costs
// less than the count rule's (ring_buddy_assign's TILEW block, step by step).
extern "C" __global__ void __launch_bounds__(NT, 1)
rb_tables(const int* sel, const float* wts, int* local_ids, u16* wbf, int* stats, int rank, int num_slots, int tile,
          int* e2p, int* p2t, int* c2b, int* c2m, int* nnc, int* tnp, int tilew, int cost_a, int cost_b) {
  __shared__ u32 s_word[RANKS];          // presence word of each block
  __shared__ int s_load[RANKS];          // present experts of each block
  __shared__ int s_x[RANKS];             // the count rule's flow: experts block b moves to rank b + 1
  __shared__ int s_tok[EXPERTS];         // tilew: tokens of each expert
  __shared__ int s_wload[RANKS];         // tilew: cost of each block
  __shared__ int s_wx[RANKS];            // tilew: cost block b moves out
  __shared__ u32 s_mt[RANKS], s_mc[RANKS];  // tilew: movers of each block, tile rule / count rule
  __shared__ int s_sum[4][RANKS];        // tilew: stay / out cost of each block, tile rule then count rule
  __shared__ int s_wc[NWARP][WIN + 1];   // a warp's rows of each local expert, then their exclusive prefix over warps
  __shared__ int s_cnt[WIN];             // rows of each local expert
  __shared__ int s_off[WIN];             // first CTA of each local expert
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  if (t < RANKS) s_word[t] = 0u;
  if (t < EXPERTS) s_tok[t] = 0;
  s_wc[warp][lane] = 0;
  s_wc[warp][lane + 32] = 0;
  // nothing of the predecessor's (the router push / the routing kernel) is read before this
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const bool v = t < num_slots;
  const int g = v ? (int)ld_dep(sel + t) : 0;
  const float w = v ? __uint_as_float(ld_dep(wts + t)) : 0.0f;
  const int blk = g >> 5, i = g & 31;
  __syncthreads();
  // census: one presence word per block (a repeated expert sets the same bit); tilew: tokens per expert
#pragma unroll
  for (int b = 0; b < RANKS; ++b) {
    const u32 m = __reduce_or_sync(FULL, (v && blk == b) ? (1u << i) : 0u);
    if (lane == 0 && m) atomicOr(&s_word[b], m);
  }
  if (tilew && v) atomicAdd(&s_tok[g & (EXPERTS - 1)], 1);
  __syncthreads();
  if (t < RANKS) s_load[t] = __popc(s_word[t]);
  // tilew, warp 1 + b: block b's expert costs and their sum
  const int tb = warp - 1;
  int cost = 0;
  if (tilew && tb >= 0 && tb < RANKS) {
    const int c = s_tok[tb * BLOCK + lane];
    cost = c > 0 ? cost_a + cost_b * ((c + tile - 1) / tile - 1) : 0;
    const int sum = (int)__reduce_add_sync(FULL, (unsigned)cost);
    if (lane == 0) s_wload[tb] = sum;
  }
  __syncthreads();
  if (t < RANKS) s_x[t] = ring_x(s_load, t);
  else if (tilew && t >= 32 && t < 32 + RANKS) s_wx[t - 32] = ring_x(s_wload, t - 32);
  __syncthreads();
  if (tilew && tb >= 0 && tb < RANKS) {
    int after = cost;  // cost of the block's experts at or above this local index
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int y = __shfl_down_sync(FULL, after, d);
      if (lane + d < 32) after += y;
    }
    const int before = after - cost, wxb = s_wx[tb];
    const bool move_t = cost > 0 && before < wxb && 2 * wxb >= before + after;
    const u32 pres = __ballot_sync(FULL, cost > 0);
    const bool move_c = cost > 0 && __popc((pres >> lane) >> 1) < s_x[tb];
    const u32 mt = __ballot_sync(FULL, move_t), mc = __ballot_sync(FULL, move_c);
    const int stay_t = (int)__reduce_add_sync(FULL, (unsigned)(move_t ? 0 : cost));
    const int stay_c = (int)__reduce_add_sync(FULL, (unsigned)(move_c ? 0 : cost));
    if (lane == 0) {
      s_mt[tb] = mt;
      s_mc[tb] = mc;
      s_sum[0][tb] = stay_t;
      s_sum[1][tb] = s_wload[tb] - stay_t;
      s_sum[2][tb] = stay_c;
      s_sum[3][tb] = s_wload[tb] - stay_c;
    }
  }
  __syncthreads();
  // movers: the count rule's x[b] present experts of block b with the highest local index, or tilew's choice
  bool take_t = false;
  if (tilew) {
    int max_t = 0, max_c = 0;
#pragma unroll
    for (int r = 0; r < RANKS; ++r) {
      max_t = imax(max_t, s_sum[0][r] + s_sum[1][(r + RANKS - 1) % RANKS]);
      max_c = imax(max_c, s_sum[2][r] + s_sum[3][(r + RANKS - 1) % RANKS]);
    }
    take_t = max_t < max_c;
  }
  const int bk = blk & (RANKS - 1);
  const bool moved = tilew ? (((take_t ? s_mt[bk] : s_mc[bk]) >> i) & 1u) != 0u
                           : __popc((s_word[bk] >> i) >> 1) < s_x[bk];
  if (t < RANKS) {
    const int p = (t + RANKS - 1) % RANKS;
    const int xt = tilew ? __popc(take_t ? s_mt[t] : s_mc[t]) : s_x[t];
    const int xp = tilew ? __popc(take_t ? s_mt[p] : s_mc[p]) : s_x[p];
    stats[t] = s_load[t];
    stats[RANKS + t] = s_load[t] - xt + xp;
  }
  const int er = (blk + (moved ? 1 : 0)) % RANKS;
  const int local = er == rank ? (blk == rank ? i : BLOCK + i) : 2 * BLOCK + g % (EXPERTS - 2 * BLOCK);
  if (v) {
    local_ids[t] = local;
    wbf[t] = bf16_rn(w);
  }
  // tables: a slot's position in its expert = the expert's slots before it (token order)
  const bool vm = v && local < WIN;
  const u32 peers = __match_any_sync(FULL, vm ? local : WIN + lane);
  const int intra = __popc(peers & ((1u << lane) - 1u));
  if (vm && intra == 0) s_wc[warp][local] = __popc(peers);
  __syncthreads();
#pragma unroll
  for (int h = 0; h < 2; ++h) {  // warp q: experts 2q and 2q + 1, lane = warp of the count
    const int e = 2 * warp + h;
    const int c = s_wc[lane][e];
    int inc = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int y = __shfl_up_sync(FULL, inc, d);
      if (lane >= d) inc += y;
    }
    s_wc[lane][e] = inc - c;
    if (lane == 31) s_cnt[e] = inc;
  }
  __syncthreads();
  if (warp == 0) {  // the experts' first CTAs: an exclusive scan of their tiles, two experts per lane
    const int a = (s_cnt[2 * lane] + tile - 1) / tile, b = (s_cnt[2 * lane + 1] + tile - 1) / tile;
    int inc = a + b;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int y = __shfl_up_sync(FULL, inc, d);
      if (lane >= d) inc += y;
    }
    const int ex = inc - a - b;
    s_off[2 * lane] = ex;
    s_off[2 * lane + 1] = ex + a;
    if (lane == 31) {
      nnc[0] = inc;
      tnp[0] = inc * tile;
    }
  }
  __syncthreads();
  if (vm) {
    const int p = s_off[local] * tile + s_wc[warp][local] + intra;
    e2p[t] = p;
    p2t[p] = t >> 3;
  } else if (v) {
    e2p[t] = -1;
  }
  {
    const int e = t / MAXT, tt = t % MAXT;
    const int c = s_cnt[e];
    if (tt < (c + tile - 1) / tile) {
      const int cta = s_off[e] + tt;
      c2b[cta] = e;
      c2m[cta] = imin((cta + 1) * tile, s_off[e] * tile + c);
    }
  }
}
"""


def takes(ranks: int, balance: bool, half: bool, slots: int, tile: Optional[int]) -> bool:
    """Whether a ``ring_buddy_assign`` call is this kernel's case: the 8-rank balanced two-window placement (by
    expert count, or by tile cost under ``tep8_tile_placement``) with tables (``meta_tile``) of at most
    ``MAX_SLOTS`` slots. Rank-uniform facts only."""
    return ranks == RANKS and bool(balance) and not half and 0 < slots <= MAX_SLOTS and tile in TILES


def _ring_x(load: List[int]) -> List[int]:
    """``ring_x`` of the source for every rank: the smallest feasible maximum, then the pointwise smallest flow."""
    t_max = 0
    for r in range(RANKS):
        arc = 0
        for n in range(1, RANKS + 1):
            arc += load[(r - n + 1 + RANKS) % RANKS]
            t_max = max(t_max, (arc + RANKS - 1) // RANKS if n == RANKS else (arc + n) // (n + 1))
    x = []
    for r in range(RANKS):
        arc = xm = 0
        for n in range(1, RANKS + 1):
            arc += load[(r - n + 1 + RANKS) % RANKS]
            xm = max(xm, arc - n * t_max)
        x.append(xm)
    return x


def model(
    flat: List[int], rank: int, tile: int, tilew: bool = False, cost: Tuple[int, int] = (20, 11)
) -> Dict[str, object]:
    """The kernel's algorithm in plain Python on one call's ``flat`` (row-major ``[rows, 8]``) ids: per slot the
    local id and ``e2p``, ``p2t`` as a dict (only the entries the kernel writes), ``c2b`` / ``c2m`` of the
    ``nnc`` CTAs, ``nnc``, ``tnp`` and ``stats``, step by step as the CUDA source computes them (``tilew``: the
    tile placement at expert costs ``cost[0] + cost[1] (tiles - 1)``)."""
    slots = len(flat)
    assert 0 < slots <= MAX_SLOTS and tile in TILES
    word = [0] * RANKS
    for g in flat:
        word[g >> 5] |= 1 << (g & 31)
    load = [bin(w).count("1") for w in word]
    x = _ring_x(load)
    if tilew:
        tok = [0] * EXPERTS
        for g in flat:
            tok[g] += 1
        costs = [cost[0] + cost[1] * ((c + tile - 1) // tile - 1) if c > 0 else 0 for c in tok]
        wload = [sum(costs[b * BLOCK : (b + 1) * BLOCK]) for b in range(RANKS)]
        wx = _ring_x(wload)
        mt, mc, sums = [0] * RANKS, [0] * RANKS, [[0] * RANKS for _ in range(4)]
        for b in range(RANKS):
            c_b = costs[b * BLOCK : (b + 1) * BLOCK]
            pres = sum(1 << j for j in range(BLOCK) if c_b[j] > 0)
            for j in range(BLOCK):
                after = sum(c_b[j:])
                before = after - c_b[j]
                if c_b[j] > 0 and before < wx[b] and 2 * wx[b] >= before + after:
                    mt[b] |= 1 << j
                if c_b[j] > 0 and bin((pres >> j) >> 1).count("1") < x[b]:
                    mc[b] |= 1 << j
            sums[0][b] = sum(c_b[j] for j in range(BLOCK) if not (mt[b] >> j) & 1)
            sums[1][b] = wload[b] - sums[0][b]
            sums[2][b] = sum(c_b[j] for j in range(BLOCK) if not (mc[b] >> j) & 1)
            sums[3][b] = wload[b] - sums[2][b]
        max_t = max(sums[0][r] + sums[1][(r + RANKS - 1) % RANKS] for r in range(RANKS))
        max_c = max(sums[2][r] + sums[3][(r + RANKS - 1) % RANKS] for r in range(RANKS))
        movers = mt if max_t < max_c else mc
        xs = [bin(m).count("1") for m in movers]
        stats = load + [load[r] - xs[r] + xs[(r + RANKS - 1) % RANKS] for r in range(RANKS)]
    else:
        stats = load + [load[r] - x[r] + x[(r + RANKS - 1) % RANKS] for r in range(RANKS)]
    local = []
    for g in flat:
        blk, i = g >> 5, g & 31
        if tilew:
            moved = (movers[blk] >> i) & 1 == 1
        else:
            moved = bin((word[blk] >> i) >> 1).count("1") < x[blk]
        er = (blk + (1 if moved else 0)) % RANKS
        local.append((i if blk == rank else BLOCK + i) if er == rank else 2 * BLOCK + g % (EXPERTS - 2 * BLOCK))
    # the warp-level ranking: per warp the peers of each slot, then an exclusive prefix over warps
    nwarp = THREADS // 32
    wc = [[0] * WINDOW for _ in range(nwarp)]
    intra = [0] * slots
    for s, e in enumerate(local):
        if e < WINDOW:
            w, lane = divmod(s, 32)
            intra[s] = sum(1 for q in range(w * 32, w * 32 + lane) if local[q] == e)
            wc[w][e] += 1
    cnt = [sum(wc[w][e] for w in range(nwarp)) for e in range(WINDOW)]
    pre = [[sum(wc[q][e] for q in range(w)) for e in range(WINDOW)] for w in range(nwarp)]
    tiles = [(c + tile - 1) // tile for c in cnt]
    off = [sum(tiles[:e]) for e in range(WINDOW)]
    nnc = sum(tiles)
    e2p, p2t = [], {}
    for s, e in enumerate(local):
        if e < WINDOW:
            p = off[e] * tile + pre[s // 32][e] + intra[s]
            e2p.append(p)
            p2t[p] = s >> 3
        else:
            e2p.append(-1)
    c2b, c2m = [0] * nnc, [0] * nnc
    for e in range(WINDOW):
        for tt in range(min(tiles[e], MAX_TILES)):
            cta = off[e] + tt
            c2b[cta] = e
            c2m[cta] = min((cta + 1) * tile, off[e] * tile + cnt[e])
    return dict(local=local, e2p=e2p, p2t=p2t, c2b=c2b, c2m=c2m, nnc=nnc, tnp=nnc * tile, stats=stats)


def _chk(r):
    if int(r[0]) != 0:
        raise RuntimeError(f"rb_tables: CUDA/NVRTC error {r!r}")
    return r[1] if len(r) == 2 else r[1:]


class Kernel:
    """``rb_tables`` compiled for one device (sm_107), launched as one PDL-dependent CTA."""

    def __init__(self, device_index: int):
        import torch
        from cuda.bindings import driver as cu
        from cuda.bindings import nvrtc

        self.cu = cu
        major, minor = torch.cuda.get_device_capability(device_index)
        if (major, minor) != (10, 7):
            raise RuntimeError("rb_tables: built for sm_107 only")
        prog = _chk(nvrtc.nvrtcCreateProgram(SRC.encode(), b"rb_tables.cu", 0, [], []))
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17", b"-default-device"]
        r = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if int(r[0]) != 0:
            n = _chk(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("rb_tables NVRTC compile failed:\n" + log.decode(errors="replace"))
        n = _chk(nvrtc.nvrtcGetCUBINSize(prog))
        self.cubin = b" " * n
        _chk(nvrtc.nvrtcGetCUBIN(prog, self.cubin))
        with torch.cuda.device(device_index):
            self.module = _chk(cu.cuModuleLoadData(self.cubin))
            self.fn = _chk(cu.cuModuleGetFunction(self.module, b"rb_tables"))
            attr = cu.CUfunction_attribute
            self.regs = int(_chk(cu.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, self.fn)))
            self.local_bytes = int(
                _chk(cu.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self.fn))
            )
        self._cfg: Dict[int, object] = {}

    def _config(self, stream: int):
        cfg = self._cfg.get(stream)
        if cfg is None:
            cu = self.cu
            cfg = cu.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = 1, 1, 1
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = THREADS, 1, 1
            cfg.sharedMemBytes = 0
            cfg.hStream = cu.CUstream(stream)
            a = cu.CUlaunchAttribute()
            a.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            a.value.programmaticStreamSerializationAllowed = 1
            cfg.attrs = [a]
            cfg.numAttrs = 1
            self._cfg[stream] = cfg
        return cfg

    def launch(self, args: Tuple[int, ...], stream: int) -> None:
        vp, ii = ctypes.c_void_p, ctypes.c_int
        types_ = (vp, vp, vp, vp, vp, ii, ii, ii, vp, vp, vp, vp, vp, vp, ii, ii, ii)
        _chk((self.cu.cuLaunchKernelEx(self._config(stream), self.fn, (args, types_), 0)[0], None))


_KERNELS: Dict[int, Kernel] = {}
#: Devices where ``rb_tables`` failed to build (the reason): their calls keep the Triton program.
_FAILED: Dict[int, str] = {}


def launch(
    selected,
    weights,
    rank: int,
    tile: int,
    local_ids,
    weights_bf16,
    stats,
    meta,
    tilew: bool = False,
    cost: Tuple[int, int] = (20, 11),
) -> bool:
    """Write ``ring_buddy_assign``'s outputs (ids, BF16 weights, stats, the ``meta`` tables) for one call with
    ``rb_tables`` (``tilew``: the tile placement at ``cost``); False (nothing launched) when the kernel is not
    compiled yet under capture or failed to build (the caller's Triton program then runs the call)."""
    import torch

    dev = selected.device.index
    k = _KERNELS.get(dev)
    if k is None:
        if dev in _FAILED or torch.cuda.is_current_stream_capturing():
            return False
        try:
            k = Kernel(dev)
        except Exception as exc:  # noqa: BLE001 - a build failure keeps the Triton program
            _FAILED[dev] = f"{type(exc).__name__}: {exc}"
            try:
                from tensorrt_llm.logger import logger as _logger

                _logger.warning(
                    f"[musespark_open] ring_buddy_assign: rb_tables inactive ({_FAILED[dev]}); the Triton program "
                    "writes the tables"
                )
            except Exception:  # noqa: BLE001 - logging only
                pass
            return False
        _KERNELS[dev] = k
        try:
            from tensorrt_llm.logger import logger as _logger

            _logger.info(
                f"[musespark_open] ring_buddy_assign: the TP8 two-window tables run rb_tables (one {THREADS}-thread "
                f"CUDA CTA, PDL; {k.regs} regs, {k.local_bytes} B local)"
            )
        except Exception:  # noqa: BLE001 - logging only
            pass
    args = (
        selected.data_ptr(),
        weights.data_ptr(),
        local_ids.data_ptr(),
        weights_bf16.data_ptr(),
        stats.data_ptr(),
        int(rank),
        int(selected.numel()),
        int(tile),
        meta["e2p"].data_ptr(),
        meta["p2t"].data_ptr(),
        meta["c2b"].data_ptr(),
        meta["c2m"].data_ptr(),
        meta["nnc"].data_ptr(),
        meta["tnp"].data_ptr(),
        int(bool(tilew)),
        int(cost[0]),
        int(cost[1]),
    )
    k.launch(args, torch.cuda.current_stream(selected.device).cuda_stream)
    return True
