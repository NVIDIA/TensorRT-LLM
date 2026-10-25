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
// FlashMLA's SM100 sparse prefill kernel (third_party/flashmla: deepseek-ai/FlashMLA 15f13e5, MIT;
// sm100/prefill/sparse/fwd/head64/phase1.cuh, d_qk = d_v = 512, no RoPE part) adapted to
// query tiles: a CTA takes TOK = 64 / H consecutive tokens of one sequence x their H heads (64 query rows,
// consecutive rows of the [T x H, 512] query buffer) and streams the union of the tokens' visible pools
// from sparse_mla_union's workspace (sorted pools, the cache row of each pool's first position, a bit per
// token) -- one TMA gather4 per pool -- with one validity mask per token (pool membership and position).
// Changes to the original: the Q tile by 2-D TMA at the tile's first row, the pool rows and masks from
// the union workspace, a mask per token, and the output stored directly by the softmax threads for the
// tile's own rows (no lse / max logits).
#include "fmla.h"

#include "sm100/prefill/sparse/fwd/head64/config.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include <kerutils/kerutils.cuh>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"
#include "sm100/prefill/sparse/common_subroutine.h"

namespace sm100::fwd::head64u {

using namespace cute;
using namespace sm100::fwd::head64;

constexpr int MAX_TOK = 8;  // tokens per tile (H >= 8)

struct SharedMemoryPlanU {
  union {
    struct {
      array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> _k_rope_pad;
      array_aligned<bf16, cosize_v<SmemLayoutKNoPE>> _k_pad[2];  // so that q_nope covers k[2]
      array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope;
    } q_full;
    struct {
      array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> k_rope;
      array_aligned<bf16, cosize_v<SmemLayoutKNoPE>> k_nope[NUM_BUFS];
    } k;
  } u;
  float p_exchange_buf[4][32 * (B_TOPK / 2)];
  bf16 s[B_H * B_TOPK];
  alignas(16) char is_k_valid[NUM_BUFS][MAX_TOK][B_TOPK / 8];
  int tok_pos[MAX_TOK];
  transac_bar_t bar_prologue_q_nope, bar_prologue_utccp_nope;
  transac_bar_t bar_qk_nope_done[NUM_BUFS];
  transac_bar_t bar_sv_done[NUM_BUFS];
  transac_bar_t bar_kv_nope_ready[NUM_BUFS][2];
  transac_bar_t bar_p_free;
  transac_bar_t bar_so_ready;
  transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
  array_aligned<uint32_t, 1> tmem_start_addr;
  float rowwise_max_buf[128], rowwise_li_buf[128];
};

struct UnionParams {
  const int2* tiles;      // {first token, tokens}
  const uint8_t* ws;      // union workspace (sparse_mla_union)
  int union_bytes, union_cap;
  const int* positions;   // [T]
  bf16* out;              // [T x H, 512]
  float sm_scale_div_log2;
  // split-KV (grid (tiles, nsplit), H <= 16): CTA (i, s) takes the s-th range of the union's 64-row
  // blocks and writes its unnormalized O rows, max (natural log units) and sum into partial
  // (mla_attn_combine's layout: [token][split][16 rows] x 514 floats); nullptr: out
  float* partial;
  int pre;  // split-KV: prologue before the PDL wait (the pre-pass triggers after its own wait)
};

template <int TOK>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
    sparse_attn_union_kernel(__grid_constant__ const UnionParams params,
                             __grid_constant__ const CUtensorMap tm_q,
                             __grid_constant__ const CUtensorMap tm_kv) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
  constexpr int H = B_H / TOK;
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_idx = threadIdx.x % 32;
  const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
  const int idx_in_warpgroup = threadIdx.x % 128;
  // split-KV decode: launched with PDL behind the union pre-pass (its CTAs resident early); the
  // union is read after the wait, through L2 (no read-only loads nvcc could hoist above it).
  // params.pre: the pre-pass triggers this grid only after its own wait (the query rows' producer
  // has completed), so the prologue (query tile, barriers, tensor memory) runs before the wait.
  if (params.partial) {
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
    if (!params.pre) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  }
  const int2 tile = params.tiles[blockIdx.x];  // (a constant table: {4 i, 4})
  const int t0 = tile.x, nt = tile.y;

  extern __shared__ char wksp_buf[];
  SharedMemoryPlanU& plan = *reinterpret_cast<SharedMemoryPlanU*>(wksp_buf);

  TiledMMA tiled_mma_P = TiledMMA_P{};
  TiledMMA tiled_mma_O = TiledMMA_O{};
  Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
  Tensor tQ_nope_part0 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
      partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<(D_V / 2) / 2>>{}));
  Tensor tQ_nope_part1 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
      partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<(D_V / 2) / 2>>{}));
  Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<B_H>, Int<D_V>>{});
  tP.data().get() = tmem_cols::P;
  tQ_nope_part0.data().get() = tmem_cols::Q;
  tQ_nope_part1.data().get() = tmem_cols::Q + 64;
  tO.data().get() = tmem_cols::O;

  if (threadIdx.x < MAX_TOK) plan.tok_pos[threadIdx.x] = threadIdx.x < nt ? params.positions[t0 + threadIdx.x] : -1;

  if (warp_idx == 0) {
    if (elect_one_sync()) {
      cute::prefetch_tma_descriptor(&tm_q);
      plan.bar_prologue_q_nope.init(1);
      fence_barrier_init();
      // the tile's 64 query rows: 8 boxes of 64 rows x 64 dims (128-byte swizzle) = SmemLayoutQNoPE
      constexpr uint64_t kHint = static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_FIRST);
      CUTE_UNROLL
      for (int a = 0; a < D_V / 64; ++a)
        cute::SM90_TMA_LOAD_2D::copy(&tm_q, reinterpret_cast<uint64_t*>(&plan.bar_prologue_q_nope),
                                     kHint, plan.u.q_full.q_nope.data() + a * B_H * 64, a * 64, t0 * H);
      cute::prefetch_tma_descriptor(&tm_kv);
      plan.bar_prologue_utccp_nope.init(1);
      CUTE_UNROLL
      for (int i = 0; i < NUM_BUFS; ++i) {
        plan.bar_qk_nope_done[i].init(1);
        plan.bar_sv_done[i].init(1);
        plan.bar_kv_nope_ready[i][0].init(1);
        plan.bar_kv_nope_ready[i][1].init(1);
        plan.bar_k_valid_ready[i].init(B_TOPK / 8);
        plan.bar_k_valid_free[i].init(128);
      }
      plan.bar_p_free.init(128);
      plan.bar_so_ready.init(128);
      fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
    TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }

  __syncthreads();
  if (params.partial && params.pre) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const uint8_t* uws = params.ws + (size_t)blockIdx.x * params.union_bytes;
  // (split s: blocks [kb0, kb0 + per) of the union's; a split without any runs one empty block)
  const int npools_all = __ldcg(reinterpret_cast<const int*>(uws));
  const int nkb_all = max(cute::ceil_div(npools_all * 4, (int)B_TOPK), 1);
  const int kb_per = (nkb_all + (int)gridDim.y - 1) / (int)gridDim.y;
  const int kb0 = min(nkb_all, (int)blockIdx.y * kb_per);
  const int npools = max(0, min(npools_all - kb0 * (B_TOPK / 4), kb_per * (B_TOPK / 4)));
  const int* upools = reinterpret_cast<const int*>(uws + 16) + kb0 * (B_TOPK / 4);
  const int* urows = reinterpret_cast<const int*>(uws + 16) + params.union_cap + kb0 * (B_TOPK / 4);
  const uint8_t* umember = uws + 16 + (size_t)params.union_cap * 8 + kb0 * (B_TOPK / 4);
  const int num_k_blocks = max(cute::ceil_div(npools * 4, (int)B_TOPK), 1);

  if (warpgroup_idx == 0) {
    // Scale & Exp warps (one query row per thread pair: row idx_in_warpgroup % 64 = token row / H)
    float mi = MAX_INIT_VAL;
    float li = 0.0f;
    float real_mi = -CUDART_INF_F;
    const int row = idx_in_warpgroup % 64;
    const int tok = row / H;

    bf16* sS_base = plan.s + lane_idx * 8 + (warp_idx & 1) * (B_H / 2) * 8 + (warp_idx / 2) * B_H * (B_TOPK / 2);
    static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

    CUTE_NO_UNROLL
    for (int k = 0; k < num_k_blocks; ++k) {
      NamedBarrier::arrive_and_wait(64, NamedBarriers::wg0_warp02_sync + (warp_idx & 1));
      plan.bar_qk_nope_done[k % NUM_BUFS].wait((k / NUM_BUFS) & 1);
      plan.bar_k_valid_ready[k % NUM_BUFS].wait((k / NUM_BUFS) & 1);
      ku::tcgen05_after_thread_sync();

      float p[NUM_ELEMS_PER_THREAD];
      retrieve_mask_and_reduce_p<NUM_ELEMS_PER_THREAD, tmem_cols::P, NamedBarriers::wg0_warp02_sync,
                                 NamedBarriers::wg0_warp13_sync, false>(
          plan.is_k_valid[k % NUM_BUFS][tok], warp_idx, lane_idx, [&]() { plan.bar_p_free.arrive(); },
          plan.p_exchange_buf, p);
      plan.bar_k_valid_free[k % NUM_BUFS].arrive();

      float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
      cur_pi_max *= params.sm_scale_div_log2;

      plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
      NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
      cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup ^ 64]);
      real_mi = max(real_mi, cur_pi_max);
      bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

      float new_max, scale_for_old;
      if (!should_scale_o) {
        scale_for_old = 1.0f;
        new_max = mi;
      } else {
        new_max = max(cur_pi_max, mi);
        scale_for_old = exp2f(mi - new_max);
      }
      mi = new_max;

      nv_bfloat162 s[NUM_ELEMS_PER_THREAD / 2];
      float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>(s, p, params.sm_scale_div_log2, new_max);
      li = fma(li, scale_for_old, cur_sum);

      if (k > 0) plan.bar_sv_done[(k - 1) % NUM_BUFS].wait(((k - 1) / NUM_BUFS) & 1);
      CUTE_UNROLL
      for (int i = 0; i < NUM_ELEMS_PER_THREAD / 8; i += 1)
        *(uint128_t*)(sS_base + B_H * 8 * i) = *(uint128_t*)(s + i * 4);

      if (k > 0 && should_scale_o) {
        ku::tcgen05_after_thread_sync();
        rescale_O<D_V, 32, tmem_cols::O>(scale_for_old);
        ku::tcgen05_before_thread_sync();
      }

      fence_view_async_shared();
      plan.bar_so_ready.arrive();
    }

    // Epilogue
    if (real_mi == -CUDART_INF_F) {
      li = 0.0f;
      mi = -CUDART_INF_F;
    }
    plan.rowwise_li_buf[idx_in_warpgroup] = li;
    NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
    li += plan.rowwise_li_buf[idx_in_warpgroup ^ 64];

    plan.bar_sv_done[(num_k_blocks - 1) % NUM_BUFS].wait(((num_k_blocks - 1) / NUM_BUFS) & 1);
    ku::tcgen05_after_thread_sync();

    float output_scale = li > 0.f ? __fdividef(1.0f, li) : 0.f;
    constexpr int B_EPI = 64;
    float2 o[B_EPI / 2];
    bool have_valid_indices = __any_sync(0xffffffff, li != 0);
    if (!have_valid_indices) {
      CUTE_UNROLL
      for (int i = 0; i < B_EPI / 2; ++i) o[i].x = o[i].y = 0.0f;
    }
    float2 output_scale_float2 = make_float2(output_scale, output_scale);
    const bool store = row < nt * H;
    bf16* orow = params.out + ((int64_t)t0 * H + row) * D_V;

    if (params.partial) {
      // the O rows through shared memory (the consumed KV buffers), then row by row with consecutive
      // threads on consecutive columns (a thread per row storing straight from its TMEM lanes would
      // write 8-byte pieces 2 KB apart)
      constexpr int SP = D_V + 2;  // staging pitch (floats)
      float* stage = reinterpret_cast<float*>(&plan.u);
      static_assert(sizeof(plan.u) >= 64 * SP * sizeof(float), "staging fits the KV buffers");
      if (idx_in_warpgroup < 64) {
        plan.rowwise_max_buf[row] = li > 0.f ? mi * 0.6931471805599453f : -CUDART_INF_F;
        plan.rowwise_max_buf[64 + row] = li;
      }
      CUTE_UNROLL
      for (int c = 0; c < 2; ++c) {
        CUTE_UNROLL
        for (int k = 0; k < (D_V / 4) / B_EPI; ++k) {
          if (have_valid_indices) {
            ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + c * 128 + k * B_EPI, o);
            cutlass::arch::fence_view_async_tmem_load();
          }
          const int col0 = c * (D_V / 2) + (idx_in_warpgroup / 64) * (D_V / 4) + k * B_EPI;
          CUTE_UNROLL
          for (int i = 0; i < B_EPI / 2; ++i)
            *reinterpret_cast<float2*>(stage + row * SP + col0 + 2 * i) = o[i];
        }
      }
      NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
      const int rows = nt * H;
      for (int r = 0; r < rows; ++r) {
        float* w = params.partial + ((size_t)(t0 + r / H) * gridDim.y + blockIdx.y) * 16 * (D_V + 2) +
                   (size_t)(r % H) * (D_V + 2);
        const float lr = plan.rowwise_max_buf[64 + r];
        if (lr > 0.f) {
          CUTE_UNROLL
          for (int j = 0; j < D_V / 256; ++j) {
            const int col = 2 * (idx_in_warpgroup + 128 * j);
            *reinterpret_cast<float2*>(w + col) = *reinterpret_cast<const float2*>(stage + r * SP + col);
          }
        }
        if (idx_in_warpgroup == 0) {
          w[D_V] = plan.rowwise_max_buf[r];
          w[D_V + 1] = lr;
        }
      }
    } else {
    CUTE_UNROLL
    for (int c = 0; c < 2; ++c) {
      CUTE_UNROLL
      for (int k = 0; k < (D_V / 4) / B_EPI; ++k) {
        if (have_valid_indices) {
          ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + c * 128 + k * B_EPI, o);
          cutlass::arch::fence_view_async_tmem_load();
        }
        const int col0 = c * (D_V / 2) + (idx_in_warpgroup / 64) * (D_V / 4) + k * B_EPI;
        CUTE_UNROLL
        for (int i = 0; i < B_EPI / 8; ++i) {
          nv_bfloat162 o_bf16[4];
          CUTE_UNROLL
          for (int j = 0; j < 4; ++j) {
            o[i * 4 + j] = ku::float2_mul(o[i * 4 + j], output_scale_float2);
            o_bf16[j] = __float22bfloat162_rn(o[i * 4 + j]);
          }
          if (store) *(uint128_t*)(orow + col0 + i * 8) = *(uint128_t*)(o_bf16);
        }
      }
    }
    }

    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, 512);
  } else if (warpgroup_idx == 1) {
    // KV producer: one TMA gather4 per pool (its 4 consecutive cache rows), 16 pools per block
    int warp_idx = cutlass::canonical_warp_idx_sync() - 4;
    constexpr int NUM_WARPS = 4, NUM_LOCAL_ROWS_PER_WARP = (B_TOPK / 4) / NUM_WARPS;
    if (elect_one_sync()) {
      CUTE_NO_UNROLL
      for (int k = 0; k < num_k_blocks; ++k) {
        int4 indices[NUM_LOCAL_ROWS_PER_WARP];
        CUTE_UNROLL
        for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
          const int pi = k * (B_TOPK / 4) + local_row * NUM_WARPS + warp_idx;
          const int r0 = pi < npools ? __ldcg(urows + pi) : -1;
          indices[local_row] = r0 >= 0 ? make_int4(r0, r0 + 1, r0 + 2, r0 + 3) : make_int4(-1, -1, -1, -1);
        }
        const bool should_skip_tma = k * (B_TOPK / 4) >= npools && k >= NUM_BUFS;

        if (k == 2) plan.bar_prologue_utccp_nope.wait(0);  // q_nope coincides with k[2]

        int cur_buf = k % NUM_BUFS;
        plan.bar_sv_done[cur_buf].wait((k / NUM_BUFS) & 1 ^ 1);
        bf16* sK_nope_base = plan.u.k.k_nope[cur_buf].data() + warp_idx * 4 * 64;

        // (split-KV decode: each row is read once per layer -> evict first, not displacing the step's
        //  other data from L2; prefill tiles: evict last, as FlashMLA)
        const int64_t kv_hint = params.partial ? (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                                               : (int64_t)TMA::CacheHintSm90::EVICT_LAST;
        auto load_kv_nope_part = [&](int part_idx) {
          CUTE_UNROLL
          for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
            CUTE_UNROLL
            for (int local_col = part_idx * (D_V / 2 / 64); local_col < (part_idx + 1) * (D_V / 2 / 64); ++local_col) {
              ku::tma_gather4(&tm_kv, plan.bar_kv_nope_ready[cur_buf][part_idx],
                              sK_nope_base + local_row * (4 * NUM_WARPS) * 64 + local_col * (B_TOPK * 64),
                              local_col * 64, indices[local_row], kv_hint);
            }
          }
        };
        if (!should_skip_tma) {
          load_kv_nope_part(0);
          load_kv_nope_part(1);
        } else {
          CUTE_UNROLL
          for (int part_idx = 0; part_idx < 2; ++part_idx)
            plan.bar_kv_nope_ready[cur_buf][part_idx].complete_transaction(NUM_LOCAL_ROWS_PER_WARP * 4 * D_V / 2 * sizeof(bf16));
        }
      }
    }
  } else {
    if (warp_idx == 8 && elect_one_sync()) {
      // MMA warp: S -> T copy of Q, then P = Q K^T (dual gemm), O += S V
      UMMA::SmemDescriptor sQ_nope_desc = UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
          make_smem_ptr(plan.u.q_full.q_nope.data()),
          tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_H * 2>, Int<64>>{})));
      plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
      plan.bar_prologue_q_nope.wait(0);
      ku::tcgen05_after_thread_sync();
      CUTE_UNROLL
      for (int tile_idx = 0; tile_idx < D_V / 64 / 2; ++tile_idx) {
        CUTE_UNROLL
        for (int subtile_idx = 0; subtile_idx < 4; ++subtile_idx) {
          SM100_UTCCP_128dp256bit_1cta::copy(sQ_nope_desc + (tile_idx * (B_H * 128 * 2) + subtile_idx * 32) / 16,
                                             tmem_cols::Q + tile_idx * 32 + subtile_idx * 8);
        }
      }
      ku::umma_arrive_noelect(plan.bar_prologue_utccp_nope);

      CUTE_NO_UNROLL
      for (int k = 0; k < num_k_blocks + 1; ++k) {
        if (k < num_k_blocks) {
          int cur_buf = k % NUM_BUFS;
          Tensor sK_nope = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()), SmemLayoutKNoPE_TiledMMA{});
          plan.bar_p_free.wait(k & 1 ^ 1);
          ku::tcgen05_after_thread_sync();
          if (k == 0) plan.bar_prologue_utccp_nope.wait(0);
          Tensor sK_nope_divided = flat_divide(sK_nope, Tile<Int<B_TOPK * 2>, Int<D_V / 4>>{})(_, _, _0{}, _);
          CUTE_UNROLL
          for (int kv_nope_part_idx = 0; kv_nope_part_idx < 2; ++kv_nope_part_idx) {
            plan.bar_kv_nope_ready[cur_buf][kv_nope_part_idx].arrive_and_expect_tx(B_TOPK * D_V / 2 * sizeof(bf16));
            plan.bar_kv_nope_ready[cur_buf][kv_nope_part_idx].wait((k / NUM_BUFS) & 1);
            ku::tcgen05_after_thread_sync();
            ku::utcmma_ts(tiled_mma_P, kv_nope_part_idx ? tQ_nope_part1 : tQ_nope_part0,
                          sK_nope_divided(_, _, kv_nope_part_idx), tP, kv_nope_part_idx == 0);
          }
          ku::umma_arrive_noelect(plan.bar_qk_nope_done[cur_buf]);
        }
        if (k > 0) {
          int cur_buf = (k - 1) % NUM_BUFS;
          Tensor sS = make_tensor(make_smem_ptr(plan.s), SmemLayoutS{});
          Tensor sV = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()), SmemLayoutV{});
          plan.bar_so_ready.wait((k - 1) & 1);
          ku::tcgen05_after_thread_sync();
          ku::utcmma_ss(tiled_mma_O, sS, sV, tO, k == 1);
          ku::umma_arrive_noelect(plan.bar_sv_done[cur_buf]);
        }
      }
    } else if (warp_idx == 9) {
      // validity masks: lane l covers rows [8 l, 8 l + 8) of a block = pools 2 l, 2 l + 1; per token,
      // the rows of the pools it sees up to its position
      if (lane_idx < B_TOPK / 8) {
        int pos[TOK];
        CUTE_UNROLL
        for (int j = 0; j < TOK; ++j) pos[j] = plan.tok_pos[j];
        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
          uint32_t m[TOK];
          CUTE_UNROLL
          for (int j = 0; j < TOK; ++j) m[j] = 0u;
          CUTE_UNROLL
          for (int pp = 0; pp < 2; ++pp) {
            const int pi = k * (B_TOPK / 4) + lane_idx * 2 + pp;
            if (pi < npools) {
              const int base = __ldcg(upools + pi) * 4;
              const uint32_t mem = (uint32_t)__ldcg(reinterpret_cast<const char*>(umember) + pi) & 0xffu;
              CUTE_UNROLL
              for (int j = 0; j < TOK; ++j) {
                const int vis = min(max(pos[j] - base + 1, 0), 4);  // rows of this pool at or before pos
                if ((mem >> j) & 1u) m[j] |= ((1u << vis) - 1u) << (pp * 4);
              }
            }
          }
          int cur_buf = k % NUM_BUFS;
          plan.bar_k_valid_free[cur_buf].wait((k / NUM_BUFS) & 1 ^ 1);
          CUTE_UNROLL
          for (int j = 0; j < TOK; ++j) plan.is_k_valid[cur_buf][j][lane_idx] = (char)m[j];
          plan.bar_k_valid_ready[cur_buf].arrive();
        }
      }
    }
  }
#endif
}

}  // namespace sm100::fwd::head64u

namespace glm {
namespace k {

namespace {
CUtensorMap make_map_rows512(const void* base, int64_t rows, int box_rows) {
  CUtensorMap m;
  uint64_t size[2] = {512, (uint64_t)rows};
  uint64_t stride[1] = {512 * sizeof(cutlass::bfloat16_t)};
  uint32_t box[2] = {64, (uint32_t)box_rows};
  uint32_t es[2] = {1, 1};
  CUresult r = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), size, stride, box, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  KU_ASSERT(r == CUDA_SUCCESS);
  return m;
}
}  // namespace

void fmla_sparse_union_fwd(const FmlaUnionArgs& a) {
  using namespace sm100::fwd::head64u;
  if (a.ntiles <= 0) return;
  KU_ASSERT(a.H >= 8 && 64 % a.H == 0);
  UnionParams p;
  p.tiles = a.tiles;
  p.ws = static_cast<const uint8_t*>(a.ws);
  p.union_bytes = a.union_bytes;
  p.union_cap = a.union_cap;
  p.positions = a.positions;
  p.out = (bf16*)a.out;
  p.sm_scale_div_log2 = a.sm_scale * 1.44269504088896340736f;
  p.partial = a.nsplit > 1 ? a.partial_ws : nullptr;
  p.pre = a.pre ? 1 : 0;
  KU_ASSERT(a.nsplit <= 1 || (a.partial_ws != nullptr && a.H <= 16));
  const CUtensorMap tq = make_map_rows512(a.q, a.q_rows, 64);
  const CUtensorMap tkv = make_map_rows512(a.kv, a.kv_rows, 1);
  constexpr size_t smem = sizeof(SharedMemoryPlanU);
  auto launch = [&](auto kernel) {
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    if (p.partial) {  // (split-KV decode: PDL, the kernel waits before reading anything)
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(a.ntiles, a.nsplit);
      cfg.blockDim = dim3(NUM_THREADS);
      cfg.dynamicSmemBytes = smem;
      cfg.stream = a.stream;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      KU_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p, tq, tkv));
    } else {
      kernel<<<dim3(a.ntiles, 1), NUM_THREADS, smem, a.stream>>>(p, tq, tkv);
    }
    KU_CHECK_KERNEL_LAUNCH();
  };
  switch (64 / a.H) {
    case 1: launch(&sparse_attn_union_kernel<1>); break;
    case 2: launch(&sparse_attn_union_kernel<2>); break;
    case 4: launch(&sparse_attn_union_kernel<4>); break;
    default: launch(&sparse_attn_union_kernel<8>); break;
  }
}

}  // namespace k
}  // namespace glm
