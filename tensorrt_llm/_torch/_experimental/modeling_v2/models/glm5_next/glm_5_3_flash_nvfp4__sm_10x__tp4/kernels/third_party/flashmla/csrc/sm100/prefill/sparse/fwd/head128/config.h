#pragma once

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "params.h"
#include "defines.h"

namespace sm100::fwd::head128 {

using namespace cute;

template<
    typename Shape_Q, typename TMA_Q,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_kv;
};

struct float2x2 {
    float2 lo, hi;
};

template<int D_QK>
struct KernelTemplate {

static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits) to avoid -inf - (-inf) = nan

static constexpr int B_H = 128;    // For 2 CTAs
static constexpr int B_TOPK = 128; // For 2 CTAs
static constexpr int NUM_BUFS = 2;
static constexpr int NUM_THREADS = 256 + 128 + 128; // 128 scale & exp threads, 128x2 TMA threads, 32 UTCMMA threads


static constexpr int D_tQ = 384, NUM_tQ_TILES = D_tQ / 64;
static constexpr int D_sQ = D_QK-D_tQ, NUM_sQ_TILES = D_sQ / 64;
static_assert(D_sQ%64 == 0 && D_tQ%64 == 0 && D_sQ + D_tQ == D_Q);

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 320: P
    // 320 ~ 512: Q[D_QK-D_tQ:]
    static constexpr int o = 0;
    static constexpr int p = 256;
    static constexpr int q = 512 - D_tQ/2;
    static_assert(p+64 <= q);
};

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutO = SmemLayoutOTiles<8>;

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutV = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<256>, Int<B_TOPK>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutSTiles = decltype(coalesce(tile_to_shape(
	UMMA::Layout_K_INTER_Atom<bf16>{},
	Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
	Step<_1, _2>{}
), Shape<_1, _1>{}));

struct SharedMemoryPlan {
    union {
        array_aligned<bf16, cosize_v<SmemLayoutQTiles<D_Q/64>>> q_full;
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutQTiles<NUM_sQ_TILES>>> sq;
            array_aligned<bf16, cosize_v<SmemLayoutV>> v;
            // NOTE K is not overlapped with q_full, so we can do k copy-in while performing S->T copy for q
            static_assert(cosize_v<SmemLayoutQTiles<D_Q/64>> <= cosize_v<SmemLayoutQTiles<NUM_sQ_TILES>> + cosize_v<SmemLayoutV>);
            array_aligned<bf16, cosize_v<SmemLayoutKTiles<D_K/64>>> k;
        } s;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } u;
    array_aligned<bf16, cosize_v<SmemLayoutSTiles<2>>> s;
    float p[(B_H/2)*B_TOPK];
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    transac_bar_t bar_prologue_q, bar_prologue_utccp;
    transac_bar_t bar_qk_part_done[NUM_BUFS], bar_qk_done[NUM_BUFS];    // Pi = QKi^T done (i.e. Ki free)
    transac_bar_t bar_sv_part_done[NUM_BUFS], bar_sv_done[NUM_BUFS];    // O += SiVi done (i.e. Vi free)
    transac_bar_t bar_k_part0_ready[NUM_BUFS], bar_k_part1_ready[NUM_BUFS];
    transac_bar_t bar_v_part0_ready[NUM_BUFS], bar_v_part1_ready[NUM_BUFS];    // Vi is ready
    transac_bar_t bar_p_free[NUM_BUFS];
    transac_bar_t bar_so_ready[NUM_BUFS];   // S and O are ready
    transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
    array_aligned<uint32_t, 1> tmem_start_addr;
    float rowwise_max_buf[128], rowwise_li_buf[128];
};

using TiledMMA_P_tQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_P_sQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}  // We use this permutation layout to let CTA0 takes V[:, 0:256] and CTA1 takes V[:, 256:512]
));

template<typename TmaParams>
static __device__ void
sparse_attn_fwd_kernel_devfunc(const SparseAttnFwdParams &params, const TmaParams &tma_params);

};

}
