#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"

namespace sm100::fwd::head64 {

using namespace cute;

template<
    typename Shape_Q_NoPE, typename TMA_Q_NoPE,
    typename Shape_Q_RoPE, typename TMA_Q_RoPE,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q_NoPE shape_Q_nope; TMA_Q_NoPE tma_Q_nope;
    Shape_Q_RoPE shape_Q_rope; TMA_Q_RoPE tma_Q_rope;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_kv_nope;
};

struct float2x2 {
    float2 lo, hi;
};

constexpr int D_Q = 576;
constexpr int D_K = 576;
constexpr int D_V = 512;
constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits) to avoid -inf - (-inf) = nan

constexpr int B_H = 64;
constexpr int B_TOPK = 64;
constexpr int NUM_BUFS = 3;
constexpr int NUM_THREADS = 128 + 128 + 128; // 128 scale & exp threads, 128 TMA threads, 32 UTCMMA threads


// Tensor memory columns
namespace tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 400: Q
    // 400 ~ 464: P
    constexpr int O = 0;
    constexpr int Q = 256;
    constexpr int Q_RoPE = 256 + 128;
    constexpr int P = 400;
}

using SmemLayoutQNoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_Q-D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutO = SmemLayoutOTiles<8>;

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKNoPE = SmemLayoutKTiles<8>;
using SmemLayoutV = decltype(coalesce(
    composition(
        SmemLayoutKNoPE{},
        Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
    )
, Shape<_1, _1>{}));

using SmemLayoutKRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<64>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKNoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK*2>, Int<D_V/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));   // Re-view K-NoPE as B_TOPK*2 x D_V/2 for dual gemm

using SmemLayoutKRoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK*2>, Int<64/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutS = decltype(coalesce(tile_to_shape(
	UMMA::Layout_K_INTER_Atom<bf16>{},
	Shape<Int<B_H>, Int<B_TOPK>>{},
	Step<_1, _2>{}
), Shape<_1, _1>{}));


struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> _k_rope_pad;
            array_aligned<bf16, cosize_v<SmemLayoutKNoPE>> _k_pad[2];   // So that q_nope covers k[2]
            array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope;
        } q_full;
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> k_rope;
            array_aligned<bf16, cosize_v<SmemLayoutKNoPE>> k_nope[NUM_BUFS];
        } k;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } u;
    float p_exchange_buf[4][32 * (B_TOPK/2)];
    union {
        bf16 s[B_H*B_TOPK];
        array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;
    } s_q_rope;
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    transac_bar_t bar_prologue_q_nope, bar_prologue_q_rope, bar_prologue_utccp_nope, bar_prologue_utccp_rope;
    transac_bar_t bar_qk_nope_done[NUM_BUFS], bar_qk_rope_done;    // Pi = QKi^T (the nope part) done
    transac_bar_t bar_sv_done[NUM_BUFS];    // O += SiVi done (i.e. O, Si and Vi are free)
    transac_bar_t bar_kv_nope_ready[NUM_BUFS][2], bar_kv_rope_ready;
    transac_bar_t bar_p_free;
    transac_bar_t bar_so_ready;   // S and O are ready
    transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
    array_aligned<uint32_t, 1> tmem_start_addr;
    float rowwise_max_buf[128], rowwise_li_buf[128];
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, B_H, 128, UMMA::Major::K, UMMA::Major::K>{}  // Here we use N = 128 = 2*B_TOPK since we're going to use implicit dual gemm: <TODO Fill link here>
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{}
));

enum NamedBarriers : int {
    wg0_sync = 0,
    wg0_warp02_sync = 1,
    wg0_warp13_sync = 2,
    pepi_sync = 3,
};


}
