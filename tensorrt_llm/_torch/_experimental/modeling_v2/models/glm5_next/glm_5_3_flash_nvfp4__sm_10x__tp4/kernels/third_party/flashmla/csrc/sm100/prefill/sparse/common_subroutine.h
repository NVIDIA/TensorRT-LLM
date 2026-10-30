#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

namespace sm100 {

/*
Load K/V indices from global memory, and generate validity mask
Each thread loads 8 indices
Should be called by lanes 0 ~ (BLOCK_TOPK/8)
*/
CUTE_DEVICE
char load_indices_and_generate_mask(
    int lane_idx,
    int* gIndices,
    int s_kv,
    int abs_pos_start,
    int topk_length
) {
    int indices[8];
    KU_LDG_256(
        gIndices + lane_idx*8, 
        indices,
        ".nc", 
        "no_allocate", 
        "evict_normal", 
        "256B"
    );
    auto is_valid = [&](int rel_pos_in_lane, int index) -> char {
        int abs_pos = abs_pos_start + lane_idx*8 + rel_pos_in_lane;
        return index >= 0 && index < s_kv && abs_pos < topk_length;
    };
    char is_ks_valid_mask = \
        is_valid(7, indices[7]) << 7 | 
        is_valid(6, indices[6]) << 6 | 
        is_valid(5, indices[5]) << 5 |
        is_valid(4, indices[4]) << 4 |
        is_valid(3, indices[3]) << 3 |
        is_valid(2, indices[2]) << 2 |
        is_valid(1, indices[1]) << 1 |
        is_valid(0, indices[0]) << 0;
    return is_ks_valid_mask;
}


/*
Get P from Tensor Memory, reduce P within shared memory, perform masking, and store back if necessary

Initially, since dual gemm is used, we have two P pieces in Tensor Memory, one occupying rows 0 ~ 63 while the other occupying rows 64 ~ 127. We'd like to have them reduced into one single P piece, stored in registers with layout:

        N       N    ---   (topk)
    +-------+-------+
    |       |       |
32  | Warp0 | Warp2 |
    |       |       |
    +-------+-------+
    |       |       |
32  | Warp1 | Warp3 |
    |       |       |
    +-------+-------+
|
(head)

where N = NUM_ELEMS_PER_THREAD
*/
template<
    int NUM_ELEMS_PER_THREAD,
    int TMEM_COL_START,
    int BARRIER_WARP02_SYNC_ID,
    int BARRIER_WARP13_SYNC_ID,
    bool STORE_BACK_P
>
CUTE_DEVICE
void retrieve_mask_and_reduce_p(
    char* k_validness_base,
    int local_warp_idx,
    int lane_idx,
    auto slot_bar_P_empty_arrival,
    float p_exchange_buf[4][32*NUM_ELEMS_PER_THREAD],
    float p[NUM_ELEMS_PER_THREAD]
) {
    using namespace cute;
    using cutlass::arch::NamedBarrier;
    static_assert(BARRIER_WARP13_SYNC_ID == BARRIER_WARP02_SYNC_ID+1);

    float p_peer[NUM_ELEMS_PER_THREAD];
    if (local_warp_idx < 2) {
        ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(TMEM_COL_START, p);
        ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(TMEM_COL_START + NUM_ELEMS_PER_THREAD, p_peer);
    } else {
        ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(TMEM_COL_START, p_peer);
        ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(TMEM_COL_START + NUM_ELEMS_PER_THREAD, p);
    }
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();
    slot_bar_P_empty_arrival();

    // Mask invalid tokens
    // We put masking before reduction, since (-inf) + anything (except nan and +inf) is (-inf), which guarantees correctness, and this can overlap with smem load
    static_assert(NUM_ELEMS_PER_THREAD == 32);
    uint32_t is_k_valid = *(uint32_t*)(k_validness_base + (local_warp_idx>=2?NUM_ELEMS_PER_THREAD/8:0));
    CUTE_UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD; i += 1) {
        if (!(is_k_valid >> i & 1))
            p[i] = -CUDART_INF_F;
    }

    // Reduce P within the cluster
    {
        // Store
        // Warp 0, 1 store their right (col 32 ~ 63) part, while warp 2, 3 store their left (row 0 ~ 31) part
        CUTE_UNROLL
        for (int i = 0; i < NUM_ELEMS_PER_THREAD/4; ++i) {
            ku::st_shared(&p_exchange_buf[local_warp_idx^2][i*32*4 + lane_idx*4], *(float4*)(p_peer + i*4));
        }
        NamedBarrier::arrive_and_wait(64, BARRIER_WARP02_SYNC_ID + (local_warp_idx&1));
        CUTE_UNROLL
        for (int i = 0; i < NUM_ELEMS_PER_THREAD/4; ++i) {
            float2 t[2];
            *(float4*)t = *(float4*)(&p_exchange_buf[local_warp_idx][i*32*4 + lane_idx*4]);
            float2* cur_p = (float2*)(p + i*4);
            cur_p[0] = ku::float2_add(cur_p[0], t[0]);
            cur_p[1] = ku::float2_add(cur_p[1], t[1]);
        }
    }

    if constexpr (STORE_BACK_P) {
        CUTE_UNROLL
        for (int i = 0; i < NUM_ELEMS_PER_THREAD/4; ++i) {
            ku::st_shared(&p_exchange_buf[local_warp_idx][i*32*4 + lane_idx*4], *(float4*)(p+i*4));
        }
    }
}

/*
Rescale O in Tensor Memory.

O should occupy 128 rows x (D_V/2) columns in Tensor Memory.
*/
template<
    int D_V,
    int CHUNK_SIZE,
    int TMEM_COL_START
>
CUTE_DEVICE
void rescale_O(
    float scale_factor
) {
    float2 scale_factor_float2 = {scale_factor, scale_factor};
    float2 o[CHUNK_SIZE/2];

    CUTE_UNROLL
    for (int chunk_idx = 0; chunk_idx < (D_V/2)/CHUNK_SIZE; ++chunk_idx) {
        // Load O
        ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(TMEM_COL_START + chunk_idx*CHUNK_SIZE, o);
        cutlass::arch::fence_view_async_tmem_load();

        // Mult
        for (int i = 0; i < CHUNK_SIZE/2; ++i) {
            o[i] = ku::float2_mul(o[i], scale_factor_float2);
        }

        // Store O
        ku::tmem_st_32dp32bNx<CHUNK_SIZE>(TMEM_COL_START + chunk_idx*CHUNK_SIZE, o);
        cutlass::arch::fence_view_async_tmem_store();
    }
}

template<int NUM_ELEMS_PER_THREAD>
CUTE_DEVICE
float get_max(
    float p[NUM_ELEMS_PER_THREAD]
) {
    float local_max = -CUDART_INF_F;
    CUTE_UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
        local_max = max(local_max, p[i]);
    }
    return local_max;
}

/*
Calculate s := exp2f(p*scale - new_max) and its sum
*/
template<int NUM_ELEMS_PER_THREAD>
CUTE_DEVICE
float get_s_from_p(
    nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2],
    float p[NUM_ELEMS_PER_THREAD],
    float scale,
    float new_max
) {
    float2 cur_sum = float2 {0.0f, 0.0f};
    float2 neg_new_max_float2 = float2 {-new_max, -new_max};
    float2 scale_float2 = float2 {scale, scale};
    CUTE_UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD/2; i += 1) {
        float2 d = ku::float2_fma(float2{p[i*2], p[i*2+1]}, scale_float2, neg_new_max_float2);
        d.x = exp2f(d.x);
        d.y = exp2f(d.y);
        cur_sum = ku::float2_add(cur_sum, d);
        s[i] = __float22bfloat162_rn(d);
    }
    return cur_sum.x + cur_sum.y;
}

}
