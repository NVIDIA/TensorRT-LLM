#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include <kerutils/kerutils.cuh>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"
#include "sm100/prefill/sparse/common_subroutine.h"
#include "config.h"

namespace sm100::fwd::head64 {

using namespace cute;

/*
Pipeline Overview:

| Copy |    MMA    |   Scale & Exp   |

KV0
KV1
KV2
        P0 = QK0^T
                    S0 = exp(P0)
                    scale(O) w.r.t P0
        P1 = QK1^T
                    S1 = exp(P1)
        O += S0V0
KV3                 scale(O) w.r.t P1
        P2 = QK2^T
                    S2 = exp(P2)
        O += S1V1
KV4                 scale(O) w.r.t P2
        P3 = QK3^T
                    S3 = exp(P3)
        O += S2V2
KV5                 scale(O) w.r.t P3

...

        O += S(n-3)V(n-3)
                    scale(O) w.r.t P(n-2)
        P(n-1) = QK(n-1)^T
                   S(n-1) = exp(P(n-1))
        O += S(n-2)V(n-2)
                   scale(O) w.r.t P(n-1)
        O += S(n-1)V(n-1)
*/

using FwdMode = SparseAttnFwdMode;

template<bool HAVE_ROPE, typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params, __grid_constant__ const TmaParams tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    // Grid shape: [s_q, 1, 1]

    const int s_q_idx = blockIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);  // num_k_blocks always >= 1

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q; // [topk]

    // Allocate tmem tensors
    TiledMMA tiled_mma_P = TiledMMA_P{};
    TiledMMA tiled_mma_O = TiledMMA_O{};
    // NOTE These tXXX tensors are only for a forged layout (so that CuTe is able to generate correct address in cute::gemm)
    Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
    Tensor tQ_nope_part0 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<(D_V/2)/2>>{})
    );
    Tensor tQ_nope_part1 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<(D_V/2)/2>>{})
    );
    Tensor tQ_rope = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<64/2>>{})
    );
    Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<B_H>, Int<D_V>>{});
    tP.data().get() = tmem_cols::P;
    tQ_nope_part0.data().get() = tmem_cols::Q;
    tQ_nope_part1.data().get() = tmem_cols::Q + 64;
    tQ_rope.data().get() = tmem_cols::Q_RoPE;
    tO.data().get() = tmem_cols::O;

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            // Copy Q
            if constexpr (HAVE_ROPE) {
                cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
            }
            cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());

            plan.bar_prologue_q_nope.init(1);
            plan.bar_prologue_q_rope.init(1);
            fence_barrier_init();
            
            if constexpr (HAVE_ROPE) {
                Tensor gQ_rope = tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx);
                Tensor sQ_rope = make_tensor(make_smem_ptr(plan.s_q_rope.q_rope.data()), SmemLayoutQRoPE{});
                ku::launch_tma_copy(tma_params.tma_Q_rope, gQ_rope, sQ_rope, plan.bar_prologue_q_rope, TMA::CacheHintSm90::EVICT_FIRST);
            }

            Tensor gQ_nope = tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, s_q_idx);
            Tensor sQ_nope = make_tensor(make_smem_ptr(plan.u.q_full.q_nope.data()), SmemLayoutQNoPE{});
            ku::launch_tma_copy(tma_params.tma_Q_nope, gQ_nope, sQ_nope, plan.bar_prologue_q_nope, TMA::CacheHintSm90::EVICT_FIRST);

            cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
            cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_nope));
            
            // Initialize other barriers
            plan.bar_prologue_utccp_rope.init(1);
            plan.bar_prologue_utccp_nope.init(1);
            CUTE_UNROLL
            for (int i = 0; i < NUM_BUFS; ++i) {
                plan.bar_qk_nope_done[i].init(1);
                plan.bar_sv_done[i].init(1);
                plan.bar_kv_nope_ready[i][0].init(1);
                plan.bar_kv_nope_ready[i][1].init(1);
                plan.bar_k_valid_ready[i].init(B_TOPK/8);
                plan.bar_k_valid_free[i].init(128);
            }
            plan.bar_p_free.init(128);
            plan.bar_so_ready.init(128);
            plan.bar_qk_rope_done.init(1);
            plan.bar_kv_rope_ready.init(64);
            fence_barrier_init();
        }

        // Initialize TMEM
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    __syncthreads();

    if (warpgroup_idx == 0) {
        // Scale & Exp warps

        // The following three numbers are 
        // - mi: max_logits used to scale Pi (i.e. O := exp2(Pi*scale - mi) @ V)
        // - li: sumexp, i.e. li := sum(exp(Pi*scale - mi))
        // - real_mi: real max logits, i.e. real_mi := max(Pi*scale)
        // where Pi is the i-th row of P, P := QK^T
        // mi and real_mi are always consistent within the two threads that
        // controls one row (i.e. thread 0+64, 1+65, 2+66, ...) after every update
        float mi = MAX_INIT_VAL;
        float li = 0.0f;
        float real_mi = -CUDART_INF_F;

        bf16* sS_base = plan.s_q_rope.s + lane_idx*8 + (warp_idx&1)*(B_H/2)*8 + (warp_idx/2)*B_H*(B_TOPK/2);
        static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            // Wait for P
            NamedBarrier::arrive_and_wait(64, NamedBarriers::wg0_warp02_sync+(warp_idx&1));
            plan.bar_qk_nope_done[k%NUM_BUFS].wait((k/NUM_BUFS)&1);
            plan.bar_k_valid_ready[k%NUM_BUFS].wait((k/NUM_BUFS)&1);    // Put the barrier wait here for more code reordering space
            ku::tcgen05_after_thread_sync();
            
            // Load P
            float p[NUM_ELEMS_PER_THREAD];
            retrieve_mask_and_reduce_p<
                NUM_ELEMS_PER_THREAD,
                tmem_cols::P,
                NamedBarriers::wg0_warp02_sync,
                NamedBarriers::wg0_warp13_sync,
                false
            >(
                plan.is_k_valid[k%NUM_BUFS],
                warp_idx, lane_idx, 
                [&]() {plan.bar_p_free.arrive();},
                plan.p_exchange_buf,
                p
            );
            plan.bar_k_valid_free[k%NUM_BUFS].arrive();
            
            // Get rowwise max of Pi
            float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
            cur_pi_max *= params.sm_scale_div_log2;

            plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup^64]);
            real_mi = max(real_mi, cur_pi_max);
            bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);
            // By this point:
            // - cur_pi_max, real_mi, and mi is identical within each row (i.e. thread 0+64, 1+65, ...)
            // - should_scale_o is identical among every warp, and is identical among threads that controls the same row (i.e. among threads 0~31+64~95; and is identical among threads 32~63+96~127)


            // Calc scale factor, and scale li
            float new_max, scale_for_old;
            if (!should_scale_o) {
                // Don't scale O
                scale_for_old = 1.0f;
                new_max = mi;
            } else {
                new_max = max(cur_pi_max, mi);
                scale_for_old = exp2f(mi - new_max);
            }
            mi = new_max;   // mi is still identical within each row

            // Calculate S
            nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2];
            float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>(s, p, params.sm_scale_div_log2, new_max);
            li = fma(li, scale_for_old, cur_sum);

            // Wait for last SV gemm, write S
            if (k > 0) {
                plan.bar_sv_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);
            }
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; i += 1) {
                *(uint128_t*)(sS_base + B_H*8*i) = *(uint128_t*)(s + i*4);
            }

            // Scale O
            if (k > 0 && should_scale_o) {
                // plan.bar_sv_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);   // NOTE We have waited for last SV gemm before
                ku::tcgen05_after_thread_sync();
                rescale_O<D_V, 32, tmem_cols::O>(scale_for_old);
                ku::tcgen05_before_thread_sync();
            }
            
            fence_view_async_shared();
            plan.bar_so_ready.arrive();
        }

        // Epilogue

        if (real_mi == -CUDART_INF_F) {
            // real_mi == -CUDART_INF_F <=> No valid TopK indices
            // We set li to 0 to fit the definition that li := exp(x[i] - mi)
            li = 0.0f;
            mi = -CUDART_INF_F;
        }
        
        // Exchange li
        plan.rowwise_li_buf[idx_in_warpgroup] = li;
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
        li += plan.rowwise_li_buf[idx_in_warpgroup^64];

        // Store mi and li
        if (idx_in_warpgroup < 64) {
            int global_index = s_q_idx*params.h_q + idx_in_warpgroup;
            float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
            cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
            params.max_logits[global_index] = real_mi*CUDART_LN2_F;
            params.lse[global_index] = cur_lse;
        }

        // Wait for the last GEMM
        plan.bar_sv_done[(num_k_blocks-1)%NUM_BUFS].wait(((num_k_blocks-1)/NUM_BUFS)&1);
        ku::tcgen05_after_thread_sync();

        // Fetch dO if necessary

        // Store O
        float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + (idx_in_warpgroup%64))*CUDART_L2E_F;
        float output_scale = __fdividef(1.0f, li + exp2f(attn_sink - mi));
        Tensor sO = make_tensor(make_smem_ptr(plan.u.o.data()), SmemLayoutO{});
        constexpr int B_EPI = 64;
        Tensor tma_gO = flat_divide(
            tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx),
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        Tensor sO_divided = flat_divide(
            sO,
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        auto thr_tma = tma_params.tma_O.get_slice(_0{});

        float2 o[B_EPI/2];
        bool have_valid_indices = __any_sync(0xffffffff, li != 0);  // Prevent some threads' li == 0 and some threads' li != 0 which lead to deadlock during ku::tmem_ld
        if (!have_valid_indices) {
            // If there are no valid indices, we set o[i] to 0 and don't load from TMEM
            CUTE_UNROLL
            for (int i = 0; i < B_EPI/2; ++i)
                o[i].x = o[i].y = 0.0f;
            output_scale = 1.0f;
        }

        float2 output_scale_float2 = make_float2(output_scale, output_scale);

        bf16* sO_addrs[8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI/8; ++i) {
            sO_addrs[i] = &sO(idx_in_warpgroup%64, i*8);
        }

        CUTE_UNROLL
        for (int c = 0; c < 2; ++c) {
            // Each tile: 64 x 256
            CUTE_UNROLL
            for (int k = 0; k < (D_V/4)/B_EPI; ++k) {
                // Load O from tO
                if (have_valid_indices) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + c*128 + k*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                }

                // Convert and store
                CUTE_UNROLL
                for (int i = 0; i < B_EPI/8; ++i) {
                    nv_bfloat162 o_bf16[4];
                    CUTE_UNROLL
                    for (int j = 0; j < 4; ++j) {
                        o[i*4+j] = ku::float2_mul(o[i*4+j], output_scale_float2);
                        o_bf16[j] = __float22bfloat162_rn(o[i*4+j]);
                    }
                    *(uint128_t*)(sO_addrs[i] + (c*(D_V/2) + (idx_in_warpgroup/64)*(D_V/4) + k*B_EPI)*64) = *(uint128_t*)(o_bf16);
                }

                // Sync
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                
                if (warp_idx == 0 && elect_one_sync()) {
                    int epi_chunk_idx = c*(D_V/2/B_EPI) + k;
                    cute::copy(
                        tma_params.tma_O,
                        thr_tma.partition_S(sO_divided(_, _, epi_chunk_idx)),
                        thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                    );
                }
                if (warp_idx == 1 && elect_one_sync()) {
                    int epi_chunk_idx = c*(D_V/2/B_EPI) + (D_V/B_EPI/4) + k;
                    cute::copy(
                        tma_params.tma_O,
                        thr_tma.partition_S(sO_divided(_, _, epi_chunk_idx)),
                        thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                    );
                }
            }
        }


        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        // Producer warp for KV
        int warp_idx = cutlass::canonical_warp_idx_sync() - 4;
        constexpr int NUM_WARPS = 4, NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/4)/NUM_WARPS;
        if (elect_one_sync()) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                int4 indices[NUM_LOCAL_ROWS_PER_WARP];
                int max_indices = -1, min_indices = params.s_kv;
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices[local_row] = __ldg((int4*)(gIndices + k*B_TOPK) + local_row*NUM_WARPS + warp_idx);
                    max_indices = max(max_indices, int4_max(indices[local_row]));
                    min_indices = min(min_indices, int4_min(indices[local_row]));
                }
                bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                bool should_skip_tma = is_all_rows_invalid && k >= NUM_BUFS;

                if (k == 2) {
                    plan.bar_prologue_utccp_nope.wait(0);   // Since q_nope coincidences with k[2]
                }

                // Copy NoPE
                int cur_buf = k%NUM_BUFS;
                plan.bar_sv_done[cur_buf].wait((k/NUM_BUFS)&1^1);
                bf16* sK_nope_base = plan.u.k.k_nope[cur_buf].data() + warp_idx*4*64;

                auto load_kv_nope_part = [&](int part_idx) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        CUTE_UNROLL
                        for (int local_col = part_idx*(D_V/2/64); local_col < (part_idx+1)*(D_V/2/64); ++local_col) {
                            ku::tma_gather4(
                                &(tma_params.tensor_map_kv_nope),
                                plan.bar_kv_nope_ready[cur_buf][part_idx],
                                sK_nope_base + local_row*(4*NUM_WARPS)*64 + local_col*(B_TOPK*64),
                                local_col*64,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                    }
                };

                if (!should_skip_tma) {
                    load_kv_nope_part(0);
                    load_kv_nope_part(1);
                } else {
                    // NOTE See head128/phase1.cuh for this TMA skipping technique
                    CUTE_UNROLL
                    for (int part_idx = 0; part_idx < 2; ++part_idx)
                        plan.bar_kv_nope_ready[cur_buf][part_idx].complete_transaction(NUM_LOCAL_ROWS_PER_WARP*4*D_V/2*sizeof(bf16));
                }
            }
        }
    } else {
        // MMA warp
        if (warp_idx == 8 && elect_one_sync()) {
            // S -> T copy for Q
            UMMA::SmemDescriptor sQ_nope_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                make_tensor(
                    make_smem_ptr(plan.u.q_full.q_nope.data()),
                    tile_to_shape(
                        UMMA::Layout_K_SW128_Atom<bf16>{},
                        Shape<Int<B_H*2>, Int<64>>{}    // We use this shape for dual gemm (TODO Link)
                    )
                )
            );
            UMMA::SmemDescriptor sQ_rope_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                make_tensor(
                    make_smem_ptr(plan.s_q_rope.q_rope.data()),
                    tile_to_shape(
                        UMMA::Layout_K_SW64_Atom<bf16>{},
                        Shape<Int<B_H*2>, Int<32>>{}
                    )
                )
            );
            
            if constexpr (HAVE_ROPE) {
                // Copy the RoPE tile: 128 rows * 32 cols (64B) (in UTCCP's view), or 64 rows * 64 cols (in our view)
                plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H*(D_Q-D_V)*sizeof(bf16));
                plan.bar_prologue_q_rope.wait(0);
                ku::tcgen05_after_thread_sync();
                CUTE_UNROLL
                for (int subtile_idx = 0; subtile_idx < 2; ++subtile_idx) {
                    // A subtile is 128 rows * 16 cols (256b, 32B) (in UTCCP's view), or 64 rows * 16 cols * 2 (in our view)
                    SM100_UTCCP_128dp256bit_1cta::copy(
                        sQ_rope_desc + (subtile_idx*32) / 16,
                        tmem_cols::Q_RoPE + subtile_idx*8
                    );
                }
                ku::umma_arrive_noelect(plan.bar_prologue_utccp_rope);
            }

            plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H*D_V*sizeof(bf16));
            plan.bar_prologue_q_nope.wait(0);
            ku::tcgen05_after_thread_sync();
            CUTE_UNROLL
            for (int tile_idx = 0; tile_idx < D_V/64/2; ++tile_idx) {
                // A tile is 128 rows * 64 cols (128B) (in UTCCP's view), or 64 rows * 128 cols (in our view)
                CUTE_UNROLL
                for (int subtile_idx = 0; subtile_idx < 4; ++subtile_idx) {
                    // A subtile is 128 rows * 16 cols (256b, 32B) (in UTCCP's view), or 64 rows * 16 cols * 2 (in our view)
                    SM100_UTCCP_128dp256bit_1cta::copy(
                        sQ_nope_desc + (tile_idx*(B_H*128*2) + subtile_idx*32) / 16,   // Remember that 4 LSBs are not included
                        tmem_cols::Q + tile_idx*32 + subtile_idx*8
                    );
                }
            }
            ku::umma_arrive_noelect(plan.bar_prologue_utccp_nope);

            if constexpr (HAVE_ROPE) {
                plan.bar_prologue_utccp_rope.wait(0);
            }

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks+1; ++k) {
                if (k < num_k_blocks) {
                    // Pi = QKi^T
                    int cur_buf = k%NUM_BUFS;
                    Tensor sK_nope = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()), SmemLayoutKNoPE_TiledMMA{});
                    Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()), SmemLayoutKRoPE_TiledMMA{});

                    plan.bar_p_free.wait(k&1^1);
                    ku::tcgen05_after_thread_sync();
                    
                    // Wait for K (RoPE)
                    // P = Q(rope) @ K(rope)^T
                    if constexpr (HAVE_ROPE) {
                        plan.bar_kv_rope_ready.wait(k&1);
                        ku::tcgen05_after_thread_sync();
                        ku::utcmma_ts(tiled_mma_P, tQ_rope, sK_rope, tP, true);
                        ku::umma_arrive_noelect(plan.bar_qk_rope_done);
                    }

                    // Wait for K (NoPE)
                    if (k == 0) {
                        plan.bar_prologue_utccp_nope.wait(0);
                    }
                    Tensor sK_nope_divided = flat_divide(sK_nope, Tile<Int<B_TOPK*2>, Int<D_V/4>>{})(_, _, _0{}, _);
                    CUTE_UNROLL
                    for (int kv_nope_part_idx = 0; kv_nope_part_idx < 2; ++kv_nope_part_idx) {
                        plan.bar_kv_nope_ready[cur_buf][kv_nope_part_idx].arrive_and_expect_tx(B_TOPK*D_V/2*sizeof(bf16));
                        plan.bar_kv_nope_ready[cur_buf][kv_nope_part_idx].wait((k/NUM_BUFS)&1);
                        ku::tcgen05_after_thread_sync();

                        // P += Q(nope) @ K(nope)^T
                        bool clear_accum = (!HAVE_ROPE) && kv_nope_part_idx == 0;
                        ku::utcmma_ts(tiled_mma_P, kv_nope_part_idx ? tQ_nope_part1 : tQ_nope_part0, sK_nope_divided(_, _, kv_nope_part_idx), tP, clear_accum);
                    }
                    ku::umma_arrive_noelect(plan.bar_qk_nope_done[cur_buf]);
                }
                if (k > 0) {
                    // O += S(i-1)V(i-1)
                    int cur_buf = (k-1)%NUM_BUFS;

                    Tensor sS = make_tensor(make_smem_ptr(plan.s_q_rope.s), SmemLayoutS{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()), SmemLayoutV{});

                    // Wait for S(i-1) and O to be scaled
                    plan.bar_so_ready.wait((k-1)&1);
                    ku::tcgen05_after_thread_sync();

                    // O += sS @ sV
                    ku::utcmma_ss(tiled_mma_O, sS, sV, tO, k == 1);
                    ku::umma_arrive_noelect(plan.bar_sv_done[cur_buf]);
                }
            }
        } else if (warp_idx == 9) {
            // KV valid loading warp
            if (lane_idx < B_TOPK/8) {
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    char k_validness_mask = load_indices_and_generate_mask(
                        lane_idx,
                        gIndices + k*B_TOPK,
                        params.s_kv,
                        k*B_TOPK,
                        topk_length
                    );

                    int cur_buf = k%NUM_BUFS;
                    plan.bar_k_valid_free[cur_buf].wait((k/NUM_BUFS)&1^1);
                    plan.is_k_valid[cur_buf][lane_idx] = k_validness_mask;
                    plan.bar_k_valid_ready[cur_buf].arrive();
                }
            }
        } else if (warp_idx == 10 || warp_idx == 11) {
            if constexpr (HAVE_ROPE) {
                int thread_idx = threadIdx.x - 10*32;
                constexpr int GROUP_SIZE = 8, NUM_GROUPS = 64/GROUP_SIZE, ROWS_PER_THREAD = B_TOPK/NUM_GROUPS;
                int group_idx = thread_idx / GROUP_SIZE, idx_in_group = thread_idx % GROUP_SIZE;
                Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()), SmemLayoutKRoPE{});
                bf16* sK_rope_base = &sK_rope(group_idx, idx_in_group*8);
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    int indices[ROWS_PER_THREAD];
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < ROWS_PER_THREAD; ++local_row)
                        indices[local_row] = __ldg(gIndices + k*B_TOPK + group_idx + local_row*NUM_GROUPS);
                    plan.bar_qk_rope_done.wait(k&1^1);
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < ROWS_PER_THREAD; ++local_row) {
                        int index = indices[local_row];
                        ku::cp_async_cacheglobal<ku::PrefetchSize::B128>(
                            params.kv + (int64_t)index*params.stride_kv_s_kv + 512 + idx_in_group*8,
                            sK_rope_base + local_row*NUM_GROUPS*32,
                            index >= 0 && index < params.s_kv
                        );  // NOTE Using cp.async instead of TMA is faster here
                        // NOTE Here we only consider the range of `index` instead of also checking against topk_length, as it's noted that under this scenario (i.e. there exists a valid index among indices[topk_length: ] that points to a token who has NaN inside)
                    }
                    cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)&(plan.bar_kv_rope_ready));
                }
            }
        }
    }


#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

template<int D_QK>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % B_TOPK == 0);   // To save some boundry checkings
    KU_ASSERT(params.h_q == B_H);  // To save some calculation
    KU_ASSERT(params.d_qk == D_QK);
    static_assert(D_QK == 576 || D_QK == 512);

    auto shape_Q_nope = make_shape(params.h_q, D_V, params.s_q);
    auto tma_Q_nope = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q_nope,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQNoPE{}
    );

    auto shape_Q_rope = make_shape(params.h_q, D_Q-D_V, params.s_q);
    auto tma_Q_rope = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q + D_V),
            make_layout(
                shape_Q_rope,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQRoPE{}
    );

    auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((bf16*)params.out),
            make_layout(
                shape_O,
                make_stride(params.d_v, _1{}, params.h_q*params.d_v)
            )
        ),
        SmemLayoutOTiles<1>{}
    );


    CUtensorMap tensor_map_kv_nope;
    {
        uint64_t size[2] = {D_V, (unsigned long)params.s_kv};
        uint64_t stride[1] = {params.stride_kv_s_kv*sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv_nope,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            params.kv,
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    TmaParams<
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_O), decltype(tma_O)
    > tma_params = {
        shape_Q_nope, tma_Q_nope,
        shape_Q_rope, tma_Q_rope,
        shape_O, tma_O,
        tensor_map_kv_nope
    };
    auto kernel = &sparse_attn_fwd_kernel<D_QK == 576, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

}
