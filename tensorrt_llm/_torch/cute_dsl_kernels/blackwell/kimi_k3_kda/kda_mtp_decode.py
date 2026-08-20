# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CuTe DSL device kernel for KDA multi-token speculative verify (conv-MTP).

Source-integrated from the ``KDA_decode_mtp`` kernel drop ("readable KDA
conv-MTP"). Specialized for the benchmarked contract: conv enabled, no bias,
no output norm, lower_bound gate, Q/K L2 norm, beta sigmoid, TILE_V=64,
ILP=2, W=4, K == V == 128, HV == H.

Per request ``n`` the kernel processes ``num_accepted_tokens[n] + 1 +
NUM_SPEC`` steps: it first *replays* the accepted draft tokens from the
``qkg/v/beta`` caches (raw conv inputs from the extended tail columns of
``cs_q/cs_k/cs_v``), then processes the ``1 + NUM_SPEC`` new tokens. The
recurrent state and the base conv windows are committed in place after the
first new (golden) token; the new spec tokens are cached for the next
round's replay. The pool invariant is therefore "state after the last
golden token; accepted drafts pending in the replay caches".

Vendoring delta vs the drop: the ``v_row_a``/``v_row_b`` readout indices in
the output stage are hoisted above the ``if/elif`` on ``USE_ZERO_ACCEPTED``
/ ``i_t >= commit_len``. In the drop they were first assigned inside the
*dynamic* ``elif`` branch, which the CuTe DSL cannot trace (names must
pre-exist before dynamic assignment), so every non-``USE_ZERO_ACCEPTED``
compile — i.e. any replay round, and any non-benchmark shape — failed with
``DSLRuntimeError: v_row_a is None``. The hoist is semantics-preserving.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Int64
from cutlass.cutlass_dsl import T, dsl_user_op

NUM_THREADS = 256
TILE_K = 128


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None) -> Int64:
    """Read the SM global timer for optional in-kernel stage profiling."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.kernel
def kda_decode_mtp_kernel(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    o: cute.Tensor,
    ht: cute.Tensor,
    qkg_cache: cute.Tensor,
    v_cache: cute.Tensor,
    beta_cache: cute.Tensor,
    smem_qk_layout: cute.Layout,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    num_accepted_tokens: cute.Tensor,
    precompute_control: cute.Tensor,
    TILE_V: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    KERNEL_WIDTH: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
    USE_SETMAXREG: cutlass.Constexpr[bool],
    USE_REGULAR_METADATA: cutlass.Constexpr[bool],
    USE_REG_Q_WEIGHTS: cutlass.Constexpr[bool],
    USE_ZERO_ACCEPTED: cutlass.Constexpr[bool],
    FUSE_PRECOMPUTE: cutlass.Constexpr[bool],
    RUNTIME_PRECOMPUTE_FLAG: cutlass.Constexpr[bool],
    stage_timing: cute.Tensor,
    PROFILE_STAGES: cutlass.Constexpr[bool],
):
    """KDA MTP decode — SMEM pre-compute + register-resident state.

    With ``PROFILE_STAGES=True``, ``stage_timing`` must be an int64 tensor
    with at least ``HV * N * 4`` elements, indexed as
    ``(i_hv * grid_n + i_n) * 4``. With profiling off it is never accessed
    and the host may pass any placeholder tensor.
    """
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    i_hv, i_n, _ = cute.arch.block_idx()
    i_h = i_hv
    if cutlass.const_expr(PROFILE_STAGES):
        t_stage0 = read_globaltimer()
        # Pre-declare so the dynamic `run_precompute` branch below only
        # reassigns (first assignment inside a dynamic branch is untraceable;
        # see the module docstring on v_row_a/v_row_b).
        t_stage1 = Int64(0)
    if cutlass.const_expr(USE_REGULAR_METADATA):
        bos = i_n * (2 * NUM_SPEC + 1)
        eos = bos + (2 * NUM_SPEC + 1)
        slot = i_n
    else:
        bos = cu_seqlens[i_n]
        eos = cu_seqlens[i_n + 1]
        slot = ssm_state_indices[i_n]
    h0_idx = slot * HV + i_hv
    hk_off = i_h * K
    hv_off = i_hv * V
    if cutlass.const_expr(USE_ZERO_ACCEPTED):
        commit_len = 0
    else:
        commit_len = num_accepted_tokens[i_n]
        # Only NUM_SPEC drafts can be pending from the previous round. Clamp
        # so a malformed count cannot drive T_loop past the t_max-sized SMEM
        # buffers or the num_spec extents of the replay caches.
        if commit_len > NUM_SPEC:
            commit_len = cutlass.Int32(NUM_SPEC)
    if cutlass.const_expr(USE_ZERO_ACCEPTED):
        T_loop = 1 + NUM_SPEC
        t_max = 1 + NUM_SPEC
    else:
        T_loop = commit_len + 1 + NUM_SPEC
        t_max = 2 * NUM_SPEC + 1
    vec_size = TILE_K // 32
    num_v_tiles = V // TILE_V
    NUM_V_ROWS = TILE_V // (NUM_THREADS // 32)
    if cutlass.const_expr(USE_REG_Q_WEIGHTS):
        q_weight_elems = 0
    else:
        q_weight_elems = KERNEL_WIDTH * K
    v_weight_elems = KERNEL_WIDTH * V
    k_weight_base = q_weight_elems
    v_weight_base = q_weight_elems + KERNEL_WIDTH * K
    conv_weight_elems = q_weight_elems + KERNEL_WIDTH * K + v_weight_elems
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max,)), 16)
    # Reserve the 8-float scratch region formerly used by the removed
    # output-norm reduction. Nothing reads or writes this allocation; it only
    # preserves the offsets and bank mapping of the shared-memory buffers below.
    smem.allocate_tensor(cutlass.Float32, cute.make_layout((8,)), 16)
    sVall = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max * V,)), 16)
    sConvW = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((conv_weight_elems,)),
        16,
    )
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_decay = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_bk = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_state = cute.make_rmem_tensor(
        cute.make_layout((NUM_V_ROWS * vec_size,), stride=(1,)), cutlass.Float32
    )
    if cutlass.const_expr(USE_REG_Q_WEIGHTS):
        r_wq = cute.make_rmem_tensor(
            cute.make_layout((KERNEL_WIDTH * vec_size,), stride=(1,)), cutlass.Float32
        )
    r_exp_A = cutlass.Float32(0.0)
    run_precompute = False
    if cutlass.const_expr(USE_REGULAR_METADATA) or eos > bos:
        if cutlass.const_expr(FUSE_PRECOMPUTE or RUNTIME_PRECOMPUTE_FLAG):
            if cutlass.const_expr(RUNTIME_PRECOMPUTE_FLAG):
                run_precompute = precompute_control[0] != 0
            else:
                run_precompute = True
            if run_precompute:
                for i in range(vec_size):
                    k_idx = i * 32 + in_warp_tid
                    for w in range(KERNEL_WIDTH - 1):
                        r_state[w * vec_size + i] = cutlass.Float32(cs_q[slot, hk_off + k_idx, w])
                        r_state[(KERNEL_WIDTH - 1) * vec_size + w * vec_size + i] = cutlass.Float32(
                            cs_k[slot, hk_off + k_idx, w]
                        )
                for w in range(KERNEL_WIDTH):
                    if tidx < K:
                        if cutlass.const_expr(not USE_REG_Q_WEIGHTS):
                            sConvW[w * K + tidx] = cutlass.Float32(w_q[hk_off + tidx, w])
                        sConvW[k_weight_base + w * K + tidx] = cutlass.Float32(
                            w_k[hk_off + tidx, w]
                        )
                for ld in range(V * KERNEL_WIDTH // NUM_THREADS):
                    flat = ld * NUM_THREADS + tidx
                    sConvW[v_weight_base + flat] = cutlass.Float32(
                        w_v[hv_off + flat % V, flat // V]
                    )
                if cutlass.const_expr(USE_REG_Q_WEIGHTS):
                    if warp_idx == 0:
                        for _w in range(KERNEL_WIDTH):
                            for _i in range(vec_size):
                                r_wq[_w * vec_size + _i] = cutlass.Float32(
                                    w_q[hk_off + _i * 32 + in_warp_tid, _w]
                                )
                cute.arch.barrier()
                if cutlass.const_expr(USE_SETMAXREG):
                    cute.arch.warpgroup_reg_dealloc(64)
                if warp_idx < 3:
                    if warp_idx == 2:
                        r_exp_A = cute.math.exp(cutlass.Float32(A_log[i_h]), fastmath=True)
                    i_t = 0
                    while i_t < T_loop:
                        if cutlass.const_expr(USE_ZERO_ACCEPTED):
                            replay_from_cache = False
                        else:
                            replay_from_cache = i_t < commit_len
                        if replay_from_cache:
                            if warp_idx == 0:
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    sQ[i_t, k_idx] = cutlass.Float32(
                                        qkg_cache[slot, i_t, 0, hk_off + k_idx]
                                    )
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    r_xq_raw = cutlass.Float32(
                                        cs_q[slot, hk_off + k_idx, KERNEL_WIDTH - 1 + i_t]
                                    )
                                    for w in range(KERNEL_WIDTH - 2):
                                        r_state[w * vec_size + i] = r_state[(w + 1) * vec_size + i]
                                    r_state[(KERNEL_WIDTH - 2) * vec_size + i] = r_xq_raw
                            elif warp_idx == 1:
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    sK[i_t, k_idx] = cutlass.Float32(
                                        qkg_cache[slot, i_t, 1, hk_off + k_idx]
                                    )
                                if in_warp_tid == 0:
                                    sBeta[i_t] = cutlass.Float32(beta_cache[slot, i_t, i_hv])
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    r_xk_raw = cutlass.Float32(
                                        cs_k[slot, hk_off + k_idx, KERNEL_WIDTH - 1 + i_t]
                                    )
                                    for w in range(KERNEL_WIDTH - 2):
                                        r_state[
                                            (KERNEL_WIDTH - 1) * vec_size + w * vec_size + i
                                        ] = r_state[
                                            (KERNEL_WIDTH - 1) * vec_size + (w + 1) * vec_size + i
                                        ]
                                    r_state[
                                        (KERNEL_WIDTH - 1) * vec_size
                                        + (KERNEL_WIDTH - 2) * vec_size
                                        + i
                                    ] = r_xk_raw
                            else:
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    r_gk_c2 = cutlass.Float32(
                                        qkg_cache[slot, i_t, 2, hk_off + k_idx]
                                    )
                                    sG[i_t, k_idx] = cute.math.exp(r_gk_c2, fastmath=True)
                        else:
                            token = bos + i_t
                            if warp_idx == 0:
                                for i_pair in range(vec_size // 2):
                                    i0 = i_pair * 2
                                    i1 = i_pair * 2 + 1
                                    k_idx0 = i0 * 32 + in_warp_tid
                                    k_idx1 = i1 * 32 + in_warp_tid
                                    r_conv_0 = 0.0
                                    r_conv_1 = 0.0
                                    if cutlass.const_expr(USE_REG_Q_WEIGHTS):
                                        _cwq_0 = r_wq[0 * vec_size + i0]
                                        _cwq_1 = r_wq[0 * vec_size + i1]
                                        r_conv_0 += r_state[0 * vec_size + i0] * _cwq_0
                                        r_conv_1 += r_state[0 * vec_size + i1] * _cwq_1
                                        _cwq_0 = r_wq[1 * vec_size + i0]
                                        _cwq_1 = r_wq[1 * vec_size + i1]
                                        r_conv_0 += r_state[1 * vec_size + i0] * _cwq_0
                                        r_conv_1 += r_state[1 * vec_size + i1] * _cwq_1
                                        _cwq_0 = r_wq[2 * vec_size + i0]
                                        _cwq_1 = r_wq[2 * vec_size + i1]
                                        r_conv_0 += r_state[2 * vec_size + i0] * _cwq_0
                                        r_conv_1 += r_state[2 * vec_size + i1] * _cwq_1
                                    else:
                                        _cwq_0 = cutlass.Float32(0.0)
                                        _cwq_1 = cutlass.Float32(0.0)
                                        for w in range(KERNEL_WIDTH - 1):
                                            _cwq_0 = sConvW[w * K + i0 * 32 + in_warp_tid]
                                            _cwq_1 = sConvW[w * K + i1 * 32 + in_warp_tid]
                                            r_conv_0 += r_state[w * vec_size + i0] * _cwq_0
                                            r_conv_1 += r_state[w * vec_size + i1] * _cwq_1
                                    if cutlass.const_expr(USE_FLAT_LAYOUT):
                                        r_xq_0 = cutlass.Float32(x_q[0, token, hk_off + k_idx0])
                                        r_xq_1 = cutlass.Float32(x_q[0, token, hk_off + k_idx1])
                                    else:
                                        r_xq_0 = cutlass.Float32(x_q[0, token, i_h, k_idx0])
                                        r_xq_1 = cutlass.Float32(x_q[0, token, i_h, k_idx1])
                                    if cutlass.const_expr(USE_REG_Q_WEIGHTS):
                                        _cwq_last_0 = r_wq[(KERNEL_WIDTH - 1) * vec_size + i0]
                                        _cwq_last_1 = r_wq[(KERNEL_WIDTH - 1) * vec_size + i1]
                                    else:
                                        _cwq_last_0 = sConvW[
                                            (KERNEL_WIDTH - 1) * K + i0 * 32 + in_warp_tid
                                        ]
                                        _cwq_last_1 = sConvW[
                                            (KERNEL_WIDTH - 1) * K + i1 * 32 + in_warp_tid
                                        ]
                                    r_conv_0 += r_xq_0 * _cwq_last_0
                                    r_conv_1 += r_xq_1 * _cwq_last_1
                                    e0 = cute.math.exp(-r_conv_0, fastmath=True)
                                    e1 = cute.math.exp(-r_conv_1, fastmath=True)
                                    sig_0 = cute.arch.rcp_approx(cutlass.Float32(1.0) + e0)
                                    sig_1 = cute.arch.rcp_approx(cutlass.Float32(1.0) + e1)
                                    r_q[i0] = r_conv_0 * sig_0
                                    r_q[i1] = r_conv_1 * sig_1
                                    r_state[0 * vec_size + i0] = r_state[1 * vec_size + i0]
                                    r_state[0 * vec_size + i1] = r_state[1 * vec_size + i1]
                                    r_state[1 * vec_size + i0] = r_state[2 * vec_size + i0]
                                    r_state[1 * vec_size + i1] = r_state[2 * vec_size + i1]
                                    r_state[2 * vec_size + i0] = r_xq_0
                                    r_state[2 * vec_size + i1] = r_xq_1
                                sum_q = 0.0
                                for i in range(vec_size):
                                    sum_q += r_q[i] * r_q[i]
                                for offset in [16, 8, 4, 2, 1]:
                                    sum_q += cute.arch.shuffle_sync_bfly(
                                        sum_q, offset=offset, mask=-1, mask_and_clamp=31
                                    )
                                rnorm_q_scaled = (
                                    cute.math.rsqrt(sum_q + 1e-06, fastmath=True) * scale
                                )
                                for i in range(vec_size):
                                    r_q[i] = r_q[i] * rnorm_q_scaled
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    sQ[i_t, k_idx] = r_q[i]
                            elif warp_idx == 1:
                                r_b_raw = cutlass.Float32(0.0)
                                if in_warp_tid == 0:
                                    r_b_raw = cutlass.Float32(beta[0, token, i_hv])
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    r_conv = (
                                        r_state[(KERNEL_WIDTH - 1) * vec_size + 0 * vec_size + i]
                                        * sConvW[k_weight_base + 0 * K + i * 32 + in_warp_tid]
                                    )
                                    r_conv += (
                                        r_state[(KERNEL_WIDTH - 1) * vec_size + 1 * vec_size + i]
                                        * sConvW[k_weight_base + 1 * K + i * 32 + in_warp_tid]
                                    )
                                    r_conv += (
                                        r_state[(KERNEL_WIDTH - 1) * vec_size + 2 * vec_size + i]
                                        * sConvW[k_weight_base + 2 * K + i * 32 + in_warp_tid]
                                    )
                                    if cutlass.const_expr(USE_FLAT_LAYOUT):
                                        r_xk = cutlass.Float32(x_k[0, token, hk_off + k_idx])
                                    else:
                                        r_xk = cutlass.Float32(x_k[0, token, i_h, k_idx])
                                    r_conv += (
                                        r_xk
                                        * sConvW[
                                            k_weight_base
                                            + (KERNEL_WIDTH - 1) * K
                                            + i * 32
                                            + in_warp_tid
                                        ]
                                    )
                                    r_conv = r_conv * cute.arch.rcp_approx(
                                        cutlass.Float32(1.0) + cute.math.exp(-r_conv, fastmath=True)
                                    )
                                    r_k[i] = r_conv
                                    r_state[(KERNEL_WIDTH - 1) * vec_size + 0 * vec_size + i] = (
                                        r_state[(KERNEL_WIDTH - 1) * vec_size + 1 * vec_size + i]
                                    )
                                    r_state[(KERNEL_WIDTH - 1) * vec_size + 1 * vec_size + i] = (
                                        r_state[(KERNEL_WIDTH - 1) * vec_size + 2 * vec_size + i]
                                    )
                                    r_state[(KERNEL_WIDTH - 1) * vec_size + 2 * vec_size + i] = r_xk
                                sum_k = 0.0
                                for i in range(vec_size):
                                    sum_k += r_k[i] * r_k[i]
                                for offset in [16, 8, 4, 2, 1]:
                                    sum_k += cute.arch.shuffle_sync_bfly(
                                        sum_k, offset=offset, mask=-1, mask_and_clamp=31
                                    )
                                rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=True)
                                for i in range(vec_size):
                                    r_k[i] = r_k[i] * rnorm_k
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    sK[i_t, k_idx] = r_k[i]
                                if in_warp_tid == 0:
                                    sBeta[i_t] = cute.arch.rcp_approx(
                                        cutlass.Float32(1.0)
                                        + cute.math.exp(-r_b_raw, fastmath=True)
                                    )
                            else:
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    r_g_raw = cutlass.Float32(g[0, token, i_hv, k_idx])
                                    r_g_raw = r_g_raw + cutlass.Float32(dt_bias[i_h * K + k_idx])
                                    exp_A_x = r_exp_A * r_g_raw
                                    sigmoid_val = cute.arch.rcp_approx(
                                        cutlass.Float32(1.0)
                                        + cute.math.exp(-exp_A_x, fastmath=True)
                                    )
                                    r_gk = lower_bound * sigmoid_val
                                    sG[i_t, k_idx] = cute.math.exp(r_gk, fastmath=True)
                                    r_decay[i] = r_gk
                                if i_t > commit_len:
                                    cache_pos = i_t - commit_len - 1
                                    for i in range(vec_size):
                                        k_idx = i * 32 + in_warp_tid
                                        qkg_cache[slot, cache_pos, 2, hk_off + k_idx] = r_decay[i]
                            if i_t == commit_len:
                                if warp_idx == 0:
                                    for i in range(vec_size):
                                        k_idx = i * 32 + in_warp_tid
                                        for w in range(KERNEL_WIDTH - 1):
                                            cs_q[slot, hk_off + k_idx, w] = r_state[
                                                w * vec_size + i
                                            ]
                                elif warp_idx == 1:
                                    for i in range(vec_size):
                                        k_idx = i * 32 + in_warp_tid
                                        for w in range(KERNEL_WIDTH - 1):
                                            cs_k[slot, hk_off + k_idx, w] = r_state[
                                                (KERNEL_WIDTH - 1) * vec_size + w * vec_size + i
                                            ]
                            if i_t > commit_len:
                                cache_pos = i_t - commit_len - 1
                                if warp_idx == 0:
                                    for i in range(vec_size):
                                        k_idx = i * 32 + in_warp_tid
                                        qkg_cache[slot, cache_pos, 0, hk_off + k_idx] = sQ[
                                            i_t, k_idx
                                        ]
                                    for i in range(vec_size):
                                        k_idx = i * 32 + in_warp_tid
                                        cs_q[slot, hk_off + k_idx, KERNEL_WIDTH - 1 + cache_pos] = (
                                            r_state[(KERNEL_WIDTH - 2) * vec_size + i]
                                        )
                                elif warp_idx == 1:
                                    for i in range(vec_size):
                                        k_idx = i * 32 + in_warp_tid
                                        qkg_cache[slot, cache_pos, 1, hk_off + k_idx] = sK[
                                            i_t, k_idx
                                        ]
                                    if in_warp_tid == 0:
                                        beta_cache[slot, cache_pos, i_hv] = sBeta[i_t]
                                    for i in range(vec_size):
                                        k_idx = i * 32 + in_warp_tid
                                        cs_k[slot, hk_off + k_idx, KERNEL_WIDTH - 1 + cache_pos] = (
                                            r_state[
                                                (KERNEL_WIDTH - 1) * vec_size
                                                + (KERNEL_WIDTH - 2) * vec_size
                                                + i
                                            ]
                                        )
                        i_t = i_t + 1
                else:
                    _v_idx = tidx - 96
                    if _v_idx < V:
                        _csv0 = cutlass.Float32(cs_v[slot, hv_off + _v_idx, 0])
                        _csv1 = cutlass.Float32(cs_v[slot, hv_off + _v_idx, 1])
                        _csv2 = cutlass.Float32(cs_v[slot, hv_off + _v_idx, 2])
                        if cutlass.const_expr(USE_ZERO_ACCEPTED):
                            _wv0 = sConvW[v_weight_base + 0 * V + _v_idx]
                            _wv1 = sConvW[v_weight_base + 1 * V + _v_idx]
                            _wv2 = sConvW[v_weight_base + 2 * V + _v_idx]
                            _wv3 = sConvW[v_weight_base + (KERNEL_WIDTH - 1) * V + _v_idx]
                            if cutlass.const_expr(USE_FLAT_LAYOUT):
                                _xv0 = cutlass.Float32(x_v[0, bos + 0, hv_off + _v_idx])
                                _xv1 = cutlass.Float32(x_v[0, bos + 1, hv_off + _v_idx])
                                _xv2 = cutlass.Float32(x_v[0, bos + 2, hv_off + _v_idx])
                            else:
                                _xv0 = cutlass.Float32(x_v[0, bos + 0, i_hv, _v_idx])
                                _xv1 = cutlass.Float32(x_v[0, bos + 1, i_hv, _v_idx])
                                _xv2 = cutlass.Float32(x_v[0, bos + 2, i_hv, _v_idx])
                            _vconv0 = _csv0 * _wv0
                            _vconv0 += _csv1 * _wv1
                            _vconv0 += _csv2 * _wv2
                            _vconv0 += _xv0 * _wv3
                            _vconv0 = _vconv0 * cute.arch.rcp_approx(
                                cutlass.Float32(1.0) + cute.math.exp(-_vconv0, fastmath=True)
                            )
                            sVall[0 * V + _v_idx] = _vconv0
                            cs_v[slot, hv_off + _v_idx, 0] = _csv1
                            cs_v[slot, hv_off + _v_idx, 1] = _csv2
                            cs_v[slot, hv_off + _v_idx, 2] = _xv0
                            _vconv1, _vconv2 = cute.arch.mul_packed_f32x2(
                                (_csv1, _csv2), (_wv0, _wv0)
                            )
                            _vconv1, _vconv2 = cute.arch.fma_packed_f32x2(
                                (_csv2, _xv0), (_wv1, _wv1), (_vconv1, _vconv2)
                            )
                            _vconv1, _vconv2 = cute.arch.fma_packed_f32x2(
                                (_xv0, _xv1), (_wv2, _wv2), (_vconv1, _vconv2)
                            )
                            _vconv1, _vconv2 = cute.arch.fma_packed_f32x2(
                                (_xv1, _xv2), (_wv3, _wv3), (_vconv1, _vconv2)
                            )
                            _vconv1 = _vconv1 * cute.arch.rcp_approx(
                                cutlass.Float32(1.0) + cute.math.exp(-_vconv1, fastmath=True)
                            )
                            _vconv2 = _vconv2 * cute.arch.rcp_approx(
                                cutlass.Float32(1.0) + cute.math.exp(-_vconv2, fastmath=True)
                            )
                            sVall[1 * V + _v_idx] = _vconv1
                            v_cache[slot, 0, hv_off + _v_idx] = _vconv1
                            cs_v[slot, hv_off + _v_idx, KERNEL_WIDTH - 1] = _xv1
                            sVall[2 * V + _v_idx] = _vconv2
                            v_cache[slot, 1, hv_off + _v_idx] = _vconv2
                            cs_v[slot, hv_off + _v_idx, KERNEL_WIDTH] = _xv2
                        else:
                            _i_t = 0
                            while _i_t < T_loop:
                                if _i_t < commit_len:
                                    sVall[_i_t * V + _v_idx] = cutlass.Float32(
                                        v_cache[slot, _i_t, hv_off + _v_idx]
                                    )
                                    _xv_replay = cutlass.Float32(
                                        cs_v[slot, hv_off + _v_idx, KERNEL_WIDTH - 1 + _i_t]
                                    )
                                    _csv0 = _csv1
                                    _csv1 = _csv2
                                    _csv2 = _xv_replay
                                else:
                                    _token_v = bos + _i_t
                                    _v_conv = 0.0
                                    _v_conv += _csv0 * sConvW[v_weight_base + 0 * V + _v_idx]
                                    _v_conv += _csv1 * sConvW[v_weight_base + 1 * V + _v_idx]
                                    _v_conv += _csv2 * sConvW[v_weight_base + 2 * V + _v_idx]
                                    if cutlass.const_expr(USE_FLAT_LAYOUT):
                                        _xv = cutlass.Float32(x_v[0, _token_v, hv_off + _v_idx])
                                    else:
                                        _xv = cutlass.Float32(x_v[0, _token_v, i_hv, _v_idx])
                                    _v_conv += (
                                        _xv
                                        * sConvW[v_weight_base + (KERNEL_WIDTH - 1) * V + _v_idx]
                                    )
                                    _v_conv = _v_conv * cute.arch.rcp_approx(
                                        cutlass.Float32(1.0)
                                        + cute.math.exp(-_v_conv, fastmath=True)
                                    )
                                    sVall[_i_t * V + _v_idx] = _v_conv
                                    _csv0 = _csv1
                                    _csv1 = _csv2
                                    _csv2 = _xv
                                    if _i_t == commit_len:
                                        cs_v[slot, hv_off + _v_idx, 0] = _csv0
                                        cs_v[slot, hv_off + _v_idx, 1] = _csv1
                                        cs_v[slot, hv_off + _v_idx, 2] = _csv2
                                    if _i_t > commit_len:
                                        _cp = _i_t - commit_len - 1
                                        v_cache[slot, _cp, hv_off + _v_idx] = _v_conv
                                        cs_v[slot, hv_off + _v_idx, KERNEL_WIDTH - 1 + _cp] = _xv
                                _i_t = _i_t + 1
                if cutlass.const_expr(PROFILE_STAGES):
                    cute.arch.barrier()
                    t_stage1 = read_globaltimer()
            else:
                cute.arch.barrier()
                if cutlass.const_expr(USE_SETMAXREG):
                    cute.arch.warpgroup_reg_dealloc(64)
                if cutlass.const_expr(PROFILE_STAGES):
                    cute.arch.barrier()
                    t_stage1 = read_globaltimer()
        else:
            cute.arch.barrier()
            if cutlass.const_expr(USE_SETMAXREG):
                cute.arch.warpgroup_reg_dealloc(64)
            if cutlass.const_expr(PROFILE_STAGES):
                cute.arch.barrier()
                t_stage1 = read_globaltimer()
        for row in range(NUM_V_ROWS):
            v_row = warp_idx * NUM_V_ROWS + row
            for i in range(vec_size):
                if cutlass.const_expr(USE_FLAT_LAYOUT):
                    r_state[row * vec_size + i] = cutlass.Float32(
                        h0[h0_idx, v_row, i * 32 + in_warp_tid]
                    )
                else:
                    r_state[row * vec_size + i] = cutlass.Float32(
                        h0[slot, i_hv, v_row, i * 32 + in_warp_tid]
                    )
        if cutlass.const_expr(USE_SETMAXREG):
            cute.arch.warpgroup_reg_alloc(72)
        cute.arch.barrier()
        for i_v in range(num_v_tiles):
            v_base = i_v * TILE_V
            if i_v > 0:
                for row in range(NUM_V_ROWS):
                    v_row = warp_idx * NUM_V_ROWS + row
                    for i in range(vec_size):
                        if cutlass.const_expr(USE_FLAT_LAYOUT):
                            r_state[row * vec_size + i] = cutlass.Float32(
                                h0[h0_idx, v_base + v_row, i * 32 + in_warp_tid]
                            )
                        else:
                            r_state[row * vec_size + i] = cutlass.Float32(
                                h0[slot, i_hv, v_base + v_row, i * 32 + in_warp_tid]
                            )
            r_v_val = cutlass.Float32(0.0)
            i_t = 0
            while i_t < T_loop:
                if in_warp_tid < NUM_V_ROWS:
                    v_idx = v_base + warp_idx * NUM_V_ROWS + in_warp_tid
                    r_v_val = sVall[i_t * V + v_idx]
                r_beta_val = sBeta[i_t]
                for i_pair in range(vec_size // 2):
                    i0 = i_pair * 2
                    i1 = i_pair * 2 + 1
                    k_idx0 = i0 * 32 + in_warp_tid
                    k_idx1 = i1 * 32 + in_warp_tid
                    r_q[i0] = sQ[i_t, k_idx0]
                    r_q[i1] = sQ[i_t, k_idx1]
                    _k0 = sK[i_t, k_idx0]
                    _k1 = sK[i_t, k_idx1]
                    r_decay[i0] = sG[i_t, k_idx0]
                    r_decay[i1] = sG[i_t, k_idx1]
                    r_bk[i0], r_bk[i1] = cute.arch.mul_packed_f32x2(
                        (r_beta_val, r_beta_val), (_k0, _k1)
                    )
                    r_k[i0], r_k[i1] = cute.arch.mul_packed_f32x2(
                        (r_decay[i0], r_decay[i1]), (_k0, _k1)
                    )
                for row_pair in range(NUM_V_ROWS // 2):
                    ra = row_pair * 2
                    rb = row_pair * 2 + 1
                    r_va = cute.arch.shuffle_sync(r_v_val, ra)
                    r_vb = cute.arch.shuffle_sync(r_v_val, rb)
                    shk_a1 = 0.0
                    shk_a2 = 0.0
                    shk_b1 = 0.0
                    shk_b2 = 0.0
                    for _pi in range(vec_size // 2):
                        _p = _pi * 2
                        shk_a1, shk_a2 = cute.arch.fma_packed_f32x2(
                            src_a=(
                                r_state[ra * vec_size + _p],
                                r_state[ra * vec_size + _p + 1],
                            ),
                            src_b=(r_k[_p], r_k[_p + 1]),
                            src_c=(shk_a1, shk_a2),
                        )
                        shk_b1, shk_b2 = cute.arch.fma_packed_f32x2(
                            src_a=(
                                r_state[rb * vec_size + _p],
                                r_state[rb * vec_size + _p + 1],
                            ),
                            src_b=(r_k[_p], r_k[_p + 1]),
                            src_c=(shk_b1, shk_b2),
                        )
                    shk_a = shk_a1 + shk_a2
                    shk_b = shk_b1 + shk_b2
                    for offset in [16, 8, 4, 2, 1]:
                        shk_a += cute.arch.shuffle_sync_bfly(
                            shk_a, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        shk_b += cute.arch.shuffle_sync_bfly(
                            shk_b, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    vn_a = r_va - shk_a
                    vn_b = r_vb - shk_b
                    shq_a1 = 0.0
                    shq_a2 = 0.0
                    shq_b1 = 0.0
                    shq_b2 = 0.0
                    for _pi in range(vec_size // 2):
                        _p = _pi * 2
                        vnbk_a0, vnbk_a1 = cute.arch.mul_packed_f32x2(
                            (vn_a, vn_a), (r_bk[_p], r_bk[_p + 1])
                        )
                        vnbk_b0, vnbk_b1 = cute.arch.mul_packed_f32x2(
                            (vn_b, vn_b), (r_bk[_p], r_bk[_p + 1])
                        )
                        r_state[ra * vec_size + _p], r_state[ra * vec_size + _p + 1] = (
                            cute.arch.fma_packed_f32x2(
                                src_a=(r_decay[_p], r_decay[_p + 1]),
                                src_b=(
                                    r_state[ra * vec_size + _p],
                                    r_state[ra * vec_size + _p + 1],
                                ),
                                src_c=(vnbk_a0, vnbk_a1),
                            )
                        )
                        r_state[rb * vec_size + _p], r_state[rb * vec_size + _p + 1] = (
                            cute.arch.fma_packed_f32x2(
                                src_a=(r_decay[_p], r_decay[_p + 1]),
                                src_b=(
                                    r_state[rb * vec_size + _p],
                                    r_state[rb * vec_size + _p + 1],
                                ),
                                src_c=(vnbk_b0, vnbk_b1),
                            )
                        )
                        shq_a1, shq_a2 = cute.arch.fma_packed_f32x2(
                            src_a=(
                                r_state[ra * vec_size + _p],
                                r_state[ra * vec_size + _p + 1],
                            ),
                            src_b=(r_q[_p], r_q[_p + 1]),
                            src_c=(shq_a1, shq_a2),
                        )
                        shq_b1, shq_b2 = cute.arch.fma_packed_f32x2(
                            src_a=(
                                r_state[rb * vec_size + _p],
                                r_state[rb * vec_size + _p + 1],
                            ),
                            src_b=(r_q[_p], r_q[_p + 1]),
                            src_c=(shq_b1, shq_b2),
                        )
                    shq_a = shq_a1 + shq_a2
                    shq_b = shq_b1 + shq_b2
                    for offset in [16, 8, 4, 2, 1]:
                        shq_a += cute.arch.shuffle_sync_bfly(
                            shq_a, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        shq_b += cute.arch.shuffle_sync_bfly(
                            shq_b, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    if in_warp_tid == 0:
                        v_row_a = warp_idx * NUM_V_ROWS + ra
                        v_row_b = warp_idx * NUM_V_ROWS + rb
                        if cutlass.const_expr(USE_ZERO_ACCEPTED):
                            o[0, bos + i_t, i_hv, v_base + v_row_a] = cutlass.BFloat16(shq_a)
                            o[0, bos + i_t, i_hv, v_base + v_row_b] = cutlass.BFloat16(shq_b)
                        elif i_t >= commit_len:
                            o[0, bos + i_t, i_hv, v_base + v_row_a] = cutlass.BFloat16(shq_a)
                            o[0, bos + i_t, i_hv, v_base + v_row_b] = cutlass.BFloat16(shq_b)
                if i_t == commit_len:
                    for row in range(NUM_V_ROWS):
                        v_row = warp_idx * NUM_V_ROWS + row
                        for i in range(vec_size):
                            if cutlass.const_expr(USE_FLAT_LAYOUT):
                                ht[h0_idx, v_base + v_row, i * 32 + in_warp_tid] = r_state[
                                    row * vec_size + i
                                ]
                            else:
                                ht[slot, i_hv, v_base + v_row, i * 32 + in_warp_tid] = r_state[
                                    row * vec_size + i
                                ]
                i_t = i_t + 1
        if cutlass.const_expr(PROFILE_STAGES):
            cute.arch.barrier()
            t_stage2 = read_globaltimer()
            if tidx == 0:
                _, grid_n, _ = cute.arch.grid_dim()
                timing_base = (i_hv * grid_n + i_n) * 4
                stage_timing[timing_base + 0] = t_stage1 - t_stage0
                stage_timing[timing_base + 1] = t_stage2 - t_stage1
                stage_timing[timing_base + 2] = t_stage2 - t_stage0
                stage_timing[timing_base + 3] = t_stage0
