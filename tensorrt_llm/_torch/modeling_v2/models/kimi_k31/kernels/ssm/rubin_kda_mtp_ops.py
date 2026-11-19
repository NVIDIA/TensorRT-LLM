# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host launchers of the Kimi K3 / K3.1 KDA multi-token speculative verify (CuTe DSL).

Compiles and launches the two kernels of the runtime verify contract,
``kda_decode_mtp_v2_kernel`` (``rubin_kda_mtp_decode_v2.py``) and
``kda_decode_mtp_2cta_kernel`` (``rubin_kda_mtp_decode_2cta.py``), the family's
versions of the engine's ``trtllm::kda_mtp_decode``. The launchers are plain
Python functions, not torch ops. One launch fuses, per generation request:
replay of previously-accepted draft tokens from the ``kg/v/beta`` caches,
causal conv + SiLU, Q/K L2 norm, beta sigmoid, lower-bound gate, the KDA
delta-rule recurrence over the ``1 + num_spec`` new tokens and the gated
RMSNorm of the output. The recurrent state and base conv windows are
committed **in place** after the first new (golden) token; the spec tokens
are cached for the next round.

State-management contract (differs from the engine's intermediate-buffer +
``update_mamba_states`` promotion flow): after this op returns, the pools
hold the state as of the last golden token, with the new spec tokens pending
in the replay caches. The next round passes ``num_accepted_tokens`` (how
many of those pending drafts the sampler accepted) and the kernel replays
them before the new tokens. No host-side promotion of KDA SSM/conv state is
required or allowed.

The compile cache is keyed by dtype / shape / stride layouts and constexpr
flags. Bias and ``pad_slot_id`` are not supported; the MXFP8 quantization of
the output is an optional epilogue.

Kernel shape contract: ``K == V == 128``, conv width ``W == 4``,
``HV == H``, one V pass (TILE_V=128, 512 threads). ``num_spec`` is a
compile-time constant per cache allocation.

Kernel selection (``_select_verify_kernel``). The runtime verify contract is
packed token rows (exactly ``1 + num_spec`` per request), the fused gated
RMSNorm and the 3xTF32 recurrence (at most ``MAX_NUM_SPEC`` drafts,
vector-aligned bf16 operands). The two kernels have the same arithmetic in
the same order and bitwise identical results:

* ``kda_decode_mtp_v2_kernel`` (one CTA or a cluster pair per tile): calls with
  at most one ``(head, request)`` tile per SM.
* ``kda_decode_mtp_2cta_kernel`` (two CTAs per SM): calls with more tiles than
  SMs.

The MXFP8 output (``quantize_output``) is the catalog quantizer's 1x32 form of
the BF16 output (the operand ``rubin_mxfp8_linear`` makes of it, bit for bit),
an epilogue of both kernels (``mxfp8_exact_store``) that runs on the kernel the
BF16 output takes.
"""

from math import gcd
from typing import Callable, Literal, NamedTuple, Optional, Tuple

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm.logger import logger

if IS_CUTLASS_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
    from cutlass.cute.runtime import from_dlpack

    from .rubin_kda_mtp_decode import MAX_NUM_SPEC, NUM_THREADS, SPLIT_MAX_TILES, T_PAD, TILE_K
    from .rubin_kda_mtp_decode_2cta import kda_decode_mtp_2cta_kernel
    from .rubin_kda_mtp_decode_v2 import kda_decode_mtp_v2_kernel
else:
    raise ImportError("Kimi K3 KDA MTP decode requires NVIDIA CUTLASS DSL")

# CTAs per SM kda_decode_mtp_2cta_kernel is compiled for (64 registers per thread).
_TWO_CTA_BLOCKS_PER_SM = 2


if IS_CUTLASS_DSL_AVAILABLE:

    class _VerifyMmaAtoms(NamedTuple):
        """tcgen05 MMA atoms and operand smem layouts shared by the verify kernels."""

        tiled_mma1: "cute.TiledMma"
        smem_layout_b1: "cute.ComposedLayout"
        tiled_mma_f32a: "cute.TiledMma"
        tiled_mma1k: "cute.TiledMma"
        smem_layout_b1k: "cute.ComposedLayout"
        tiled_mma2h: "cute.TiledMma"
        smem_layout_b2h: "cute.ComposedLayout"
        smem_layout_a2: "cute.ComposedLayout"
        tiled_mma2q: "cute.TiledMma"
        smem_layout_b2q: "cute.ComposedLayout"
        tiled_mma3: "cute.TiledMma"
        smem_layout_a3: "cute.ComposedLayout"
        smem_layout_b3: "cute.ComposedLayout"
        tiled_mma4: "cute.TiledMma"

    def _verify_mma_atoms(V: int, K: int) -> _VerifyMmaAtoms:
        """The verify MMAs: tf32 operands (3xTF32) with fp32 accumulators.

        GEMM1: M = V rows of the state (TMEM), N = 3 * T_PAD operand rows [K~; Q~; K~lo];
        ``tiled_mma_f32a``, GEMM1's atom, is also the fp32 A-fragment layout of the (V x K) state
        in TMEM (the GEMM2 accumulator seed / state store views).
        GEMM1b: the S0_lo sweep only needs K~hi (N = T_PAD). GEMM2: (V x T_PAD) x (T_PAD x K),
        both operands in smem, issued as two N = K/2 column halves so the state store can start
        on half 0; half 0 also carries the output-combination GEMM (N = T_PAD (Bcoef) + K/2).
        GEMM3: T x T coupling matrices, A3 = [Kf; Qf] padded to 64 rows, B3 = Kb (16 rows).
        ``tiled_mma4`` only provides the accumulator layout of the output combination
        D4[v, t] = sum_s V'[s, v] Bcoef[t, s], the first T_PAD columns of GEMM2 half 0.
        """
        f32 = cutlass.Float32

        def mma(shape_mn, source):
            return sm100_utils.make_trivial_tiled_mma(
                f32,
                OperandMajorMode.K,
                OperandMajorMode.K,
                f32,
                tcgen05.CtaGroup.ONE,
                shape_mn,
                source,
            )

        tmem, smem = tcgen05.OperandSource.TMEM, tcgen05.OperandSource.SMEM
        tiled_mma1 = tiled_mma_f32a = mma((V, 3 * T_PAD), tmem)
        tiled_mma1k = mma((V, T_PAD), tmem)
        tiled_mma2h = mma((V, K // 2), smem)
        tiled_mma2q = mma((V, T_PAD + K // 2), smem)
        tiled_mma3 = mma((4 * T_PAD, T_PAD), smem)
        return _VerifyMmaAtoms(
            tiled_mma1=tiled_mma1,
            smem_layout_b1=sm100_utils.make_smem_layout_b(tiled_mma1, (V, 3 * T_PAD, K), f32, 1),
            tiled_mma_f32a=tiled_mma_f32a,
            tiled_mma1k=tiled_mma1k,
            smem_layout_b1k=sm100_utils.make_smem_layout_b(tiled_mma1k, (V, T_PAD, K), f32, 1),
            tiled_mma2h=tiled_mma2h,
            smem_layout_b2h=sm100_utils.make_smem_layout_b(tiled_mma2h, (V, K // 2, T_PAD), f32, 1),
            smem_layout_a2=sm100_utils.make_smem_layout_a(tiled_mma2h, (V, K // 2, T_PAD), f32, 1),
            tiled_mma2q=tiled_mma2q,
            smem_layout_b2q=sm100_utils.make_smem_layout_b(
                tiled_mma2q, (V, T_PAD + K // 2, T_PAD), f32, 1
            ),
            tiled_mma3=tiled_mma3,
            smem_layout_a3=sm100_utils.make_smem_layout_a(
                tiled_mma3, (4 * T_PAD, T_PAD, K), f32, 1
            ),
            smem_layout_b3=sm100_utils.make_smem_layout_b(
                tiled_mma3, (4 * T_PAD, T_PAD, K), f32, 1
            ),
            tiled_mma4=mma((V, T_PAD), smem),
        )

    class _StateTmaAtoms(NamedTuple):
        """TMA load of the initial state S0 and bulk store of the committed state."""

        tma_atom_l: "cpasync.CopyAtom"
        mS_tma: "cute.Tensor"
        box_layout: "cute.ComposedLayout"
        tma_atom_t: "cpasync.CopyAtom"
        mT_tma: "cute.Tensor"
        box_layout_t: "cute.ComposedLayout"

    def _state_tiles(state: cute.Tensor, V: int, K: int) -> cute.Tensor:
        """(V, K, tiles) view of a V-first state pool with its own strides (the per-layer state
        cache may be a strided slice of a larger buffer); the tile mode is the pool index (flat
        ``[pool, V, K]`` layout) or the (head, slot) pair (``[pool, H, V, K]``)."""
        stride = state.layout.stride
        if cute.rank(state.shape) == 3:
            return cute.make_tensor(
                state.iterator,
                cute.make_layout((V, K, state.shape[0]), stride=(stride[1], stride[2], stride[0])),
            )
        return cute.make_tensor(
            state.iterator,
            cute.make_layout(
                (V, K, (state.shape[1], state.shape[0])),
                stride=(stride[2], stride[3], (stride[1], stride[0])),
            ),
        )

    def _state_tma_atoms(
        h0: cute.Tensor, ht: cute.Tensor, V: int, K: int, split_v: int
    ) -> _StateTmaAtoms:
        """S0 is loaded from ``h0`` and the committed state stored to ``ht`` in 128B-swizzled
        boxes of 32 fp32 or 64 bf16 state columns; each CTA of a ``split_v`` group stores its
        V / split_v rows."""
        box_cols = 128 // (h0.element_type.width // 8)
        box_atom = sm100_utils.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, h0.element_type
        )
        box_layout = cute.tile_to_shape(box_atom, (V, box_cols), order=(0, 1))
        box_layout_t = cute.tile_to_shape(box_atom, (V // split_v, box_cols), order=(0, 1))
        tma_atom_l, mS_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            _state_tiles(h0, V, K),
            box_layout,
            (V, box_cols),
        )
        tma_atom_t, mT_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            _state_tiles(ht, V, K),
            box_layout_t,
            (V // split_v, box_cols),
        )
        return _StateTmaAtoms(tma_atom_l, mS_tma, box_layout, tma_atom_t, mT_tma, box_layout_t)

    def _launch_verify(launched, HV: int, N, split_v: int, stream, **launch_kwargs) -> None:
        if split_v == 2:
            # cluster pair: the two CTAs of a (head, request) must be co-scheduled (in-place state)
            launched.launch(
                grid=(HV, N, 2),
                block=[NUM_THREADS, 1, 1],
                cluster=(1, 1, 2),
                stream=stream,
                **launch_kwargs,
            )
        else:
            launched.launch(
                grid=(HV, N, 1), block=[NUM_THREADS, 1, 1], stream=stream, **launch_kwargs
            )

    @cute.jit
    def _run_kda_decode_mtp_v2(
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
        onorm_g: cute.Tensor,
        onorm_weight: cute.Tensor,
        o: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        v_cache: cute.Tensor,
        beta_cache: cute.Tensor,
        ssm_state_indices: cute.Tensor,
        num_accepted_tokens: cute.Tensor,
        o8: cute.Tensor,
        osf: cute.Tensor,
        N: cutlass.Int32,
        scale: cutlass.Constexpr[float],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        NUM_SPEC: cutlass.Constexpr[int],
        KERNEL_WIDTH: cutlass.Constexpr[int],
        lower_bound: cutlass.Constexpr[float],
        onorm_eps: cutlass.Constexpr[float],
        USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
        SPLIT_V: cutlass.Constexpr[int],
        MXFP8_OUT: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        """Host side of ``kda_decode_mtp_v2_kernel`` (packed rows, fused gated RMSNorm,
        3xTF32): the tf32 MMA atoms, operand smem layouts and state TMA atoms, then the launch.
        With ``MXFP8_OUT`` the kernel writes the E4M3 ``o8`` and the R128c4 scales ``osf`` in the
        catalog quantizer's 1x32 form instead of ``o`` (else both are placeholders)."""
        mma = _verify_mma_atoms(V, K)
        state_tma = _state_tma_atoms(h0, h0, V, K, SPLIT_V)
        launched = kda_decode_mtp_v2_kernel(
            h0,
            x_q,
            x_k,
            x_v,
            w_q,
            w_k,
            w_v,
            cs_q,
            cs_k,
            cs_v,
            A_log,
            g,
            dt_bias,
            beta,
            onorm_g,
            onorm_weight,
            o,
            k_cache,
            g_cache,
            v_cache,
            beta_cache,
            ssm_state_indices,
            num_accepted_tokens,
            cute.make_layout((2 * NUM_SPEC + 1, K), stride=(K, 1)),
            mma.tiled_mma1,
            mma.tiled_mma3,
            mma.tiled_mma1k,
            mma.smem_layout_b1,
            mma.smem_layout_a2,
            mma.smem_layout_a3,
            mma.smem_layout_b3,
            mma.smem_layout_b1k,
            mma.tiled_mma4,
            state_tma.tma_atom_t,
            state_tma.mT_tma,
            state_tma.box_layout,
            state_tma.tma_atom_l,
            state_tma.mS_tma,
            mma.tiled_mma2h,
            mma.smem_layout_b2h,
            state_tma.box_layout_t,
            mma.tiled_mma2q,
            mma.smem_layout_b2q,
            o8,
            osf,
            scale,
            HV,
            K,
            V,
            NUM_SPEC,
            KERNEL_WIDTH,
            lower_bound,
            onorm_eps,
            USE_FLAT_LAYOUT,
            SPLIT_V,
            MXFP8_OUT,
        )
        _launch_verify(launched, HV, N, SPLIT_V, stream)

    @cute.jit
    def _run_kda_decode_mtp_2cta(
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
        onorm_g: cute.Tensor,
        onorm_weight: cute.Tensor,
        o: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        v_cache: cute.Tensor,
        beta_cache: cute.Tensor,
        ssm_state_indices: cute.Tensor,
        num_accepted_tokens: cute.Tensor,
        o8: cute.Tensor,
        osf: cute.Tensor,
        N: cutlass.Int32,
        scale: cutlass.Constexpr[float],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        NUM_SPEC: cutlass.Constexpr[int],
        KERNEL_WIDTH: cutlass.Constexpr[int],
        lower_bound: cutlass.Constexpr[float],
        onorm_eps: cutlass.Constexpr[float],
        USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
        MXFP8_OUT: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        """Host side of ``kda_decode_mtp_2cta_kernel`` (fp32 or bf16 state), launched for two
        CTAs per SM (64 registers per thread)."""
        mma = _verify_mma_atoms(V, K)
        state_tma = _state_tma_atoms(h0, h0, V, K, 1)
        launched = kda_decode_mtp_2cta_kernel(
            h0,
            x_q,
            x_k,
            x_v,
            w_q,
            w_k,
            w_v,
            cs_q,
            cs_k,
            cs_v,
            A_log,
            g,
            dt_bias,
            beta,
            onorm_g,
            onorm_weight,
            o,
            k_cache,
            g_cache,
            v_cache,
            beta_cache,
            ssm_state_indices,
            num_accepted_tokens,
            o8,
            osf,
            mma.tiled_mma1,
            mma.tiled_mma3,
            mma.tiled_mma1k,
            mma.smem_layout_b1,
            mma.smem_layout_a2,
            mma.smem_layout_a3,
            mma.smem_layout_b3,
            mma.smem_layout_b1k,
            mma.tiled_mma4,
            mma.tiled_mma2h,
            mma.smem_layout_b2h,
            mma.tiled_mma2q,
            mma.smem_layout_b2q,
            state_tma.tma_atom_l,
            state_tma.mS_tma,
            state_tma.box_layout,
            state_tma.tma_atom_t,
            state_tma.mT_tma,
            scale,
            HV,
            K,
            V,
            NUM_SPEC,
            KERNEL_WIDTH,
            lower_bound,
            onorm_eps,
            USE_FLAT_LAYOUT,
            MXFP8_OUT,
        )
        _launch_verify(launched, HV, N, 1, stream, min_blocks_per_mp=_TWO_CTA_BLOCKS_PER_SM)


def _require_stride_layout(
    *,
    x_q,
    x_k,
    x_v,
    w_q,
    w_k,
    w_v,
    cs_q,
    cs_k,
    cs_v,
    g,
    beta,
    A_log,
    dt_bias,
    recurrent_state,
    k_cache,
    g_cache,
    v_cache,
    beta_cache,
    ssm_state_indices,
    num_accepted_tokens,
    out,
    onorm_g,
    onorm_weight,
    H,
    HV,
    K,
    V,
    W,
    num_spec,
    T_total,
    quantize_output,
):
    if x_q.ndim != 4 or x_k.ndim != 4 or x_v.ndim != 4:
        raise ValueError("Expected x_q/x_k/x_v to have shape [1, T, H, D].")
    if x_q.shape != (1, T_total, H, K) or x_k.shape != (1, T_total, H, K):
        raise ValueError(f"Expected x_q/x_k shape [1, {T_total}, {H}, {K}].")
    if x_v.shape != (1, T_total, HV, V):
        raise ValueError(f"Expected x_v shape [1, {T_total}, {HV}, {V}].")
    if g.ndim != 4 or g.shape != (1, T_total, HV, K):
        raise ValueError(f"Expected g shape [1, {T_total}, {HV}, {K}].")
    if beta.ndim != 3 or beta.shape != (1, T_total, HV):
        raise ValueError(f"Expected beta shape [1, {T_total}, {HV}].")
    if quantize_output:
        if out.ndim != 2 or out.shape != (T_total, HV * V):
            raise ValueError(f"Expected FP8 out shape [{T_total}, {HV * V}].")
        if out.dtype is not torch.float8_e4m3fn:
            raise TypeError("Expected FP8 out dtype torch.float8_e4m3fn.")
    elif out.ndim != 4 or out.shape != (1, T_total, HV, V):
        raise ValueError(f"Expected BF16 out shape [1, {T_total}, {HV}, {V}].")
    if onorm_g.ndim != 4 or onorm_g.shape != (1, T_total, HV, V):
        raise ValueError(f"Expected onorm_g shape [1, {T_total}, {HV}, {V}].")
    if onorm_g.stride(-1) != 1 or onorm_g.stride(-2) != V:
        raise ValueError("Expected onorm_g to have contiguous head/value blocks.")
    if onorm_weight.ndim != 1 or onorm_weight.shape[0] != V:
        raise ValueError(f"Expected onorm_weight shape [{V}].")
    if onorm_weight.stride(0) != 1:
        raise ValueError("Expected onorm_weight to be contiguous.")

    last_dim_tensors = {
        "x_q": x_q,
        "x_k": x_k,
        "x_v": x_v,
        "g": g,
        "beta": beta,
        "recurrent_state": recurrent_state,
        "k_cache": k_cache,
        "g_cache": g_cache,
        "v_cache": v_cache,
        "beta_cache": beta_cache,
    }
    if not quantize_output:
        last_dim_tensors["out"] = out
    for name, tensor in last_dim_tensors.items():
        if tensor.stride(-1) != 1:
            raise ValueError(f"Expected {name} to be contiguous in its last dimension.")

    if w_q.shape != (H * K, W) or w_k.shape != (H * K, W) or w_v.shape != (HV * V, W):
        raise ValueError(f"Expected w_q/w_k shape [{H * K}, {W}] and w_v shape [{HV * V}, {W}].")
    if w_q.stride(1) != 1 or w_k.stride(1) != 1 or w_v.stride(1) != 1:
        raise ValueError("Expected w_q/w_k/w_v to be contiguous in the kernel-width axis.")

    if A_log.ndim != 1 or A_log.shape[0] != H:
        raise ValueError(f"Expected A_log shape [{H}].")
    if dt_bias.ndim != 1 or dt_bias.shape[0] != H * K:
        raise ValueError(f"Expected dt_bias shape [{H * K}].")

    state_s = W - 1 + num_spec
    if cs_q.ndim != 3 or cs_k.ndim != 3 or cs_v.ndim != 3:
        raise ValueError("Expected cs_q/cs_k/cs_v to have shape [pool, dim, S].")
    if cs_q.shape[1] != H * K or cs_k.shape[1] != H * K:
        raise ValueError(f"Expected cs_q/cs_k shape [pool, {H * K}, S].")
    if cs_v.shape[1] != HV * V:
        raise ValueError(f"Expected cs_v shape [pool, {HV * V}, S].")
    if cs_q.shape[2] < state_s or cs_k.shape[2] < state_s or cs_v.shape[2] < state_s:
        raise ValueError(f"Expected conv-state S dimension to be at least {state_s}.")
    if cs_q.stride(1) != 1 or cs_k.stride(1) != 1 or cs_v.stride(1) != 1:
        raise ValueError(
            "Expected cs_q/cs_k/cs_v to use dim-contiguous layout "
            "(allocate as [pool, S, dim] and transpose(1, 2))."
        )

    pool_size = recurrent_state.shape[0]
    if recurrent_state.ndim != 4 or recurrent_state.shape[1:] != (HV, V, K):
        raise ValueError(
            f"Expected recurrent_state shape [pool, {HV}, {V}, {K}] (V-first pool layout)."
        )
    if recurrent_state.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"recurrent_state must be fp32 or bf16, got {recurrent_state.dtype}")
    if k_cache.ndim != 3 or k_cache.shape[1:] != (num_spec, H * K):
        raise ValueError(f"Expected k_cache shape [pool, {num_spec}, {H * K}].")
    if g_cache.ndim != 3 or g_cache.shape[1:] != (num_spec, H * K):
        raise ValueError(f"Expected g_cache shape [pool, {num_spec}, {H * K}].")
    if g_cache.dtype != torch.float32:
        raise ValueError("g_cache (the replayed gates) must be fp32.")
    if v_cache.ndim != 3 or v_cache.shape[1:] != (num_spec, HV * V):
        raise ValueError(f"Expected v_cache shape [pool, {num_spec}, {HV * V}].")
    if beta_cache.ndim != 3 or beta_cache.shape[1:] != (num_spec, HV):
        raise ValueError(f"Expected beta_cache shape [pool, {num_spec}, {HV}].")
    if (
        k_cache.shape[0] < pool_size
        or g_cache.shape[0] < pool_size
        or v_cache.shape[0] < pool_size
        or beta_cache.shape[0] < pool_size
    ):
        raise ValueError("Expected cache pool dimensions to cover recurrent_state rows.")

    if ssm_state_indices.ndim != 1 or num_accepted_tokens.ndim != 1:
        raise ValueError("Expected ssm_state_indices and num_accepted_tokens to be 1D.")
    num_requests = ssm_state_indices.shape[0]
    if T_total != num_requests * (num_spec + 1):
        raise ValueError("Expected exactly num_spec + 1 new-token rows per request.")
    if num_accepted_tokens.shape[0] < pool_size:
        raise ValueError("Expected slot-indexed num_accepted_tokens to cover the state pool.")


def _fits_32bit_stride(tensor: torch.Tensor) -> bool:
    int32_max = 2**31 - 1
    max_offset = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride()):
        stride = abs(int(stride))
        if stride > int32_max:
            return False
        if size:
            max_offset += (int(size) - 1) * stride
            if max_offset > int32_max:
                return False
    return True


def _from_dlpack_arg(tensor: torch.Tensor, *, assumed_align: int):
    return from_dlpack(
        tensor,
        assumed_align=assumed_align,
        use_32bit_stride=_fits_32bit_stride(tensor),
    )


def _beta_cache_assumed_align(beta_cache: torch.Tensor) -> int:
    """Return the alignment shared by KDA per-layer beta-cache views.

    The producer allocates ``[layers, slots, num_spec, local_heads]``. After
    selecting a layer, ``slots * stride(0)`` is the physical layer span even
    when the head rows are padded. Combine that span with the current pointer
    and dtype, capped at the CuTe bridge's useful 16-byte guarantee.
    """
    layer_span_bytes = beta_cache.shape[0] * beta_cache.stride(0) * beta_cache.element_size()
    return gcd(16, beta_cache.data_ptr(), layer_span_bytes)


def _dlpack_arg(tensor: torch.Tensor, *, assumed_align: int):
    # Alignment is deliberately mandatory: layout dynamism does not imply
    # arbitrary pointer alignment. Each call site must state the guarantee
    # provided by that tensor's producer and view pattern.
    arg = _from_dlpack_arg(tensor, assumed_align=assumed_align)
    for dim, stride in enumerate(tensor.stride()):
        if stride == 1:
            return arg.mark_layout_dynamic(dim)
    return arg.mark_layout_dynamic()


def _layout_key(tensor: torch.Tensor, *, assumed_align: int):
    arg = _dlpack_arg(tensor, assumed_align=assumed_align)
    shape_mask = arg.dynamic_shapes_mask
    stride_mask = arg.dynamic_strides_mask
    shape = tuple(None if dynamic else size for size, dynamic in zip(tensor.shape, shape_mask))
    stride = tuple(
        None if dynamic else value for value, dynamic in zip(tensor.stride(), stride_mask)
    )
    return (tensor.dtype, shape, stride, _fits_32bit_stride(tensor), assumed_align)


def _vec4_rows(tensor: torch.Tensor, dim: int = -1) -> bool:
    """True when every group of four consecutive elements along the unit-stride dimension `dim`
    that starts at a multiple of four is one aligned 8 B (bf16) or 16 B (fp32) access: the v2
    kernel moves its per-token rows with such vector loads and stores."""
    if tensor.dtype not in (torch.bfloat16, torch.float32) or tensor.stride(dim) != 1:
        return False
    if tensor.data_ptr() % (4 * tensor.element_size()):
        return False
    return all(
        stride % 4 == 0 for d, stride in enumerate(tensor.stride()) if d != dim % tensor.ndim
    )


def _verify_operands_ok(*, w_q, w_k, w_v, state, projections, rows, conv_caches, g_cache) -> bool:
    """Operands of the runtime verify contract that ``kda_decode_mtp_v2_kernel`` and
    ``kda_decode_mtp_2cta_kernel`` take.

    Layouts for their vector accesses (see `_vec4_rows`): token rows and replay caches
    vectorized along their last dimension, the dim-contiguous conv caches along the channel
    dimension, conv weights as contiguous [channel][4] rows. Dtypes: fp32 or bf16 state, bf16
    new-token projections and conv caches, fp32 replay gates.
    """
    weights_ok = all(
        w.dtype == w_q.dtype and w.is_contiguous() and w.shape[1] == 4 and _vec4_rows(w)
        for w in (w_q, w_k, w_v)
    )
    return (
        weights_ok
        and all(_vec4_rows(t) for t in rows)
        and all(_vec4_rows(t, dim=1) for t in conv_caches)
        and state.dtype in (torch.float32, torch.bfloat16)
        and all(t.dtype == torch.bfloat16 for t in (*projections, *conv_caches))
        and g_cache.dtype == torch.float32
    )


_num_sms_cache = {}


def _num_sms(device: torch.device) -> int:
    dev = torch.device(device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    if index not in _num_sms_cache:
        _num_sms_cache[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _num_sms_cache[index]


def _select_verify_kernel(
    *,
    num_spec: int,
    num_tiles: int,
    num_sms: int,
    operands_ok: Callable[[], bool],
) -> Literal["v2", "two_cta"]:
    """The verify kernel a call runs, by the rule in the module docstring. A call with more than
    ``MAX_NUM_SPEC`` drafts, or whose operands fail ``operands_ok`` (`_verify_operands_ok`,
    evaluated last), is rejected."""
    if num_spec > MAX_NUM_SPEC:
        raise ValueError(f"KDA MTP verify: at most {MAX_NUM_SPEC} drafts per request")
    if not operands_ok():
        raise ValueError(
            "KDA MTP verify: the packed, fused-norm verify takes 16-byte-vectorizable bf16 "
            "projections and conv caches, contiguous conv weights, fp32 replay gates and an fp32 "
            "or bf16 state"
        )
    if num_tiles > num_sms:
        return "two_cta"
    return "v2"


def _try_flatten_args(
    *,
    recurrent_state: torch.Tensor,
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    T_total: int,
    H: int,
    HV: int,
    K: int,
    V: int,
) -> Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        h0 = recurrent_state.view(-1, V, K)
        x_q_flat = x_q.view(1, T_total, H * K)
        x_k_flat = x_k.view(1, T_total, H * K)
        x_v_flat = x_v.view(1, T_total, HV * V)
    except RuntimeError:
        return False, recurrent_state, x_q, x_k, x_v
    return True, h0, x_q_flat, x_k_flat, x_v_flat


# Layout-and-constexpr-keyed compile cache. Request count and packed-token
# length are dynamic; batches sharing the same kernel variant reuse one
# artifact even when their launch grid and token-buffer extents differ.
_compiled_cache = {}


def kda_mtp_decode_impl(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    cs_q: torch.Tensor,
    cs_k: torch.Tensor,
    cs_v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    v_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_spec: int,
    num_accepted_tokens: torch.Tensor,
    lower_bound: float,
    scale: Optional[float],
    onorm_g: torch.Tensor,
    onorm_weight: torch.Tensor,
    onorm_eps: float,
    quantize_output: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Launch the fused KDA MTP verify kernel. See the module docstring.

    Args (device tensors unless noted):
        x_q/x_k/x_v: post-projection, pre-conv token states
            ``[1, T_total, H, 128]`` bf16 — new tokens only, packed uniformly by
            request, ``1 + num_spec`` rows each.
        w_q/w_k/w_v: conv weights ``[H*128, W]`` fp32, width-contiguous.
        cs_q/cs_k/cs_v: extended conv caches ``[pool, H*128, >= W-1+M]``
            of raw projection inputs, bf16 (the runtime's caches), dim-contiguous.
            Columns ``[0, W-1)`` are the committed window; tail columns hold raw
            pending-draft inputs. Mutated.
        g, beta: raw gate ``[1, T, H, 128]`` and beta ``[1, T, H]`` bf16.
        A_log, dt_bias: fp32 ``[H]`` / ``[H*128]``.
        recurrent_state: pool ``[pool, H, V, K]`` fp32 or bf16, **V-first** layout
            (matches the executor ssm pool and the single-token decode
            kernel). Committed in place.
        k_cache/g_cache/v_cache/beta_cache: replay caches ``[pool, M, H*K]`` (k,
            fp32 or bf16) / ``[pool, M, H*K]`` (gate, fp32) / ``[pool, M, H*V]`` /
            ``[pool, M, H]`` (fp32 or bf16). Mutated.
        ssm_state_indices: per-request state-pool slots; ``num_accepted_tokens``
            holds the accepted counts by slot.
        onorm_g, onorm_weight, onorm_eps: the gated RMSNorm of the output, applied in
            the CuTe epilogue (``onorm_g`` may be section-strided).
        quantize_output: return the normalized activation as the catalog
            ``mxfp8_quantize``'s MXFP8 form: one UE8M0 scale per 32 values (four
            distinct scale bytes per (row, head)) with that quantizer's
            instructions, so the codes and the scales of the rows are bit for bit
            what ``mxfp8_quantize(out, True, 32)`` makes of the BF16 output. The
            kernel is the one the BF16 output takes (both have the epilogue).

    Returns either BF16 ``[1, T_total, H, V]`` or, with
    ``quantize_output=True``, E4M3 ``[T_total, H*V]`` and packed UE8M0 scales
    in the flat uint8 R128c4 layout consumed by the CuTe MXFP8 GEMM.
    """
    _, T_total, _, D = x_q.shape
    H = A_log.shape[0]
    HV = g.shape[2] if g.ndim == 4 else H
    K = D
    V_dim = x_v.shape[-1]
    W = w_q.shape[1]
    if K != TILE_K or V_dim != 128 or W != 4:
        raise ValueError("specialized kernel expects K=128, V=128, W=4")
    if HV != H:
        raise ValueError("specialized kernel expects HV == H")
    if scale is None:
        scale = K**-0.5

    N = ssm_state_indices.shape[0]
    if quantize_output:
        out = torch.empty(
            (T_total, HV * V_dim),
            dtype=torch.float8_e4m3fn,
            device=x_q.device,
        )
        output_scale = torch.empty(
            ((T_total + 127) // 128 * 128 * HV * 4,),
            dtype=torch.uint8,
            device=x_q.device,
        )
    else:
        # The output contains new-token rows only, and the kernel writes every head/value
        # element: no zero-fill.
        out = torch.empty((1, T_total, HV, V_dim), dtype=x_q.dtype, device=x_q.device)
    if num_accepted_tokens.dtype != torch.int32:
        num_accepted_tokens = num_accepted_tokens.to(torch.int32)
    _require_stride_layout(
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        cs_q=cs_q,
        cs_k=cs_k,
        cs_v=cs_v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        k_cache=k_cache,
        g_cache=g_cache,
        v_cache=v_cache,
        beta_cache=beta_cache,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        out=out,
        onorm_g=onorm_g,
        onorm_weight=onorm_weight,
        H=H,
        HV=HV,
        K=K,
        V=V_dim,
        W=W,
        num_spec=num_spec,
        T_total=T_total,
        quantize_output=quantize_output,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    use_flat_layout, h0_arg, x_q_arg, x_k_arg, x_v_arg = _try_flatten_args(
        recurrent_state=recurrent_state,
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        T_total=T_total,
        H=H,
        HV=HV,
        K=K,
        V=V_dim,
    )
    beta_cache_assumed_align = _beta_cache_assumed_align(beta_cache)
    # Two CTAs per (head, request) while they fit in one wave (see SPLIT_MAX_TILES).
    split_v = 2 if HV * N <= SPLIT_MAX_TILES else 1

    kernel = _select_verify_kernel(
        num_spec=num_spec,
        num_tiles=HV * N,
        num_sms=_num_sms(x_q.device),
        operands_ok=lambda: _verify_operands_ok(
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            state=h0_arg,
            projections=(x_q, x_k, x_v),
            rows=(x_q_arg, x_k_arg, x_v_arg, g, onorm_g, k_cache, g_cache, v_cache),
            conv_caches=(cs_q, cs_k, cs_v),
            g_cache=g_cache,
        ),
    )
    if quantize_output:
        logger.info_once(
            "Kimi K3.1 Rubin MXFP8 hand-off: rubin_kda_mtp_verify writes o_proj's operand "
            "in the catalog quantizer's 1x32 form (no standalone quantization)",
            key="kda_mtp_verify quantize_output",
        )
    # Shared operands of kda_decode_mtp_v2_kernel and kda_decode_mtp_2cta_kernel.
    args = [
        (h0_arg, 16),
        (x_q_arg, 16),
        (x_k_arg, 16),
        (x_v_arg, 16),
        (w_q, 16),
        (w_k, 16),
        (w_v, 16),
        (cs_q, 16),
        (cs_k, 16),
        (cs_v, 16),
        (A_log, 16),
        (g, 16),
        (dt_bias, 16),
        (beta, 16),
        (onorm_g, 16),
        (onorm_weight, 16),
        (out, 16),
        (k_cache, 16),
        (g_cache, 16),
        (v_cache, 16),
        (beta_cache, beta_cache_assumed_align),
        (ssm_state_indices, 4),
        (num_accepted_tokens, 4),
    ]
    if quantize_output:
        if kernel == "v2":
            # The BF16 output is not written (the output-gate view stands in for it).
            args[16] = (onorm_g, 16)
        # E4M3 values [T, HV * V] and R128c4 UE8M0 scales of the MXFP8 output
        args += [(out, 16), (output_scale, 16)]
    else:
        # Placeholders of the MXFP8 output operands
        args += [(out, 16), (out, 16)]
    if kernel == "two_cta":
        runner = _run_kda_decode_mtp_2cta
        variant = {}
    else:
        runner = _run_kda_decode_mtp_v2
        variant = {"SPLIT_V": split_v}
    constexprs = dict(
        scale=scale,
        HV=HV,
        K=K,
        V=V_dim,
        NUM_SPEC=num_spec,
        KERNEL_WIDTH=W,
        lower_bound=float(lower_bound),
        onorm_eps=float(onorm_eps),
        USE_FLAT_LAYOUT=use_flat_layout,
        MXFP8_OUT=bool(quantize_output),
        **variant,
    )
    key = (
        kernel,
        tuple(_layout_key(t, assumed_align=a) for t, a in args),
        tuple(sorted(constexprs.items())),
    )
    if key not in _compiled_cache:
        logger.info(f"kda_mtp_decode: compiling {kernel} variant {constexprs}")
        _compiled_cache[key] = cute.compile(
            runner,
            *[_dlpack_arg(t, assumed_align=a) for t, a in args],
            N,
            **constexprs,
            stream=stream,
        )
    _compiled_cache[key](*[_dlpack_arg(t, assumed_align=a) for t, a in args], N, stream)
    if quantize_output:
        return out, output_scale
    return out
