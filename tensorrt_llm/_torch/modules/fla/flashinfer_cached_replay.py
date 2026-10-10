# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer ring-buffer adapter for GDN speculative-decoding replay.

Wraps FlashInfer's ``gated_delta_rule_mtp_ucache_flush``, which plays the role of
``cached_replay.fused_recurrent_gated_delta_rule_cached_replay_update`` but keeps
the replay history in a 32-row circular buffer per request instead of a
two-buffer layout. Differences from the Triton path that callers must handle:
  * History length and ring start are passed per batch row, not per slot:
    gather them with ``state_indices`` before the call.
  * The kernel only reads the ring cursors. The caller advances them once per
    step after acceptance: on flush ``base = (base + len) % 32; len = accepted``,
    otherwise ``len += accepted``.
  * Block-strided state pools need FlashInfer's strided mode, which also
    requires q/k/v to be column slices of one packed buffer.
"""

from typing import Optional

import torch

from tensorrt_llm._utils import is_flashinfer_gdn_decode_supported_arch

FLASHINFER_REPLAY_RING_SIZE = 32

try:
    import flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_flush as _fi_ring_module
    from flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_flush import (
        gated_delta_rule_mtp_ucache_flush as _fi_ring_replay,
    )

    # Strided mode is required for block-strided state pools. Enable it via the
    # module flag FlashInfer reads on every call, so it works regardless of
    # import order. It only applies when q/k/v share a token stride.
    _FLASHINFER_RING_REPLAY_AVAILABLE = (
        hasattr(_fi_ring_module, "_STRIDED_QKV")
        and _fi_ring_module.RING_SLOTS == FLASHINFER_REPLAY_RING_SIZE
    )
    if _FLASHINFER_RING_REPLAY_AVAILABLE:
        _fi_ring_module._STRIDED_QKV = True
except (ImportError, RuntimeError):
    _FLASHINFER_RING_REPLAY_AVAILABLE = False


def is_flashinfer_cached_replay_available() -> bool:
    """Whether the FlashInfer ring replay kernel is importable on this GPU."""
    return _FLASHINFER_RING_REPLAY_AVAILABLE and is_flashinfer_gdn_decode_supported_arch()


def _align16(t: torch.Tensor) -> torch.Tensor:
    # FlashInfer requires every tensor to start at a 16-byte aligned address.
    # ``a`` starts HV bf16 values into the in_proj_ba output (e.g. byte 8 when
    # HV=4), and index tensors may be slices of larger buffers. Copy into fresh
    # (aligned) storage only when misaligned.
    if t.data_ptr() % 16 != 0:
        return t.clone(memory_format=torch.contiguous_format)
    return t


@torch.compiler.disable
def flashinfer_cached_replay_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    state_indices: torch.Tensor,
    k_ring: torch.Tensor,
    u_ring: torch.Tensor,
    g_ring: torch.Tensor,
    hist_len: torch.Tensor,
    cache_base: torch.Tensor,
    history_size: int,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Verify T draft tokens per request using the ring-buffer replay history.

    Args:
        q, k: ``[N, T, H, K]``. v: ``[N, T, HV, V]``.
        a, b: raw gating inputs ``[N, T, HV]``; gating is applied in-kernel
            from ``A_log`` and ``dt_bias`` (``[HV]``).
        ssm_states: checkpoint pool ``[slots, HV, V, K]``; dim 0 may be strided.
        state_indices: ``[N]`` slot per request; ``-1`` rows are skipped.
        k_ring, u_ring, g_ring: ``[slots, H, 32, K]``, ``[slots, HV, 32, V]``,
            ``[slots, HV, 32]`` fp32.
        hist_len, cache_base: ``[N]`` int32 per-row history length and ring
            start, gathered from the per-slot values.
        history_size: rows the history may hold before it is folded into
            ``ssm_states``.
        output: preallocated ``[N, T, HV, V]``; pass it under CUDA graphs.

    Returns:
        ``output`` of shape ``[N, T, HV, V]``.
    """
    N, T = q.shape[0], q.shape[1]
    HV, V = v.shape[2], v.shape[3]
    if output is None:
        output = q.new_empty(N, T, HV, V)
    # a/b are column slices of in_proj_ba, so each token's values are 2*HV
    # apart. Strided mode (q/k/v sharing a token stride) reads them in place;
    # non-strided mode needs them tightly packed.
    if not (q.stride(1) == k.stride(1) == v.stride(1)):
        a, b = a.contiguous(), b.contiguous()
    _fi_ring_replay(
        A_log=A_log,
        a=_align16(a),
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=_align16(b),
        initial_state_source=ssm_states,
        initial_state_indices=_align16(state_indices.int()),
        use_qk_l2norm_in_kernel=True,
        scale=scale,
        output=output,
        k_cache=k_ring,
        u_cache=u_ring,
        g_cache=g_ring,
        hist_len=_align16(hist_len),
        cache_base=_align16(cache_base),
        # Flush when the T new rows no longer fit, matching the Triton
        # kernel's ``pnat + T > history_size``.
        flush_min=history_size - T + 1,
        # The caller advances the cursors once per step, after acceptance.
        restart_hist_on_flush=False,
    )
    return output
