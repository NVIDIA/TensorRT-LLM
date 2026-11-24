# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA verification support of the generation target: the in-tree multi-token verify op loader and the
replay bookkeeping of the KDA cache manager."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional

import torch
import triton
import triton.language as tl

try:
    from tensorrt_llm._utils import get_sm_version as _tllm_get_sm_version
except ImportError:  # pragma: no cover — source-loader stub path
    _tllm_get_sm_version = None


def _default_get_sm_version() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return -1
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor


def get_kda_sm_version() -> int:
    """Return the runtime SM version used for KDA kernel selection.

    Prefers ``tensorrt_llm._utils.get_sm_version`` when the real package is
    importable so environment-side overrides propagate. Falls back to a
    plain CUDA-property probe when we are executing under the source-loader
    stub subtree.
    """
    if _tllm_get_sm_version is not None:
        try:
            return int(_tllm_get_sm_version())
        except RuntimeError:
            # torch raises RuntimeError when no CUDA device is usable;
            # the property probe below handles that case itself.
            return _default_get_sm_version()
    return _default_get_sm_version()


def is_kda_optimized_supported() -> bool:
    """The optimized prefill/decode kernels need Blackwell sm_100 or Rubin sm_107."""
    return get_kda_sm_version() in (100, 103, 107)


# ---------------------------------------------------------------------------
# In-tree KDA multi-token verify op (CuTe DSL, trtllm::k31r_kda_mtp_decode).
# ---------------------------------------------------------------------------

_MTP_MODULE: Optional[ModuleType] = None
_MTP_IMPORT_ERROR: Optional[Exception] = None


def _load_mtp_module() -> ModuleType:
    """Import the in-tree MTP verify custom-op module (registers the op)."""
    global _MTP_MODULE, _MTP_IMPORT_ERROR
    if _MTP_MODULE is not None:
        return _MTP_MODULE
    if _MTP_IMPORT_ERROR is not None:
        raise _MTP_IMPORT_ERROR
    try:
        module = importlib.import_module(".rubin_ull_kda_cute_dsl_kimi_k3_kda_mtp_ops", __package__)
    except Exception as exc:  # typically ImportError when CuTe DSL is unavailable
        _MTP_IMPORT_ERROR = exc
        raise
    _MTP_MODULE = module
    return module


def is_intree_mtp_available() -> bool:
    """True when the in-tree CuTe DSL MTP verify op can be imported."""
    try:
        _load_mtp_module()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# KDA replay bookkeeping and fused-verify sizing, copied from
# tensorrt_llm/_torch/modules/kimi_kda/_kda_kernels.py (kernels/SOURCE.json). The one change:
# commit_kda_pending_drafts' replay from the replay caches (no draft records) loads the op module
# it used there through _load_commit_module.
# ---------------------------------------------------------------------------


def _load_commit_module() -> ModuleType:
    """The fused-verify op module of the replay from the replay caches (``kda_mtp_commit_pending_drafts``)."""
    return importlib.import_module("tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops")


@triton.jit
def _copy_kda_replay_conv_window_kernel(
    conv_ptr,
    q_cache_ptr,
    k_cache_ptr,
    v_cache_ptr,
    state_indices_ptr,
    conv_stride_slot,
    conv_stride_dim,
    conv_stride_window,
    cache_stride_slot,
    cache_stride_dim,
    cache_stride_window,
    PROJECTION_SIZE: tl.constexpr,
    COMMITTED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    request = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < PROJECTION_SIZE * COMMITTED
    dim = offsets // COMMITTED
    window = offsets % COMMITTED
    slot = tl.load(state_indices_ptr + request).to(tl.int64)

    conv_offset = slot * conv_stride_slot + dim * conv_stride_dim + window * conv_stride_window
    cache_offset = slot * cache_stride_slot + dim * cache_stride_dim + window * cache_stride_window
    section_offset = PROJECTION_SIZE * conv_stride_dim
    tl.store(q_cache_ptr + cache_offset, tl.load(conv_ptr + conv_offset, mask=mask), mask=mask)
    tl.store(
        k_cache_ptr + cache_offset,
        tl.load(conv_ptr + conv_offset + section_offset, mask=mask),
        mask=mask,
    )
    tl.store(
        v_cache_ptr + cache_offset,
        tl.load(conv_ptr + conv_offset + 2 * section_offset, mask=mask),
        mask=mask,
    )


def copy_kda_replay_conv_window(
    conv_pool: torch.Tensor,
    q_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    """Copy selected packed ``W - 1`` rows into KDA replay caches.

    The live convolution pool is ``[slots, 3D, W - 1]``. Replay caches are
    ``[slots, D, W - 1 + num_spec]`` with a dim-contiguous inner layout.
    """
    if conv_pool.ndim != 3 or conv_pool.shape[1] % 3:
        raise ValueError(f"Expected packed rank-3 KDA convolution pool, got {conv_pool.shape}")
    projection_size = conv_pool.shape[1] // 3
    committed = conv_pool.shape[2]
    caches = (q_cache, k_cache, v_cache)
    expected_prefix = (conv_pool.shape[0], projection_size)
    if any(cache.ndim != 3 or cache.shape[:2] != expected_prefix for cache in caches):
        raise ValueError("KDA replay convolution caches do not match the live pool geometry")
    if any(cache.shape[2] < committed for cache in caches):
        raise ValueError("KDA replay convolution caches are shorter than the committed window")
    if any(cache.stride() != q_cache.stride() for cache in caches[1:]):
        raise ValueError("KDA replay convolution caches must share one layout")
    if state_indices.ndim != 1 or state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("KDA replay state indices must be a rank-1 int32 or int64 tensor")
    if any(tensor.device != conv_pool.device for tensor in (*caches, state_indices)):
        raise ValueError("KDA replay convolution tensors must be on one device")
    if state_indices.numel() == 0:
        return

    block_size = 256
    grid = (
        state_indices.numel(),
        triton.cdiv(projection_size * committed, block_size),
    )
    with torch.cuda.device(conv_pool.device.index):
        _copy_kda_replay_conv_window_kernel[grid](
            conv_pool,
            q_cache,
            k_cache,
            v_cache,
            state_indices,
            conv_pool.stride(0),
            conv_pool.stride(1),
            conv_pool.stride(2),
            q_cache.stride(0),
            q_cache.stride(1),
            q_cache.stride(2),
            PROJECTION_SIZE=projection_size,
            COMMITTED=committed,
            BLOCK_SIZE=block_size,
        )


@triton.jit
def _store_kda_replay_conv_window_kernel(
    conv_ptr,
    q_cache_ptr,
    k_cache_ptr,
    v_cache_ptr,
    state_indices_ptr,
    num_accepted_ptr,
    conv_stride_slot,
    conv_stride_dim,
    conv_stride_window,
    cache_stride_slot,
    cache_stride_dim,
    cache_stride_window,
    PROJECTION_SIZE: tl.constexpr,
    COMMITTED: tl.constexpr,
    NUM_SPEC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    request = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < PROJECTION_SIZE * COMMITTED
    dim = offsets // COMMITTED
    window = offsets % COMMITTED
    slot = tl.load(state_indices_ptr + request).to(tl.int64)
    # The window after the accepted drafts starts at their count, clamped
    # like the verify kernels clamp it.
    shift = tl.minimum(tl.maximum(tl.load(num_accepted_ptr + request), 0), NUM_SPEC)

    conv_offset = slot * conv_stride_slot + dim * conv_stride_dim + window * conv_stride_window
    cache_offset = (
        slot * cache_stride_slot + dim * cache_stride_dim + (window + shift) * cache_stride_window
    )
    section_offset = PROJECTION_SIZE * conv_stride_dim
    tl.store(conv_ptr + conv_offset, tl.load(q_cache_ptr + cache_offset, mask=mask), mask=mask)
    tl.store(
        conv_ptr + conv_offset + section_offset,
        tl.load(k_cache_ptr + cache_offset, mask=mask),
        mask=mask,
    )
    tl.store(
        conv_ptr + conv_offset + 2 * section_offset,
        tl.load(v_cache_ptr + cache_offset, mask=mask),
        mask=mask,
    )


def store_kda_replay_conv_window(
    conv_pool: torch.Tensor,
    q_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> None:
    """Store each selected slot's conv window after its accepted drafts in
    the live convolution pool.

    The replay caches hold the window before the golden token in columns
    ``[0, W - 1)`` and the raw inputs of the round's drafts after it, so the
    ``W - 1`` columns starting at the request's ``num_accepted_tokens`` entry
    are the window the next verify would start from. They are raw
    convolution inputs, so storing them in a bf16 pool is exact.
    """
    projection_size = conv_pool.shape[1] // 3
    committed = conv_pool.shape[2]
    caches = (q_cache, k_cache, v_cache)
    if any(cache.shape[1] != projection_size or cache.shape[2] < committed for cache in caches):
        raise ValueError("KDA replay convolution caches do not match the live pool geometry")
    if any(cache.stride() != q_cache.stride() for cache in caches[1:]):
        raise ValueError("KDA replay convolution caches must share one layout")
    if num_accepted_tokens.shape != state_indices.shape:
        raise ValueError("Expected one accepted-draft count per state index")
    if state_indices.numel() == 0:
        return

    block_size = 256
    grid = (
        state_indices.numel(),
        triton.cdiv(projection_size * committed, block_size),
    )
    with torch.cuda.device(conv_pool.device.index):
        _store_kda_replay_conv_window_kernel[grid](
            conv_pool,
            q_cache,
            k_cache,
            v_cache,
            state_indices,
            num_accepted_tokens,
            conv_pool.stride(0),
            conv_pool.stride(1),
            conv_pool.stride(2),
            q_cache.stride(0),
            q_cache.stride(1),
            q_cache.stride(2),
            PROJECTION_SIZE=projection_size,
            COMMITTED=committed,
            NUM_SPEC=q_cache.shape[2] - committed,
            BLOCK_SIZE=block_size,
        )


@triton.jit
def _replay_kda_token_records_kernel(
    state_ptr,
    records_ptr,
    state_indices_ptr,
    num_accepted_ptr,
    state_stride_slot,
    records_stride_slot,
    NUM_SPEC: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    rows = tl.program_id(2) * BLOCK_V + tl.arange(0, BLOCK_V)
    keys = tl.arange(0, K)
    slot = tl.load(state_indices_ptr + request).to(tl.int64)
    accepted = tl.minimum(tl.maximum(tl.load(num_accepted_ptr + request), 0), NUM_SPEC)
    state_ptrs = (
        state_ptr + slot * state_stride_slot + (head * V + rows[:, None]) * K + keys[None, :]
    )
    state = tl.load(state_ptrs)
    records = records_ptr + slot * records_stride_slot
    for t in range(0, accepted):
        vn = tl.load(records + (t * H + head) * V + rows)
        beta_k = tl.load(records + NUM_SPEC * H * V + (t * H + head) * K + keys)
        decay = tl.load(records + NUM_SPEC * H * (V + K) + (t * H + head) * K + keys)
        state = tl.fma(decay[None, :], state, vn[:, None] * beta_k[None, :])
    tl.store(state_ptrs, state)


def replay_kda_token_records(
    recurrent_state: torch.Tensor,
    state_tok: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> None:
    """Replay each selected slot's accepted drafts onto its pool state from
    the per-draft records the Kimi K3 verify kernels keep in ``state_tok``.

    A slot of ``state_tok`` (``[slots, 3, num_spec, H, K]`` fp32) holds the
    records of the last round's drafts: the row innovations ``vn``
    ``[num_spec][H][V]``, then ``beta * k`` and the decay
    ``[num_spec][H][K]`` (V = K). Draft ``t`` updates the state as the
    kernels do, ``S = fma(decay, S, vn (beta * k))``, so the result is
    bit-identical to the state their next verify starts from.
    """
    _, num_spec, num_heads, k_dim = state_tok.shape[1:]
    v_dim = recurrent_state.shape[-2]
    if state_tok.shape[1] != 3 or v_dim != k_dim:
        raise ValueError(
            "replay_kda_token_records: expected the drafts' records "
            "[slots, 3, num_spec, H, K] with V = K, got "
            f"{tuple(state_tok.shape)} for a state of "
            f"{tuple(recurrent_state.shape)}"
        )
    if state_indices.numel() == 0:
        return
    block_v = 32
    grid = (state_indices.numel(), num_heads, v_dim // block_v)
    with torch.cuda.device(recurrent_state.device.index):
        _replay_kda_token_records_kernel[grid](
            recurrent_state,
            state_tok,
            state_indices,
            num_accepted_tokens,
            recurrent_state.stride(0),
            state_tok.stride(0),
            NUM_SPEC=num_spec,
            H=num_heads,
            V=v_dim,
            K=k_dim,
            BLOCK_V=block_v,
        )


def commit_kda_pending_drafts(
    recurrent_state: torch.Tensor,
    conv_pool: torch.Tensor,
    conv_q: torch.Tensor,
    conv_k: torch.Tensor,
    conv_v: torch.Tensor,
    k_cache: Optional[torch.Tensor],
    g_cache: Optional[torch.Tensor],
    v_cache: Optional[torch.Tensor],
    beta_cache: Optional[torch.Tensor],
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_tok: Optional[torch.Tensor] = None,
) -> None:
    """Commit one layer's pending accepted drafts into its live pools.

    ``num_accepted_tokens[n]`` is the number of drafts of slot
    ``state_indices[n]`` the last verify accepted, which the next verify
    would take up first. Afterwards ``recurrent_state`` and ``conv_pool``
    hold the state and the conv window that verify would start from (the
    window bit for bit), and the caller clears the slot's count:

    * with the Kimi K3 verify kernels' draft records (``state_tok``), the
      drafts are replayed from those (``replay_kda_token_records``), and the
      replay caches, which are then not allocated, may be None;
    * otherwise they are replayed from the replay caches by the fused verify
      kernel's own replay steps (``kda_mtp_commit_pending_drafts``). The
      tensor-core verify computes a chunk in one pass, so this commit
      rounds the state differently from a verify that replays the drafts
      itself.
    """
    num_spec = conv_q.shape[2] - conv_pool.shape[2]
    if state_tok is not None:
        replay_kda_token_records(recurrent_state, state_tok, state_indices, num_accepted_tokens)
    else:
        assert (
            k_cache is not None
            and g_cache is not None
            and v_cache is not None
            and beta_cache is not None
        )
        _load_commit_module().kda_mtp_commit_pending_drafts(
            recurrent_state,
            conv_q,
            conv_k,
            conv_v,
            k_cache,
            g_cache,
            v_cache,
            beta_cache,
            state_indices,
            num_accepted_tokens,
            num_spec,
            conv_width=conv_pool.shape[2] + 1,
        )
    store_kda_replay_conv_window(
        conv_pool, conv_q, conv_k, conv_v, state_indices, num_accepted_tokens
    )


def is_kda_mtp_verify_available() -> bool:
    """True when the fused multi-token verify kernel can run here.

    Used by the executor (cache-manager sizing) to decide whether to
    allocate the KDA replay caches instead of the legacy intermediate
    verification buffers.
    """
    return is_kda_optimized_supported() and is_intree_mtp_available()


def kda_mtp_max_num_spec() -> int:
    """Largest draft count the fused multi-token verify kernel supports."""
    from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda.kda_mtp_decode import (
        MAX_NUM_SPEC,
    )

    return MAX_NUM_SPEC


def kda_mtp_replay_num_spec(num_spec: int) -> Optional[int]:
    """Draft count to size the KDA replay caches for, or ``None``.

    ``None`` keeps the legacy intermediate verification buffers, so the
    module runs the sequential FLA verify: either the fused kernel cannot
    run here, or ``num_spec`` exceeds what it supports.
    """
    if not is_kda_mtp_verify_available():
        return None
    max_num_spec = kda_mtp_max_num_spec()
    if num_spec > max_num_spec:
        from tensorrt_llm.logger import logger

        logger.warning_once(
            f"KDA fused multi-token verify supports at most {max_num_spec} draft "
            f"tokens, got {num_spec}; using the sequential FLA verify.",
            key="kda_mtp_num_spec_fallback",
        )
        return None
    return num_spec
