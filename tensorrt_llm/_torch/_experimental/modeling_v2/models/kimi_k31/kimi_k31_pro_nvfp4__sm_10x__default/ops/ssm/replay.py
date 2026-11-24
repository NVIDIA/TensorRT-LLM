# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA FlashInfer prefill, native decode/verification and portable FLA dispatch."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional

import torch
import triton
import triton.language as tl

from ....ops.ssm.store_kda_replay_conv_window import store_kda_replay_conv_window

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
    """The target runs optimized prefill/decode on SM100, SM103, and SM107."""
    return get_kda_sm_version() in (100, 103, 107)


def fused_kda_post_conv(
    packed: torch.Tensor,
    num_heads: int,
    head_dim: int,
    l2_norm_eps: float = 1e-6,
    *,
    normalize_qk: bool = True,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Portable layout conversion for the unfused convolution fallback.

    The production FlashInfer path produces this layout directly in the
    convolution kernel and normalizes Q/K within the recurrent kernel.
    """
    if packed.ndim != 2 or packed.shape[0] != 3 * num_heads * head_dim:
        raise ValueError("Expected packed [3 * H * D, T] convolution output")
    if packed.shape[1] > 1 and 1 not in packed.stride():
        raise ValueError("Packed KDA post-conv requires a unit stride along tokens or features")
    qkv = tuple(
        t.t().reshape(1, packed.shape[1], num_heads, head_dim).contiguous()
        for t in packed.chunk(3, dim=0)
    )
    if normalize_qk:
        q, k, v = qkv

        def normalize(t):
            f = t.float()
            return (f * torch.rsqrt((f * f).sum(-1, keepdim=True) + l2_norm_eps)).to(t.dtype)

        qkv = (normalize(q), normalize(k), v)
    if out is not None:
        if len(out) != 3 or any(
            dst.shape != src.shape
            or dst.dtype != src.dtype
            or dst.device != src.device
            or not dst.is_contiguous()
            for dst, src in zip(out, qkv)
        ):
            raise ValueError("Invalid KDA layout output buffers")
        for dst, src in zip(out, qkv):
            dst.copy_(src)
        return out
    return qkv


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
        _load_mtp_module().kda_mtp_commit_pending_drafts(
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


# ---------------------------------------------------------------------------
# In-tree KDA prefill op (CuTe DSL, trtllm::kda_prefill).
# ---------------------------------------------------------------------------

_PREFILL_MODULE: Optional[ModuleType] = None
_PREFILL_IMPORT_ERROR: Optional[Exception] = None


def _load_prefill_module() -> ModuleType:
    """Import the in-tree prefill custom-op module (registers the op)."""
    global _PREFILL_MODULE, _PREFILL_IMPORT_ERROR
    if _PREFILL_MODULE is not None:
        return _PREFILL_MODULE
    if _PREFILL_IMPORT_ERROR is not None:
        raise _PREFILL_IMPORT_ERROR
    try:
        module = importlib.import_module(
            "tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_custom_ops"
        )
    except ImportError as exc:
        _PREFILL_IMPORT_ERROR = exc
        raise
    _PREFILL_MODULE = module
    return module


def is_intree_prefill_available() -> bool:
    """True when the vendored prefill toolchain and operator are available."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        return False
    try:
        _load_prefill_module()
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# In-tree KDA multi-token verify op (CuTe DSL, trtllm::kda_mtp_decode).
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
        module = importlib.import_module(
            "tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops"
        )
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


# ---------------------------------------------------------------------------
# Dispatch API used by the KimiKDALinearAttention module.
# ---------------------------------------------------------------------------
