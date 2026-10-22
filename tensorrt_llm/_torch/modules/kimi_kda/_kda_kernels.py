# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA FlashInfer prefill, native decode/verification and portable FLA dispatch."""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from . import _kda_decode

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
    from ...cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

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
# Kernel Factory MTP verify (trtllm::kimi_k3_kda_mtp_verify_kf).
# ---------------------------------------------------------------------------

# Default off. Startup snapshot per dispatch instance, strict 0/1 parsing.
_KF_VERIFY_ENV = "TRTLLM_K3_KDA_VERIFY_KF"
# (heads per rank, num_spec) the kernel takes: attention-DP ranks (96 heads) with 3 or 5 drafts, TP8 ranks (12 heads)
# with 3 drafts. TP8 with 5 drafts stays on the CuTe DSL kernels (no measurable gain).
_KF_VERIFY_ROUTES = frozenset({(96, 3), (96, 5), (12, 3)})
# The Kimi K3 constants the kernel bakes in.
_KF_LOWER_BOUND = -5.0
_KF_SCALE = 128**-0.5
_KF_ONORM_EPS = 1e-5
_KF_BF16_INPUTS = (
    "x_q",
    "x_k",
    "x_v",
    "g",
    "beta",
    "onorm_g",
    "cs_q",
    "cs_k",
    "cs_v",
    "recurrent_state",
    "k_cache",
    "v_cache",
    "beta_cache",
)
# The constants the kernel reads as float32. The mixer casts its parameters to BF16 (A_log and dt_bias among them), so
# it passes FP32 copies of those two (see mtp_verify).
_KF_FP32_INPUTS = ("w_q", "w_k", "w_v", "A_log", "dt_bias", "onorm_weight")
# Tensors read through tensor maps or bulk copies: (name, stride dims that must be whole 16-byte units). beta and the
# beta cache are read through tensor maps with boxes of 8 heads, so their rows must be padded to a multiple of 8 heads
# (the [f_a | b] projection and the cache manager do; a separate [T, 12] beta output at TP8 is not eligible).
_KF_ROW_ALIGNED = (
    ("x_q", (1,)),
    ("x_k", (1,)),
    ("x_v", (1,)),
    ("g", (1,)),
    ("beta", (1,)),
    ("onorm_g", (1,)),
    ("cs_q", (0, 2)),
    ("cs_k", (0, 2)),
    ("cs_v", (0, 2)),
    ("k_cache", (0, 1)),
    ("g_cache", (0, 1)),
    ("v_cache", (0, 1)),
    ("beta_cache", (0, 1)),
    ("recurrent_state", (0,)),
    ("w_q", ()),
    ("w_k", ()),
    ("w_v", ()),
)
_kf_logged: set = set()


def _parse_switch(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _log_kf_once(key: str, message: str) -> None:
    """Log once per process and key: which routes ran and why other calls stayed on the CuTe DSL kernels."""
    if key in _kf_logged:
        return
    _kf_logged.add(key)
    try:
        from tensorrt_llm.logger import logger
    except ImportError:  # pragma: no cover - source-loader stub path
        return
    logger.info(message)


def _kf_layout_ok(kwargs: dict, num_heads: int, num_spec: int) -> bool:
    """The layouts that ``kimi_k3_kda_mtp_verify_kf`` checks (the production op accepts more of them)."""
    dim = num_heads * 128
    x_q, indices = kwargs["x_q"], kwargs["ssm_state_indices"]
    if x_q.dim() != 4 or indices.dim() != 1 or kwargs["A_log"].dim() != 1:
        return False
    rows = x_q.shape[1]
    if rows != indices.shape[0] * (num_spec + 1):
        return False
    if indices.dtype == torch.int32 and not indices.is_contiguous():
        return False
    for name in ("x_q", "x_k", "x_v", "g", "onorm_g"):  # [1, T, H, 128], contiguous head blocks
        t = kwargs[name]
        if t.shape != (1, rows, num_heads, 128) or t.stride(3) != 1 or t.stride(2) != 128:
            return False
    beta = kwargs["beta"]
    if beta.shape != (1, rows, num_heads) or beta.stride(2) != 1:
        return False
    for name, shape in (
        ("w_q", (dim, 4)),
        ("w_k", (dim, 4)),
        ("w_v", (dim, 4)),
        ("A_log", (num_heads,)),
        ("dt_bias", (dim,)),
        ("onorm_weight", (128,)),
    ):
        t = kwargs[name]
        if t.dtype != torch.float32 or t.shape != shape or not t.is_contiguous():
            return False
    state = kwargs["recurrent_state"]
    if state.dim() != 4 or state.shape[1:] != (num_heads, 128, 128):
        return False
    if state.stride()[1:] != (128 * 128, 128, 1):
        return False
    pool = state.shape[0]
    # Accepted counts may have any slot stride (KV cache manager v2 keeps a count block per slot).
    accepted = kwargs["num_accepted_tokens"]
    if accepted.dim() != 1 or accepted.shape[0] < pool:
        return False
    for name in ("cs_q", "cs_k", "cs_v"):  # [>= pool, H * 128, >= 3 + num_spec], dim-contiguous
        t = kwargs[name]
        if t.dim() != 3 or t.shape[0] < pool or t.shape[1] != dim or t.shape[2] < 3 + num_spec:
            return False
        if t.stride(1) != 1:
            return False
    for name, width in (
        ("k_cache", dim),
        ("g_cache", dim),
        ("v_cache", dim),
        ("beta_cache", num_heads),
    ):
        t = kwargs[name]
        if (
            t.dim() != 3
            or t.shape[0] < pool
            or t.shape[1:] != (num_spec, width)
            or t.stride(2) != 1
        ):
            return False
    return True


def _kf_verify_ineligibility(kwargs: dict, mxfp8_output: bool) -> Optional[str]:
    """Why the Kernel Factory verify kernel cannot take a ``mtp_verify`` call, or None.

    Mirrors the checks of ``kimi_k3_kda_mtp_verify_kf`` (which raises on a violation), so
    enabling the switch never turns a call that the CuTe DSL kernels take into an error:
    such a call stays on the CuTe DSL kernels, and each distinct reason is logged once.
    """
    if not mxfp8_output:
        return "BF16 output"
    if not kwargs.get("packed_token_layout", False) or not kwargs.get("fuse_output_norm", False):
        return "not the packed fused-norm layout"
    if kwargs.get("cu_seqlens") is not None:
        return "cu_seqlens"
    if kwargs.get("onorm_g") is None or kwargs.get("onorm_weight") is None:
        return "no output-norm inputs"
    num_heads = kwargs["A_log"].shape[0]
    if (num_heads, kwargs["num_spec"]) not in _KF_VERIFY_ROUTES:
        return f"{num_heads} heads with num_spec {kwargs['num_spec']}"
    scale = kwargs.get("scale")
    if (
        kwargs.get("lower_bound") != _KF_LOWER_BOUND
        or (scale is not None and scale != _KF_SCALE)
        or kwargs.get("onorm_eps", 1e-5) != _KF_ONORM_EPS
    ):
        return "constants"
    if any(kwargs[name].dtype != torch.bfloat16 for name in _KF_BF16_INPUTS):
        return "dtype"
    if kwargs["g_cache"].dtype != torch.float32:
        return "dtype"
    for name in _KF_FP32_INPUTS:
        if kwargs[name].dtype != torch.float32:
            return f"{name} is {kwargs[name].dtype}, not torch.float32"
    if not _kf_layout_ok(kwargs, num_heads, kwargs["num_spec"]):
        return "layout"
    for name, dims in _KF_ROW_ALIGNED:
        tensor = kwargs[name]
        units = 16 // tensor.element_size()
        if tensor.data_ptr() % 16 != 0 or any(
            tensor.stride(d) <= 0 or tensor.stride(d) % units != 0 for d in dims
        ):
            return f"{name} alignment"
    return None


def _with_fp32_gate_constants(
    kwargs: dict, A_log_f32: Optional[torch.Tensor], dt_bias_f32: Optional[torch.Tensor]
) -> dict:
    """The verify arguments with FP32 copies of ``A_log`` / ``dt_bias``, which the KF kernel reads as float32.

    The copies hold the same values as the parameters (BF16 -> FP32 is exact). Without them, or with other shapes,
    the arguments stay as they are and the eligibility check names a parameter that is not float32.
    """
    if A_log_f32 is None or dt_bias_f32 is None:
        return kwargs
    if A_log_f32.shape != kwargs["A_log"].shape or dt_bias_f32.shape != kwargs["dt_bias"].shape:
        return kwargs
    return {**kwargs, "A_log": A_log_f32, "dt_bias": dt_bias_f32}


def _kf_mtp_verify(
    *,
    enable_pdl: bool,
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
    onorm_g: torch.Tensor,
    onorm_weight: torch.Tensor,
    lower_bound: float = _KF_LOWER_BOUND,
    scale: Optional[float] = None,
    onorm_eps: float = _KF_ONORM_EPS,
    **unused,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``trtllm::kda_mtp_decode_fp8`` (packed layout, fused output norm) through the KF kernel."""
    num_tokens = x_q.shape[1]
    num_heads = A_log.shape[0]
    head_dim = x_v.shape[-1]
    if scale is None:
        scale = x_q.shape[-1] ** -0.5
    out = torch.empty(
        (num_tokens, num_heads * head_dim), dtype=torch.float8_e4m3fn, device=x_q.device
    )
    out_scale = torch.empty(
        ((num_tokens + 127) // 128 * 128 * num_heads * 4,), dtype=torch.uint8, device=x_q.device
    )
    # The kernel waits for its programmatic-launch predecessor before reading anything, so the int32
    # conversions (a kernel each, when needed) do not restrict PDL.
    torch.ops.trtllm.kimi_k3_kda_mtp_verify_kf(
        x_q,
        x_k,
        x_v,
        g,
        beta,
        onorm_g,
        w_q,
        w_k,
        w_v,
        A_log,
        dt_bias,
        onorm_weight,
        ssm_state_indices.to(torch.int32),
        num_accepted_tokens.to(torch.int32),
        cs_q,
        cs_k,
        cs_v,
        recurrent_state,
        k_cache,
        g_cache,
        v_cache,
        beta_cache,
        out,
        out_scale,
        num_spec,
        lower_bound,
        scale,
        onorm_eps,
        enable_pdl,
    )
    return out, out_scale


# ---------------------------------------------------------------------------
# Dispatch API used by the KimiKDALinearAttention module.
# ---------------------------------------------------------------------------


class KDAKernelDispatch:
    """Kernel dispatch state for one ``KimiKDALinearAttention`` instance.

    Attributes
    ----------
    prefill_kernel_path : str
        Selected prefill path: ``"optimized"`` or ``"fla"``.
    decode_kernel_path : str
        Selected decode path: ``"optimized"`` or ``"fla"``.
    verify_kernel_path : str
        Selected multi-token verify path: ``"optimized"`` (fused
        ``trtllm::kda_mtp_decode`` replay kernel) or ``"fla"`` (sequential
        per-step ``fused_recurrent_kda`` with intermediate-buffer state
        promotion).
    Notes
    -----
    Prefill, decode, and verify dispatch are decided independently. All
    require a supported GPU; the in-tree CuTe DSL ops additionally require
    their modules to be importable.
    """

    _selection_logged = False

    def __init__(
        self,
        use_optimized_prefill: bool = True,
        use_optimized_decode: bool = True,
        use_optimized_verify: bool = True,
    ) -> None:
        optimized_supported = is_kda_optimized_supported()
        self.prefill_kernel_path = "fla"
        if use_optimized_prefill and optimized_supported and is_intree_prefill_available():
            self.prefill_kernel_path = "optimized"
        self.decode_kernel_path = (
            "optimized" if use_optimized_decode and optimized_supported else "fla"
        )
        self.verify_kernel_path = "fla"
        if use_optimized_verify and optimized_supported and is_intree_mtp_available():
            self.verify_kernel_path = "optimized"
        # Kernel Factory verify kernel for the eligible optimized verify calls
        # (see _kf_verify_ineligibility), default off.
        kf_switch = _parse_switch(_KF_VERIFY_ENV)  # an invalid value raises on every path
        self._kf_verify = False
        if kf_switch and self.verify_kernel_path == "optimized":
            if not hasattr(torch.ops.trtllm, "kimi_k3_kda_mtp_verify_kf"):
                raise RuntimeError(
                    f"{_KF_VERIFY_ENV}=1 needs trtllm::kimi_k3_kda_mtp_verify_kf; "
                    "rebuild the TensorRT-LLM C++ extension"
                )
            self._kf_verify = True
        # One line per process so runs record which paths actually executed
        # (the fallback is otherwise silent).
        if not KDAKernelDispatch._selection_logged:
            KDAKernelDispatch._selection_logged = True
            try:
                from tensorrt_llm.logger import logger
            except ImportError:  # pragma: no cover — source-loader stub path
                pass
            else:
                logger.info(
                    f"KDA kernel dispatch: prefill={self.prefill_kernel_path} "
                    f"decode={self.decode_kernel_path} "
                    f"verify={self.verify_kernel_path} "
                    f"verify_kf={self._kf_verify}"
                )

    def mtp_verify(
        self,
        *,
        mxfp8_output: bool = False,
        A_log_f32: Optional[torch.Tensor] = None,
        dt_bias_f32: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the fused KDA multi-token verify kernel.

        Thin passthrough to ``trtllm::kda_mtp_decode``, or with
        ``mxfp8_output`` to ``trtllm::kda_mtp_decode_fp8``, which always
        fuses the output norm and returns the E4M3 core and its R128c4
        scales (see
        ``custom_ops/cute_dsl_kimi_k3_kda_mtp_ops.py`` for the full
        argument and state-management contract). Only defined on the
        optimized path; the FLA fallback is the module's sequential
        per-step loop with intermediate-buffer promotion.

        ``A_log_f32`` / ``dt_bias_f32``: FP32 copies of ``A_log`` / ``dt_bias``
        (the mixer casts those parameters to BF16), read only by the Kernel
        Factory verify kernel; the CuTe DSL kernels read ``A_log`` / ``dt_bias``.
        """
        if self.verify_kernel_path != "optimized":
            raise RuntimeError(
                "mtp_verify called on non-optimized path; use the module's "
                "sequential FLA verify fallback instead."
            )
        if self._kf_verify:
            kf_kwargs = _with_fp32_gate_constants(kwargs, A_log_f32, dt_bias_f32)
            reason = _kf_verify_ineligibility(kf_kwargs, mxfp8_output)
            if reason is None:
                route = (kf_kwargs["A_log"].shape[0], kf_kwargs["num_spec"])
                _log_kf_once(
                    f"route {route}",
                    f"{_KF_VERIFY_ENV}: Kernel Factory verify kernel for {route[0]} heads, "
                    f"num_spec {route[1]}",
                )
                # PDL follows the stack-wide TRTLLM_ENABLE_PDL switch. The kernel waits for its
                # producer before its first global access and never triggers early.
                enable_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
                return _kf_mtp_verify(enable_pdl=enable_pdl, **kf_kwargs)
            _log_kf_once(
                reason,
                f"{_KF_VERIFY_ENV}: CuTe DSL verify kernels for an ineligible call ({reason})",
            )
        _load_mtp_module()  # registers both the BF16 and the MXFP8-output ops
        if mxfp8_output:
            if not kwargs.pop("fuse_output_norm", True):
                raise ValueError("the MXFP8-output KDA MTP op always fuses the output norm")
            return torch.ops.trtllm.kda_mtp_decode_fp8(**kwargs)
        return torch.ops.trtllm.kda_mtp_decode(**kwargs)

    def can_use_indexed_prefill(
        self,
        *,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
        num_sequences: int,
        num_tokens: int,
    ) -> bool:
        """Return whether prefill can update this V-first pool directly."""
        if self.prefill_kernel_path != "optimized" or num_tokens == 0:
            return False
        if (
            cu_seqlens is None
            or cu_seqlens.dtype != torch.int64
            or cu_seqlens.shape != (num_sequences + 1,)
            or not cu_seqlens.is_contiguous()
        ):
            return False
        if state_pool.dtype not in (torch.bfloat16, torch.float32) or state_pool.ndim != 4:
            return False
        _, H, V, K = state_pool.shape
        if state_pool.stride()[1:] != (V * K, K, 1):
            return False
        if (
            (V, K) != (128, 128)
            or state_pool.stride(0) < H * V * K
            or state_pool.stride(0) * state_pool.element_size() % 16
        ):
            return False
        if state_pool.data_ptr() % 16:
            return False
        if (
            state_indices.ndim != 1
            or state_indices.shape[0] != num_sequences
            or state_indices.dtype != torch.int32
            or not state_indices.is_contiguous()
        ):
            return False
        if (
            has_initial_states.ndim != 1
            or has_initial_states.shape[0] != num_sequences
            or has_initial_states.dtype != torch.bool
            or not has_initial_states.is_contiguous()
        ):
            return False
        if (
            state_pool.device != state_indices.device
            or state_pool.device != has_initial_states.device
        ):
            return False
        return True

    def prefill_chunk_kda(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        safe_gate: bool,
        lower_bound: Optional[float],
        cu_seqlens: Optional[torch.Tensor],
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
        state_pool: Optional[torch.Tensor] = None,
        state_indices: Optional[torch.Tensor] = None,
        varlen_is_aligned: Optional[bool] = None,
        single_sequence_length: Optional[int] = None,
        max_seqlen: int = 0,
        seq_order: Optional[torch.Tensor] = None,
        cu_chunks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run prefill on raw Q/K/V and gate/beta logits.

        FlashInfer updates indexed BF16/FP32 V-first state directly; FLA uses
        dense FP32 initial/final states. Both normalize Q/K internally.
        """
        if state_pool is not None:
            if self.prefill_kernel_path != "optimized" or initial_state is not None:
                raise ValueError("Indexed prefill requires FlashInfer and no dense initial state")
            if not safe_gate or lower_bound is None:
                raise ValueError("FlashInfer requires a finite negative gate bound")
            _load_prefill_module()
            out = torch.ops.trtllm.kda_prefill(
                q,
                k,
                v,
                g,
                beta.to(torch.bfloat16),
                state_pool,
                state_indices,
                cu_seqlens,
                A_log.float().contiguous(),
                dt_bias.float().contiguous(),
                scale,
                lower_bound,
                getattr(self, "prefill_mode", "auto"),
                max_seqlen,
                seq_order,
                cu_chunks,
            )
            return out, None

        from fla.ops.kda import chunk_kda

        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )

    def decode_kda(self, **kwargs) -> torch.Tensor:
        """Run the fused KDA single-token decode kernel.

        This is only defined on the optimized path; the FLA fallback runs
        ``fla.ops.kda.fused_recurrent_kda`` directly and does not use this
        wrapper (see ``_decode_via_fla`` on the module).
        """
        if self.decode_kernel_path != "optimized":
            raise RuntimeError(
                "decode_kda called on non-optimized path; use FLA path via "
                "the module's fallback handling instead."
            )
        return _kda_decode.run_kda_decode_fusion_cuda(**kwargs)
