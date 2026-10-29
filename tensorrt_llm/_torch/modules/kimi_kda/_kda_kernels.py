# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel dispatch for the in-tree KDA module.

Both optimized KDA kernels are source-integrated into TensorRT-LLM: the
chunked prefill (CuTe DSL ``trtllm::kda_prefill``, see
``tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_custom_ops.py``) and the
fused CUDA C++ single-token decode (``trtllm::kda_decode`` thop op wrapped
by ``_kda_decode``). Neither requires the external
``exisiting_optimization_work`` collection at runtime.

On Blackwell (sm_100/sm_103) with the CuTe DSL toolchain available the
dispatch runs the optimized kernels. On any other arch (or when the in-tree
prefill op is unavailable) the dispatch falls back to FLA's ``chunk_kda`` /
``fused_recurrent_kda`` on-device references.

Callers are the ``KimiKDALinearAttention`` module. The module owns the
HF-parity semantics (Q/K/V projections, conv, gating, ``o_norm``,
``o_proj``); the dispatch here only wraps the kernel-level computation of
the delta-rule inner loop plus its state update.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ...flashinfer_utils import get_env_enable_pdl
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


@triton.jit(do_not_specialize=["num_tokens", "num_out_tokens", "beta_row_stride"])
def _fused_kda_post_conv_kernel(
    packed_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    beta_ptr,
    beta_out_ptr,
    num_tokens,
    num_out_tokens,
    beta_row_stride,
    l2_norm_eps,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    """Normalize and transpose channel-major packed Q/K/V in one launch.

    Rows in ``[num_tokens, num_out_tokens)`` are written as exact zeros, so
    the outputs are already zero-padded for the KDA prefill op. When
    ``HAS_BETA``, beta is cast to fp32 and zero-padded the same way.
    """
    token_offsets = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    head_idx = tl.program_id(1)
    dim_offsets = tl.arange(0, BLOCK_DIM)
    token_mask = token_offsets < num_tokens
    out_token_mask = token_offsets < num_out_tokens
    dim_mask = dim_offsets < HEAD_DIM
    mask = token_mask[:, None] & dim_mask[None, :]
    out_mask = out_token_mask[:, None] & dim_mask[None, :]

    feature_offsets = head_idx * HEAD_DIM + dim_offsets
    projection_size = NUM_HEADS * HEAD_DIM
    num_tokens_i64 = num_tokens.to(tl.int64)
    source_offsets = feature_offsets[None, :].to(tl.int64) * num_tokens_i64 + token_offsets[
        :, None
    ].to(tl.int64)
    section_stride = projection_size * num_tokens_i64
    output_offsets = (
        token_offsets[:, None].to(tl.int64) * projection_size + feature_offsets[None, :]
    )

    # Masked-off rows load 0 and normalize to exactly +0 (0 / sqrt(eps)).
    q = tl.load(packed_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    q /= tl.sqrt(tl.sum(q * q, axis=1) + l2_norm_eps)[:, None]
    tl.store(q_out_ptr + output_offsets, q, mask=out_mask)

    k = tl.load(
        packed_ptr + section_stride + source_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    k /= tl.sqrt(tl.sum(k * k, axis=1) + l2_norm_eps)[:, None]
    tl.store(k_out_ptr + output_offsets, k, mask=out_mask)

    v = tl.load(
        packed_ptr + 2 * section_stride + source_offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(v_out_ptr + output_offsets, v, mask=out_mask)

    if HAS_BETA:
        token_i64 = token_offsets.to(tl.int64)
        beta = tl.load(
            beta_ptr + token_i64 * beta_row_stride + head_idx, mask=token_mask, other=0.0
        ).to(tl.float32)
        tl.store(beta_out_ptr + token_i64 * NUM_HEADS + head_idx, beta, mask=out_token_mask)


def fused_kda_post_conv(
    packed: torch.Tensor,
    num_heads: int,
    head_dim: int,
    l2_norm_eps: float = 1e-6,
    padded_tokens: Optional[int] = None,
    beta: Optional[torch.Tensor] = None,
):
    """Convert packed channel-major convolution output to KDA Q/K/V.

    ``packed`` has shape ``[3 * num_heads * head_dim, tokens]``. The
    returned tensors are contiguous ``[1, out_tokens, num_heads, head_dim]``;
    Q and K are L2-normalized along the head dimension. ``out_tokens`` is
    ``padded_tokens`` when given (rows past ``tokens`` are exact zeros),
    else ``tokens``.

    If ``beta`` (``[..., tokens, num_heads]`` with unit head stride) is
    given, a fourth output holds it cast to fp32 and zero-padded to
    ``[1, out_tokens, num_heads]``.
    """
    projection_size = num_heads * head_dim
    if packed.ndim != 2 or packed.shape[0] != 3 * projection_size:
        raise ValueError(
            "Packed KDA post-conv expected shape "
            f"[{3 * projection_size}, tokens], got {tuple(packed.shape)}"
        )
    if not packed.is_contiguous():
        raise ValueError("Packed KDA post-conv requires a contiguous tensor")

    num_tokens = packed.shape[1]
    out_tokens = num_tokens if padded_tokens is None else padded_tokens
    if out_tokens < num_tokens:
        raise ValueError(f"padded_tokens={out_tokens} is smaller than tokens={num_tokens}")
    output_shape = (1, out_tokens, num_heads, head_dim)
    q_out = torch.empty(output_shape, dtype=packed.dtype, device=packed.device)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)
    beta_out = None
    beta_2d = None
    if beta is not None:
        if beta.ndim == 2 and beta.shape == (num_tokens, num_heads):
            beta_2d = beta
        elif beta.ndim == 3 and beta.shape == (1, num_tokens, num_heads):
            beta_2d = beta[0]
        if beta_2d is None or beta_2d.stride(-1) != 1 or beta_2d.device != packed.device:
            raise ValueError(
                f"beta must be [tokens={num_tokens}, heads={num_heads}] or "
                f"[1, tokens={num_tokens}, heads={num_heads}] on {packed.device} "
                f"with unit head stride, got shape {tuple(beta.shape)}, "
                f"stride {beta.stride()}, device {beta.device}"
            )
        beta_out = torch.empty(
            (1, out_tokens, num_heads), dtype=torch.float32, device=packed.device
        )
    if out_tokens == 0:
        return (q_out, k_out, v_out) if beta is None else (q_out, k_out, v_out, beta_out)

    block_tokens = 16
    block_dim = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(out_tokens, block_tokens), num_heads)
    _fused_kda_post_conv_kernel[grid](
        packed,
        q_out,
        k_out,
        v_out,
        beta_2d if beta_2d is not None else q_out,
        beta_out if beta_out is not None else q_out,
        num_tokens,
        out_tokens,
        beta_2d.stride(0) if beta_2d is not None else 0,
        l2_norm_eps,
        num_heads,
        head_dim,
        block_tokens,
        block_dim,
        beta is not None,
        num_warps=8,
        num_stages=3,
    )
    return (q_out, k_out, v_out) if beta is None else (q_out, k_out, v_out, beta_out)


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
    except Exception as exc:  # typically ImportError when CuTe DSL is unavailable
        _PREFILL_IMPORT_ERROR = exc
        raise
    _PREFILL_MODULE = module
    return module


def is_intree_prefill_available() -> bool:
    """True when the in-tree CuTe DSL prefill op can be imported."""
    try:
        _load_prefill_module()
        return True
    except Exception:
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


# ---------------------------------------------------------------------------
# Generated MTP verify kernel (trtllm::kimi_k3_kda_mtp_verify_kf).
# ---------------------------------------------------------------------------

# Default off. Startup snapshot per dispatch instance, strict 0/1 parsing.
_KF_VERIFY_ENV = "TRTLLM_K3_KDA_VERIFY_KF"
# (heads per rank, num_spec) the kernel takes: attention-DP ranks (96 heads) with 3 or 5 drafts. TP8 ranks (12 heads)
# stay on the CuTe DSL kernels, whose PDL variant (TRTLLM_K3_KDA_VERIFY_PDL) is at least as fast there.
_KF_VERIFY_ROUTES = frozenset({(96, 3), (96, 5)})
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
    """Why the generated verify kernel cannot take a ``mtp_verify`` call, or None.

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
        enable_pdl: Optional[bool] = None,
    ) -> None:
        self._enable_pdl = get_env_enable_pdl() if enable_pdl is None else enable_pdl
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
        # Generated verify kernel for the eligible optimized verify calls
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

        Thin passthrough to the BF16 or MXFP8-output MTP op (see
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
                    f"{_KF_VERIFY_ENV}: generated verify kernel for {route[0]} heads, "
                    f"num_spec {route[1]}",
                )
                # PDL as for the CuTe DSL verify kernels (TRTLLM_K3_KDA_VERIFY_PDL with TRTLLM_ENABLE_PDL).
                enable_pdl = _load_mtp_module()._verify_pdl_enabled()
                return _kf_mtp_verify(enable_pdl=enable_pdl, **kf_kwargs)
            _log_kf_once(
                reason,
                f"{_KF_VERIFY_ENV}: CuTe DSL verify kernels for an ineligible call ({reason})",
            )
        _load_mtp_module()  # registers both BF16 and FP8 MTP ops
        if mxfp8_output:
            kwargs.pop("fuse_output_norm", None)
            op = torch.ops.trtllm.kda_mtp_decode_fp8
        else:
            op = torch.ops.trtllm.kda_mtp_decode
        return op(**kwargs)

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
        if cu_seqlens is not None:
            if cu_seqlens.shape[0] - 1 != num_sequences:
                return False
        if state_pool.dtype != torch.float32 or state_pool.ndim != 4:
            return False
        _, H, V, K = state_pool.shape
        if state_pool.stride()[1:] != (V * K, K, 1):
            return False
        if state_pool.stride(0) < H * V * K or state_pool.stride(0) % 4:
            return False
        if state_pool.data_ptr() % 16:
            return False
        if (
            state_indices.ndim != 1
            or state_indices.shape[0] != num_sequences
            or state_indices.dtype not in (torch.int32, torch.int64)
            or not state_indices.is_contiguous()
        ):
            return False
        if state_indices.data_ptr() % 16:
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run KDA chunked prefill.

        Q and K must be L2-normalized by the caller. On the optimized path,
        beta sigmoid runs inside the in-tree CuTe DSL op.

        On the FLA path it calls ``fla.ops.kda.chunk_kda`` directly with the
        matching flags so the semantics are byte-equivalent.

        The optimized kernel always reads and writes indexed V-first
        ``[slots, H, V, K]`` state-pool rows. Callers provide the executor
        pool and reset fresh rows before entering the kernel. The FLA fallback
        retains the batch-dense ``initial_state``/``final_state`` contract.

        ``chunk_indices``, ``varlen_is_aligned``, and
        ``single_sequence_length`` are prepared once from Kimi K3 runtime
        metadata. They keep the optimized path from reading ``cu_seqlens``
        back from the GPU in every KDA layer.
        """
        use_indexed_state = state_pool is not None
        if use_indexed_state and (initial_state is not None or state_indices is None):
            raise ValueError(
                "Indexed KDA prefill requires state_indices and does not accept initial_state."
            )
        use_optimized = self.prefill_kernel_path == "optimized" and use_indexed_state

        if use_indexed_state and not use_optimized:
            raise RuntimeError(
                "Indexed KDA prefill requires the optimized prefill path; "
                "use the FLA state path for this batch."
            )

        if use_optimized:
            import torch.nn.functional as F

            real_T = q.shape[1]
            if (
                cu_seqlens is not None
                and single_sequence_length is not None
                and q.shape[1] != single_sequence_length
            ):
                # Caller already produced chunk-padded inputs (zero tails);
                # only the output needs trimming back to the real length.
                real_T = single_sequence_length
                padded_T = q.shape[1]
                if (
                    cu_seqlens.shape[0] != 2
                    or padded_T % chunk_size != 0
                    or not 0 < padded_T - real_T < chunk_size
                    or any(t.shape[1] != padded_T for t in (k, v, beta))
                    or g.shape[1] not in (real_T, padded_T)
                ):
                    raise ValueError(
                        "Pre-padded KDA prefill inputs must be one sequence padded to the "
                        f"next {chunk_size}-token boundary (real_T={real_T}, "
                        f"q/k/v/g/beta lengths={[t.shape[1] for t in (q, k, v, g, beta)]})."
                    )
            elif cu_seqlens is not None:
                # The op's varlen single-seq path (Phase 2.1) expects the
                # caller to zero-pad the packed tensors to a chunk multiple
                # (FLA convention) while cu_seqlens keeps the real length;
                # the op re-sentinels g's tail itself. Without this the op
                # would run the mask-free kernel on a partial final chunk.
                # Multi-seq varlen runs the masked path and needs no pad.
                if cu_seqlens.shape[0] == 2 and real_T % chunk_size != 0:
                    pad = chunk_size - real_T % chunk_size
                    q = F.pad(q, (0, 0, 0, 0, 0, pad))
                    k = F.pad(k, (0, 0, 0, 0, 0, pad))
                    v = F.pad(v, (0, 0, 0, 0, 0, pad))
                    g = F.pad(g, (0, 0, 0, 0, 0, pad))
                    beta = F.pad(beta, (0, 0, 0, pad))

            A_log_kernel = A_log.detach() if A_log is not None else None
            dt_bias_kernel = dt_bias.detach() if dt_bias is not None else None

            _load_prefill_module()  # registers trtllm::kda_prefill
            out = torch.ops.trtllm.kda_prefill(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                state_pool=state_pool,
                state_indices=state_indices,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=chunk_size,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                A_log=A_log_kernel,
                dt_bias=dt_bias_kernel,
                varlen_is_aligned=varlen_is_aligned,
                single_sequence_length=single_sequence_length,
            )
            if out.shape[1] != real_T:
                out = out[:, :real_T]
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
            use_qk_l2norm_in_kernel=False,
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
        return _kda_decode.run_kda_decode_fusion_cuda(enable_pdl=self._enable_pdl, **kwargs)
