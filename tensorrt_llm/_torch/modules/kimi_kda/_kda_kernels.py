# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA FlashInfer prefill, native decode/verification and portable FLA dispatch."""

from __future__ import annotations

import importlib
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
    replay_indices_ptr,
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
    replay_slot = tl.load(replay_indices_ptr + request).to(tl.int64)

    conv_offset = slot * conv_stride_slot + dim * conv_stride_dim + window * conv_stride_window
    cache_offset = (
        replay_slot * cache_stride_slot + dim * cache_stride_dim + window * cache_stride_window
    )
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
    replay_indices: Optional[torch.Tensor] = None,
) -> None:
    """Copy selected packed ``W - 1`` rows into KDA replay caches.

    The live convolution pool is ``[slots, 3D, W - 1]``. Replay caches are
    ``[replay_slots, D, W - 1 + num_spec]`` with a dim-contiguous inner layout.
    Row ``state_indices[n]`` of the pool goes to row ``replay_indices[n]`` of
    the caches; without ``replay_indices`` the caches share the pool's slots.
    """
    if conv_pool.ndim != 3 or conv_pool.shape[1] % 3:
        raise ValueError(f"Expected packed rank-3 KDA convolution pool, got {conv_pool.shape}")
    projection_size = conv_pool.shape[1] // 3
    committed = conv_pool.shape[2]
    caches = (q_cache, k_cache, v_cache)
    if any(cache.ndim != 3 or cache.shape[1] != projection_size for cache in caches):
        raise ValueError("KDA replay convolution caches do not match the live pool geometry")
    if replay_indices is None:
        if any(cache.shape[0] != conv_pool.shape[0] for cache in caches):
            raise ValueError("KDA replay convolution caches do not match the live pool slots")
        replay_indices = state_indices
    if any(cache.shape[2] < committed for cache in caches):
        raise ValueError("KDA replay convolution caches are shorter than the committed window")
    if any(cache.stride() != q_cache.stride() for cache in caches[1:]):
        raise ValueError("KDA replay convolution caches must share one layout")
    for indices in (state_indices, replay_indices):
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("KDA replay state indices must be a rank-1 int32 or int64 tensor")
    if replay_indices.shape != state_indices.shape:
        raise ValueError("Expected one replay index per state index")
    if any(
        tensor.device != conv_pool.device for tensor in (*caches, state_indices, replay_indices)
    ):
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
            replay_indices,
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
                    f"verify={self.verify_kernel_path}"
                )

    def mtp_verify(
        self, *, mxfp8_output: bool = False, **kwargs
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
        """
        if self.verify_kernel_path != "optimized":
            raise RuntimeError(
                "mtp_verify called on non-optimized path; use the module's "
                "sequential FLA verify fallback instead."
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
