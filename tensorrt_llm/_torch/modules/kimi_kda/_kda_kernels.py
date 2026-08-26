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
from types import ModuleType
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ...flashinfer_utils import get_env_enable_pdl
from ..fla.index import prepare_chunk_indices
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
    """The optimized prefill/decode kernels are Blackwell sm_100 only."""
    return get_kda_sm_version() in (100, 103)


@triton.jit(do_not_specialize=["num_tokens"])
def _fused_kda_post_conv_kernel(
    packed_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    num_tokens,
    l2_norm_eps,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Normalize and transpose channel-major packed Q/K/V in one launch."""
    token_offsets = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    head_idx = tl.program_id(1)
    dim_offsets = tl.arange(0, BLOCK_DIM)
    token_mask = token_offsets < num_tokens
    dim_mask = dim_offsets < HEAD_DIM
    mask = token_mask[:, None] & dim_mask[None, :]

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

    q = tl.load(packed_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    q /= tl.sqrt(tl.sum(q * q, axis=1) + l2_norm_eps)[:, None]
    tl.store(q_out_ptr + output_offsets, q, mask=mask)

    k = tl.load(
        packed_ptr + section_stride + source_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    k /= tl.sqrt(tl.sum(k * k, axis=1) + l2_norm_eps)[:, None]
    tl.store(k_out_ptr + output_offsets, k, mask=mask)

    v = tl.load(
        packed_ptr + 2 * section_stride + source_offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(v_out_ptr + output_offsets, v, mask=mask)


def fused_kda_post_conv(
    packed: torch.Tensor,
    num_heads: int,
    head_dim: int,
    l2_norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert packed channel-major convolution output to KDA Q/K/V.

    ``packed`` has shape ``[3 * num_heads * head_dim, tokens]``. The
    returned tensors are contiguous ``[1, tokens, num_heads, head_dim]``;
    Q and K are L2-normalized along the head dimension.
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
    output_shape = (1, num_tokens, num_heads, head_dim)
    q_out = torch.empty(output_shape, dtype=packed.dtype, device=packed.device)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)
    if num_tokens == 0:
        return q_out, k_out, v_out

    block_tokens = 16
    block_dim = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(num_tokens, block_tokens), num_heads)
    _fused_kda_post_conv_kernel[grid](
        packed,
        q_out,
        k_out,
        v_out,
        num_tokens,
        l2_norm_eps,
        num_heads,
        head_dim,
        block_tokens,
        block_dim,
        num_warps=8,
        num_stages=3,
    )
    return q_out, k_out, v_out


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

    def mtp_verify(self, **kwargs) -> torch.Tensor:
        """Run the fused KDA multi-token verify kernel.

        Thin passthrough to ``trtllm::kda_mtp_decode`` (see
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
        _load_mtp_module()  # registers trtllm::kda_mtp_decode
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
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
    ) -> bool:
        """Return whether prefill can update this V-first pool directly."""
        if self.prefill_kernel_path != "optimized" or num_tokens == 0:
            return False
        if cu_seqlens is not None:
            if chunk_indices is None:
                from fla.ops.utils.index import prepare_chunk_indices

                chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
            if chunk_indices.shape[0] < 4:
                return False
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
        if use_optimized and cu_seqlens is not None:
            if chunk_indices is None:
                chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
            # The persistent K123 scheduler needs at least 4 total chunks
            # (cgs_per_head = NT // 4 cooperative groups per head). The
            # eqlen path guarantees this by padding to a 256-token multiple
            # inside the op; varlen has no such pad, so small varlen
            # batches (short-prompt contexts, NT < 4) launch with a
            # zero-size grid -> DSLCudaRuntimeError. Route them to the FLA
            # reference path (negligible perf impact at these sizes). The
            # check must happen before dispatch: the FLA fallback applies
            # Q/K normalization and beta sigmoid in its own kernels.
            if chunk_indices.shape[0] < 4:
                use_optimized = False

        if use_indexed_state and not use_optimized:
            raise RuntimeError(
                "Indexed KDA prefill requires the optimized prefill path; "
                "use the FLA state path for this batch."
            )

        if use_optimized:
            import torch.nn.functional as F

            real_T = q.shape[1]
            if cu_seqlens is not None:
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
