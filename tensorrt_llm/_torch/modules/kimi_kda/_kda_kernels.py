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
        if state_pool.dtype != torch.float32 or state_pool.ndim != 4:
            return False
        _, H, V, K = state_pool.shape
        if state_pool.stride()[1:] != (V * K, K, 1):
            return False
        if state_pool.stride(0) < H * V * K or state_pool.stride(0) % 4:
            return False
        if state_pool.data_ptr() % 16:
            return False
        num_sequences = state_indices.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
        if (
            state_indices.ndim != 1
            or state_indices.shape[0] != num_sequences
            or state_indices.dtype not in (torch.int32, torch.int64)
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run KDA chunked prefill.

        On the optimized path this replays the preprocessing that FLA's
        ``ChunkKDAFunction.forward`` performs when the caller enables
        ``use_qk_l2norm_in_kernel``, ``use_beta_sigmoid_in_kernel``,
        ``use_gate_in_kernel``, and ``state_v_first``; then dispatches to
        the in-tree ``trtllm::kda_prefill`` CuTe DSL op.

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
                from fla.ops.utils.index import prepare_chunk_indices

                chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
            # The persistent K123 scheduler needs at least 4 total chunks
            # (cgs_per_head = NT // 4 cooperative groups per head). The
            # eqlen path guarantees this by padding to a 256-token multiple
            # inside the op; varlen has no such pad, so small varlen
            # batches (short-prompt contexts, NT < 4) launch with a
            # zero-size grid -> DSLCudaRuntimeError. Route them to the FLA
            # reference path (negligible perf impact at these sizes). The
            # check must happen HERE, before the l2norm/beta-sigmoid
            # pre-transforms below: the FLA path applies both in-kernel.
            if chunk_indices.shape[0] < 4:
                use_optimized = False

        if use_indexed_state and not use_optimized:
            raise RuntimeError(
                "Indexed KDA prefill requires the optimized prefill path; "
                "use the FLA state path for this batch."
            )

        if use_optimized:
            import torch.nn.functional as F
            from fla.modules.l2norm import l2norm_fwd
            from fla.ops.common.gate import fused_beta_sigmoid

            q, _ = l2norm_fwd(q)
            k, _ = l2norm_fwd(k)
            beta = fused_beta_sigmoid(beta, scale=1.0).to(torch.bfloat16)

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
