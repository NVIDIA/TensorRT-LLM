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


def _load_fla_chunk_kda() -> ModuleType:
    return importlib.import_module("fla.ops.kda")


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

    def get_prefill_source(self) -> str:
        if self.prefill_kernel_path == "optimized":
            return _load_prefill_module().__file__ or "<custom_ops.cute_dsl_kimi_k3>"
        return _load_fla_chunk_kda().__file__ or "<fla.ops.kda>"

    def get_decode_source(self) -> str:
        if self.decode_kernel_path == "optimized":
            return _kda_decode.__file__ or "<kimi_kda._kda_decode>"
        return _load_fla_chunk_kda().__file__ or "<fla.ops.kda>"

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
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run KDA chunked prefill.

        On the optimized path this replays the preprocessing that FLA's
        ``ChunkKDAFunction.forward`` performs when the caller enables
        ``use_qk_l2norm_in_kernel``, ``use_beta_sigmoid_in_kernel``,
        ``use_gate_in_kernel``, and ``state_v_first``; then dispatches to
        the in-tree ``trtllm::kda_prefill`` CuTe DSL op.

        On the FLA path it calls ``fla.ops.kda.chunk_kda`` directly with the
        matching flags so the semantics are byte-equivalent.

        State layout contract (both paths): ``initial_state`` is consumed
        and ``final_state`` returned in the V-first ``[N, H, V, K]`` layout —
        the layout of the executor's ssm pool and of the fused decode
        kernel. The in-tree prefill op natively uses the FLA-default K-first
        ``[N, H, K, V]`` layout, so the optimized path transposes at both
        boundaries (K == V == 128 for Kimi K3, so shapes alone cannot catch
        a mix-up — the transpose is semantic).
        """
        use_optimized = self.prefill_kernel_path == "optimized"
        chunk_indices = None
        if use_optimized and cu_seqlens is not None:
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

            if initial_state is not None:
                # Pool V-first [N, H, V, K] -> op K-first [N, H, K, V].
                initial_state = initial_state.transpose(-1, -2).contiguous()

            out, final_state = torch.ops.trtllm.kda_prefill(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=chunk_size,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                use_gate_in_kernel=True,
                A_log=A_log_kernel,
                dt_bias=dt_bias_kernel,
            )
            if out.shape[1] != real_T:
                out = out[:, :real_T]
            if final_state is not None and final_state.numel() > 0:
                # Op K-first [N, H, K, V] -> pool V-first [N, H, V, K].
                # .contiguous() also detaches the result from the op's
                # shared per-shape S_out scratch, which the next same-shape
                # call overwrites.
                final_state = final_state.transpose(-1, -2).contiguous()
            return out, final_state

        from fla.ops.kda import chunk_kda

        o, final_state = chunk_kda(
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
        return o, final_state

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
