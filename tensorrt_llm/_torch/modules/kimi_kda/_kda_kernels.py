# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel dispatch for the in-tree KDA module.

The optimized ``KDA_prefill`` (CuTe / cuTile + Triton chunked prefill) and
``KDA_decode`` (fused CUDA C++ single-token decode) kernels live in the
NVIDIA+Moonshot internal collaboration collection at
``exisiting_optimization_work/`` and are Blackwell sm_100 only. They are not
open-sourced; the KDA module imports them via env-configurable paths so the
production tree carries no source copies.

On sm_100 with the optimization tree available the dispatch runs the
optimized kernels. On any other arch (or when the optimization tree is
unreachable at runtime) the dispatch falls back to FLA's ``chunk_kda`` /
``fused_recurrent_kda`` on-device references.

Callers are the ``KimiKDALinearAttention`` module. The module owns the
HF-parity semantics (Q/K/V projections, conv, gating, ``o_norm``,
``o_proj``); the dispatch here only wraps the kernel-level computation of
the delta-rule inner loop plus its state update.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from types import ModuleType
from typing import Optional, Tuple

import torch

try:
    from tensorrt_llm._utils import get_sm_version as _tllm_get_sm_version
except Exception:  # pragma: no cover — source-loader stub path
    _tllm_get_sm_version = None


KIMI_KDA_OPTIMIZED_KERNEL_ENV = "KIMI_KDA_OPTIMIZED_KERNEL_DIR"
"""Environment variable pointing at the ``exisiting_optimization_work`` root.

The KDA module resolves the ``KDA_prefill/benchmark`` and ``KDA_decode``
sub-directories from this root. If unset the module falls back to the FLA
reference path.
"""


def _resolve_optimization_root() -> Optional[str]:
    root = os.environ.get(KIMI_KDA_OPTIMIZED_KERNEL_ENV)
    if root and os.path.isdir(root):
        return root
    return None


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
        except Exception:
            return _default_get_sm_version()
    return _default_get_sm_version()


def is_kda_optimized_supported() -> bool:
    """The optimized prefill/decode kernels are Blackwell sm_100 only."""
    return get_kda_sm_version() in (100, 103)


# ---------------------------------------------------------------------------
# KDA_prefill kernel loading (CuTe / cuTile + Triton chunked prefill).
# ---------------------------------------------------------------------------

_PREFILL_MODULE: Optional[ModuleType] = None


def _load_prefill_module(root: str) -> ModuleType:
    global _PREFILL_MODULE
    if _PREFILL_MODULE is not None:
        return _PREFILL_MODULE

    benchmark_dir = os.path.join(root, "KDA_prefill", "benchmark")
    chunk_fwd_path = os.path.join(benchmark_dir, "chunk_fwd.py")
    if not os.path.isfile(chunk_fwd_path):
        raise FileNotFoundError(
            f"KDA_prefill chunk_fwd not found at {chunk_fwd_path}. "
            f"Set {KIMI_KDA_OPTIMIZED_KERNEL_ENV} to point at "
            "exisiting_optimization_work/."
        )
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)

    spec = importlib.util.spec_from_file_location(
        "tensorrt_llm._torch.modules.kimi_kda._optimized_kda_prefill",
        chunk_fwd_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {chunk_fwd_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _PREFILL_MODULE = module
    return module


def _load_fla_chunk_kda() -> ModuleType:
    return importlib.import_module("fla.ops.kda")


# ---------------------------------------------------------------------------
# KDA_decode kernel loading (fused CUDA extension).
# ---------------------------------------------------------------------------

_DECODE_MODULE: Optional[ModuleType] = None


def _load_decode_module(root: str) -> ModuleType:
    global _DECODE_MODULE
    if _DECODE_MODULE is not None:
        return _DECODE_MODULE

    decode_dir = os.path.join(root, "KDA_decode")
    if not os.path.isdir(decode_dir):
        raise FileNotFoundError(
            f"KDA_decode dir not found at {decode_dir}. "
            f"Set {KIMI_KDA_OPTIMIZED_KERNEL_ENV} to point at "
            "exisiting_optimization_work/."
        )
    if decode_dir not in sys.path:
        sys.path.insert(0, decode_dir)
    module = importlib.import_module("kda_decode_fusion_cuda")
    _DECODE_MODULE = module
    return module


# ---------------------------------------------------------------------------
# Dispatch API used by the KimiKDALinearAttention module.
# ---------------------------------------------------------------------------


class KDAKernelDispatch:
    """Kernel dispatch state for one ``KimiKDALinearAttention`` instance.

    Attributes
    ----------
    kernel_path : str
        ``"optimized"`` when the sm_100 CuTe/Triton chunked prefill and fused
        CUDA decode kernels are selected, ``"fla"`` when the FLA references
        are selected.
    optimization_root : Optional[str]
        Root path of the ``exisiting_optimization_work`` collection when
        available, else ``None``.

    Notes
    -----
    Dispatch is decided at construction time using
    ``is_kda_optimized_supported()`` and the presence of the optimization
    collection. Callers can force the fallback by constructing with
    ``force_use_fallback=True``.
    """

    def __init__(self, force_use_fallback: bool = False) -> None:
        self.force_use_fallback = force_use_fallback
        root = _resolve_optimization_root()
        if force_use_fallback or root is None or not is_kda_optimized_supported():
            self.kernel_path = "fla"
            self.optimization_root: Optional[str] = None
            self._prefill_module: Optional[ModuleType] = None
            self._decode_module: Optional[ModuleType] = None
        else:
            self.kernel_path = "optimized"
            self.optimization_root = root
            self._prefill_module = None
            self._decode_module = None

    def get_prefill_source(self) -> str:
        if self.kernel_path == "optimized":
            module = _load_prefill_module(self.optimization_root)
            self._prefill_module = module
            return module.__file__
        return _load_fla_chunk_kda().__file__ or "<fla.ops.kda>"

    def get_decode_source(self) -> str:
        if self.kernel_path == "optimized":
            module = _load_decode_module(self.optimization_root)
            self._decode_module = module
            return module.__file__
        return _load_fla_chunk_kda().__file__ or "<fla.ops.kda>"

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
        ``use_gate_in_kernel``, and ``transpose_state_layout``; then
        dispatches to the CuTe/Triton chunked forward.

        On the FLA path it calls ``fla.ops.kda.chunk_kda`` directly with the
        matching flags so the semantics are byte-equivalent.
        """
        if self.kernel_path == "optimized":
            from fla.modules.l2norm import l2norm_fwd
            from fla.ops.common.gate import fused_beta_sigmoid
            from fla.ops.utils.index import prepare_chunk_indices

            q, _ = l2norm_fwd(q)
            k, _ = l2norm_fwd(k)
            beta = fused_beta_sigmoid(beta, scale=1.0).to(torch.bfloat16)

            chunk_indices = None
            if cu_seqlens is not None:
                chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

            A_log_kernel = A_log.detach() if A_log is not None else None
            dt_bias_kernel = dt_bias.detach() if dt_bias is not None else None

            module = _load_prefill_module(self.optimization_root)
            self._prefill_module = module

            out_tuple = module.chunk_kda_fwd(
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
            return out_tuple[0], out_tuple[1]

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
            transpose_state_layout=True,
            cu_seqlens=cu_seqlens,
        )
        return o, final_state

    def decode_kda(self, **kwargs) -> torch.Tensor:
        """Run the fused KDA single-token decode kernel.

        This is only defined on the optimized path; the FLA fallback runs
        ``fla.ops.kda.fused_recurrent_kda`` directly and does not use this
        wrapper (see ``_decode_via_fla`` on the module).
        """
        if self.kernel_path != "optimized":
            raise RuntimeError(
                "decode_kda called on non-optimized path; use FLA path via "
                "the module's fallback handling instead."
            )
        module = _load_decode_module(self.optimization_root)
        self._decode_module = module
        return module.run_kda_decode_fusion_cuda(**kwargs)

    def precompile_decode(self, verbose: bool = False) -> None:
        """Trigger the KDA_decode extension JIT build early.

        On the FLA fallback path this is a no-op.
        """
        if self.kernel_path != "optimized":
            return
        module = _load_decode_module(self.optimization_root)
        self._decode_module = module
        if hasattr(module, "precompile_kda_decode_fusion_extension"):
            module.precompile_kda_decode_fusion_extension(verbose=verbose)
