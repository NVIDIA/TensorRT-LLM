# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import enum
import functools
import json
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch
from packaging.version import InvalidVersion, Version

from tensorrt_llm.logger import logger

from ..flashinfer_utils import get_env_enable_pdl

_BACKEND_ENV = "TRTLLM_LOW_M_GEMM_BACKEND"
_SHAPE_LOG_ENV = "TRTLLM_LOW_M_GEMM_SHAPE_LOG"
_SHAPE_LOG_SCHEMA_VERSION = 1
_FLASHINFER_BACKEND = "cute-dsl"
_MIN_FLASHINFER_VERSION = Version("0.6.17.dev20260806")
_SUPPORTED_SMS = {100, 103}
_MAX_FLASHINFER_M = 32


class LowMGemmBackend(str, enum.Enum):
    """Runtime choices exposed by the low-M BF16 dispatcher."""

    OFF = "off"
    AUTO = "auto"
    FLASHINFER = "flashinfer"
    CUBLAS = "cublas"


def _get_rank() -> int:
    for name in ("RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 0


def _get_world_size() -> int:
    for name in ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 1


def _rank_path(path: str) -> Path:
    rank = _get_rank()
    if "{rank}" in path:
        return Path(path.format(rank=rank))
    resolved = Path(path)
    if _get_world_size() > 1:
        return resolved.with_name(f"{resolved.name}.rank{rank}")
    return resolved


def _normalize_backend(value: str) -> LowMGemmBackend:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "": LowMGemmBackend.OFF,
        "0": LowMGemmBackend.OFF,
        "none": LowMGemmBackend.OFF,
        "off": LowMGemmBackend.OFF,
        "auto": LowMGemmBackend.AUTO,
        "flashinfer": LowMGemmBackend.FLASHINFER,
        "flashinfer-cute-dsl": LowMGemmBackend.FLASHINFER,
        "cute-dsl": LowMGemmBackend.FLASHINFER,
        "cublas": LowMGemmBackend.CUBLAS,
        "cublaslt": LowMGemmBackend.CUBLAS,
    }
    if normalized not in aliases:
        choices = ", ".join(backend.value for backend in LowMGemmBackend)
        raise ValueError(f"Invalid {_BACKEND_ENV}={value!r}; expected one of {choices}.")
    return aliases[normalized]


def _configured_backend() -> LowMGemmBackend:
    return _normalize_backend(os.environ.get(_BACKEND_ENV, "off"))


@functools.cache
def _device_sm(device_index: int) -> int:
    major, minor = torch.cuda.get_device_capability(device_index)
    return major * 10 + minor


def _current_sm(device: torch.device) -> int:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _device_sm(device_index)


def _prefer_cublas_for_auto(m: int, n: int, k: int, sm: int) -> bool:
    """Return whether cuBLAS wins for a measured low-M Blackwell shape.

    FlashInfer selects its direct versus split-K tactic internally.  This
    narrow crossover handles shapes where the packaged CuTe DSL kernels are
    slower than the normal cuBLAS Linear path in the 2026-08-06 nightly.
    Explicit ``flashinfer`` selection remains an unconditional override.
    """

    if sm != 103:
        return False
    if (n, k) == (8192, 128):
        return m >= 8
    if (n, k) == (15520, 8192):
        return m >= 15
    return (m, n, k) == (16, 2304, 8192)


class _ShapeCollector:
    """Debug-only hot-shape collector with durable runtime-shape discovery."""

    def __init__(self, output_path: str):
        self.output_path = _rank_path(output_path)
        self._counts: dict[tuple[Any, ...], int] = defaultdict(int)
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._persisted_problem_keys: set[tuple[Any, ...]] = set()
        self._pending_problem_keys: set[tuple[Any, ...]] = set()
        self._warmup_complete = False
        atexit.register(self.flush)

    def record(
        self,
        module: torch.nn.Module,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        cuda_graph: bool,
    ) -> None:
        input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
        layer = getattr(module, "_low_m_gemm_name", module.__class__.__name__)
        op_name = _infer_op_name(layer)
        batch_shape = tuple(int(dim) for dim in input_tensor.shape[:-1])
        key = (
            int(input_2d.shape[0]),
            int(weight.shape[0]),
            int(weight.shape[1]),
            str(input_tensor.dtype).removeprefix("torch."),
            False,
            True,
            cuda_graph,
            bias is not None,
            batch_shape,
            layer,
            op_name,
        )
        problem_key = key[:8]
        with self._lock:
            self._counts[key] += 1
            flush_new_runtime_shape = (
                self._warmup_complete
                and problem_key not in self._persisted_problem_keys
                and problem_key not in self._pending_problem_keys
            )
            if flush_new_runtime_shape:
                self._pending_problem_keys.add(problem_key)
        if flush_new_runtime_shape:
            # SRT terminates serving workers after a benchmark, so their atexit
            # handlers are not reliable. Persist each shape first discovered
            # after warmup immediately. This collector is debug-only and is
            # never enabled for a performance measurement.
            try:
                self.flush()
            finally:
                with self._lock:
                    self._pending_problem_keys.discard(problem_key)

    def flush(self, *, mark_warmup_complete: bool = False) -> None:
        with self._flush_lock:
            with self._lock:
                if mark_warmup_complete:
                    self._warmup_complete = True
                counts = dict(self._counts)
            if not counts:
                return
            rows = []
            for key, count in counts.items():
                (
                    m,
                    n,
                    k,
                    dtype,
                    trans_a,
                    trans_b,
                    cuda_graph,
                    has_bias,
                    batch_shape,
                    layer,
                    op_name,
                ) = key
                rows.append(
                    {
                        "m": m,
                        "n": n,
                        "k": k,
                        "dtype": dtype,
                        "trans_a": trans_a,
                        "trans_b": trans_b,
                        "cuda_graph": cuda_graph,
                        "has_bias": has_bias,
                        "batch_shape": list(batch_shape),
                        "layer": layer,
                        "op_name": op_name,
                        "call_count": count,
                    }
                )
            rows.sort(
                key=lambda row: (
                    -row["call_count"],
                    row["m"],
                    row["n"],
                    row["k"],
                    row["layer"],
                )
            )
            payload = {
                "schema_version": _SHAPE_LOG_SCHEMA_VERSION,
                "rank": _get_rank(),
                "world_size": _get_world_size(),
                "shapes": rows,
            }
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.output_path.with_suffix(f"{self.output_path.suffix}.tmp")
            temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            temporary_path.replace(self.output_path)
            with self._lock:
                self._persisted_problem_keys.update(key[:8] for key in counts)


def _infer_op_name(layer: str) -> str:
    lowered = layer.lower()
    patterns = (
        ("mtp", "mtp"),
        ("lm_head", "lm_head"),
        ("lmhead", "lm_head"),
        ("qkv", "attention_qkv"),
        ("q_proj", "attention_qkv"),
        ("k_proj", "attention_qkv"),
        ("v_proj", "attention_qkv"),
        ("o_proj", "attention_o_proj"),
        ("out_proj", "out_proj"),
        ("in_proj", "in_proj"),
        ("shared_expert", "shared_expert"),
    )
    for needle, op_name in patterns:
        if needle in lowered:
            return op_name
    # Appended MTP fusion projections may be named simply ``fc``.
    if lowered.endswith(".fc"):
        return "mtp_fusion_fc"
    return "linear"


class LowMGemmDispatcher:
    """Dispatch eligible BF16 GEMMs to FlashInfer's low-M CuTe DSL backend."""

    def __init__(self) -> None:
        self.backend = _configured_backend()
        self.cuda_graph_enabled = True
        self._prepared = False
        self._flashinfer_mm = None
        self._engaged: set[tuple[int, int, int, bool]] = set()
        self._collector = (
            _ShapeCollector(os.environ[_SHAPE_LOG_ENV]) if os.environ.get(_SHAPE_LOG_ENV) else None
        )

    @property
    def enabled(self) -> bool:
        return self.backend not in (LowMGemmBackend.OFF, LowMGemmBackend.CUBLAS)

    def prepare(self, model: torch.nn.Module, cuda_graph_enabled: bool) -> None:
        self.cuda_graph_enabled = cuda_graph_enabled
        for name, module in model.named_modules():
            if module.__class__.__name__.endswith("Linear"):
                setattr(module, "_low_m_gemm_name", name)

        if self._prepared or not self.enabled:
            self._prepared = True
            return

        self._prepare_flashinfer()
        self._prepared = True

    def _prepare_flashinfer(self) -> None:
        try:
            import flashinfer as flashinfer_module
            from flashinfer import mm_bf16
            from flashinfer.cute_dsl.utils import is_cute_dsl_available
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "FlashInfer low-M GEMM requires flashinfer-python "
                f">={_MIN_FLASHINFER_VERSION} and nvidia-cutlass-dsl."
            ) from error
        version_value = getattr(flashinfer_module, "__version__", "unknown")
        try:
            flashinfer_version = Version(version_value)
        except InvalidVersion as error:
            raise RuntimeError(f"FlashInfer has an invalid version: {version_value!r}.") from error
        if flashinfer_version < _MIN_FLASHINFER_VERSION:
            raise RuntimeError(
                "FlashInfer low-M GEMM requires flashinfer-python "
                f">={_MIN_FLASHINFER_VERSION}; found {flashinfer_version}."
            )
        if not is_cute_dsl_available():
            raise RuntimeError("FlashInfer low-M GEMM requires nvidia-cutlass-dsl.")
        if not callable(getattr(mm_bf16, "is_backend_supported", None)):
            raise RuntimeError(
                "Installed FlashInfer mm_bf16 has no backend capability API; "
                "the required PR #4266 implementation is not active."
            )
        self._flashinfer_mm = mm_bf16
        logger.info(
            "FlashInfer low-M BF16 GEMM is using the packaged cache-free "
            f"direct/split-K heuristic from flashinfer-python {flashinfer_version}."
        )

    def _is_candidate_shape(
        self, input_tensor: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> bool:
        # FlashInfer's DLPack bridge is inference-only and intentionally does
        # not build an autograd graph. Preserve normal Linear semantics for
        # training or any grad-enabled caller.
        if (
            torch.is_grad_enabled()
            or input_tensor.ndim < 1
            or weight.ndim != 2
            or not input_tensor.is_cuda
            or not weight.is_cuda
            or input_tensor.device != weight.device
            or input_tensor.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or not input_tensor.is_contiguous()
            or not weight.is_contiguous()
        ):
            return False
        k = int(input_tensor.shape[-1])
        if k <= 0 or int(weight.shape[1]) != k or k % 128:
            return False
        m = input_tensor.numel() // k
        if not 1 <= m <= _MAX_FLASHINFER_M:
            return False
        if input_tensor.data_ptr() % 32 or weight.data_ptr() % 32:
            return False
        if bias is not None and (
            not bias.is_cuda
            or bias.device != input_tensor.device
            or bias.dtype != torch.bfloat16
            or bias.shape != (weight.shape[0],)
            or not bias.is_contiguous()
        ):
            return False
        sm = _current_sm(input_tensor.device)
        if sm not in _SUPPORTED_SMS:
            return False
        return True

    def _is_flashinfer_supported(self, device: torch.device) -> bool:
        if self._flashinfer_mm is None:
            self._prepare_flashinfer()
        sm = _current_sm(device)
        return bool(self._flashinfer_mm.is_backend_supported(_FLASHINFER_BACKEND, sm))

    def apply(
        self,
        module: torch.nn.Module,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (
            self._collector is not None
            and input_tensor.is_cuda
            and input_tensor.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
        ):
            self._collector.record(module, input_tensor, weight, bias, self.cuda_graph_enabled)
        if not self.enabled:
            return None
        if not self._prepared:
            # Unit tests and direct module users do not construct ModelEngine.
            self.prepare(module, cuda_graph_enabled=False)
        if not self._is_candidate_shape(input_tensor, weight, bias):
            return None

        input_2d = input_tensor.detach().view(-1, input_tensor.shape[-1])
        m = int(input_2d.shape[0])
        n = int(weight.shape[0])
        k = int(weight.shape[1])
        sm = _current_sm(input_tensor.device)
        if self.backend == LowMGemmBackend.AUTO and _prefer_cublas_for_auto(m, n, k, sm):
            return None

        if not self._is_flashinfer_supported(input_tensor.device):
            return None

        engaged_key = (m, n, k, bias is not None)
        if engaged_key not in self._engaged:
            self._engaged.add(engaged_key)
            logger.info(
                f"Low-M BF16 GEMM: routing M={m} N={n} K={k} "
                f"bias={bias is not None} to FlashInfer {_FLASHINFER_BACKEND}'s "
                "packaged direct/split-K heuristic."
            )

        inference_weight = weight.detach()
        inference_bias = bias.detach() if bias is not None else None
        output = self._flashinfer_mm(
            input_2d,
            inference_weight.t(),
            bias=inference_bias,
            pdl=get_env_enable_pdl(),
            out_dtype=torch.bfloat16,
            backend=_FLASHINFER_BACKEND,
        )
        return output.view(*input_tensor.shape[:-1], weight.shape[0])

    def flush_shape_log(self) -> None:
        if self._collector is not None:
            self._collector.flush(mark_warmup_complete=True)


_DISPATCHER = LowMGemmDispatcher()
LOW_M_GEMM_ACTIVE = _DISPATCHER.enabled or _DISPATCHER._collector is not None


def prepare_low_m_gemm(model: torch.nn.Module, cuda_graph_enabled: bool) -> None:
    """Label Linear modules and initialize the cache-free FlashInfer path."""

    _DISPATCHER.prepare(model, cuda_graph_enabled)


def apply_low_m_gemm(
    module: torch.nn.Module,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return a low-M BF16 result, or ``None`` for the normal path."""

    return _DISPATCHER.apply(module, input_tensor, weight, bias)


def flush_low_m_gemm_shape_log() -> None:
    """Persist the debug shape inventory at a deterministic warmup boundary."""

    _DISPATCHER.flush_shape_log()


__all__ = [
    "LOW_M_GEMM_ACTIVE",
    "LowMGemmBackend",
    "apply_low_m_gemm",
    "flush_low_m_gemm_shape_log",
    "prepare_low_m_gemm",
]
