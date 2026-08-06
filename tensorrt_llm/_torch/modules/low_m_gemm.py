# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import enum
import hashlib
import json
import os
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.version import __version__ as trtllm_version

from ..flashinfer_utils import get_env_enable_pdl

_BACKEND_ENV = "TRTLLM_LOW_M_GEMM_BACKEND"
_DISPATCH_CACHE_ENV = "TRTLLM_LOW_M_GEMM_TUNING_CACHE"
_FLASHINFER_CACHE_ENV = "TRTLLM_FLASHINFER_AUTOTUNER_CACHE"
_FLASHINFER_COMMIT_ENV = "TRTLLM_FLASHINFER_COMMIT"
_FLASHINFER_SOURCE_ROOT_ENV = "TRTLLM_FLASHINFER_SOURCE_ROOT"
_DISABLE_CACHE_ENV = "TRTLLM_DISABLE_GEMM_TUNING_CACHE"
_SHAPE_LOG_ENV = "TRTLLM_LOW_M_GEMM_SHAPE_LOG"
_DISPATCH_CACHE_SCHEMA_VERSION = 3
_SHAPE_LOG_SCHEMA_VERSION = 1
_FLASHINFER_BACKEND = "cute-dsl"
_SUPPORTED_SMS = {100, 103}
_MAX_FLASHINFER_M = 32


class LowMGemmBackend(str, enum.Enum):
    """Runtime choices exposed by the low-M BF16 dispatcher."""

    OFF = "off"
    AUTO = "auto"
    FLASHINFER = "flashinfer"
    CUBLAS = "cublas"


@dataclass(frozen=True)
class GemmDispatchKey:
    """Properties that can change the best low-M GEMM implementation."""

    sm: int
    m: int
    n: int
    k: int
    a_type: str = "bf16"
    b_type: str = "bf16"
    c_type: str = "bf16"
    trans_a: bool = False
    trans_b: bool = True
    has_bias: bool = False
    cuda_graph: bool = True

    def cache_key(self) -> str:
        transpose = ("t" if self.trans_a else "n") + ("t" if self.trans_b else "n")
        bias = "bias" if self.has_bias else "nobias"
        execution = "graph" if self.cuda_graph else "eager"
        return (
            f"sm{self.sm}:{self.a_type}:{self.m}x{self.n}x{self.k}:{transpose}:{bias}:{execution}"
        )


@dataclass(frozen=True)
class GemmTuningResult:
    """One persisted dispatcher decision produced by offline tuning."""

    backend: str
    algorithm: Optional[str] = None
    tactic: Optional[dict[str, Any]] = None
    latency_us: Optional[float] = None
    baseline_us: Optional[float] = None
    measurements: Optional[dict[str, Any]] = None


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as cache_file:
        for chunk in iter(lambda: cache_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _flashinfer_commit(flashinfer_module: Any) -> Optional[str]:
    """Return a verifiable FlashInfer source revision for cache provenance."""

    package_commit = getattr(flashinfer_module, "__git_version__", None)
    if package_commit == "unknown":
        package_commit = None
    declared_commit = os.environ.get(_FLASHINFER_COMMIT_ENV)
    if package_commit is not None and declared_commit is not None:
        if package_commit != declared_commit:
            raise RuntimeError(
                f"{_FLASHINFER_COMMIT_ENV}={declared_commit} does not match "
                f"the imported FlashInfer revision {package_commit}."
            )
    return package_commit or declared_commit


def _current_sm(device: torch.device) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def _configure_flashinfer_source_layout() -> None:
    source_value = os.environ.get(_FLASHINFER_SOURCE_ROOT_ENV)
    if not source_value:
        return
    source_root = Path(source_value).resolve()
    source_csrc = source_root / "csrc"
    source_include = source_root / "include"
    if not source_csrc.is_dir() or not source_include.is_dir():
        raise RuntimeError(
            f"{_FLASHINFER_SOURCE_ROOT_ENV} is not a FlashInfer source checkout: {source_root}"
        )

    import flashinfer
    from flashinfer.jit import env as jit_env

    imported_root = Path(flashinfer.__file__).resolve().parents[1]
    if imported_root != source_root:
        raise RuntimeError(
            f"Imported FlashInfer from {imported_root}, but "
            f"{_FLASHINFER_SOURCE_ROOT_ENV}={source_root}."
        )
    # An installed wheel stores build inputs under flashinfer/data. A source
    # overlay stores them at repository top level. Configure this before any
    # FlashInfer JIT helper is first invoked.
    jit_env.FLASHINFER_CSRC_DIR = source_csrc
    jit_env.FLASHINFER_INCLUDE_DIR = source_include


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
    """Offline-cache dispatcher for FlashInfer direct/split-K versus cuBLAS."""

    def __init__(self) -> None:
        self.backend = _configured_backend()
        if self.backend in (LowMGemmBackend.AUTO, LowMGemmBackend.FLASHINFER):
            _configure_flashinfer_source_layout()
        self.cuda_graph_enabled = True
        self._dispatch_cache: dict[str, GemmTuningResult] = {}
        self._dispatch_metadata: dict[str, Any] = {}
        self._prepared = False
        self._flashinfer_mm = None
        self._flashinfer_direct_tactic = None
        self._flashinfer_run_direct = None
        self._flashinfer_splitk_tactic = None
        self._flashinfer_run_splitk = None
        self._engaged: set[tuple[str, int, int, int, bool]] = set()
        self._cache_misses: set[str] = set()
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

        self._load_dispatch_cache()
        needs_flashinfer = self.backend == LowMGemmBackend.FLASHINFER or any(
            result.backend == LowMGemmBackend.FLASHINFER.value
            for result in self._dispatch_cache.values()
        )
        if needs_flashinfer:
            self._prepare_flashinfer()
        self._prepared = True

    def _load_dispatch_cache(self) -> None:
        if self.backend != LowMGemmBackend.AUTO:
            return
        if os.environ.get(_DISABLE_CACHE_ENV, "0") == "1":
            logger.warning(
                f"{_DISABLE_CACHE_ENV}=1: low-M GEMM auto mode will use its cache-miss heuristic."
            )
            return
        cache_value = os.environ.get(_DISPATCH_CACHE_ENV)
        if not cache_value:
            raise RuntimeError(
                f"{_BACKEND_ENV}=auto requires {_DISPATCH_CACHE_ENV}; run the "
                "offline tuner first, use flashinfer for heuristic-only testing, "
                "or set cublas to roll back."
            )
        cache_path = Path(cache_value)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Low-M GEMM dispatch cache does not exist: {cache_path}")
        document = json.loads(cache_path.read_text(encoding="utf-8"))
        if document.get("schema_version") != _DISPATCH_CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported low-M GEMM cache schema in {cache_path}: "
                f"{document.get('schema_version')!r}"
            )
        self._dispatch_metadata = document.get("metadata", {})
        if not isinstance(self._dispatch_metadata, dict):
            raise RuntimeError(f"Low-M GEMM cache {cache_path} has invalid metadata.")
        expected_pdl = self._dispatch_metadata.get("pdl")
        if not isinstance(expected_pdl, bool):
            raise RuntimeError(
                f"Low-M GEMM cache {cache_path} has no Boolean PDL runtime identity."
            )
        actual_pdl = bool(get_env_enable_pdl())
        if expected_pdl != actual_pdl:
            raise RuntimeError(
                f"Low-M GEMM cache was tuned with PDL={expected_pdl}, but runtime PDL={actual_pdl}."
            )
        expected_cuda_version = self._dispatch_metadata.get("cuda_version")
        if not isinstance(expected_cuda_version, str):
            raise RuntimeError(f"Low-M GEMM cache {cache_path} has no CUDA runtime identity.")
        actual_cuda_version = torch.version.cuda or "none"
        if expected_cuda_version != actual_cuda_version:
            raise RuntimeError(
                "CUDA version does not match the low-M GEMM dispatch cache: "
                f"expected {expected_cuda_version}, got {actual_cuda_version}."
            )
        entries = document.get("entries")
        if not isinstance(entries, dict):
            raise RuntimeError(f"Low-M GEMM cache {cache_path} has no object-valued entries.")
        self._dispatch_cache = {key: GemmTuningResult(**value) for key, value in entries.items()}
        logger.info(f"Loaded {len(self._dispatch_cache)} low-M GEMM decisions from {cache_path}.")

    def _prepare_flashinfer(self) -> None:
        try:
            import flashinfer as flashinfer_module
            from flashinfer import mm_bf16
            from flashinfer.cute_dsl.utils import is_cute_dsl_available
            from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
                DirectTactic,
                run_direct_dense,
            )
            from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
                SplitKTactic,
                run_splitk_dense,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "FlashInfer low-M GEMM requires a FlashInfer build containing "
                "PR #4266 and nvidia-cutlass-dsl."
            ) from error
        if not is_cute_dsl_available():
            raise RuntimeError("FlashInfer low-M GEMM requires nvidia-cutlass-dsl.")
        if not callable(getattr(mm_bf16, "is_backend_supported", None)):
            raise RuntimeError(
                "Installed FlashInfer mm_bf16 has no backend capability API; "
                "the pinned PR #4266 build is not active."
            )
        expected_arch = self._dispatch_metadata.get("gpu_arch")
        actual_arch = f"sm{_current_sm(torch.device('cuda'))}"
        if expected_arch is not None and expected_arch != actual_arch:
            raise RuntimeError(
                f"Low-M GEMM cache targets {expected_arch}, but this rank uses {actual_arch}."
            )
        expected_version = self._dispatch_metadata.get("flashinfer_version")
        if expected_version is not None and expected_version != flashinfer_module.__version__:
            raise RuntimeError(
                "FlashInfer version does not match the low-M GEMM dispatch "
                f"cache: expected {expected_version}, got "
                f"{flashinfer_module.__version__}."
            )
        expected_trtllm_version = self._dispatch_metadata.get("trtllm_version")
        if expected_trtllm_version is not None and expected_trtllm_version != trtllm_version:
            raise RuntimeError(
                "TensorRT-LLM version does not match the low-M GEMM dispatch "
                f"cache: expected {expected_trtllm_version}, got "
                f"{trtllm_version}."
            )
        expected_dispatcher_digest = self._dispatch_metadata.get("trtllm_low_m_gemm_sha256")
        if expected_dispatcher_digest is not None:
            actual_dispatcher_digest = _sha256(Path(__file__))
            if actual_dispatcher_digest != expected_dispatcher_digest:
                raise RuntimeError(
                    "TensorRT-LLM low-M GEMM source does not match the offline "
                    "dispatch cache; regenerate the cache with this checkout."
                )
        expected_commit = self._dispatch_metadata.get("flashinfer_commit")
        actual_commit = _flashinfer_commit(flashinfer_module)
        if (
            expected_commit is not None
            and self.backend == LowMGemmBackend.AUTO
            and actual_commit != expected_commit
        ):
            raise RuntimeError(
                "Imported FlashInfer does not match the cache's pinned "
                f"commit {expected_commit}; got {actual_commit!r}."
            )
        self._flashinfer_mm = mm_bf16
        self._flashinfer_direct_tactic = DirectTactic
        self._flashinfer_run_direct = run_direct_dense
        self._flashinfer_splitk_tactic = SplitKTactic
        self._flashinfer_run_splitk = run_splitk_dense

        cache_disabled = os.environ.get(_DISABLE_CACHE_ENV, "0") == "1"
        cache_value = os.environ.get(_FLASHINFER_CACHE_ENV)
        if cache_disabled or not cache_value:
            if self.backend == LowMGemmBackend.AUTO and not cache_disabled:
                raise RuntimeError(
                    f"{_BACKEND_ENV}=auto selected FlashInfer entries but "
                    f"{_FLASHINFER_CACHE_ENV} is unset."
                )
            logger.warning(
                "FlashInfer BF16 autotune cache is unavailable for provenance "
                "validation; forced/cache-miss routing will use FlashInfer's "
                "heuristic."
            )
            return

        cache_path = Path(cache_value)
        if not cache_path.is_file():
            raise FileNotFoundError(f"FlashInfer autotune cache does not exist: {cache_path}")
        expected_digest = self._dispatch_metadata.get("flashinfer_cache_sha256")
        if expected_digest is not None:
            actual_digest = _sha256(cache_path)
            if actual_digest != expected_digest:
                raise RuntimeError(
                    "FlashInfer autotune cache checksum does not match the "
                    "TRT-LLM dispatch cache; regenerate the pair together."
                )
        # Do not load this file into FlashInfer's runtime AutoTuner. Its
        # default mapper aliases M=24 to M=32, whereas this dispatch cache is
        # exact-M. The selected runner/tactic is launched directly below.
        logger.info(f"Validated paired FlashInfer BF16 autotune cache {cache_path}.")

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

    @staticmethod
    def _tactic_values(
        result: GemmTuningResult,
        expected_fields: tuple[str, ...],
    ) -> dict[str, int]:
        tactic = result.tactic
        if not isinstance(tactic, dict) or set(tactic) != set(expected_fields):
            raise RuntimeError(
                f"Invalid {result.algorithm!r} tactic in low-M GEMM cache: "
                f"expected {expected_fields}, got {tactic!r}."
            )
        return {field: int(tactic[field]) for field in expected_fields}

    def _launch_cached_flashinfer(
        self,
        input_2d: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        result: GemmTuningResult,
    ) -> torch.Tensor:
        output = torch.empty(
            (input_2d.shape[0], weight.shape[0]),
            device=input_2d.device,
            dtype=torch.bfloat16,
        )
        weight_t = weight.t()
        pdl = get_env_enable_pdl()
        if result.algorithm == "simt":
            if bias is not None:
                raise RuntimeError("FlashInfer direct/SIMT GEMM does not support bias.")
            values = self._tactic_values(
                result,
                ("block_size", "outputs_per_block", "rows_per_block"),
            )
            tactic = self._flashinfer_direct_tactic(**values)
            self._flashinfer_run_direct(input_2d, weight_t, output, pdl, tactic)
            return output
        if result.algorithm == "splitk":
            values = self._tactic_values(
                result,
                ("mma_m", "mma_n", "split_k", "ab_stages"),
            )
            tactic = self._flashinfer_splitk_tactic(**values)
            self._flashinfer_run_splitk(input_2d, weight_t, bias, output, pdl, tactic)
            return output
        raise RuntimeError(
            f"FlashInfer cache entry has unsupported algorithm {result.algorithm!r}."
        )

    def _make_key(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> GemmDispatchKey:
        input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
        return GemmDispatchKey(
            sm=_current_sm(input_tensor.device),
            m=int(input_2d.shape[0]),
            n=int(weight.shape[0]),
            k=int(weight.shape[1]),
            has_bias=bias is not None,
            cuda_graph=self.cuda_graph_enabled,
        )

    def _select_backend(self, key: GemmDispatchKey) -> LowMGemmBackend:
        if self.backend != LowMGemmBackend.AUTO:
            return self.backend
        if os.environ.get(_DISABLE_CACHE_ENV, "0") != "1":
            result = self._dispatch_cache.get(key.cache_key())
            if result is not None:
                return _normalize_backend(result.backend)
        # FlashInfer's measured fallback chooses direct for the smallest shapes
        # and split-K for the rest of M<=32. This is intentionally only a cache
        # miss policy; offline tuning should cover every hot production shape.
        if key.cache_key() not in self._cache_misses:
            self._cache_misses.add(key.cache_key())
            logger.warning(
                f"No offline low-M GEMM decision for {key.cache_key()}; using "
                "the FlashInfer direct/split-K fallback heuristic."
            )
        return LowMGemmBackend.FLASHINFER

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

        key = self._make_key(input_tensor, weight, bias)
        selected = self._select_backend(key)
        if selected == LowMGemmBackend.CUBLAS:
            return None
        if selected != LowMGemmBackend.FLASHINFER:
            raise RuntimeError(f"Unsupported cached low-M GEMM backend: {selected.value}")
        if not self._is_flashinfer_supported(input_tensor.device):
            return None

        engaged_key = (selected.value, key.m, key.n, key.k, key.has_bias)
        if engaged_key not in self._engaged:
            self._engaged.add(engaged_key)
            result = self._dispatch_cache.get(key.cache_key())
            detail = ""
            if result is not None and result.algorithm:
                detail = f" ({result.algorithm})"
            logger.info(
                f"Low-M BF16 GEMM: routing M={key.m} N={key.n} K={key.k} "
                f"bias={key.has_bias} to FlashInfer {_FLASHINFER_BACKEND}{detail}."
            )

        input_2d = input_tensor.detach().view(-1, input_tensor.shape[-1])
        inference_weight = weight.detach()
        inference_bias = bias.detach() if bias is not None else None
        result = self._dispatch_cache.get(key.cache_key())
        if self.backend == LowMGemmBackend.AUTO and result is not None:
            output = self._launch_cached_flashinfer(
                input_2d,
                inference_weight,
                inference_bias,
                result,
            )
        else:
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
    """Label Linear modules and load read-only offline tuning caches."""

    _DISPATCHER.prepare(model, cuda_graph_enabled)


def apply_low_m_gemm(
    module: torch.nn.Module,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return a tuned low-M BF16 result, or ``None`` for the normal path."""

    return _DISPATCHER.apply(module, input_tensor, weight, bias)


def flush_low_m_gemm_shape_log() -> None:
    """Persist the debug shape inventory at a deterministic warmup boundary."""

    _DISPATCHER.flush_shape_log()


def write_dispatch_cache(
    path: Path, metadata: dict[str, Any], entries: dict[str, GemmTuningResult]
) -> None:
    """Write a dispatcher cache atomically for the offline tuning tool."""

    payload = {
        "schema_version": _DISPATCH_CACHE_SCHEMA_VERSION,
        "metadata": metadata,
        "entries": {key: asdict(result) for key, result in sorted(entries.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


__all__ = [
    "GemmDispatchKey",
    "GemmTuningResult",
    "LOW_M_GEMM_ACTIVE",
    "LowMGemmBackend",
    "apply_low_m_gemm",
    "flush_low_m_gemm_shape_log",
    "prepare_low_m_gemm",
    "write_dispatch_cache",
]
