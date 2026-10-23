# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify startup failures into coarse categories and log one actionable hint.

The classifier only depends on the standard library. Third-party exceptions
(torch, Hugging Face, pydantic, MPI, ...) are recognized by class name and
module, so importing this module never imports those packages.

Classification metadata (category, hint, stage, rank, summary) is stored in the
exception's ``__dict__``, so it survives pickling across the worker/proxy
process boundary.
"""

import dataclasses
import errno
import re
import subprocess
from typing import Literal, Optional, get_args

from tensorrt_llm.logger import logger

StartupErrorCategory = Literal[
    "config_input", "resource", "unsupported", "model_load", "runtime_init", "other"
]
STARTUP_ERROR_CATEGORIES: tuple[str, ...] = get_args(StartupErrorCategory)

_META_ATTR = "_trtllm_startup_error"
_MAX_CHAIN_DEPTH = 8
_MAX_SUMMARY_CHARS = 300

_DOCS = "https://nvidia.github.io/TensorRT-LLM"
_HINTS = {
    "config": (
        "Check the command-line options and the --config / --extra_llm_api_options file "
        "against the documentation."
    ),
    "import": "A required Python package is missing or broken; install it in this environment.",
    "host_name": "The host name could not be resolved; check --host.",
    "hf_gated": (
        "The model is gated on Hugging Face. Accept the license on the model page and set HF_TOKEN."
    ),
    "hf_denied": (
        "Hugging Face denied access: the repo id does not exist, or the model is gated or "
        "private. Check the repo id; for gated or private models set HF_TOKEN."
    ),
    "hf_not_found": (
        "The model was not found locally or on Hugging Face. Check the local path, "
        "or the repo id and revision."
    ),
    "hf_network": (
        "Hugging Face is not reachable. Set HF_ENDPOINT to a mirror, or download the model "
        "first and pass the local path."
    ),
    "gpu_oom": (
        "GPU memory ran out. Lower kv_cache_config.free_gpu_memory_fraction, max_batch_size "
        "or max_num_tokens, reduce the CUDA graph batch sizes, or use more GPUs "
        "(tensor_parallel_size)."
    ),
    "host_memory": (
        "The host ran out of memory or file handles. Check the container memory limit, "
        "the /dev/shm size (--shm-size) and `ulimit -n`."
    ),
    "disk": "The disk is full. Free space in the Hugging Face cache (HF_HOME) or move it.",
    "port": "The listen address is not available. Use another --port or stop the process using it.",
    "no_gpu": (
        "No GPU is usable. Check that the container has GPUs (docker --gpus, or nvidia.com/gpu "
        "in Kubernetes), that CUDA_VISIBLE_DEVICES does not hide them, and that the driver "
        "supports this CUDA version."
    ),
    "unsupported": (
        "This model, GPU, quantization or feature combination is not supported by this "
        f"TensorRT-LLM version. See {_DOCS}/models/supported-models.html and "
        f"{_DOCS}/supported-hardware.html."
    ),
    "model_load": (
        "The model files could not be loaded. Check that the checkpoint is complete, "
        "in Hugging Face format and consistent with its config; re-download it if needed."
    ),
    "distributed": (
        "Multi-GPU initialization failed. Check that every rank has a GPU of its own (two ranks "
        "on one GPU fail with NCCL 'invalid usage'), rerun with NCCL_DEBUG=INFO, and check the "
        "MPI / Slurm launch and the network setup."
    ),
    "gpu_busy": (
        "The GPU is busy or unavailable. Check `nvidia-smi` for other processes on it and for "
        "an exclusive compute mode."
    ),
    "jit": (
        "Kernel compilation failed. Check that the CUDA toolkit (nvcc) is installed and "
        "that the JIT cache directory is writable."
    ),
    "internal": (
        "This looks like an internal TensorRT-LLM error. Please report it at "
        "https://github.com/NVIDIA/TensorRT-LLM/issues with the full log."
    ),
    "worker_died": (
        "A worker process exited without reporting an error; check its log above. If it was "
        "killed, check host memory (OOM killer, dmesg), the container memory limit and /dev/shm."
    ),
}
_CATEGORY_HINTS = {
    "config_input": _HINTS["config"],
    "resource": "",
    "unsupported": _HINTS["unsupported"],
    "model_load": _HINTS["model_load"],
    "runtime_init": _HINTS["internal"],
    "other": "",
}

_UNSUPPORTED_RE = re.compile(
    r"\bnot (?:yet |currently )?supported\b|\bunsupported\b(?! operand)"
    r"|\bonly supported (?:on|for|with|by)\b|\brequires? (?:sm ?\d|compute capability)"
    r"|\bunknown (?:model )?architecture\b",
    re.IGNORECASE,
)
_OOM_RE = re.compile(
    r"out of memory|cudaErrorMemoryAllocation|CUDA_ERROR_OUT_OF_MEMORY", re.IGNORECASE
)
_WORKER_DIED_RE = re.compile(r"died during initialization|exited unexpectedly", re.IGNORECASE)
_GPU_BUSY_RE = re.compile(r"busy or unavailable|cudaErrorDevicesUnavailable", re.IGNORECASE)
_NCCL_RE = re.compile(
    r"\bnccl\b.{0,40}\b(?:error|failure|failed)\b"
    r"|\bnccl(?:InternalError|SystemError|UnhandledCudaError|InvalidUsage|RemoteError)\b",
    re.IGNORECASE,
)
_MODEL_LOAD_MODULES = (
    "transformers.",
    "safetensors",
    "tensorrt_llm._torch.models.checkpoints",
    "tensorrt_llm._torch.pyexecutor.model_loader",
)
_JIT_MODULES = (
    "flashinfer",
    "triton",
    "deep_gemm",
    "cutlass",
    "tvm_ffi",
    "torch.utils.cpp_extension",
)
_BUG_TYPES = (
    AssertionError,
    AttributeError,
    IndexError,
    KeyError,
    NameError,
    TypeError,
    ZeroDivisionError,
)
# torch.AcceleratorError.error_code values (cudaError_t).
_CUDA_OOM_CODES = {2}
_CUDA_NO_DEVICE_CODES = {35, 100, 803, 804}  # insufficient driver, no device, compat issues
_CUDA_BUSY_CODES = {46}  # cudaErrorDevicesUnavailable: exclusive mode or in use
_CUDA_NO_KERNEL_IMAGE_CODES = {209}


@dataclasses.dataclass(frozen=True)
class StartupErrorInfo:
    """Classification of one startup failure.

    Args:
        category: Coarse category, one of ``STARTUP_ERROR_CATEGORIES``.
        summary: ``"<ExceptionType>: <message>"`` of the deciding exception, on one line.
        hint: Actionable next step for the user; may be empty.
        stage: Initialization stage that failed (e.g. ``executor_creation/model``); may be empty.
        rank: MPI rank that failed, when known.
    """

    category: str
    summary: str
    hint: str = ""
    stage: str = ""
    rank: Optional[int] = None


def _meta(exc: BaseException) -> dict:
    meta = exc.__dict__.get(_META_ATTR)
    if meta is None:
        meta = {}
        setattr(exc, _META_ATTR, meta)
    return meta


def tag_startup_error(exc: BaseException, category: StartupErrorCategory, hint: str = ""):
    """Attach a category (and optionally a specific hint) at a raise site.

    Usage: ``raise tag_startup_error(ValueError("..."), "resource", hint="...")``.

    Returns:
        The same exception object.
    """
    meta = _meta(exc)
    meta["category"] = category
    meta["hint"] = hint or _CATEGORY_HINTS[category]
    return exc


def mark_startup_stage(exc: Optional[BaseException], stage: str) -> None:
    """Record the initialization stage on ``exc`` unless an inner stage already did."""
    if exc is None or not stage:
        return
    try:
        _meta(exc).setdefault("stage", stage)
    except Exception:  # noqa: BLE001 - annotating must never replace the original error
        pass


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Return the cause/context chain, deepest first, including pydantic inner errors."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen and len(chain) < _MAX_CHAIN_DEPTH:
        seen.add(id(current))
        chain.append(current)
        # Errors raised inside pydantic validators count as deeper than the ValidationError.
        inner = [e for e in _pydantic_inner_errors(current) if id(e) not in seen]
        chain.extend(reversed(inner))
        seen.update(id(e) for e in inner)
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    chain.reverse()
    return chain


def _has_class(exc: BaseException, names: tuple[str, ...], module_prefix: str = "") -> bool:
    return any(
        cls.__name__ in names and cls.__module__.startswith(module_prefix)
        for cls in type(exc).__mro__
    )


def _is_pydantic_validation_error(exc: BaseException) -> bool:
    return _has_class(exc, ("ValidationError",), "pydantic")


def _pydantic_inner_errors(exc: BaseException) -> list[BaseException]:
    if not _is_pydantic_validation_error(exc):
        return []
    try:
        details = exc.errors()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - best effort; the outer error still classifies
        return []
    return [
        d["ctx"]["error"]
        for d in details
        if isinstance(d.get("ctx"), dict) and isinstance(d["ctx"].get("error"), BaseException)
    ]


def _message(exc: BaseException) -> str:
    try:
        return str(exc)
    except Exception:  # noqa: BLE001 - a broken __str__ must not break classification
        return ""


def _summarize(exc: BaseException, hint: str = "") -> str:
    """One bounded line; a message that already spells out the hint drops it."""
    message = _message(exc)
    if hint:
        message = message.replace(hint, "")
    text = " ".join(message.split())
    if len(text) > _MAX_SUMMARY_CHARS:
        text = text[: _MAX_SUMMARY_CHARS - 3] + "..."
    name = type(exc).__name__
    return f"{name}: {text}" if text else name


def _raise_module(exc: BaseException) -> str:
    tb = exc.__traceback__
    if tb is None:
        return ""
    while tb.tb_next is not None:
        tb = tb.tb_next
    return tb.tb_frame.f_globals.get("__name__", "")


def _tag_rule(exc: BaseException) -> Optional[tuple[str, str]]:
    meta = exc.__dict__.get(_META_ATTR) or {}
    if meta.get("category"):
        return meta["category"], meta.get("hint", "")
    return None


def _typed_rule(exc: BaseException) -> Optional[tuple[str, str]]:
    # Resources.
    if _has_class(exc, ("OutOfMemoryError",), "torch"):
        return "resource", _HINTS["gpu_oom"]
    if isinstance(exc, MemoryError):
        return "resource", _HINTS["host_memory"]
    if _has_class(exc, ("gaierror",), "socket"):
        return "config_input", _HINTS["host_name"]
    if isinstance(exc, OSError) and exc.errno is not None:
        if exc.errno in (errno.EADDRINUSE, errno.EADDRNOTAVAIL):
            return "resource", _HINTS["port"]
        if exc.errno == errno.ENOSPC:
            return "resource", _HINTS["disk"]
        if exc.errno in (errno.ENOMEM, errno.EMFILE):
            return "resource", _HINTS["host_memory"]
    if _has_class(exc, ("AcceleratorError",), "torch"):
        code = getattr(exc, "error_code", None)
        if code in _CUDA_OOM_CODES:
            return "resource", _HINTS["gpu_oom"]
        if code in _CUDA_NO_DEVICE_CODES:
            return "resource", _HINTS["no_gpu"]
        if code in _CUDA_BUSY_CODES:
            return "resource", _HINTS["gpu_busy"]
        if code in _CUDA_NO_KERNEL_IMAGE_CODES:
            return "unsupported", _HINTS["unsupported"]
        return "runtime_init", _HINTS["internal"]

    # User input. Subclasses are checked before their bases.
    if _has_class(exc, ("GatedRepoError",), "huggingface_hub"):
        return "config_input", _HINTS["hf_gated"]
    if _has_class(
        exc,
        ("RepositoryNotFoundError", "RevisionNotFoundError", "HFValidationError"),
        "huggingface_hub",
    ):
        return "config_input", _HINTS["hf_not_found"]
    if _has_class(exc, ("LocalEntryNotFoundError", "OfflineModeIsEnabled"), "huggingface_hub"):
        return "config_input", _HINTS["hf_network"]
    if _has_class(exc, ("EntryNotFoundError",), "huggingface_hub"):
        return "model_load", _HINTS["model_load"]
    if _has_class(exc, ("HfHubHTTPError",), "huggingface_hub") or any(
        cls.__module__.split(".")[0] in ("requests", "httpx", "urllib3")
        for cls in type(exc).__mro__
    ):
        # The Hub answers 401 for repos that do not exist as well as for private ones.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return "config_input", _HINTS["hf_denied"]
        if status == 404:
            return "config_input", _HINTS["hf_not_found"]
        return "config_input", _HINTS["hf_network"]
    if _is_pydantic_validation_error(exc):
        # A validator that rejects a feature combination is a product gap, not a typo.
        if any(_UNSUPPORTED_RE.search(_message(e)) for e in _pydantic_inner_errors(exc)):
            return "unsupported", _HINTS["unsupported"]
        return "config_input", _HINTS["config"]
    if _has_class(exc, ("YAMLError",), "yaml") or _has_class(exc, ("ClickException",), "click"):
        return "config_input", _HINTS["config"]
    if _has_class(exc, ("ConfigOverrideError",), "tensorrt_llm"):
        return "config_input", _HINTS["config"]
    if isinstance(exc, ImportError):
        return "config_input", _HINTS["import"]

    # Product support.
    if isinstance(exc, NotImplementedError):
        return "unsupported", _HINTS["unsupported"]

    # Model files.
    if _has_class(exc, ("SafetensorError",)):
        return "model_load", _HINTS["model_load"]

    # Runtime initialization.
    if _has_class(exc, ("Exception",), "mpi4py") or _has_class(
        exc, ("DistBackendError", "DistNetworkError", "DistStoreError"), "torch"
    ):
        return "runtime_init", _HINTS["distributed"]
    if _has_class(exc, ("CompilationError", "OutOfResources"), "triton") or isinstance(
        exc, subprocess.CalledProcessError
    ):
        return "runtime_init", _HINTS["jit"]
    return None


def _heuristic_rule(exc: BaseException) -> Optional[tuple[str, str]]:
    message = _message(exc)
    if _OOM_RE.search(message):
        return "resource", _HINTS["gpu_oom"]
    if _GPU_BUSY_RE.search(message):
        return "resource", _HINTS["gpu_busy"]
    if _NCCL_RE.search(message):
        return "runtime_init", _HINTS["distributed"]
    if _WORKER_DIED_RE.search(message):
        return "runtime_init", _HINTS["worker_died"]
    if _UNSUPPORTED_RE.search(message):
        return "unsupported", _HINTS["unsupported"]
    module = _raise_module(exc)
    if module.startswith(_MODEL_LOAD_MODULES):
        return "model_load", _HINTS["model_load"]
    if module.startswith(_JIT_MODULES):
        return "runtime_init", _HINTS["jit"]
    if isinstance(exc, KeyError) and module.startswith("tensorrt_llm._torch.models"):
        return "model_load", _HINTS["model_load"]
    if isinstance(exc, _BUG_TYPES) and module.startswith("tensorrt_llm"):
        return "runtime_init", _HINTS["internal"]
    return None


def classify_startup_error(exc: BaseException, rank: Optional[int] = None) -> StartupErrorInfo:
    """Classify ``exc`` by walking its cause chain from the deepest exception outward.

    Three passes, each taking the first match: raise-site tags, typed rules
    (exception types, errno, CUDA error codes), then local-only heuristics
    (message text and raising module). Unmatched errors are ``other``.
    Never raises.

    Args:
        exc: The failure to classify.
        rank: MPI rank to record when the caller knows it, e.g. in a worker.
    """
    try:
        info = _classify(exc)
    except Exception:  # noqa: BLE001 - classification must never replace the original error
        info = StartupErrorInfo("other", type(exc).__name__)
    return info if rank is None else dataclasses.replace(info, rank=rank)


def _classify(exc: BaseException) -> StartupErrorInfo:
    chain = _exception_chain(exc)
    decided: Optional[tuple[BaseException, str, str]] = None
    for rule in (_tag_rule, _typed_rule, _heuristic_rule):
        for e in chain:
            hit = rule(e)
            if hit is not None:
                decided = (e, *hit)
                break
        if decided is not None:
            break
    if decided is None:
        decided = (chain[0], "other", "")
    decider, category, hint = decided

    metas = [e.__dict__.get(_META_ATTR) or {} for e in chain]
    stage = next((m["stage"] for m in metas if m.get("stage")), "")
    rank = next((m["rank"] for m in metas if m.get("rank") is not None), None)
    summary = (decider.__dict__.get(_META_ATTR) or {}).get("summary") or _summarize(decider, hint)
    return StartupErrorInfo(category, summary, hint, stage, rank)


def wrap_worker_startup_error(message: str, info: Optional[StartupErrorInfo]) -> RuntimeError:
    """Build the error reporting a worker's initialization failure, with its classification.

    The error holds only strings and ints, so it pickles across processes, and its
    message alone names the rank, stage and root cause for channels that forward text.

    Args:
        message: Fixed prefix, e.g. ``"Executor worker returned error"``.
        info: Classification computed in the worker process, or None if unavailable.
    """
    if info is None:
        return RuntimeError(message)
    where = "" if info.rank is None else f" on rank {info.rank}"
    if info.stage:
        where += f" during {info.stage}"
    error = RuntimeError(f"{message}{where}: {info.summary}")
    _meta(error).update(dataclasses.asdict(info))
    return error


def is_worker_startup_error(exc: BaseException) -> bool:
    """Whether ``exc`` came from ``wrap_worker_startup_error`` with a worker classification."""
    return "summary" in (exc.__dict__.get(_META_ATTR) or {})


def format_startup_error(info: StartupErrorInfo, what: str) -> str:
    """Render the single ERROR line for a classified startup failure."""
    where = "" if info.rank is None else f" on rank {info.rank}"
    if info.stage:
        where += f" during {info.stage}"
    line = f"{what} [{info.category}]{where}: {info.summary}"
    return f"{line} Hint: {info.hint}" if info.hint else line


def log_startup_error(exc: BaseException, what: str) -> Optional[StartupErrorInfo]:
    """Classify ``exc`` and log one ERROR line, unless it was already logged.

    Never raises, so callers can use it right before re-raising ``exc``.

    Args:
        exc: The exception about to propagate.
        what: Leading words of the line, e.g. ``"LLM initialization failed"``.

    Returns:
        The classification, or None if nothing was logged.
    """
    try:
        if any((e.__dict__.get(_META_ATTR) or {}).get("logged") for e in _exception_chain(exc)):
            return None
        info = classify_startup_error(exc)
        _meta(exc)["logged"] = True
        logger.error(format_startup_error(info, what))
        return info
    except Exception:  # noqa: BLE001 - diagnostics must never mask the original error
        return None
