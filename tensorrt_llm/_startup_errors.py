# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify startup failures into coarse categories and log one actionable hint.

The categories, reasons, hints and category exception classes do not depend on
a phase; they are defined in ``tensorrt_llm._errors``. This module is the
startup boundary: it classifies a failure and logs the hint line only where
initialization fails, i.e. in the ``LLM`` constructor, the executor worker and
proxy, and the CLI boundary until an ``LLM`` has initialized in this process.

Declared categories decide first: a ``category`` attribute (the classes in
``tensorrt_llm._errors``, or any exception class that sets one), the worker's
classification carried by the proxy, and the ``[<category>]`` marker in the text
that ``wrap_worker_startup_error`` forwards for a worker.
Failures that are not declared (builtin, third-party, C++, CUDA and NCCL errors,
and TensorRT-LLM errors raised as plain builtins) are classified by exception
type, errno and CUDA error code, then by the fixed message phrases and raising
modules below. Unmatched errors are ``other``, as are errors that look internal
to TensorRT-LLM, which get the ``internal`` reason and hint.

The classifier only depends on the standard library. Third-party exceptions
(torch, Hugging Face, pydantic, MPI, ...) are recognized by class name and
module, so importing this module never imports those packages.

Classification metadata (category, reason, hint, stage, rank, summary) is stored
in the exception's ``__dict__``, so it survives pickling across the worker/proxy
process boundary.
"""

import dataclasses
import errno
import re
from typing import Optional

from tensorrt_llm._errors import DEFAULT_REASONS, ERROR_CATEGORIES, ERROR_HINTS
from tensorrt_llm.logger import logger

# Categories a raise site can declare; "other" is only the classifier's fallback.
_DECLARABLE_CATEGORIES = tuple(c for c in ERROR_CATEGORIES if c != "other")
_MARKER_RE = re.compile(r"\[(" + "|".join(_DECLARABLE_CATEGORIES) + r")\] ")

_META_ATTR = "_trtllm_startup_error"
_MAX_CHAIN_DEPTH = 8
_MAX_SUMMARY_CHARS = 300

_UNSUPPORTED_RE = re.compile(
    r"\bnot (?:yet |currently )?supported\b|\bunsupported\b(?! operand)"
    r"|\b(?:does|do) not support\b|\bonly supported (?:on|for|with|by)\b|\bonly supports\b"
    r"|\bsupported on sm ?\d"
    r"|\brequires? (?:an? |the )?(?:sm ?\d|compute capability|hopper|blackwell)"
    r"|\bunknown (?:model )?architecture\b"
    r"|\bnot included in this TensorRT-LLM build\b",
    re.IGNORECASE,
)
_OOM_RE = re.compile(
    r"out of memory|cudaErrorMemoryAllocation|CUDA_ERROR_OUT_OF_MEMORY", re.IGNORECASE
)
_WORKER_DIED_RE = re.compile(r"died during initialization|exited unexpectedly", re.IGNORECASE)
# C++ kernel compilation: DeepGEMM's NVRTC JIT and the trtllm-gen FMHA nvcc probe.
_CPP_JIT_RE = re.compile(r"NVRTC compilation failed|Cannot parse NVCC version", re.IGNORECASE)
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
# CLI code opens the files that the user named on the command line, e.g. --config.
_CLI_MODULES = (
    "tensorrt_llm.commands",
    "tensorrt_llm.bench.benchmark",
    "tensorrt_llm.llmapi.disagg_utils",
)
_JIT_MODULES = (
    "flashinfer",
    "triton",
    "deep_gemm",
    "cutlass",
    "tvm_ffi",
    "torch.utils.cpp_extension",
)
# Builtin types that, raised inside tensorrt_llm without a declared category,
# usually mean an internal error: "other" with the "internal" reason.
_INTERNAL_ERROR_TYPES = (
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
# Message phrases recognized when no raise site declared the category: text from
# C++, CUDA, NCCL and third-party code, and TensorRT-LLM errors raised as plain
# builtins. New TensorRT-LLM errors declare their category instead.
_MESSAGE_RULES = (
    (_OOM_RE, "resource", "gpu_oom"),
    (_GPU_BUSY_RE, "resource", "gpu_busy"),
    (_NCCL_RE, "runtime_init", "distributed"),
    (_WORKER_DIED_RE, "runtime_init", "worker_died"),
    (_CPP_JIT_RE, "runtime_init", "jit"),
    (_UNSUPPORTED_RE, "unsupported", "unsupported"),
)

# Set once an LLM in this process has finished initializing.
_llm_initialized = False


def mark_llm_initialized() -> None:
    """Record that an ``LLM`` in this process finished initializing.

    From then on the CLI boundary does not classify uncaught exceptions: they are
    failures of a running process, not startup failures. A later ``LLM`` that
    fails to initialize is still classified by its own boundary.
    """
    global _llm_initialized
    _llm_initialized = True


def llm_initialized() -> bool:
    """Whether an ``LLM`` in this process has finished initializing."""
    return _llm_initialized


@dataclasses.dataclass(frozen=True)
class StartupErrorInfo:
    """Classification of one startup failure.

    Args:
        category: Coarse category, one of ``ERROR_CATEGORIES``.
        summary: ``"<ExceptionType>: <message>"`` of the deciding exception, on one line.
        hint: Actionable next step for the user; may be empty.
        stage: Initialization stage that failed (e.g. ``executor_creation/model``); may be empty.
        rank: MPI rank that failed, when known.
        reason: Key of ``ERROR_HINTS`` naming the specific cause; may be empty.
    """

    category: str
    summary: str
    hint: str = ""
    stage: str = ""
    rank: Optional[int] = None
    reason: str = ""


def _meta(exc: BaseException) -> dict:
    meta = exc.__dict__.get(_META_ATTR)
    if meta is None:
        meta = {}
        setattr(exc, _META_ATTR, meta)
    return meta


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
    """One bounded line; drops the category marker and text that repeats the hint."""
    message = _MARKER_RE.sub("", _message(exc), count=1)
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


# (category, reason, hint) decided by a rule.
_Hit = tuple[str, str, str]


def _hit(category: str, reason: str) -> _Hit:
    return category, reason, ERROR_HINTS.get(reason, "")


def _declared_rule(exc: BaseException) -> Optional[_Hit]:
    # Any exception class declares its category with a ``category`` attribute.
    category = getattr(exc, "category", None)
    if isinstance(category, str) and category in _DECLARABLE_CATEGORIES:
        reason = getattr(exc, "reason", None)
        if not isinstance(reason, str) or not reason:
            reason = DEFAULT_REASONS[category]
        hint = getattr(exc, "hint", None)
        if not isinstance(hint, str) or not hint:
            hint = ERROR_HINTS.get(reason, "")
        return category, reason, hint
    # The worker's classification, carried by the proxy's wrapper.
    meta = exc.__dict__.get(_META_ATTR) or {}
    if meta.get("category"):
        return meta["category"], meta.get("reason", ""), meta.get("hint", "")
    # The worker's classification, forwarded as text only (wrap_worker_startup_error).
    match = _MARKER_RE.search(_message(exc))
    if match is None:
        return None
    category = match.group(1)
    # The marker decides the category; a recognized phrase can still add a specific hint.
    phrase = _phrase_rule(exc)
    if phrase is not None and phrase[0] == category:
        return phrase
    return _hit(category, DEFAULT_REASONS[category])


def _typed_rule(exc: BaseException) -> Optional[_Hit]:
    # Resources.
    if _has_class(exc, ("OutOfMemoryError",), "torch"):
        return _hit("resource", "gpu_oom")
    if isinstance(exc, MemoryError):
        return _hit("resource", "host_memory")
    if _has_class(exc, ("gaierror",), "socket"):
        return _hit("config_input", "host_name")
    if isinstance(exc, OSError) and exc.errno is not None:
        if exc.errno in (errno.EADDRINUSE, errno.EADDRNOTAVAIL):
            return _hit("resource", "port")
        if exc.errno == errno.ENOSPC:
            return _hit("resource", "disk")
        if exc.errno in (errno.ENOMEM, errno.EMFILE):
            return _hit("resource", "host_memory")
    if _has_class(exc, ("AcceleratorError",), "torch"):
        code = getattr(exc, "error_code", None)
        if code in _CUDA_OOM_CODES:
            return _hit("resource", "gpu_oom")
        if code in _CUDA_NO_DEVICE_CODES:
            return _hit("resource", "no_gpu")
        if code in _CUDA_BUSY_CODES:
            return _hit("resource", "gpu_busy")
        if code in _CUDA_NO_KERNEL_IMAGE_CODES:
            return _hit("unsupported", "unsupported")
        return _hit("other", "internal")

    # User input. A file missing from a repo that was found is a model-file problem;
    # LocalEntryNotFoundError (offline, not cached) subclasses EntryNotFoundError.
    if _has_class(exc, ("EntryNotFoundError",), "huggingface_hub") and not _has_class(
        exc, ("LocalEntryNotFoundError",), "huggingface_hub"
    ):
        return _hit("model_load", "model_load")
    if _has_class(
        exc,
        (
            "HfHubHTTPError",
            "GatedRepoError",
            "RepositoryNotFoundError",
            "RevisionNotFoundError",
            "HFValidationError",
            "LocalEntryNotFoundError",
            "OfflineModeIsEnabled",
        ),
        "huggingface_hub",
    ) or any(
        cls.__module__.split(".")[0] in ("requests", "httpx", "urllib3")
        for cls in type(exc).__mro__
    ):
        return _hit("config_input", "hf")
    if _is_pydantic_validation_error(exc):
        # A validator that rejects a feature combination is a product gap, not a typo.
        if any(_UNSUPPORTED_RE.search(_message(e)) for e in _pydantic_inner_errors(exc)):
            return _hit("unsupported", "unsupported")
        return _hit("config_input", "config")
    if _has_class(exc, ("YAMLError",), "yaml") or _has_class(exc, ("ClickException",), "click"):
        return _hit("config_input", "config")
    if _has_class(exc, ("ConfigOverrideError",), "tensorrt_llm"):
        return _hit("config_input", "config")
    if isinstance(exc, ImportError):
        return _hit("config_input", "import")

    # Product support.
    if isinstance(exc, NotImplementedError):
        return _hit("unsupported", "unsupported")

    # Model files.
    if _has_class(exc, ("SafetensorError",)):
        return _hit("model_load", "model_load")

    # Runtime initialization.
    if _has_class(exc, ("Exception",), "mpi4py") or _has_class(
        exc, ("DistBackendError", "DistNetworkError", "DistStoreError"), "torch"
    ):
        return _hit("runtime_init", "distributed")
    if _has_class(exc, ("CompilationError", "OutOfResources"), "triton") or _has_class(
        exc, ("CalledProcessError",), "subprocess"
    ):
        return _hit("runtime_init", "jit")
    return None


def _phrase_rule(exc: BaseException) -> Optional[_Hit]:
    message = _message(exc)
    for pattern, category, reason in _MESSAGE_RULES:
        if pattern.search(message):
            return _hit(category, reason)
    return None


def _heuristic_rule(exc: BaseException) -> Optional[_Hit]:
    phrase = _phrase_rule(exc)
    if phrase is not None:
        return phrase
    module = _raise_module(exc)
    if module.startswith(_MODEL_LOAD_MODULES):
        return _hit("model_load", "model_load")
    if isinstance(exc, FileNotFoundError) and module.startswith(_CLI_MODULES):
        return _hit("config_input", "config")
    if module.startswith(_JIT_MODULES):
        return _hit("runtime_init", "jit")
    if isinstance(exc, KeyError) and module.startswith("tensorrt_llm._torch.models"):
        return _hit("model_load", "model_load")
    if isinstance(exc, _INTERNAL_ERROR_TYPES) and module.startswith("tensorrt_llm"):
        return _hit("other", "internal")
    return None


def classify_startup_error(exc: BaseException, rank: Optional[int] = None) -> StartupErrorInfo:
    """Classify ``exc`` by walking its cause chain from the deepest exception outward.

    Three passes, each taking the first match: declared categories (a
    ``category`` attribute, the worker's classification carried by the proxy or
    by the ``[<category>]`` marker in its forwarded text), typed rules (exception
    types, errno, CUDA error codes), then local-only heuristics (message phrases
    and raising module).
    Internal-looking errors are ``other`` with the ``internal`` reason; unmatched
    errors are ``other`` with no reason. Never raises.

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
    decided: Optional[tuple[BaseException, str, str, str]] = None
    for rule in (_declared_rule, _typed_rule, _heuristic_rule):
        for e in chain:
            hit = rule(e)
            if hit is not None:
                decided = (e, *hit)
                break
        if decided is not None:
            break
    if decided is None:
        decided = (chain[0], "other", "", "")
    decider, category, reason, hint = decided

    metas = [e.__dict__.get(_META_ATTR) or {} for e in chain]
    stage = next((m["stage"] for m in metas if m.get("stage")), "")
    rank = next((m["rank"] for m in metas if m.get("rank") is not None), None)
    summary = (decider.__dict__.get(_META_ATTR) or {}).get("summary") or _summarize(decider, hint)
    return StartupErrorInfo(category, summary, hint, stage, rank, reason)


def wrap_worker_startup_error(message: str, info: Optional[StartupErrorInfo]) -> RuntimeError:
    """Build the error reporting a worker's initialization failure, with its classification.

    The error holds only strings and ints, so it pickles across processes. Its
    message alone names the rank, stage, ``[<category>]`` marker and root cause,
    so the worker's classification also survives channels that forward only
    text, such as a non-leader rank under ``trtllm-llmapi-launch``.

    Args:
        message: Fixed prefix, e.g. ``"Executor worker returned error"``.
        info: Classification computed in the worker process, or None if unavailable.
    """
    if info is None:
        return RuntimeError(message)
    where = "" if info.rank is None else f" on rank {info.rank}"
    if info.stage:
        where += f" during {info.stage}"
    summary = info.summary
    if info.category in _DECLARABLE_CATEGORIES:
        summary = f"[{info.category}] {summary}"
    error = RuntimeError(f"{message}{where}: {summary}")
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
