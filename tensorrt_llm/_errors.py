# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error categories, the exception classes that declare them, and standard hints.

A category says what went wrong, and so who acts on it. A raise site that knows
the category declares it, preferably by raising the category's class:

- ``ConfigInputError`` (a ``ValueError``): invalid user input;
- ``UnsupportedError`` (a ``ValueError``): a model, GPU, quantization or feature
  combination that this version does not support;
- ``ResourceError`` (a ``RuntimeError``): missing or exhausted GPUs, memory, disk or ports;
- ``ModelLoadError`` (a ``RuntimeError``): model files that cannot be read or do not
  match the model;
- ``RuntimeInitError`` (a ``RuntimeError``): bringing up the runtime failed: worker
  processes, distributed setup or kernel compilation.

``unsupported`` is a model, GPU, quantization or feature request that the product
could reasonably support later; invalid or misspelled values are ``config_input``.

Each class derives only from its builtin, so callers can always catch it with
``except ValueError`` or ``except RuntimeError``. The category travels with the
type; the message carries no marker. Any other exception class declares a
category with a ``category`` class attribute. ``"other"`` is never declared: it
is the classifier's fallback for errors that cannot be attributed to the other
categories.

The reason is a key of ``ERROR_HINTS`` naming the specific cause, e.g.
``"gpu_oom"``; its hint is the next step for the user. Raise sites pick a
standard hint by passing its reason.
"""

from typing import ClassVar, Literal, get_args

# Usage telemetry reports the category of a startup failure as the exit report's
# ``errorCategory`` (see tensorrt_llm/usage/schemas/README.md). The names are a
# data contract: renaming one breaks its trend, and adding one also needs a
# telemetry schema update.
ErrorCategory = Literal[
    "config_input", "resource", "unsupported", "model_load", "runtime_init", "other"
]
ERROR_CATEGORIES: tuple[str, ...] = get_args(ErrorCategory)

_DOCS_SITE = "https://nvidia.github.io/TensorRT-LLM"
# Standard hints keyed by reason.
ERROR_HINTS = {
    "config": "Check the LLM arguments and the config file against the documentation.",
    "import": "A required Python package is missing or broken; install it in this environment.",
    "host_name": "The host name could not be resolved; check --host.",
    "hf": (
        "The model could not be fetched from Hugging Face. Check the local path or the repo id "
        "and revision; for a gated or private model, accept its license and set HF_TOKEN; "
        "without access to the Hub, set HF_ENDPOINT to a mirror or pass a local path."
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
        f"TensorRT-LLM version. See {_DOCS_SITE}/models/supported-models.html and "
        f"{_DOCS_SITE}/supported-hardware.html."
    ),
    "model_load": (
        "The model files could not be loaded. Check that the checkpoint is complete, "
        "in Hugging Face format and consistent with its config; re-download it if needed."
    ),
    "runtime_init": (
        "Bringing up the runtime failed (worker processes, multi-GPU communication or kernel "
        "compilation). Check the MPI / Slurm launch and the NCCL setup, the CUDA toolkit "
        "(nvcc) for kernel compilation, and the host memory and /dev/shm limits."
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
# A category's reason when nothing more specific is known; "" has no hint.
DEFAULT_REASONS: dict[str, str] = {
    "config_input": "config",
    "resource": "",
    "unsupported": "unsupported",
    "model_load": "model_load",
    "runtime_init": "runtime_init",
    "other": "",
}


class _CategorizedError:
    """Code shared by the category classes; not an exception type and not for ``except``.

    ``category`` is a class attribute; ``reason`` and ``hint`` are instance
    attributes, so they survive pickling across processes. ``str()`` is the
    plain message.

    Args:
        message: What failed.
        reason: Key of ``ERROR_HINTS`` naming the specific cause; defaults to the category's.
        hint: Next step for the user; defaults to the reason's standard hint.
    """

    category: ClassVar[ErrorCategory]

    def __init__(self, message: str = "", *, reason: str = "", hint: str = ""):
        super().__init__(message)
        self.reason = reason or DEFAULT_REASONS[self.category]
        self.hint = hint or ERROR_HINTS.get(self.reason, "")


class ConfigInputError(_CategorizedError, ValueError):
    """Invalid user input: arguments, configuration files or launch parameters."""

    category = "config_input"


class UnsupportedError(_CategorizedError, ValueError):
    """A model, GPU, quantization or feature combination that this version does not support."""

    category = "unsupported"


class ResourceError(_CategorizedError, RuntimeError):
    """Missing or exhausted resources: GPUs, memory, disk or network ports."""

    category = "resource"


class ModelLoadError(_CategorizedError, RuntimeError):
    """Model files that cannot be read or do not match the model."""

    category = "model_load"


class RuntimeInitError(_CategorizedError, RuntimeError):
    """Bringing up the runtime failed: worker processes, distributed setup or kernels."""

    category = "runtime_init"
