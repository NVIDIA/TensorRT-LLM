# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compile cache helpers for Prims-TS batched GEMM launches.

Compiled GEMMs are kept in an in-process dictionary and persisted under the
FlashInfer JIT directory (module ``prims_ts_moe_gemm``), so another process
loads a GEMM it would otherwise recompile. The on-disk module is invalidated
as a whole when the batched GEMM sources, the CuTe-DSL stack or the compile
architecture change; within it, a kernel is named after everything else its
code depends on (see :func:`_disk_kernel_name`).
"""

from __future__ import annotations

import enum
import functools
import hashlib
import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from flashinfer.jit.cute_dsl_core import _get_compile_arch
from ..utils import get_prims_ts_compile_options

# TensorRT-LLM: distinct from the installed flashinfer's own Prims-TS MoE
# module, whose different sources would otherwise invalidate this one.
_DISK_MODULE_NAME = "trtllm_prims_ts_moe_gemm"


def stable_config_hash(cfg: Any, **constexpr_args: Any) -> str:
    """Return a stable cache key for compile-time TS kernel arguments."""
    if hasattr(cfg, "compile_signature"):
        payload = cfg.compile_signature()
    elif is_dataclass(cfg):
        payload = asdict(cfg)
    elif hasattr(cfg, "__dict__"):
        payload = dict(cfg.__dict__)
    else:
        payload = repr(cfg)
    return json.dumps(
        {
            "cfg": payload,
            "constexpr_args": constexpr_args,
        },
        sort_keys=True,
        default=str,
    )


@functools.cache
def get_compile_options() -> str:
    return get_prims_ts_compile_options()


_COMPILED_GEMM_CACHE: dict[tuple[str, str, str, int, str], Any] = {}


def clear_compiled_gemm_cache() -> None:
    """Release process-local compiled kernels after all launches synchronize."""
    _COMPILED_GEMM_CACHE.clear()


def _compile_target_key(io: dict) -> tuple[int, str]:
    """Return the CUDA device and CuTe-DSL architecture for this compilation."""
    import torch

    device_index = None
    for value in io.get("_keepalive", ()):
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            device_index = value.device.index
            break
    if device_index is None:
        device_index = torch.cuda.current_device()

    # The architecture is resolved from the current device.
    with torch.cuda.device(device_index):
        return int(device_index), _get_compile_arch()


@functools.cache
def _kernel_source_files() -> tuple[str, ...]:
    """Sources whose content invalidates every persisted GEMM.

    The kernel and its ``cute.compile`` entry point live in ``batched_gemm``,
    whose generated code depends outside the package only on the CuTe-DSL
    bootstrap in ``cutlass_dsl.py`` and the FP4 conversion helpers in
    ``flashinfer/cute_dsl/fp4_common.py``. The MoE adapters only choose the
    config and build the launch arguments, both of which are part of each
    kernel's name instead.
    """
    from flashinfer.cute_dsl import fp4_common

    root = Path(__file__).resolve().parents[1]
    files = [
        *(root / "batched_gemm").rglob("*.py"),
        root / "cutlass_dsl.py",
        # TensorRT-LLM: the vendored tree is not inside the flashinfer package,
        # so the FP4 helpers come from the installed flashinfer.
        Path(fp4_common.__file__).resolve(),
    ]
    return tuple(sorted(str(path) for path in files))


def _type_key(value: Any) -> Any:
    """JSON-encodable, process-independent form of one compile-time property."""
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, enum.Enum):
        return f"{type(value).__qualname__}.{value.name}"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_type_key(item) for item in value]
    return repr(value)


def _arg_type_key(value: Any) -> Any:
    """What ``cute.compile`` specializes on for one launch argument.

    A CuTe pointer contributes its ``__cache_key__`` (element type, address
    space and assumed alignment); Python scalars only their type, since they
    become dynamic kernel arguments; tuples their structure.
    """
    cache_key = getattr(value, "__cache_key__", None)
    if cache_key is not None:
        return [type(value).__qualname__, _type_key(cache_key)]
    if isinstance(value, (tuple, list)):
        return [_arg_type_key(item) for item in value]
    return type(value).__qualname__


def _disk_kernel_name(cfg_hash: str, fc1_or_fc2: str, io: dict) -> str:
    """Name of a GEMM in the persistent module: everything its code depends on
    besides the module-level sources, CuTe-DSL stack and compile architecture.

    Includes the capability of the current device, which must be the device
    the GEMM is compiled for.
    """
    import torch

    from ..batched_gemm.batched_gemm_run import (
        _compile_arg_tuple,
        _cute_compile_options,
    )

    # The stream is the last compile argument and the constexpr config the
    # one before it; the config is keyed through ``cfg_hash``.
    launch_args = _compile_arg_tuple(io, None)[:-2]
    identity = json.dumps(
        {
            "cfg": cfg_hash,
            "args": [_arg_type_key(arg) for arg in launch_args],
            "options": _cute_compile_options(),
            # Read verbatim: the kernel and the compile entry point parse it
            # differently (line info, task-manager verification).
            "debug_checks": os.environ.get("FLASHINFER_PRIMS_TS_DEBUG_CHECKS"),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()
    # A family target such as sm100f covers GPUs with different capabilities.
    major, minor = torch.cuda.get_device_capability()
    return f"{fc1_or_fc2}_sm{major}{minor}_{digest}"


@functools.cache
def _kernel_source_sha256() -> str | None:
    """Hash of :func:`_kernel_source_files`; None when a file cannot be read,
    which disables persistence rather than weakening the key."""
    from flashinfer.jit._cute_dsl_source import _hash_source_files
    from flashinfer.jit.core import logger

    try:
        return _hash_source_files(_kernel_source_files())
    except (OSError, SyntaxError, ValueError) as exc:
        logger.warning(
            f"Cannot hash the Prims-TS GEMM sources ({exc}); compiled GEMMs "
            "are not persisted."
        )
        return None


def _compile_or_load(
    cfg_hash: str, fc1_or_fc2: str, io: dict, stream: Any, device_index: int
) -> Any:
    """Load the GEMM from the persistent module, compiling it on a miss."""
    import torch

    from flashinfer.jit.cute_dsl_core import JitSpecCuteDsl, cute_dsl_cache_disabled
    from ..batched_gemm.batched_gemm_run import _compile_for_launch

    def compile_fn():
        return _compile_for_launch(io, stream)

    source_sha256 = _kernel_source_sha256()
    # TensorRT-LLM: a flashinfer whose JitSpecCuteDsl has no ``enable_tvm_ffi``
    # persists only TVM-FFI kernels, and these GEMMs are compiled without it.
    persists_plain_kernels = (
        "enable_tvm_ffi" in inspect.signature(JitSpecCuteDsl.__init__).parameters
    )
    # ``cute.compile``, the spec's module directory and metadata, and the
    # kernel name all resolve the target architecture from the current device.
    with torch.cuda.device(device_index):
        if (
            cute_dsl_cache_disabled()
            or source_sha256 is None
            or not persists_plain_kernels
        ):
            return compile_fn()
        spec = JitSpecCuteDsl(
            _DISK_MODULE_NAME,
            _disk_kernel_name(cfg_hash, fc1_or_fc2, io),
            compile_fn,
            source_sha256,
            enable_tvm_ffi=False,
        )
        kernel = spec.try_load()
        if kernel is None:
            # The module lock is held only to commit the artifact, so the
            # threads of a parallel tactic precompile (and other ranks) keep
            # compiling concurrently.
            spec.compile_and_persist()
            kernel = spec.load()
    return kernel


def get_compiled_gemm(cfg_hash: str, fc1_or_fc2: str, io: dict, stream: Any) -> Any:
    """Compile and cache a TS GEMM for the supplied launch IO.

    ``cfg_hash`` includes only the normalized config and other constexpr
    operands. Runtime problem shape and IO objects are passed to
    ``cute.compile`` to establish argument types, but the generated kernel body
    does not depend on their values and they must not be part of the cache key.
    """

    device_index, compile_arch = _compile_target_key(io)
    key = (
        cfg_hash,
        fc1_or_fc2,
        get_compile_options(),
        device_index,
        compile_arch,
    )
    if key in _COMPILED_GEMM_CACHE:
        return _COMPILED_GEMM_CACHE[key]
    compiled = _compile_or_load(cfg_hash, fc1_or_fc2, io, stream, device_index)
    _COMPILED_GEMM_CACHE[key] = compiled
    return compiled
