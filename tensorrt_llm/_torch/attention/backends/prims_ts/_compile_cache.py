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

"""Persistent compilation shared by the PrimTS attention adapters."""

from collections.abc import Callable
import dataclasses
import enum
import functools
import hashlib
from pathlib import Path
from typing import Any

from ._worktile_scope import with_scalar_work_tile_info


def _stable_value(value: object) -> object:
    if isinstance(value, enum.Enum):
        return (type(value).__module__, type(value).__qualname__, value.name)
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, type):
        return (value.__module__, value.__qualname__)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (field.name, _stable_value(getattr(value, field.name)))
                for field in dataclasses.fields(value)
                if field.compare
            ),
        )
    if isinstance(value, (tuple, list)):
        return tuple(_stable_value(item) for item in value)
    raise TypeError(f"no stable cache-key encoding for {type(value).__name__}")


def _compile_cache_key(spec: object, options: str, extra: tuple = ()) -> str | None:
    """Hash compile identity, excluding only device placement and non-key fields.

    MLA's non-comparing kernel object is represented by its kernel_signature.
    Unsupported values bypass persistence rather than risking a stale artifact.
    """
    if dataclasses.is_dataclass(spec) and not isinstance(spec, type):
        spec = tuple(
            (field.name, getattr(spec, field.name))
            for field in dataclasses.fields(spec)
            if field.compare and field.name != "device_index"
        )
    try:
        encoded = repr(_stable_value((spec, options, extra)))
    except TypeError:
        return None
    return hashlib.sha256(encoded.encode()).hexdigest()


@functools.cache
def _kernel_source_files() -> tuple[str, ...]:
    # Family adapters share masks, reductions, sparse/Sage preparation and
    # compatibility helpers. Invalidate them together when any source changes.
    root = Path(__file__).resolve().parent
    return tuple(sorted(str(path) for path in root.rglob("*.py")))


@functools.cache
def _kernel_source_sha256(paths: tuple[str, ...], strict: bool) -> str:
    # Loaded PrimTS sources are fixed for the lifetime of this process. Strict
    # mode includes imported helpers, so it needs a separate digest.
    from flashinfer.jit.cute_dsl_core import _hash_source_files

    return _hash_source_files(paths)


def _attention_kernel_identity(
    kind: str, spec: object, options: str, extra: tuple
) -> tuple[str, str] | None:
    from flashinfer.jit.core import logger
    from flashinfer.jit.cute_dsl_core import cute_dsl_cache_disabled

    from tensorrt_llm._torch.attention.backends._cute_dsl_compat import cute_dsl_strict_cache_key

    if cute_dsl_cache_disabled():
        return None
    key = _compile_cache_key(spec, options, extra)
    if key is None:
        return None
    import torch

    # A family target such as sm100f can cover physical GPUs with different
    # host-selected tuning defaults. The module metadata keys the target;
    # the specialization must also distinguish the physical capability.
    major, minor = torch.cuda.get_device_capability()
    try:
        source_hash = _kernel_source_sha256(
            _kernel_source_files(), cute_dsl_strict_cache_key()
        )
    except (OSError, TypeError, SyntaxError, ValueError) as exc:
        logger.warning(
            f"Cannot hash PrimTS attention sources ({exc}); bypassing the disk cache."
        )
        return None
    return f"{kind}_sm{major}{minor}_{key}", source_hash


def is_attention_kernel_cached(
    family: str,
    kind: str,
    spec: object,
    options: str,
    compile_fn: Callable[[], Any],
    *,
    extra: tuple = (),
) -> bool:
    from tensorrt_llm._torch.attention.backends._cute_dsl_compat import JitSpecCuteDsl

    identity = _attention_kernel_identity(kind, spec, options, extra)
    if identity is None:
        return False
    name, source_hash = identity
    return JitSpecCuteDsl(
        f"prims_ts_{family}", name, compile_fn, source_hash
    ).is_compiled


@with_scalar_work_tile_info
def build_and_load_attention_kernel(
    family: str,
    kind: str,
    spec: object,
    options: str,
    compile_fn: Callable[[], Any],
    *,
    extra: tuple = (),
) -> Any:
    """Build or load under the caller's selected CUDA device context."""
    from tensorrt_llm._torch.attention.backends._cute_dsl_compat import (
        _compile_in_process,
        build_and_load_cute_dsl_kernel,
    )

    identity = _attention_kernel_identity(kind, spec, options, extra)
    if identity is None:
        return _compile_in_process(compile_fn)
    name, source_hash = identity
    return build_and_load_cute_dsl_kernel(
        f"prims_ts_{family}",
        name,
        compile_fn,
        source_sha256=source_hash,
    )
