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
import contextlib
import contextvars
import dataclasses
import enum
import functools
import hashlib
import os
from pathlib import Path
from typing import Any


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


# Warm-up precompile modes. Under ``precompile_only()`` the attention kernels
# are compiled through ``JitSpecCuteDsl.compile_and_persist()`` (the
# ``cute.compile`` runs outside the module-level file lock that
# ``build_and_load()`` holds for a whole build, so sibling processes compile
# different kernels concurrently), committed to disk, and not loaded: the
# serving process loads them afterwards. Under ``probe_cached()`` nothing is
# compiled; a kernel missing from the cache raises ``KernelNotCached``. Both
# modes return a placeholder in place of the kernel.
_PRECOMPILE_MODE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "prims_ts_precompile_mode", default=None
)
_SKIP_BUSY: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "prims_ts_precompile_skip_busy", default=False
)


class KernelNotCached(Exception):
    """Raised under ``probe_cached()`` for a kernel that is not on disk."""


class KernelBusy(Exception):
    """Raised under ``precompile_only(skip_busy=True)`` while another process compiles the kernel."""


@contextlib.contextmanager
def precompile_only(*, skip_busy: bool = False):
    """Compile and persist kernels without loading them (see module note).

    With ``skip_busy`` a kernel whose per-kernel lock another process holds
    raises ``KernelBusy`` instead of waiting, so a worker can defer it and
    compile something else first.
    """
    token = _PRECOMPILE_MODE.set("persist")
    busy_token = _SKIP_BUSY.set(skip_busy)
    try:
        yield
    finally:
        _SKIP_BUSY.reset(busy_token)
        _PRECOMPILE_MODE.reset(token)


@contextlib.contextmanager
def probe_cached():
    """Report, without compiling, whether every kernel a plan needs is on disk."""
    token = _PRECOMPILE_MODE.set("probe")
    try:
        yield
    finally:
        _PRECOMPILE_MODE.reset(token)


def expected_cache_meta() -> dict:
    """The on-disk module metadata this process would write (arch, DSL version, source hash)."""
    return _make_spec(
        "prims_ts", "metadata", lambda: None, _kernel_source_files()
    ).expected_meta


def _unloaded_kernel(*args, **kwargs):
    raise RuntimeError(
        "PrimTS kernel was compiled in precompile-only mode and never loaded; "
        "load it through a normal plan() outside precompile_only()."
    )


def is_placeholder(compiled) -> bool:
    """True for the stand-in returned under the precompile modes."""
    return compiled is _unloaded_kernel


def cache_compiled(compile_fn):
    """Memoize loaded kernels, bypassing the memo for probe/persist plans.

    A plan may return a kernel tuple or a callable. Bypassing the whole memo
    keeps placeholders out of either shape and probes disk even after a normal
    plan has loaded a kernel into this process.
    """
    cached = functools.cache(compile_fn)

    @functools.wraps(compile_fn)
    def wrapper(*args, **kwargs):
        if _PRECOMPILE_MODE.get() is not None:
            return compile_fn(*args, **kwargs)
        return cached(*args, **kwargs)

    wrapper.cache_clear = cached.cache_clear
    wrapper.cache_info = cached.cache_info
    return wrapper


def _make_spec(module_name: str, kernel_name: str, compile_fn, key_files):
    from flashinfer.jit.cute_dsl_core import JitSpecCuteDsl, _hash_source_files

    return JitSpecCuteDsl(
        module_name, kernel_name, compile_fn, _hash_source_files(tuple(key_files))
    )


def _precompile_kernel(
    module_name: str, kernel_name: str, compile_fn, key_files
) -> None:
    """Compile one kernel into the on-disk cache unless it is already there.

    One compile per kernel per workspace: a sibling that wants the same kernel
    waits on the per-kernel lock (or raises ``KernelBusy`` under ``skip_busy``)
    and then finds it on disk. The lock lives in FlashInfer's tmp directory,
    not in the module directory: creating that directory before the first
    export would let ``compile_and_persist()`` of a sibling read a missing
    meta.json as a stale module and wipe the exports already there.
    """
    from filelock import FileLock, Timeout
    from flashinfer.jit.core import get_tmpdir

    spec = _make_spec(module_name, kernel_name, compile_fn, key_files)
    if spec.is_compiled:
        return
    lock = FileLock(
        get_tmpdir() / f"{spec.module_dir_name}.{spec.kernel_name}.lock",
        thread_local=False,
    )
    if _SKIP_BUSY.get():
        try:
            lock.acquire(timeout=0)
        except Timeout:
            raise KernelBusy(spec.name) from None
    else:
        lock.acquire()
    try:
        if spec.is_compiled:
            return
        spec.compile_and_persist()
        if not spec.is_compiled:
            raise RuntimeError(f"PrimTS kernel was not persisted: {spec.name}")
    finally:
        lock.release()


def _probe_kernel(module_name: str, kernel_name: str, compile_fn, key_files) -> None:
    spec = _make_spec(module_name, kernel_name, compile_fn, key_files)
    if not spec.is_compiled:
        raise KernelNotCached(spec.name)


@functools.cache
def _kernel_source_files() -> tuple[str, ...]:
    # Family adapters share masks, reductions, sparse/Sage preparation and
    # compatibility helpers. Invalidate them together when any source changes.
    root = Path(__file__).resolve().parent
    return tuple(sorted(str(path) for path in root.rglob("*.py")))


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
    from flashinfer.jit.cute_dsl_core import (
        build_and_load_cute_dsl_kernel,
        cute_dsl_cache_disabled,
    )

    mode = _PRECOMPILE_MODE.get()
    if mode == "persist" and os.environ.get("FLASHINFER_DISABLE_JIT"):
        raise RuntimeError(
            "PrimTS warm-up precompile cannot compile with FLASHINFER_DISABLE_JIT set."
        )
    if cute_dsl_cache_disabled():
        if mode is not None:
            raise RuntimeError(
                "PrimTS warm-up precompile needs the on-disk CuTe DSL cache "
                "(FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 is set)."
            )
        return compile_fn()

    key = _compile_cache_key(spec, options, extra)
    if key is None:
        if mode is not None:
            raise RuntimeError(
                "PrimTS warm-up precompile cannot key this kernel for the on-disk cache."
            )
        return compile_fn()
    import torch

    # A family target such as sm100f can cover physical GPUs with different
    # host-selected tuning defaults. The module metadata keys the target;
    # the specialization must also distinguish the physical capability.
    major, minor = torch.cuda.get_device_capability()
    if mode == "persist":
        _precompile_kernel(
            f"prims_ts_{family}",
            f"{kind}_sm{major}{minor}_{key}",
            compile_fn,
            _kernel_source_files(),
        )
        return _unloaded_kernel
    if mode == "probe":
        _probe_kernel(
            f"prims_ts_{family}",
            f"{kind}_sm{major}{minor}_{key}",
            compile_fn,
            _kernel_source_files(),
        )
        return _unloaded_kernel
    return build_and_load_cute_dsl_kernel(
        f"prims_ts_{family}",
        f"{kind}_sm{major}{minor}_{key}",
        compile_fn,
        extra_key_files=_kernel_source_files(),
    )
