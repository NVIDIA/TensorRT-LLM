# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL cache operations for PrimTS with the pinned FlashInfer 0.7.3 runtime."""

import json
import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import RLock
from typing import Any, TextIO

from filelock import FileLock
from flashinfer.jit import cute_dsl_core
from flashinfer.jit.core import MissingJITCacheError, get_tmpdir, logger
from flashinfer.jit.cute_dsl_core import JitSpecCuteDsl as _FlashInferJitSpecCuteDsl
from flashinfer.jit.cute_dsl_core import _hash_source_files, _read_meta, cute_dsl_cache_disabled

# FlashInfer 0.7.3 hashes the explicitly selected source files.
cute_dsl_strict_cache_key = getattr(cute_dsl_core, "cute_dsl_strict_cache_key", lambda: False)

# CuTe DSL compiler options are process-global, even for distinct kernels.
_compile_lock = RLock()


def _reset_compile_lock() -> None:
    global _compile_lock
    _compile_lock = RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_compile_lock)


def _compile_in_process(compile_fn: Callable[[], Any]) -> Any:
    with _compile_lock:
        return compile_fn()


_SHARED_ENV = "TRTLLM_FLASHINFER_SHARED_CUTE_DSL"
_MANAGED_WORKSPACE_ENV = "TRTLLM_FLASHINFER_WORKSPACE_MANAGED"
_WORKSPACE_ENV = "FLASHINFER_WORKSPACE_BASE"
_SHARED_DIR_NAME = "cute_dsl"


def shared_jit_dir() -> Path | None:
    """The PrimTS CuTe DSL cache that the ranks on this host share, if any.

    TensorRT-LLM gives every MPI rank its own FlashInfer workspace, a ``rank-N``
    slot, so that no rank replaces compiler inputs another rank is reading.
    CuTe DSL kernels have no such inputs: each compiles in-process and is
    published as an object file plus a ``meta.json`` marker, both by atomic
    rename and under file locks. Their cache therefore lives once per host, in
    a ``cute_dsl`` directory beside the slots that mirrors the FlashInfer JIT
    directory's layout below a workspace. None for a workspace TensorRT-LLM
    does not manage, or with ``TRTLLM_FLASHINFER_SHARED_CUTE_DSL=0``.
    """
    if os.environ.get(_SHARED_ENV, "1") == "0" or os.environ.get(_MANAGED_WORKSPACE_ENV) != "1":
        return None
    workspace = os.environ.get(_WORKSPACE_ENV)
    if not workspace or not Path(workspace).name.startswith("rank-"):
        # Temporary fallback workspaces are unique to their process.
        return None
    workspace = Path(workspace)
    try:
        relative = Path(cute_dsl_core.jit_env.FLASHINFER_JIT_DIR).relative_to(workspace)
    except ValueError:
        return None
    return workspace.parent / _SHARED_DIR_NAME / relative


def cache_dir() -> Path:
    """The directory holding this process's PrimTS CuTe DSL kernels and their locks."""
    return shared_jit_dir() or Path(cute_dsl_core.jit_env.FLASHINFER_JIT_DIR)


class JitSpecCuteDsl(_FlashInferJitSpecCuteDsl):
    """Compile each specialization under its own lock and publish under the module lock."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._shared_dir = shared_jit_dir()
        if self._shared_dir is not None:
            self.module_dir = self._shared_dir / self.module_dir_name
            self.object_path = self.module_dir / f"{self.kernel_name}.o"

    def _lock_dir(self) -> Path:
        if self._shared_dir is None:
            return get_tmpdir()
        lock_dir = self._shared_dir / "tmp"
        lock_dir.mkdir(parents=True, exist_ok=True)
        return lock_dir

    @property
    def lock_path(self) -> Path:
        return self._lock_dir() / f"{self.module_dir_name}.{self.kernel_name}.lock"

    @property
    def module_lock_path(self) -> Path:
        return self._lock_dir() / f"{self.module_dir_name}.lock"

    def _same_cache_generation(self, marker: TextIO) -> bool | None:
        """False if the marker was replaced or removed, None if it cannot be checked."""
        try:
            return os.path.samestat(os.fstat(marker.fileno()), self.meta_path.stat())
        except (FileNotFoundError, NotADirectoryError):
            return False
        except OSError:
            return None

    def _cached_entry(self, *, load: bool) -> Any:
        # meta.json is the generation's commit marker, published after the
        # object. Keep its inode alive across the lookup to detect replacement
        # without writing a lock file on cache hits.
        try:
            marker = self.meta_path.open()
        except OSError:
            return None
        with marker:
            try:
                if json.load(marker) != self.expected_meta or not self.object_path.is_file():
                    return None
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return None

            try:
                value = self._load_from_disk() if load else True
            except Exception:
                # A sibling can invalidate the object after the existence check.
                # Preserve genuine loader errors for the same (or unknown) generation.
                if self._same_cache_generation(marker) is False:
                    return None
                raise
            return value if self._same_cache_generation(marker) else None

    @property
    def is_compiled(self) -> bool:
        return self._cached_entry(load=False) is True

    def try_load(self) -> Any:
        """The cached kernel iff its object and metadata belong to the same generation."""
        try:
            return self._cached_entry(load=True)
        except Exception as e:
            logger.warning(
                f"Failed to load cached CuTe-DSL kernel {self.object_path}: {e}. Recompiling."
            )
            return None

    def build_and_load(self) -> Any:
        if os.environ.get("FLASHINFER_DISABLE_JIT"):
            cached = self.try_load()
            if cached is None:
                raise MissingJITCacheError(
                    f"JIT compilation is disabled via FLASHINFER_DISABLE_JIT and no usable "
                    f"cached kernel was found for {self.name}.",
                    spec=self,
                )
            return cached
        return super().build_and_load()

    def build(self) -> None:
        logger.info(f"Compiling CuTe-DSL kernel {self.name}")
        self._compiled_kernel = _compile_in_process(self.compile_fn)
        with FileLock(self.module_lock_path, thread_local=False):
            try:
                if self.module_dir.exists() and _read_meta(self.meta_path) != self.expected_meta:
                    logger.info(f"Invalidating stale CuTe-DSL module {self.module_dir_name}")
                    # Retire the generation before removing any of its objects.
                    self.meta_path.unlink(missing_ok=True)
                    shutil.rmtree(self.module_dir)
                self._export()
            except Exception as error:
                # Keep the in-process kernel when the disk cache cannot be written.
                logger.warning(f"Failed to persist {self.name} to {self.object_path}: {error}")

    def compile_and_persist(self) -> None:
        """Meant for parallel precompilation workers."""
        with FileLock(self.lock_path, thread_local=False):
            self._timed_build()

    def load(self) -> Any:
        kernel = self._cached_entry(load=True)
        if kernel is not None:
            return kernel
        if self._compiled_kernel is not None:
            return self._compiled_kernel
        raise RuntimeError(f"No valid cached CuTe-DSL kernel for {self.name}")


def build_and_load_cute_dsl_kernel(
    module_name: str,
    kernel_name: str,
    compile_fn: Callable[[], Any],
    extra_key_files: Sequence[str] = (),
    *,
    source_sha256: str | None = None,
) -> Any:
    if cute_dsl_cache_disabled():
        return _compile_in_process(compile_fn)
    if source_sha256 is None:
        try:
            source_sha256 = _hash_source_files(tuple(extra_key_files))
        except (OSError, TypeError, SyntaxError, ValueError) as error:
            logger.warning(
                f"Cannot hash sources for {module_name}/{kernel_name}: {error}. Bypassing cache."
            )
            return _compile_in_process(compile_fn)
    return JitSpecCuteDsl(module_name, kernel_name, compile_fn, source_sha256).build_and_load()
