# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Parallel CuTe DSL kernel builds for the PrimTS attention backends."""

import json
import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import RLock
from typing import Any

from filelock import FileLock
from flashinfer.jit.core import get_tmpdir, logger
from flashinfer.jit.cute_dsl_core import JitSpecCuteDsl as _FlashInferJitSpecCuteDsl
from flashinfer.jit.cute_dsl_core import _hash_source_files, _read_meta, cute_dsl_cache_disabled

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


class JitSpecCuteDsl(_FlashInferJitSpecCuteDsl):
    """Compile each specialization under its own lock and publish under the module lock."""

    @property
    def lock_path(self) -> Path:
        return get_tmpdir() / f"{self.module_dir_name}.{self.kernel_name}.lock"

    @property
    def module_lock_path(self) -> Path:
        return get_tmpdir() / f"{self.module_dir_name}.lock"

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

            value = self._load_from_disk() if load else True

            try:
                if os.path.samestat(os.fstat(marker.fileno()), self.meta_path.stat()):
                    return value
            except OSError:
                return None
        return None

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

    def build(self) -> None:
        logger.info(f"Compiling CuTe-DSL kernel {self.name}")
        self._compiled_kernel = _compile_in_process(self.compile_fn)
        with FileLock(self.module_lock_path, thread_local=False):
            try:
                if self.module_dir.exists() and _read_meta(self.meta_path) != self.expected_meta:
                    logger.info(f"Invalidating stale CuTe-DSL module {self.module_dir_name}")
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
) -> Any:
    if cute_dsl_cache_disabled():
        return _compile_in_process(compile_fn)
    try:
        source_hash = _hash_source_files(tuple(extra_key_files))
    except (OSError, TypeError, SyntaxError, ValueError) as error:
        logger.warning(
            f"Cannot hash sources for {module_name}/{kernel_name}: {error}. Bypassing cache."
        )
        return _compile_in_process(compile_fn)
    return JitSpecCuteDsl(module_name, kernel_name, compile_fn, source_hash).build_and_load()
