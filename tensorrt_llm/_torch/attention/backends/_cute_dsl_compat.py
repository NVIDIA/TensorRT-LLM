# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Parallel CuTe DSL kernel builds for the PrimTS attention backends."""

import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from filelock import FileLock
from flashinfer.jit.core import get_tmpdir, logger
from flashinfer.jit.cute_dsl_core import JitSpecCuteDsl as _FlashInferJitSpecCuteDsl
from flashinfer.jit.cute_dsl_core import _hash_source_files, _read_meta, cute_dsl_cache_disabled


class JitSpecCuteDsl(_FlashInferJitSpecCuteDsl):
    """Compile each specialization under its own lock and publish under the module lock."""

    @property
    def lock_path(self) -> Path:
        return get_tmpdir() / f"{self.module_dir_name}.{self.kernel_name}.lock"

    @property
    def module_lock_path(self) -> Path:
        return get_tmpdir() / f"{self.module_dir_name}.lock"

    def try_load(self) -> Any:
        with FileLock(self.module_lock_path, thread_local=False):
            if not self.is_compiled:
                return None
            try:
                return self._load_from_disk()
            except Exception as error:
                logger.warning(f"Failed to load {self.object_path}: {error}. Recompiling.")
                return None

    def build(self) -> None:
        logger.info(f"Compiling CuTe-DSL kernel {self.name}")
        self._compiled_kernel = self.compile_fn()
        with FileLock(self.module_lock_path, thread_local=False):
            if self.module_dir.exists() and _read_meta(self.meta_path) != self.expected_meta:
                logger.info(f"Invalidating stale CuTe-DSL module {self.module_dir_name}")
                shutil.rmtree(self.module_dir, ignore_errors=True)
            try:
                self._export()
            except Exception as error:
                # Keep the in-process kernel when the disk cache cannot be written.
                logger.warning(f"Failed to persist {self.name} to {self.object_path}: {error}")

    def compile_and_persist(self) -> None:
        """Meant for parallel precompilation workers."""
        with FileLock(self.lock_path, thread_local=False):
            self._timed_build()

    def load(self) -> Any:
        with FileLock(self.module_lock_path, thread_local=False):
            if self.is_compiled:
                return self._load_from_disk()
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
        return compile_fn()
    try:
        source_hash = _hash_source_files(tuple(extra_key_files))
    except (OSError, TypeError, SyntaxError, ValueError) as error:
        logger.warning(
            f"Cannot hash sources for {module_name}/{kernel_name}: {error}. Bypassing cache."
        )
        return compile_fn()
    return JitSpecCuteDsl(module_name, kernel_name, compile_fn, source_hash).build_and_load()
