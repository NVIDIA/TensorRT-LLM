# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import platform
import traceback
from typing import Any, Callable

from ..logger import logger

IS_FLASHINFER_AVAILABLE = False


def get_env_enable_pdl() -> bool:
    enabled = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    if enabled and not getattr(get_env_enable_pdl, "_printed", False):
        logger.info("PDL enabled")
        setattr(get_env_enable_pdl, "_printed", True)
    return enabled


if platform.system() != "Windows":
    try:
        import flashinfer
        logger.info(f"flashinfer is available: {flashinfer.__version__}")
        IS_FLASHINFER_AVAILABLE = True
    except ImportError:
        traceback.print_exc()
        print(
            "flashinfer is not installed properly, please try pip install or building from source codes"
        )

if IS_FLASHINFER_AVAILABLE:
    from .flashinfer_cubin_pool import install_shared_cubin_pool

    install_shared_cubin_pool()


def _jit_dir_scope() -> str:
    """Where FlashInfer's JIT modules land: the ranks whose builds go to the
    same directory on this host split them."""
    from flashinfer.jit import env as jit_env

    jit_dir = os.path.realpath(str(jit_env.FLASHINFER_JIT_DIR))
    return f"{os.uname().nodename}:{jit_dir}"


def queue_flashinfer_module_build(name: str, gen_spec: Callable[[],
                                                                Any]) -> None:
    """In a KernelBuilder rehearsal, queue the build of the FlashInfer JIT
    module that ``gen_spec()`` describes on the rank's worker processes, once
    per ``name``; outside a rehearsal, nothing.

    ``gen_spec`` is a module-level function, which the worker imports. With
    shared JIT builds, the ranks that build into the same FlashInfer JIT
    directory on this host split the modules (``BuildStrategy.SHARED``).
    """
    if not IS_FLASHINFER_AVAILABLE:
        return
    from .kernel_builder import (BuildConfig, BuildStrategy, BuildTask,
                                 KernelBuilder)

    builder = KernelBuilder.get()
    if not builder.rehearsing():
        return
    # Not thread-safe: in a rehearsal the build goes to a worker process.
    config = BuildConfig(strategy=BuildStrategy.SHARED,
                         share_scope=_jit_dir_scope,
                         use_build_pool_in_rehearsal=True)
    task = BuildTask(f"{__name__}:build_flashinfer_module", (gen_spec, ))
    builder.build("flashinfer", [(name, task, config)])


def build_flashinfer_module(gen_spec: Callable[[], Any]) -> bool:
    """In a worker process: build the FlashInfer JIT module ``gen_spec()``
    describes, without loading it, under the lock FlashInfer's own
    ``build_and_load`` takes, so that the first call that needs the module
    finds it built, or waits on that lock, and loads it. False when there is
    nothing to build: an AOT or built module, or FlashInfer's JIT turned off.
    """
    if os.environ.get("FLASHINFER_DISABLE_JIT"):
        return False
    spec = gen_spec()
    if getattr(spec, "is_aot", False) or spec.is_compiled:
        return False
    spec.build(need_lock=True)
    return True
