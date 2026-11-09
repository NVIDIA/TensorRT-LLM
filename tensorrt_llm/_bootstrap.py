# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Package bootstrap mechanics for ``tensorrt_llm``.

Two phases, in this order, both driven from ``tensorrt_llm/__init__.py``:

1. :func:`_prepare_environment` -- process environment and compiled-artifact
    cache defaults, DLL search path, Python-library preload and vendored
    ``triton_kernels`` precedence. It must run *before* ``torch`` and before
    any TensorRT-LLM shared object is loaded. ``__init__`` runs it and
    ``import torch`` inside :func:`_mimalloc_arenas_off_thp`, which keeps
    torch's CPU allocator off transparent huge pages.
2. :func:`_init` -- custom-op library loading and MPI initialization.  It runs
   after the package's own imports have completed.

Module scope here is deliberately limited to the standard library.  ``__init__``
imports this module ahead of ``import torch``, so anything imported at module
scope would be pulled in before phase 1 has run -- including ``torch`` itself,
which is exactly what phase 1 exists to prepare for.  Phase 2's imports are
therefore inside :func:`_init`; by the time it is called ``torch``,
``tensorrt_llm.bindings``, ``tensorrt_llm._utils`` and ``tensorrt_llm.logger``
are already in ``sys.modules``, so this defers the *statement* and not the first
import of any module.
"""

import contextlib
import functools
import os
import platform
import sys
import threading
import time
import warnings
from collections.abc import Iterator
from pathlib import Path

# Disable UCC to WAR allgather issue before NGC PyTorch 25.12 upgrade.
os.environ["OMPI_MCA_coll_ucc_enable"] = "0"

_inited = False

_INDUCTOR_QUIESCE_POOL_ENV = "TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL"
_MIMALLOC_ALLOW_THP_ENV = "MIMALLOC_ALLOW_THP"
_PR_SET_THP_DISABLE = 41
_PR_GET_THP_DISABLE = 42
# PR_SET_THP_DISABLE flag (Linux 6.18+): THP stays allowed where MADV_HUGEPAGE
# asks for it. PR_GET_THP_DISABLE then reads 1 | this.
_PR_THP_DISABLE_EXCEPT_ADVISED = 1 << 1
_UNIFIED_CACHE_ROOT_ENV = "TRTLLM_CACHE_DIR"
_TRTLLM_DG_CACHE_ENV = "TRTLLM_DG_CACHE_DIR"
_TRTLLM_DG_DUMP_CUBIN_ENV = "TRTLLM_DG_JIT_DUMP_CUBIN"
_UNIFIED_CACHE_ENV_VARS = (
    "TLLM_AUTOTUNER_CACHE_PATH",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "CUTE_DSL_CACHE_DIR",
    "DG_JIT_CACHE_DIR",
    _TRTLLM_DG_CACHE_ENV,
    "CUDA_CACHE_PATH",
)


def _unified_cache_defaults(cache_root: str) -> dict[str, str]:
    """Return cache environment defaults rooted at ``cache_root``."""
    cache_root = os.path.expanduser(cache_root)
    return {
        "TLLM_AUTOTUNER_CACHE_PATH": os.path.join(cache_root, "autotuner", "cache.json"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(cache_root, "inductor"),
        "TRITON_CACHE_DIR": os.path.join(cache_root, "triton"),
        "TORCH_EXTENSIONS_DIR": os.path.join(cache_root, "torch_extensions"),
        "FLASHINFER_WORKSPACE_BASE": os.path.join(cache_root, "flashinfer"),
        "CUTE_DSL_CACHE_DIR": os.path.join(cache_root, "cute_dsl"),
        "DG_JIT_CACHE_DIR": os.path.join(cache_root, "deep_gemm"),
        _TRTLLM_DG_CACHE_ENV: os.path.join(cache_root, "trtllm_deep_gemm"),
        "CUDA_CACHE_PATH": os.path.join(cache_root, "cuda"),
    }


def _setup_unified_cache() -> None:
    """Configure compiled-artifact cache defaults before framework imports."""
    cache_root = os.environ.get(_UNIFIED_CACHE_ROOT_ENV)
    if cache_root:
        for name, path in _unified_cache_defaults(cache_root).items():
            os.environ.setdefault(name, path)

    if os.environ.get(_TRTLLM_DG_CACHE_ENV):
        os.environ.setdefault(_TRTLLM_DG_DUMP_CUBIN_ENV, "1")


def _set_numexpr_thread_default() -> None:
    # TRT-LLM uses NumExpr only for scalar checkpoint formulas. Avoid creating
    # its helper pool unless the embedding application explicitly opts in.
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@functools.lru_cache(maxsize=1)
def _libc_prctl():
    """libc's ``prctl`` with its argument types set, or None where it is unavailable."""
    import ctypes

    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (AttributeError, OSError):
        return None
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    return prctl


def _thp_disable_flag() -> int | None:
    """Return this process's PR_GET_THP_DISABLE value, or None where it cannot be read.

    0 allows transparent huge pages, 1 disables them, and
    1 | _PR_THP_DISABLE_EXCEPT_ADVISED disables them except for memory advised
    MADV_HUGEPAGE.
    """
    prctl = _libc_prctl() if sys.platform == "linux" else None
    if prctl is None:
        return None
    value = prctl(_PR_GET_THP_DISABLE, 0, 0, 0, 0)
    return None if value < 0 else value


def _set_thp_disable_flag(value: int) -> None:
    """Set this process's THP-disable flag to a value `_thp_disable_flag` returned.

    Warns if the kernel refuses, e.g. under a seccomp policy that lets the flag
    be read but not set; the process then keeps the flag it has.
    """
    import ctypes

    prctl = _libc_prctl()
    flags = value & _PR_THP_DISABLE_EXCEPT_ADVISED
    if prctl is not None and prctl(_PR_SET_THP_DISABLE, value & 1, flags, 0, 0) == 0:
        return
    reason = os.strerror(ctypes.get_errno()) if prctl is not None else "prctl is unavailable"
    warnings.warn(
        f"Could not set the transparent huge page disable flag of this process to {value}: "
        f"{reason}",
        RuntimeWarning,
        stacklevel=2,
    )


@contextlib.contextmanager
def _mimalloc_arenas_off_thp() -> Iterator[None]:
    """Keep torch's mimalloc arenas off transparent huge pages while torch loads.

    ``tensorrt_llm/__init__.py`` runs phase 1 and ``import torch`` in this
    context. On Linux it sets ``MIMALLOC_ALLOW_THP=0`` unless the caller set the
    variable, and on exit, also when the body raises, withdraws it and puts the
    process's THP-disable flag back if the import changed it.

    PyTorch's CPU allocator on aarch64 is mimalloc, which by default advises its
    1 GiB arenas MADV_HUGEPAGE. With 64 KiB base pages a transparent huge page is
    512 MiB. Once a free inside the arena splits one, khugepaged later collapses
    the range again, copying 512 MiB while it holds the process's mmap lock, so a
    page fault in any thread of the process (the executor touching a fresh heap
    page) waits tens of milliseconds, and its tensor-parallel peers wait with it.
    mimalloc reads the option once, when torch loads it, so the variable only has
    to be there for the import. It is set on every Linux platform but matters
    only where torch's allocator is mimalloc.

    The default only takes effect when tensorrt_llm is imported before torch. MPI
    pool workers re-run the launching script's module-level imports, so they
    follow its order. A value the caller sets wins and is left as it is: a
    caller-set ``MIMALLOC_ALLOW_THP=0`` keeps transparent huge pages disabled for
    the whole process and the processes it starts, including memory that asks
    for them.

    With the option off, mimalloc also disables transparent huge pages for the
    whole process (PR_SET_THP_DISABLE) when it initializes. Only torch's arenas
    need to stay off huge pages: memory that asks for them with MADV_HUGEPAGE
    (the MoE load balancer's NUMA pool, the KVCacheManagerV2 host tier under
    TLLM_KV_CACHE_MANAGER_V2_THP=1) must keep getting them. Withdrawing the
    variable also keeps it from later libraries that bundle their own mimalloc
    (pyarrow does) and from child processes that do not import TensorRT LLM.

    Torch's arenas then stay off huge pages only under the ``madvise`` THP mode
    (/sys/kernel/mm/transparent_hugepage/enabled), where memory that was not
    advised gets none. Under ``always`` they are eligible again, as they are
    without this default; the flag is put back there too, because keeping it
    would also deny huge pages to the memory that asks for them.
    """
    if sys.platform != "linux" or _MIMALLOC_ALLOW_THP_ENV in os.environ:
        yield
        return
    before = _thp_disable_flag()
    os.environ[_MIMALLOC_ALLOW_THP_ENV] = "0"
    try:
        yield
    finally:
        os.environ.pop(_MIMALLOC_ALLOW_THP_ENV, None)
        after = _thp_disable_flag()
        if before is not None and after is not None and after != before:
            _set_thp_disable_flag(before)


def _keep_inductor_compile_pool_alive() -> None:
    # Inductor shuts down its async-compile worker pool after a short idle
    # period. A compile submitted while that shutdown is in flight can wait
    # forever, and a rank stuck there stalls every collective of its group
    # (e.g. a torch.compile'd helper recompiling inside a TP forward).
    os.environ.setdefault(_INDUCTOR_QUIESCE_POOL_ENV, "0")


def _add_trt_llm_dll_directory():
    on_windows = platform.system() == "Windows"
    if on_windows:
        import sysconfig

        os.add_dll_directory(Path(sysconfig.get_paths()["purelib"]) / "tensorrt_llm" / "libs")


def _preload_python_lib():
    """
    Preload Python library.

    On Linux, the python executable links to libpython statically,
    so the dynamic library `libpython3.x.so` is not loaded.
    When using virtual environment on top of non-system Python installation,
    our libraries installed under `$VENV_PREFIX/lib/python3.x/site-packages/`
    have difficulties loading `$PREFIX/lib/libpython3.x.so.1.0` on their own,
    since venv does not symlink `libpython3.x.so` into `$VENV_PREFIX/lib/`,
    and the relative path from `$VENV_PREFIX` to `$PREFIX` is arbitrary.

    We preload the libraries here since the Python executable under `$PREFIX/bin`
    can easily find the library.
    """
    on_linux = platform.system() == "Linux"
    if on_linux:
        from ctypes import cdll

        v_major, v_minor, *_ = sys.version_info
        pythonlib = f"libpython{v_major}.{v_minor}.so"
        _ = cdll.LoadLibrary(pythonlib + ".1.0")
        _ = cdll.LoadLibrary(pythonlib)


def _setup_vendored_triton_kernels():
    """Ensure our vendored triton_kernels takes precedence over any existing installation.

    Some environments bundle triton_kernels, which can conflict with our vendored version. This function:
    1. Clears any pre-loaded triton_kernels from sys.modules
    2. Temporarily adds our package root to sys.path
    3. Imports triton_kernels (caching our version in sys.modules)
    4. Removes the package root from sys.path
    """

    # Clear any pre-loaded triton_kernels from cache
    for mod in list(sys.modules.keys()):
        if mod == "triton_kernels" or mod.startswith("triton_kernels."):
            del sys.modules[mod]

    # Temporarily add our package root to sys.path
    root = Path(__file__).parent.parent

    vendored = root / "triton_kernels"
    if not vendored.exists():
        raise RuntimeError(f"Vendored triton_kernels module not found at {vendored}")

    should_add_to_path = str(root) not in sys.path
    if should_add_to_path:
        sys.path.insert(0, str(root))

    import triton_kernels  # noqa: F401

    if should_add_to_path:
        sys.path.remove(str(root))


def _prepare_environment() -> None:
    """Phase 1: environment and library preparation, before the Torch import."""
    _set_numexpr_thread_default()
    _keep_inductor_compile_pool_alive()
    _setup_unified_cache()
    _add_trt_llm_dll_directory()
    _preload_python_lib()
    _setup_vendored_triton_kernels()


def _init(log_level: object = None) -> None:
    """Phase 2: custom-op registration and MPI initialization, after imports."""
    global _inited
    if _inited:
        return
    _inited = True

    import torch

    from ._utils import print_all_stacks
    from .bindings import MpiComm
    from .logger import logger

    if log_level is not None:
        logger.set_level(log_level)

    if os.getenv("TRT_LLM_NO_LIB_INIT", "0") == "1":
        logger.info("Skipping TensorRT LLM init.")
        return

    logger.info("Starting TensorRT LLM init.")

    project_dir = str(Path(__file__).parent.absolute())

    # Load FT decoder layer and torch custom ops.
    if platform.system() == "Windows":
        ft_decoder_lib = project_dir + "/libs/th_common.dll"
    else:
        ft_decoder_lib = project_dir + "/libs/libth_common.so"
    try:
        torch.classes.load_library(ft_decoder_lib)
        from ._torch.custom_ops import _register_fake

        _register_fake()
    except Exception as e:
        msg = (
            "\nFATAL: Decoding operators failed to load. This may be caused by an incompatibility "
            "between PyTorch and TensorRT-LLM. Please rebuild and install TensorRT-LLM."
        )
        raise ImportError(str(e) + msg)

    MpiComm.local_init()

    def _print_stacks():
        counter = 0
        while True:
            time.sleep(print_stacks_period)
            counter += 1
            logger.error(f"Printing stacks {counter} times")
            print_all_stacks()

    print_stacks_period = int(os.getenv("TRTLLM_PRINT_STACKS_PERIOD", "-1"))
    if print_stacks_period > 0:
        print_stacks_thread = threading.Thread(target=_print_stacks, daemon=True)
        print_stacks_thread.start()

    logger.info("TensorRT LLM inited.")
