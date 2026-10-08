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

Initialization is driven from ``tensorrt_llm/__init__.py``:

1. :func:`_prepare_environment` sets process defaults before importing PyTorch.
2. PyTorch's build metadata selects the CUDA or ROCm execution path without
   initializing a device.
3. Only CUDA calls :func:`_prepare_native_environment` for TensorRT library
   preparation, followed by :func:`_init` for custom ops and MPI.

Module scope deliberately uses only the standard library. ROCm imports never
preload native TensorRT libraries, import MPI bindings, or load NVIDIA kernels.
"""

import os
import platform
import sys
import threading
import time
from pathlib import Path

# Disable UCC to WAR allgather issue before NGC PyTorch 25.12 upgrade.
os.environ["OMPI_MCA_coll_ucc_enable"] = "0"

_inited = False


def _set_numexpr_thread_default() -> None:
    # TRT-LLM uses NumExpr only for scalar checkpoint formulas. Avoid creating
    # its helper pool unless the embedding application explicitly opts in.
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


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
    """Set backend-independent environment defaults before importing PyTorch."""
    _set_numexpr_thread_default()


def _prepare_native_environment() -> None:
    """Prepare only NVIDIA libraries after PyTorch has identified its backend.

    ROCm does not need TensorRT DLL paths, a shared libpython, or vendored NVIDIA
    Triton kernels. CUDA calls this before any TensorRT-LLM shared object loads.
    """
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
