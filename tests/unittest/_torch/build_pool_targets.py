# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builds for test_build_pool.py and test_kernel_builder.py, whose worker processes import this
module.

Its "kernels" record the process that compiled them, and pickle. Imported in a worker process
(``IN_WORKER_ENV`` set), this module makes ``tensorrt_llm._jit_share`` export and load them; the
tests make it load them in the test process.
"""

import os
import pickle
import time
from typing import NamedTuple, Tuple

from tensorrt_llm import _jit_share
from tensorrt_llm._torch.build_pool import process_built

IN_WORKER_ENV = "BUILD_POOL_TEST_IN_WORKER"
# Variables of a launcher that the tests set in the test process.
LAUNCHER_VARS = ("SLURM_JOB_ID", "OMPI_COMM_WORLD_SIZE", "PMIX_RANK")
NAME = "build_pool_test"
# The kernels' names would hold the GPU's compute capability.
IDENTITY = "build-pool-test"


class Kernel(NamedTuple):
    value: int
    pid: int
    launcher_vars: Tuple[str, ...]


def export_kernel(kernel, obj: str, name: str, meta: str) -> None:
    with open(meta, "w") as f:
        f.write("{}")
    with open(obj + ".part", "wb") as f:
        pickle.dump(kernel, f)
    os.replace(obj + ".part", obj)


def load_kernel(obj: str, name: str, meta: str):
    with open(obj, "rb") as f:
        return pickle.load(f)


@process_built(NAME, use_build_pool_in_warmup=True, use_build_pool_in_rehearsal=True)
def compile_kernel(value: int, in_worker: str = "") -> Kernel:
    """Compile a kernel; in a worker process, ``in_worker="raise"`` raises and
    ``in_worker="exit"`` ends the process instead."""
    if os.environ.get(IN_WORKER_ENV) == "1":
        if in_worker == "raise":
            raise ValueError(f"cannot compile {value}")
        if in_worker == "exit":
            os._exit(3)
    launcher_vars = tuple(k for k in LAUNCHER_VARS if k in os.environ)
    return Kernel(value, os.getpid(), launcher_vars)


@process_built(NAME + "_in_rank")
def compile_kernel_in_rank(value: int) -> Kernel:
    """A compile function whose kernels build in the rank's process."""
    return compile_kernel.raw(value)


def build_by_running(value: int) -> None:
    """A build that runs an op: its call compiles the kernel, which the worker exports."""
    compile_kernel(value)


def record(path: str, value: int) -> None:
    """A build whose result reaches the rank through a file, as a compiler's disk cache."""
    with open(path, "w") as f:
        f.write(f"{value} {os.getpid()}")


def span(path: str, seconds: float) -> None:
    """A build that takes ``seconds`` and records when it started and ended, in ``path``."""
    start = time.time()
    time.sleep(seconds)
    with open(path, "w") as f:
        f.write(f"{start} {time.time()}")


def fail() -> None:
    raise ValueError("a build that fails")


def double(value: int) -> int:
    """A build whose function returns a value: twice ``value``."""
    return 2 * value


def unpicklable() -> object:
    """A build whose function returns a value that does not pickle."""
    return lambda: None


def hold(started: str, release: str, timeout: float = 120.0) -> None:
    """A build that creates the file ``started`` and runs until the file ``release`` exists,
    for at most ``timeout`` seconds."""
    open(started, "w").close()
    deadline = time.monotonic() + timeout
    while not os.path.exists(release):
        if time.monotonic() > deadline:
            raise TimeoutError(f"{release} was not created within {timeout}s")
        time.sleep(0.05)


_context_lost = False


def lose_context() -> None:
    """A build that costs the worker its CUDA context, as a kernel fault does."""
    global _context_lost
    _context_lost = True
    raise RuntimeError("an illegal memory access")


class LoadsOnlyHere:
    """An argument that pickles here and does not load in a worker process."""

    def __reduce__(self):
        return (_load_only_here, ())


def _load_only_here() -> LoadsOnlyHere:
    if os.environ.get(IN_WORKER_ENV) == "1":
        raise ImportError("not importable in a worker")
    return LoadsOnlyHere()


if os.environ.get(IN_WORKER_ENV) == "1":
    from tensorrt_llm._torch import build_pool as _build_pool

    _build_pool._cuda_context_broken = lambda: _context_lost
    _jit_share._export_kernel = export_kernel
    _jit_share._load_kernel = load_kernel
    _jit_share._cute_identity = lambda: IDENTITY
