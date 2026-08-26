# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generic process-launch-target detection, used by the guest bootstrap.

Opts build tooling and Ray infrastructure subtrees out of coverage instrumentation, and
identifies an `mpi4py.futures` pool worker so only that process type defers activation.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# mpi4py.futures always spawns a pool worker as `-m mpi4py.futures.server`
# (mpi4py/futures/_lib.py client_spawn(): `args = get_python_flags() + python_args +
# ['-m', get_spawn_module()]`, where get_spawn_module() == "mpi4py.futures.server").
#
# `is_mpi_pool_worker()` below scans sys.orig_argv for that literal token rather than
# requiring launch_target() to resolve it as the actual -m module, because one TRT-LLM
# code path overrides python_args (FlashInfer per-process workspace isolation passes
# "-c <bootstrap script>"): client_spawn still appends the trailing "-m
# mpi4py.futures.server" tokens in that case, even though they land after "-c" and so
# are not parsed as a real -m flag -- the bootstrap script imports mpi4py.futures.server
# itself instead. The token scan is the one check that matches both spawn paths.
_MPI_POOL_WORKER_MODULE = "mpi4py.futures.server"

# Interpreter options that consume the token after them, so the scan below can tell a script
# path from an option argument (``python -X importtime -m pytest`` must still yield "pytest").
_OPTIONS_WITH_ARGUMENT = frozenset(("-W", "-X", "--check-hash-based-pycs"))


def launch_target() -> tuple[Optional[str], Optional[str]]:
    """How this interpreter was started, as ``(module, script_basename)``.

    Exactly one is set: ``module`` for ``-m pkg.mod``, ``script_basename`` for a
    script or console-script path. Both are ``None`` for ``-c`` and for a bare
    REPL. Parsing stops at the launch target, so a program's own arguments can
    never be mistaken for the interpreter's.
    """
    argv = list(getattr(sys, "orig_argv", sys.argv) or [""])[1:]
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "-m":
            return (argv[index + 1] if index + 1 < len(argv) else None), None
        if token == "-c":
            return None, None
        if token in _OPTIONS_WITH_ARGUMENT:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return None, os.path.basename(token)
    return None, None


def is_dependency_build_process() -> bool:
    """Return True for pip / setuptools / native build-tool processes, which opt the subtree out.

    These are spawned by pip and the PEP 517 backend rather than by our own code, so there is
    nobody to hand them an explicit role; the launch target is the available signal.
    """
    module, script = launch_target()
    if module is not None:
        return module.split(".", 1)[0] in {"pip", "build", "pyproject_hooks", "setuptools"}
    if script is None:
        return False
    script = script.lower()
    return script in {
        "pip",
        "pip3",
        "cmake",
        "ninja",
        "ninja-build",
        "meson",
        "setup.py",
        "_in_process.py",
    }


def is_ray_infra_process() -> bool:
    """Return True for Ray infrastructure / worker processes, which opt the subtree out.

    The Ray stage (TLLM_DISABLE_MPI=1) nests Ray under mpi4py pool workers: each pool worker
    calls ``ray.init(address="local")`` which spawns ``raylet``, ``gcs_server``, dashboard,
    log_monitor, autoscaler.monitor, runtime_env.agent, and pre-starts up to 224 ``default_worker.py``
    processes. All of them inherit ``CBTS_COVERAGE_CONFIG``/``PYTHONPATH`` via default env inheritance.

    Activating CBTS in ``default_worker.py`` adds enough Python startup / sys.monitoring PY_START
    overhead that the workers can't register with raylet before its ``worker_pool.cc:600`` timeout,
    so raylet keeps spawning more, and the driver's ``ray.init()`` hangs in ``RegisterClient`` forever
    (observed in test_disaggregated_* under the Ray orchestrator stage). Opt out here so Ray's
    hot spawn path stays fast; the mpi4py pool worker still records coverage for the LLM API surface,
    and the RayGPUWorker actor itself lives inside a ``default_worker.py`` so it's uninstrumented too.

    raylet spawns these, not our own code, so there is nobody to hand them an explicit role;
    the launch target is the available signal.
    """
    module, script = launch_target()
    if module is not None:
        # Dotted-prefix match over the module namespace, so e.g.
        # ray.autoscaler._private.monitor is covered without matching a path that
        # merely contains the text.
        return module == "ray" or module.startswith(
            ("ray.autoscaler.", "ray.dashboard.", "ray._private.")
        )
    return script in {"default_worker.py", "setup_worker.py"}


def is_mpi_pool_worker() -> bool:
    """Return True in an ``mpi4py.futures`` pool worker process (spawned via ``MPI_Comm_spawn``).

    This is the *only* process type that needs deferred PY_START activation: it's the one
    place a slow instrumented import can trip ``MpiPoolSession``'s ``wait_shutdown`` identity
    barrier (``tensorrt_llm/llmapi/mpi_session.py``), whose deadline the process spawn plus
    this cold import shares. Every other product process (``trtllm-serve``, disagg helpers,
    ...) has no such budget to protect and should activate immediately instead.
    """
    argv = getattr(sys, "orig_argv", sys.argv) or ()
    return any(_MPI_POOL_WORKER_MODULE in token for token in argv)
