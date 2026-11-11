# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest plugin wiring for automatic MPI session reuse (repo-wide).

Loaded via ``-p test_common.session_reuse_hooks`` from each test tree's
pytest.ini. Factory installation is LAZY: nothing is patched until the test
suite itself imports tensorrt_llm's executor modules (which only happens for
tests that create MPI pools), so suites that never create pools pay nothing —
not even the tensorrt_llm import.
"""

import functools
import multiprocessing.process
import os
import subprocess

import pytest

from test_common.session_reuse import REUSE

# Node-id substrings that are opted out of session reuse as a class, in
# addition to the per-test ``private_mpi_session`` marker. Torch compile's
# Userbuffers Manager is process-global and assumes one Engine per process, so it
# cannot safely serve a later Engine whose pool size increases. Remove entries
# once the underlying state is re-initialized per engine.
_PRIVATE_NODEID_PATTERNS = (
    "torch_compile=true",  # Parameterized compile cases.
    "piecewise_cuda_graph",  # Cases with an unconditional compile config.
)


def _is_private_nodeid(nodeid: str) -> bool:
    lowered = nodeid.lower()
    return any(pat.lower() in lowered for pat in _PRIVATE_NODEID_PATTERNS)


# A test's child processes cannot use the cached pools but run on the GPUs
# their idle workers occupy, so starting one during setup or call drains the
# cache first. nvidia-smi only queries devices, so it is exempt. Spawns through
# subprocess.Popen and multiprocessing are covered; os.fork and os.posix_spawn
# are not.
_POPEN_INIT = subprocess.Popen.__init__
_PROCESS_START = multiprocessing.process.BaseProcess.start
_drain_on_spawn = False


def _is_device_query(args) -> bool:
    if isinstance(args, (str, bytes, os.PathLike)):
        args = os.fsdecode(args).split()
    try:
        program = os.path.basename(os.fsdecode(args[0]))
    except (IndexError, TypeError):
        return False
    return os.path.splitext(program)[0] == "nvidia-smi"


def _drain_before_spawn(child: str) -> None:
    global _drain_on_spawn
    # Processes started by the drain itself must not re-enter it.
    _drain_on_spawn = False
    try:
        REUSE.drain(reason=f"child process {child}")
    finally:
        _drain_on_spawn = True


@functools.wraps(_POPEN_INIT)
def _popen_init(self, args, *popenargs, **kwargs):
    if _drain_on_spawn and not _is_device_query(args):
        _drain_before_spawn(repr(args))
    _POPEN_INIT(self, args, *popenargs, **kwargs)


@functools.wraps(_PROCESS_START)
def _process_start(self):
    if _drain_on_spawn:
        _drain_before_spawn(repr(self.name))
    _PROCESS_START(self)


def _set_drain_on_spawn(enabled: bool) -> None:
    global _drain_on_spawn
    if enabled:
        subprocess.Popen.__init__ = _popen_init
        multiprocessing.process.BaseProcess.start = _process_start
    _drain_on_spawn = enabled


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "private_mpi_session: opt this test out of automatic MPI session "
        "reuse; it gets a fresh private pool and the reuse cache is drained "
        "first (required for LLM tests using RPC, whose pool is created before "
        "the RPC executor, and other tests needing process isolation)",
    )


def pytest_runtest_setup(item):
    if not REUSE.enabled:
        return
    opt_out = item.get_closest_marker("private_mpi_session") is not None or _is_private_nodeid(
        item.nodeid
    )
    if opt_out:
        REUSE.drain()
    REUSE.suspend(opt_out)
    REUSE.install_pool_factory_if_loaded()
    _set_drain_on_spawn(True)


def pytest_runtest_teardown(item):
    _set_drain_on_spawn(False)


# Failure fence: a failed test's pool can pass the health probe (workers alive
# and responsive) while still carrying worker-side state the probe cannot see,
# and would otherwise serve up to max_uses-1 more tests. Treat any failed item
# as untrusted and drain ALL cached pools after its teardown (by which time
# the pool has been returned to the cache). This is a conservative fallback
# for contamination the probe cannot detect — not a replacement for the
# broken-pool retirement path — and costs nothing on the passing-test path.
_FAILED_ITEMS: set = set()


def pytest_runtest_logreport(report):
    if report.failed:
        _FAILED_ITEMS.add(report.nodeid)


def pytest_runtest_logfinish(nodeid, location):
    if nodeid in _FAILED_ITEMS:
        _FAILED_ITEMS.discard(nodeid)
        if REUSE.enabled:
            REUSE.drain()


def pytest_sessionfinish(session, exitstatus):
    REUSE.drain()
