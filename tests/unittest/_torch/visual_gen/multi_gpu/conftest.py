# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest hooks for the rank-process pool used by the tests in this package.

- Failure fence: a test can fail *after* its pool call returned (an assertion
  in the test body), in which case the pool never saw the failure and would
  stay cached. Treat any failed test as untrusted and shut every cached pool
  down after its teardown, the same rule the MPI session reuse applies.
- Session end: shut the pools down before pytest exits so their GPU memory is
  released on the pool's terms rather than by the interpreter's exit sequence.
- Module lifetime: with TRTLLM_TEST_VISUALGEN_POOL_LIFETIME=module, shut the
  pools down at every test file's teardown.
- Deadline: at test setup, when pytest-timeout's clock is running, hand the
  pool the test's effective pytest-timeout (marker, then --timeout, then
  PYTEST_TIMEOUT, then the ini key: the plugin's own precedence). Every pool
  call in the test is then bounded by the time remaining.
"""

import os

import pytest

from ._worker_pool import lifetime, set_test_deadline, shutdown_all


def pytest_sessionstart(session):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        print("[visualgen-pool] pytest-xdist worker: pool lifetime forced to test", flush=True)


_FAILED_ITEMS: set[str] = set()


def _effective_pytest_timeout(item) -> float | None:
    marker = item.get_closest_marker("timeout")
    if marker is not None:
        value = marker.kwargs.get("timeout", marker.args[0] if marker.args else None)
        if value is not None:
            return float(value)
    value = item.config.getoption("timeout", default=None)
    if value is None:
        value = os.environ.get("PYTEST_TIMEOUT") or None
    if value is None:
        try:
            value = item.config.getini("timeout") or None
        except ValueError:  # pytest-timeout not installed: no such ini key
            value = None
    return float(value) if value is not None else None


def pytest_runtest_setup(item):
    set_test_deadline(_effective_pytest_timeout(item))


def end_of_module() -> None:
    """Module-teardown action: under lifetime ``module``, shut the pools down."""
    if lifetime() == "module":
        shutdown_all()


@pytest.fixture(autouse=True, scope="module")
def _pool_module_lifetime():
    yield
    end_of_module()


def pytest_runtest_logreport(report):
    if report.failed:
        _FAILED_ITEMS.add(report.nodeid)


def pytest_runtest_logfinish(nodeid, location):
    if nodeid in _FAILED_ITEMS:
        _FAILED_ITEMS.discard(nodeid)
        shutdown_all()


def pytest_sessionfinish(session, exitstatus):
    shutdown_all()
