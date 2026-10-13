# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest hooks for the rank-process pool used by the tests in this package.

- Failure fence: a test can fail *after* its pool call returned (an assertion
  in the test body), in which case the pool never saw the failure and would
  stay cached. Treat any failed test as untrusted and shut every cached pool
  down after its teardown, the same rule the MPI session reuse applies.
- Session end: shut the pools down before pytest exits so their GPU memory is
  released on the pool's terms rather than by the interpreter's exit sequence.
"""

from ._worker_pool import shutdown_all

_FAILED_ITEMS: set[str] = set()


def pytest_runtest_logreport(report):
    if report.failed:
        _FAILED_ITEMS.add(report.nodeid)


def pytest_runtest_logfinish(nodeid, location):
    if nodeid in _FAILED_ITEMS:
        _FAILED_ITEMS.discard(nodeid)
        shutdown_all()


def pytest_sessionfinish(session, exitstatus):
    shutdown_all()
