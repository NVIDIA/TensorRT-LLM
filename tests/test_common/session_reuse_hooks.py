# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest plugin wiring for automatic MPI session reuse (repo-wide).

Loaded via ``-p test_common.session_reuse_hooks`` from each test tree's
pytest.ini. Factory installation is LAZY: nothing is patched until the test
suite itself imports tensorrt_llm's executor modules (which only happens for
tests that create MPI pools), so suites that never create pools pay nothing —
not even the tensorrt_llm import.
"""

from test_common.session_reuse import REUSE


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "private_mpi_session: opt this test out of automatic MPI session "
        "reuse; it gets a fresh private pool and the reuse cache is drained "
        "first (use for Ray/custom executors or tests needing isolation; RPC "
        "executors are handled automatically at their construction seam)",
    )


def pytest_runtest_setup(item):
    if not REUSE.enabled:
        return
    opt_out = item.get_closest_marker("private_mpi_session") is not None
    if opt_out:
        REUSE.drain()
    REUSE.suspend(opt_out)
    REUSE.install_pool_factory_if_loaded()


def pytest_sessionfinish(session, exitstatus):
    REUSE.drain()
