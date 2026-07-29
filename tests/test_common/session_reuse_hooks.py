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

# Node-id substrings that are opted out of session reuse as a class, in
# addition to the per-test ``private_mpi_session`` marker. AutoDeploy's
# executor adapter keeps worker-process state sized for the previous engine's
# config (graph/sequence-length shaped buffers); on a reused pool the next
# test hits e.g. "The expanded size of the tensor (128) must match the
# existing size (256)". Remove entries once the underlying state is
# re-initialized per engine.
_PRIVATE_NODEID_PATTERNS = (
    "autodeploy",
    "auto_deploy",
    "/test_ad_",
)


def _is_private_nodeid(nodeid: str) -> bool:
    lowered = nodeid.lower()
    return any(pat in lowered for pat in _PRIVATE_NODEID_PATTERNS)


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
    opt_out = item.get_closest_marker("private_mpi_session") is not None or _is_private_nodeid(
        item.nodeid
    )
    if opt_out:
        REUSE.drain()
    REUSE.suspend(opt_out)
    REUSE.install_pool_factory_if_loaded()


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
