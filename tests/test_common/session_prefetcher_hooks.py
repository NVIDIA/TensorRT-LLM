# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest plugin wiring for the session prefetcher (repo-wide, demand-driven).

Loaded via ``-p test_common.session_prefetcher_hooks`` from each test tree's
pytest.ini. Factory installation is LAZY: nothing is patched until the test
suite itself imports tensorrt_llm's executor modules (which only happens for
tests that create MPI pools), so suites that never create pools pay nothing —
not even the tensorrt_llm import.
"""

import pytest

from test_common.session_prefetcher import PREFETCHER


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "prefetch_session(n_gpus): session spec for the background session prefetcher",
    )
    config.addinivalue_line(
        "markers",
        "prefetch_model_dir(path): model dir the test loads, for page-cache warming",
    )


def pytest_collection_modifyitems(items):
    PREFETCHER.on_collection(items)


def pytest_runtest_setup(item):
    PREFETCHER.install_pool_factory_if_loaded()
    PREFETCHER.on_test_setup(item)


def pytest_sessionfinish(session, exitstatus):
    PREFETCHER.dispose()


@pytest.fixture
def prefetched_mpi_session(request):
    """An ``MpiPoolSession`` for this test's ``prefetch_session(N)`` marker.

    Returns the pool prefetched in the background while the PREVIOUS test was
    running when available, else builds one synchronously. Pass it to
    ``LLM(..., _mpi_session=prefetched_mpi_session)``.
    """
    marker = request.node.get_closest_marker("prefetch_session")
    assert marker is not None and marker.args, (
        "prefetched_mpi_session requires @pytest.mark.prefetch_session(n_workers)"
    )
    n_workers = marker.args[0]
    session = PREFETCHER.take(n_workers)
    if session is None:
        from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

        session = MpiPoolSession(n_workers=n_workers)
    yield session
    session.shutdown()
