# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""llmapi-level pytest hooks wiring the opt-in session prefetcher.

See tests/test_common/session_prefetcher.py; enabled by
TRTLLM_TEST_PREFETCH_SESSION=1, no-op otherwise.
"""

import pytest
from test_common.session_prefetcher import PREFETCHER


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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "prefetch_session(n_gpus): session spec for the background session prefetcher",
    )
    config.addinivalue_line(
        "markers",
        "prefetch_model_dir(path): model dir the test loads, for page-cache warming",
    )
    if PREFETCHER.enabled:
        # Shadow mode: intercept bare LLM(...) pool creation so tests consume
        # prefetched pools with zero test changes.
        PREFETCHER.install_pool_factory()


def pytest_sessionfinish(session, exitstatus):
    PREFETCHER.dispose()


def pytest_collection_modifyitems(items):
    PREFETCHER.on_collection(items)


def pytest_runtest_setup(item):
    PREFETCHER.on_test_setup(item)
