# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest plugin wiring for the session prefetcher (repo-wide, demand-driven).

Loaded via ``pytest_plugins`` in each test tree's top-level conftest. Factory
installation is LAZY: nothing is patched until the test suite itself imports
tensorrt_llm's executor modules (which only happens for tests that create MPI
pools), so suites that never create pools pay nothing — not even the
tensorrt_llm import.
"""

import os

from test_common.session_prefetcher import PREFETCHER


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "prefetch_model_dir(path): model dir the test loads, for page-cache warming",
    )


def pytest_runtest_setup(item):
    # Last-line fail-open: prefetch is an optimization wired into EVERY test's
    # setup, so an unexpected error here must degrade to baseline speed via
    # the kill switch — never error the suite.
    try:
        PREFETCHER.install_pool_factory_if_loaded()
        PREFETCHER.on_test_setup(item)
    except Exception as e:
        os.environ["TRTLLM_TEST_PREFETCH_SESSION"] = "0"
        print(f"[session-prefetch] disabled by unexpected error: {e}", flush=True)


def pytest_sessionfinish(session, exitstatus):
    try:
        PREFETCHER.dispose()
    except Exception as e:  # never fail the session over cleanup
        print(f"[session-prefetch] dispose failed: {e}", flush=True)
