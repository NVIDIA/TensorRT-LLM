# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest plugin wiring for the session prefetcher (repo-wide, demand-driven).

Loaded via ``pytest_plugins`` in each test tree's top-level conftest. Factory
installation is LAZY: nothing is patched until the test suite itself imports
tensorrt_llm's executor modules (which only happens for tests that create MPI
pools), so suites that never create pools pay nothing — not even the
tensorrt_llm import.
"""

from test_common.session_prefetcher import PREFETCHER


def pytest_configure(config):
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
