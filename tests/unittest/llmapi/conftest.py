# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared MPI-pool support for eligible LLM API unit tests."""

import pytest
from test_common.grouped_test_utils import SharedMpiSessionRegistry, share_torch_llm_mpi_sessions


def pytest_collection_modifyitems(items):
    # The shared pool deliberately keeps its manager thread alive until the
    # session fixture tears down.
    for item in items:
        item.add_marker(pytest.mark.threadleak(enabled=False))


@pytest.fixture(scope="session", autouse=True)
def _shared_mpi_pools():
    registry = SharedMpiSessionRegistry()
    try:
        with share_torch_llm_mpi_sessions(registry):
            yield
    finally:
        registry.shutdown()
