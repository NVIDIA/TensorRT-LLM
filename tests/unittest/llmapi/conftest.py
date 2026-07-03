# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""llmapi-level pytest hooks wiring the opt-in session prefetcher.

See tests/test_common/session_prefetcher.py; enabled by
TRTLLM_TEST_PREFETCH_SESSION=1, no-op otherwise.
"""

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
    PREFETCHER.on_test_setup(item)
