# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared pytest fixtures for host_perf module-level tests.

Provides a session-scoped fixture that uploads accumulated module-level
performance results (scheduler, sampler, kv_cache) to OpenSearch at the
end of the test session.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def module_perf_db_finalizer():
    """Upload accumulated module perf results to OpenSearch after all tests."""
    yield
    from .regression_helper import get_collected_results, post_module_perf_to_db

    if get_collected_results():
        try:
            post_module_perf_to_db()
        except Exception as e:
            from defs.trt_test_alternative import print_info

            print_info(f"[module_perf_db] Failed to upload to OpenSearch: {e}")
