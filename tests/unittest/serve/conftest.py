# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest


@pytest.fixture
def serve_env(tmp_path):
    """A :class:`test_common.serve.app_factory.ServeTestEnv` per test."""
    # Imported here, not at module level: this conftest also governs checks
    # that need no tensorrt_llm build (the ratchet, the agent-docs test).
    from test_common.serve.app_factory import isolated_serve_env

    with isolated_serve_env(tmp_path) as env:
        yield env
