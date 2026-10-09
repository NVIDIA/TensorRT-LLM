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
import importlib
from importlib.util import find_spec

import pytest


def _poison_api():
    """(poison_reason, take_poison, num_live_managers), or None if the backend has no latch.

    Only the C++ backend has one. The package is reachable under two different names depending
    on whether it was imported top-level or through tensorrt_llm.
    """
    name = (
        "kv_cache_manager_v2"
        if find_spec("kv_cache_manager_v2") is not None
        else "tensorrt_llm.runtime.kv_cache_manager_v2"
    )
    module = importlib.import_module(name)
    if not hasattr(module, "take_poison"):
        return None
    return module.poison_reason, module.take_poison, module.num_live_managers


@pytest.fixture(autouse=True)
def check_kvcm2_not_poisoned():
    """Fails a test that leaves KVCM2 poisoned, and blames the right test when it does.

    A violation detected inside a destructor is not attached to any API call, so without this it
    would only reach the log. Checking on the way in as well means a latch that could not be
    cleared -- because a manager from the previous test is still alive -- is reported against the
    test that caused it rather than the next one to run.
    """
    api = _poison_api()
    if api is None:
        yield
        return
    poison_reason, take_poison, num_live_managers = api

    stale = poison_reason()
    if stale is not None:
        pytest.fail(
            f"KVCM2 was already poisoned before this test started, by an earlier test: {stale}"
        )

    yield

    reason = take_poison()
    if reason is not None:
        live = num_live_managers()
        detail = (
            ""
            if live == 0
            else (
                f" A manager is still alive ({live}), so the latch could not be cleared and the tests"
                " after this one will not run."
            )
        )
        pytest.fail(f"KVCM2 was poisoned during this test: {reason}{detail}")
