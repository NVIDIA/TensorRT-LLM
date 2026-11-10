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
"""Every GPU memory monitor test starts and ends with the monitor's
process-wide state at its defaults, whatever the previous test left."""

from collections.abc import Iterator

import pytest

from tensorrt_llm._torch.memory.monitor import (
    claims,
    providers,
    settings,
    stacks,
    startup,
    tagging,
    triggers,
)

_SETTING_VARS = (
    settings.ENV_VAR,
    settings.RANK_ENV_VAR,
    settings.STACKS_ENV_VAR,
    settings.PERIODIC_INTERVAL_VAR,
    settings.OOM_INTERVAL_VAR,
    settings.DIR_VAR,
)


def _reset_monitor() -> None:
    settings.configure(None)
    startup.clear()
    claims.clear_claims()
    tagging.set_frozen_map(tagging.TagSnapshot(supported=False))
    providers.clear_registered_providers()
    triggers.uninstall_oom_reporter()
    stacks.stop_stack_recording()


@pytest.fixture(autouse=True)
def _monitor_state_is_reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear settings, startup records, claims, tags, providers and arming.

    The environment variables are removed here; a file's own fixture that
    sets one runs after this one. Pool registrations are left alone: tests
    that depend on them clear them, and clearing before every test would
    make each GPU test classify every pool again.
    """
    for var in _SETTING_VARS:
        monkeypatch.delenv(var, raising=False)
    _reset_monitor()
    yield
    _reset_monitor()
