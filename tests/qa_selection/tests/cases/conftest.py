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
"""Mock of `tests/integration/defs/conftest.py`: the skip decorators, built the same way.

Staged beside a mock module and loaded by the run under test, never by this suite.

Each decorator is a module-level `pytest.mark.skipif(<probe>, reason=...)`, imported
by name into a test module, as the real ones are. Each reason is copied byte for
byte from the real decorator and is never read from `core/rules.json` -- editing
one there is meant to break these criteria.

Each condition probes the collecting host, frozen at import. `get_sm_version()`
answers as the real one does without a GPU, so `skip_pre_blackwell`'s condition is
True here: a target that keeps its test shows the condition was never read.
"""

import platform
import time

import pytest


def get_sm_version():
    """What the real probe returns when `torch.cuda.is_available()` is False."""
    return 0


skip_pre_blackwell = pytest.mark.skipif(
    get_sm_version() < 100,
    reason="This test is not supported in pre-Blackwell architecture",
)

skip_post_blackwell = pytest.mark.skipif(
    get_sm_version() >= 100,
    reason="This test is not supported in post-Blackwell architecture",
)

skip_arm = pytest.mark.skipif(
    "aarch64" in platform.machine(),
    reason="This test is not supported on ARM architecture",
)

# No rule in `core/rules.json` matches this reason, deliberately.
skip_flaky_on_tuesdays = pytest.mark.skipif(
    time.localtime().tm_wday == 1,
    reason="This test is flaky on Tuesdays",
)
