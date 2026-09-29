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
"""Mock suite: a memory demand at two levels. Collected by the run under test.

Every catalogued machine has between 81 559 and 284 208 MiB, so 1 000 never
blocks and 1 000 000 always does -- a corrected figure in `core/profiles.json`
cannot break this module.
"""

import pytest


def test_unmarked():
    pass


@pytest.mark.skip_less_device_memory(1000000)
class TestHungry:
    @pytest.mark.skip_less_device_memory(1000)
    def test_modest_method(self):
        pass
