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
"""Mock suite: how marks are read. Collected by the run under test, never run here.

- A GPU bound at two levels: the class asks for more than any machine here
  has and the method for less, so a run that keeps the method read the
  closest marker only.
- A memory demand at two levels: every catalogued machine has between 81 559
  and 284 208 MiB, so 1 000 never blocks and 1 000 000 always does.
- Both GPU markers on one test, agreeing at 8 -- the shape at
  `tests/integration/defs/examples/test_deepseek_v4_pro.py:138,140` -- and
  disagreeing, the smaller listed first, a shape no collected item has.
- A skip the rule table encodes, and one it declines to. Without the first,
  `deselected_by_reason` would be omitted from the record entirely.
- A skip on a class, and marks on one `pytest.param` each -- the two other
  places the integration suite puts them.
"""

import pytest
from conftest import skip_flaky_on_tuesdays, skip_pre_blackwell


def test_unmarked():
    pass


@pytest.mark.skip_less_device(8)
class TestClosestGpus:
    @pytest.mark.skip_less_device(2)
    def test_method_bound(self):
        pass


@pytest.mark.skip_less_device_memory(1000000)
class TestEveryLevelMemory:
    @pytest.mark.skip_less_device_memory(1000)
    def test_method_bound(self):
        pass


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_mpi_world_size(8)
def test_agreeing_bounds():
    pass


@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_mpi_world_size(8)
def test_disagreeing_bounds():
    pass


@skip_pre_blackwell
def test_curated_skip():
    pass


@skip_flaky_on_tuesdays
def test_uncurated_skip():
    pass


@skip_pre_blackwell
class TestClassSkip:
    def test_method(self):
        pass


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("skip", marks=skip_pre_blackwell, id="skip"),
        pytest.param("eight_gpus", marks=pytest.mark.skip_less_device(8), id="eight_gpus"),
    ],
)
def test_param(case):
    pass
