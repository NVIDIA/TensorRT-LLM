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
"""Mock suite: both GPU markers on one test, agreeing at 8.

Modelled on `tests/integration/defs/examples/test_deepseek_v4_pro.py:138,140`,
whose `skip_less_device_memory` is left off -- it decides feasibility, not
routing. Collected by the run under test, never run here.
"""

import pytest


def test_unmarked():
    pass


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_mpi_world_size(8)
def test_needs_eight_of_both():
    pass
