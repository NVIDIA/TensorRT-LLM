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
"""Mock suite: a GPU bound at two levels. Collected by the run under test.

The class asks for more than any machine here has and the method asks for
less, so a run that keeps the method can only have read the closest marker.
"""

import pytest


def test_unmarked():
    pass


@pytest.mark.skip_less_device(8)
class TestEightGpus:
    @pytest.mark.skip_less_device(2)
    def test_needs_two(self):
        pass
