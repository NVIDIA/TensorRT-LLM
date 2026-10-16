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
"""Mock suite: the architecture gates. Collected by the run under test, never run here.

Each test name says what its decorator keeps: `skip_post_blackwell` is encoded
`sm >= 100`, so `test_hopper_and_older` survives on Hopper and older.
`test_unmarked` is the control every mock module carries: it states no
requirement, so it survives on every machine.
"""

from conftest import skip_arm, skip_post_blackwell, skip_pre_blackwell


def test_unmarked():
    pass


@skip_pre_blackwell
def test_blackwell_and_newer():
    pass


@skip_post_blackwell
def test_hopper_and_older():
    pass


@skip_arm
def test_x86_only():
    pass
