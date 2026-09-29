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
"""Mock suite: one skip the rule table encodes, one it declines to.

Both are needed: without the curated skip, `deselected_by_reason` would be
omitted from the record entirely.
"""

from conftest import skip_flaky_on_tuesdays, skip_pre_blackwell


def test_unmarked():
    pass


@skip_flaky_on_tuesdays
def test_skipped_for_an_uncurated_reason():
    pass


@skip_pre_blackwell
def test_skipped_by_the_table():
    pass
