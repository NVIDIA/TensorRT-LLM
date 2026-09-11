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
"""Tests for the public telemetry architecture allowlist."""

import pytest

from tensorrt_llm.usage.architecture_allowlist import PUBLIC_HF_ARCHITECTURES

pytestmark = pytest.mark.cpu_only


def test_allowlist_entries_are_well_formed() -> None:
    """Runtime entries are valid schema-bounded identifiers."""
    assert PUBLIC_HF_ARCHITECTURES
    assert all(
        name.isidentifier() and name.strip() == name and len(name) <= 256
        for name in PUBLIC_HF_ARCHITECTURES
    )
