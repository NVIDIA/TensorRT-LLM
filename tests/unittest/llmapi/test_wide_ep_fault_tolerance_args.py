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
"""Tests for the rank identity that a Wide-EP fault tolerance node supervisor gives each rank."""

import pytest

from tensorrt_llm import _wide_ep_fault_tolerance
from tensorrt_llm._wide_ep_fault_tolerance import FAULT_TOLERANCE_RANK_ENV

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("replacement", [False, True], ids=["original", "replacement"])
def test_rank_identity_from_node_supervisor(monkeypatch, replacement):
    """Check that the node supervisor gives the rank, the world size and the replacement flag."""
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, f"2,4,4,{int(replacement)}")
    identity = _wide_ep_fault_tolerance.wide_ep_fault_tolerance_rank()
    assert identity == (2, 4, 4, replacement)
    assert identity.env_value() == f"2,4,4,{int(replacement)}"
    monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV)
    assert _wide_ep_fault_tolerance.wide_ep_fault_tolerance_rank() is None
