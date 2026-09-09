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
"""CI-routed CPU coverage for Ulysses attention with replicated K/V."""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest

from .test_ulysses_attention import (
    MODULES_AVAILABLE,
    _logic_ulysses_replicated_kv_unequal_lengths,
    run_test_in_distributed,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Required modules not available")
def test_ulysses_replicated_kv_unequal_lengths():
    """Unequal replicated context and generated padding match unpadded SDPA."""
    run_test_in_distributed(
        world_size=2,
        test_fn=_logic_ulysses_replicated_kv_unequal_lengths,
        use_cuda=False,
    )
