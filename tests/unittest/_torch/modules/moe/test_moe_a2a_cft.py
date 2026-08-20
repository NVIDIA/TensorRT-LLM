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

import pytest

from tensorrt_llm._torch.distributed.moe_alltoall import get_force_cft as get_force_cft_standalone
from tensorrt_llm._torch.distributed.moe_alltoall import should_use_cft as should_use_cft_standalone
from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_one_sided import (
    FORCE_CFT_ENV,
    get_force_cft,
    should_use_cft,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("2", None),
        ("true", None),
        (" 1 ", None),
        ("0", False),
        ("1", True),
    ],
)
def test_get_force_cft(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool | None):
    if value is None:
        monkeypatch.delenv(FORCE_CFT_ENV, raising=False)
    else:
        monkeypatch.setenv(FORCE_CFT_ENV, value)

    assert get_force_cft() is expected
    assert get_force_cft_standalone() is expected


@pytest.mark.parametrize(
    ("can_use_cft", "force_cft", "runtime_max_tokens_per_rank", "expected"),
    [
        (True, None, 128, True),
        (True, None, 129, False),
        (True, False, 1, False),
        (True, True, 129, True),
        (False, True, 1, False),
        (False, None, 1, False),
    ],
)
def test_should_use_cft(
    can_use_cft: bool,
    force_cft: bool | None,
    runtime_max_tokens_per_rank: int,
    expected: bool,
):
    assert should_use_cft(can_use_cft, force_cft, 128, runtime_max_tokens_per_rank) is expected
    assert (
        should_use_cft_standalone(can_use_cft, force_cft, 128, runtime_max_tokens_per_rank)
        is expected
    )
