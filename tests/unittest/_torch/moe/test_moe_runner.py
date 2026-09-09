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

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.custom_ops import torch_custom_ops
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner


def _make_runner(x_dtype: torch.dtype, weight_dtype: torch.dtype) -> MoERunner:
    runner = MoERunner.__new__(MoERunner)
    runner.x_dtype = x_dtype
    runner.weight_dtype = weight_dtype
    runner.fused_moe_runner = Mock()
    runner.fused_moe_runner.get_tactic_num.side_effect = {1: 37, 2: 71}.get
    return runner


@pytest.mark.parametrize("gemm_idx, expected", [(1, 36), (2, 70)])
def test_sm107_bf16_moe_uses_last_tactic_for_fallback(
    monkeypatch: pytest.MonkeyPatch, gemm_idx: int, expected: int
) -> None:
    monkeypatch.setattr(torch_custom_ops, "get_sm_version", lambda: 107)
    runner = _make_runner(torch.bfloat16, torch.bfloat16)

    assert runner._resolve_fallback_tactic(-1, gemm_idx) == expected


@pytest.mark.parametrize(
    "sm_version,x_dtype,weight_dtype",
    [
        (100, torch.bfloat16, torch.bfloat16),
        (107, torch.float16, torch.float16),
        (107, torch.bfloat16, torch.float8_e4m3fn),
    ],
)
def test_moe_fallback_is_unchanged_outside_sm107_bf16(
    monkeypatch: pytest.MonkeyPatch,
    sm_version: int,
    x_dtype: torch.dtype,
    weight_dtype: torch.dtype,
) -> None:
    monkeypatch.setattr(torch_custom_ops, "get_sm_version", lambda: sm_version)
    runner = _make_runner(x_dtype, weight_dtype)

    assert runner._resolve_fallback_tactic(-1, 1) == -1
    runner.fused_moe_runner.get_tactic_num.assert_not_called()


def test_sm107_bf16_moe_preserves_tuned_tactic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_custom_ops, "get_sm_version", lambda: 107)
    runner = _make_runner(torch.bfloat16, torch.bfloat16)

    assert runner._resolve_fallback_tactic(4, 1) == 4
    runner.fused_moe_runner.get_tactic_num.assert_not_called()
