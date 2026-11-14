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
"""Qwen3.5's MoE block asks the routed experts' backend for the shared expert's SMs per forward, through
MoEImplBase.reserve_sms_for_concurrent_kernels."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextSparseMoeBlock
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE

CEILING = 8
# 64 requests of an MTP 6 decode step (7 rows each): _large_step_rows.
RESERVE_ROWS = 64 * 7


class _Backend:
    def __init__(self) -> None:
        self.asked: list[int] = []

    def sm_reserve_ceiling(self) -> int:
        return CEILING

    def reserve_sms_for_concurrent_kernels(self, num_sms: int) -> int:
        self.asked.append(num_sms)
        return num_sms


def _block(reserve_rows: int, aux_stream: object = object()) -> SimpleNamespace:
    experts = ConfigurableMoE.__new__(ConfigurableMoE)
    object.__setattr__(experts, "backend", _Backend())
    return SimpleNamespace(
        experts=experts, aux_stream=aux_stream, _shared_expert_reserve_rows=reserve_rows
    )


@pytest.mark.parametrize(
    "multi_stream,rows,expect",
    [
        (True, RESERVE_ROWS, CEILING),
        (True, RESERVE_ROWS + 70, CEILING),
        (True, RESERVE_ROWS - 1, 0),
        (False, RESERVE_ROWS, 0),
    ],
)
def test_shared_expert_sms_are_asked_per_forward(multi_stream, rows, expect) -> None:
    """The backend's ceiling where the shared expert runs beside the routed experts (multi-stream) on a step of at
    least _shared_expert_reserve_rows rows; 0 for smaller or in-order steps, so the grid keeps every SM."""
    block = _block(RESERVE_ROWS)
    with with_multi_stream(multi_stream):
        Qwen3NextSparseMoeBlock._reserve_sms_for_shared_expert(block, rows)
    assert block.experts.backend.asked == [expect]


def test_no_reserve_without_the_declared_rows_or_an_aux_stream() -> None:
    """Without the decode kernels' rows (no MTP, or another model) nothing is asked; without an aux stream the shared
    expert runs in order, so the grid keeps every SM."""
    block = _block(0)
    with with_multi_stream(True):
        Qwen3NextSparseMoeBlock._reserve_sms_for_shared_expert(block, RESERVE_ROWS)
    assert block.experts.backend.asked == []
    block = _block(RESERVE_ROWS, aux_stream=None)
    with with_multi_stream(True):
        Qwen3NextSparseMoeBlock._reserve_sms_for_shared_expert(block, RESERVE_ROWS)
    assert block.experts.backend.asked == [0]
