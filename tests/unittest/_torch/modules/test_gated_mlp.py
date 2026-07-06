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

from collections.abc import Callable

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.modules.gated_mlp import GatedMLP


class _RecordingProjection(nn.Module):
    def __init__(self, has_any_quant: bool = False) -> None:
        super().__init__()
        self.has_any_quant = has_any_quant
        self.last_input_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input_shape = tuple(x.shape)
        return torch.cat((x, x), dim=-1)


class _CompiledProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.has_any_quant = False
        self.linear = nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _GateUpProjectionWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _make_test_mlp()
        self.mlp.gate_up_proj = _CompiledProjection()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp._run_gate_up_proj(x)


def _make_test_mlp(*, pad_rows: int = 16, has_any_quant: bool = False) -> GatedMLP:
    mlp = GatedMLP.__new__(GatedMLP)
    nn.Module.__init__(mlp)
    mlp.small_m_fc1_pad_rows = pad_rows
    mlp.gate_up_proj = _RecordingProjection(has_any_quant=has_any_quant)
    return mlp


@pytest.mark.parametrize(
    ("num_rows", "expected_projection_rows"),
    [(1, 1), (8, 8), (9, 16), (12, 16), (15, 16), (16, 16), (17, 17)],
)
def test_small_m_bf16_fc1_padding(num_rows: int, expected_projection_rows: int) -> None:
    mlp = _make_test_mlp()
    x = torch.randn(num_rows, 32, dtype=torch.bfloat16)

    output = mlp._run_gate_up_proj(x)

    assert mlp.gate_up_proj.last_input_shape == (expected_projection_rows, 32)
    assert output.shape == (num_rows, 64)
    torch.testing.assert_close(output, torch.cat((x, x), dim=-1))


@pytest.mark.parametrize(
    ("dtype", "has_any_quant"),
    [(torch.float16, False), (torch.bfloat16, True)],
)
def test_small_m_fc1_padding_skips_other_paths(dtype: torch.dtype, has_any_quant: bool) -> None:
    mlp = _make_test_mlp(has_any_quant=has_any_quant)
    x = torch.randn(12, 32, dtype=dtype)

    output = mlp._run_gate_up_proj(x)

    assert mlp.gate_up_proj.last_input_shape == (12, 32)
    assert output.shape == (12, 64)


@pytest.mark.parametrize("pad_rows", [-1, 8, 15, 17])
def test_small_m_fc1_padding_rejects_invalid_configuration(pad_rows: int) -> None:
    with pytest.raises(ValueError, match="small_m_fc1_pad_rows must be 0 or 16"):
        GatedMLP(hidden_size=32, intermediate_size=64, bias=False, small_m_fc1_pad_rows=pad_rows)


def test_small_m_fc1_padding_uses_one_dynamic_compiled_graph() -> None:
    compile_count = 0

    def backend(
        graph: torch.fx.GraphModule, _example_inputs: list[torch.Tensor]
    ) -> Callable[..., torch.Tensor]:
        nonlocal compile_count
        compile_count += 1
        return graph.forward

    compiled = torch.compile(
        _GateUpProjectionWrapper(), backend=backend, dynamic=True, fullgraph=True
    )
    for num_rows in (8, 9, 12, 15, 16, 17):
        x = torch.randn(num_rows, 32, dtype=torch.bfloat16)
        output = compiled(x)
        assert output.shape == (num_rows, 64)

    assert compile_count == 1
