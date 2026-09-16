# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Callable
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.modules import gated_mlp as gated_mlp_module
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP


def _make_gate_up_proj(
    projected: torch.Tensor,
    *,
    partitioned: bool,
) -> nn.Module:
    gate_up_proj = nn.Module()
    gate_up_proj.has_nvfp4 = True
    gate_up_proj.has_bias = False
    gate_up_proj.partition_plan = SimpleNamespace(enabled=partitioned)
    gate_up_proj.can_use_cute_dsl_nvfp4_swiglu_blackwell = Mock(return_value=not partitioned)
    gate_up_proj.forward = Mock(return_value=projected)
    return gate_up_proj


def _make_down_proj() -> nn.Module:
    down_proj = nn.Module()
    down_proj.has_fp8_qdq = False
    down_proj.has_w4a8_nvfp4_fp8 = False
    # The unfused branch calls ``_can_fuse_swiglu_fp8_quant``, whose first
    # check reads this attribute, so a stand-in without it raises
    # AttributeError before the assertions below are reached.
    down_proj.has_fp8_block_scales = False
    down_proj.forward = Mock(side_effect=lambda value, **kwargs: value + 1)
    return down_proj


def test_gate_up_partition_falls_back_to_swiglu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mlp = GatedMLP(hidden_size=2, intermediate_size=2, bias=False)
    mlp.use_cute_dsl_blockscaling_mm = True
    projected = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mlp.gate_up_proj = _make_gate_up_proj(projected, partitioned=True)
    mlp.down_proj = _make_down_proj()

    gate, up = projected.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    swiglu = Mock(side_effect=lambda value, **kwargs: F.silu(value[..., :2]) * value[..., 2:])
    monkeypatch.setattr(gated_mlp_module, "swiglu", swiglu)
    fused_gate_up_swiglu = Mock(side_effect=AssertionError("Blackwell fused op must not run"))
    monkeypatch.setattr(mlp, "_fused_gate_up_swiglu", fused_gate_up_swiglu)

    inputs = torch.tensor([[5.0, 6.0]])
    output = mlp(inputs)

    assert not mlp._can_fuse_gate_up_swiglu()
    mlp.gate_up_proj.forward.assert_called_once_with(inputs)
    # ``GatedMLP`` forwards all three SwiGLU shape parameters on every call;
    # they are None here because this layer is plain SwiGLU.
    swiglu.assert_called_once_with(
        projected, swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None
    )
    mlp.down_proj.forward.assert_called_once()
    down_args, down_kwargs = mlp.down_proj.forward.call_args
    torch.testing.assert_close(down_args[0], activated)
    assert down_kwargs == {"all_reduce_params": None, "layer_idx": None}
    fused_gate_up_swiglu.assert_not_called()
    torch.testing.assert_close(output, activated + 1)


@pytest.mark.parametrize(
    "swiglu_limit, expected",
    [(None, True), (float("inf"), True), (7.0, False)],
)
def test_swiglu_limit_controls_gate_up_fusion_capability(
    swiglu_limit: float | None,
    expected: bool,
) -> None:
    mlp = GatedMLP(
        hidden_size=2,
        intermediate_size=2,
        bias=False,
        swiglu_limit=swiglu_limit,
        use_cute_dsl_blockscaling_mm=True,
    )

    assert mlp.gate_up_proj.use_cute_dsl_nvfp4_swiglu_blackwell is expected


@pytest.mark.parametrize(
    ("activation", "expected"),
    [
        pytest.param(F.silu, True, id="plain-swiglu"),
        pytest.param(lambda value: value, False, id="custom-swiglu-oai"),
    ],
)
def test_activation_controls_fp8_quant_fusion_capability(
    monkeypatch: pytest.MonkeyPatch,
    activation: Callable[[torch.Tensor], torch.Tensor],
    expected: bool,
) -> None:
    mlp = GatedMLP(
        hidden_size=2,
        intermediate_size=2,
        bias=False,
        activation=activation,
    )
    down_proj = nn.Module()
    down_proj.has_fp8_block_scales = True
    down_proj.use_cute_dsl_blockscaling_mm = True
    down_proj.disable_deep_gemm = False
    mlp.down_proj = down_proj
    monkeypatch.setattr(gated_mlp_module, "get_sm_version", lambda: 107)
    monkeypatch.setattr(gated_mlp_module, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", True)

    assert mlp._can_fuse_swiglu_fp8_quant() is expected
