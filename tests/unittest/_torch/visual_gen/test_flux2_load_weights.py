# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for loading FLUX.2 feed-forward checkpoint tensors."""

import pytest
import torch

from tensorrt_llm._torch.modules.linear import Linear, WeightMode, WeightsLoadingConfig
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2 import (
    Flux2Transformer2DModel,
    _split_linear_in_weights,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_NVFP4_BLOCK_SCALE = (
    (torch.arange(512 * 16).remainder(127) + 1).reshape(512, 16).to(torch.float8_e4m3fn)
)


@pytest.mark.parametrize(
    "weight_dtype, row_scales, shared_scales",
    [
        pytest.param(torch.bfloat16, {}, {}, id="bf16"),
        pytest.param(
            torch.float8_e4m3fn,
            {},
            {"weight_scale": torch.tensor(0.25), "input_scale": torch.tensor(0.5)},
            id="fp8",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            {},
            {"weight_scale": torch.tensor([0.25]), "input_scale": torch.tensor([0.5])},
            id="fp8_scale_shape_1",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            {
                "weight_scale": torch.arange(8, dtype=torch.float32).reshape(4, 2) + 1,
                "weight_scale_inv": torch.arange(8, dtype=torch.float32).reshape(4, 2) + 2,
            },
            {"input_scale": torch.tensor(0.5)},
            id="fp8_block_scales",
        ),
        pytest.param(
            torch.uint8,
            {"weight_scale": _NVFP4_BLOCK_SCALE},
            {"weight_scale_2": torch.tensor(0.25), "input_scale": torch.tensor(0.5)},
            id="nvfp4",
        ),
        pytest.param(
            torch.uint8,
            {"weight_scale": _NVFP4_BLOCK_SCALE},
            {
                "weight_scale_2": torch.tensor(0.25),
                "input_scale": torch.tensor(0.5),
                "pre_quant_scale": torch.arange(256, dtype=torch.float32).remainder(127) + 1,
            },
            id="nvfp4_pre_quant_scale",
        ),
    ],
)
def test_split_linear_in_weights(
    weight_dtype: torch.dtype,
    row_scales: dict[str, torch.Tensor],
    shared_scales: dict[str, torch.Tensor],
) -> None:
    """Split row tensors and preserve shared scales exactly for both feed-forward modules."""
    stored_input_size = 128 if weight_dtype == torch.uint8 else 256
    row_tensors = {
        "weight": torch.arange(512 * stored_input_size)
        .remainder(251)
        .reshape(512, stored_input_size)
        .to(weight_dtype),
        "bias": torch.arange(512, dtype=torch.bfloat16),
        **row_scales,
    }
    weights = {"transformer_blocks.0.ff.down_proj.weight": torch.tensor([[3.0]])}
    expected = {key: value.clone() for key, value in weights.items()}
    for module in ("transformer_blocks.0.ff", "transformer_blocks.0.ff_context"):
        for suffix, tensor in row_tensors.items():
            weights[f"{module}.linear_in.{suffix}"] = tensor
            midpoint = tensor.shape[0] // 2
            expected[f"{module}.linear_in_gate.{suffix}"] = tensor[:midpoint].clone()
            expected[f"{module}.linear_in_up.{suffix}"] = tensor[midpoint:].clone()
        for suffix, tensor in shared_scales.items():
            weights[f"{module}.linear_in.{suffix}"] = tensor
            expected[f"{module}.linear_in_gate.{suffix}"] = tensor.clone()
            expected[f"{module}.linear_in_up.{suffix}"] = tensor.clone()

    actual = _split_linear_in_weights(weights)

    assert actual.keys() == expected.keys()
    for key, reference in expected.items():
        assert actual[key].shape == reference.shape, key
        assert actual[key].dtype == reference.dtype, key
        assert torch.equal(
            actual[key].contiguous().reshape(-1).view(torch.uint8),
            reference.contiguous().reshape(-1).view(torch.uint8),
        ), key


@pytest.mark.parametrize("tensor", [torch.tensor(0.5), torch.ones(4)], ids=["scalar", "vector"])
def test_split_linear_in_weights_rejects_unknown_suffix(tensor: torch.Tensor) -> None:
    """Reject unknown checkpoint entries instead of guessing their split axis."""
    with pytest.raises(ValueError, match="Unsupported linear_in checkpoint tensor:.*future_scale"):
        _split_linear_in_weights({"transformer_blocks.0.ff.linear_in.future_scale": tensor})


def test_load_weights_static_fp8_linear_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load a static FP8 checkpoint through FLUX.2 into fused gate/up weights and scales."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = DiffusionModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
        dynamic_weight_quant=False,
        device="cpu",
    )
    gate_up_proj = Linear(
        8,
        16,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=config.quant_config,
        reduce_output=False,
        weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_GATE_UP_LINEAR),
        fused_weight_shard_indices_mapping={"gate": (0, 8), "up": (8, 8)},
    )
    load_shard = gate_up_proj.load_shard
    loaded_shards = {}

    def load_shard_on_cpu(weights, tensor_name, **kwargs):
        """Keep the production shard loader on CPU despite its caller's CUDA device."""
        loaded_shards[kwargs["name"]] = weights
        kwargs["device"] = torch.device("cpu")
        return load_shard(weights, tensor_name, **kwargs)

    monkeypatch.setattr(gate_up_proj, "load_shard", load_shard_on_cpu)

    # Only the feed-forward projection is needed to exercise the model's weight loader.
    model = Flux2Transformer2DModel.__new__(Flux2Transformer2DModel)
    torch.nn.Module.__init__(model)
    model.model_config = config
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"ff": torch.nn.ModuleDict({"gate_up_proj": gate_up_proj})})]
    )
    weight = torch.arange(128).remainder(31).reshape(16, 8).to(torch.float8_e4m3fn)
    expected_weight = weight.view(torch.uint8).clone()
    input_scale = torch.tensor(0.5)
    weight_scale = torch.tensor(0.25)

    model.load_weights(
        {
            "transformer_blocks.0.ff.linear_in.weight": weight,
            "transformer_blocks.0.ff.linear_in.input_scale": input_scale,
            "transformer_blocks.0.ff.linear_in.weight_scale": weight_scale,
        }
    )

    assert gate_up_proj.weight.dtype == torch.float8_e4m3fn
    assert torch.equal(gate_up_proj.weight[:8].view(torch.uint8), expected_weight[:8])
    assert torch.equal(gate_up_proj.weight[8:].view(torch.uint8), expected_weight[8:])
    assert loaded_shards.keys() == {"gate", "up"}
    for shard in loaded_shards.values():
        torch.testing.assert_close(shard["input_scale"], input_scale, rtol=0, atol=0)
        torch.testing.assert_close(shard["weight_scale"], weight_scale, rtol=0, atol=0)
    torch.testing.assert_close(gate_up_proj.input_scale, input_scale, rtol=0, atol=0)
    torch.testing.assert_close(gate_up_proj.weight_scale, weight_scale, rtol=0, atol=0)
