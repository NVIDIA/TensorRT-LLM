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

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import (
    HfWeightMapper,
    convert_compressed_tensors_nvfp4_weights,
)
from tensorrt_llm._torch.models.modeling_utils import duplicate_kv_weight


@pytest.mark.parametrize(
    "duplicate_kv",
    [HfWeightMapper()._duplicate_kv, duplicate_kv_weight],
    ids=["hf_weight_mapper", "legacy_modeling_utils"],
)
def test_duplicate_kv_bias_preserves_head_boundaries(duplicate_kv) -> None:
    bias = torch.tensor([10, 11, 20, 21])

    duplicated_bias = duplicate_kv(bias, num_kv_heads=2, tensor_parallel_size=4)

    expected_bias = torch.tensor([10, 11, 10, 11, 20, 21, 20, 21])
    torch.testing.assert_close(duplicated_bias, expected_bias)


class _LazySlice:
    """Stands in for a lazily read safetensors slice, which ``[...]`` loads."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def __getitem__(self, index):
        return self._tensor[index]


def test_compressed_tensors_nvfp4_names_and_global_scales_are_converted() -> None:
    prefix = "model.layers.0.mlp.experts.0.w1"
    packed = torch.tensor([[1, 2]], dtype=torch.uint8)
    block_scale = torch.ones(1, 1).to(torch.float8_e4m3fn)
    norm = torch.tensor([3.0])
    weights = {
        f"{prefix}.weight_packed": packed,
        f"{prefix}.weight_scale": block_scale,
        # One entry is 0, as stored for an expert that saw no calibration data.
        f"{prefix}.weight_global_scale": torch.tensor([4.0, 0.0]),
        f"{prefix}.input_global_scale": _LazySlice(torch.tensor([8.0])),
        "model.norm.weight": norm,
    }

    converted = convert_compressed_tensors_nvfp4_weights(weights)

    assert set(converted.keys()) == {
        f"{prefix}.weight",
        f"{prefix}.weight_scale",
        f"{prefix}.weight_scale_2",
        f"{prefix}.input_scale",
        "model.norm.weight",
    }
    assert converted[f"{prefix}.weight"] is packed
    assert converted[f"{prefix}.weight_scale"] is block_scale
    assert converted["model.norm.weight"] is norm
    torch.testing.assert_close(converted[f"{prefix}.weight_scale_2"], torch.tensor([0.25, 0.0]))
    torch.testing.assert_close(converted[f"{prefix}.input_scale"], torch.tensor([0.125]))


def test_compressed_tensors_nvfp4_uninvertible_global_scales_log_one_warning(monkeypatch) -> None:
    import tensorrt_llm._torch.models.checkpoints.hf.weight_mapper as weight_mapper

    log = Mock()
    monkeypatch.setattr(weight_mapper, "logger", log)
    weights = {
        "model.layers.0.mlp.experts.0.w1.weight_global_scale": torch.tensor([4.0]),
        "model.layers.0.mlp.experts.1.w1.weight_global_scale": torch.tensor([0.0]),
        "model.layers.0.mlp.experts.2.w1.weight_global_scale": torch.tensor([-1.0]),
        "model.layers.0.mlp.experts.3.w1.input_global_scale": _LazySlice(
            torch.tensor([float("nan")])
        ),
    }

    converted = convert_compressed_tensors_nvfp4_weights(weights)

    for i in (1, 2):
        torch.testing.assert_close(
            converted[f"model.layers.0.mlp.experts.{i}.w1.weight_scale_2"], torch.tensor([0.0])
        )
    torch.testing.assert_close(
        converted["model.layers.0.mlp.experts.3.w1.input_scale"], torch.tensor([0.0])
    )
    assert log.warning.call_count == 1
    message = log.warning.call_args[0][0]
    assert "2 with negative or non-finite values" in message
    assert "experts.2.w1.weight_global_scale" in message
    assert "experts.3.w1.input_global_scale" in message
    assert "1 with 0" in message
    assert "experts.1.w1.weight_global_scale" in message

    log.reset_mock()
    convert_compressed_tensors_nvfp4_weights(
        {"model.layers.0.mlp.experts.0.w1.weight_global_scale": torch.tensor([4.0])}
    )
    log.warning.assert_not_called()


def test_compressed_tensors_nvfp4_conversion_returns_other_checkpoints_unchanged() -> None:
    prefix = "model.layers.0.mlp.experts.0.w1"
    weights = {
        f"{prefix}.weight": torch.tensor([[1, 2]], dtype=torch.uint8),
        f"{prefix}.weight_scale_2": torch.tensor([0.25]),
        f"{prefix}.input_scale": torch.tensor([0.125]),
    }

    assert convert_compressed_tensors_nvfp4_weights(weights) is weights


@pytest.mark.parametrize(
    "names",
    [
        ("model.layers.0.self_attn.o_proj.weight_packed", "model.layers.0.self_attn.o_proj.weight"),
        (
            "model.layers.0.mlp.experts.0.w1.weight_global_scale",
            "model.layers.0.mlp.experts.0.w1.weight_scale_2",
        ),
    ],
    ids=["weight", "weight_scale_2"],
)
def test_compressed_tensors_nvfp4_conversion_rejects_colliding_names(names) -> None:
    weights = {name: torch.tensor([4.0]) for name in names}

    with pytest.raises(ValueError, match="maps both"):
        convert_compressed_tensors_nvfp4_weights(weights)


def test_compressed_tensors_nvfp4_conversion_keeps_consumable_weights() -> None:
    packed = torch.tensor([[1, 2]], dtype=torch.uint8)
    source = ConsumableWeightsDict({"model.layers.0.self_attn.o_proj.weight_packed": packed})

    converted = convert_compressed_tensors_nvfp4_weights(source)

    assert isinstance(converted, ConsumableWeightsDict)
    assert converted["model.layers.0.self_attn.o_proj.weight"] is packed
    # The converted dict owns the tensors, so consuming it can release them.
    assert len(source) == 0
