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

import math
from collections.abc import Sequence

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.models.quant_checkpoint_layout import (
    PackedMxfp4ExpertsCheckpointLayout,
)

HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 32
LAYER = 3


def _layout() -> PackedMxfp4ExpertsCheckpointLayout:
    return PackedMxfp4ExpertsCheckpointLayout(
        expert_key_pattern=(
            r"layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
            r"(?P<projection>w[123])\.(?P<tensor_kind>weight|scale)"
        ),
        runtime_gate_up_order=("w3", "w1"),
        runtime_down_projection="w2",
        expert_block_size=32,
    )


def _key(layer: int, expert: int, weight_name: str, tensor_kind: str) -> str:
    return f"layers.{layer}.ffn.experts.{expert}.{weight_name}.{tensor_kind}"


def _raw_bytes(shape: Sequence[int], offset: int) -> torch.Tensor:
    values = (torch.arange(math.prod(shape), dtype=torch.int64) + offset) % 256
    return values.to(torch.uint8).reshape(tuple(shape))


def _raw_uint8(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8 and tensor.is_contiguous():
        return tensor
    return tensor.contiguous().view(torch.uint8)


def _scale_tensor(raw: torch.Tensor, scale_dtype: torch.dtype) -> torch.Tensor:
    if scale_dtype == torch.uint8:
        return raw
    return raw.view(scale_dtype)


def _state_dict(
    *,
    experts: Sequence[int],
    layer: int = LAYER,
    hidden_size: int = HIDDEN_SIZE,
    intermediate_size: int = INTERMEDIATE_SIZE,
    int8_weights: bool = False,
    scale_dtype: torch.dtype = torch.uint8,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    weight_offsets = {"w1": 17, "w2": 89, "w3": 151}
    scale_offsets = {"w1": 23, "w2": 101, "w3": 179}

    for expert in experts:
        for weight_name in ("w1", "w2", "w3"):
            if weight_name in ("w1", "w3"):
                weight_shape = (intermediate_size, hidden_size // 2)
                scale_shape = (intermediate_size, hidden_size // 32)
            else:
                weight_shape = (hidden_size, intermediate_size // 2)
                scale_shape = (hidden_size, intermediate_size // 32)

            weight_raw = _raw_bytes(weight_shape, weight_offsets[weight_name] + 37 * expert)
            scale_raw = _raw_bytes(scale_shape, scale_offsets[weight_name] + 41 * expert)
            if int8_weights:
                weight_raw = weight_raw.view(torch.int8)
            state[_key(layer, expert, weight_name, "weight")] = weight_raw
            state[_key(layer, expert, weight_name, "scale")] = _scale_tensor(scale_raw, scale_dtype)

    state["layers.0.ffn.shared_experts.w1.weight"] = torch.empty(0, dtype=torch.uint8)
    state["layers.0.ffn.experts.0.w4.weight"] = torch.empty(0, dtype=torch.uint8)
    return state


def _metadata_from_state(state: dict[str, torch.Tensor]) -> dict[str, dict[str, object]]:
    layout = _layout()
    metadata = {}
    for name, tensor in state.items():
        parsed = layout.parse_key(name)
        if parsed is None:
            continue
        dtype = "I8" if parsed.tensor_kind == "weight" else "F8_E8M0"
        metadata[name] = {"dtype": dtype, "shape": list(tensor.shape)}
    return metadata


def _packed_weight(state: dict[str, torch.Tensor], expert: int, weight_name: str) -> torch.Tensor:
    raw = _raw_uint8(state[_key(LAYER, expert, weight_name, "weight")])
    if weight_name in ("w1", "w3"):
        return raw.view(INTERMEDIATE_SIZE, HIDDEN_SIZE // 32, 16)
    return raw.view(HIDDEN_SIZE, INTERMEDIATE_SIZE // 32, 16)


def _scale(state: dict[str, torch.Tensor], expert: int, weight_name: str) -> torch.Tensor:
    return _raw_uint8(state[_key(LAYER, expert, weight_name, "scale")])


def _unpack_nibbles(packed_bytes: torch.Tensor) -> torch.Tensor:
    raw = _raw_uint8(packed_bytes)
    low_nibbles = torch.bitwise_and(raw, 0x0F)
    high_nibbles = torch.bitwise_right_shift(raw, 4)
    return torch.stack((low_nibbles, high_nibbles), dim=-1).reshape(*raw.shape[:-1], -1)


def _parsed_values(parsed: object) -> tuple[int, int, str, str]:
    return (
        getattr(parsed, "layer"),
        getattr(parsed, "expert"),
        getattr(parsed, "projection"),
        getattr(parsed, "tensor_kind"),
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("layers.0.ffn.experts.1.w1.weight", (0, 1, "w1", "weight")),
        ("layers.12.ffn.experts.34.w2.scale", (12, 34, "w2", "scale")),
        ("layers.5.ffn.experts.8.w3.weight", (5, 8, "w3", "weight")),
    ),
)
def test_layout_parser_accepts_deepseek_style_routed_expert_keys(
    name: str,
    expected: tuple[int, int, str, str],
) -> None:
    parsed = _layout().parse_key(name)

    assert parsed is not None
    assert _parsed_values(parsed) == expected


@pytest.mark.parametrize(
    "name",
    (
        "model.layers.0.ffn.experts.1.w1.weight",
        "layers.0.ffn.shared_experts.1.w1.weight",
        "layers.0.ffn.experts.1.w4.weight",
        "layers.0.ffn.experts.1.w1.bias",
        "layers.0.ffn.experts.-1.w1.weight",
        "layers.0.ffn.experts.1.w1.weight.extra",
    ),
)
def test_layout_parser_rejects_noncanonical_keys(name: str) -> None:
    assert _layout().parse_key(name) is None


@pytest.mark.parametrize("int8_weights", (False, True))
def test_layout_stacks_blocks_and_scales_in_runtime_order_and_preserves_raw_nibbles(
    int8_weights: bool,
) -> None:
    state = _state_dict(experts=(0, 1), int8_weights=int8_weights)

    packed = _layout().pack_experts(
        state,
        layer=LAYER,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_experts=2,
    )

    assert packed.expert_indices == (0, 1)
    assert packed.gate_up_blocks.shape == (2, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 32, 16)
    assert packed.gate_up_scales.shape == (2, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 32)
    assert packed.down_blocks.shape == (2, HIDDEN_SIZE, INTERMEDIATE_SIZE // 32, 16)
    assert packed.down_scales.shape == (2, HIDDEN_SIZE, INTERMEDIATE_SIZE // 32)
    assert packed.gate_up_blocks.dtype == torch.uint8
    assert packed.gate_up_scales.dtype == torch.uint8
    assert packed.down_blocks.dtype == torch.uint8
    assert packed.down_scales.dtype == torch.uint8

    for expert_pos, expert in enumerate((0, 1)):
        expected_w3 = _packed_weight(state, expert, "w3")
        expected_w1 = _packed_weight(state, expert, "w1")
        expected_w2 = _packed_weight(state, expert, "w2")

        assert torch.equal(packed.gate_up_blocks[expert_pos, :INTERMEDIATE_SIZE], expected_w3)
        assert torch.equal(packed.gate_up_blocks[expert_pos, INTERMEDIATE_SIZE:], expected_w1)
        assert torch.equal(packed.down_blocks[expert_pos], expected_w2)
        assert torch.equal(
            packed.gate_up_scales[expert_pos, :INTERMEDIATE_SIZE], _scale(state, expert, "w3")
        )
        assert torch.equal(
            packed.gate_up_scales[expert_pos, INTERMEDIATE_SIZE:], _scale(state, expert, "w1")
        )
        assert torch.equal(packed.down_scales[expert_pos], _scale(state, expert, "w2"))

        got_w3_nibbles = _unpack_nibbles(
            packed.gate_up_blocks[expert_pos, :INTERMEDIATE_SIZE].reshape(
                INTERMEDIATE_SIZE, HIDDEN_SIZE // 2
            )
        )
        expected_w3_nibbles = _unpack_nibbles(
            _raw_uint8(state[_key(LAYER, expert, "w3", "weight")])
        )
        assert torch.equal(got_w3_nibbles, expected_w3_nibbles)


def test_layout_preserves_float8_e8m0_scale_raw_bytes_when_dtype_exists() -> None:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        pytest.skip("torch.float8_e8m0fnu is not available in this torch build")

    state = _state_dict(experts=(0,), scale_dtype=e8m0_dtype)

    packed = _layout().pack_experts(
        state,
        layer=LAYER,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_experts=1,
    )

    assert packed.gate_up_scales.dtype == torch.uint8
    assert packed.down_scales.dtype == torch.uint8
    assert torch.equal(packed.gate_up_scales[0, :INTERMEDIATE_SIZE], _scale(state, 0, "w3"))
    assert torch.equal(packed.gate_up_scales[0, INTERMEDIATE_SIZE:], _scale(state, 0, "w1"))
    assert torch.equal(packed.down_scales[0], _scale(state, 0, "w2"))


@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size", "message"),
    (
        (0, INTERMEDIATE_SIZE, "hidden_size.*positive"),
        (63, INTERMEDIATE_SIZE, "hidden_size.*divisible by 32"),
        (HIDDEN_SIZE, 31, "intermediate_size.*divisible by 32"),
    ),
)
def test_layout_validates_model_dimensions(
    hidden_size: int,
    intermediate_size: int,
    message: str,
) -> None:
    state = _state_dict(experts=(0,))

    with pytest.raises(ValueError, match=message):
        _layout().pack_experts(
            state,
            layer=LAYER,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )


def test_layout_reports_bad_shape_with_tensor_name_and_expected_shape() -> None:
    state = _state_dict(experts=(0,))
    state[_key(LAYER, 0, "w3", "weight")] = torch.zeros(
        (INTERMEDIATE_SIZE, HIDDEN_SIZE // 2 + 1), dtype=torch.uint8
    )

    with pytest.raises(
        ValueError,
        match=r"layers\.3\.ffn\.experts\.0\.w3\.weight.*expected \[32, 32\]",
    ):
        _layout().pack_experts(
            state,
            layer=LAYER,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
        )


@pytest.mark.parametrize(
    ("tensor_kind", "dtype", "message"),
    (
        ("weight", torch.float32, "packed.*dtype"),
        ("scale", torch.float32, "scale.*dtype"),
    ),
)
def test_layout_validates_tensor_dtypes(
    tensor_kind: str,
    dtype: torch.dtype,
    message: str,
) -> None:
    state = _state_dict(experts=(0,))
    state[_key(LAYER, 0, "w3", tensor_kind)] = torch.zeros(
        tuple(state[_key(LAYER, 0, "w3", tensor_kind)].shape), dtype=dtype
    )

    with pytest.raises(ValueError, match=message):
        _layout().pack_experts(
            state,
            layer=LAYER,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
        )


def test_layout_reports_missing_key_with_full_tensor_name() -> None:
    state = _state_dict(experts=(0,))
    missing_key = _key(LAYER, 0, "w2", "scale")
    del state[missing_key]

    with pytest.raises(ValueError, match=missing_key.replace(".", r"\.")):
        _layout().pack_experts(
            state,
            layer=LAYER,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            num_experts=1,
        )


def test_validate_checkpoint_metadata_requires_all_runtime_projections_per_expert() -> None:
    metadata = _metadata_from_state(_state_dict(experts=(0,)))
    missing_weight = _key(LAYER, 0, "w2", "weight")
    missing_scale = _key(LAYER, 0, "w2", "scale")
    del metadata[missing_weight]
    del metadata[missing_scale]

    with pytest.raises(ValueError, match=r"missing required projection w2"):
        _layout().validate_checkpoint_metadata(metadata)


def test_validate_checkpoint_metadata_accepts_complete_projection_sets() -> None:
    metadata = _metadata_from_state(_state_dict(experts=(0, 1)))

    assert _layout().validate_checkpoint_metadata(metadata) == 6


def test_load_runtime_buffers_removes_consumed_source_keys_and_keeps_unrelated_keys() -> None:
    layout = _layout()
    source_state = _state_dict(experts=(0, 1))
    expected = layout.pack_experts(
        source_state,
        layer=LAYER,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_experts=2,
    )
    prefix = "root."
    state = {prefix + name: tensor for name, tensor in source_state.items()}
    consumed_keys = {
        prefix + _key(LAYER, expert, projection, tensor_kind)
        for expert in (0, 1)
        for projection in ("w1", "w2", "w3")
        for tensor_kind in ("weight", "scale")
    }
    unrelated_keys = set(state) - consumed_keys

    layout.load_runtime_buffers(
        state,
        prefix,
        layer=LAYER,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        target_gate_up_blocks="layers.3.ffn.experts.gate_up_proj_blocks",
        target_gate_up_scales="layers.3.ffn.experts.gate_up_proj_scales",
        target_down_blocks="layers.3.ffn.experts.down_proj_blocks",
        target_down_scales="layers.3.ffn.experts.down_proj_scales",
        num_experts=2,
    )

    assert consumed_keys.isdisjoint(state)
    assert unrelated_keys.issubset(state)
    assert torch.equal(
        state[prefix + "layers.3.ffn.experts.gate_up_proj_blocks"],
        expected.gate_up_blocks,
    )
    assert torch.equal(
        state[prefix + "layers.3.ffn.experts.gate_up_proj_scales"],
        expected.gate_up_scales,
    )
    assert torch.equal(
        state[prefix + "layers.3.ffn.experts.down_proj_blocks"],
        expected.down_blocks,
    )
    assert torch.equal(
        state[prefix + "layers.3.ffn.experts.down_proj_scales"],
        expected.down_scales,
    )


def test_layout_rejects_duplicate_explicit_expert_indices() -> None:
    state = _state_dict(experts=(0, 1))

    with pytest.raises(ValueError, match="[Dd]uplicate"):
        _layout().pack_experts(
            state,
            layer=LAYER,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            expert_indices=(1, 1),
            num_experts=2,
        )


def test_explicit_expert_subset_order() -> None:
    state = _state_dict(experts=(0, 1, 2, 3, 4))

    subset = _layout().pack_experts(
        state,
        layer=LAYER,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        expert_indices=(3, 1),
        num_experts=5,
    )
    assert subset.expert_indices == (3, 1)
    assert torch.equal(subset.down_scales[0], _scale(state, 3, "w2"))
    assert torch.equal(subset.down_scales[1], _scale(state, 1, "w2"))
