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

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    DeepseekV4MoE,
)
from tensorrt_llm._torch.auto_deploy.models.quant_checkpoint_layout import (
    PackedMxfp4ExpertsCheckpointLayout,
)

HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 32
LAYER = 3
_E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


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


def _small_deepseek_v4_config(**overrides: object) -> DeepseekV4Config:
    values = {
        "vocab_size": 16,
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "q_lora_rank": 16,
        "qk_rope_head_dim": 8,
        "o_lora_rank": 16,
        "o_groups": 2,
        "sliding_window": 4,
        "compress_ratios": (0,),
        "index_n_heads": 2,
        "index_head_dim": 16,
        "index_topk": 2,
        "moe_intermediate_size": INTERMEDIATE_SIZE,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "num_hash_layers": 0,
        "max_position_embeddings": 16,
        "ad_rope_cache_len": 16,
        "ad_compress_max_seq_len": 16,
        "hc_mult": 1,
        "hc_sinkhorn_iters": 1,
        "ad_use_mxfp4_experts": True,
    }
    values.update(overrides)
    return DeepseekV4Config(**values)


def _key(layer: int, expert: int, weight_name: str, tensor_kind: str) -> str:
    return f"layers.{layer}.ffn.experts.{expert}.{weight_name}.{tensor_kind}"


def _raw_bytes(shape: Sequence[int], offset: int) -> torch.Tensor:
    values = (torch.arange(math.prod(shape), dtype=torch.int64) + offset) % 256
    return values.to(torch.uint8).reshape(tuple(shape))


def _controlled_mxfp4_weight(
    shape: tuple[int, int], offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = shape
    assert cols % 32 == 0
    codes = (torch.arange(rows * cols, dtype=torch.int64) + offset) % len(_E2M1_VALUES)
    codes = codes.reshape(rows, cols).to(torch.uint8)
    block_codes = codes.reshape(rows, cols // 32, 32)
    packed = (block_codes[..., 0::2] | (block_codes[..., 1::2] << 4)).to(torch.uint8)
    scales = torch.full((rows, cols // 32), 127, dtype=torch.uint8)
    return packed.reshape(rows, cols // 2), scales


def _controlled_mxfp4_state(
    *,
    experts: Sequence[int],
    layer: int,
    hidden_size: int = HIDDEN_SIZE,
    intermediate_size: int = INTERMEDIATE_SIZE,
) -> dict[str, torch.Tensor]:
    state = {}
    for expert in experts:
        for projection, shape, offset in (
            ("w1", (intermediate_size, hidden_size), 3),
            ("w2", (hidden_size, intermediate_size), 7),
            ("w3", (intermediate_size, hidden_size), 11),
        ):
            weight, scales = _controlled_mxfp4_weight(shape, offset + 17 * expert)
            state[_key(layer, expert, projection, "weight")] = weight
            state[_key(layer, expert, projection, "scale")] = scales
    return state


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


def test_deepseek_v4_load_state_dict_skips_mtp_checkpoint_keys() -> None:
    model = DeepseekV4ForCausalLM(_small_deepseek_v4_config(ad_use_mxfp4_experts=False)).eval()

    incompatible = model.load_state_dict(
        {"model.mtp.0.attn.wq_a.weight": torch.empty(1)},
        strict=False,
    )

    assert incompatible.unexpected_keys == []


def test_deepseek_v4_load_state_dict_accepts_packed_mxfp4_runtime_buffers() -> None:
    layer = 0
    source_state = {
        name: tensor
        for name, tensor in _state_dict(experts=(0, 1), layer=layer).items()
        if f"layers.{layer}.ffn.experts." in name and ".w4." not in name
    }
    expected = _layout().pack_experts(
        source_state,
        layer=layer,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_experts=2,
    )
    checkpoint_state = {f"model.{name}": tensor for name, tensor in source_state.items()}
    model = DeepseekV4ForCausalLM(_small_deepseek_v4_config()).eval()

    incompatible = model.load_state_dict(checkpoint_state, strict=False)

    assert incompatible.unexpected_keys == []
    moe = model.layers[layer].ffn
    assert torch.equal(moe.experts.gate_up_proj_blocks, expected.gate_up_blocks)
    assert torch.equal(moe.experts.gate_up_proj_scales, expected.gate_up_scales)
    assert torch.equal(moe.experts.down_proj_blocks, expected.down_blocks)
    assert torch.equal(moe.experts.down_proj_scales, expected.down_scales)


def test_deepseek_v4_mxfp4_runtime_buffers_are_shape_complete_for_export() -> None:
    config = _small_deepseek_v4_config()
    moe = DeepseekV4MoE(config, layer_idx=0).eval()

    assert moe.experts.gate_up_proj_blocks.shape == (
        config.n_routed_experts,
        2 * config.moe_intermediate_size,
        config.hidden_size // 32,
        16,
    )
    assert moe.experts.gate_up_proj_scales.shape == (
        config.n_routed_experts,
        2 * config.moe_intermediate_size,
        config.hidden_size // 32,
    )
    assert moe.experts.down_proj_blocks.shape == (
        config.n_routed_experts,
        config.hidden_size,
        config.moe_intermediate_size // 32,
        16,
    )
    assert moe.experts.down_proj_scales.shape == (
        config.n_routed_experts,
        config.hidden_size,
        config.moe_intermediate_size // 32,
    )


def test_deepseek_v4_remap_hook_finds_mxfp4_loader_under_non_iterable_layers() -> None:
    layer = 0
    source_state = {
        name: tensor
        for name, tensor in _state_dict(experts=(0, 1), layer=layer).items()
        if f"layers.{layer}.ffn.experts." in name and ".w4." not in name
    }
    expected = _layout().pack_experts(
        source_state,
        layer=layer,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_experts=2,
    )
    checkpoint_state = {f"model.{name}": tensor for name, tensor in source_state.items()}
    config = _small_deepseek_v4_config()
    moe = DeepseekV4MoE(config, layer_idx=layer).eval()
    container = torch.nn.Module()
    container.layers = torch.nn.Module()
    container.layers.block = torch.nn.Module()
    container.layers.block.ffn = moe

    DeepseekV4ForCausalLM._remap_load_state_hook(container, checkpoint_state, "")

    assert torch.equal(
        checkpoint_state["layers.0.ffn.experts.gate_up_proj_blocks"],
        expected.gate_up_blocks,
    )
    assert torch.equal(
        checkpoint_state["layers.0.ffn.experts.gate_up_proj_scales"],
        expected.gate_up_scales,
    )
    assert torch.equal(
        checkpoint_state["layers.0.ffn.experts.down_proj_blocks"],
        expected.down_blocks,
    )
    assert torch.equal(
        checkpoint_state["layers.0.ffn.experts.down_proj_scales"],
        expected.down_scales,
    )
    assert moe.experts.gate_up_proj_blocks.shape == expected.gate_up_blocks.shape
    assert moe.experts.down_proj_blocks.shape == expected.down_blocks.shape


def test_deepseek_v4_remap_hook_packs_graph_module_style_mxfp4_buffers() -> None:
    layer = 0
    source_state = {
        name: tensor
        for name, tensor in _state_dict(experts=(0, 1), layer=layer).items()
        if f"layers.{layer}.ffn.experts." in name and ".w4." not in name
    }
    expected = _layout().pack_experts(
        source_state,
        layer=layer,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_experts=2,
    )
    checkpoint_state = {f"model.{name}": tensor for name, tensor in source_state.items()}

    experts = torch.nn.Module()
    experts.register_buffer(
        "gate_up_proj_blocks",
        torch.empty(expected.gate_up_blocks.shape, dtype=torch.uint8),
    )
    experts.register_buffer(
        "gate_up_proj_scales",
        torch.empty(expected.gate_up_scales.shape, dtype=torch.uint8),
    )
    experts.register_buffer(
        "down_proj_blocks",
        torch.empty(expected.down_blocks.shape, dtype=torch.uint8),
    )
    experts.register_buffer(
        "down_proj_scales",
        torch.empty(expected.down_scales.shape, dtype=torch.uint8),
    )
    block = torch.nn.Module()
    block.ffn = torch.nn.Module()
    block.ffn.experts = experts
    container = torch.nn.Module()
    container.layers = torch.nn.ModuleList([block])

    DeepseekV4ForCausalLM._remap_load_state_hook(container, checkpoint_state, "")

    assert torch.equal(
        checkpoint_state["layers.0.ffn.experts.gate_up_proj_blocks"],
        expected.gate_up_blocks,
    )
    assert torch.equal(
        checkpoint_state["layers.0.ffn.experts.gate_up_proj_scales"],
        expected.gate_up_scales,
    )
    assert torch.equal(
        checkpoint_state["layers.0.ffn.experts.down_proj_blocks"],
        expected.down_blocks,
    )
    assert torch.equal(
        checkpoint_state["layers.0.ffn.experts.down_proj_scales"],
        expected.down_scales,
    )


def test_deepseek_v4_graph_mxfp4_hook_packs_full_expert_count_for_local_buffers() -> None:
    layer = 0
    source_state = {
        name: tensor
        for name, tensor in _state_dict(experts=(0, 1, 2, 3), layer=layer).items()
        if f"layers.{layer}.ffn.experts." in name and ".w4." not in name
    }
    expected = _layout().pack_experts(
        source_state,
        layer=layer,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_experts=4,
    )
    checkpoint_state = {f"model.{name}": tensor for name, tensor in source_state.items()}

    experts = torch.nn.Module()
    experts.register_buffer(
        "gate_up_proj_blocks",
        torch.empty(expected.gate_up_blocks[2:4].shape, dtype=torch.uint8),
    )
    experts.register_buffer(
        "gate_up_proj_scales",
        torch.empty(expected.gate_up_scales[2:4].shape, dtype=torch.uint8),
    )
    experts.register_buffer(
        "down_proj_blocks",
        torch.empty(expected.down_blocks[2:4].shape, dtype=torch.uint8),
    )
    experts.register_buffer(
        "down_proj_scales",
        torch.empty(expected.down_scales[2:4].shape, dtype=torch.uint8),
    )
    block = torch.nn.Module()
    block.ffn = torch.nn.Module()
    block.ffn.experts = experts
    container = torch.nn.Module()
    container.layers = torch.nn.ModuleList([block])

    DeepseekV4ForCausalLM._remap_load_state_hook(container, checkpoint_state, "")

    assert torch.equal(
        checkpoint_state["layers.0.ffn.experts.gate_up_proj_blocks"],
        expected.gate_up_blocks,
    )
    assert torch.equal(
        checkpoint_state["layers.0.ffn.experts.gate_up_proj_scales"],
        expected.gate_up_scales,
    )
    assert torch.equal(
        checkpoint_state["layers.0.ffn.experts.down_proj_blocks"],
        expected.down_blocks,
    )
    assert torch.equal(
        checkpoint_state["layers.0.ffn.experts.down_proj_scales"],
        expected.down_scales,
    )


def test_deepseek_v4_moe_forward_uses_routing_driven_packed_mxfp4_experts() -> None:
    layer = 0
    config = _small_deepseek_v4_config(
        num_experts_per_tok=2,
        num_hash_layers=1,
        swiglu_limit=0.75,
    )
    moe = DeepseekV4MoE(config, layer_idx=layer).eval()
    source_state = _controlled_mxfp4_state(experts=(0, 1), layer=layer)
    packed = _layout().pack_experts(
        source_state,
        layer=layer,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_experts=2,
    )
    moe.experts.gate_up_proj_blocks = packed.gate_up_blocks
    moe.experts.gate_up_proj_scales = packed.gate_up_scales
    moe.experts.down_proj_blocks = packed.down_blocks
    moe.experts.down_proj_scales = packed.down_scales

    for expert in moe.experts:
        expert.w1.weight.data.zero_()
        expert.w2.weight.data.zero_()
        expert.w3.weight.data.zero_()
    for param in moe.shared_experts.parameters():
        param.data.zero_()
    moe.gate.weight.data.zero_()
    tid2eid = torch.zeros_like(moe.gate.tid2eid)
    tid2eid[:, 0] = torch.arange(tid2eid.shape[0], dtype=torch.long) % 2
    tid2eid[:, 1] = 1 - tid2eid[:, 0]
    moe.gate.tid2eid.copy_(tid2eid)

    hidden_states = torch.linspace(
        -0.25,
        0.25,
        steps=3 * HIDDEN_SIZE,
        dtype=torch.float32,
    ).reshape(1, 3, HIDDEN_SIZE)
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    hidden_states_flat = hidden_states.reshape(-1, HIDDEN_SIZE)
    selected_experts, routing_weights = moe.gate(hidden_states_flat, input_ids.reshape(-1))

    expected = torch.ops.auto_deploy.torch_mxfp4_moe_from_routing(
        hidden_states_flat,
        selected_experts,
        routing_weights.to(hidden_states.dtype),
        packed.gate_up_blocks,
        moe.experts.gate_up_proj_bias,
        packed.gate_up_scales,
        1.0,
        config.swiglu_limit,
        packed.down_blocks,
        moe.experts.down_proj_bias,
        packed.down_scales,
        "up_gate",
        "deepseek",
    ).reshape_as(hidden_states)
    actual = moe(hidden_states, input_ids)

    assert expected.abs().max() > 0
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
