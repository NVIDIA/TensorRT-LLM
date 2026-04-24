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
import torch

from tensorrt_llm._torch.auto_deploy.transform.library.deepseek_v4_mxfp4 import (
    DeepSeekV4ExpertKey,
    DeepSeekV4MXFP4LoaderError,
    expert_parallel_slice,
    load_deepseek_v4_mxfp4_experts,
    parse_deepseek_v4_expert_key,
    slice_deepseek_v4_mxfp4_experts,
)

_HAS_E8M0 = hasattr(torch, "float8_e8m0fnu")
_requires_e8m0 = pytest.mark.skipif(
    not _HAS_E8M0,
    reason="torch.float8_e8m0fnu is not available in this PyTorch build",
)


def _pack_fp4(logical_values: torch.Tensor) -> torch.Tensor:
    assert logical_values.dtype == torch.uint8
    assert logical_values.shape[-1] % 2 == 0
    low = logical_values[..., 0::2] & 0x0F
    high = (logical_values[..., 1::2] & 0x0F) << 4
    return (low | high).contiguous().view(torch.int8)


def _unpack_fp4(packed_values: torch.Tensor) -> torch.Tensor:
    raw = packed_values.view(torch.uint8)
    unpacked = torch.empty(
        (*raw.shape[:-1], raw.shape[-1] * 2), dtype=torch.uint8, device=raw.device
    )
    unpacked[..., 0::2] = raw & 0x0F
    unpacked[..., 1::2] = raw >> 4
    return unpacked


def _logical_fp4(rows: int, cols: int, offset: int) -> torch.Tensor:
    return (torch.arange(rows * cols, dtype=torch.uint8).reshape(rows, cols) + offset) & 0x0F


def _scale_bytes(rows: int, cols: int, offset: int) -> torch.Tensor:
    return (
        torch.arange(rows * cols, dtype=torch.uint8).reshape(rows, cols) * 3 + offset
    ).contiguous()


def _make_expert_state(
    *,
    layer_idx: int,
    expert_idx: int,
    hidden_size: int,
    intermediate_size: int,
    scale_dtype: torch.dtype = torch.uint8,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    prefix = f"layers.{layer_idx}.ffn.experts.{expert_idx}"
    logical = {
        "w1": _logical_fp4(intermediate_size, hidden_size, 1 + expert_idx),
        "w2": _logical_fp4(hidden_size, intermediate_size, 5 + expert_idx),
        "w3": _logical_fp4(intermediate_size, hidden_size, 9 + expert_idx),
    }
    scales = {
        "w1": _scale_bytes(intermediate_size, hidden_size // 32, 17 + expert_idx),
        "w2": _scale_bytes(hidden_size, intermediate_size // 32, 29 + expert_idx),
        "w3": _scale_bytes(intermediate_size, hidden_size // 32, 43 + expert_idx),
    }

    state = {}
    for proj in ("w1", "w2", "w3"):
        state[f"{prefix}.{proj}.weight"] = _pack_fp4(logical[proj])
        scale = scales[proj]
        if scale_dtype != torch.uint8:
            scale = scale.view(scale_dtype)
        state[f"{prefix}.{proj}.scale"] = scale
    return state, {"logical": logical, "scales": scales}


def _make_state_dict(
    *,
    num_experts: int,
    layer_idx: int = 3,
    hidden_size: int = 64,
    intermediate_size: int = 32,
    scale_dtype: torch.dtype = torch.uint8,
) -> tuple[dict[str, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    state: dict[str, torch.Tensor] = {}
    expected = {}
    for expert_idx in range(num_experts):
        expert_state, expert_expected = _make_expert_state(
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            scale_dtype=scale_dtype,
        )
        state.update(expert_state)
        expected[expert_idx] = expert_expected
    return state, expected


def test_parse_deepseek_v4_expert_key_accepts_only_routed_expert_tensors() -> None:
    assert parse_deepseek_v4_expert_key("layers.12.ffn.experts.34.w2.scale") == (
        DeepSeekV4ExpertKey(layer_idx=12, expert_idx=34, proj="w2", suffix="scale")
    )
    assert parse_deepseek_v4_expert_key("layers.0.ffn.experts.1.w3.weight") == (
        DeepSeekV4ExpertKey(layer_idx=0, expert_idx=1, proj="w3", suffix="weight")
    )

    assert parse_deepseek_v4_expert_key("layers.0.ffn.shared_experts.w1.weight") is None
    assert parse_deepseek_v4_expert_key("layers.0.ffn.experts.1.w4.weight") is None
    assert parse_deepseek_v4_expert_key("model.layers.0.ffn.experts.1.w1.weight") is None


def test_loader_stacks_packed_blocks_in_w3_w1_runtime_order() -> None:
    hidden_size = 64
    intermediate_size = 32
    state, expected = _make_state_dict(
        num_experts=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    layout = load_deepseek_v4_mxfp4_experts(
        state,
        layer_idx=3,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=2,
    )

    assert layout.gate_up_blocks.shape == (2, 2 * intermediate_size, hidden_size // 32, 16)
    assert layout.gate_up_scales.shape == (2, 2 * intermediate_size, hidden_size // 32)
    assert layout.down_blocks.shape == (2, hidden_size, intermediate_size // 32, 16)
    assert layout.down_scales.shape == (2, hidden_size, intermediate_size // 32)
    assert layout.gate_up_blocks.dtype == torch.uint8
    assert layout.gate_up_scales.dtype == torch.uint8
    assert layout.down_blocks.dtype == torch.uint8
    assert layout.down_scales.dtype == torch.uint8
    assert layout.expert_indices == (0, 1)

    first_expert = expected[0]
    gate_up_packed = layout.gate_up_blocks[0].reshape(2 * intermediate_size, hidden_size // 2)
    down_packed = layout.down_blocks[0].reshape(hidden_size, intermediate_size // 2)

    torch.testing.assert_close(
        _unpack_fp4(gate_up_packed[:intermediate_size]),
        first_expert["logical"]["w3"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _unpack_fp4(gate_up_packed[intermediate_size:]),
        first_expert["logical"]["w1"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _unpack_fp4(down_packed),
        first_expert["logical"]["w2"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        layout.gate_up_scales[0, :intermediate_size],
        first_expert["scales"]["w3"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        layout.gate_up_scales[0, intermediate_size:],
        first_expert["scales"]["w1"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(layout.down_scales[0], first_expert["scales"]["w2"], rtol=0, atol=0)


@_requires_e8m0
def test_loader_preserves_e8m0_scale_exponent_bytes() -> None:
    hidden_size = 64
    intermediate_size = 32
    state, expected = _make_state_dict(
        num_experts=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        scale_dtype=torch.float8_e8m0fnu,
    )

    layout = load_deepseek_v4_mxfp4_experts(
        state,
        layer_idx=3,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=1,
    )

    w3_scale_key = "layers.3.ffn.experts.0.w3.scale"
    numeric_conversion = state[w3_scale_key].to(torch.uint8)

    torch.testing.assert_close(
        layout.gate_up_scales[0, :intermediate_size],
        expected[0]["scales"]["w3"],
        rtol=0,
        atol=0,
    )
    assert not torch.equal(numeric_conversion, expected[0]["scales"]["w3"])


def test_loader_validates_required_shapes() -> None:
    state, _ = _make_state_dict(num_experts=1, hidden_size=64, intermediate_size=32)
    state["layers.3.ffn.experts.0.w1.weight"] = torch.zeros((32, 31), dtype=torch.int8)

    with pytest.raises(DeepSeekV4MXFP4LoaderError, match=r"w1\.weight.*expected \[32, 32\]"):
        load_deepseek_v4_mxfp4_experts(
            state,
            layer_idx=3,
            hidden_size=64,
            intermediate_size=32,
            num_experts=1,
        )


def test_loader_rejects_missing_expert_keys() -> None:
    state, _ = _make_state_dict(num_experts=1, hidden_size=64, intermediate_size=32)
    del state["layers.3.ffn.experts.0.w2.scale"]

    with pytest.raises(DeepSeekV4MXFP4LoaderError, match="Missing.*w2\\.scale"):
        load_deepseek_v4_mxfp4_experts(
            state,
            layer_idx=3,
            hidden_size=64,
            intermediate_size=32,
            num_experts=1,
        )


def test_loader_can_stack_explicit_expert_subset_in_requested_order() -> None:
    state, expected = _make_state_dict(num_experts=4, hidden_size=64, intermediate_size=32)

    layout = load_deepseek_v4_mxfp4_experts(
        state,
        layer_idx=3,
        hidden_size=64,
        intermediate_size=32,
        num_experts=4,
        expert_indices=(3, 1),
    )

    assert layout.expert_indices == (3, 1)
    torch.testing.assert_close(
        _unpack_fp4(layout.gate_up_blocks[0, :32].reshape(32, 32)),
        expected[3]["logical"]["w3"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _unpack_fp4(layout.gate_up_blocks[1, :32].reshape(32, 32)),
        expected[1]["logical"]["w3"],
        rtol=0,
        atol=0,
    )


def test_expert_parallel_split_slices_only_expert_dimension() -> None:
    state, _ = _make_state_dict(num_experts=8, hidden_size=64, intermediate_size=32)
    layout = load_deepseek_v4_mxfp4_experts(
        state,
        layer_idx=3,
        hidden_size=64,
        intermediate_size=32,
        num_experts=8,
    )

    assert expert_parallel_slice(8, ep_size=2, ep_rank=0) == slice(0, 4)
    assert expert_parallel_slice(8, ep_size=2, ep_rank=1) == slice(4, 8)

    rank_one = slice_deepseek_v4_mxfp4_experts(layout, ep_size=2, ep_rank=1)

    assert rank_one.expert_indices == (4, 5, 6, 7)
    assert rank_one.gate_up_blocks.shape == layout.gate_up_blocks[4:8].shape
    torch.testing.assert_close(rank_one.gate_up_blocks, layout.gate_up_blocks[4:8])
    torch.testing.assert_close(rank_one.gate_up_scales, layout.gate_up_scales[4:8])
    torch.testing.assert_close(rank_one.down_blocks, layout.down_blocks[4:8])
    torch.testing.assert_close(rank_one.down_scales, layout.down_scales[4:8])
