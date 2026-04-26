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

import json
import operator
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import mxfp4_moe
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.deepseek_v4_moe import (
    deepseek_v4_limited_swiglu,
    deepseek_v4_moe_reference,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.mxfp4_moe import (
    _deepseek_v4_swiglu_torch,
    _interleave_deepseek_v4_gate_up,
    _routing_from_prebuilt_metadata,
    _routing_from_precomputed,
    localize_deepseek_v4_routes_for_mxfp4,
    prepare_deepseek_v4_route_metadata,
    prepare_deepseek_v4_route_metadata_tensors,
)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4MoE,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.deepseek_v4_moe import (
    DeepSeekV4MoELowering,
    DeepSeekV4MoELoweringError,
)
from tensorrt_llm._torch.auto_deploy.transform.library.deepseek_v4_mxfp4 import (
    DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
    DEEPSEEK_V4_MXFP4_BYTES_PER_BLOCK,
    load_deepseek_v4_mxfp4_experts,
)
from tensorrt_llm._torch.auto_deploy.transform.library.sharding_ir import ApplyShardingHints
from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig
from tensorrt_llm._torch.auto_deploy.utils.e8m0 import e8m0_to_fp32, e8m0_to_uint8
from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_op_args, is_op

_DEEPSEEK_V4_REAL_CKPT_ENV = "DEEPSEEK_V4_FLASH_CHECKPOINT"
_DEFAULT_DEEPSEEK_V4_REAL_CKPT = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
    "bmarimuthu/dev/hf_home/manual/deepseek-ai__DeepSeek-V4-Flash"
)
_ROUTE_METADATA_ARG_NAMES = (
    "sorted_routing_weights",
    "expert_histogram",
    "sorted_route_indices",
    "inverse_route_indices",
    "expert_offsets",
    "expert_block_offsets",
    "expert_block_schedule",
)


def _expert_mlp(
    hidden_states: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
    swiglu_limit: float,
    *,
    apply_swiglu_limit: bool,
) -> torch.Tensor:
    gate = F.linear(hidden_states.to(w1_weight.dtype), w1_weight).float()
    up = F.linear(hidden_states.to(w3_weight.dtype), w3_weight).float()
    if apply_swiglu_limit:
        hidden = F.silu(torch.clamp(gate, max=swiglu_limit)) * torch.clamp(
            up, min=-swiglu_limit, max=swiglu_limit
        )
    else:
        hidden = F.silu(gate) * up
    return F.linear(hidden.to(w2_weight.dtype), w2_weight).to(hidden_states.dtype)


def _manual_from_routing(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    routed_w1_weight: torch.Tensor,
    routed_w2_weight: torch.Tensor,
    routed_w3_weight: torch.Tensor,
    shared_w1_weight: torch.Tensor | None,
    shared_w2_weight: torch.Tensor | None,
    shared_w3_weight: torch.Tensor | None,
    swiglu_limit: float,
) -> torch.Tensor:
    hidden_shape = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, hidden_shape[-1])
    output = torch.zeros_like(hidden_flat)

    for expert_idx in range(routed_w1_weight.shape[0]):
        token_idx, route_idx = torch.where(selected_experts == expert_idx)
        if token_idx.numel() == 0:
            continue
        expert_output = _expert_mlp(
            hidden_flat[token_idx],
            routed_w1_weight[expert_idx],
            routed_w2_weight[expert_idx],
            routed_w3_weight[expert_idx],
            swiglu_limit,
            apply_swiglu_limit=True,
        )
        output[token_idx] += (
            expert_output * routing_weights[token_idx, route_idx, None].to(expert_output.dtype)
        ).to(output.dtype)

    if shared_w1_weight is not None:
        assert shared_w2_weight is not None
        assert shared_w3_weight is not None
        output += _expert_mlp(
            hidden_flat,
            shared_w1_weight,
            shared_w2_weight,
            shared_w3_weight,
            swiglu_limit,
            apply_swiglu_limit=False,
        )

    return output.view(hidden_shape)


def _stacked_weights(
    num_experts: int = 4,
    hidden_size: int = 3,
    intermediate_size: int = 5,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w1 = (
        torch.arange(num_experts * intermediate_size * hidden_size, dtype=torch.float32)
        .reshape(num_experts, intermediate_size, hidden_size)
        .div(37.0)
        .sub(0.5)
        .to(dtype)
    )
    w2 = (
        torch.arange(num_experts * hidden_size * intermediate_size, dtype=torch.float32)
        .reshape(num_experts, hidden_size, intermediate_size)
        .div(29.0)
        .sub(0.25)
        .to(dtype)
    )
    w3 = (
        torch.arange(num_experts * intermediate_size * hidden_size, dtype=torch.float32)
        .reshape(num_experts, intermediate_size, hidden_size)
        .div(31.0)
        .sub(0.75)
        .to(dtype)
    )
    return w1, w2, w3


def _run_ir_sharding(
    gm: torch.fx.GraphModule,
    *,
    rank: int,
    world_size: int,
) -> tuple[torch.fx.GraphModule, object]:
    transform = ApplyShardingHints.from_kwargs(stage=Stages.SHARDING)
    shared_config = SharedConfig(
        local_rank=rank,
        world_size=world_size,
        dist_config=DistConfig(
            world_size=world_size,
            rank=rank,
            tp_size=world_size,
            moe_ep_size=world_size,
        ),
    )
    return transform._apply(gm, None, None, shared_config)


def _deepseek_v4_real_checkpoint_path() -> Path:
    checkpoint_path = Path(
        os.environ.get(_DEEPSEEK_V4_REAL_CKPT_ENV, _DEFAULT_DEEPSEEK_V4_REAL_CKPT)
    )
    if not checkpoint_path.exists():
        pytest.skip(
            f"DeepSeek V4 real checkpoint not found at {checkpoint_path}; set "
            f"{_DEEPSEEK_V4_REAL_CKPT_ENV} to run this coverage."
        )
    index_path = checkpoint_path / "model.safetensors.index.json"
    if not index_path.exists():
        pytest.skip(f"DeepSeek V4 safetensors index not found at {index_path}.")
    return checkpoint_path


def _load_real_checkpoint_tensors(
    checkpoint_path: Path,
    tensor_names: list[str],
) -> dict[str, torch.Tensor]:
    safetensors = pytest.importorskip("safetensors")
    index_path = checkpoint_path / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    missing = [name for name in tensor_names if name not in weight_map]
    assert missing == []

    by_file: dict[str, list[str]] = {}
    for name in tensor_names:
        by_file.setdefault(weight_map[name], []).append(name)

    tensors: dict[str, torch.Tensor] = {}
    for filename, file_tensor_names in by_file.items():
        with safetensors.safe_open(
            checkpoint_path / filename,
            framework="pt",
            device="cpu",
        ) as handle:
            for name in file_tensor_names:
                tensors[name] = handle.get_tensor(name)
    return tensors


def _real_mxfp4_weight_blocks(
    tensor: torch.Tensor,
    *,
    rows: int,
    logical_cols: int,
) -> torch.Tensor:
    assert tensor.dtype in (torch.int8, torch.uint8)
    assert tensor.shape == (
        rows,
        logical_cols // 2,
    )
    return (
        tensor.view(torch.uint8)
        .contiguous()
        .view(
            rows,
            logical_cols // DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
            DEEPSEEK_V4_MXFP4_BYTES_PER_BLOCK,
        )
    )


def _real_mxfp4_scale_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor.contiguous()
    return e8m0_to_uint8(tensor).contiguous()


def _assert_layout_matches_real_checkpoint(
    layout,
    state_dict: dict[str, torch.Tensor],
    *,
    layer_idx: int,
    expert_indices: tuple[int, ...],
    hidden_size: int,
    intermediate_size: int,
) -> None:
    assert layout.expert_indices == expert_indices
    assert layout.gate_up_blocks.shape == (
        len(expert_indices),
        2 * intermediate_size,
        hidden_size // DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        DEEPSEEK_V4_MXFP4_BYTES_PER_BLOCK,
    )
    assert layout.gate_up_scales.shape == (
        len(expert_indices),
        2 * intermediate_size,
        hidden_size // DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
    )
    assert layout.down_blocks.shape == (
        len(expert_indices),
        hidden_size,
        intermediate_size // DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        DEEPSEEK_V4_MXFP4_BYTES_PER_BLOCK,
    )
    assert layout.down_scales.shape == (
        len(expert_indices),
        hidden_size,
        intermediate_size // DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
    )

    for local_idx, expert_idx in enumerate(expert_indices):
        prefix = f"layers.{layer_idx}.ffn.experts.{expert_idx}"
        w3_blocks = _real_mxfp4_weight_blocks(
            state_dict[f"{prefix}.w3.weight"],
            rows=intermediate_size,
            logical_cols=hidden_size,
        )
        w1_blocks = _real_mxfp4_weight_blocks(
            state_dict[f"{prefix}.w1.weight"],
            rows=intermediate_size,
            logical_cols=hidden_size,
        )
        w2_blocks = _real_mxfp4_weight_blocks(
            state_dict[f"{prefix}.w2.weight"],
            rows=hidden_size,
            logical_cols=intermediate_size,
        )
        w3_scales = _real_mxfp4_scale_bytes(state_dict[f"{prefix}.w3.scale"])
        w1_scales = _real_mxfp4_scale_bytes(state_dict[f"{prefix}.w1.scale"])
        w2_scales = _real_mxfp4_scale_bytes(state_dict[f"{prefix}.w2.scale"])

        torch.testing.assert_close(
            layout.gate_up_blocks[local_idx, :intermediate_size],
            w3_blocks,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            layout.gate_up_blocks[local_idx, intermediate_size:],
            w1_blocks,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(layout.down_blocks[local_idx], w2_blocks, rtol=0, atol=0)
        torch.testing.assert_close(
            layout.gate_up_scales[local_idx, :intermediate_size],
            w3_scales,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            layout.gate_up_scales[local_idx, intermediate_size:],
            w1_scales,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(layout.down_scales[local_idx], w2_scales, rtol=0, atol=0)


def test_from_routing_matches_tiny_bf16_reference_with_precomputed_routing() -> None:
    hidden_states = torch.tensor(
        [
            [0.5, -1.0, 1.5],
            [1.25, 0.25, -0.75],
            [-0.5, 0.75, 0.125],
        ],
        dtype=torch.bfloat16,
    )
    selected_experts = torch.tensor([[0, 2], [3, 1], [2, 0]], dtype=torch.int32)
    routing_weights = torch.tensor([[0.75, 0.25], [0.4, 0.6], [0.125, 0.875]], dtype=torch.float32)
    routed_w1, routed_w2, routed_w3 = _stacked_weights()
    shared_w1 = torch.linspace(-0.25, 0.5, 15, dtype=torch.float32).view(5, 3).to(torch.bfloat16)
    shared_w2 = torch.linspace(0.1, 0.8, 15, dtype=torch.float32).view(3, 5).to(torch.bfloat16)
    shared_w3 = torch.linspace(-0.6, 0.3, 15, dtype=torch.float32).view(5, 3).to(torch.bfloat16)

    expected = _manual_from_routing(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_w1,
        routed_w2,
        routed_w3,
        shared_w1,
        shared_w2,
        shared_w3,
        swiglu_limit=10.0,
    )
    actual = torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_w1,
        routed_w2,
        routed_w3,
        shared_w1,
        shared_w2,
        shared_w3,
        10.0,
        "moe",
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)


def test_limited_swiglu_limit_zero_zeroes_up_branch() -> None:
    gate = torch.tensor([[20.0, -20.0]])
    up = torch.tensor([[15.0, -15.0]])

    actual = deepseek_v4_limited_swiglu(gate, up, swiglu_limit=0.0)

    torch.testing.assert_close(actual, torch.zeros_like(actual))


def test_limited_swiglu_limit_ten_clamps_gate_and_up() -> None:
    gate = torch.tensor([[20.0, -20.0]])
    up = torch.tensor([[15.0, -15.0]])

    actual = deepseek_v4_limited_swiglu(gate, up, swiglu_limit=10.0)
    expected = F.silu(torch.tensor([[10.0, -20.0]])) * torch.tensor([[10.0, -10.0]])

    torch.testing.assert_close(actual, expected)


def test_deepseek_v4_mxfp4_swiglu_matches_reference_without_gptoss_offset() -> None:
    gate_up_interleaved = torch.tensor([[20.0, 15.0, -20.0, -15.0]], dtype=torch.float32)

    actual = _deepseek_v4_swiglu_torch(gate_up_interleaved, alpha=1.0, limit=10.0)

    expected_gate = torch.tensor([[10.0, -20.0]], dtype=torch.float32)
    expected_up = torch.tensor([[10.0, -10.0]], dtype=torch.float32)
    expected = F.silu(expected_gate) * expected_up
    gptoss_style = F.silu(expected_gate) * (expected_up + 1)

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, gptoss_style)


def test_deepseek_v4_mxfp4_interleaves_checkpoint_gate_up_order() -> None:
    checkpoint_order = torch.tensor([[30, 31, 32, 10, 11, 12]], dtype=torch.uint8)

    actual = _interleave_deepseek_v4_gate_up(checkpoint_order)

    expected = torch.tensor([[10, 30, 11, 31, 12, 32]], dtype=torch.uint8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prepare_weights_scales_interleaves_deepseek_gate_up_before_swizzle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeSwizzledTensor:
        shape: torch.Size | None = None

    swizzle_calls = []

    def _fake_swizzle_mxfp4(
        weight: torch.Tensor, weight_scale: torch.Tensor
    ) -> tuple[_FakeSwizzledTensor, _FakeSwizzledTensor]:
        swizzle_calls.append((weight.clone(), weight_scale.clone()))
        return _FakeSwizzledTensor(), _FakeSwizzledTensor()

    monkeypatch.setattr(mxfp4_moe, "_swizzle_mxfp4", _fake_swizzle_mxfp4)
    mxfp4_moe._clear_mxfp4_weights_scales_cache()
    hidden_size = 64
    intermediate_size = 32
    gate_up_blocks = (
        torch.arange(2 * intermediate_size, dtype=torch.uint8)
        .view(1, 2 * intermediate_size, 1, 1)
        .expand(1, 2 * intermediate_size, hidden_size // 32, 16)
        .contiguous()
    )
    gate_up_scales = (
        torch.arange(2 * intermediate_size, dtype=torch.uint8)
        .view(1, 2 * intermediate_size, 1)
        .expand(1, 2 * intermediate_size, hidden_size // 32)
        .contiguous()
    )
    down_blocks = torch.zeros((1, hidden_size, intermediate_size // 32, 16), dtype=torch.uint8)
    down_scales = torch.zeros((1, hidden_size, intermediate_size // 32), dtype=torch.uint8)

    mxfp4_moe._prepare_weights_scales(
        hidden_size,
        gate_up_blocks,
        gate_up_scales,
        down_blocks,
        down_scales,
        interleave_gate_up=True,
    )

    gate_up_weight, gate_up_scale = swizzle_calls[0]
    expected_prefix = torch.tensor([32, 0, 33, 1, 34, 2, 35, 3], dtype=torch.uint8)
    torch.testing.assert_close(gate_up_weight[0, 0, :8], expected_prefix, rtol=0, atol=0)
    torch.testing.assert_close(gate_up_scale[0, 0, :8], expected_prefix, rtol=0, atol=0)


def test_from_routing_limit_zero_zeroes_routed_output_without_shared_expert() -> None:
    hidden_states = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    selected_experts = torch.tensor([[0], [0]], dtype=torch.int64)
    routing_weights = torch.ones((2, 1), dtype=torch.float32)
    routed_w1 = torch.tensor([[[20.0], [-20.0]]])
    routed_w2 = torch.tensor([[[1.0, 2.0]]])
    routed_w3 = torch.tensor([[[15.0], [-15.0]]])

    actual = torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_w1,
        routed_w2,
        routed_w3,
        None,
        None,
        None,
        0.0,
        "moe",
    )

    torch.testing.assert_close(actual, torch.zeros_like(hidden_states))


def test_from_routing_limit_ten_clamps_gate_and_up_inside_expert() -> None:
    hidden_states = torch.tensor([[1.0]], dtype=torch.float32)
    selected_experts = torch.tensor([[0]], dtype=torch.int64)
    routing_weights = torch.ones((1, 1), dtype=torch.float32)
    routed_w1 = torch.tensor([[[20.0], [-20.0]]])
    routed_w2 = torch.tensor([[[1.0, 2.0]]])
    routed_w3 = torch.tensor([[[15.0], [-15.0]]])

    actual = torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_w1,
        routed_w2,
        routed_w3,
        None,
        None,
        None,
        10.0,
        "moe",
    )
    expected_hidden = F.silu(torch.tensor([[10.0, -20.0]])) * torch.tensor([[10.0, -10.0]])
    expected = expected_hidden @ torch.tensor([[1.0], [2.0]])

    torch.testing.assert_close(actual, expected)


def test_canonical_moe_uses_router_surface() -> None:
    hidden_states = torch.tensor([[0.25, -0.5, 1.0], [1.0, 0.5, -0.25]], dtype=torch.float32)
    router_weight = torch.tensor(
        [
            [0.5, -0.25, 0.75],
            [-0.25, 0.5, 0.125],
            [0.75, 0.125, -0.5],
            [0.25, 0.25, 0.25],
        ],
        dtype=torch.float32,
    )
    router_bias = torch.tensor([0.0, 0.25, -0.25, 0.125], dtype=torch.float32)
    routed_w1, routed_w2, routed_w3 = _stacked_weights(
        num_experts=4, hidden_size=3, intermediate_size=2, dtype=torch.float32
    )

    expected = deepseek_v4_moe_reference(
        hidden_states,
        input_ids=None,
        router_weight=router_weight,
        router_bias=router_bias,
        tid2eid=None,
        routed_w1_weight=routed_w1,
        routed_w2_weight=routed_w2,
        routed_w3_weight=routed_w3,
        shared_w1_weight=None,
        shared_w2_weight=None,
        shared_w3_weight=None,
        top_k=2,
        route_scale=1.5,
        swiglu_limit=10.0,
        is_hash_layer=False,
    )
    actual = torch.ops.auto_deploy.torch_deepseek_v4_moe.default(
        hidden_states,
        None,
        router_weight,
        router_bias,
        None,
        routed_w1,
        routed_w2,
        routed_w3,
        None,
        None,
        None,
        2,
        1.5,
        10.0,
        False,
        "moe",
    )

    torch.testing.assert_close(actual, expected)


def test_from_routing_fake_returns_meta_output_shape() -> None:
    hidden_states = torch.empty((2, 3, 4), dtype=torch.bfloat16, device="meta")
    selected_experts = torch.empty((6, 2), dtype=torch.int64, device="meta")
    routing_weights = torch.empty((6, 2), dtype=torch.float32, device="meta")
    routed_w1 = torch.empty((4, 5, 4), dtype=torch.bfloat16, device="meta")
    routed_w2 = torch.empty((4, 4, 5), dtype=torch.bfloat16, device="meta")
    routed_w3 = torch.empty((4, 5, 4), dtype=torch.bfloat16, device="meta")

    actual = torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_w1,
        routed_w2,
        routed_w3,
        None,
        None,
        None,
        10.0,
        "moe",
    )

    assert actual.shape == hidden_states.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device.type == "meta"


def test_deepseek_v4_route_metadata_prepares_sort_histogram_and_indices() -> None:
    selected_experts = torch.tensor([[2, 0, 2], [1, 0, 3]], dtype=torch.int64)
    routing_weights = torch.tensor(
        [[0.2, 0.1, 0.3], [0.4, 0.5, 0.6]],
        dtype=torch.float32,
    )

    route_metadata = prepare_deepseek_v4_route_metadata(
        selected_experts,
        routing_weights,
        num_experts=4,
    )
    metadata_tensors = prepare_deepseek_v4_route_metadata_tensors(
        selected_experts,
        routing_weights,
        num_experts=4,
    )

    expected_sorted = torch.tensor([1, 4, 3, 0, 2, 5], dtype=torch.int32)
    expected_inverse = torch.tensor([3, 0, 4, 2, 1, 5], dtype=torch.int32)
    expected_histogram = torch.tensor([2, 1, 2, 1], dtype=torch.int32)
    expected_offsets = torch.tensor([0, 2, 3, 5, 6], dtype=torch.int32)

    assert route_metadata.num_experts == 4
    assert route_metadata.top_k == 3
    assert route_metadata.num_routes == 6
    torch.testing.assert_close(
        route_metadata.flat_expert_ids,
        selected_experts.reshape(-1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(route_metadata.sorted_route_indices, expected_sorted)
    torch.testing.assert_close(route_metadata.inverse_route_indices, expected_inverse)
    torch.testing.assert_close(route_metadata.expert_histogram, expected_histogram)
    torch.testing.assert_close(
        route_metadata.sorted_routing_weights,
        routing_weights.reshape(-1)[expected_sorted.to(torch.int64)],
    )
    torch.testing.assert_close(metadata_tensors.expert_offsets, expected_offsets)
    torch.testing.assert_close(metadata_tensors.expert_histogram, expected_histogram)
    torch.testing.assert_close(metadata_tensors.sorted_route_indices, expected_sorted)
    torch.testing.assert_close(metadata_tensors.inverse_route_indices, expected_inverse)


def _assert_routing_bridge_matches(
    actual: tuple[object, object, object],
    expected: tuple[object, object, object],
) -> None:
    actual_routing, actual_gather, actual_scatter = actual
    expected_routing, expected_gather, expected_scatter = expected

    assert actual_routing.n_expts_tot == expected_routing.n_expts_tot
    assert actual_routing.n_expts_act == expected_routing.n_expts_act
    torch.testing.assert_close(actual_routing.gate_scal, expected_routing.gate_scal)
    torch.testing.assert_close(actual_routing.expt_hist, expected_routing.expt_hist)
    torch.testing.assert_close(
        actual_routing.expt_data.slice_sizes,
        expected_routing.expt_data.slice_sizes,
    )
    torch.testing.assert_close(
        actual_routing.expt_data.slice_offs,
        expected_routing.expt_data.slice_offs,
    )
    torch.testing.assert_close(
        actual_routing.expt_data.block_offs_data,
        expected_routing.expt_data.block_offs_data,
    )
    torch.testing.assert_close(
        actual_routing.expt_data.block_schedule_data,
        expected_routing.expt_data.block_schedule_data,
    )
    torch.testing.assert_close(actual_gather.src_indx, expected_gather.src_indx)
    torch.testing.assert_close(actual_gather.dst_indx, expected_gather.dst_indx)
    torch.testing.assert_close(actual_scatter.src_indx, expected_scatter.src_indx)
    torch.testing.assert_close(actual_scatter.dst_indx, expected_scatter.dst_indx)


def _assert_prebuilt_metadata_matches_precomputed(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> None:
    metadata_tensors = prepare_deepseek_v4_route_metadata_tensors(
        selected_experts,
        routing_weights,
        num_experts,
    )
    actual = _routing_from_prebuilt_metadata(
        *metadata_tensors.as_tuple(),
        num_experts=num_experts,
        top_k=selected_experts.shape[1],
    )
    expected = _routing_from_precomputed(selected_experts, routing_weights, num_experts)

    _assert_routing_bridge_matches(actual, expected)


def test_deepseek_v4_prebuilt_route_metadata_matches_precomputed_bridge() -> None:
    selected_experts = torch.tensor(
        [
            [2, 0, 2, 7, 1, 0],
            [1, 0, 3, 7, 4, 4],
            [6, 2, 5, 3, 1, 0],
        ],
        dtype=torch.int64,
    )
    routing_weights = torch.arange(1, 19, dtype=torch.float32).view(3, 6).div(19.0)

    _assert_prebuilt_metadata_matches_precomputed(
        selected_experts,
        routing_weights,
        num_experts=8,
    )


def test_deepseek_v4_prebuilt_route_metadata_handles_all_routes_to_one_expert() -> None:
    selected_experts = torch.full((5, 6), 17, dtype=torch.int64)
    routing_weights = torch.linspace(0.05, 1.5, 30, dtype=torch.float32).view(5, 6)
    metadata_tensors = prepare_deepseek_v4_route_metadata_tensors(
        selected_experts,
        routing_weights,
        num_experts=32,
    )

    expected_histogram = torch.zeros(32, dtype=torch.int32)
    expected_histogram[17] = selected_experts.numel()

    torch.testing.assert_close(metadata_tensors.expert_histogram, expected_histogram)
    torch.testing.assert_close(
        metadata_tensors.sorted_route_indices,
        torch.arange(selected_experts.numel(), dtype=torch.int32),
    )
    _assert_prebuilt_metadata_matches_precomputed(
        selected_experts,
        routing_weights,
        num_experts=32,
    )


def test_deepseek_v4_prebuilt_route_metadata_handles_sparse_experts_32_topk6() -> None:
    selected_experts = torch.tensor(
        [
            [0, 31, 0, 31, 7, 7],
            [7, 0, 31, 7, 0, 31],
            [31, 31, 7, 0, 7, 0],
        ],
        dtype=torch.int64,
    )
    routing_weights = torch.arange(1, 19, dtype=torch.float32).view(3, 6).div(18.0)

    metadata_tensors = prepare_deepseek_v4_route_metadata_tensors(
        selected_experts,
        routing_weights,
        num_experts=32,
    )

    expected_histogram = torch.zeros(32, dtype=torch.int32)
    expected_histogram[0] = 6
    expected_histogram[7] = 6
    expected_histogram[31] = 6
    expected_sorted = torch.argsort(selected_experts.reshape(-1), stable=True).to(torch.int32)
    expected_offsets = torch.cumsum(
        torch.cat((torch.zeros(1, dtype=torch.int32), expected_histogram)),
        dim=0,
    ).to(torch.int32)

    torch.testing.assert_close(metadata_tensors.expert_histogram, expected_histogram)
    torch.testing.assert_close(metadata_tensors.sorted_route_indices, expected_sorted)
    torch.testing.assert_close(metadata_tensors.expert_offsets, expected_offsets)
    torch.testing.assert_close(
        metadata_tensors.sorted_routing_weights,
        routing_weights.reshape(-1)[expected_sorted.to(torch.int64)],
    )
    _assert_prebuilt_metadata_matches_precomputed(
        selected_experts,
        routing_weights,
        num_experts=32,
    )


def test_deepseek_v4_prebuilt_route_metadata_uses_32_local_experts_topk6_shape() -> None:
    selected_experts = torch.arange(30, dtype=torch.int64).view(5, 6).mul(7).add(3).remainder(32)
    routing_weights = torch.arange(1, 31, dtype=torch.float32).view(5, 6).div(31.0)

    metadata_tensors = prepare_deepseek_v4_route_metadata_tensors(
        selected_experts,
        routing_weights,
        num_experts=32,
    )

    assert metadata_tensors.sorted_routing_weights.shape == (30,)
    assert metadata_tensors.expert_histogram.shape == (32,)
    assert metadata_tensors.sorted_route_indices.shape == (30,)
    assert metadata_tensors.inverse_route_indices.shape == (30,)
    assert metadata_tensors.expert_offsets.shape == (33,)
    assert metadata_tensors.expert_block_offsets.shape[1] == 33
    assert metadata_tensors.expert_block_schedule.shape[1] == 30
    _assert_prebuilt_metadata_matches_precomputed(
        selected_experts,
        routing_weights,
        num_experts=32,
    )


def test_deepseek_v4_route_metadata_custom_op_fake_uses_32_experts_topk6_shape() -> None:
    selected_experts = torch.empty((7, 6), dtype=torch.int64, device="meta")
    routing_weights = torch.empty((7, 6), dtype=torch.float32, device="meta")

    (
        sorted_routing_weights,
        expert_histogram,
        sorted_route_indices,
        inverse_route_indices,
        expert_offsets,
        expert_block_offsets,
        expert_block_schedule,
    ) = torch.ops.auto_deploy.torch_deepseek_v4_mxfp4_route_metadata.default(
        selected_experts,
        routing_weights,
        32,
    )

    assert sorted_routing_weights.shape == (42,)
    assert sorted_routing_weights.dtype == torch.float32
    assert expert_histogram.shape == (32,)
    assert sorted_route_indices.shape == (42,)
    assert inverse_route_indices.shape == (42,)
    assert expert_offsets.shape == (33,)
    assert expert_block_offsets.shape[1] == 33
    assert expert_block_schedule.shape[1] == 32


def test_deepseek_v4_route_metadata_localizes_observed_ep8_rank_slice() -> None:
    selected_experts = torch.tensor([[0, 31, 32, 63, 64, 95, 224, 255]], dtype=torch.int64)
    routing_weights = torch.arange(1, 9, dtype=torch.float32).view(1, 8)

    ep_metadata = localize_deepseek_v4_routes_for_mxfp4(
        selected_experts,
        routing_weights,
        moe_ep_size=8,
        moe_ep_rank=7,
        num_global_experts=256,
    )
    route_metadata = prepare_deepseek_v4_route_metadata(
        ep_metadata.selected_experts_local,
        ep_metadata.routing_weights_local,
        num_experts=ep_metadata.experts_per_rank,
    )

    expected_mask = torch.tensor([[False, False, False, False, False, False, True, True]])
    expected_local = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 31]], dtype=torch.int64)
    expected_weights = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0]])
    expected_histogram = torch.zeros(32, dtype=torch.int32)
    expected_histogram[0] = 7
    expected_histogram[31] = 1

    assert ep_metadata.local_expert_start == 224
    assert ep_metadata.local_expert_end == 256
    assert ep_metadata.experts_per_rank == 32
    assert torch.equal(ep_metadata.rank_mask, expected_mask)
    torch.testing.assert_close(ep_metadata.selected_experts_local, expected_local)
    torch.testing.assert_close(ep_metadata.routing_weights_local, expected_weights)
    torch.testing.assert_close(route_metadata.expert_histogram, expected_histogram)
    _assert_prebuilt_metadata_matches_precomputed(
        ep_metadata.selected_experts_local,
        ep_metadata.routing_weights_local,
        num_experts=ep_metadata.experts_per_rank,
    )


def test_deepseek_v4_route_metadata_localizes_ep8_rank3_topk6_nonlocal_routes() -> None:
    selected_experts = torch.tensor(
        [
            [95, 96, 101, 127, 128, 255],
            [0, 100, 121, 190, 97, 32],
        ],
        dtype=torch.int64,
    )
    routing_weights = torch.arange(1, 13, dtype=torch.float32).view(2, 6)

    ep_metadata = localize_deepseek_v4_routes_for_mxfp4(
        selected_experts,
        routing_weights,
        moe_ep_size=8,
        moe_ep_rank=3,
        num_global_experts=256,
    )
    route_metadata = prepare_deepseek_v4_route_metadata(
        ep_metadata.selected_experts_local,
        ep_metadata.routing_weights_local,
        num_experts=ep_metadata.experts_per_rank,
    )

    expected_mask = torch.tensor(
        [
            [False, True, True, True, False, False],
            [False, True, True, False, True, False],
        ]
    )
    expected_local = torch.tensor(
        [
            [0, 0, 5, 31, 0, 0],
            [0, 4, 25, 0, 1, 0],
        ],
        dtype=torch.int64,
    )
    expected_weights = torch.tensor(
        [
            [0.0, 2.0, 3.0, 4.0, 0.0, 0.0],
            [0.0, 8.0, 9.0, 0.0, 11.0, 0.0],
        ]
    )
    expected_histogram = torch.zeros(32, dtype=torch.int32)
    expected_histogram[0] = 7
    expected_histogram[1] = 1
    expected_histogram[4] = 1
    expected_histogram[5] = 1
    expected_histogram[25] = 1
    expected_histogram[31] = 1

    assert ep_metadata.local_expert_start == 96
    assert ep_metadata.local_expert_end == 128
    assert ep_metadata.experts_per_rank == 32
    assert torch.equal(ep_metadata.rank_mask, expected_mask)
    torch.testing.assert_close(ep_metadata.selected_experts_local, expected_local)
    torch.testing.assert_close(ep_metadata.routing_weights_local, expected_weights)
    torch.testing.assert_close(route_metadata.expert_histogram, expected_histogram)
    _assert_prebuilt_metadata_matches_precomputed(
        ep_metadata.selected_experts_local,
        ep_metadata.routing_weights_local,
        num_experts=ep_metadata.experts_per_rank,
    )


def test_mxfp4_from_routing_uses_prebuilt_metadata_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    selected_experts = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)
    routing_weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float32)
    metadata_tensors = prepare_deepseek_v4_route_metadata_tensors(
        selected_experts,
        routing_weights,
        num_experts=2,
    )
    captured = {}

    def _fail_precomputed(*args, **kwargs):
        raise AssertionError("precomputed route metadata path should not run")

    def _fake_core_from_routing(
        hidden_states: torch.Tensor,
        gate_up_blocks: torch.Tensor,
        gate_up_bias: torch.Tensor,
        gate_up_scales: torch.Tensor,
        alpha: float,
        limit: float,
        down_blocks: torch.Tensor,
        down_bias: torch.Tensor,
        down_scales: torch.Tensor,
        routing_data,
        gather_idx,
        scatter_idx,
        **kwargs,
    ) -> torch.Tensor:
        del (
            gate_up_blocks,
            gate_up_bias,
            gate_up_scales,
            alpha,
            limit,
            down_blocks,
            down_bias,
            down_scales,
            kwargs,
        )
        captured["routing"] = (routing_data, gather_idx, scatter_idx)
        return torch.full_like(hidden_states, 3)

    monkeypatch.setattr(mxfp4_moe, "_routing_from_precomputed", _fail_precomputed)
    monkeypatch.setattr(
        mxfp4_moe,
        "_run_mxfp4_mlp_core_from_routing",
        _fake_core_from_routing,
    )

    actual = torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing.default(
        hidden_states,
        selected_experts,
        routing_weights,
        torch.empty((2, 8, 2, 16), dtype=torch.uint8),
        torch.empty((2, 8), dtype=torch.float32),
        torch.empty((2, 8, 2), dtype=torch.uint8),
        1.0,
        10.0,
        torch.empty((2, 4, 2, 16), dtype=torch.uint8),
        torch.empty((2, 4), dtype=torch.float32),
        torch.empty((2, 4, 2), dtype=torch.uint8),
        "moe",
        *metadata_tensors.as_tuple(),
    )
    expected_routing = _routing_from_prebuilt_metadata(
        *metadata_tensors.as_tuple(),
        num_experts=2,
        top_k=2,
    )

    torch.testing.assert_close(actual, torch.full_like(hidden_states, 3))
    _assert_routing_bridge_matches(captured["routing"], expected_routing)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mxfp4_precomputed_routing_accepts_non_power_of_two_topk() -> None:
    selected_experts = torch.tensor(
        [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]],
        dtype=torch.int64,
        device="cuda",
    )
    routing_weights = torch.arange(12, dtype=torch.float32, device="cuda").view(2, 6)

    routing_data, gather_idx, scatter_idx = _routing_from_precomputed(
        selected_experts,
        routing_weights,
        num_experts=256,
    )

    expected_order = torch.argsort(selected_experts.reshape(-1), stable=True).to(torch.int32)
    expected_inverse = torch.argsort(expected_order, stable=True).to(torch.int32)
    expected_hist = torch.zeros(256, dtype=torch.int32, device="cuda")
    expected_hist[:12] = 1

    assert routing_data.n_expts_act == 6
    assert routing_data.n_expts_tot == 256
    torch.testing.assert_close(routing_data.expt_hist, expected_hist)
    torch.testing.assert_close(gather_idx.src_indx, expected_order)
    torch.testing.assert_close(gather_idx.dst_indx, expected_inverse)
    torch.testing.assert_close(scatter_idx.src_indx, expected_inverse)
    torch.testing.assert_close(scatter_idx.dst_indx, expected_order)
    torch.testing.assert_close(
        routing_data.gate_scal,
        routing_weights.reshape(-1)[expected_order.to(torch.int64)],
    )

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_routing_data, captured_gather_idx, captured_scatter_idx = (
            _routing_from_precomputed(
                selected_experts,
                routing_weights,
                num_experts=256,
            )
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured_routing_data.expt_hist, expected_hist)
    torch.testing.assert_close(captured_gather_idx.src_indx, expected_order)
    torch.testing.assert_close(captured_gather_idx.dst_indx, expected_inverse)
    torch.testing.assert_close(captured_scatter_idx.src_indx, expected_inverse)
    torch.testing.assert_close(captured_scatter_idx.dst_indx, expected_order)
    torch.testing.assert_close(
        captured_routing_data.gate_scal,
        routing_weights.reshape(-1)[expected_order.to(torch.int64)],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_route_metadata_custom_op_replays_in_cuda_graph_32_topk6() -> None:
    selected_experts = torch.tensor(
        [
            [31, 0, 17, 17, 3, 31],
            [8, 3, 0, 24, 24, 17],
            [31, 8, 8, 3, 0, 24],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    routing_weights = torch.arange(1, 19, dtype=torch.float32, device="cuda").view(3, 6).div(19.0)

    expected = torch.ops.auto_deploy.torch_deepseek_v4_mxfp4_route_metadata.default(
        selected_experts,
        routing_weights,
        32,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = torch.ops.auto_deploy.torch_deepseek_v4_mxfp4_route_metadata.default(
            selected_experts,
            routing_weights,
            32,
        )
    graph.replay()
    torch.cuda.synchronize()

    for actual, expected_tensor in zip(captured, expected):
        torch.testing.assert_close(actual, expected_tensor)


class _FromRoutingExportModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        routed_w1, routed_w2, routed_w3 = _stacked_weights(
            num_experts=2, hidden_size=3, intermediate_size=4, dtype=torch.float32
        )
        self.routed_w1 = nn.Parameter(routed_w1, requires_grad=False)
        self.routed_w2 = nn.Parameter(routed_w2, requires_grad=False)
        self.routed_w3 = nn.Parameter(routed_w3, requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default(
            hidden_states,
            selected_experts,
            routing_weights,
            self.routed_w1,
            self.routed_w2,
            self.routed_w3,
            None,
            None,
            None,
            10.0,
            "moe",
        )


def test_from_routing_custom_op_exports() -> None:
    module = _FromRoutingExportModule()
    hidden_states = torch.randn(3, 3, dtype=torch.float32)
    selected_experts = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int64)
    routing_weights = torch.full((3, 2), 0.5, dtype=torch.float32)

    exported = torch.export.export(module, (hidden_states, selected_experts, routing_weights))
    targets = [node.target for node in exported.graph_module.graph.nodes]

    assert torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default in targets


def _uint8_pattern(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    values = torch.arange(int(torch.tensor(shape).prod().item()), dtype=torch.int64)
    return values.reshape(shape).add(offset).remainder(251).to(torch.uint8)


class _MXFP4FromRoutingShardingModule(nn.Module):
    def __init__(self, num_experts: int = 4) -> None:
        super().__init__()
        self.ffn = nn.Module()
        self.ffn.gate_up_blocks = nn.Parameter(
            _uint8_pattern((num_experts, 8, 2, 16), 1),
            requires_grad=False,
        )
        self.ffn.gate_up_bias = nn.Parameter(
            torch.arange(num_experts * 8, dtype=torch.float32).reshape(num_experts, 8),
            requires_grad=False,
        )
        self.ffn.gate_up_scales = nn.Parameter(
            _uint8_pattern((num_experts, 8, 2), 11),
            requires_grad=False,
        )
        self.ffn.down_blocks = nn.Parameter(
            _uint8_pattern((num_experts, 4, 2, 16), 23),
            requires_grad=False,
        )
        self.ffn.down_bias = nn.Parameter(
            torch.arange(num_experts * 4, dtype=torch.float32).reshape(num_experts, 4).add(100),
            requires_grad=False,
        )
        self.ffn.down_scales = nn.Parameter(
            _uint8_pattern((num_experts, 4, 2), 37),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing.default(
            hidden_states,
            selected_experts,
            routing_weights,
            self.ffn.gate_up_blocks,
            self.ffn.gate_up_bias,
            self.ffn.gate_up_scales,
            1.0,
            10.0,
            self.ffn.down_blocks,
            self.ffn.down_bias,
            self.ffn.down_scales,
            "moe",
        )


def test_triton_mxfp4_from_routing_custom_op_exports() -> None:
    module = _MXFP4FromRoutingShardingModule(num_experts=2)
    hidden_states = torch.randn(3, 4, dtype=torch.bfloat16)
    selected_experts = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.int64)
    routing_weights = torch.full((3, 2), 0.5, dtype=torch.float32)

    exported = torch.export.export(module, (hidden_states, selected_experts, routing_weights))
    targets = [node.target for node in exported.graph_module.graph.nodes]

    assert torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing.default in targets
    assert torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default not in targets


def _trace_mxfp4_from_routing_sharding_module(
    num_experts: int = 4,
    top_k: int = 2,
) -> torch.fx.GraphModule:
    gm = torch.fx.symbolic_trace(_MXFP4FromRoutingShardingModule(num_experts))
    params_and_buffers = dict(gm.named_parameters())
    params_and_buffers.update(dict(gm.named_buffers()))
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.target == "hidden_states":
                node.meta["val"] = torch.empty((3, 4), dtype=torch.bfloat16)
            elif node.target == "selected_experts":
                node.meta["val"] = torch.empty((3, top_k), dtype=torch.int64)
            elif node.target == "routing_weights":
                node.meta["val"] = torch.empty((3, top_k), dtype=torch.float32)
        elif node.op == "get_attr" and node.target in params_and_buffers:
            node.meta["val"] = params_and_buffers[node.target].detach()
        elif is_op(
            node,
            torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing,
        ):
            node.meta["val"] = torch.empty((3, 4), dtype=torch.bfloat16)
    return gm


def _meta_parameter(shape: tuple[int, ...], dtype: torch.dtype) -> nn.Parameter:
    return nn.Parameter(torch.empty(shape, dtype=dtype, device="meta"), requires_grad=False)


class _LoweredDeepSeekV4EP8ShapeRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].ffn = nn.Module()
        ffn = self.layers[0].ffn
        ffn.mxfp4_gate_up_blocks = _meta_parameter((256, 4096, 128, 16), torch.uint8)
        ffn.register_buffer(
            "mxfp4_gate_up_bias",
            torch.empty((256, 4096), dtype=torch.float32, device="meta"),
            persistent=False,
        )
        ffn.mxfp4_gate_up_scales = _meta_parameter((256, 4096, 128), torch.uint8)
        ffn.mxfp4_down_blocks = _meta_parameter((256, 4096, 64, 16), torch.uint8)
        ffn.register_buffer(
            "mxfp4_down_bias",
            torch.empty((256, 4096), dtype=torch.float32, device="meta"),
            persistent=False,
        )
        ffn.mxfp4_down_scales = _meta_parameter((256, 4096, 64), torch.uint8)
        ffn.shared_experts = nn.Module()
        for proj, weight_shape, scale_shape in (
            ("w1", (2048, 4096), (16, 32)),
            ("w2", (4096, 2048), (32, 16)),
            ("w3", (2048, 4096), (16, 32)),
        ):
            linear = nn.Module()
            linear.weight = _meta_parameter(weight_shape, torch.float8_e4m3fn)
            linear.register_buffer(
                "weight_scale_inv",
                torch.empty(scale_shape, dtype=torch.float32, device="meta"),
            )
            setattr(ffn.shared_experts, proj, linear)


def _nested_attr(module: nn.Module, name: str) -> torch.Tensor:
    value: object = module
    for part in name.split("."):
        value = getattr(value, part)
    assert isinstance(value, torch.Tensor)
    return value


def _get_attr_with_meta(graph: torch.fx.Graph, root: nn.Module, name: str) -> torch.fx.Node:
    node = graph.create_node("get_attr", name)
    node.meta["val"] = _nested_attr(root, name).detach()
    return node


def _call_fp8_linear_with_meta(
    graph: torch.fx.Graph,
    input_node: torch.fx.Node,
    input_shape: tuple[int, int],
    weight_node: torch.fx.Node,
    scale_node: torch.fx.Node,
    tp_mode: str,
) -> torch.fx.Node:
    output_features = int(weight_node.meta["val"].shape[0])
    if tp_mode == "rowwise":
        output_features = int(weight_node.meta["val"].shape[0])
    node = graph.call_function(
        torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear.default,
        args=(input_node, weight_node, None, [], [scale_node], [], [], tp_mode, None, 1, "moe"),
    )
    node.meta["val"] = torch.empty(
        (*input_shape[:-1], output_features),
        dtype=torch.bfloat16,
        device="meta",
    )
    return node


def _make_lowered_deepseek_v4_ep8_shape_graph() -> torch.fx.GraphModule:
    root = _LoweredDeepSeekV4EP8ShapeRoot()
    graph = torch.fx.Graph()
    hidden_states = graph.placeholder("hidden_states")
    hidden_states.meta["val"] = torch.empty((2, 4096), dtype=torch.bfloat16, device="meta")
    selected_experts = graph.placeholder("selected_experts")
    selected_experts.meta["val"] = torch.empty((2, 6), dtype=torch.int64, device="meta")
    routing_weights = graph.placeholder("routing_weights")
    routing_weights.meta["val"] = torch.empty((2, 6), dtype=torch.float32, device="meta")

    gate_up_blocks = _get_attr_with_meta(graph, root, "layers.0.ffn.mxfp4_gate_up_blocks")
    gate_up_bias = _get_attr_with_meta(graph, root, "layers.0.ffn.mxfp4_gate_up_bias")
    gate_up_scales = _get_attr_with_meta(graph, root, "layers.0.ffn.mxfp4_gate_up_scales")
    down_blocks = _get_attr_with_meta(graph, root, "layers.0.ffn.mxfp4_down_blocks")
    down_bias = _get_attr_with_meta(graph, root, "layers.0.ffn.mxfp4_down_bias")
    down_scales = _get_attr_with_meta(graph, root, "layers.0.ffn.mxfp4_down_scales")
    routed = graph.call_function(
        torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing.default,
        args=(
            hidden_states,
            selected_experts,
            routing_weights,
            gate_up_blocks,
            gate_up_bias,
            gate_up_scales,
            1.0,
            10.0,
            down_blocks,
            down_bias,
            down_scales,
            "moe",
        ),
    )
    routed.meta["val"] = torch.empty((2, 4096), dtype=torch.bfloat16, device="meta")

    shared_w1 = _get_attr_with_meta(graph, root, "layers.0.ffn.shared_experts.w1.weight")
    shared_w1_scale = _get_attr_with_meta(
        graph,
        root,
        "layers.0.ffn.shared_experts.w1.weight_scale_inv",
    )
    shared_w2 = _get_attr_with_meta(graph, root, "layers.0.ffn.shared_experts.w2.weight")
    shared_w2_scale = _get_attr_with_meta(
        graph,
        root,
        "layers.0.ffn.shared_experts.w2.weight_scale_inv",
    )
    shared_w3 = _get_attr_with_meta(graph, root, "layers.0.ffn.shared_experts.w3.weight")
    shared_w3_scale = _get_attr_with_meta(
        graph,
        root,
        "layers.0.ffn.shared_experts.w3.weight_scale_inv",
    )
    shared_gate = _call_fp8_linear_with_meta(
        graph, hidden_states, (2, 4096), shared_w1, shared_w1_scale, "colwise"
    )
    shared_up = _call_fp8_linear_with_meta(
        graph, hidden_states, (2, 4096), shared_w3, shared_w3_scale, "colwise"
    )
    shared_act = graph.call_function(torch.ops.aten.silu.default, args=(shared_gate,))
    shared_act.meta["val"] = torch.empty((2, 2048), dtype=torch.bfloat16, device="meta")
    shared_hidden = graph.call_function(torch.ops.aten.mul.Tensor, args=(shared_act, shared_up))
    shared_hidden.meta["val"] = torch.empty((2, 2048), dtype=torch.bfloat16, device="meta")
    shared = _call_fp8_linear_with_meta(
        graph, shared_hidden, (2, 2048), shared_w2, shared_w2_scale, "rowwise"
    )
    shared_reduced = graph.call_function(
        torch.ops.auto_deploy.all_reduce.default,
        args=(shared,),
        kwargs={"layer_type": "moe"},
    )
    shared_reduced.meta["val"] = torch.empty((2, 4096), dtype=torch.bfloat16, device="meta")
    output = graph.call_function(torch.ops.aten.add.Tensor, args=(routed, shared_reduced))
    output.meta["val"] = torch.empty((2, 4096), dtype=torch.bfloat16, device="meta")
    graph.output(output)
    return torch.fx.GraphModule(root, graph)


def _shift_state_value(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor.to(torch.int64).add(19).remainder(251).to(torch.uint8)
    return tensor + 19


def _is_all_reduce_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and "all_reduce" in str(node.target)


def _assert_mxfp4_node_uses_local_route_metadata(
    moe_node: torch.fx.Node,
    selected_local: torch.fx.Node,
    routing_local: torch.fx.Node,
    num_local_experts: int,
) -> None:
    assert is_op(
        moe_node,
        torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing,
    )
    metadata_nodes = extract_op_args(moe_node, *_ROUTE_METADATA_ARG_NAMES)
    assert len(metadata_nodes) == len(_ROUTE_METADATA_ARG_NAMES)
    assert all(isinstance(node, torch.fx.Node) for node in metadata_nodes)
    route_metadata = metadata_nodes[0].args[0]
    assert all(node.args[0] is route_metadata for node in metadata_nodes)
    assert is_op(route_metadata, torch.ops.auto_deploy.torch_deepseek_v4_mxfp4_route_metadata)
    assert route_metadata.args == (selected_local, routing_local, num_local_experts)


def test_deepseek_v4_mxfp4_from_routing_shards_experts_and_localizes_ids() -> None:
    gm = _trace_mxfp4_from_routing_sharding_module()
    full_state = {name: tensor.detach().clone() for name, tensor in gm.named_parameters()}

    transformed, info = _run_ir_sharding(gm, rank=1, world_size=2)

    assert info.num_matches == 1
    for name, full_tensor in full_state.items():
        local_tensor = transformed.get_parameter(name)
        assert local_tensor.shape[0] == 2
        torch.testing.assert_close(local_tensor, full_tensor[2:4], rtol=0, atol=0)

    moe_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing,
        )
    )
    selected_local = moe_node.args[1]
    routing_local = moe_node.args[2]
    assert selected_local.target == operator.mul
    assert routing_local.target == operator.mul
    assert selected_local.args[1] is routing_local.args[1]
    assert selected_local.args[0].target == operator.sub
    assert selected_local.args[0].args[1] == 2
    assert selected_local.args[1].target == torch.logical_and
    _assert_mxfp4_node_uses_local_route_metadata(
        moe_node,
        selected_local,
        routing_local,
        num_local_experts=2,
    )
    assert any(
        node.args and node.args[0] is moe_node and _is_all_reduce_node(node)
        for node in transformed.graph.nodes
    )

    shifted_full_state = {name: _shift_state_value(tensor) for name, tensor in full_state.items()}
    load_result = transformed.load_state_dict(shifted_full_state, strict=False)

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    for name, full_tensor in shifted_full_state.items():
        torch.testing.assert_close(
            transformed.get_parameter(name),
            full_tensor[2:4],
            rtol=0,
            atol=0,
        )


def test_deepseek_v4_mxfp4_from_routing_ep8_localizes_rank7_global_ids() -> None:
    gm = _trace_mxfp4_from_routing_sharding_module(num_experts=16)
    full_state = {name: tensor.detach().clone() for name, tensor in gm.named_parameters()}

    transformed, info = _run_ir_sharding(gm, rank=7, world_size=8)

    assert info.num_matches == 1
    for name, full_tensor in full_state.items():
        local_tensor = transformed.get_parameter(name)
        assert local_tensor.shape[0] == 2
        torch.testing.assert_close(local_tensor, full_tensor[14:16], rtol=0, atol=0)

    moe_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing,
        )
    )
    selected_local = moe_node.args[1]
    routing_local = moe_node.args[2]
    assert selected_local.target == operator.mul
    assert routing_local.target == operator.mul
    assert selected_local.args[1] is routing_local.args[1]
    local_ids = selected_local.args[0]
    assert local_ids.target == operator.sub
    assert local_ids.args[1] == 14
    rank_mask = selected_local.args[1]
    ge_lower, lt_upper = rank_mask.args
    assert rank_mask.target == torch.logical_and
    assert ge_lower.target == torch.ge
    assert ge_lower.args[1] == 14
    assert lt_upper.target == torch.lt
    assert lt_upper.args[1] == 16
    _assert_mxfp4_node_uses_local_route_metadata(
        moe_node,
        selected_local,
        routing_local,
        num_local_experts=2,
    )


def test_deepseek_v4_mxfp4_from_routing_ep8_uses_32_local_experts_topk6_metadata() -> None:
    gm = _trace_mxfp4_from_routing_sharding_module(num_experts=256, top_k=6)
    full_state = {name: tensor.detach().clone() for name, tensor in gm.named_parameters()}

    transformed, info = _run_ir_sharding(gm, rank=5, world_size=8)

    assert info.num_matches == 1
    for name, full_tensor in full_state.items():
        local_tensor = transformed.get_parameter(name)
        assert local_tensor.shape[0] == 32
        torch.testing.assert_close(local_tensor, full_tensor[160:192], rtol=0, atol=0)

    moe_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing,
        )
    )
    selected_local = moe_node.args[1]
    routing_local = moe_node.args[2]
    assert moe_node.args[6] == 1.0
    assert moe_node.args[7] == 10.0
    assert selected_local.args[0].target == operator.sub
    assert selected_local.args[0].args[1] == 160
    _assert_mxfp4_node_uses_local_route_metadata(
        moe_node,
        selected_local,
        routing_local,
        num_local_experts=32,
    )
    assert any(
        node.args and node.args[0] is moe_node and _is_all_reduce_node(node)
        for node in transformed.graph.nodes
    )


def test_lowered_deepseek_v4_ep8_exact_local_shapes_metadata_and_collectives() -> None:
    gm = _make_lowered_deepseek_v4_ep8_shape_graph()

    transformed, info = _run_ir_sharding(gm, rank=6, world_size=8)

    assert info.num_matches == 5
    ffn = transformed.get_submodule("layers.0.ffn")
    assert tuple(ffn.mxfp4_gate_up_blocks.shape) == (32, 4096, 128, 16)
    assert tuple(ffn.mxfp4_gate_up_bias.shape) == (32, 4096)
    assert tuple(ffn.mxfp4_gate_up_scales.shape) == (32, 4096, 128)
    assert tuple(ffn.mxfp4_down_blocks.shape) == (32, 4096, 64, 16)
    assert tuple(ffn.mxfp4_down_bias.shape) == (32, 4096)
    assert tuple(ffn.mxfp4_down_scales.shape) == (32, 4096, 64)
    assert tuple(ffn.shared_experts.w1.weight.shape) == (256, 4096)
    assert tuple(ffn.shared_experts.w2.weight.shape) == (4096, 256)
    assert tuple(ffn.shared_experts.w3.weight.shape) == (256, 4096)

    moe_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing,
        )
    )
    selected_local = moe_node.args[1]
    routing_local = moe_node.args[2]
    assert moe_node.args[6] == 1.0
    assert moe_node.args[7] == 10.0
    assert selected_local.args[0].target == operator.sub
    assert selected_local.args[0].args[1] == 192
    _assert_mxfp4_node_uses_local_route_metadata(
        moe_node,
        selected_local,
        routing_local,
        num_local_experts=32,
    )

    all_reduce_nodes = [node for node in transformed.graph.nodes if _is_all_reduce_node(node)]
    shared_w2_node = _shared_fp8_linear_nodes(transformed)["layers.0.ffn.shared_experts.w2.weight"]
    routed_reduces = [node for node in all_reduce_nodes if node.args[0] is moe_node]
    shared_reduces = [node for node in all_reduce_nodes if node.args[0] is shared_w2_node]
    assert len(routed_reduces) == 1
    assert len(shared_reduces) == 1
    add_node = next(
        node for node in transformed.graph.nodes if node.target == torch.ops.aten.add.Tensor
    )
    assert add_node.args == (routed_reduces[0], shared_reduces[0])
    assert not any(
        is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing)
        or is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_moe)
        for node in transformed.graph.nodes
    )


class _CanonicalMoEModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router_weight = nn.Parameter(torch.randn(3, 2), requires_grad=False)
        self.router_bias = nn.Parameter(torch.zeros(3), requires_grad=False)
        routed_w1, routed_w2, routed_w3 = _stacked_weights(
            num_experts=3, hidden_size=2, intermediate_size=4, dtype=torch.float32
        )
        self.routed_w1 = nn.Parameter(routed_w1, requires_grad=False)
        self.routed_w2 = nn.Parameter(routed_w2, requires_grad=False)
        self.routed_w3 = nn.Parameter(routed_w3, requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_deepseek_v4_moe.default(
            hidden_states,
            None,
            self.router_weight,
            self.router_bias,
            None,
            self.routed_w1,
            self.routed_w2,
            self.routed_w3,
            None,
            None,
            None,
            2,
            1.0,
            10.0,
            False,
            "moe",
        )


def _trace_canonical_moe() -> torch.fx.GraphModule:
    return torch.fx.symbolic_trace(_CanonicalMoEModule())


def test_lowering_bridge_rejects_non_layered_graph_with_clear_error() -> None:
    gm = _trace_canonical_moe()
    transform = DeepSeekV4MoELowering.from_kwargs(stage=Stages.POST_LOAD_FUSION)

    with pytest.raises(
        DeepSeekV4MoELoweringError,
        match="requires routed w1 weights.*stack of per-expert get_attr",
    ):
        transform._apply(gm, None, None, SharedConfig())


def test_lowering_skeleton_can_opt_into_reference_graph_for_tests() -> None:
    module = _CanonicalMoEModule()
    gm = torch.fx.symbolic_trace(module)
    transform = DeepSeekV4MoELowering.from_kwargs(
        stage=Stages.POST_LOAD_FUSION, allow_reference_lowering=True
    )

    lowered, info = transform._apply(gm, None, None, SharedConfig())
    targets = [node.target for node in lowered.graph.nodes]

    assert info.num_matches == 1
    assert torch.ops.auto_deploy.torch_deepseek_v4_router.default in targets
    assert torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default in targets
    assert not any(
        is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_moe) for node in lowered.graph.nodes
    )

    hidden_states = torch.randn(4, 2, dtype=torch.float32)
    torch.testing.assert_close(lowered(hidden_states), module(hidden_states))


@pytest.mark.parametrize(
    ("num_hash_layers", "expected_is_hash_layer"),
    [(0, False), (1, True)],
)
def test_lowering_bridge_production_uses_triton_router_source_op(
    num_hash_layers: int,
    expected_is_hash_layer: bool,
) -> None:
    lowered, info = _lower_layered_moe_model(num_hash_layers=num_hash_layers)

    assert info.num_matches == 1
    torch_router_nodes = [
        node
        for node in lowered.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_router)
    ]
    triton_router_nodes = [
        node
        for node in lowered.graph.nodes
        if is_op(node, torch.ops.auto_deploy.triton_deepseek_v4_router)
    ]
    assert torch_router_nodes == []
    assert len(triton_router_nodes) == 1

    router_node = triton_router_nodes[0]
    assert router_node.args[7] == expected_is_hash_layer
    if expected_is_hash_layer:
        assert router_node.args[3] is None
        assert isinstance(router_node.args[4], torch.fx.Node)
    else:
        assert isinstance(router_node.args[3], torch.fx.Node)
        assert router_node.args[4] is None
    mxfp4_node = next(
        node
        for node in lowered.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing,
        )
    )
    selected_experts = mxfp4_node.args[1]
    routing_weights = mxfp4_node.args[2]

    assert selected_experts.target == operator.getitem
    assert selected_experts.args == (router_node, 0)
    assert routing_weights.target == operator.getitem
    assert routing_weights.args == (router_node, 1)
    assert torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default not in [
        node.target for node in lowered.graph.nodes
    ]


def test_lowering_bridge_ep8_production_uses_triton_router_packed_mxfp4_and_collectives() -> None:
    lowered, lowering_info = _lower_layered_moe_model(
        n_routed_experts=256,
        num_experts_per_tok=6,
        num_hash_layers=1,
    )

    transformed, sharding_info = _run_ir_sharding(lowered, rank=6, world_size=8)

    assert lowering_info.num_matches == 1
    assert sharding_info.num_matches >= 5
    assert not any(
        is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_router)
        or is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing)
        or is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_moe)
        for node in transformed.graph.nodes
    )

    router_nodes = [
        node
        for node in transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.triton_deepseek_v4_router)
    ]
    assert len(router_nodes) == 1
    router_node = router_nodes[0]
    assert router_node.args[7] is True

    mxfp4_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing,
        )
    )
    selected_local = mxfp4_node.args[1]
    routing_local = mxfp4_node.args[2]
    selected_global = selected_local.args[0].args[0]
    routing_global = routing_local.args[0]

    assert selected_local.args[0].target == operator.sub
    assert selected_local.args[0].args[1] == 192
    assert selected_global.target == operator.getitem
    assert selected_global.args == (router_node, 0)
    assert routing_global.target == operator.getitem
    assert routing_global.args == (router_node, 1)
    _assert_mxfp4_node_uses_local_route_metadata(
        mxfp4_node,
        selected_local,
        routing_local,
        num_local_experts=32,
    )

    ffn = transformed.get_submodule("layers.0.ffn")
    assert tuple(ffn.mxfp4_gate_up_blocks.shape) == (32, 64, 2, 16)
    assert tuple(ffn.mxfp4_gate_up_bias.shape) == (32, 64)
    assert tuple(ffn.mxfp4_gate_up_scales.shape) == (32, 64, 2)
    assert tuple(ffn.mxfp4_down_blocks.shape) == (32, 64, 1, 16)
    assert tuple(ffn.mxfp4_down_bias.shape) == (32, 64)
    assert tuple(ffn.mxfp4_down_scales.shape) == (32, 64, 1)

    fp8_nodes = _shared_fp8_linear_nodes(transformed)
    expected_modes = {
        "layers.0.ffn.shared_experts.w1.weight": "colwise",
        "layers.0.ffn.shared_experts.w2.weight": "rowwise",
        "layers.0.ffn.shared_experts.w3.weight": "colwise",
    }
    assert {
        weight_name: extract_op_args(node, "tp_mode")[0]
        for weight_name, node in fp8_nodes.items()
        if weight_name in expected_modes
    } == expected_modes

    all_reduce_nodes = [node for node in transformed.graph.nodes if _is_all_reduce_node(node)]
    shared_w2_node = fp8_nodes["layers.0.ffn.shared_experts.w2.weight"]
    routed_reduces = [node for node in all_reduce_nodes if node.args[0] is mxfp4_node]
    shared_reduces = [node for node in all_reduce_nodes if node.args[0] is shared_w2_node]
    assert len(routed_reduces) == 1
    assert len(shared_reduces) == 1
    assert any(
        node.target == torch.ops.aten.add.Tensor
        and node.args == (routed_reduces[0], shared_reduces[0])
        for node in transformed.graph.nodes
    )


class _LayeredMoEBlock(nn.Module):
    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.ffn = DeepseekV4MoE(config, layer_idx=0)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return self.ffn(hidden_states, input_ids)


class _LayeredMoEModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        moe_intermediate_size: int = 32,
        num_hash_layers: int = 0,
        n_routed_experts: int = 2,
        num_experts_per_tok: int = 1,
    ) -> None:
        super().__init__()
        config = DeepseekV4Config(
            vocab_size=16,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            n_routed_experts=n_routed_experts,
            n_shared_experts=1,
            num_experts_per_tok=num_experts_per_tok,
            num_hash_layers=num_hash_layers,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.0,
            swiglu_limit=10.0,
        )
        self.layers = nn.ModuleList([_LayeredMoEBlock(config)])

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return self.layers[0](hidden_states, input_ids)


def _lower_layered_moe_model(
    hidden_size: int = 64,
    moe_intermediate_size: int = 32,
    num_hash_layers: int = 0,
    n_routed_experts: int = 2,
    num_experts_per_tok: int = 1,
) -> tuple[torch.fx.GraphModule, object]:
    model = _LayeredMoEModel(
        hidden_size,
        moe_intermediate_size,
        num_hash_layers,
        n_routed_experts,
        num_experts_per_tok,
    )
    gm = torch_export_to_gm(
        model,
        args=(torch.randn(1, 2, hidden_size), torch.ones(1, 2, dtype=torch.long)),
    )
    transform = DeepSeekV4MoELowering.from_kwargs(stage=Stages.PATTERN_MATCHER)
    return transform._apply(gm, None, None, SharedConfig())


def _shared_fp8_linear_nodes(gm: torch.fx.GraphModule) -> dict[str, torch.fx.Node]:
    nodes = {}
    for node in gm.graph.nodes:
        if not is_op(node, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear):
            continue
        [weight] = extract_op_args(node, "weight_quantized")
        if isinstance(weight, torch.fx.Node) and weight.op == "get_attr":
            nodes[str(weight.target)] = node
    return nodes


def _fp8_pattern(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    values = torch.arange(int(torch.tensor(shape).prod().item()), dtype=torch.float32)
    values = values.add(offset).remainder(29).sub(14).div(16).reshape(shape)
    return values.to(torch.float8_e4m3fn)


def test_lowering_bridge_shared_fp8_linears_have_tp_hints_and_all_reduce() -> None:
    lowered, info = _lower_layered_moe_model()

    assert info.num_matches == 1
    fp8_nodes = _shared_fp8_linear_nodes(lowered)
    expected_modes = {
        "layers.0.ffn.shared_experts.w1.weight": "colwise",
        "layers.0.ffn.shared_experts.w2.weight": "rowwise",
        "layers.0.ffn.shared_experts.w3.weight": "colwise",
    }
    actual_modes = {
        weight_name: extract_op_args(node, "tp_mode")[0]
        for weight_name, node in fp8_nodes.items()
        if weight_name in expected_modes
    }

    assert actual_modes == expected_modes

    shared_w2_node = fp8_nodes["layers.0.ffn.shared_experts.w2.weight"]
    shared_all_reduces = [
        node
        for node in lowered.graph.nodes
        if is_op(node, torch.ops.auto_deploy.all_reduce) and node.args[0] is shared_w2_node
    ]
    assert len(shared_all_reduces) == 1
    assert extract_op_args(shared_all_reduces[0], "layer_type") == ["moe"]


def test_lowering_bridge_shared_fp8_linears_shard_weights_and_scales() -> None:
    lowered, _ = _lower_layered_moe_model(hidden_size=256, moe_intermediate_size=256)

    transformed, info = _run_ir_sharding(lowered, rank=1, world_size=2)

    assert info.num_matches >= 4
    shared_w1 = transformed.get_submodule("layers.0.ffn.shared_experts.w1")
    shared_w2 = transformed.get_submodule("layers.0.ffn.shared_experts.w2")
    shared_w3 = transformed.get_submodule("layers.0.ffn.shared_experts.w3")
    assert shared_w1.weight.shape == (128, 256)
    assert shared_w2.weight.shape == (256, 128)
    assert shared_w3.weight.shape == (128, 256)
    assert shared_w1.weight_scale_inv.shape == (1, 2)
    assert shared_w2.weight_scale_inv.shape == (2, 1)
    assert shared_w3.weight_scale_inv.shape == (1, 2)
    all_reduce_nodes = [node for node in transformed.graph.nodes if _is_all_reduce_node(node)]
    assert all_reduce_nodes
    assert all(node.kwargs == {} for node in all_reduce_nodes)
    assert all(len(node.args) == 2 for node in all_reduce_nodes)
    mxfp4_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing,
        )
    )
    shared_w2_node = _shared_fp8_linear_nodes(transformed)["layers.0.ffn.shared_experts.w2.weight"]
    assert any(node.args[0] is mxfp4_node for node in all_reduce_nodes)
    assert any(node.args[0] is shared_w2_node for node in all_reduce_nodes)

    full_state = {
        "layers.0.ffn.shared_experts.w1.weight": _fp8_pattern((256, 256), 1),
        "layers.0.ffn.shared_experts.w1.weight_scale_inv": torch.arange(
            4, dtype=torch.float32
        ).reshape(2, 2),
        "layers.0.ffn.shared_experts.w2.weight": _fp8_pattern((256, 256), 3),
        "layers.0.ffn.shared_experts.w2.weight_scale_inv": torch.arange(4, dtype=torch.float32)
        .reshape(2, 2)
        .add(10),
        "layers.0.ffn.shared_experts.w3.weight": _fp8_pattern((256, 256), 5),
        "layers.0.ffn.shared_experts.w3.weight_scale_inv": torch.arange(4, dtype=torch.float32)
        .reshape(2, 2)
        .add(20),
    }
    load_result = transformed.load_state_dict(full_state, strict=False)

    assert load_result.unexpected_keys == []
    torch.testing.assert_close(
        shared_w1.weight.float(),
        full_state["layers.0.ffn.shared_experts.w1.weight"][128:256].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        shared_w2.weight.float(),
        full_state["layers.0.ffn.shared_experts.w2.weight"][:, 128:256].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        shared_w3.weight.float(),
        full_state["layers.0.ffn.shared_experts.w3.weight"][128:256].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        shared_w1.weight_scale_inv,
        full_state["layers.0.ffn.shared_experts.w1.weight_scale_inv"][1:2],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        shared_w2.weight_scale_inv,
        full_state["layers.0.ffn.shared_experts.w2.weight_scale_inv"][:, 1:2],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        shared_w3.weight_scale_inv,
        full_state["layers.0.ffn.shared_experts.w3.weight_scale_inv"][1:2],
        rtol=0,
        atol=0,
    )


def _pack_fp4(logical_values: torch.Tensor) -> torch.Tensor:
    low = logical_values[..., 0::2] & 0x0F
    high = (logical_values[..., 1::2] & 0x0F) << 4
    return (low | high).contiguous().view(torch.int8)


def _logical_fp4(rows: int, cols: int, offset: int) -> torch.Tensor:
    return (torch.arange(rows * cols, dtype=torch.uint8).reshape(rows, cols) + offset) & 0x0F


def _packed_checkpoint_state() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    hidden_size = 64
    intermediate_size = 32
    for expert_idx in range(2):
        prefix = f"layers.0.ffn.experts.{expert_idx}"
        state[f"{prefix}.w1.weight"] = _pack_fp4(
            _logical_fp4(intermediate_size, hidden_size, 1 + expert_idx)
        )
        state[f"{prefix}.w2.weight"] = _pack_fp4(
            _logical_fp4(hidden_size, intermediate_size, 5 + expert_idx)
        )
        state[f"{prefix}.w3.weight"] = _pack_fp4(
            _logical_fp4(intermediate_size, hidden_size, 9 + expert_idx)
        )
        state[f"{prefix}.w1.scale"] = torch.full(
            (intermediate_size, hidden_size // 32), 17 + expert_idx, dtype=torch.uint8
        )
        state[f"{prefix}.w2.scale"] = torch.full(
            (hidden_size, intermediate_size // 32), 29 + expert_idx, dtype=torch.uint8
        )
        state[f"{prefix}.w3.scale"] = torch.full(
            (intermediate_size, hidden_size // 32), 43 + expert_idx, dtype=torch.uint8
        )

    for proj, shape in (("w1", (32, 64)), ("w2", (64, 32)), ("w3", (32, 64))):
        state[f"layers.0.ffn.shared_experts.{proj}.weight"] = torch.zeros(
            shape, dtype=torch.float8_e4m3fn
        )
        state[f"layers.0.ffn.shared_experts.{proj}.scale"] = torch.ones((1, 1), dtype=torch.float32)
    return state


def test_mxfp4_loader_matches_real_deepseek_v4_checkpoint_layout() -> None:
    checkpoint_path = _deepseek_v4_real_checkpoint_path()
    hidden_size = 4096
    intermediate_size = 2048

    for layer_idx, expert_indices in ((0, (0, 1, 255)), (42, (0, 255))):
        tensor_names = [
            f"layers.{layer_idx}.ffn.experts.{expert_idx}.{proj}.{suffix}"
            for expert_idx in expert_indices
            for proj in ("w1", "w2", "w3")
            for suffix in ("weight", "scale")
        ]
        state_dict = _load_real_checkpoint_tensors(checkpoint_path, tensor_names)

        layout = load_deepseek_v4_mxfp4_experts(
            state_dict,
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            expert_indices=expert_indices,
        )

        _assert_layout_matches_real_checkpoint(
            layout,
            state_dict,
            layer_idx=layer_idx,
            expert_indices=expert_indices,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )


def test_lowering_bridge_loads_real_deepseek_v4_mxfp4_checkpoint_block() -> None:
    checkpoint_path = _deepseek_v4_real_checkpoint_path()
    hidden_size = 4096
    intermediate_size = 2048
    layer_idx = 0
    expert_indices = (0, 1)
    tensor_names = [
        f"layers.{layer_idx}.ffn.experts.{expert_idx}.{proj}.{suffix}"
        for expert_idx in expert_indices
        for proj in ("w1", "w2", "w3")
        for suffix in ("weight", "scale")
    ]
    tensor_names.extend(
        f"layers.{layer_idx}.ffn.shared_experts.{proj}.{suffix}"
        for proj in ("w1", "w2", "w3")
        for suffix in ("weight", "scale")
    )
    real_state = _load_real_checkpoint_tensors(checkpoint_path, tensor_names)
    model = _LayeredMoEModel(hidden_size=hidden_size, moe_intermediate_size=intermediate_size)
    gm = torch_export_to_gm(
        model,
        args=(torch.randn(1, 1, hidden_size), torch.ones(1, 1, dtype=torch.long)),
    )
    transform = DeepSeekV4MoELowering.from_kwargs(stage=Stages.PATTERN_MATCHER)

    lowered, info = transform._apply(gm, None, None, SharedConfig())

    assert info.num_matches == 1
    assert not any(
        name.startswith("layers.0.ffn.experts.") for name, _ in lowered.named_parameters()
    )

    state_dict = {
        name: tensor.detach().clone()
        for name, tensor in lowered.state_dict().items()
        if not name.startswith("layers.0.ffn.mxfp4_") and not name.endswith(".weight_scale_inv")
    }
    state_dict.update(real_state)
    load_result = lowered.load_state_dict(state_dict, strict=False)
    expected_layout = load_deepseek_v4_mxfp4_experts(
        real_state,
        layer_idx=layer_idx,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=len(expert_indices),
    )

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    _assert_layout_matches_real_checkpoint(
        expected_layout,
        real_state,
        layer_idx=layer_idx,
        expert_indices=expert_indices,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    torch.testing.assert_close(
        lowered.get_parameter("layers.0.ffn.mxfp4_gate_up_blocks"),
        expected_layout.gate_up_blocks,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        lowered.get_parameter("layers.0.ffn.mxfp4_gate_up_scales"),
        expected_layout.gate_up_scales,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        lowered.get_parameter("layers.0.ffn.mxfp4_down_blocks"),
        expected_layout.down_blocks,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        lowered.get_parameter("layers.0.ffn.mxfp4_down_scales"),
        expected_layout.down_scales,
        rtol=0,
        atol=0,
    )

    for proj in ("w1", "w2", "w3"):
        weight_name = f"layers.0.ffn.shared_experts.{proj}.weight"
        scale_name = f"layers.0.ffn.shared_experts.{proj}.scale"
        scale_buffer_name = f"layers.0.ffn.shared_experts.{proj}.weight_scale_inv"
        torch.testing.assert_close(
            lowered.get_parameter(weight_name).float(),
            real_state[weight_name].float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            lowered.get_buffer(scale_buffer_name),
            e8m0_to_fp32(real_state[scale_name]),
            rtol=0,
            atol=0,
        )


def test_lowering_bridge_loads_packed_mxfp4_checkpoint_without_dense_shape_mismatch() -> None:
    model = _LayeredMoEModel()
    gm = torch_export_to_gm(
        model,
        args=(torch.randn(1, 2, 64), torch.ones(1, 2, dtype=torch.long)),
    )
    transform = DeepSeekV4MoELowering.from_kwargs(stage=Stages.PATTERN_MATCHER)

    lowered, info = transform._apply(gm, None, None, SharedConfig())
    targets = [node.target for node in lowered.graph.nodes]

    assert info.num_matches == 1
    assert torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing.default in targets
    assert torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear.default in targets
    assert torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default not in targets
    assert not any(
        is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_moe) for node in lowered.graph.nodes
    )

    state_dict = {
        name: tensor.detach().clone()
        for name, tensor in lowered.state_dict().items()
        if not name.startswith("layers.0.ffn.mxfp4_") and not name.endswith(".weight_scale_inv")
    }
    state_dict.update(_packed_checkpoint_state())
    load_result = lowered.load_state_dict(state_dict, strict=False)
    expected_layout = load_deepseek_v4_mxfp4_experts(
        _packed_checkpoint_state(),
        layer_idx=0,
        hidden_size=64,
        intermediate_size=32,
        num_experts=2,
    )

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    assert "layers.0.ffn.mxfp4_gate_up_bias" not in lowered.state_dict()
    assert "layers.0.ffn.mxfp4_down_bias" not in lowered.state_dict()
    torch.testing.assert_close(
        lowered.get_buffer("layers.0.ffn.mxfp4_gate_up_bias"),
        torch.zeros((2, 64), dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        lowered.get_buffer("layers.0.ffn.mxfp4_down_bias"),
        torch.zeros((2, 64), dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        lowered.get_parameter("layers.0.ffn.mxfp4_gate_up_blocks"),
        expected_layout.gate_up_blocks,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        lowered.get_parameter("layers.0.ffn.mxfp4_down_scales"),
        expected_layout.down_scales,
        rtol=0,
        atol=0,
    )
    assert (
        lowered.get_parameter("layers.0.ffn.shared_experts.w1.weight").dtype == torch.float8_e4m3fn
    )
    torch.testing.assert_close(
        lowered.get_buffer("layers.0.ffn.shared_experts.w1.weight_scale_inv"),
        torch.ones((1, 1), dtype=torch.float32),
        rtol=0,
        atol=0,
    )
