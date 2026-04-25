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

from collections.abc import Generator

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import mxfp4_moe


class _FakeSwizzledTensor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.shape: torch.Size | None = None


@pytest.fixture(autouse=True)
def _clear_mxfp4_cache() -> Generator[None, None, None]:
    mxfp4_moe._clear_mxfp4_weights_scales_cache()
    yield
    mxfp4_moe._clear_mxfp4_weights_scales_cache()


def _make_inputs(
    hidden_size: int = 64,
    intermediate_size: int = 64,
    num_experts: int = 2,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_up_blocks = torch.zeros(
        (num_experts, intermediate_size * 2, hidden_size // 32, 16), dtype=torch.uint8
    )
    gate_up_scales = torch.ones(
        (num_experts, intermediate_size * 2, hidden_size // 32), dtype=torch.uint8
    )
    down_blocks = torch.zeros(
        (num_experts, hidden_size, intermediate_size // 32, 16), dtype=torch.uint8
    )
    down_scales = torch.ones((num_experts, hidden_size, intermediate_size // 32), dtype=torch.uint8)
    return hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales


def _install_fake_swizzle(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def _fake_swizzle_mxfp4(
        weight: torch.Tensor, weight_scale: torch.Tensor
    ) -> tuple[_FakeSwizzledTensor, _FakeSwizzledTensor]:
        calls.append((tuple(weight.shape), tuple(weight_scale.shape)))
        call_idx = len(calls)
        return _FakeSwizzledTensor(f"weight_{call_idx}"), _FakeSwizzledTensor(f"scale_{call_idx}")

    monkeypatch.setattr(mxfp4_moe, "_swizzle_mxfp4", _fake_swizzle_mxfp4)
    return calls


def test_prepare_weights_scales_reuses_cached_swizzles_for_same_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_swizzle(monkeypatch)
    hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales = _make_inputs()

    first = mxfp4_moe._prepare_weights_scales(
        hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales
    )
    second = mxfp4_moe._prepare_weights_scales(
        hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales
    )

    assert len(calls) == 2
    assert all(first_item is second_item for first_item, second_item in zip(first, second))
    assert first[0].shape == torch.Size([2, hidden_size, 128])
    assert first[2].shape == torch.Size([2, 64, hidden_size])
    assert calls == [
        ((2, 32, 128), (2, 2, 128)),
        ((2, 32, 64), (2, 2, 64)),
    ]


def test_prepare_weights_scales_reuses_cached_swizzles_for_inference_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_swizzle(monkeypatch)
    with torch.inference_mode():
        (
            hidden_size,
            gate_up_blocks,
            gate_up_scales,
            down_blocks,
            down_scales,
        ) = _make_inputs()

    first = mxfp4_moe._prepare_weights_scales(
        hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales
    )
    second = mxfp4_moe._prepare_weights_scales(
        hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales
    )

    assert len(calls) == 2
    assert all(first_item is second_item for first_item, second_item in zip(first, second))


def test_prepare_weights_scales_cache_key_invalidates_for_new_tensors_and_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_swizzle(monkeypatch)
    hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales = _make_inputs()

    first = mxfp4_moe._prepare_weights_scales(
        hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales
    )
    same_shape_new_gate = gate_up_blocks.clone()
    second = mxfp4_moe._prepare_weights_scales(
        hidden_size, same_shape_new_gate, gate_up_scales, down_blocks, down_scales
    )
    smaller_gate_up_blocks = gate_up_blocks[:, :64].contiguous()
    smaller_gate_up_scales = gate_up_scales[:, :64].contiguous()
    smaller_down_blocks = down_blocks[:, :, :1].contiguous()
    smaller_down_scales = down_scales[:, :, :1].contiguous()
    third = mxfp4_moe._prepare_weights_scales(
        hidden_size,
        smaller_gate_up_blocks,
        smaller_gate_up_scales,
        smaller_down_blocks,
        smaller_down_scales,
    )

    assert len(calls) == 6
    assert all(first_item is not second_item for first_item, second_item in zip(first, second))
    assert all(first_item is not third_item for first_item, third_item in zip(first, third))
    assert calls[4:] == [
        ((2, 32, 64), (2, 2, 64)),
        ((2, 16, 64), (2, 1, 64)),
    ]


def test_prepare_weights_scales_cache_key_invalidates_for_in_place_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_swizzle(monkeypatch)
    hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales = _make_inputs()
    original_data_ptr = gate_up_blocks.untyped_storage().data_ptr()
    original_version = gate_up_blocks._version

    first = mxfp4_moe._prepare_weights_scales(
        hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales
    )
    gate_up_blocks.add_(1)
    second = mxfp4_moe._prepare_weights_scales(
        hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales
    )

    assert gate_up_blocks.untyped_storage().data_ptr() == original_data_ptr
    assert gate_up_blocks._version > original_version
    assert len(calls) == 4
    assert all(first_item is not second_item for first_item, second_item in zip(first, second))
