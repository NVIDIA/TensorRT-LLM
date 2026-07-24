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

"""Tests for MegaMoE-CuteDSL NVFP4 streaming source weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

if TYPE_CHECKING:
    from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
    from tensorrt_llm._torch.modules.fused_moe.quantization import NVFP4MegaMoECuteDslMethod

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs 1 CUDA GPU (initial-load sources are rematerialized on cuda)",
)

NUM_EXPERTS = 4
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 64

_STREAMED_PARAMS = ("w3_w1_weight", "w3_w1_weight_scale", "w2_weight", "w2_weight_scale")


def _load_classes() -> tuple[type[MoEWeightLoadingMode], type[NVFP4MegaMoECuteDslMethod]]:
    from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
    from tensorrt_llm._torch.modules.fused_moe.quantization import NVFP4MegaMoECuteDslMethod

    return MoEWeightLoadingMode, NVFP4MegaMoECuteDslMethod


class _StreamingMoEModule(nn.Module):
    """Minimal single-rank module for the quant-method load path."""

    def __init__(self, weight_loading_mode: MoEWeightLoadingMode) -> None:
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.hidden_size = HIDDEN_SIZE
        self.intermediate_size_per_partition = INTERMEDIATE_SIZE
        self.expand_intermediate_size_per_partition = 2 * INTERMEDIATE_SIZE
        self.expert_size_per_partition = NUM_EXPERTS
        self.initial_local_expert_ids = list(range(NUM_EXPERTS))
        self.tp_size = 1
        self.tp_rank = 0
        self.ep_size = 1
        self.ep_rank = 0
        self.dtype = torch.bfloat16
        self.bias = False
        self.weight_loading_mode = weight_loading_mode
        # No EPLB in this test: need_load_shared_weights() must be False.
        self.layer_load_balancer = None

    def _add_raw_shared_weights_for_unmap(self, weight_tensors: list[torch.Tensor]) -> None:
        # Only forwards to the dynamic load balancer in production; no-op here.
        del weight_tensors


def _w13_input_scale(expert_id: int) -> float:
    return 0.5 + 0.125 * expert_id


def _w2_input_scale(expert_id: int) -> float:
    return 0.25 + 0.0625 * expert_id


def _make_vanilla_weights(seed: int = 20260708) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand_u8(*shape: int) -> torch.Tensor:
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=gen)

    def rand_fp8(*shape: int) -> torch.Tensor:
        # Any byte payload works (pure moves/reinterprets); stay clear of the
        # 0x7f/0xff NaN encodings for hygiene.
        raw = torch.randint(1, 120, shape, dtype=torch.uint8, device="cuda", generator=gen)
        return raw.view(torch.float8_e4m3fn)

    weights = {}
    for e in range(NUM_EXPERTS):
        weights[f"{e}.w1.weight"] = rand_u8(INTERMEDIATE_SIZE, HIDDEN_SIZE // 2)
        weights[f"{e}.w3.weight"] = rand_u8(INTERMEDIATE_SIZE, HIDDEN_SIZE // 2)
        weights[f"{e}.w2.weight"] = rand_u8(HIDDEN_SIZE, INTERMEDIATE_SIZE // 2)
        weights[f"{e}.w1.weight_scale"] = rand_fp8(INTERMEDIATE_SIZE, HIDDEN_SIZE // 16)
        weights[f"{e}.w3.weight_scale"] = rand_fp8(INTERMEDIATE_SIZE, HIDDEN_SIZE // 16)
        weights[f"{e}.w2.weight_scale"] = rand_fp8(HIDDEN_SIZE, INTERMEDIATE_SIZE // 16)
        # w1/w3 input scales must match per expert (parent PWAL asserts).
        weights[f"{e}.w1.input_scale"] = torch.tensor(_w13_input_scale(e), dtype=torch.float32)
        weights[f"{e}.w3.input_scale"] = torch.tensor(_w13_input_scale(e), dtype=torch.float32)
        weights[f"{e}.w2.input_scale"] = torch.tensor(_w2_input_scale(e), dtype=torch.float32)
        # w1/w3 weight_scale_2 must match per expert (reconcile warns/maxes).
        ws13 = torch.tensor(0.01 * (e + 1), dtype=torch.float32)
        weights[f"{e}.w1.weight_scale_2"] = ws13
        weights[f"{e}.w3.weight_scale_2"] = ws13.clone()
        weights[f"{e}.w2.weight_scale_2"] = torch.tensor(0.02 * (e + 1), dtype=torch.float32)
    return weights


def _fresh(
    seed: int = 20260708,
) -> tuple[
    NVFP4MegaMoECuteDslMethod,
    _StreamingMoEModule,
    dict[str, torch.Tensor],
]:
    mode_cls, method_cls = _load_classes()
    module = _StreamingMoEModule(mode_cls.VANILLA)
    method = method_cls()
    with torch.device("cuda"):
        method.create_weights(module)
    return method, module, _make_vanilla_weights(seed)


def _load(
    method: NVFP4MegaMoECuteDslMethod,
    module: _StreamingMoEModule,
    bucket: dict[str, torch.Tensor],
    allow_partial_loading: bool,
) -> None:
    mode_cls, _ = _load_classes()
    method.load_weights(
        module, bucket, mode_cls.VANILLA, allow_partial_loading=allow_partial_loading
    )


def _streamed_numels(module: _StreamingMoEModule) -> dict[str, int]:
    return {name: getattr(module, name).data.numel() for name in _STREAMED_PARAMS}


def _assert_sources_freed(module: _StreamingMoEModule, context: str = "") -> None:
    numels = _streamed_numels(module)
    assert all(n == 0 for n in numels.values()), (
        f"streamed source params should be 0-element placeholders {context}: {numels}"
    )


def _expected_mega_fc2(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([weights[f"{e}.w2.weight"] for e in range(NUM_EXPERTS)])


def _expected_mega_fc1(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    per_slot = []
    for e in range(NUM_EXPERTS):
        gate = weights[f"{e}.w1.weight"].view(INTERMEDIATE_SIZE // 16, 16, HIDDEN_SIZE // 2)
        up = weights[f"{e}.w3.weight"].view(INTERMEDIATE_SIZE // 16, 16, HIDDEN_SIZE // 2)
        per_slot.append(
            torch.stack([gate, up], dim=1).reshape(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 2)
        )
    return torch.stack(per_slot)


def _expected_fc1_norm_const() -> torch.Tensor:
    return torch.tensor(
        [1.0 / _w2_input_scale(e) for e in range(NUM_EXPERTS)],
        dtype=torch.float32,
        device="cuda",
    )


def test_initial_streaming_load_layer_atomic() -> None:
    method, module, weights = _fresh()

    # create_weights replaces streamed sources with empty placeholders.
    _assert_sources_freed(module, "right after create_weights")

    _load(method, module, weights, allow_partial_loading=False)
    _assert_sources_freed(module, "after the initial eager load")

    assert torch.equal(module.mega_fc2_weight.data, _expected_mega_fc2(weights))
    assert torch.equal(module.mega_fc1_weight.data, _expected_mega_fc1(weights))
    assert torch.allclose(module.fc1_norm_const.data, _expected_fc1_norm_const())


def test_partial_load_rejected_before_source_materialization() -> None:
    method, module, weights = _fresh()
    _assert_sources_freed(module, "before partial load")

    with pytest.raises(NotImplementedError, match="only supports full initial weight loading"):
        _load(method, module, weights, allow_partial_loading=True)

    _assert_sources_freed(module, "after rejected partial load")
