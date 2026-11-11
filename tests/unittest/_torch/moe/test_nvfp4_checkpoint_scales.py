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
"""An NVFP4 MoE refuses checkpoint experts whose scales would not load.

The NVFP4 loaders read every scale with ``weights.get`` and zero-pad a block
scale that is narrower than its destination, so without a check an expert
stored in another format -- an MXFP4 expert has one E8M0 scale per 32 inputs
and no global scales -- would load without an error and compute garbage. These
tests build such experts on the CPU and check that both loading paths refuse
them while an NVFP4 expert still loads.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    NVFP4CutlassFusedMoEMethod,
    NVFP4PlusCuteDslFusedMoEMethod,
    NVFP4TRTLLMGenFusedMoEMethod,
)

pytestmark = pytest.mark.cpu_only

HIDDEN = 64
INTERMEDIATE = 32


class _RecordsScales:
    """Records the block scales a method is handed instead of laying them out."""

    def __init__(self) -> None:
        super().__init__()
        self.loaded = []

    def load_expert_w3_w1_weight_scale_nvfp4(
        self,
        module: SimpleNamespace,
        w1_weight_scale: torch.Tensor,
        w3_weight_scale: torch.Tensor,
        dst_w3_w1_weight_scale: torch.Tensor,
        expert_idx: int = -1,
    ) -> None:
        self.loaded.append(("w3_w1", w1_weight_scale, w3_weight_scale))

    def load_expert_w2_weight_scale_nvfp4(
        self,
        module: SimpleNamespace,
        w2_weight_scale: torch.Tensor,
        dst_w2_weight_scale: torch.Tensor,
    ) -> None:
        self.loaded.append(("w2", w2_weight_scale))


class _RecordingMethod(_RecordsScales, NVFP4CutlassFusedMoEMethod):
    """The vanilla NVFP4 method: 16-input blocks and global scales."""


class _MxBlockMethod(_RecordsScales, NVFP4PlusCuteDslFusedMoEMethod):
    """The NVFP4_PLUS method: 32-input blocks and no global scales."""


def _module(scaling_vector_size: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        layer_idx=61,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        scaling_vector_size=scaling_vector_size,
        is_gated_activation=True,
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        _add_raw_shared_weights_for_unmap=lambda tensors: None,
    )


def _expert(
    expert_id: int = 0, block: int = 16, global_scales: bool = True, drop: tuple[str, ...] = ()
) -> dict[str, torch.Tensor]:
    """One checkpoint expert: packed weights, block scales of `block` inputs."""
    weights = {}
    for proj, rows, cols in (
        ("w1", INTERMEDIATE, HIDDEN),
        ("w3", INTERMEDIATE, HIDDEN),
        ("w2", HIDDEN, INTERMEDIATE),
    ):
        prefix = f"{expert_id}.{proj}"
        weights[f"{prefix}.weight"] = torch.zeros(rows, cols // 2, dtype=torch.uint8)
        weights[f"{prefix}.weight_scale"] = torch.ones(rows, cols // block, dtype=torch.uint8)
        if global_scales:
            weights[f"{prefix}.weight_scale_2"] = torch.tensor(1.0)
            weights[f"{prefix}.input_scale"] = torch.tensor(1.0)
    for key in drop:
        del weights[key]
    return weights


def _load_block_scales(
    method: NVFP4CutlassFusedMoEMethod,
    module: SimpleNamespace,
    weights: dict[str, torch.Tensor],
    require_complete: bool = True,
) -> dict[int, dict[str, torch.Tensor]]:
    tmp_weight_scale_2 = {}
    method.load_fp4_weight_block_scales(
        module,
        weights,
        [0],
        torch.empty(1, 2 * INTERMEDIATE, HIDDEN // 16, dtype=torch.uint8),
        torch.empty(1, HIDDEN, INTERMEDIATE // 16, dtype=torch.uint8),
        tmp_weight_scale_2,
        require_complete=require_complete,
    )
    return tmp_weight_scale_2


def test_nvfp4_expert_loads() -> None:
    method = _RecordingMethod()
    tmp_weight_scale_2 = _load_block_scales(method, _module(), _expert())
    assert [entry[0] for entry in method.loaded] == ["w3_w1", "w2"]
    assert sorted(tmp_weight_scale_2[0]) == ["w1", "w2", "w3"]


def test_mxfp4_expert_is_refused() -> None:
    # The MTP block of an NVFP4 DeepSeek-V4 export: 32-input E8M0 block
    # scales and no global scales, configured as an NVFP4 layer.
    method = _RecordingMethod()
    with pytest.raises(ValueError, match=r"MoE layer 61: 0\.w1\.weight_scale .* need 4 columns"):
        _load_block_scales(method, _module(), _expert(block=32, global_scales=False))
    assert method.loaded == []


@pytest.mark.parametrize("key", ["0.w1.weight_scale", "0.w3.weight_scale", "0.w2.weight_scale"])
def test_missing_block_scale_is_refused(key: str) -> None:
    method = _RecordingMethod()
    with pytest.raises(ValueError, match=f"has no {key}; an NVFP4 expert needs one block scale"):
        _load_block_scales(method, _module(), _expert(drop=(key,)))
    assert method.loaded == []


@pytest.mark.parametrize("key", ["0.w1.weight_scale_2", "0.w2.weight_scale_2"])
def test_missing_global_weight_scale_is_refused(key: str) -> None:
    method = _RecordingMethod()
    with pytest.raises(ValueError, match=f"has no {key}; an NVFP4 expert needs its global"):
        _load_block_scales(method, _module(), _expert(drop=(key,)))
    assert method.loaded == []


def test_input_scale_is_not_required() -> None:
    # Activations quantized dynamically need no checkpoint input_scale.
    method = _RecordingMethod()
    _load_block_scales(
        method,
        _module(),
        _expert(drop=("0.w1.input_scale", "0.w3.input_scale", "0.w2.input_scale")),
    )
    assert len(method.loaded) == 2


def test_method_with_wider_blocks_and_no_global_scales_loads_its_format() -> None:
    # The check follows the method's own block width and global-scale
    # contract, so a method declared for 32-input blocks without global scales
    # still loads such an expert.
    method = _MxBlockMethod()
    _load_block_scales(
        method, _module(scaling_vector_size=32), _expert(block=32, global_scales=False)
    )
    assert [entry[0] for entry in method.loaded] == ["w3_w1", "w2"]


def test_partial_reload_bucket_without_weights_loads_its_scales() -> None:
    method = _RecordingMethod()
    weights = {k: v for k, v in _expert().items() if not k.endswith(".weight")}
    weights.pop("0.w2.weight_scale")
    _load_block_scales(method, _module(), weights, require_complete=False)
    assert [entry[0] for entry in method.loaded] == ["w3_w1"]


def _bucket(weights: dict[str, torch.Tensor], *suffixes: str) -> dict[str, torch.Tensor]:
    return {k: v for k, v in weights.items() if k.split(".", 1)[1] in suffixes}


_PROJECTION_KEYS = {
    proj: tuple(f"{proj}.{kind}" for kind in ("weight", "weight_scale", "weight_scale_2"))
    for proj in ("w1", "w2", "w3")
}


@pytest.mark.parametrize(
    "buckets",
    [
        # By projection, as the bucketed refit tests of the other MoE methods send them.
        (_PROJECTION_KEYS["w1"] + _PROJECTION_KEYS["w3"], _PROJECTION_KEYS["w2"]),
        (_PROJECTION_KEYS["w1"], _PROJECTION_KEYS["w3"], _PROJECTION_KEYS["w2"]),
        # One tensor kind per bucket, as a partial update_weights may send them: every
        # weight arrives without its scales.
        tuple(
            (f"{proj}.{kind}",)
            for kind in ("weight", "weight_scale", "weight_scale_2")
            for proj in ("w1", "w3", "w2")
        ),
    ],
    ids=["w3w1_then_w2", "per_projection", "per_tensor_kind"],
)
def test_partial_reload_buckets_load(buckets: tuple[tuple[str, ...], ...]) -> None:
    method = _RecordingMethod()
    expert = _expert()
    for suffixes in buckets:
        _load_block_scales(method, _module(), _bucket(expert, *suffixes), require_complete=False)
    assert {entry[0] for entry in method.loaded} == {"w3_w1", "w2"}


def test_narrow_block_scale_in_a_partial_reload_is_refused() -> None:
    method = _RecordingMethod()
    expert = _expert(block=32, global_scales=False)
    with pytest.raises(ValueError, match=r"0\.w2\.weight_scale .* need 2 columns"):
        _load_block_scales(
            method, _module(), _bucket(expert, "w2.weight_scale"), require_complete=False
        )
    assert method.loaded == []


class _ScalesOnlyMethod(_RecordingMethod):
    """Runs FusedMoEMethodBase.load_weights with the weight layout stubbed out."""

    def load_expert_weights_to_dst(
        self,
        module: SimpleNamespace,
        weights: dict[str, torch.Tensor],
        weight_loading_mode: MoEWeightLoadingMode,
        load_expert_ids: list[int],
        dst_w3_w1_weights_tensor: torch.Tensor,
        dst_w2_weights_tensor: torch.Tensor,
        dst_w3_w1_bias_tensor: torch.Tensor | None,
        dst_w2_bias_tensor: torch.Tensor | None,
        allow_partial_loading: bool = False,
    ) -> None:
        pass


def _loadable_module() -> SimpleNamespace:
    module = _module()
    module.num_experts = 1
    module.initial_local_expert_ids = [0]
    module.bias = False
    module.layer_load_balancer = None
    module.w3_w1_weight = torch.empty(1, 2 * INTERMEDIATE, HIDDEN // 2, dtype=torch.uint8)
    module.w2_weight = torch.empty(1, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8)
    module.w3_w1_weight_scale = torch.empty(1, 2 * INTERMEDIATE, HIDDEN // 16, dtype=torch.uint8)
    module.w2_weight_scale = torch.empty(1, HIDDEN, INTERMEDIATE // 16, dtype=torch.uint8)
    return module


@pytest.mark.parametrize(
    "bucket_suffixes",
    [
        (_PROJECTION_KEYS["w1"] + _PROJECTION_KEYS["w3"], _PROJECTION_KEYS["w2"]),
        tuple(
            (f"{proj}.{kind}",)
            for kind in ("weight", "weight_scale", "weight_scale_2")
            for proj in ("w1", "w3", "w2")
        ),
    ],
    ids=["w3w1_then_w2", "per_tensor_kind"],
)
def test_partial_update_weights_buckets_pass_load_weights(
    bucket_suffixes: tuple[tuple[str, ...], ...],
) -> None:
    # The path of a partial update_weights: every bucket goes through load_weights
    # with allow_partial_loading=True, and a bucket may hold a weight without its scales.
    method = _ScalesOnlyMethod()
    module = _loadable_module()
    expert = _expert()
    for suffixes in bucket_suffixes:
        method.load_weights(
            module,
            _bucket(expert, *suffixes),
            MoEWeightLoadingMode.VANILLA,
            allow_partial_loading=True,
        )
    assert {entry[0] for entry in method.loaded} == {"w3_w1", "w2"}
    assert sorted(module.tmp_weight_scale_2[0]) == ["w1", "w2", "w3"]


@pytest.mark.parametrize(
    "method_cls",
    [NVFP4CutlassFusedMoEMethod, NVFP4PlusCuteDslFusedMoEMethod, NVFP4TRTLLMGenFusedMoEMethod],
)
def test_load_weights_hands_the_partial_flag_to_the_scale_checks(
    method_cls: type[NVFP4CutlassFusedMoEMethod],
) -> None:
    # FusedMoEMethodBase.load_weights passes allow_partial_loading only to a load_quant_scales
    # that names it; an override that drops it would demand complete experts from every bucket.
    assert "allow_partial_loading" in inspect.getfullargspec(method_cls.load_quant_scales).args


def test_streamed_expert_with_narrow_block_scale_is_refused() -> None:
    method = _RecordingMethod()
    module = _module()
    module.expert_size_per_partition = 1
    module.initial_local_expert_ids = [0]
    module._streamed_expert_slots = set()
    expert = _expert(block=32, global_scales=False)
    kwargs = {
        f"{proj}_{kind}": expert.get(f"0.{proj}.{kind}", torch.tensor(1.0))
        for proj in ("w1", "w2", "w3")
        for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    }
    with pytest.raises(ValueError, match=r"w1_weight_scale has shape \(32, 2\)"):
        method.load_streaming_nvfp4_expert(module, global_expert_id=0, local_slot_id=0, **kwargs)
    assert module._streamed_expert_slots == set()
