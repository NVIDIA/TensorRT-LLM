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
them before writing anything, while an NVFP4 expert still loads.
"""

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


class _RecordsLoads:
    """Records the weights and block scales a method is handed instead of laying them out.

    A weight write also marks its destination, so a test can tell whether a refused
    load changed the module's weights.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loaded = []
        self.written = []

    def load_expert_w3_w1_weight(
        self,
        module: SimpleNamespace,
        w1_weight: torch.Tensor | None,
        w3_weight: torch.Tensor | None,
        dst_w3_w1_weight: torch.Tensor,
        expert_idx: int = -1,
    ) -> None:
        if w1_weight is not None or w3_weight is not None:
            self.written.append("w3_w1")
            dst_w3_w1_weight.fill_(1)

    def load_expert_w2_weight(
        self,
        module: SimpleNamespace,
        w2_weight: torch.Tensor | None,
        dst_w2_weight: torch.Tensor,
    ) -> None:
        if w2_weight is not None:
            self.written.append("w2")
            dst_w2_weight.fill_(1)

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

    def process_weights_after_loading(self, module: SimpleNamespace) -> None:
        pass


class _RecordingMethod(_RecordsLoads, NVFP4CutlassFusedMoEMethod):
    """The vanilla NVFP4 method: 16-input blocks and global scales."""


class _MxBlockMethod(_RecordsLoads, NVFP4PlusCuteDslFusedMoEMethod):
    """The NVFP4_PLUS method: 32-input blocks and no global scales."""


class _TrtllmGenMethod(_RecordsLoads, NVFP4TRTLLMGenFusedMoEMethod):
    """The TRTLLM-Gen NVFP4 method, which loads its shared global scales itself."""


def _module(scaling_vector_size: int = 16, num_experts: int = 1) -> SimpleNamespace:
    """An MoE module with what FusedMoEMethodBase.load_weights reads; weights start zeroed."""
    return SimpleNamespace(
        layer_idx=61,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        scaling_vector_size=scaling_vector_size,
        is_gated_activation=True,
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        _add_raw_shared_weights_for_unmap=lambda tensors: None,
        num_experts=num_experts,
        initial_local_expert_ids=list(range(num_experts)),
        bias=False,
        layer_load_balancer=None,
        w3_w1_weight=torch.zeros(num_experts, 2 * INTERMEDIATE, HIDDEN // 2, dtype=torch.uint8),
        w2_weight=torch.zeros(num_experts, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8),
        w3_w1_weight_scale=torch.empty(
            num_experts, 2 * INTERMEDIATE, HIDDEN // 16, dtype=torch.uint8
        ),
        w2_weight_scale=torch.empty(num_experts, HIDDEN, INTERMEDIATE // 16, dtype=torch.uint8),
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


def _load(
    method: _RecordsLoads,
    module: SimpleNamespace,
    weights: dict[str, torch.Tensor],
    allow_partial_loading: bool = False,
) -> None:
    method.load_weights(
        module, weights, MoEWeightLoadingMode.VANILLA, allow_partial_loading=allow_partial_loading
    )


def _assert_nothing_written(method: _RecordsLoads, module: SimpleNamespace) -> None:
    assert method.written == []
    assert method.loaded == []
    assert not module.w3_w1_weight.any()
    assert not module.w2_weight.any()


def test_nvfp4_expert_loads() -> None:
    method = _RecordingMethod()
    module = _module()
    _load(method, module, _expert())
    assert method.written == ["w3_w1", "w2"]
    assert [entry[0] for entry in method.loaded] == ["w3_w1", "w2"]
    assert sorted(module.tmp_weight_scale_2[0]) == ["w1", "w2", "w3"]


def test_input_scale_is_not_required() -> None:
    # Activations quantized dynamically need no checkpoint input_scale.
    method = _RecordingMethod()
    _load(
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
    _load(method, _module(scaling_vector_size=32), _expert(block=32, global_scales=False))
    assert [entry[0] for entry in method.loaded] == ["w3_w1", "w2"]


def _narrow(expert: dict[str, torch.Tensor], *keys: str) -> dict[str, torch.Tensor]:
    """``expert`` with the block scales ``keys`` of a 32-input block (MXFP4's E8M0 width)."""
    return {**expert, **{k: _expert(int(k.split(".", 1)[0]), block=32)[k] for k in keys}}


def _two_experts_the_second_narrow() -> dict[str, torch.Tensor]:
    return {**_expert(0), **_narrow(_expert(1), "1.w1.weight_scale")}


def _refusal(
    weights, refused, method_cls=None, scaling_vector_size=16, num_experts=1, partial=False
):
    return pytest.param(
        method_cls or _RecordingMethod, scaling_vector_size, num_experts, weights, partial, refused
    )


_NEEDS_BLOCK = "_RecordingMethod needs one block scale per 16 inputs"
_NARROW_W1 = r"0\.w1\.weight_scale has shape \(32, 2\)"


@pytest.mark.parametrize(
    "method_cls, scaling_vector_size, num_experts, weights, allow_partial_loading, refused",
    [
        # The MTP block of an NVFP4 DeepSeek-V4 export: 32-input E8M0 block scales and no
        # global scales, configured as an NVFP4 layer; the message names layer and format.
        _refusal(
            _expert(block=32, global_scales=False),
            r"MoE layer 61: " + _NARROW_W1 + r"; _RecordingMethod expects one scale per 16 "
            r"inputs, 4 columns for 64 inputs\. The checkpoint stores",
        ),
        *(
            _refusal(_expert(drop=(k,)), f"has no {k}; {_NEEDS_BLOCK}")
            for k in ("0.w1.weight_scale", "0.w3.weight_scale", "0.w2.weight_scale")
        ),
        *(
            _refusal(_expert(drop=(k,)), f"has no {k}; _RecordingMethod needs the global")
            for k in ("0.w1.weight_scale_2", "0.w2.weight_scale_2")
        ),
        _refusal(_expert(block=32), "NVFP4_PLUS loads MXFP4 checkpoints", _MxBlockMethod, 32),
        # Every expert is checked before the first is written.
        _refusal(
            _two_experts_the_second_narrow(),
            r"1\.w1\.weight_scale has shape \(32, 2\)",
            num_experts=2,
        ),
        # Partial reload buckets: a weight with its narrow block scale, and a scale alone.
        _refusal(
            {"0.w1.weight": _expert()["0.w1.weight"], **_narrow({}, "0.w1.weight_scale")},
            _NARROW_W1,
            partial=True,
        ),
        _refusal(
            _narrow({}, "0.w2.weight_scale"),
            r"0\.w2\.weight_scale has shape \(64, 1\);",
            partial=True,
        ),
    ],
    ids=[
        "mxfp4_expert",
        "no_w1_block_scale",
        "no_w3_block_scale",
        "no_w2_block_scale",
        "no_w1_global_scale",
        "no_w2_global_scale",
        "nvfp4_plus_with_global_scales",
        "second_expert_narrow",
        "partial_bucket_narrow",
        "partial_bucket_scale_only_narrow",
    ],
)
def test_refused_load_leaves_the_weights_unchanged(
    method_cls: type[_RecordsLoads],
    scaling_vector_size: int,
    num_experts: int,
    weights: dict[str, torch.Tensor],
    allow_partial_loading: bool,
    refused: str,
) -> None:
    """A checkpoint the method cannot read is refused, naming what is wrong, before any write."""
    method = method_cls()
    module = _module(scaling_vector_size=scaling_vector_size, num_experts=num_experts)
    with pytest.raises(ValueError, match=refused):
        _load(method, module, weights, allow_partial_loading=allow_partial_loading)
    _assert_nothing_written(method, module)


def test_shared_expert_with_narrow_block_scale_is_refused_before_any_write() -> None:
    # With online EPLB the rank also loads the load balancer's shared copies, of
    # experts it does not hold locally; those are checked with the local ones.
    method = _RecordingMethod()
    module = _module()
    module.num_experts = 2
    module.layer_load_balancer = SimpleNamespace(
        need_load_shared_weights=lambda: True, get_load_expert_ids=lambda: [1]
    )
    with pytest.raises(ValueError, match=r"1\.w1\.weight_scale has shape \(32, 2\)"):
        _load(method, module, _two_experts_the_second_narrow())
    _assert_nothing_written(method, module)
    assert not hasattr(module, "local_shared_w3_w1_tensors")
    assert not hasattr(module, "local_shared_w3_w1_scale_tensors")


def test_partial_reload_bucket_without_weights_loads_its_scales() -> None:
    method = _RecordingMethod()
    weights = {k: v for k, v in _expert().items() if not k.endswith(".weight")}
    weights.pop("0.w2.weight_scale")
    _load(method, _module(), weights, allow_partial_loading=True)
    assert [entry[0] for entry in method.loaded] == ["w3_w1"]
    assert method.written == []


def _bucket(weights: dict[str, torch.Tensor], *suffixes: str) -> dict[str, torch.Tensor]:
    return {k: v for k, v in weights.items() if k.split(".", 1)[1] in suffixes}


_PROJECTION_KEYS = {
    proj: tuple(f"{proj}.{kind}" for kind in ("weight", "weight_scale", "weight_scale_2"))
    for proj in ("w1", "w2", "w3")
}


@pytest.mark.parametrize(
    "method_cls, scaling_vector_size",
    [(_RecordingMethod, 16), (_MxBlockMethod, 32), (_TrtllmGenMethod, 16)],
    ids=["cutlass", "nvfp4_plus", "trtllm_gen"],
)
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
def test_partial_reload_buckets_load(
    buckets: tuple[tuple[str, ...], ...],
    method_cls: type[_RecordsLoads],
    scaling_vector_size: int,
) -> None:
    # The path of a partial update_weights: every bucket goes through load_weights
    # with allow_partial_loading=True, and a bucket may hold a weight without its scales.
    method = method_cls()
    module = _module(scaling_vector_size=scaling_vector_size)
    global_scales = method.checkpoint_has_global_scales
    expert = _expert(block=scaling_vector_size, global_scales=global_scales)
    for suffixes in buckets:
        _load(method, module, _bucket(expert, *suffixes), allow_partial_loading=True)
    assert set(method.written) == {"w3_w1", "w2"}
    assert {entry[0] for entry in method.loaded} == {"w3_w1", "w2"}
    if global_scales:
        assert sorted(module.tmp_weight_scale_2[0]) == ["w1", "w2", "w3"]


def _streaming_module() -> SimpleNamespace:
    module = _module()
    module.expert_size_per_partition = 1
    module.initial_local_expert_ids = [0]
    module._streamed_expert_slots = set()
    return module


def _streaming_kwargs(expert: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
    return {
        f"{proj}_{kind}": expert.get(f"0.{proj}.{kind}")
        for proj in ("w1", "w2", "w3")
        for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    }


@pytest.mark.parametrize(
    "expert, refused",
    [
        (_narrow(_expert(), "0.w1.weight_scale"), r"0\.w1\.weight_scale has shape \(32, 2\);"),
        (_narrow(_expert(), "0.w3.weight_scale"), r"0\.w3\.weight_scale has shape \(32, 2\);"),
        (_narrow(_expert(), "0.w2.weight_scale"), r"0\.w2\.weight_scale has shape \(64, 1\);"),
        (_expert(drop=("0.w2.weight_scale_2",)), r"has no 0\.w2\.weight_scale_2;"),
    ],
    ids=["narrow_w1", "narrow_w3", "narrow_w2", "no_w2_global_scale"],
)
def test_streamed_expert_is_refused(expert: dict[str, torch.Tensor], refused: str) -> None:
    """The streaming loader refuses a narrow block scale or a missing global scale, and claims
    no slot."""
    method = _RecordingMethod()
    module = _streaming_module()
    with pytest.raises(ValueError, match=refused):
        method.load_streaming_nvfp4_expert(
            module, global_expert_id=0, local_slot_id=0, **_streaming_kwargs(expert)
        )
    assert module._streamed_expert_slots == set()
    assert method.loaded == []
