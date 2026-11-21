# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An NVFP4 MoE refuses, before any write, checkpoint experts whose scales it would misload."""

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
_SHAPES = {"w1": (INTERMEDIATE, HIDDEN), "w3": (INTERMEDIATE, HIDDEN), "w2": (HIDDEN, INTERMEDIATE)}


class _RecordsLoads:
    """Records the projections whose weights and block scales a method is handed."""

    def __init__(self) -> None:
        super().__init__()
        self.loaded = []
        self.written = []

    def load_expert_w3_w1_weight(
        self, module, w1_weight, w3_weight, dst_w3_w1_weight, expert_idx=-1
    ):
        if w1_weight is not None or w3_weight is not None:
            self.written.append("w3_w1")
            dst_w3_w1_weight.fill_(1)

    def load_expert_w2_weight(self, module, w2_weight, dst_w2_weight):
        if w2_weight is not None:
            self.written.append("w2")
            dst_w2_weight.fill_(1)

    def load_expert_w3_w1_weight_scale_nvfp4(
        self, module, w1_weight_scale, w3_weight_scale, dst_w3_w1_weight_scale, expert_idx=-1
    ):
        self.loaded.append("w3_w1")

    def load_expert_w2_weight_scale_nvfp4(self, module, w2_weight_scale, dst_w2_weight_scale):
        self.loaded.append("w2")

    def process_weights_after_loading(self, module) -> None:
        pass


_RecordingMethod = type("_RecordingMethod", (_RecordsLoads, NVFP4CutlassFusedMoEMethod), {})
_MxBlockMethod = type("_MxBlockMethod", (_RecordsLoads, NVFP4PlusCuteDslFusedMoEMethod), {})
_TrtllmGenMethod = type("_TrtllmGenMethod", (_RecordsLoads, NVFP4TRTLLMGenFusedMoEMethod), {})


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


def _expert(expert_id=0, block=16, global_scales=True, drop=()) -> dict[str, torch.Tensor]:
    """One checkpoint expert: packed weights, block scales of `block` inputs."""
    weights = {}
    for proj, (rows, cols) in _SHAPES.items():
        prefix = f"{expert_id}.{proj}"
        weights[f"{prefix}.weight"] = torch.zeros(rows, cols // 2, dtype=torch.uint8)
        weights[f"{prefix}.weight_scale"] = torch.ones(rows, cols // block, dtype=torch.uint8)
        if global_scales:
            weights[f"{prefix}.weight_scale_2"] = torch.tensor(1.0)
            weights[f"{prefix}.input_scale"] = torch.tensor(1.0)
    for key in drop:
        del weights[key]
    return weights


def _narrow(expert: dict[str, torch.Tensor], *keys: str) -> dict[str, torch.Tensor]:
    """``expert`` with the block scales ``keys`` of a 32-input block (MXFP4's E8M0 width)."""
    return {**expert, **{k: _expert(int(k.split(".", 1)[0]), block=32)[k] for k in keys}}


def _load(method, module, weights, mode: str = "full") -> None:
    """``load_weights`` (whole experts, a partial-reload bucket, or with the EPLB shared copy of
    expert 1), or the streaming loader of expert 0."""
    if mode == "streamed":
        module.expert_size_per_partition = 1
        module._streamed_expert_slots = set()
        tensors = {
            f"{proj}_{kind}": weights.get(f"0.{proj}.{kind}")
            for proj in _SHAPES
            for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
        }
        method.load_streaming_nvfp4_expert(module, global_expert_id=0, local_slot_id=0, **tensors)
        return
    if mode == "shared":
        module.num_experts = 2
        module.layer_load_balancer = SimpleNamespace(
            need_load_shared_weights=lambda: True, get_load_expert_ids=lambda: [1]
        )
    method.load_weights(
        module, weights, MoEWeightLoadingMode.VANILLA, allow_partial_loading=mode == "partial"
    )


@pytest.mark.parametrize(
    "method_cls, block, mode",
    [
        (_RecordingMethod, 16, "full"),
        (_MxBlockMethod, 32, "full"),
        (_RecordingMethod, 16, "partial"),
        (_MxBlockMethod, 32, "partial"),
        (_TrtllmGenMethod, 16, "partial"),
    ],
    ids=["cutlass", "nvfp4_plus", "cutlass_partial", "nvfp4_plus_partial", "trtllm_gen_partial"],
)
def test_expert_in_the_method_format_loads(method_cls, block: int, mode: str) -> None:
    """No false refusal: a full load needs no input_scale (dynamic activations), and a partial
    reload may send every tensor in its own bucket, each weight without its scales."""
    method = method_cls()
    module = _module(scaling_vector_size=block)
    expert = _expert(block=block, global_scales=method.checkpoint_has_global_scales)
    if mode == "partial":
        for kind in ("weight", "weight_scale", "weight_scale_2"):
            for proj in ("w1", "w3", "w2"):
                bucket = {k: v for k, v in expert.items() if k == f"0.{proj}.{kind}"}
                _load(method, module, bucket, mode)
    else:
        _load(method, module, {k: v for k, v in expert.items() if "input_scale" not in k})
    assert set(method.written) == {"w3_w1", "w2"}
    assert set(method.loaded) == {"w3_w1", "w2"}
    if mode == "partial" and method.checkpoint_has_global_scales:
        assert sorted(module.tmp_weight_scale_2[0]) == ["w1", "w2", "w3"]


_NARROW_W1 = r"0\.w1\.weight_scale has shape \(32, 2\)"
_NARROW_SECOND = r"1\.w1\.weight_scale has shape \(32, 2\)"


def _second_expert_narrow() -> dict[str, torch.Tensor]:
    return {**_expert(0), **_narrow(_expert(1), "1.w1.weight_scale")}


@pytest.mark.parametrize(
    "method_cls, block, num_experts, weights, mode, refused",
    [
        # The MTP block of an NVFP4 DeepSeek-V4 export: 32-input E8M0 block scales and no
        # global scales, configured as an NVFP4 layer; the message names layer and format.
        (
            _RecordingMethod, 16, 1, _expert(block=32, global_scales=False), "full",
            r"MoE layer 61: " + _NARROW_W1 + r"; _RecordingMethod expects one scale per 16 "
            r"inputs, 4 columns for 64 inputs\. The checkpoint stores",
        ),
        (
            _RecordingMethod, 16, 1, _expert(drop=("0.w2.weight_scale",)), "full",
            "has no 0.w2.weight_scale; _RecordingMethod needs one block scale per 16 inputs",
        ),
        (
            _RecordingMethod, 16, 1, _expert(drop=("0.w1.weight_scale_2",)), "full",
            "has no 0.w1.weight_scale_2; _RecordingMethod needs the global",
        ),
        (_MxBlockMethod, 32, 1, _expert(block=32), "full", "NVFP4_PLUS loads MXFP4 checkpoints"),
        # Every expert is checked before the first is written, the EPLB shared copies too.
        (_RecordingMethod, 16, 2, _second_expert_narrow(), "full", _NARROW_SECOND),
        (_RecordingMethod, 16, 1, _second_expert_narrow(), "shared", _NARROW_SECOND),
        # A partial-reload bucket holding a narrow block scale alone.
        (
            _RecordingMethod, 16, 1, _narrow({}, "0.w2.weight_scale"), "partial",
            r"0\.w2\.weight_scale has shape \(64, 1\);",
        ),
        (_RecordingMethod, 16, 1, _narrow(_expert(), "0.w1.weight_scale"), "streamed", _NARROW_W1),
        (
            _RecordingMethod, 16, 1, _expert(drop=("0.w2.weight_scale_2",)), "streamed",
            r"has no 0\.w2\.weight_scale_2;",
        ),
    ],
    ids=["mxfp4_expert", "no_block_scale", "no_global_scale", "nvfp4_plus_with_global_scales",
         "second_expert_narrow", "shared_expert_narrow", "partial_bucket_narrow", "streamed_narrow",
         "streamed_no_global_scale"],
)  # fmt: skip
def test_refused_load_leaves_the_weights_unchanged(
    method_cls, block: int, num_experts: int, weights, mode: str, refused: str
) -> None:
    """Regression: an expert with narrow (MXFP4) or missing block or global scales loaded."""
    method = method_cls()
    module = _module(scaling_vector_size=block, num_experts=num_experts)
    with pytest.raises(ValueError, match=refused):
        _load(method, module, weights, mode)
    assert method.written == [] and method.loaded == []
    assert not module.w3_w1_weight.any() and not module.w2_weight.any()
    assert not getattr(module, "_streamed_expert_slots", None)
    assert not hasattr(module, "local_shared_w3_w1_tensors")
