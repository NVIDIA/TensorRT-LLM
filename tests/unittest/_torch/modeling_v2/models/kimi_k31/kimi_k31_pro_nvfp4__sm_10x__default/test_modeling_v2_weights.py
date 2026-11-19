# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 default target's routed-expert load (CPU).

An expert of the NVIDIA NVFP4 checkpoint goes into its slot with FC1's rows in the order the GPU's CuTe DSL FC1 kernel
reads (the engine's interleave for that kernel), the block scales 128x4 swizzled and each GEMM's FP32 alpha its
weight's global scale. A non-unit input scale, which the kernels cannot serve, is refused by name.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default import (
    weights,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (
    mxfp8_scales_to_r128c4,
)
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    interleave_gate_and_linear,
    interleave_linear_and_gate,
)

__extra_import_path__ = [".."]
from _k31_utils import cpu_or_sm  # noqa: E402 — needs the path declared above

pytestmark = [pytest.mark.cpu_only, cpu_or_sm(103, 107)]

_INTER, _HIDDEN, _SLOTS = 128, 128, 2


@pytest.mark.parametrize(
    "sm_version, engine_interleave, group",
    [(107, interleave_gate_and_linear, 16), (103, interleave_linear_and_gate, 64)],
    ids=["sm107", "sm103"],
)
def test_fc1_rows_are_the_engine_order_of_the_gpus_kernel(sm_version, engine_interleave, group):
    g = torch.Generator().manual_seed(sm_version)
    gate, up = torch.randint(0, 256, (2, _INTER, 8), generator=g, dtype=torch.uint8)
    engine = engine_interleave(torch.cat([up, gate]), group, dim=0)
    assert torch.equal(weights.interleave_fc1_rows(gate, up, sm_version), engine)


def _checkpoint(seed: int, **overrides) -> dict:
    """One expert's checkpoint tensors at index 3 of ``layer.experts``."""
    g = torch.Generator().manual_seed(seed)

    def proj(rows, cols, ws2):
        return {
            "weight": torch.randint(0, 256, (rows, cols // 2), generator=g, dtype=torch.uint8),
            "weight_scale": torch.randint(
                0, 256, (rows, cols // 16), generator=g, dtype=torch.uint8
            ),
            "weight_scale_2": torch.tensor(ws2),
            "input_scale": torch.tensor(1.0),
        }

    tensors = {}
    for name, (rows, cols, ws2) in {
        "w1": (_INTER, _HIDDEN, 0.5),
        "w3": (_INTER, _HIDDEN, 0.5),
        "w2": (_HIDDEN, _INTER, 0.25),
    }.items():
        for kind, value in proj(rows, cols, ws2).items():
            tensors[f"layer.experts.3.{name}.{kind}"] = value
    tensors.update(overrides)
    return tensors


def _slots(sm_version: int) -> SimpleNamespace:
    def zeros(*shape, dtype=torch.uint8):
        return torch.zeros(_SLOTS, *shape, dtype=dtype)

    return SimpleNamespace(
        fc1_weight=zeros(2 * _INTER, _HIDDEN // 2),
        fc1_weight_scale=zeros(2 * _INTER, _HIDDEN // 16),
        fc2_weight=zeros(_HIDDEN, _INTER // 2),
        fc2_weight_scale=zeros(_HIDDEN, _INTER // 16),
        fc1_alpha=torch.zeros(_SLOTS),
        fc2_alpha=torch.zeros(_SLOTS),
        initial_local_expert_ids=[10, 11],
        sm_version=sm_version,
        _nvfp4_loaded_slots=set(),
    )


@pytest.mark.parametrize("sm_version", [103, 107])
def test_an_expert_loads_into_its_slot_in_the_gpus_layout(sm_version):
    ckpt = _checkpoint(sm_version)
    slots = _slots(sm_version)
    weights._load_nvfp4_expert(slots, "layer.experts", 3, 11, 1, ckpt.__getitem__)

    def read(w, kind):
        return ckpt[f"layer.experts.3.{w}.{kind}"]

    fc1 = weights.interleave_fc1_rows(read("w1", "weight"), read("w3", "weight"), sm_version)
    fc1_scale = weights.interleave_fc1_rows(
        read("w1", "weight_scale"), read("w3", "weight_scale"), sm_version
    )
    assert torch.equal(slots.fc1_weight[1], fc1)
    assert torch.equal(slots.fc1_weight_scale[1].flatten(), mxfp8_scales_to_r128c4(fc1_scale))
    assert torch.equal(slots.fc2_weight[1], read("w2", "weight"))
    assert torch.equal(
        slots.fc2_weight_scale[1].flatten(), mxfp8_scales_to_r128c4(read("w2", "weight_scale"))
    )
    assert slots.fc1_alpha.tolist() == [0.0, 0.5] and slots.fc2_alpha.tolist() == [0.0, 0.25]
    assert slots._nvfp4_loaded_slots == {1}
    # The other slot is untouched.
    assert not slots.fc1_weight[0].any() and not slots.fc2_weight_scale[0].any()


def test_the_load_refuses_a_non_unit_input_scale():
    ckpt = _checkpoint(0, **{"layer.experts.3.w2.input_scale": torch.tensor(0.5)})
    with pytest.raises(ValueError, match="unit input scales"):
        weights._load_nvfp4_expert(_slots(107), "layer.experts", 3, 11, 1, ckpt.__getitem__)
