# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TRTLLM-runner routed experts' scalars of the dep and tep8 targets (the family's `loaders.expert_scalars`) are bit
for bit what stock's TRTLLM-Gen NVFP4 loader computes for this rank's experts: the layer-wide FC1 / FC2 input scales
(reciprocal of the largest checkpoint input_scale, `_finalize_input_scales`), `fc31_alpha`
(`load_expert_fc31_alpha_nvfp4`), `fc2_alpha` (`load_expert_fc2_alpha_nvfp4`) and `fc31_scale_c = fc2_input_scale *
fc31_alpha` (`_compute_fc31_scale_c`)."""

import importlib

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="imports the target (CUDA build)"
)
W = importlib.import_module("tensorrt_llm._torch.modeling_v2.models.deepseek_v41.loaders")


@pytest.mark.parametrize("target", ["dep", "tep8_moe_tp4_ep2"])
def test_both_targets_load_their_experts_with_the_familys_scalars(target):
    """dep's runner took each expert's own FC2 input scale (`o1 = gate_alpha / input_scale`, `o2 = input_scale *
    weight_scale_2`) where stock, dep2 and tep8 take the layer's one (the reciprocal of the largest); both targets now
    load through the same family loader."""
    wts = importlib.import_module(
        f"tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__{target}.weights"
    )
    assert wts.load_experts is W.load_experts


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_expert_scalars_are_the_loaders(seed):
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4FusedMoEMethod

    gen = torch.Generator().manual_seed(seed)
    experts, window = 384, slice(192, 384)
    # checkpoint-like positive scalars spanning a few binades
    isc1 = torch.exp2(torch.randn(experts, generator=gen) * 2 - 10)
    isc2 = torch.exp2(torch.randn(experts, generator=gen) * 2 - 12)
    ws1 = torch.exp2(torch.randn(experts, generator=gen) - 14)
    ws2 = torch.exp2(torch.randn(experts, generator=gen) - 14)
    got = W.expert_scalars(isc1, isc2, ws1[window], ws2[window])

    fc31_input_scale = isc1.max().reciprocal()
    fc2_input_scale = isc2.max().reciprocal()
    alpha1 = torch.empty(experts)
    alpha2 = torch.empty(experts)
    for e in range(experts):
        NVFP4FusedMoEMethod.load_expert_fc31_alpha_nvfp4(
            None, ws1[e], ws1[e], fc31_input_scale, alpha1[e]
        )
        NVFP4FusedMoEMethod.load_expert_fc2_alpha_nvfp4(None, ws2[e], fc2_input_scale, alpha2[e])
    assert torch.equal(got["g1"], fc31_input_scale.reshape(1))
    assert torch.equal(got["o1_gate"], alpha1[window])
    assert torch.equal(got["o2"], alpha2[window])
    assert torch.equal(got["o1"], fc2_input_scale * alpha1[window])
