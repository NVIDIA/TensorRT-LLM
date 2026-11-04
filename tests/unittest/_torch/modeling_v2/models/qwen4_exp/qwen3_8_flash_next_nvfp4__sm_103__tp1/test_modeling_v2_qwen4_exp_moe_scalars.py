# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qwen4_exp's routed-expert scalars, derived on CPU from the checkpoint's scale rows.

CuteDSL requantizes FC1's output with ONE global scale per layer, where trtllm-gen folds
each expert's `down_proj.input_scale` into its own FC1 scalar. The CuteDSL scalars are
exact for any spread of `down_proj.input_scale` across experts: they are the engine's
`max(w2 input_scale)` convention.
"""

from __future__ import annotations

import pytest
import torch

__extra_import_path__ = ["../../.."]  # noqa: F841 -- repo's file-scoped import hook

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only

EXPERTS = 8


def _rows(isc2: torch.Tensor, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    isc1 = torch.full((EXPERTS,), 0.0123)
    ws2_1 = torch.rand(EXPERTS, generator=gen) * 1e-3 + 1e-4
    return {
        "l0_e_isc1": isc1,
        "l0_e_isc1_up": isc1.clone(),
        "l0_e_ws2_1": ws2_1,
        "l0_e_ws2_1_up": ws2_1.clone(),
        "l0_e_isc2": isc2,
        "l0_e_ws2_2": torch.rand(EXPERTS, generator=gen) * 1e-3 + 1e-4,
    }


def test_uniform_isc2_keeps_the_per_expert_products_bitwise() -> None:
    w = _rows(torch.full((EXPERTS,), 0.0371))
    got = M._moe_layer_scalars(w, 0, "CUTEDSL")
    assert torch.equal(got["alpha2"], w["l0_e_isc2"] * w["l0_e_ws2_2"])
    assert torch.equal(got["global_sf"], 1.0 / w["l0_e_isc2"][:1])
    assert got["global_sf"].shape == (1,)
    assert torch.equal(got["alpha1"], w["l0_e_isc1"] * w["l0_e_ws2_1"])


def test_isc2_spread_is_folded_into_alpha2() -> None:
    """Whatever the spread (here 40x), `global_sf * alpha2[e]` is expert e's own `ws2_2[e]`,
    the product trtllm-gen forms from `(1 / isc2[e]) * (isc2[e] * ws2_2[e])`: the FC2 value an
    expert produces is the same whether its input was quantized at its own `1 / isc2[e]` or
    at the layer's single scale."""
    isc2 = 0.01 * torch.linspace(1.0, 40.0, EXPERTS)
    w = _rows(isc2.flip(0))
    cute = M._moe_layer_scalars(w, 0, "CUTEDSL")
    gen = M._moe_layer_scalars(w, 0, "TRTLLM")
    isc2 = w["l0_e_isc2"]
    assert torch.equal(cute["global_sf"], 1.0 / isc2.max().reshape(1))
    # The CuteDSL FC1 quantizes at the smallest per-expert global scale: no expert's FC2
    # input is scaled above what the checkpoint calibrated for it.
    assert (cute["global_sf"] <= 1.0 / isc2).all()
    assert torch.allclose(
        cute["global_sf"] * cute["alpha2"], (1.0 / isc2) * gen["output2_scale_scalar"], rtol=1e-6
    )
    assert torch.allclose(cute["global_sf"] * cute["alpha2"], w["l0_e_ws2_2"], rtol=1e-6)


def test_gate_up_input_scale_mismatch_is_refused() -> None:
    w = _rows(torch.full((EXPERTS,), 0.0371))
    w["l0_e_isc1_up"] = w["l0_e_isc1_up"] * 1.5
    with pytest.raises(AssertionError, match="gate/up input_scale differ"):
        M._moe_layer_scalars(w, 0, "CUTEDSL")
