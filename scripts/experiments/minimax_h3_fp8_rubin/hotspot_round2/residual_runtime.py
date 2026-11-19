# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-local residual experiment and a reviewed optional KF producer callback."""

import json
from pathlib import Path

import cache_env  # noqa: F401 -- configure caches before importing CUDA libraries
import fp8_producers as producers
import residual_kernel
import torch

from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4 import (
    modeling as scenario,
)

ORIGINAL_FORWARD = None
STATE = {"kf_enabled": False, "kf_function": None, "validate_kf": False}
KF_CHECKS = []


@torch.library.custom_op("h3_fp8_round3::swiglu", mutates_args=())
def swiglu(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not STATE["kf_enabled"] or STATE["kf_function"] is None:
        return producers.swiglu(x, scale)
    if STATE["validate_kf"] and len(KF_CHECKS) < 1300:
        ref = producers.swiglu(x, scale)
        value = STATE["kf_function"](x, scale)
        equal = torch.equal(ref.view(torch.uint8), value.view(torch.uint8))
        KF_CHECKS.append(
            dict(
                index=len(KF_CHECKS),
                shape=list(x.shape),
                scale=float(scale.item()),
                byte_equal=bool(equal),
            )
        )
        if len(KF_CHECKS) % 50 == 0:
            Path("/profile/results/kf-real-checks.json").write_text(json.dumps(KF_CHECKS, indent=2))
        return ref
    if STATE["validate_kf"]:
        return producers.swiglu(x, scale)
    return STATE["kf_function"](x, scale)


@swiglu.register_fake
def _(x, scale):
    return torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=torch.float8_e4m3fn)


def forward(
    self, hidden_states, temb, adaln_indices, rotary_emb, key_padding_mask, timestep, valid_seq_len
):
    mod = self.adaln_proj.table(temb)
    quantized = producers.norm(
        hidden_states,
        self.norm1.weight,
        mod,
        adaln_indices,
        self.attn.qkv_proj.input_scale,
        self.norm1.variance_epsilon,
    )
    attn_output = self.attn(quantized, rotary_emb, key_padding_mask, timestep, valid_seq_len)
    hidden_states, quantized = producers.gate_norm(
        hidden_states,
        attn_output,
        self.norm2.weight,
        mod,
        adaln_indices,
        self.ff.gate_up_proj.input_scale,
        self.norm2.variance_epsilon,
    )
    ff_output = self.ff(quantized.reshape(-1, quantized.shape[-1])).reshape_as(hidden_states)
    return residual_kernel.residual(hidden_states, ff_output, mod, adaln_indices)


def install(install_residual=True):
    global ORIGINAL_FORWARD
    ORIGINAL_FORWARD = scenario.ScenarioBlock._forward_fused_adaln
    activation = GatedMLP._apply_activation

    def apply(self, x, *, has_lora=False):
        if (
            x.shape[-1] == 28672
            and self.down_proj.has_fp8_qdq
            and not has_lora
            and self._is_plain_swiglu()
            and self.swiglu_limit is None
            and self.down_proj.input_scale is not None
        ):
            return swiglu(x, self.down_proj.input_scale)
        return activation(self, x, has_lora=has_lora)

    GatedMLP._apply_activation = apply
    if install_residual:
        scenario.ScenarioBlock._forward_fused_adaln = forward
