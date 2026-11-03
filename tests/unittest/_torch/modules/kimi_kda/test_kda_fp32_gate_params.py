# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA ``A_log`` / ``dt_bias`` precision follows ``kda_fp32_gate_params``.

Kimi K3 / K3.1 checkpoints store both as F32 and the gate math consumes them in
fp32. Kimi K3 rounds them to bf16 with the rest of the KDA mixer; Kimi K3.1 keeps
the checkpoint's fp32 values end to end.
"""

import pytest
import torch
from kda_module_test_utils import make_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _checkpoint_style_gate_values(attention):
    gen = torch.Generator(device="cuda").manual_seed(3)
    # Values a bf16 round trip would change, as the checkpoint's are.
    a_log = (torch.rand(attention.A_log.shape, device="cuda", generator=gen) * 2.7).add_(2.0**-14)
    dt_bias = torch.rand(attention.dt_bias.shape, device="cuda", generator=gen).mul_(-6.0)
    assert not torch.equal(a_log.bfloat16().float(), a_log)
    return a_log, dt_bias


def _load_gate_values(attention, a_log, dt_bias) -> None:
    with torch.no_grad():
        # The Kimi weight loader copies checkpoint tensors into the parameters.
        attention.A_log.copy_(a_log)
        attention.dt_bias.copy_(dt_bias)
    attention.finalize_decode_weights()


def test_k31_checkpoint_fp32_gate_params_survive_load() -> None:
    attention = make_attention(
        heads=2, hidden_size=512, bf16_weights=True, kda_fp32_gate_params=True
    )
    assert attention.A_log.dtype == torch.float32
    assert attention.dt_bias.dtype == torch.float32

    a_log, dt_bias = _checkpoint_style_gate_values(attention)
    _load_gate_values(attention, a_log, dt_bias)

    assert torch.equal(attention.A_log, a_log)
    assert torch.equal(attention.dt_bias, dt_bias)
    # The fp32 copies the decode / verify kernels read are the exact values too.
    assert torch.equal(attention._A_log_f32, a_log)
    assert torch.equal(attention._dt_bias_f32, dt_bias)


def test_k3_gate_params_stay_bf16() -> None:
    attention = make_attention(heads=2, hidden_size=512, bf16_weights=True)
    assert attention.A_log.dtype == torch.bfloat16
    assert attention.dt_bias.dtype == torch.bfloat16

    a_log, dt_bias = _checkpoint_style_gate_values(attention)
    _load_gate_values(attention, a_log, dt_bias)

    # The kernels read fp32 copies of the bf16-rounded values.
    assert torch.equal(attention._A_log_f32, a_log.bfloat16().float())
    assert torch.equal(attention._dt_bias_f32, dt_bias.bfloat16().float())
