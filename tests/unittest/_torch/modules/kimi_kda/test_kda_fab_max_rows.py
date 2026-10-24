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
"""Gate of the fused KDA verify ``[f_a | b] -> f_b`` projection (TRTLLM_K3_KDA_FAB_FUSED).

``k3_kda_fab`` takes at most 16 heads and ``MAX_M`` rows (its default tiling covers fewer rows for
more heads: 224 at Kimi K3's 12 heads per TP8 rank). It is sized to run beside the q|k|v|g GEMM,
so the mixer routes the verify projections to it only when the switch is on, the rows stay
within both ``_KDA_FAB_FUSED_MAX_ROWS`` (TRTLLM_K3_KDA_FAB_MAX_ROWS, default 128) and the aux
stream's ``_KDA_BFA_MULTISTREAM_MAX_ROWS`` (default 128), and the kernel accepts the operands.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs an SM100/SM103/SM107 GPU",
)

H = 12  # Kimi K3 KDA heads per rank at TP8
HIDDEN = 7168


def _mixer():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.modules.kimi_kda import kimi_kda_mixer

    return kimi_kda_mixer


def _weights(heads=H):
    """[f_a | b] rows padded to a multiple of 8 with zeros, as the mixer merges them."""
    rows = 128 + heads
    w_bfa = torch.zeros(rows + (-rows) % 8, HIDDEN, device="cuda")
    w_bfa[:rows] = torch.randn(rows, HIDDEN, device="cuda") * 0.02
    w_fb = (torch.randn(heads * 128, 128, device="cuda") * 0.05).bfloat16()
    return w_bfa.bfloat16(), w_fb


def _x(batch, steps):
    return torch.randn(batch, steps, HIDDEN, device="cuda").bfloat16()


def test_defaults():
    mix = _mixer()
    if "TRTLLM_K3_KDA_FAB_FUSED" not in os.environ:
        assert mix._KDA_FAB_FUSED is False
    if "TRTLLM_K3_KDA_FAB_MAX_ROWS" not in os.environ:
        assert mix._KDA_FAB_FUSED_MAX_ROWS == 128


def test_off_by_default_switch(monkeypatch):
    mix = _mixer()
    monkeypatch.setattr(mix, "_KDA_FAB_FUSED", False)
    w_bfa, w_fb = _weights()
    assert mix._use_fused_fab(_x(2, 6), w_bfa, w_fb, H) is False


@pytest.mark.parametrize(
    "cap,aux_cap,batch,steps,expect",
    [
        (128, 128, 2, 6, True),  # TP8 C15: 12 rows
        (128, 128, 16, 6, True),  # TP8 batch 16 x d5: 96 rows
        (128, 128, 16, 8, True),  # 128 rows
        (128, 128, 17, 8, False),  # 136 rows above the default caps
        (256, 128, 28, 8, False),  # 224 rows: off the aux stream, so not fused
        (128, 256, 28, 8, False),  # 224 rows above the fused cap
        (256, 256, 28, 8, True),  # 224 rows: the largest default tiling at 12 heads
        (256, 256, 32, 8, False),  # 256 rows: within both caps, beyond the tiling at 12 heads
        (1024, 1024, 128, 7, False),  # above the kernel's MAX_M
    ],
)
def test_fused_fab_respects_caps(monkeypatch, cap, aux_cap, batch, steps, expect):
    mix = _mixer()
    monkeypatch.setattr(mix, "_KDA_FAB_FUSED", True)
    monkeypatch.setattr(mix, "_KDA_FAB_FUSED_MAX_ROWS", cap)
    monkeypatch.setattr(mix, "_KDA_BFA_MULTISTREAM_MAX_ROWS", aux_cap)
    w_bfa, w_fb = _weights()
    assert mix._use_fused_fab(_x(batch, steps), w_bfa, w_fb, H) is expect


def test_fused_fab_rejects_attention_dp_heads(monkeypatch):
    """96 heads per rank (DEP16) exceed the kernel's 144 [f_a | b] rows: cuBLAS keeps them."""
    mix = _mixer()
    monkeypatch.setattr(mix, "_KDA_FAB_FUSED", True)
    w_bfa, w_fb = _weights(heads=96)
    assert mix._use_fused_fab(_x(2, 4), w_bfa, w_fb, 96) is False


@pytest.mark.parametrize("batch,steps", [(2, 6), (16, 6)])
def test_fused_fab_matches_the_projections(monkeypatch, batch, steps):
    """The fused kernel against the cuBLAS [f_a | b] GEMM, slice and f_b GEMM it replaces (the
    mixer's unfused path): FP32 accumulation in another K order, so beta and the BF16 f_a may
    round to a neighbouring BF16 value (up to 2^-7 relative), and the forget gate moves by at most
    2^-7 |f_a| @ |W_fb| on top of its own rounding; at least 99% of the elements within 1 ulp."""
    mix = _mixer()
    monkeypatch.setattr(mix, "_KDA_FAB_FUSED", True)
    torch.manual_seed(3)
    w_bfa, w_fb = _weights()
    x = _x(batch, steps)
    assert mix._use_fused_fab(x, w_bfa, w_fb, H)
    x2d = x.reshape(-1, HIDDEN)
    beta, forget_gate = torch.ops.trtllm.k3_kda_fab_proj(x2d, w_bfa, w_fb, H)
    bfa = torch.nn.functional.linear(x2d, w_bfa)
    ref_fa = bfa[:, :128].contiguous()
    ref_beta = bfa[:, 128 : 128 + H]
    ref_fg = torch.nn.functional.linear(ref_fa, w_fb)
    torch.testing.assert_close(beta.float(), ref_beta.float(), rtol=2**-7, atol=1e-3)
    err = (forget_gate.float() - ref_fg.float()).abs()
    own = ref_fg.float().abs() * 2**-7 + 1e-4
    bound = (ref_fa.float().abs() @ w_fb.float().abs().t()) * 2**-7 + own
    assert bool((err <= bound).all()), float((err - bound).max())
    assert float((err <= own).float().mean()) > 0.99
    # Deterministic: a second launch is bit-identical.
    beta2, forget_gate2 = torch.ops.trtllm.k3_kda_fab_proj(x2d, w_bfa, w_fb, H)
    assert torch.equal(beta, beta2) and torch.equal(forget_gate, forget_gate2)
