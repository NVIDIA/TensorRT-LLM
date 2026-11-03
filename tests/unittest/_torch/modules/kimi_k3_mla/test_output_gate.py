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
"""The fused Kimi K3 MLA output gate must equal ``attn_output * gate.sigmoid()`` bit for bit."""

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_k3_mla import output_gate
from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate import (
    SIGMOID_GATE_MIN_TOKENS,
    sigmoid_gate_mul,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    same = a.view(torch.int16) == b.view(torch.int16)
    return int((~(same | (torch.isnan(a) & torch.isnan(b)))).sum())


def test_every_bf16_gate_value_is_bitwise():
    dev = torch.device("cuda")
    every = torch.arange(-32768, 32768, dtype=torch.int32, device=dev).to(torch.int16)
    gate = every.view(torch.bfloat16).repeat(24).view(128, 12288)
    torch.manual_seed(0)
    # Denormal through near-overflow magnitudes, so denormal sigmoids and products are covered.
    mags = torch.tensor([1e-40, 1e-38, 1e-30, 1e-6, 1e-2, 1.0, 3.0, 1e4, 1e30, 3e38], device=dev)
    pick = torch.randint(0, len(mags), gate.shape, device=dev)
    attn = (torch.randn(gate.shape, device=dev) * mags[pick]).to(torch.bfloat16)
    got = sigmoid_gate_mul(attn, gate)
    assert got is not None
    assert _mismatches(got, attn * gate.sigmoid()) == 0


@pytest.mark.parametrize("tokens", [SIGMOID_GATE_MIN_TOKENS, 1000, 4096])
def test_model_shapes_are_bitwise(tokens):
    dev = torch.device("cuda")
    attn = torch.randn(tokens, 96 * 128, device=dev).to(torch.bfloat16)
    gate = (torch.randn(tokens, 96 * 128, device=dev) * 4).to(torch.bfloat16)
    got = sigmoid_gate_mul(attn, gate)
    assert got is not None
    assert _mismatches(got, attn * gate.sigmoid()) == 0


@pytest.mark.parametrize(
    "tokens,cols,lead", [(1, 768, 2272), (24, 768, 2272), (112, 1536, 2272), (96, 768, 2272)]
)
def test_strided_decode_gate_is_bitwise(tokens, cols, lead, monkeypatch):
    """The gate as a column slice of the packed [latent | g_proj] GEMM output (row stride lead + cols)."""
    monkeypatch.setattr(output_gate, "_min_tokens", SIGMOID_GATE_MIN_TOKENS)
    dev = torch.device("cuda")
    torch.manual_seed(tokens + cols)
    packed = (torch.randn(tokens, lead + cols, device=dev) * 4).to(torch.bfloat16)
    gate = packed[:, lead:]
    attn = torch.randn(tokens, cols, device=dev).to(torch.bfloat16)
    assert sigmoid_gate_mul(attn, gate) is None  # below the default row minimum
    got = sigmoid_gate_mul(attn, gate, min_tokens=1)
    assert got is not None
    assert _mismatches(got, attn * gate.sigmoid()) == 0


def test_every_bf16_gate_value_strided_is_bitwise():
    dev = torch.device("cuda")
    every = torch.arange(-32768, 32768, dtype=torch.int32, device=dev).to(torch.int16)
    gate_full = torch.zeros(64, 1024 + 1536, dtype=torch.int16, device=dev)
    gate_full[:, 1024:] = every.repeat(2)[: 64 * 1536].view(64, 1536)
    gate = gate_full.view(torch.bfloat16)[:, 1024:]
    torch.manual_seed(1)
    mags = torch.tensor([1e-40, 1e-38, 1e-30, 1e-6, 1e-2, 1.0, 3.0, 1e4, 1e30, 3e38], device=dev)
    pick = torch.randint(0, len(mags), gate.shape, device=dev)
    attn = (torch.randn(gate.shape, device=dev) * mags[pick]).to(torch.bfloat16)
    got = sigmoid_gate_mul(attn, gate, min_tokens=1)
    assert got is not None
    assert _mismatches(got, attn * gate.sigmoid()) == 0


@pytest.mark.parametrize("width", [1536, 768])
@pytest.mark.parametrize("tokens", [1, 7, 28, 56, 112])
def test_small_batches_are_bitwise(tokens, width, monkeypatch):
    """Kimi K3.1 decode shapes (96 heads x 128 over TP8 / TP16) once small batches are enabled."""
    monkeypatch.setattr(output_gate, "_min_tokens", 1)
    dev = torch.device("cuda")
    attn = (torch.randn(tokens, width, device=dev) * 2).to(torch.bfloat16)
    gate = (torch.randn(tokens, width, device=dev) * 4).to(torch.bfloat16)
    got = sigmoid_gate_mul(attn, gate)
    assert got is not None
    assert _mismatches(got, attn * gate.sigmoid()) == 0
    assert sigmoid_gate_mul(attn[:0], gate[:0]) is None


@pytest.mark.parametrize("env", [None, "0", "1"])
def test_small_batches_default(env, monkeypatch):
    """Off by default (Kimi K3); a Kimi K3.1 model enables it unless the env var says otherwise."""
    monkeypatch.setattr(output_gate, "_SMALL_BATCHES_ENV", env)
    monkeypatch.setattr(output_gate, "_min_tokens", 1 if env == "1" else SIGMOID_GATE_MIN_TOKENS)
    output_gate.enable_small_batches()
    assert output_gate._min_tokens == (SIGMOID_GATE_MIN_TOKENS if env == "0" else 1)


def test_unsupported_inputs_fall_back(monkeypatch):
    monkeypatch.setattr(output_gate, "_min_tokens", SIGMOID_GATE_MIN_TOKENS)
    dev = torch.device("cuda")
    x = torch.randn(256, 1024, device=dev).to(torch.bfloat16)
    assert (
        sigmoid_gate_mul(x[: SIGMOID_GATE_MIN_TOKENS - 1], x[: SIGMOID_GATE_MIN_TOKENS - 1]) is None
    )
    assert sigmoid_gate_mul(x.float(), x.float()) is None
    assert sigmoid_gate_mul(x[:, :512], x[:, 512:]) is None
    assert sigmoid_gate_mul(x, x[:128]) is None
    # A gate whose elements are not contiguous within a row.
    assert sigmoid_gate_mul(x[:, :512], x[:, ::2], min_tokens=1) is None
    assert sigmoid_gate_mul(x[:, :512].contiguous(), x[:, ::2], min_tokens=1) is None
