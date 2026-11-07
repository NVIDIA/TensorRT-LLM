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
"""``mla_kv_to_fp8`` writes the attention op's FP8 context K / V bytes."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers the C++ ops)
from tensorrt_llm._torch.attention.kernels.mla_kv_fp8 import mla_kv_to_fp8

NOPE, ROPE, V, LORA = 128, 64, 128, 512

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="needs FP8 conversion instructions (SM89+)",
)


def _input(rows: int, cols: int, seed: int) -> torch.Tensor:
    """BF16 values spanning the E4M3 range, with saturating values and signed zeros."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, cols, device="cuda", generator=generator) * 64
    x.view(-1)[::97] = 1000.0
    x.view(-1)[1::97] = -1000.0
    x.view(-1)[2::97] = -0.0
    return x.to(torch.bfloat16)


def _cpp_unit_scale_fp8(x: torch.Tensor) -> torch.Tensor:
    unit = torch.ones(1, dtype=torch.float32, device="cuda")
    return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x.contiguous(), unit)[0]


@pytest.mark.parametrize("tokens,heads", [(1, 96), (31, 96), (1000, 96), (4099, 12), (65553, 8)])
@pytest.mark.parametrize("strided_pe", [False, True], ids=["contiguous_pe", "latent_cache_pe"])
def test_mla_kv_to_fp8_matches_cat_and_cpp_quant(tokens: int, heads: int, strided_pe: bool):
    """K / V are the unit-scale C++ FP8 bytes of the BF16 [T, H, nope + rope] concat of k_nope
    and k_pe and of the V columns."""
    kv = _input(tokens, heads * (NOPE + V), seed=tokens)
    latent = _input(tokens, LORA + ROPE, seed=tokens + 1)
    k_pe = latent[:, LORA:] if strided_pe else latent[:, LORA:].contiguous()
    k8, v8 = mla_kv_to_fp8(kv, k_pe, heads, NOPE, V)
    k_bf16 = torch.cat(
        (kv[:, : heads * NOPE].view(-1, heads, NOPE), k_pe[:, None].expand(-1, heads, -1)), -1
    ).view(tokens, -1)
    for x8, x in ((k8, k_bf16), (v8, kv[:, heads * NOPE :])):
        assert x8.dtype == torch.float8_e4m3fn and x8.is_contiguous() and x8.shape == x.shape
        assert torch.equal(x8.view(torch.uint8), _cpp_unit_scale_fp8(x).view(torch.uint8))


def test_mla_kv_to_fp8_rejects_bad_layouts():
    heads, tokens = 4, 16
    kv = torch.randn(tokens, heads * (NOPE + V), device="cuda").to(torch.bfloat16)
    latent = torch.randn(tokens, LORA + ROPE, device="cuda").to(torch.bfloat16)
    k, v = mla_kv_to_fp8(kv[:0], latent[:0, LORA:], heads, NOPE, V)
    assert k.shape == (0, heads * (NOPE + ROPE)) and v.shape == (0, heads * V)
    bad_calls = (
        (kv[:, 1:], latent[:, LORA:]),  # kv width
        (kv.float(), latent[:, LORA:]),  # kv dtype
        (kv, latent[:, LORA - 1 : -1]),  # odd k_pe offset
        (kv, latent[:, LORA : LORA + 48]),  # k_pe width not a power of two
        (kv[:-1], latent[:, LORA:]),  # token counts differ
    )
    for kv_arg, pe_arg in bad_calls:
        with pytest.raises(ValueError):
            mla_kv_to_fp8(kv_arg, pe_arg, heads, NOPE, V)
