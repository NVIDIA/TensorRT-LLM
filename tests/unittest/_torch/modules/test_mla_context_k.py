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
"""MLA context K: per-head k_nope followed by the head-shared k_pe."""

import pytest
import torch

from tensorrt_llm._torch.modules.mla import mla_context_k

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the context-K kernel is a Triton GPU kernel"
)

NOPE, ROPE, V, LORA = 128, 64, 128, 512


def _k_nope(num_tokens, num_heads, nope=NOPE):
    # The K half of kv_b_proj's [tokens, heads * (nope + v)] output.
    kv = torch.randn(num_tokens, num_heads * (nope + V), device="cuda", dtype=torch.bfloat16)
    return kv[:, : num_heads * nope].view(num_tokens, num_heads, nope)


def _k_pe(num_tokens, layout, rope=ROPE):
    if layout == "latent":
        # The rope slice of a [tokens, kv_lora_rank + rope] latent-cache row.
        latent = torch.randn(num_tokens, LORA + rope, device="cuda", dtype=torch.bfloat16)
        return latent[:, LORA:].view(num_tokens, 1, rope)
    k_pe = torch.randn(num_tokens, rope, device="cuda", dtype=torch.bfloat16)
    return k_pe if layout == "2d" else k_pe.view(num_tokens, 1, rope)


def _reference(k_nope, k_pe):
    k_pe = k_pe.reshape(k_pe.shape[0], 1, k_pe.shape[-1])
    return torch.cat((k_nope, k_pe.expand(-1, k_nope.shape[1], -1)), dim=-1)


@pytest.mark.parametrize("layout", ["chunk", "latent", "2d"])
@pytest.mark.parametrize("num_heads", [1, 12, 96])
@pytest.mark.parametrize("num_tokens", [0, 1, 33, 4100])
def test_context_k_matches_cat(num_tokens, num_heads, layout):
    k_nope = _k_nope(num_tokens, num_heads)
    k_pe = _k_pe(num_tokens, layout)

    k = mla_context_k(k_nope, k_pe)

    assert k.shape == (num_tokens, num_heads, NOPE + ROPE)
    assert k.is_contiguous()
    torch.testing.assert_close(k, _reference(k_nope, k_pe), rtol=0, atol=0)


def test_context_k_falls_back_for_unsupported_head_dims():
    # tl.arange needs power-of-two widths; other widths take torch.cat.
    k_nope = _k_nope(7, 4, nope=96)
    k_pe = _k_pe(7, "chunk", rope=48)

    torch.testing.assert_close(
        mla_context_k(k_nope, k_pe), _reference(k_nope, k_pe), rtol=0, atol=0
    )
