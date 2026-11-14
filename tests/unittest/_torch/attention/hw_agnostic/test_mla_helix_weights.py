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
"""Host-side construction and aliasing of Helix V projection weights."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.mla import MLA

pytestmark = pytest.mark.cpu_only


def _v_up_builder(kv_b_weight, kv_b_scale, *, num_heads, cp_size, cp_rank, nope, v, k):
    """The fields of an MLA module that ``MLA._build_helix_v_up_weight`` reads."""
    module = SimpleNamespace(
        mapping=SimpleNamespace(has_cp_helix=lambda: True, cp_rank=cp_rank),
        dtype=torch.bfloat16,
        kv_b_proj=SimpleNamespace(
            weight=kv_b_weight,
            weight_scale=kv_b_scale,
            partition_plan=SimpleNamespace(enabled=False, op_kind=None),
        ),
        num_heads_tp=num_heads,
        num_heads_tp_cp=num_heads // cp_size,
        qk_nope_head_dim=nope,
        v_head_dim=v,
        kv_lora_rank=k,
        v_b_proj_dequant=None,
        helix_v_b_proj_full=None,
        _helix_v_up_before_exchange=False,
    )
    module._bind_helix_v_up_weight = partial(MLA._bind_helix_v_up_weight, module)
    return module


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_helix_v_up_weight_from_fp8_kv_b(cp_rank: int) -> None:
    """MLA dequantizes every tp-local head's V weight from the loaded FP8 kv_b_proj; this rank's
    dequantized V chunk becomes a view of it."""
    h, n, v, k, cp = 4, 128, 128, 256, 2
    generator = torch.Generator().manual_seed(cp_rank)
    weight = (torch.randn(h * (n + v), k, generator=generator) * 64).to(torch.float8_e4m3fn)
    scale = torch.rand(h * (n + v) // 128, k // 128, generator=generator) + 0.5
    module = _v_up_builder(
        weight, scale, num_heads=h, cp_size=cp, cp_rank=cp_rank, nope=n, v=v, k=k
    )
    module.v_b_proj_dequant = torch.empty(h // cp, v, k, dtype=torch.bfloat16)
    MLA._build_helix_v_up_weight(module)

    value_codes = weight[h * n :].view(h, v, k)
    value_scale = scale[h * n // 128 :].view(h, v // 128, k // 128)
    expanded = value_scale.repeat_interleave(128, 1).repeat_interleave(128, 2)
    full = (value_codes.float() * expanded).to(torch.bfloat16)
    assert module._helix_v_up_before_exchange
    assert torch.equal(module.helix_v_b_proj_full, full)
    chunk = slice(cp_rank * h // cp, (cp_rank + 1) * h // cp)
    assert torch.equal(module.v_b_proj_dequant, full[chunk])
    assert (
        module.v_b_proj_dequant.untyped_storage().data_ptr()
        == module.helix_v_b_proj_full.untyped_storage().data_ptr()
    )


def test_helix_v_up_weight_views_bf16_kv_b() -> None:
    h, n, v, k = 4, 64, 128, 256
    weight = torch.randn(h * (n + v), k, dtype=torch.bfloat16)
    module = _v_up_builder(weight, None, num_heads=h, cp_size=2, cp_rank=0, nope=n, v=v, k=k)
    MLA._build_helix_v_up_weight(module)
    assert module._helix_v_up_before_exchange
    assert torch.equal(module.helix_v_b_proj_full, weight[h * n :].view(h, v, k))
    assert module.helix_v_b_proj_full.data_ptr() == weight[h * n :].data_ptr()


def test_helix_v_up_weight_needs_loaded_fp8_scales():
    """A kv_b_proj whose scales are already in a GEMM layout keeps V after the exchange."""
    h, n, v, k = 4, 128, 128, 256
    weight = torch.zeros(h * (n + v), k, dtype=torch.float8_e4m3fn)
    packed_scale = torch.zeros(k // 128 // 4 + 1, h * (n + v), dtype=torch.int32)
    module = _v_up_builder(
        weight, packed_scale, num_heads=h, cp_size=2, cp_rank=0, nope=n, v=v, k=k
    )
    MLA._build_helix_v_up_weight(module)
    assert not module._helix_v_up_before_exchange
    assert module.helix_v_b_proj_full is None
