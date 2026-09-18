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
"""Host-gated, all-layer GDN state reset (gdn_mixer._reset_prefill_states).

The per-layer masked reset used to run in every GDN layer of every iteration
with context requests. The all-layer kernel clears the same slots of every
layer in one launch; these tests pin it bit-exact against the per-layer loop
on a C++-manager-style strided pool and check the host gate around it.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.gdn_mixer import _reset_gdn_states_all_layers
from tensorrt_llm._torch.modules.mamba.recurrent_state_cache import reset_recurrent_state_rows

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cpp_style_pool(num_layers, num_slots, heads, head_dim, d_state, conv_dim, d_conv, ssm_dtype, dev):
    """One byte pool per (layer, slot) holding ssm then conv, viewed like CppMambaHybridCacheManager."""
    ssm_bytes = heads * head_dim * d_state * ssm_dtype.itemsize
    conv_bytes = conv_dim * (d_conv - 1) * 2
    pool = torch.randint(1, 255, (num_layers, num_slots, ssm_bytes + conv_bytes), dtype=torch.uint8, device=dev)
    ssm = pool[:, :, :ssm_bytes].view(ssm_dtype).view(num_layers, num_slots, heads, head_dim, d_state)
    conv = pool[:, :, ssm_bytes:].view(torch.bfloat16).view(num_layers, num_slots, conv_dim, d_conv - 1)
    return pool, ssm, conv


@needs_cuda
@pytest.mark.parametrize("ssm_dtype", [torch.bfloat16, torch.float32], ids=["bf16_ssm", "fp32_ssm"])
def test_all_layer_reset_matches_per_layer_loop(ssm_dtype):
    dev = "cuda"
    torch.manual_seed(0)
    L, S = 3, 8
    pool, ssm, conv = _cpp_style_pool(L, S, 4, 32, 32, 96, 4, ssm_dtype, dev)
    assert ssm.stride(1) != ssm[0, 0].numel(), "test pool must be strided along the slot dim"
    original = pool.clone()
    state_indices = torch.tensor([2, 5, 7, -1], dtype=torch.int32, device=dev)
    has_initial = torch.tensor([False, True, False, False], dtype=torch.bool, device=dev)

    ref_pool = pool.clone()
    ssm_bytes = ssm[0, 0].numel() * ssm_dtype.itemsize
    ref_ssm = ref_pool[:, :, :ssm_bytes].view(ssm_dtype).view(ssm.shape)
    ref_conv = ref_pool[:, :, ssm_bytes:].view(torch.bfloat16).view(conv.shape)
    for layer in range(L):
        reset_recurrent_state_rows(ref_ssm[layer], state_indices, has_initial, ref_conv[layer])

    _reset_gdn_states_all_layers(ssm, conv, state_indices, has_initial)
    torch.cuda.synchronize()

    assert torch.equal(pool, ref_pool)
    for layer in range(L):
        for slot in (2, 7):  # no initial state -> zeroed in every layer
            assert not ssm[layer, slot].any() and not conv[layer, slot].any()
        for slot in (0, 1, 3, 4, 5, 6):  # has initial state / not in the batch -> untouched
            assert torch.equal(pool[layer, slot], original[layer, slot])


class _Meta:
    prefill_needs_state_reset = False
    state_reset_done = False


def test_reset_prefill_states_gate_and_hoist(monkeypatch):
    """No launch when nothing needs a reset; one all-layer launch per iteration otherwise."""
    from tensorrt_llm._torch.modules.mamba import gdn_mixer as gm

    calls = []
    monkeypatch.setattr(gm, "_reset_gdn_states_all_layers", lambda *a: calls.append("all"))
    monkeypatch.setattr(gm, "reset_recurrent_state_rows", lambda *a: calls.append("layer"))
    ssm = torch.zeros(2, 3, 4, 8, 8)
    conv = torch.zeros(2, 3, 16, 3)
    attn = SimpleNamespace(kv_cache_manager=SimpleNamespace(all_ssm_states=ssm, all_conv_states=conv))
    layer = SimpleNamespace(_reset_prefill_states=gm.Qwen3NextGatedDeltaNet._reset_prefill_states)
    args = (attn, None, ssm[0], conv[0], torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.bool))

    meta = _Meta()
    layer._reset_prefill_states(layer, attn, meta, *args[2:])
    assert calls == []                                  # gated off: nothing to reset
    meta.prefill_needs_state_reset = True
    layer._reset_prefill_states(layer, attn, meta, *args[2:])
    layer._reset_prefill_states(layer, attn, meta, *args[2:])
    assert calls == ["all"] and meta.state_reset_done   # first layer resets every layer, the next skips
    meta.state_reset_done = False
    attn_no_pool = SimpleNamespace(kv_cache_manager=SimpleNamespace())
    layer._reset_prefill_states(layer, attn_no_pool, meta, *args[2:])
    layer._reset_prefill_states(layer, attn_no_pool, meta, *args[2:])
    assert calls == ["all", "layer", "layer"] and not meta.state_reset_done   # no all-layer pool: per-layer fallback
