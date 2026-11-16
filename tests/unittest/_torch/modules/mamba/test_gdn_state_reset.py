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


def _cpp_style_pool(
    num_layers, num_slots, heads, head_dim, d_state, conv_dim, d_conv, ssm_dtype, dev
):
    """One byte pool per (layer, slot) holding ssm then conv, viewed like the real Mamba layer cache."""
    ssm_bytes = heads * head_dim * d_state * ssm_dtype.itemsize
    conv_bytes = conv_dim * (d_conv - 1) * 2
    pool = torch.randint(
        1, 255, (num_layers, num_slots, ssm_bytes + conv_bytes), dtype=torch.uint8, device=dev
    )
    ssm = (
        pool[:, :, :ssm_bytes].view(ssm_dtype).view(num_layers, num_slots, heads, head_dim, d_state)
    )
    conv = (
        pool[:, :, ssm_bytes:]
        .view(torch.bfloat16)
        .view(num_layers, num_slots, conv_dim, d_conv - 1)
    )
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


@needs_cuda
def test_reset_prefill_states_runs_once_per_iteration_on_the_real_pool():
    """The first GDN layer clears every layer's slots; later layers and gated-off iterations leave the pool alone."""
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet

    reset = Qwen3NextGatedDeltaNet._reset_prefill_states
    dev = "cuda"
    torch.manual_seed(0)
    L, S = 3, 6
    pool, ssm, conv = _cpp_style_pool(L, S, 4, 32, 32, 96, 4, torch.bfloat16, dev)
    original = pool.clone()
    slots = torch.tensor([1, 4], dtype=torch.int32, device=dev)
    fresh = torch.tensor([False, False], dtype=torch.bool, device=dev)
    attn = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(all_ssm_states=ssm, all_conv_states=conv)
    )

    def run_layer(meta, layer):
        reset(None, attn, meta, ssm[layer], conv[layer], slots, fresh)
        torch.cuda.synchronize()

    # Nothing to reset this iteration: no layer touches the pool.
    quiet = SimpleNamespace(prefill_needs_state_reset=False, state_reset_done=False)
    for layer in range(L):
        run_layer(quiet, layer)
    assert torch.equal(pool, original)

    meta = SimpleNamespace(prefill_needs_state_reset=True, state_reset_done=False)
    run_layer(meta, 0)
    assert meta.state_reset_done
    for layer in range(L):  # layer 0 cleared the slots of every layer
        for slot in (1, 4):
            assert not ssm[layer, slot].any() and not conv[layer, slot].any()
        for slot in (0, 2, 3, 5):
            assert torch.equal(pool[layer, slot], original[layer, slot])

    # The remaining layers skip: state written after the reset survives them.
    ssm[1, 1].fill_(3)
    run_layer(meta, 1)
    run_layer(meta, 2)
    assert (ssm[1, 1] == 3).all()

    # The next iteration clears again.
    meta.state_reset_done = False
    run_layer(meta, 1)
    assert not ssm[1, 1].any()


@needs_cuda
def test_reset_prefill_states_falls_back_to_the_layer_without_an_all_layer_pool():
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet

    dev = "cuda"
    torch.manual_seed(1)
    L, S = 2, 4
    pool, ssm, conv = _cpp_style_pool(L, S, 4, 32, 32, 96, 4, torch.bfloat16, dev)
    original = pool.clone()
    slots = torch.tensor([2], dtype=torch.int32, device=dev)
    fresh = torch.tensor([False], dtype=torch.bool, device=dev)
    attn = SimpleNamespace(kv_cache_manager=SimpleNamespace())  # exposes no all-layer pool
    meta = SimpleNamespace(prefill_needs_state_reset=True, state_reset_done=False)

    Qwen3NextGatedDeltaNet._reset_prefill_states(None, attn, meta, ssm[0], conv[0], slots, fresh)
    torch.cuda.synchronize()

    assert not meta.state_reset_done
    assert not ssm[0, 2].any() and not conv[0, 2].any()
    assert torch.equal(pool[1], original[1])  # only this layer's rows were cleared
