# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Inkling 2 KV cache manager: block reuse refused, convolution slot ownership."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.inkling2.inkling_2_small_nvfp4__sm_103__tep4 import (
    cache,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only


def test_block_reuse_is_refused_before_any_allocation(monkeypatch):
    def base_init(self, *args, **kwargs):
        self.enable_block_reuse = True
        self.gen_preemption_headroom = 0

    monkeypatch.setattr(KVCacheManagerV2, "__init__", base_init)
    # Anything past the refusal would read the config or allocate on the GPU.
    with pytest.raises(ValueError, match="enable_block_reuse"):
        cache.Inkling2KVCacheManager(
            pretrained_config=SimpleNamespace(), mapping=None, max_batch_size=4
        )


def _small_text_config() -> SimpleNamespace:
    """Two layers (the second global) of the Inkling 2 text config, small enough for a CPU."""
    return SimpleNamespace(
        num_hidden_layers=2,
        local_layer_ids=[0],
        hidden_size=64,
        vocab_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        d_rel=16,
        rel_extent=64,
        sliding_window_size=64,
        sconv_kernel_size=4,
        n_layers_per_attn_res_block=6,
        n_routed_experts=8,
        num_experts_per_tok=2,
        intermediate_size=128,
        route_scale=8.0,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        final_logit_softcapping=30.0,
        qk_l2_clip_threshold=22.627416997969522,
        rel_l2_clip_threshold=8.0,
        log_scaling_alpha=0.1,
        log_scaling_n_floor=8192,
        rms_norm_eps=1e-6,
    )


def test_conv_slots_cover_requests_paused_by_gen_preemption(monkeypatch):
    """Paused requests stay resident with their windows: the pool holds them beyond the batch."""

    def base_init(self, *args, **kwargs):
        self.enable_block_reuse = False
        self.gen_preemption_headroom = 9

    monkeypatch.setattr(KVCacheManagerV2, "__init__", base_init)
    # The windows are allocated on the current CUDA device; a CPU stands in for it here.
    zeros = torch.zeros
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch, "zeros", lambda *shape, dtype=None, device=None: zeros(*shape, dtype=dtype)
    )
    manager = cache.Inkling2KVCacheManager(
        pretrained_config=SimpleNamespace(text_config=_small_text_config()),
        mapping=SimpleNamespace(tp_size=4),
        max_batch_size=23,
    )
    assert manager.num_conv_slots == 23 + 9 + cache.SPARE_SLOTS == 40
    assert all(t.shape[0] == 40 for t in manager.conv_state(1).values())

    resident = [f"request-{i}" for i in range(23 + 9)]
    slots = manager.conv_slots(resident).tolist()
    assert len(set(slots)) == 32
    # The earliest owners keep their slots while the rest are resident.
    assert manager.conv_slots(resident[:3]).tolist() == slots[:3]
    # The spare slots (CUDA-graph padding, warmup) are still there.
    assert len(set(manager.conv_slots([f"spare-{i}" for i in range(8)]).tolist())) == 8
    with pytest.raises(RuntimeError, match="all 40 short-convolution slots are in use"):
        manager.conv_slots(["one-too-many"])


def test_a_released_slot_goes_to_the_next_request_and_others_keep_theirs():
    slots = cache.ConvSlots(3)
    a, b = slots.acquire(["a", "b"])
    assert a != b
    # A live request, paused or not, keeps its slot from step to step.
    assert slots.acquire(["b", "a"]) == [b, a]
    slots.release("a")
    (c,) = slots.acquire(["c"])
    assert c == a
    assert slots.acquire(["b", "c"]) == [b, c]
    # Releasing a request that holds no slot changes nothing.
    slots.release("never-seen")
    slots.release("a")
    assert slots.acquire(["b", "c"]) == [b, c]


def test_slots_run_out_only_when_every_one_is_held():
    slots = cache.ConvSlots(2)
    slots.acquire(["a", "b"])
    with pytest.raises(RuntimeError, match="all 2 short-convolution slots are in use"):
        slots.acquire(["c"])
    slots.release("b")
    assert slots.acquire(["c"]) == [1]
