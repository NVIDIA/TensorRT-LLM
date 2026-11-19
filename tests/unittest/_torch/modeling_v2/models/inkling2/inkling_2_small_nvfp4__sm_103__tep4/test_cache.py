# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Inkling 2 KV cache manager: block reuse refused, convolution slot ownership."""

from types import SimpleNamespace

import pytest

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
