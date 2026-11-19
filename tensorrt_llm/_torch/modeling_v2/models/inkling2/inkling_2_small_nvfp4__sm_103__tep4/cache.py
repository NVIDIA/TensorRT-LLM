# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`KVCacheManagerV2` with Inkling 2's short-convolution windows.

Every decoder layer convolves four streams over the last 4 tokens: its keys and values (this
rank's KV heads) and its attention and MoE outputs (the full hidden width). A request's step
needs the 3 inputs of each stream before its first new token, so the manager keeps them, per
layer and request, in a slot of four tensors `[slots, 3, C]` (bf16): `k`, `v`, `attn`, `mlp`.

A request takes a slot the first time a step names it and keeps it until the engine frees its
KV cache. The slot pool holds `max_batch_size` requests plus the engine's padding and warmup
requests. The windows are not part of the KV cache's prefix reuse, so block reuse must be off.
"""

from __future__ import annotations

from typing import Iterable

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

from ..config import InklingDims

#: Slots beyond the resident requests: CUDA-graph padding and warmup requests.
SPARE_SLOTS = 8
#: Inputs a width-4 window keeps.
WINDOW = 3


class ConvSlots:
    """Which short-convolution slot each live request owns.

    A request keeps its slot, and so its windows, from its first step until it is released, also
    while it is paused; a released slot goes to the next new request.
    """

    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self._slot_of: dict[int, int] = {}
        self._free = list(range(num_slots - 1, -1, -1))

    def acquire(self, request_ids: Iterable[int]) -> list[int]:
        """Each request's slot; a request seen for the first time gets a free one."""
        slots = []
        for request_id in request_ids:
            slot = self._slot_of.get(request_id)
            if slot is None:
                if not self._free:
                    raise RuntimeError(
                        f"all {self.num_slots} short-convolution slots are in use; "
                        "more requests are live than the KV cache manager admits"
                    )
                slot = self._free.pop()
                self._slot_of[request_id] = slot
            slots.append(slot)
        return slots

    def release(self, request_id: int) -> None:
        """Free the request's slot; a request that never got one is ignored."""
        slot = self._slot_of.pop(request_id, None)
        if slot is not None:
            self._free.append(slot)


class Inkling2KVCacheManager(KVCacheManagerV2):
    """The KV cache plus each request's short-convolution windows, for every rank-local layer.

    Block reuse is refused: a reused prefix would come without the convolution windows of the
    tokens it covers.
    """

    def __init__(
        self, *args, pretrained_config=None, mapping=None, max_batch_size: int = 1, **kwargs
    ):
        super().__init__(
            *args,
            pretrained_config=pretrained_config,
            mapping=mapping,
            max_batch_size=max_batch_size,
            **kwargs,
        )
        if self.enable_block_reuse:
            raise ValueError(
                "Inkling2KVCacheManager does not support KV cache block reuse: a reused prefix "
                "comes without its short-convolution windows. Set "
                "kv_cache_config.enable_block_reuse to false."
            )
        text = getattr(pretrained_config, "text_config", None) or pretrained_config
        dims = InklingDims.from_config(text)
        tp = mapping.tp_size if mapping is not None else 1
        kv_width = dims.num_kv_heads * dims.head_dim // tp
        # Every sequence the manager admits holds its windows: a retiring request keeps its slot
        # one more iteration under the overlap scheduler, and a request paused by GEN preemption
        # keeps its slot, so the base manager's own count sizes the pool.
        self.num_conv_slots = self.max_admissible_sequences + SPARE_SLOTS
        widths = {"k": kv_width, "v": kv_width, "attn": dims.hidden_size, "mlp": dims.hidden_size}
        device = torch.device("cuda", torch.cuda.current_device())
        self._conv_state = [
            {
                name: torch.zeros(
                    self.num_conv_slots, WINDOW, width, dtype=torch.bfloat16, device=device
                )
                for name, width in widths.items()
            }
            for _ in range(dims.num_layers)
        ]
        self._conv_slots = ConvSlots(self.num_conv_slots)

    def conv_state(self, layer: int) -> dict:
        """Layer `layer`'s windows: `k`, `v`, `attn`, `mlp`, each `[slots, 3, C]` bf16."""
        return self._conv_state[layer]

    def conv_slots(self, request_ids: Iterable[int]) -> torch.Tensor:
        """Each request's slot, as int32 on the device; a request seen for the first time gets one."""
        return torch.tensor(self._conv_slots.acquire(request_ids), dtype=torch.int32).to(
            self._conv_state[0]["k"].device, non_blocking=True
        )

    def free_resources(self, request, pin_on_release: bool = False):
        self._conv_slots.release(request.py_request_id)
        return super().free_resources(request, pin_on_release)
