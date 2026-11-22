# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact timestep-keyed AdaLN tables for immutable, instance-owned H3 weights."""

import itertools
import types
import weakref
from collections import OrderedDict

import torch

_IDENTIFIERS = itertools.count()
_OWNERS = weakref.WeakValueDictionary()


@torch.library.custom_op("h3_fp8::conditioning_time", mutates_args=())
def time_project(conditioning: torch.Tensor, cache_id: int) -> torch.Tensor:
    owner = _OWNERS[cache_id]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("The conditioning cache is not qualified inside CUDA Graph capture")
    owner.key = None
    if (
        conditioning.device.type == "cuda"
        and conditioning.ndim == 1
        and conditioning.numel() in (1, 2)
    ):
        value = conditioning.detach().contiguous().cpu()
        key = (tuple(value.shape), str(value.dtype), bytes(value.view(torch.uint8).tolist()))
        owner.key_reads += 1
        owner.remember(key)
        owner.key = key
    return owner.original_time(conditioning)


@time_project.register_fake
def _time_fake(conditioning, cache_id):
    return torch.empty(
        (conditioning.shape[0], _OWNERS[cache_id].time_dim),
        device=conditioning.device,
        dtype=torch.float32,
    )


@torch.library.custom_op("h3_fp8::conditioning_table", mutates_args=())
def table_lookup(
    temb: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    cache_id: int,
    block_id: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    owner = _OWNERS[cache_id]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("The conditioning cache is not qualified inside CUDA Graph capture")
    if block_id.device.type != "cpu":
        raise ValueError("Conditioning block identifiers must stay on the CPU")
    index = int(block_id.item())
    if owner.key is None:
        owner.fallbacks += 1
        return owner.tables[index](temb)
    key = (index, owner.key, tuple(temb.shape), str(temb.dtype), str(temb.device))
    if key not in owner.cache:
        owner.cache[key] = owner.tables[index](temb)
        owner.misses += 1
    else:
        owner.hits += 1
    return owner.cache[key]


@table_lookup.register_fake
def _table_fake(temb, weight, bias, cache_id, block_id, output_dtype):
    return torch.empty(
        (temb.shape[0] * 3, 6 * 5376),
        device=temb.device,
        dtype=output_dtype,
    )


def _time_forward(module, conditioning: torch.Tensor) -> torch.Tensor:
    return time_project(conditioning, module._h3_fp8_cache_id)


def _table(module, temb: torch.Tensor) -> torch.Tensor:
    return table_lookup(
        temb,
        module.linear.weight,
        module.linear.bias,
        module._h3_fp8_cache_id,
        module._h3_fp8_block_id,
        module.linear.dtype,
    )


class ConditioningCache:
    """One immutable pipeline's scalar-conditioning lookup; no latents are cached.

    The exact conditioning bytes, shape and dtype form the request key. The key
    read and its synchronization execute inside each Transformer call. At most
    ``max_keys`` keys are kept; a new key evicts the least recently used one with
    its tables, so the cache follows the schedules in use and stays bounded.
    LoRA, weight offload and in-place weight reload are outside this scenario.
    """

    def __init__(self, transformer, max_keys: int = 64) -> None:
        self.identifier = next(_IDENTIFIERS)
        self.max_keys = max_keys
        self.key = None
        self.keys = OrderedDict()
        self.cache = {}
        self.evictions = 0
        self.hits = self.misses = self.fallbacks = self.key_reads = 0
        self.time_dim = int(transformer.time_proj.num_channels)
        self.original_time = transformer.time_proj.forward
        self.tables = [block.adaln_proj.table for block in transformer.transformer_blocks]
        _OWNERS[self.identifier] = self
        self._install(transformer)

    def remember(self, key) -> None:
        """Mark ``key`` as the most recently used; evict the oldest key beyond ``max_keys``."""
        if key in self.keys:
            self.keys.move_to_end(key)
            return
        self.keys[key] = None
        while len(self.keys) > self.max_keys:
            old, _ = self.keys.popitem(last=False)
            for entry in [entry for entry in self.cache if entry[1] == old]:
                del self.cache[entry]
            self.evictions += 1

    def _install(self, transformer) -> None:
        transformer.time_proj._h3_fp8_cache_id = self.identifier
        transformer.time_proj.forward = types.MethodType(_time_forward, transformer.time_proj)
        for index, block in enumerate(transformer.transformer_blocks):
            block.adaln_proj._h3_fp8_cache_id = self.identifier
            block.adaln_proj._h3_fp8_block_id = torch.tensor(index, device="cpu", dtype=torch.int64)
            block.adaln_proj.table = types.MethodType(_table, block.adaln_proj)
