# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact timestep-keyed AdaLN tables for immutable, instance-owned H3 weights."""

import itertools
import types
import weakref

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
        if key in owner.keys or len(owner.keys) < owner.max_keys:
            owner.keys.add(key)
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
    block_id: int,
) -> torch.Tensor:
    owner = _OWNERS[cache_id]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("The conditioning cache is not qualified inside CUDA Graph capture")
    if owner.key is None:
        owner.fallbacks += 1
        return owner.tables[block_id](temb)
    key = (block_id, owner.key, tuple(temb.shape), str(temb.dtype), str(temb.device))
    if key not in owner.cache:
        owner.cache[key] = owner.tables[block_id](temb)
        owner.misses += 1
    else:
        owner.hits += 1
    return owner.cache[key]


@table_lookup.register_fake
def _table_fake(temb, weight, bias, cache_id, block_id):
    return torch.empty(
        (temb.shape[0] * 3, 6 * 5376),
        device=temb.device,
        dtype=_OWNERS[cache_id].table_dtypes[block_id],
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
    )


class ConditioningCache:
    """One immutable pipeline's scalar-conditioning lookup; no latents are cached.

    The exact conditioning bytes, shape and dtype form the request key. The key
    read and its synchronization execute inside each Transformer call. A bounded
    key set keeps requests with new schedules from growing the cache indefinitely.
    LoRA, weight offload and in-place weight reload are outside this scenario.
    """

    def __init__(self, transformer, max_keys: int = 64) -> None:
        self.identifier = next(_IDENTIFIERS)
        self.max_keys = max_keys
        self.key = None
        self.keys = set()
        self.cache = {}
        self.hits = self.misses = self.fallbacks = self.key_reads = 0
        self.time_dim = int(transformer.time_proj.num_channels)
        self.original_time = transformer.time_proj.forward
        self.tables = [block.adaln_proj.table for block in transformer.transformer_blocks]
        self.table_dtypes = [
            block.adaln_proj.linear.dtype for block in transformer.transformer_blocks
        ]
        _OWNERS[self.identifier] = self
        transformer.time_proj._h3_fp8_cache_id = self.identifier
        transformer.time_proj.forward = types.MethodType(_time_forward, transformer.time_proj)
        for index, block in enumerate(transformer.transformer_blocks):
            block.adaln_proj._h3_fp8_cache_id = self.identifier
            block.adaln_proj._h3_fp8_block_id = index
            block.adaln_proj.table = types.MethodType(_table, block.adaln_proj)
