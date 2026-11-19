# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep exact AdaLN cache bookkeeping behind explicit opaque compiler boundaries.

Same immutable-weight / one-pipeline scope as v1. The timestep key is read once,
inside a time-projection op. Table ops depend on its actual temb output and take
the actual weight/bias tensors, so key mutation precedes lookup on this stream.
This avoids tracing a changing Python dictionary into every compiled block.
"""

import types

import round4_cache_base as base
import torch

ACTIVE = None


@torch.library.custom_op("h3_fp8_round4_cache::time", mutates_args=())
def time_project(conditioning: torch.Tensor) -> torch.Tensor:
    owner = ACTIVE
    assert owner is not None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Conditioning-keyed cache lookup is not qualified inside CUDA Graph capture"
        )
    eligible = (
        conditioning.device.type == "cuda"
        and conditioning.ndim == 1
        and conditioning.numel() in (1, 2)
    )
    owner.key = None
    if eligible:
        value = conditioning.detach().contiguous().cpu()
        owner.key = (tuple(value.shape), str(value.dtype), bytes(value.view(torch.uint8).tolist()))
        owner.key_reads += 1
        if owner.key not in owner.keys:
            if len(owner.keys) < owner.max_keys:
                owner.keys.add(owner.key)
            else:
                owner.key = None
    return owner.original_time(conditioning)


@time_project.register_fake
def _(conditioning):
    assert ACTIVE is not None
    return torch.empty(
        (conditioning.shape[0], ACTIVE.time_dim), device=conditioning.device, dtype=torch.float32
    )


@torch.library.custom_op("h3_fp8_round4_cache::table", mutates_args=())
def table_lookup(temb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    owner = ACTIVE
    assert owner is not None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Conditioning-keyed cache lookup is not qualified inside CUDA Graph capture"
        )
    index = owner.weight_to_index[weight.data_ptr()]
    original = owner.tables[index]
    if owner.key is None:
        owner.fallbacks += 1
        return original(temb)
    key = (index, owner.key, tuple(temb.shape), str(temb.dtype), str(temb.device))
    if key not in owner.cache:
        owner.misses += 1
        owner.cache[key] = original(temb)
    else:
        owner.hits += 1
        if owner.validate:
            reference = original(temb)
            value = owner.cache[key]
            equal = bool(torch.equal(value.view(torch.uint8), reference.view(torch.uint8)))
            row = dict(index=len(owner.checks), block=index, byte_equal=equal)
            owner.checks.append(row)
            assert equal, row
    return owner.cache[key]


@table_lookup.register_fake
def _(temb, weight, bias):
    return torch.empty((temb.shape[0] * 3, 6 * 5376), device=temb.device, dtype=weight.dtype)


def time_forward(self, conditioning, *args, **kwargs):
    assert not args and not kwargs
    return time_project(conditioning)


def table(self, temb):
    return table_lookup(temb, self.linear.weight, self.linear.bias)


class Patch(base.Patch):
    def __init__(self, transformer, max_keys=64):
        super().__init__(transformer, max_keys=max_keys)
        self.time_dim = int(transformer.time_proj.num_channels)
        self.weight_to_index = {}
        for index, block in enumerate(self.blocks):
            linear = block.adaln_proj.linear
            assert linear.bias is not None and linear.weight.shape[0] == 3 * 6 * 5376
            self.weight_to_index[linear.weight.data_ptr()] = index
        assert len(self.weight_to_index) == 50
        self.bound_time = types.MethodType(time_forward, transformer.time_proj)
        self.bound_tables = [types.MethodType(table, block.adaln_proj) for block in self.blocks]

    def configure(self, enabled, validate=False):
        global ACTIVE
        if enabled:
            ACTIVE = self
        super().configure(enabled, validate)
