# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Research-only exact conditioning-keyed cache for immutable post-load weights.

The actual conditioning tensor bytes are read once at time_proj per Transformer
call, including that synchronization cost in requests. No latent/hidden output is
cached. Weights must remain frozen for this instance; no LoRA/offload/training.
Module identities/parameter pointers/dtypes are guarded. This is not a general
in-place-weight-mutation detector or a public serving cache implementation.
"""

import types

import torch


def stamp(module):
    return tuple(
        (name, p.data_ptr(), tuple(p.shape), str(p.dtype), str(p.device))
        for name, p in module.named_parameters()
    )


class Patch:
    def __init__(self, transformer, max_keys=64):
        self.model = transformer
        self.max_keys = max_keys
        self.key = None
        self.keys = set()
        self.cache = {}
        self.hits = self.misses = self.fallbacks = self.key_reads = 0
        self.validate = False
        self.checks = []
        self.original_time = transformer.time_proj.forward
        self.previous_time = (
            "forward" in vars(transformer.time_proj),
            vars(transformer.time_proj).get("forward"),
        )
        self.blocks = transformer.transformer_blocks
        self.tables = [b.adaln_proj.table for b in self.blocks]
        self.previous_tables = [
            ("table" in vars(b.adaln_proj), vars(b.adaln_proj).get("table")) for b in self.blocks
        ]
        self.modules = [transformer.time_proj, transformer.time_embedder]
        self.modules.extend(b.adaln_proj for b in self.blocks)
        self.stamps = [stamp(m) for m in self.modules]
        for module in self.modules:
            assert not module.training

        def project(owner, conditioning, *args, **kwargs):
            # The exact values, shape and dtype are keys; pointer reuse is irrelevant.
            eligible = (
                conditioning.device.type == "cuda"
                and conditioning.ndim == 1
                and conditioning.numel() in (1, 2)
            )
            if eligible:
                value = conditioning.detach().contiguous().cpu()
                self.key = (
                    tuple(value.shape),
                    str(value.dtype),
                    bytes(value.view(torch.uint8).tolist()),
                )
                self.key_reads += 1
                if self.key not in self.keys:
                    if len(self.keys) < self.max_keys:
                        self.keys.add(self.key)
                    else:
                        self.key = None
            else:
                self.key = None
            return self.original_time(conditioning, *args, **kwargs)

        self.bound_time = types.MethodType(project, transformer.time_proj)
        self.bound_tables = []
        for index, original in enumerate(self.tables):

            def table(owner, temb, index=index, original=original):
                if self.key is None:
                    self.fallbacks += 1
                    return original(temb)
                key = (index, self.key, tuple(temb.shape), str(temb.dtype), str(temb.device))
                if key not in self.cache:
                    self.misses += 1
                    self.cache[key] = original(temb)
                else:
                    self.hits += 1
                    if self.validate:
                        reference = original(temb)
                        value = self.cache[key]
                        equal = bool(
                            torch.equal(value.view(torch.uint8), reference.view(torch.uint8))
                        )
                        row = dict(index=len(self.checks), block=index, byte_equal=equal)
                        self.checks.append(row)
                        assert equal, row
                return self.cache[key]

            self.bound_tables.append(types.MethodType(table, self.blocks[index].adaln_proj))

    def configure(self, enabled, validate=False):
        self.validate = validate
        self.key = None
        if enabled:
            assert all(
                not m.training and stamp(m) == expected
                for m, expected in zip(self.modules, self.stamps)
            )
            self.model.time_proj.forward = self.bound_time
            for block, table in zip(self.blocks, self.bound_tables):
                block.adaln_proj.table = table
        else:
            self._restore(self.model.time_proj, "forward", self.previous_time)
            for block, previous in zip(self.blocks, self.previous_tables):
                self._restore(block.adaln_proj, "table", previous)

    @staticmethod
    def _restore(module, name, previous):
        if previous[0]:
            setattr(module, name, previous[1])
        elif name in vars(module):
            delattr(module, name)
