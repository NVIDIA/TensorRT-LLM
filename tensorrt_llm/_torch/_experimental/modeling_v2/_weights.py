# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The weight table and the three passes over it, shared by every model.

A model states its weights once, as a `W` per role, and subclasses
`ModelWeights` to say how the dimensions are derived. Declaration, checkpoint
manifest and load all unroll that one table, so a role cannot be declared
without a source or loaded into a shape nobody declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, what happens on the way.

    `shape` takes the dimension bundle and returns the declared shape; every
    shape in these models is derived from configuration rather than constant.

    `src` is a checkpoint key template with `{p}` standing for the layer
    prefix, or a sequence of `(template, slice_fn)` pairs when one parameter is
    assembled from several checkpoint tensors. `slice_fn` takes the dimension
    bundle and returns an index into the destination parameter.

    `transform` runs on the source tensor before the copy, on the
    destination's device. `None` means the checkpoint tensor is already the
    declared shape.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | tuple[tuple[str, Callable], ...]
    dtype: torch.dtype | None = None
    transform: Callable | None = None
    per_layer: bool = True


class ModelWeights:
    """Shared base for a model's weight table.

    Subclasses supply `WEIGHTS` and `dims`; everything else here is one pass
    over that table. A model whose load differs structurally overrides `load`
    rather than growing a flag on this class.
    """

    WEIGHTS: tuple[W, ...] = ()

    LAYER_PREFIX: str = "model.layers.{i}"

    # Param-key suffix after which to release transform scratch. The expert
    # transforms allocate several GB per layer; without this, peak load
    # memory is the whole model rather than one layer.
    RELEASE_AFTER: str | None = None

    def dims(self, core) -> SimpleNamespace:
        """The dimension bundle every pass is computed from.

        The single place a derived width or padded size is worked out. Reads
        `core.model_config` only, so it can run before the core has set
        anything on itself.
        """
        raise NotImplementedError

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares.

        Meta-init intercepts `torch.empty` here -- real CUDA storage arrives
        when the engine materializes the registry.
        """
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            keys = (
                [f"l{i}_{entry.name}" for i in range(d.num_layers)]
                if entry.per_layer
                else [entry.name]
            )
            for key in keys:
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def manifest(self, core) -> dict[str, list[tuple[str, Any, Callable | None]]]:
        """param key -> [(checkpoint key, index into the param | None, transform | None)]."""
        d = self.dims(core)
        rows: dict[str, list[tuple[str, Any, Callable | None]]] = {}
        for entry in self.WEIGHTS:
            targets = (
                [(f"l{i}_{entry.name}", self.LAYER_PREFIX.format(i=i)) for i in range(d.num_layers)]
                if entry.per_layer
                else [(entry.name, "")]
            )
            for key, prefix in targets:
                if isinstance(entry.src, str):
                    rows[key] = [(entry.src.format(p=prefix), None, entry.transform)]
                else:
                    rows[key] = [
                        (template.format(p=prefix), slice_fn(d), entry.transform)
                        for template, slice_fn in entry.src
                    ]
        return rows

    def expected_unconsumed(self, core, weights) -> set[str]:
        """Checkpoint keys this rank is meant to leave untouched."""
        return set()

    def load(self, model, weights) -> None:
        core = model.model
        manifest = self.manifest(core)
        consumed: set[str] = set()

        def fill(param: nn.Parameter, ckpt_key: str, index, transform) -> None:
            assert ckpt_key in weights, f"checkpoint key missing: {ckpt_key}"
            src = weights[ckpt_key][:]  # materialize the lazy slice
            dst = param.data if index is None else param.data[index]
            if transform is not None:
                src = transform(src.to(dst.device, non_blocking=True), core)
            assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (ckpt_key, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.add(ckpt_key)

        for param_key, sources in manifest.items():
            for ckpt_key, index, transform in sources:
                fill(core.w[param_key], ckpt_key, index, transform)
            if self.RELEASE_AFTER is not None and param_key.endswith(self.RELEASE_AFTER):
                torch.cuda.empty_cache()

        # Shell-registered exception: the base class owns lm_head (untied).
        fill(model.lm_head.weight, "lm_head.weight", None, None)

        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed - self.expected_unconsumed(core, weights)
        assert not leftover, f"unconsumed checkpoint keys: {sorted(leftover)[:8]}"
