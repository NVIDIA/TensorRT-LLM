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

#: `layers=ALL_LAYERS` is the common case and the default. A sentinel rather
#: than a lambda so the table reads as data, and rather than `True` so the
#: field has one type of answer: which layers.
ALL_LAYERS = object()


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, what happens on the way.

    `shape` takes the dimension bundle and returns the declared shape; every
    shape in these models is derived from configuration rather than constant.

    `src` says where the bytes come from, in one of three forms:

    * a checkpoint key template, with `{p}` standing for the layer prefix;
    * a sequence of `(template, slice_fn)` pairs, when one parameter is
      assembled from several checkpoint tensors side by side -- `slice_fn`
      takes the dimension bundle and returns an index into the destination;
    * a callable `(d, i) -> [(key, index), ...]`, taking the dimension bundle
      and the layer index. The first two forms are special cases of it; this
      one exists for sources that are not a fixed template -- a per-rank
      expert window, or a prefix that differs between layer families. A `key`
      may itself be a tuple, meaning the transform is handed those tensors
      together.

    `layers` says which layers carry this weight: `ALL_LAYERS` for every one,
    `None` for a weight that is not per-layer at all, or a callable
    `d -> iterable[int]` when only some layers have it.

    `transform` runs on the source tensor before the copy, on the
    destination's device. `None` means the checkpoint tensor is already the
    declared shape.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | tuple[tuple[str, Callable], ...] | Callable
    dtype: torch.dtype | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        """The layers this weight exists on, or None if it is not per-layer."""
        if self.layers is None:
            return None
        if self.layers is ALL_LAYERS:
            return list(range(d.num_layers))
        return list(self.layers(d))

    def sources(self, d: SimpleNamespace, i: int | None, prefix: str) -> list[tuple[Any, Any]]:
        """`[(checkpoint key or key tuple, index into the param | None), ...]`."""
        if callable(self.src):
            return list(self.src(d, i))
        if isinstance(self.src, str):
            return [(self.src.format(p=prefix), None)]
        return [(template.format(p=prefix), slice_fn(d)) for template, slice_fn in self.src]


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
            for key, _, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None, str]]:
        """`[(param key, layer index | None, checkpoint layer prefix), ...]`.

        One place decides which parameters an entry declares, so `declare` and
        `manifest` cannot disagree about whether a weight exists.
        """
        layers = entry.layer_indices(d)
        if layers is None:
            return [(entry.name, None, "")]
        return [(f"l{i}_{entry.name}", i, self.LAYER_PREFIX.format(i=i)) for i in layers]

    def manifest(self, core) -> dict[str, list[tuple[Any, Any, Callable | None]]]:
        """param key -> [(checkpoint key, index into the param | None, transform | None)]."""
        d = self.dims(core)
        rows: dict[str, list[tuple[Any, Any, Callable | None]]] = {}
        for entry in self.WEIGHTS:
            for key, i, prefix in self._targets(entry, d):
                rows[key] = [
                    (ckpt, index, entry.transform) for ckpt, index in entry.sources(d, i, prefix)
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
