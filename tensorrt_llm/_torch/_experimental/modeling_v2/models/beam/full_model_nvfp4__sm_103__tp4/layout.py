# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recipe, stored geometry and four-rank ownership of the Beam NVFP4-expert checkpoint layout.

The checkpoint is a ModelOpt NVFP4 export (``hf_quant_config.json``: ``quant_algo`` NVFP4,
group size 16, static activation scales, no KV-cache quantization). Only the routed experts'
gate/up/down projections are quantized; every other tensor keeps its stored precision: the
attention, dense feed-forward, norm, embedding and head tensors are BF16, the router's weight and
correction bias FP32. The four-rank configuration and every logical tensor's interval are those
of the BF16 layout (``layout_bf16``); this module changes only
what is stored.

An expert projection of logical shape ``[N, K]`` is stored as four tensors::

    <prefix>.weight          U8       [N, K/2]    two E2M1 codes per byte; the low nibble holds
                                                  the even K index, the high nibble the next one
    <prefix>.weight_scale    F8_E4M3  [N, K/16]   one block scale per 16 consecutive K elements
    <prefix>.weight_scale_2  F32      []          the projection's global weight scale
    <prefix>.input_scale     F32      []          the static activation scale s

Its logical weight is ``W[n, k] = e2m1(code[n, k]) * e4m3(weight_scale[n, k // 16]) *
weight_scale_2``. An activation row ``x`` is quantized with ``g = 1 / s`` (per 16-element block
``b``: ``scale_b = e4m3(g * max|x_b| / 6)``, ``code = e2m1(x * g / scale_b)``), and a projection
is ``alpha * (codes x block scales) @ (codes x block scales)^T`` with ``alpha = s *
weight_scale_2``.

Rank r owns intermediate ``[640 r, 640 r + 640)`` of every expert, on whole 16-element blocks::

    gate_proj, up_proj [2560, 5120]  rows [640 r, 640 r + 640)   packed [640, 2560],  scales [640, 320]
    down_proj          [5120, 2560]  cols [640 r, 640 r + 640)   packed [5120, 320] = byte columns
                                                                 [320 r, 320 r + 320); scales
                                                                 [5120, 40] = scale columns
                                                                 [40 r, 40 r + 40)

Every rank holds every projection's two FP32 scalars unchanged. Each sparse layer keeps each
projection kind of its 256 experts in banks (``layers.<L>.experts.<gate|up|down>``): packed codes
``[256, rows, K/2]``, block scales in the GEMM's 128x4 swizzled order ``[256, rows * K/16]``, and
the four scalars ``input_scale``, ``weight_scale_2``, ``g`` and ``alpha`` ``[256]`` FP32.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import json
import os
import re
from collections import OrderedDict
from typing import Any, Mapping, Optional

from . import layout_bf16 as bf16_layout
from .layout_bf16 import (
    BF16,
    EMBED,
    F32,
    FINAL_NORM,
    HEAD,
    WORLD_SIZE,
    TensorSpec,
    check_config,
    check_rank,
)

CONFIG_FILE = "config.json"
QUANT_CONFIG_FILE = "hf_quant_config.json"
U8 = "U8"
E4M3 = "F8_E4M3"
#: Elements per NVFP4 block scale.
GROUP = 16
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
QUANT_ALGO = "NVFP4"
#: The quantizer's 128-row x 4-column scale tile.
SWIZZLE_ROWS, SWIZZLE_COLS = 128, 4

__all__ = [
    "BF16",
    "E4M3",
    "EMBED",
    "F32",
    "FINAL_NORM",
    "GROUP",
    "HEAD",
    "PROJECTIONS",
    "U8",
    "WORLD_SIZE",
    "QuantSpec",
    "TensorSpec",
    "check_config",
    "check_partition",
    "check_rank",
    "check_recipe",
    "ownership_ledger",
    "stored_entries",
    "swizzled_numel",
    "tensor_specs",
]


def swizzled_numel(rows: int, cols: int) -> int:
    """Bytes of a ``[rows, cols]`` block-scale matrix in the 128x4 swizzled layout (padded)."""
    pad = lambda v, m: -(-v // m) * m  # noqa: E731
    return pad(rows, SWIZZLE_ROWS) * pad(cols, SWIZZLE_COLS)


@dataclasses.dataclass(frozen=True)
class QuantSpec:
    """One NVFP4 expert projection: logical geometry, four stored tensors, four-rank rule."""

    name: str  # "model.layers.<L>.mlp.experts.<E>.<proj>"
    shape: tuple  # logical [N, K]
    layer: int
    expert: int
    proj: str  # one of PROJECTIONS
    split_dim: int  # 0: output rows (gate/up), 1: input columns (down)

    @property
    def family(self) -> str:
        return "expert_" + self.proj.removesuffix("_proj")

    @property
    def bank(self) -> str:
        return f"layers.{self.layer}.experts.{self.proj.removesuffix('_proj')}"

    @property
    def packed_shape(self) -> tuple:
        return (self.shape[0], self.shape[1] // 2)

    @property
    def scale_shape(self) -> tuple:
        return (self.shape[0], self.shape[1] // GROUP)

    def stored(self) -> "OrderedDict[str, tuple]":
        """The four stored tensors: name -> (dtype, shape)."""
        return OrderedDict(
            (
                (self.name + ".weight", (U8, self.packed_shape)),
                (self.name + ".weight_scale", (E4M3, self.scale_shape)),
                (self.name + ".weight_scale_2", (F32, ())),
                (self.name + ".input_scale", (F32, ())),
            )
        )

    def interval(self, rank: int, world_size: int = WORLD_SIZE) -> tuple:
        """``(dim, start, stop)`` of rank ``rank``'s logical slice, on whole quantization blocks."""
        check_rank(rank, world_size)
        extent = self.shape[self.split_dim]
        if extent % (world_size * GROUP):
            raise ValueError(
                f"{self.name}: dim {self.split_dim} of {extent} does not split into "
                f"{world_size} whole {GROUP}-element blocks"
            )
        size = extent // world_size
        return self.split_dim, rank * size, (rank + 1) * size

    def packed_interval(self, rank: int, world_size: int = WORLD_SIZE) -> tuple:
        """``(dim, start, stop)`` of the rank's slice of the packed U8 tensor (byte columns)."""
        dim, start, stop = self.interval(rank, world_size)
        return (dim, start, stop) if dim == 0 else (dim, start // 2, stop // 2)

    def scale_interval(self, rank: int, world_size: int = WORLD_SIZE) -> tuple:
        """``(dim, start, stop)`` of the rank's slice of the block-scale tensor (scale columns)."""
        dim, start, stop = self.interval(rank, world_size)
        return (dim, start, stop) if dim == 0 else (dim, start // GROUP, stop // GROUP)

    def local_shape(self, rank: int, world_size: int = WORLD_SIZE) -> tuple:
        dim, start, stop = self.interval(rank, world_size)
        shape = list(self.shape)
        shape[dim] = stop - start
        return tuple(shape)

    def local_packed_shape(self, rank: int, world_size: int = WORLD_SIZE) -> tuple:
        rows, cols = self.local_shape(rank, world_size)
        return (rows, cols // 2)

    def local_scale_shape(self, rank: int, world_size: int = WORLD_SIZE) -> tuple:
        rows, cols = self.local_shape(rank, world_size)
        return (rows, cols // GROUP)


def tensor_specs(config: Any, layers: Optional[tuple] = None) -> "OrderedDict[str, Any]":
    """Every logical tensor of the selected layers and the outer tensors.

    Keys are checkpoint names for higher-precision tensors (``TensorSpec``, stored BF16 or F32)
    and projection prefixes for NVFP4 expert projections (``QuantSpec``). Geometry and
    intervals are the BF16 layout's; the stored dtype of every non-expert matrix is BF16.
    """
    specs: "OrderedDict[str, Any]" = OrderedDict()
    for name, spec in bf16_layout.tensor_specs(config, layers).items():
        if spec.expert is not None:
            prefix = name.removesuffix(".weight")
            proj = prefix.rsplit(".", 1)[1]
            specs[prefix] = QuantSpec(
                prefix, tuple(spec.shape), spec.layer, spec.expert, proj, spec.split_dim
            )
        elif spec.stored_dtype == bf16_layout.FP8:
            specs[name] = dataclasses.replace(spec, stored_dtype=BF16)
        else:
            specs[name] = spec
    return specs


def stored_entries(specs: Mapping[str, Any]) -> "OrderedDict[str, tuple]":
    """Every stored tensor ``specs`` needs: name -> (dtype, shape)."""
    out: "OrderedDict[str, tuple]" = OrderedDict()
    for spec in specs.values():
        if isinstance(spec, QuantSpec):
            out.update(spec.stored())
        else:
            out[spec.name] = (spec.stored_dtype, tuple(spec.shape))
    return out


def check_partition(specs: Mapping[str, Any], world_size: int = WORLD_SIZE) -> None:
    """Raise unless every split tensor's rank intervals tile its axis exactly, in rank order.

    For an expert projection the logical, packed-byte and block-scale intervals must each tile
    their axis, and every logical boundary must fall on a quantization block boundary.
    """
    bf16_layout.check_partition(
        {n: s for n, s in specs.items() if not isinstance(s, QuantSpec)}, world_size
    )
    for spec in specs.values():
        if not isinstance(spec, QuantSpec):
            continue
        for kind, extent, interval in (
            ("logical", spec.shape[spec.split_dim], spec.interval),
            ("packed", spec.packed_shape[spec.split_dim], spec.packed_interval),
            ("scale", spec.scale_shape[spec.split_dim], spec.scale_interval),
        ):
            cursor = 0
            for rank in range(world_size):
                _, start, stop = interval(rank, world_size)
                if start != cursor or stop <= start:
                    raise ValueError(
                        f"{spec.name}: rank {rank} {kind} interval [{start}, {stop}) leaves a gap "
                        "or overlap"
                    )
                cursor = stop
            if cursor != extent:
                raise ValueError(f"{spec.name}: {kind} intervals cover {cursor} of {extent}")


def ownership_ledger(specs: Mapping[str, Any], world_size: int = WORLD_SIZE) -> list:
    """One plain-data row per logical tensor: its rule and every rank's intervals and storage.

    Higher-precision rows are the BF16 layout's. An expert row records, per rank, the logical,
    packed-byte and block-scale intervals, the bank and the slot (the global expert id).
    """
    rows = bf16_layout.ownership_ledger(
        OrderedDict((n, s) for n, s in specs.items() if not isinstance(s, QuantSpec)), world_size
    )
    for spec in specs.values():
        if not isinstance(spec, QuantSpec):
            continue
        ranks = []
        for rank in range(world_size):
            dim, start, stop = spec.interval(rank, world_size)
            _, pstart, pstop = spec.packed_interval(rank, world_size)
            _, sstart, sstop = spec.scale_interval(rank, world_size)
            ranks.append(
                {
                    "rank": rank,
                    "replicated": False,
                    "dim": dim,
                    "start": start,
                    "stop": stop,
                    "packed": [pstart, pstop],
                    "scale": [sstart, sstop],
                    "local_packed_shape": list(spec.local_packed_shape(rank, world_size)),
                    "local_scale_shape": list(spec.local_scale_shape(rank, world_size)),
                    "scalars": "replicated",
                    "bank": spec.bank,
                    "slot": spec.expert,
                }
            )
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "layer": spec.layer,
                "expert": spec.expert,
                "shape": list(spec.shape),
                "stored": {k: [d, list(s)] for k, (d, s) in spec.stored().items()},
                "ranks": ranks,
            }
        )
    return rows


def _module_names(specs: Mapping[str, Any]) -> tuple:
    """(quantized expert modules, higher-precision Linear modules) of ``specs``."""
    quantized, kept = [], []
    for name, spec in specs.items():
        if isinstance(spec, QuantSpec):
            quantized.append(name)
        elif name.endswith(".weight") and len(spec.shape) == 2 and name != EMBED:
            kept.append(name.removesuffix(".weight"))
    return quantized, kept


def check_recipe(quant_config: Any, specs: Mapping[str, Any]) -> dict:
    """Raise ValueError unless ``hf_quant_config.json`` is the recipe this layout computes.

    ModelOpt NVFP4, group size 16, no KV-cache quantization, and an exclusion list (``fnmatch``
    patterns over module names) that excludes every higher-precision Linear module of ``specs``
    (attention, router, dense feed-forward, head) and no expert projection. Returns a summary.
    """
    if not isinstance(quant_config, Mapping):
        raise ValueError("hf_quant_config.json must be a JSON object")
    q = quant_config.get("quantization")
    if not isinstance(q, Mapping):
        raise ValueError("hf_quant_config.json has no 'quantization' object")
    if q.get("quant_algo") != QUANT_ALGO:
        raise ValueError(f"unsupported quant_algo {q.get('quant_algo')!r}; expected {QUANT_ALGO}")
    if q.get("group_size") != GROUP or type(q.get("group_size")) is not int:
        raise ValueError(f"unsupported group_size {q.get('group_size')!r}; expected {GROUP}")
    if q.get("kv_cache_quant_algo") is not None:
        raise ValueError(
            f"unsupported kv_cache_quant_algo {q.get('kv_cache_quant_algo')!r}; expected none"
        )
    patterns = q.get("exclude_modules")
    if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
        raise ValueError("exclude_modules must be a list of strings")
    excluded = re.compile("|".join(f"(?:{fnmatch.translate(p)})" for p in patterns) or "(?!)")
    quantized, kept = _module_names(specs)
    wrongly_excluded = [m for m in quantized if excluded.match(m)]
    if wrongly_excluded:
        raise ValueError(f"exclude_modules excludes quantized experts: {wrongly_excluded[:4]}")
    not_excluded = [m for m in kept if not excluded.match(m)]
    if not_excluded:
        raise ValueError(f"higher-precision modules are not excluded: {not_excluded[:4]}")
    return {
        "quant_algo": QUANT_ALGO,
        "group_size": GROUP,
        "kv_cache_quant_algo": None,
        "exclude_patterns": len(patterns),
        "quantized_modules": len(quantized),
        "excluded_linear_modules": len(kept),
    }


def checkpoint_dir_of(pretrained_config: Any) -> Optional[str]:
    """The checkpoint directory a loaded configuration came from, or None."""
    return getattr(pretrained_config, "name_or_path", None) or getattr(
        pretrained_config, "_name_or_path", None
    )


def read_config(directory: Any, name: str = CONFIG_FILE) -> dict:
    """The JSON object stored as ``name`` in checkpoint ``directory``; ValueError otherwise."""
    if not directory:
        raise ValueError(f"no checkpoint directory names {name} ({directory!r})")
    path = os.path.join(os.fspath(directory), name)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is not a readable JSON file: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return data


def check_checkpoint(checkpoint_dir: Any) -> dict:
    """Raise ValueError unless this target computes the checkpoint; return its config.json.

    The one validator of what the target serves, used by routing (a refusal there is the
    ``NoTarget`` reason) and by the target's construction, so the two cannot disagree: the
    checkpoint's ``config.json`` is what the layout computes (``check_config``) and its
    ``hf_quant_config.json`` the NVFP4 recipe of the whole model (``check_recipe``). Reads
    the two files; allocates nothing.
    """
    config = read_config(checkpoint_dir)
    check_config(config)
    check_recipe(read_config(checkpoint_dir, QUANT_CONFIG_FILE), tensor_specs(config))
    return config
