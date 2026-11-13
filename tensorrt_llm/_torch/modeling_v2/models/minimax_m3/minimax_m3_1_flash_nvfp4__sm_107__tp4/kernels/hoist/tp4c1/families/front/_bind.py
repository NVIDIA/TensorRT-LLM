# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the front/scorer/topk/decode/oproj/post variant modules (no CUDA at import)."""

from __future__ import annotations

import importlib
import os
import tempfile

from ...api import Footprint, StageFn


def knobs_table():
    """The process-wide legacy knob table every sparse-layer _src copy reads as `_K`."""
    from ...runtime.knobs import KNOBS

    return KNOBS


def build_root(tag: str) -> str:
    """Node-local build root for legacy torch-extension builds, never a shared filesystem."""
    d = os.path.join(tempfile.gettempdir(), "m3llk-legacy", tag)
    os.makedirs(d, exist_ok=True)
    return d


def bind(owner: str, values: dict) -> None:
    """Pin knobs (str values). None means "must stay unset": refused if anything set it."""
    K = knobs_table()
    unset = [k for k, v in values.items() if v is None and k in K]
    if unset:
        raise ValueError(f"{owner}: {unset} must be unset (bound {[K[k] for k in unset]})")
    K.bind({k: str(v) for k, v in values.items() if v is not None}, owner)


def legacy(name: str):
    """Import a legacy module; the core finder serves it from this preset's _src copies (Variant.legacy_modules)."""
    return importlib.import_module(name)


def g3_knobs(llk) -> dict:
    """The node-local build root of the lamport all-reduce extension (comm lamport_allreduce_host)."""
    return {"G3_BUILD": build_root("g3")}


def stage(variant, fn, *, reads, writes, pdl, footprint=Footprint(), keys=()):
    return StageFn(
        name=variant.id,
        fn=fn,
        stages=variant.stages,
        reads=tuple(reads),
        writes=tuple(writes),
        pdl=pdl,
        footprint=footprint,
        keys=tuple(keys),
    )


__all__ = ["bind", "build_root", "g3_knobs", "knobs_table", "legacy", "stage"]
