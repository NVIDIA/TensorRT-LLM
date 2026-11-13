# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the router and moe variant modules. No CUDA, no env, no file I/O at import."""

from __future__ import annotations

import importlib
import json
import os
import tempfile

from ...api import Footprint, StageFn

# a 2^31-poll trap cap, not a globaltimer deadline
HANG_CAP_NS = (1 << 31) * 200
RX_DEADLINE_NS = (
    2_000_000_000  # lbh._deadline: H9_RH_DEADLINE_NS unset -> 2 s globaltimer, then __trap
)
VR200_SMS = 212


def knobs():
    from ...runtime.knobs import KNOBS

    return KNOBS


def bind(owner: str, values: dict) -> None:
    """Pin the knobs this variant's Params render to (process-wide; a different bound value raises KnobConflict)."""
    knobs().bind({k: str(v) for k, v in values.items()}, owner)


def base_knobs(llk) -> dict:
    """Knobs every legacy module in this layer reads from the process env: the build root (TMPDIR) and RANK."""
    return {"TMPDIR": tempfile.gettempdir(), "RANK": str(llk.rank)}


def g3_knobs(llk) -> dict:
    """The pr_ar workspace knobs (post.g3-prar's values, so a second binder never conflicts)."""
    from ..front._bind import g3_knobs as _g3

    return _g3(llk)


def legacy(name: str):
    """Import a legacy module by its dotted name (served from the _src copies by toolchain.legacy)."""
    return importlib.import_module(name)


def legacy_attr(spec: str):
    mod, attr = spec.split(":")
    return getattr(legacy(mod), attr)


def jdump(d: dict, compact: bool) -> str:
    """The exact json spelling the legacy modules expect for a knob value."""
    return json.dumps(d, separators=(",", ":")) if compact else json.dumps(d)


# legacy factories allocate their own tensors outside the arenas; size not measured yet
UNMEASURED = Footprint(measured=False, in_arena=False)


def stage(variant, fn, *, reads, writes, pdl, footprint=UNMEASURED, keys=(), kernels=()):
    return StageFn(
        name=variant.id,
        fn=fn,
        stages=variant.stages,
        reads=tuple(reads),
        writes=tuple(writes),
        pdl=pdl,
        footprint=footprint,
        keys=tuple(keys),
        kernels=tuple(kernels),
    )


def chain(*fns):
    """One Stage that launches several legacy slot callables in order (o_proj then post)."""
    fns = tuple(fns)

    def fn():
        for f in fns:
            f()

    return fn


def engine_dir(name: str) -> str:
    """The MoE engine package (host code + the CuTe DSL kernel it loads per configuration) under kernels/moe/engine."""
    from ...toolchain.paths import pkg_root

    return os.path.join(pkg_root(), "kernels", "moe", "engine", name)


__all__ = [
    "UNMEASURED",
    "HANG_CAP_NS",
    "RX_DEADLINE_NS",
    "VR200_SMS",
    "base_knobs",
    "bind",
    "chain",
    "g3_knobs",
    "jdump",
    "knobs",
    "engine_dir",
    "legacy",
    "legacy_attr",
    "stage",
]


# sentences shared by the waiver reasons
FS_C1 = (
    "The formal schedule model proves the C1 chain as built free of residency deadlock on GPC layouts g8/g16, "
    "for any CTA dispatch order, 2 ranks x 2 layers"
)
FS_C2 = (
    "The formal schedule model proves the C2 chain as built free of residency deadlock on GPC layouts g8/g16/g24, "
    "for any dispatch order, 2 ranks x 2 and x 3 layers"
)
FS_CAVEAT = (
    "The proof assumes cluster co-scheduling and the SM107 GPC layout, neither verified on hardware, and a "
    "chain stream with no co-tenant: a foreign resident spinner deadlocks C2, and C1 survives one only by "
    "smem headroom"
)
NO_TP4 = "NOT PROVED: the formal schedule model has no TP4 C1 chain spec"


def waiver(stage: str, rule: str, reason: str) -> dict:
    """One preset.toml [[waivers]] row."""
    return {"stage": stage, "rule": rule, "reason": reason}
