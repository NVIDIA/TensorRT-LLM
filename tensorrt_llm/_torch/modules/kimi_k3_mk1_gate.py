# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Switches and the plan check of the Kimi K3 specialized-decode ports (MK1) on Rubin.

Each port is opt-in through its own environment variable, read at use time, and takes effect
only where it was validated: SM107 with the CuTe DSL Rubin kernels (the predicate the K3 FP8
weight-read GEMM uses). Elsewhere a set switch is ignored with one warning per switch.

* :func:`mk1_value` / :func:`mk1_enabled` read a switch; :func:`mk1_cache_key` returns the
  switches' effective values for a CuTe compile-cache key, so a compiled kernel never
  outlives the setting it was compiled for.
* Ports that change the kernels a step runs, or how ranks hand data to each other, must
  decide the same way on every rank. :func:`mk1_plan_hash` is a pure function of the batch,
  the draft length and the static configuration; :func:`check_mk1_plan` all-gathers it (at
  warmup, before any graph capture) and raises if any rank differs.
"""

import functools
import hashlib
import json
import os
from typing import Any, Callable, Mapping, Sequence, Tuple

import numpy as np

from ..._utils import get_sm_version
from ...logger import logger


@functools.cache
def mk1_platform_supported() -> bool:
    """SM107 with the CuTe DSL Rubin kernels: the only platform the MK1 ports were validated on."""
    from ..cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE


def mk1_value(name: str, choices: Sequence[int] = (0, 1)) -> int:
    """Effective value of switch ``name``: its integer value, or 0 when unset or unsupported.

    Raises ValueError when the variable is set to anything outside ``choices``.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        value = None
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {raw!r}")
    if value == 0 or mk1_platform_supported():
        return value
    logger.warning_once(
        f"{name}={raw} is ignored: the Kimi K3 MK1 ports need SM107 with the CuTe DSL Rubin kernels.",
        key=f"mk1_gate_{name}",
    )
    return 0


def mk1_enabled(name: str) -> bool:
    """Whether the on/off switch ``name`` is set to 1 on a supported platform."""
    return mk1_value(name) != 0


def mk1_cache_key(*switches: Tuple[str, Sequence[int]]) -> Tuple[Tuple[str, int], ...]:
    """``((name, effective value), ...)`` of ``(name, choices)`` pairs, for a compile-cache key."""
    return tuple((name, mk1_value(name, choices)) for name, choices in switches)


def mk1_plan_hash(batch: int, draft_len: int, config: Mapping[str, Any]) -> int:
    """Signed 64-bit hash of a decode plan.

    A pure function of the batch (rows per rank), the draft length and ``config``, the
    static, JSON-serializable settings the plan depends on (switches, TP size, shapes). Key
    order does not matter. Nothing that varies between ranks may go into ``config``.
    """
    payload = json.dumps(
        {"batch": int(batch), "draft_len": int(draft_len), "config": config},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=True)


def check_mk1_plan(
    plan_hash: int, allgather_int64: Callable[[Sequence[int]], np.ndarray], what: str
) -> None:
    """Raise RuntimeError unless every rank computed ``plan_hash``.

    ``allgather_int64`` is a fixed-size int64 all-gather over the ranks that run the plan
    together, such as ``Distributed.tp_allgather_int64``. Call it at warmup, before capture:
    one rank on another plan would otherwise wait forever on a peer's mailbox or trigger.
    """
    gathered = np.asarray(allgather_int64([plan_hash]), dtype=np.int64).reshape(-1)
    mismatched = [rank for rank, value in enumerate(gathered.tolist()) if value != plan_hash]
    if mismatched:
        raise RuntimeError(
            f"Kimi K3 MK1 plan mismatch for {what}: ranks {mismatched} differ from "
            f"{plan_hash:#018x} (all: {[f'{v:#018x}' for v in gathered.tolist()]})"
        )
