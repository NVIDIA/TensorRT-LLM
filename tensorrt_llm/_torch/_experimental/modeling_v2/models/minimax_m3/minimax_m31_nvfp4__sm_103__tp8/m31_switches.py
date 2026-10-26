# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kill switches of this MiniMax-M3.1 target's default-on items (each target keeps its own copy: targets share
only the catalog).

Every item is on by default; ``TRTLLM_M31_DISABLE_<ID>=1`` turns item ``<ID>`` off and restores the path it replaced.
The environment is read once, at import, so every rank and every step of a process sees the same setting (a step must
not change its graph mid-run). Each item lists its id and what the switch restores where it reads it.

Values (case-insensitive, surrounding blanks ignored): ``1`` / ``true`` / ``on`` / ``yes`` set a switch; empty / ``0`` /
``false`` / ``off`` / ``no`` leave it unset; anything else raises at import, naming the variable, so a typo fails loudly
instead of silently turning an item on or off.
"""

import os
from typing import FrozenSet

_PREFIX = "TRTLLM_M31_DISABLE_"
_SET = frozenset(("1", "true", "on", "yes"))
_UNSET = frozenset(("", "0", "false", "off", "no"))


def _is_set(name: str, value: str) -> bool:
    """Whether the switch variable ``name`` is set to ``value``; raises on a value of neither kind."""
    word = value.strip().lower()
    if word in _SET:
        return True
    if word in _UNSET:
        return False
    raise ValueError(
        f"{name}={value!r}: expected 1 / true / on / yes (set) or 0 / false / off / no / empty (unset)"
    )


# The item ids switched off in this process, read once at import.
_DISABLED: FrozenSet[str] = frozenset(
    name[len(_PREFIX) :]
    for name, value in os.environ.items()
    if name.startswith(_PREFIX) and _is_set(name, value)
)


def m31_disabled(item_id: str) -> bool:
    """Whether ``TRTLLM_M31_DISABLE_<item_id>`` turned the item off for this process."""
    return item_id in _DISABLED


# Items that are OFF by default (rule 1b without a passing gate c, or a failed gate c): ``TRTLLM_M31_ENABLE_<ID>=1``
# opts in; the item's ``TRTLLM_M31_DISABLE_<ID>`` still wins.
_ENABLE_PREFIX = "TRTLLM_M31_ENABLE_"

_ENABLED: FrozenSet[str] = frozenset(
    name[len(_ENABLE_PREFIX) :]
    for name, value in os.environ.items()
    if name.startswith(_ENABLE_PREFIX) and _is_set(name, value)
)


def m31_enabled(item_id: str) -> bool:
    """Whether ``TRTLLM_M31_ENABLE_<item_id>`` opted a default-off item in for this process."""
    return item_id in _ENABLED
