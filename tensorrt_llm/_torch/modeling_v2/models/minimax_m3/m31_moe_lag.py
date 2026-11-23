# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion-ring sizing and lag admission for m3_moe.

The ring has no consumer-to-producer empty handshake. Give every group its own
slot and retain the conservative admission cap on experimental phase lags. This policy belongs
to this engine lineage; the vendored m3ll engine has a separate ring policy.
"""

FC1_DONE_SLOTS = 4
MAX_PHASE_LAG = FC1_DONE_SLOTS - 1
DEFAULT_PHASE_LAG = 2


def _check_lag(lag: int) -> None:
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
        raise ValueError(f"m3_moe: phase lag must be a non-negative int, got {lag!r}")


def completion_ring_slots(max_groups: int) -> int:
    """Keep four slots for small calls; give every group a slot for larger calls."""
    if isinstance(max_groups, bool) or not isinstance(max_groups, int) or not 1 <= max_groups <= 8:
        raise ValueError(f"m3_moe: group capacity must be an int in 1..8, got {max_groups!r}")
    return FC1_DONE_SLOTS if max_groups <= FC1_DONE_SLOTS else 8


def phase_lag_config(lag: int, max_groups: int, *, allow_unsafe: bool = False) -> dict:
    """Validate host admission and preserve the existing default-lag kernel key."""
    _check_lag(lag)
    slots = completion_ring_slots(max_groups)
    config = {} if lag == DEFAULT_PHASE_LAG else {"lag": lag}
    if slots != FC1_DONE_SLOTS:
        config["fc1d"] = slots
    if lag > MAX_PHASE_LAG:
        if not allow_unsafe:
            raise ValueError(
                f"m3_moe: phase lag {lag} exceeds {MAX_PHASE_LAG}; the fc1_done "
                f"protocol has no empty barrier; larger lags need runtime validation. "
                "allow_unsafe=True is for explicit unsafe experiments only."
            )
        config["lag_unsafe"] = 1
    return config


def validate_kernel_lag(
    lag: int, max_groups: int, *, slots: int, allow_unsafe: bool = False
) -> None:
    """Reject unsafe direct kernel configs when a call can wrap the ring."""
    _check_lag(lag)
    completion_ring_slots(max_groups)
    if slots not in (4, 8):
        raise ValueError(f"m3_moe: fc1_done ring must have four or eight slots, got {slots}")
    if slots < max_groups and not allow_unsafe:
        raise ValueError(
            f"m3_moe: fc1_done {slots}-slot ring can alias across {max_groups} groups on idle CTAs"
        )
    if max_groups > FC1_DONE_SLOTS:
        phase_lag_config(lag, max_groups, allow_unsafe=allow_unsafe)
