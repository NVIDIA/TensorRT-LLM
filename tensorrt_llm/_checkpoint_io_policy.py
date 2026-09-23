# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint I/O names and display metadata, independent of loader execution."""

from typing import Final, Literal, NamedTuple, TypeAlias, get_args

CheckpointIoPolicy: TypeAlias = Literal[
    "auto", "native", "rank_striped_read_ahead", "demand_ordered_rank_striped_read_ahead"
]

AUTO_IO_POLICY: Final = "auto"
NATIVE_IO_POLICY: Final = "native"
RANK_STRIPED_IO_POLICY: Final = "rank_striped_read_ahead"
DEMAND_ORDERED_IO_POLICY: Final = "demand_ordered_rank_striped_read_ahead"

CHECKPOINT_IO_POLICIES: tuple[CheckpointIoPolicy, ...] = get_args(CheckpointIoPolicy)
RANK_STRIPED_IO_POLICIES = (RANK_STRIPED_IO_POLICY, DEMAND_ORDERED_IO_POLICY)
EXECUTABLE_CHECKPOINT_IO_POLICIES = (NATIVE_IO_POLICY,) + RANK_STRIPED_IO_POLICIES


class CheckpointIoPolicyInfo(NamedTuple):
    """Human-readable metadata; eligibility and dispatch remain in the loader."""

    display_name: str
    description: str
    kind: Literal["selector", "implementation"]


CHECKPOINT_IO_POLICY_INFO: dict[CheckpointIoPolicy, CheckpointIoPolicyInfo] = {
    AUTO_IO_POLICY: CheckpointIoPolicyInfo(
        "Automatic selection",
        "Select basic rank-striped read-ahead where eligible, otherwise native I/O.",
        "selector",
    ),
    NATIVE_IO_POLICY: CheckpointIoPolicyInfo(
        "Native checkpoint loading",
        "Preserve the existing checkpoint I/O and materialization path.",
        "implementation",
    ),
    RANK_STRIPED_IO_POLICY: CheckpointIoPolicyInfo(
        "Rank-cooperative read-ahead",
        "Read complete checkpoint chunks across node-local ranks while native materialization runs.",
        "implementation",
    ),
    DEMAND_ORDERED_IO_POLICY: CheckpointIoPolicyInfo(
        "Demand-ordered rank-cooperative read-ahead",
        "Prioritize complete checkpoint chunks using native materialization demand hints.",
        "implementation",
    ),
}
