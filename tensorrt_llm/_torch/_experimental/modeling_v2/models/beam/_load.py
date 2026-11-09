# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect rank-local checkpoint failures before any rank leaves a load phase."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping


def load_failure_details(mapping: Mapping, ok: bool, error: BaseException | None) -> str | None:
    """Return the same failing-rank summary on every rank, or None after complete success."""
    outcome = (ok, repr(error) if error is not None else "incomplete tensor coverage")
    outcomes = [outcome]
    if mapping.world_size > 1:
        from tensorrt_llm._torch.distributed.communicator import Distributed

        outcomes = Distributed.get(mapping).allgather(outcome)
    failures = [
        f"rank {rank}: {cause}" for rank, (passed, cause) in enumerate(outcomes) if not passed
    ]
    return "; ".join(failures) or None
