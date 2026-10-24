# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal contract for model-owned checkpoint discovery and resolution."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class ResolvedCheckpoint:
    path: str
    pipeline_config: dict[str, Any] = field(default_factory=dict)


class CheckpointAdapter(Protocol):
    """A recognized checkpoint; resolution may fetch supporting components."""

    def resolve(self) -> ResolvedCheckpoint: ...


# Probes inspect local metadata only. Downloads and filesystem changes belong
# in resolve(), after exactly one pipeline has claimed the checkpoint.
CheckpointProbe = Callable[[Path, Mapping[str, Any]], CheckpointAdapter | None]
