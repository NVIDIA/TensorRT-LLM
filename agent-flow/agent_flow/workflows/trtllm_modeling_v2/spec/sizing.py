# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How many devices one engine of a checkpoint needs, from the weights' size alone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: The share of a device's memory one engine's weights may take: the rest holds the KV cache,
#: activations and CUDA graphs. A member below the width this implies cannot serve the model.
#: (A 344-GiB checkpoint has served disaggregated on two 277-GiB devices per engine: 62%.)
ENGINE_WEIGHT_FRACTION = 0.75


def checkpoint_weight_bytes(checkpoint: str | Path | None) -> int | None:
    """The checkpoint's weights in bytes: its safetensors index's total, else its files' sizes."""
    if not checkpoint:
        return None
    root = Path(str(checkpoint))
    try:
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        total = int((index.get("metadata") or {}).get("total_size") or 0)
        if total > 0:
            return total
    except (OSError, ValueError, TypeError):
        pass
    try:
        total = sum(f.stat().st_size for f in root.glob("*.safetensors"))
    except OSError:
        return None
    return total or None


def engine_width(weight_bytes: int | None, gpu_memory_gib: Any) -> int | None:
    """The fewest devices (a power of two) one engine's weights fit on; ``None`` unknown."""
    memory = gpu_memory_gib
    if not weight_bytes or not isinstance(memory, (int, float)) or memory <= 0:
        return None
    room = ENGINE_WEIGHT_FRACTION * float(memory) * 2**30
    width = 1
    while weight_bytes / width > room:
        width *= 2
    return width


__all__ = ["ENGINE_WEIGHT_FRACTION", "checkpoint_weight_bytes", "engine_width"]
