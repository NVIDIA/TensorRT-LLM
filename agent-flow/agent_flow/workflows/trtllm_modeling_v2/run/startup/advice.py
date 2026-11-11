# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a spec is likely to cost or miss, read off its deployments before any GPU is spent.

Advice, never a verdict: each line says what to do, none stops a run, and a spec this
module cannot read gets no advice rather than an error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ...checks import anchors
from ...spec import deployments as deployments_mod

#: The share of a device's memory one engine's weights may take: the rest holds the KV cache,
#: activations and CUDA graphs.
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
    except (OSError, ValueError, TypeError, AttributeError):
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


def spec_warnings(task: Mapping[str, Any]) -> list[str]:
    """The advice a spec gets, one line each.

    - the first deployment holds the checkpoint on too few devices
      (``environment.gpu_memory_gib`` says how much one has);
    - the first deployment has no anchor: no benchmark, or a benchmark with no declared
      anchor and no reference row carrying exactly its tags, so ``onboarded`` can only
      mean it boots;
    - a deployment is scored without CUDA graphs: a full benchmark takes 10 to 50 times
      longer, and what is scored is not the captured path the target serves.

    An anchor that exists is never questioned: a small model's measured score is low.
    """
    try:
        return _warnings(task)
    except Exception:  # noqa: BLE001 - advice never stops what it advises on
        return []


def _warnings(task: Mapping[str, Any]) -> list[str]:
    found = deployments_mod.resolve(task)
    said: list[str] = []
    core = found[0]
    memory = (task.get("environment") or {}).get("gpu_memory_gib")
    weights = checkpoint_weight_bytes(task.get("checkpoint_path"))
    width = engine_width(weights, memory)
    narrow = min(server.ranks for server in core.servers.values())
    if width is not None and narrow < width:
        said.append(
            f"first deployment too narrow: `{core.name}` holds the checkpoint's "
            f"{weights / 2**30:.0f} GiB of weights on {narrow} device(s), more than "
            f"{ENGINE_WEIGHT_FRACTION:.0%} of each {memory:g}-GiB device, which leaves the "
            f"KV cache and CUDA graphs no room, if it loads at all; start at {width} or more"
        )
    if not core.benchmarks:
        said.append(
            f"no accuracy anchor: `{core.name}` names no benchmark, so `onboarded` means it "
            f"boots and answers the boot prompts, and nothing judges its accuracy"
        )
    for benchmark in core.benchmarks:
        anchor = anchors.resolve(task, core, benchmark)
        if anchor.kind == anchors.NONE:
            said.append(
                f"no accuracy anchor: `{core.name}` on `{benchmark.name}` has {anchor.source}, "
                f"so the run cannot judge its accuracy; measure one with an implementation "
                f"that already serves the checkpoint and write the row, or declare `anchor` "
                f"with its `source`"
            )
    eager = [
        d.name
        for d in found
        if d.benchmarks
        and any(
            "cuda_graph_config" in s.options and s.options["cuda_graph_config"] is None
            for s in d.servers.values()
        )
    ]
    if eager:
        said.append(
            f"accuracy without CUDA graphs: {', '.join(f'`{name}`' for name in eager)} "
            f"scored with `cuda_graph_config: null`: a full benchmark takes 10-50x longer, each "
            f"fix-and-recheck pays it again, and what is scored is not the captured path the "
            f"target serves; turn the graphs on unless the eager score is the point"
        )
    return said


__all__ = ["ENGINE_WEIGHT_FRACTION", "checkpoint_weight_bytes", "engine_width", "spec_warnings"]
