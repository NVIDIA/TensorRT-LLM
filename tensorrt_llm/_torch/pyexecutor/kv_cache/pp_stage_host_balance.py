# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-stage host tier sizing for a pipeline-parallel DFlash context server.

A prefix is reusable on a pipeline-parallel server only as deep as every stage
still holds it, so the stage that holds the fewest full-attention tokens over
all of its cache tiers caps the reusable depth of all stages. A configured
``host_cache_size`` gives every rank the same host tier, while the stages'
full-attention token costs differ: under the last-stage DFlash draft placement
the last stage gives a larger byte share to its sliding-window life cycle,
because every sliding-window page there also carries the draft KV. The full
pages that the other stages hold beyond the last stage's count are then never
reused.

On a stage of a pipeline-parallel DFlash context server
(``disaggregation.pp_dflash_context.pp_dflash_context_enabled``, so only at
``pipeline_parallel_size > 1``) with an explicit ``host_cache_size``, the KV
cache manager V2 resizes each rank's host tier so that
every stage holds the same number of full-attention tokens over its device,
host, remote host and store tiers, while the host tiers of all ranks keep the
configured total. All ranks must share one host for the total to stay a
node-level budget; otherwise, and whenever one rank cannot describe its tiers,
every rank keeps the configured size.

This module imports nothing from the package, so it can be tested on its own.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: Prefix of the balance's log lines.
LOG_TAG = "[pp_stage_host_balance]"

# Host tier sizes are rounded down to whole 2 MiB huge pages, so the balanced
# sizes never add up to more than the configured total.
HOST_TIER_GRANULARITY = 2 << 20


def full_attention_token_cost(
    pool_ratio: Optional[Sequence[float]],
    layer_bytes_per_token: Sequence[int],
    windows: Sequence[Optional[int]],
    sink_tokens: Sequence[Optional[int]],
    max_seq_len: int,
    tokens_per_block: int,
) -> Optional[Tuple[float, float]]:
    """Bytes of a cache tier per full-attention token, and the full-attention share.

    ``pool_ratio`` gives each life cycle a byte share of every cache tier, in
    the order the manager registers life cycles: by first appearance in the
    layer list, keyed by window (``None``, non-positive or at least
    ``max_seq_len`` meaning full attention) and sink blocks. A tier of ``B``
    bytes then holds about ``B * share / bytes_per_token`` tokens of the
    full-attention life cycle, so its cost is ``bytes_per_token / share``.

    That holds only while the full-attention life cycle has pools of its own.
    The cache levels below the device put life cycles whose pages have the
    same size into one pool, where they take pages from each other as they go,
    and the device pools life cycles with the same slot sizes together; either
    way ``share`` no longer bounds the full-attention pages.

    Returns ``(cost, share)``, or ``None`` when the stage has no
    ``pool_ratio``, a ratio without exactly one entry per life cycle, no
    full-attention life cycle or more than one, another life cycle with the
    same bytes per token as the full-attention one, or a non-positive share.
    """
    if pool_ratio is None:
        return None
    if not (len(layer_bytes_per_token) == len(windows) == len(sink_tokens)):
        return None
    life_cycle_bytes: Dict[tuple, int] = {}
    for layer_bytes, window, sinks in zip(layer_bytes_per_token, windows, sink_tokens):
        if window is not None and (window <= 0 or window >= max_seq_len):
            window = None
        sink_blocks = -(-(sinks or 0) // tokens_per_block)
        key = (window, sink_blocks)
        life_cycle_bytes[key] = life_cycle_bytes.get(key, 0) + int(layer_bytes)
    if len(pool_ratio) != len(life_cycle_bytes):
        return None
    full = [
        (float(share), layer_bytes)
        for share, (key, layer_bytes) in zip(pool_ratio, life_cycle_bytes.items())
        if key[0] is None
    ]
    if len(full) != 1:
        return None
    share, layer_bytes = full[0]
    if share <= 0.0 or layer_bytes <= 0:
        return None
    if sum(1 for other_bytes in life_cycle_bytes.values() if other_bytes == layer_bytes) > 1:
        return None
    return layer_bytes / share, share


@dataclass(frozen=True)
class StageTierShape:
    """One rank's cache tiers as the balance sees them."""

    hostname: str
    # Bytes of a cache tier per full-attention token (``full_attention_token_cost``).
    full_token_cost: float
    # The configured host tier.
    host_bytes: int
    # The device tier plus the remote host tier and the store level, which the balance does
    # not resize.
    other_bytes: int


@dataclass(frozen=True)
class StageHostPlan:
    """The balanced host tier of every rank, in rank order."""

    host_bytes: Tuple[int, ...]
    # Full-attention tokens every stage holds over all of its tiers.
    full_tokens: float


def plan_stage_host_tiers(
    shapes: Sequence[Optional[StageTierShape]],
    granularity: int = HOST_TIER_GRANULARITY,
) -> Tuple[Optional[StageHostPlan], str]:
    """Host tier sizes that give every rank the same full-attention tokens.

    Rank ``s`` holds ``(host_s + other_s) / cost_s`` full-attention tokens.
    Equal tokens ``T`` on every rank with the configured host total kept give
    ``host_s = T * cost_s - other_s`` and ``T = (sum host + sum other) / sum
    cost``. Each size is rounded down to ``granularity``, so the total never
    exceeds the configured one.

    Returns ``(plan, "")``, or ``(None, reason)`` when a rank has no shape,
    the ranks are on more than one host, or a rank would get less than one
    ``granularity`` of host tier. The result depends only on ``shapes``, so
    every rank that gathers the same list decides alike.
    """
    if not shapes:
        return None, "no ranks"
    missing = [rank for rank, shape in enumerate(shapes) if shape is None]
    if missing:
        return None, f"ranks {missing} cannot describe their full-attention tier cost"
    hostnames = sorted({shape.hostname for shape in shapes})
    if len(hostnames) > 1:
        return None, f"the ranks span {len(hostnames)} hosts"
    total_cost = math.fsum(shape.full_token_cost for shape in shapes)
    total_bytes = sum(shape.host_bytes + shape.other_bytes for shape in shapes)
    full_tokens = total_bytes / total_cost
    host_bytes = []
    for rank, shape in enumerate(shapes):
        target = full_tokens * shape.full_token_cost - shape.other_bytes
        size = int(math.floor(target / granularity)) * granularity if target > 0 else 0
        if size < granularity:
            return None, (
                f"rank {rank} would get {max(target, 0.0) / (1 << 30):.2f} GiB of host tier"
            )
        host_bytes.append(size)
    return StageHostPlan(tuple(host_bytes), full_tokens), ""


def describe_plan(shapes: Sequence[StageTierShape], plan: StageHostPlan) -> List[str]:
    """One line per rank: configured and balanced host tier, other tiers, tokens."""
    gib = float(1 << 30)
    lines = []
    for rank, (shape, size) in enumerate(zip(shapes, plan.host_bytes)):
        tokens = (size + shape.other_bytes) / shape.full_token_cost
        lines.append(
            f"rank {rank}: host tier {shape.host_bytes / gib:.2f} -> {size / gib:.2f} GiB, "
            f"device + remote host + store {shape.other_bytes / gib:.2f} GiB, "
            f"{shape.full_token_cost:.1f} B per full-attention token, "
            f"{tokens / 1e6:.3f} M full-attention tokens"
        )
    return lines
