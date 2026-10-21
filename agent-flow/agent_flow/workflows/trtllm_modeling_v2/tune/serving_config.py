# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The serving config a perf measurement runs at, derived from its workload.

The settings that decide whether a workload is served well at all follow
from the workload, so they are derived rather than searched:

- ``max_batch_size``: the highest concurrency measured;
- ``max_seq_len``: ISL + OSL, plus the draft length so a speculative verify
  step at the last position still fits;
- ``max_num_tokens``: room for a whole prompt plus one token (and its
  drafts) for every other request in the same iteration, rounded up to a
  power of two and kept within [8192, 32768];
- chunked prefill: on when a prompt does not fit ``max_num_tokens``;
- CUDA graphs on, every decode batch size up to ``max_batch_size`` captured
  (up to ``DENSE_CAPTURE_LIMIT``; past it, padding rounds up to the next
  captured size).

These are the settings the one GPU-measured campaign kept when it searched
them (a dense capture list and a token budget of 2 x ISL at ISL 8192,
concurrency 128). Anything a workload needs beyond them is a hypothesis for
a tuner turn, not a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import yaml

MIN_MAX_NUM_TOKENS = 8192
MAX_MAX_NUM_TOKENS = 32768
#: Above this many concurrent requests, capture powers of two and pad between them.
DENSE_CAPTURE_LIMIT = 256


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def capture_batch_sizes(max_batch_size: int) -> list[int]:
    """Every size up to the limit, then powers of two up to ``max_batch_size``."""
    dense = list(range(1, min(max_batch_size, DENSE_CAPTURE_LIMIT) + 1))
    size = _next_pow2(DENSE_CAPTURE_LIMIT + 1)
    while dense[-1] < max_batch_size:
        dense.append(min(size, max_batch_size))
        size *= 2
    return dense


def derive_serving_config(
    *,
    isl: int,
    osl: int,
    concurrency: Sequence[int],
    draft_len: int = 0,
) -> dict[str, Any]:
    """The LLM-API options a workload of ``isl``/``osl`` at ``concurrency`` runs with."""
    max_batch_size = max(int(c) for c in concurrency)
    tokens = isl + max_batch_size * (1 + draft_len)
    max_num_tokens = min(max(_next_pow2(tokens), MIN_MAX_NUM_TOKENS), MAX_MAX_NUM_TOKENS)
    return {
        "max_batch_size": max_batch_size,
        "max_seq_len": isl + osl + draft_len,
        "max_num_tokens": max_num_tokens,
        "enable_chunked_prefill": isl > max_num_tokens,
        "cuda_graph_config": {
            "enable_padding": True,
            "batch_sizes": capture_batch_sizes(max_batch_size),
        },
    }


def write_serving_config(path: Path, config: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


__all__ = [
    "DENSE_CAPTURE_LIMIT",
    "capture_batch_sizes",
    "derive_serving_config",
    "write_serving_config",
]
