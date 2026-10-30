# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission metrics, initialized after serving configures multiprocessing."""

from threading import Lock
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from prometheus_client import Counter

_counter: "Counter | None" = None
_counter_lock = Lock()


def count_ignored_kv_hints(
    reason: Literal["untrusted", "unknown_version", "oversized", "malformed"],
) -> None:
    """Count one ignored envelope at its admission edge, before rank broadcast.

    Args:
        reason: Fixed rejection category; never contains producer metadata.
    """
    global _counter
    with _counter_lock:
        if _counter is None:
            from prometheus_client import Counter

            _counter = Counter(
                "trtllm_kv_hints_ignored_total",
                "Optional KV hint envelopes ignored at request ingress.",
                ("reason",),
            )
    _counter.labels(reason=reason).inc()
