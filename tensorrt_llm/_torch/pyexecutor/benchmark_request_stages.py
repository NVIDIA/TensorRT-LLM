# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parsing of the internal disaggregated benchmark request schedule."""

from typing import NamedTuple


class BenchmarkRequestStage(NamedTuple):
    """A fill target and the total number of requests in one measurement."""

    fill_target: int
    num_requests: int


def parse_benchmark_request_stages(value: str) -> list[BenchmarkRequestStage]:
    """Parse ``C``, ``C,C,...`` or ``C/N,C/N,...`` without losing budgets.

    Args:
        value: The ``TLLM_BENCHMARK_REQ_QUEUES_SIZE`` value. A bare ``C``
            means ``C/C``.

    Returns:
        One stage per entry; ``[(0, 0)]`` for ``"0"``, which disables the gate.

    Raises:
        ValueError: An entry is malformed or violates ``0 < C <= N``.
    """
    if value.strip() == "0":
        return [BenchmarkRequestStage(0, 0)]
    stages = []
    for entry in value.split(","):
        parts = entry.strip().split("/")
        if len(parts) not in (1, 2) or any(
            not part.strip().isascii() or not part.strip().isdecimal() for part in parts
        ):
            raise ValueError(
                "TLLM_BENCHMARK_REQ_QUEUES_SIZE expects C or C/N entries "
                f"separated by commas; got {value!r}"
            )
        fill_target = int(parts[0])
        num_requests = int(parts[-1])
        if fill_target <= 0 or num_requests < fill_target:
            raise ValueError(
                f"TLLM_BENCHMARK_REQ_QUEUES_SIZE requires 0 < C <= N for every stage; got {entry!r}"
            )
        stages.append(BenchmarkRequestStage(fill_target, num_requests))
    return stages
