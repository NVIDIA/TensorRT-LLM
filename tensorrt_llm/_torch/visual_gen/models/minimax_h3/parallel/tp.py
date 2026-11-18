# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def shard_bounds(size: int, tp_size: int, rank: int) -> tuple[int, int]:
    chunk, extra = divmod(size, tp_size)
    start = rank * chunk + min(rank, extra)
    return start, start + chunk + (rank < extra)


def head_bounds(num_kv_heads: int, tp_size: int, rank: int, ulysses_size: int) -> tuple[int, int]:
    """Partition KV heads across TP ranks in complete Ulysses groups."""
    assert num_kv_heads % ulysses_size == 0
    groups = num_kv_heads // ulysses_size
    assert groups >= tp_size
    start, end = shard_bounds(groups, tp_size, rank)
    return start * ulysses_size, end * ulysses_size
