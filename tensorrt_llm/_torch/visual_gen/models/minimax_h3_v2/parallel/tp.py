# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm.quantization.mode import QuantAlgo


def shard_bounds(size: int, tp_size: int, rank: int) -> tuple[int, int]:
    chunk, extra = divmod(size, tp_size)
    start = rank * chunk + min(rank, extra)
    return start, start + chunk + (rank < extra)


def projection_shard_bounds(
    width: int, tp_size: int, rank: int, quant_algo: QuantAlgo | None, deferred: bool
) -> tuple[int, int]:
    if tp_size == 1:
        return 0, width
    if deferred:
        if width % tp_size:
            raise ValueError(
                f"Deferred H3 projection width {width} must divide across {tp_size} ranks"
            )
        return shard_bounds(width, tp_size, rank)
    alignment = {QuantAlgo.FP8_BLOCK_SCALES: 128, QuantAlgo.NVFP4: 32}.get(quant_algo, 1)
    if width % alignment:
        raise ValueError(
            f"TP projection width {width} must be divisible by quantization alignment {alignment}"
        )
    start, end = shard_bounds(width // alignment, tp_size, rank)
    return start * alignment, end * alignment


def head_bounds(num_kv_heads: int, tp_size: int, rank: int, ulysses_size: int) -> tuple[int, int]:
    """Partition KV heads across TP ranks in complete Ulysses groups."""
    assert num_kv_heads % ulysses_size == 0
    groups = num_kv_heads // ulysses_size
    assert groups >= tp_size
    start, end = shard_bounds(groups, tp_size, rank)
    return start * ulysses_size, end * ulysses_size
