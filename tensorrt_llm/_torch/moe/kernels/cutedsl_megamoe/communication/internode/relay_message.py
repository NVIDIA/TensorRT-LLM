# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Byte layout of one Stage 1 relay message (port of ``pre_unification/comm/nccl_gin_layout.py``).

A relay message carries one node-deduplicated token to its destination node: the 16-byte
metadata slot (one :class:`TokenSrcMetadata` record plus zero padding, invariant I6), the
route words (``node_expert | topk_slot << 16`` for every top-k route owned by the destination
node, ``0xFFFF`` otherwise), the top-k weights (FP32 bits, zero where not routed), the FP4 token
row and its scale factors. Every plane starts 16-byte aligned so the pack and the fan-out can
move 16-byte vectors; the message stride rounds to ``alignment`` (4096 bytes at the production
shape: hidden 7168, top-k 4).
"""

from __future__ import annotations

from dataclasses import dataclass


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class RelayMessageLayout:
    src_metadata_offset: int
    route_words_offset: int
    topk_weights_offset: int
    token_data_offset: int
    sf_data_offset: int
    message_bytes: int


def build_relay_message_layout(*, hidden_bytes: int, sf_bytes: int, topk: int, alignment: int = 128) -> RelayMessageLayout:
    for name, value in (("hidden_bytes", hidden_bytes), ("sf_bytes", sf_bytes), ("topk", topk), ("alignment", alignment)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    cursor = 16  # metadata record (8 B) + zero padding (8 B)
    route_words_offset = cursor
    cursor += _round_up(topk * 4, 16)
    topk_weights_offset = cursor
    cursor += _round_up(topk * 4, 16)
    token_data_offset = cursor
    cursor += hidden_bytes
    sf_data_offset = cursor
    cursor += sf_bytes
    layout = RelayMessageLayout(
        src_metadata_offset=0,
        route_words_offset=route_words_offset,
        topk_weights_offset=topk_weights_offset,
        token_data_offset=token_data_offset,
        sf_data_offset=sf_data_offset,
        message_bytes=_round_up(cursor, alignment),
    )
    if any(value % 16 for value in (hidden_bytes, sf_bytes, layout.token_data_offset, layout.sf_data_offset, layout.message_bytes)):
        raise ValueError(f"the 16-byte vector pack needs 16-byte aligned planes and sizes, got {layout} for hidden_bytes={hidden_bytes} sf_bytes={sf_bytes}.")
    return layout


__all__ = ["RelayMessageLayout", "build_relay_message_layout"]
