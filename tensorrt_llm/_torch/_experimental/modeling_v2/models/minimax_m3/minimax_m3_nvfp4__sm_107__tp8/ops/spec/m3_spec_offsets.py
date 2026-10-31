# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The overlap scheduler's speculative step correction as one launch (``RuntimeHooks.spec_offsets_op``)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.logger import logger

# The engagement line is logged once per process (module state: __call__ reads no instance state).
_ENGAGED: list = []


class M3SpecOffsets(OpWrapper):
    """``positions += position_offsets`` and ``kv_lens += kv_lens_offsets`` (int32, contiguous 1-D views, equal lengths
    per pair) in one plain launch: trtllm::minimax_m3_sm107_tp8_spec_offsets (minimaxM3Sm107Tp8/specOffsets.cu), the
    same wrapping int32 adds the two ATen ops do. ``__call__`` returns False, having written nothing, for other
    layouts (the caller runs the ATen adds)."""

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="1 request, 4 tokens (draft length 3)", spec=dict(tokens=4, seqs=1)),
        Cell(why="8 requests, 32 tokens", spec=dict(tokens=32, seqs=8)),
        Cell(why="wrap-around values", spec=dict(tokens=8, seqs=2, wrap=True)),
    )

    note = "Integer adds: the reference (the two in-place ATen adds) is exact."

    def __call__(self, pos, pos_off, kv, kv_off) -> bool:
        ts = (pos, pos_off, kv, kv_off)
        if (
            any(
                t.dtype != torch.int32 or t.dim() != 1 or t.stride(0) != 1 or not t.is_cuda
                for t in ts
            )
            or pos.numel() != pos_off.numel()
            or kv.numel() != kv_off.numel()
        ):
            return False
        torch.ops.trtllm.minimax_m3_sm107_tp8_spec_offsets(pos, pos_off, kv, kv_off)
        if not _ENGAGED:
            _ENGAGED.append(True)
            logger.info("[M3 host] spec step correction: one launch taken (spec_offsets_op)")
        return True

    def reference(self, pos, pos_off, kv, kv_off) -> None:
        pos += pos_off
        kv += kv_off

    def is_valid(self, pos, pos_off, kv, kv_off) -> None:
        assert pos.numel() == pos_off.numel() and kv.numel() == kv_off.numel(), (
            "equal lengths per pair"
        )


m3_spec_offsets = M3SpecOffsets()
