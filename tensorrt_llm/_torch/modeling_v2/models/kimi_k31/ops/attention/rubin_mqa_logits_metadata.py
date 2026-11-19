# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer prefill: DeepGEMM's per-SM schedule of one MQA-logits call (``schedule_meta``).

Wraps the pybind ``tensorrt_llm.deep_gemm.get_mqa_logits_metadata``. The family DSA metadata builds
one schedule per query tile of each prefill chunk, once per step, and every DSA layer's logits call
of that tile passes it as ``schedule_meta``. With it the kernel splits the keys over the SMs
instead of running the query blocks grid-stride, so a tile does not cost whole waves of
4-row query blocks.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell, OpWrapper
from tensorrt_llm.deep_gemm import get_mqa_logits_metadata as _deep_gemm_mqa_logits_metadata

#: Query rows per DeepGEMM MQA-logits block.
BLOCK_Q = 4
#: Keys per unit of schedule work (two 128-key compute blocks).
SPLIT_KV = 256


def _spec(rows, keys, heads=32, **overrides) -> dict:
    """``rows`` causal query rows over ``keys`` gathered keys, ``heads`` index heads."""
    spec = dict(rows=rows, keys=keys, heads=heads, causal=True)
    spec.update(overrides)
    return spec


class RubinMqaLogitsMetadata(OpWrapper):
    """Returns DeepGEMM's schedule of one ``fp8_fp4_mqa_logits`` call over query rows with key
    windows ``[cu_seqlen_ks[r], cu_seqlen_ke[r])`` into ``num_kv_tokens`` keys and ``num_heads``
    heads: an int32 device tensor the call takes as ``schedule_meta``.

    The schedule only places work; the logits of a call with it are bitwise those without it (each
    (row, key) dot product and the head sum run in the same order). The windows are read on the
    device, so the schedule follows the values the step's copy wrote.
    """

    # No architecture listed until a GPU run drives the cells on sm_107.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one 2k-row chunk over its own 2k keys (no cached prefix)", spec=_spec(2048, 2048)
        ),
        Cell(
            why="a tile cut by the per-call budget: 1,795 rows behind a 64k prefix",
            spec=_spec(1795, 65536 + 1795),
        ),
        Cell(why="a packed chunk of 8 short prompts", spec=_spec(4096, 4096, packed=8)),
        Cell(why="rows not a multiple of the 4-row query block", spec=_spec(1023, 5000)),
        Cell(why="a single query row", spec=_spec(1, 2049)),
    )
    note = """
    The flat targets' indexer prefill (DEP16 and TEP8 context) scores each query tile with
    DeepGEMM's FP8 MQA logits and this schedule. The family DSA metadata builds the schedules in
    prepare(), on the step's stream, after the windows' host-to-device copies.
    """

    def __call__(
        self,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        num_kv_tokens: int,
        num_heads: int,
    ) -> torch.Tensor:
        return _deep_gemm_mqa_logits_metadata(cu_seqlen_ks, cu_seqlen_ke, num_kv_tokens, num_heads)

    def reference(
        self,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        num_kv_tokens: int,
        num_heads: int,
    ) -> torch.Tensor:
        """The work the schedule covers, per query block: ``[blocks, 2]`` int32 of (first key,
        key end) over the block's rows, clamped to ``num_kv_tokens``. DeepGEMM's own layout of
        the schedule is not specified here; the entry's test checks that a logits call with the
        schedule equals one without it. ``num_heads`` is ignored."""
        rows = cu_seqlen_ks.shape[0]
        blocks = -(-rows // BLOCK_Q)
        ks = torch.full((blocks * BLOCK_Q,), num_kv_tokens, dtype=torch.int64)
        ke = torch.zeros((blocks * BLOCK_Q,), dtype=torch.int64)
        ks[:rows] = cu_seqlen_ks.cpu().to(torch.int64)
        ke[:rows] = cu_seqlen_ke.cpu().to(torch.int64)
        first = ks.view(blocks, BLOCK_Q).min(dim=1).values.clamp(0, num_kv_tokens)
        end = ke.view(blocks, BLOCK_Q).max(dim=1).values.clamp(0, num_kv_tokens)
        return torch.stack([first, end], dim=1).to(torch.int32)

    def is_valid(
        self,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        num_kv_tokens: int,
        num_heads: int,
    ) -> None:
        # The schedule must describe the call it is passed to: the same windows and key count.
        assert cu_seqlen_ks.dtype == torch.int32 and cu_seqlen_ke.dtype == torch.int32, (
            "the windows must be int32, as the logits call takes them"
        )
        assert cu_seqlen_ks.shape == cu_seqlen_ke.shape and cu_seqlen_ks.dim() == 1
        assert num_kv_tokens > 0, "a call over no keys has nothing to schedule"


rubin_mqa_logits_metadata = RubinMqaLogitsMetadata()
